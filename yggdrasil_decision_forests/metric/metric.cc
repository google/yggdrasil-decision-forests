/*
 * Copyright 2021 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "yggdrasil_decision_forests/metric/metric.h"

#include <functional>
#include <random>
#include <vector>

#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "boost/math/distributions/binomial.hpp"
#include "boost/math/distributions/chi_squared.hpp"
#include "boost/math/distributions/students_t.hpp"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/metric/labels.h"
#include "yggdrasil_decision_forests/metric/ranking_mrr.h"
#include "yggdrasil_decision_forests/metric/ranking_ndcg.h"
#include "yggdrasil_decision_forests/metric/uplift.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace metric {

namespace {

// Compute the AUC (area under the curve) of the ROC curve.
double computeAUC(const google::protobuf::RepeatedPtrField<proto::Roc::Point>& curve) {
  double auc = 0.0;
  for (size_t roc_point_idx = 0; roc_point_idx < curve.size() - 1;
       roc_point_idx++) {
    const auto& cur = curve[roc_point_idx];
    const auto& next = curve[roc_point_idx + 1];
    auc += (internal::RocFPR(cur) - internal::RocFPR(next)) *
           (internal::RocTPR(next) + internal::RocTPR(cur)) / 2;
  }
  return auc;
}

double computePrAuc(const google::protobuf::RepeatedPtrField<proto::Roc::Point>& curve) {
  double pr_auc = 0.0;
  for (size_t roc_point_idx = 0; roc_point_idx < curve.size() - 1;
       roc_point_idx++) {
    const auto& cur = curve[roc_point_idx];
    const auto& next = curve[roc_point_idx + 1];
    // "roc.curve()" is sorted by threshold value, therefore this formulation
    // corresponds to the lower trapezoidal rule.
    pr_auc += (internal::RocPrecision(cur) + internal::RocPrecision(next)) *
              (internal::RocTPR(cur) - internal::RocTPR(next)) / 2;
  }
  return pr_auc;
}

double computeAP(const google::protobuf::RepeatedPtrField<proto::Roc::Point>& curve) {
  double ap = 0.0;
  for (size_t roc_point_idx = 0; roc_point_idx < curve.size() - 1;
       roc_point_idx++) {
    const auto& cur = curve[roc_point_idx];
    const auto& next = curve[roc_point_idx + 1];
    ap += internal::RocPrecision(next) *
          (internal::RocTPR(cur) - internal::RocTPR(next));
  }
  return ap;
}

// Returns the index of the item in "items" with the greatest "prediction"
// value. If multiple item have the same greatest prediction value, returns the
// smallest index. Returns "-1" if items is empty.
int GreatestPredictionIndex(
    const std::vector<RankingLabelAndPrediction>& items) {
  if (ABSL_PREDICT_FALSE(items.empty())) {
    return -1;
  }
  int max_idx = 0;
  float max_prediction = items.front().prediction;

  for (size_t index = 1; index < items.size(); index++) {
    if (items[index].prediction > max_prediction) {
      max_prediction = items[index].prediction;
      max_idx = index;
    }
  }
  return max_idx;
}

// Extract the lower and upper bounds from the samples (using the "getter") and
// store the values using the "setter".
void SetLowerAndUpperBounds(
    const std::vector<proto::Roc>& samples,
    const std::function<double(const proto::Roc&)>& getter,
    const std::function<void(const double, proto::Roc*)>& setter,
    proto::Roc* roc) {
  const auto ci = internal::GetQuantiles(samples, getter, 0.025f, 0.975f);
  setter(ci.first, roc->mutable_bootstrap_lower_bounds_95p());
  setter(ci.second, roc->mutable_bootstrap_upper_bounds_95p());
}

// Build the ROC curve from a set of sorted predictions.
// "prediction_selected_count" specifies the number of times each prediction is
// selected. If "prediction_selected_count" is empty, all predictions are
// selected once.
void BuildROCCurveFromSortedPredictions(
    const std::vector<BinaryPrediction>& sorted_predictions,
    const std::vector<int>& prediction_selected_count,
    const double sum_positive_label, const double sum_negative_label,
    google::protobuf::RepeatedPtrField<proto::Roc::Point>* curve) {
  CHECK(prediction_selected_count.empty() ||
        prediction_selected_count.size() == sorted_predictions.size());
  curve->Clear();
  // Initialize accumulator.
  proto::Roc::Point accumulator;
  accumulator.set_tp(sum_positive_label);
  accumulator.set_fp(sum_negative_label);
  accumulator.set_tn(0);
  accumulator.set_fn(0);
  // Note: The threshold will be set at the end.
  *curve->Add() = accumulator;

  // ROC Curve.
  for (size_t prediction_idx = 0; prediction_idx < sorted_predictions.size();
       prediction_idx++) {
    const auto& prediction = sorted_predictions[prediction_idx];

    int count = 1;
    if (!prediction_selected_count.empty()) {
      count = prediction_selected_count[prediction_idx];
      if (count == 0) {
        continue;
      }
    }

    if (accumulator.threshold() != prediction.predict_true) {
      *curve->Add() = accumulator;
      accumulator.set_threshold(prediction.predict_true);
    }
    const auto weight_times_count = prediction.weight * count;
    if (prediction.label) {
      accumulator.set_tp(accumulator.tp() - weight_times_count);
      accumulator.set_fn(accumulator.fn() + weight_times_count);
    } else {
      accumulator.set_fp(accumulator.fp() - weight_times_count);
      accumulator.set_tn(accumulator.tn() + weight_times_count);
    }
  }
  // Note: The threshold will be set at the end.
  *curve->Add() = accumulator;

  // Top right point. Note: Why -1? -> We need a finite threshold smaller than
  // any observed prediction.
  // Bottom left point. Note: Why +1? -> We need a finite threshold larger
  // than any observed prediction.
  if (curve->size() >= 2) {
    (*curve)[0].set_threshold((*curve)[0].threshold() - 1);
    const auto n = curve->size();
    (*curve)[n - 1].set_threshold((*curve)[n - 2].threshold() + 1);
  }
}

// Return the index of a x@y metric from the y constraint value.
utils::StatusOr<int> XAtYMetricIndexFromConstraint(
    const google::protobuf::RepeatedPtrField<proto::Roc::XAtYMetric>& x_at_ys,
    const float y_constraint, const float margin = 0.0001f) {
  for (int x_at_y_idx = 0; x_at_y_idx < x_at_ys.size(); x_at_y_idx++) {
    if (std::abs(x_at_ys[x_at_y_idx].y_metric_constraint() - y_constraint) <=
        margin) {
      return x_at_y_idx;
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("No x@y metric found with constraint: ", y_constraint));
}

// Estimates the 95% CI of a set of measures using bootstrapping.
void BootstrapMetricEstimate(
    const std::vector<std::pair<float, float>>& values_and_weights,
    const int64_t num_samples, proto::MetricEstimate* metric) {
  std::vector<float> samples(num_samples);
  utils::RandomEngine rnd;
  std::uniform_int_distribution<int64_t> group_dist(
      0, values_and_weights.size() - 1);

  // Bootstrapping.
  for (int64_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    double sum_weighted_values = 0;
    double sum_weights = 0;
    for (int64_t item_idx = 0; item_idx < values_and_weights.size();
         item_idx++) {
      const int64_t selected_group_idx = group_dist(rnd);
      sum_weighted_values += values_and_weights[selected_group_idx].first;
      sum_weights += values_and_weights[selected_group_idx].second;
    }
    double weighted_mean = 0;
    if (sum_weights > 0) {
      weighted_mean = sum_weighted_values / sum_weights;
    }
    samples[sample_idx] = weighted_mean;
  }
  // Estimate confidence intervals using samples.
  std::sort(samples.begin(), samples.end());
  const auto quantile_1_idx = 0.025f * samples.size();
  const auto quantile_2_idx = 0.975f * samples.size();

  metric->mutable_bootstrap_based_95p()->set_lower(samples[quantile_1_idx]);
  metric->mutable_bootstrap_based_95p()->set_upper(samples[quantile_2_idx]);
}

void FinalizeRankingMetricsFromSampledPredictions(
    const proto::EvaluationOptions& option, proto::EvaluationResults* eval) {
  const auto num_preds = eval->sampled_predictions_size();
  CHECK_GT(num_preds, 0);
  auto& ranking = *eval->mutable_ranking();

  // sorted_prediction_by_group[i].pred_and_label_relevance[j] is the predicted
  // score and label for the j-th example for the i-th group.
  // sorted_prediction_by_group[i][j] are stored by decreasing label (i.e.
  // relevance) value.
  struct Item {
    float weight;
    std::vector<RankingLabelAndPrediction> pred_and_label_relevance;
  };
  absl::flat_hash_map<uint64_t, Item> grouped_examples;

  // Populate the groups.
  for (const auto& prediction : eval->sampled_predictions()) {
    auto& group = grouped_examples[prediction.ranking().group_id()];
    group.weight = prediction.weight();
    group.pred_and_label_relevance.push_back(
        {/*.prediction =*/prediction.ranking().relevance(),
         /*.label =*/prediction.ranking().ground_truth_relevance()});
  }

  if (!option.ranking().allow_only_one_group() &&
      grouped_examples.size() == 1) {
    LOG(FATAL) << "During the ranking evaluation, all items ("
               << eval->sampled_predictions().size()
               << ") (e.g. documents) are in the same group (e.g. "
                  "query). Make sure the dataspec was also "
                  "initialized using the test dataset, or that the ranking "
                  "group is of type HASH.";
  }

  NDCGCalculator ndcg_computer(option.ranking().ndcg_truncation());
  MRRCalculator mrr_computer(option.ranking().mrr_truncation());

  double sum_weighted_ndcg = 0;
  double sum_weights = 0;
  double sum_default_weighted_ndcg = 0;

  double sum_weighted_mrr = 0;
  double sum_weighted_precision_at_1 = 0;

  // NDCGs and weights of each group. Used for the computation of confidence
  // intervals using bootstrapping.
  std::vector<std::pair<float, float>> individual_ndcgs, individual_mrrs;
  individual_ndcgs.reserve(grouped_examples.size());
  individual_mrrs.reserve(grouped_examples.size());

  size_t min_num_items_in_group = std::numeric_limits<size_t>::max();
  size_t max_num_items_in_group = std::numeric_limits<size_t>::min();

  // Sort the prediction in each group.
  for (auto& group : grouped_examples) {
    // Sort the groups by decreasing label value.
    std::sort(
        group.second.pred_and_label_relevance.begin(),
        group.second.pred_and_label_relevance.end(),
        [](const RankingLabelAndPrediction& a,
           const RankingLabelAndPrediction& b) { return a.label > b.label; });

    // NDCG
    const auto weighted_ndcg =
        group.second.weight *
        ndcg_computer.NDCG(group.second.pred_and_label_relevance);
    individual_ndcgs.emplace_back(weighted_ndcg, group.second.weight);

    sum_weighted_ndcg += weighted_ndcg;
    sum_default_weighted_ndcg +=
        group.second.weight *
        ndcg_computer.DefaultNDCG(group.second.pred_and_label_relevance);

    // MRR
    const auto weighted_mrr =
        group.second.weight *
        mrr_computer.MRR(group.second.pred_and_label_relevance);
    individual_mrrs.emplace_back(weighted_mrr, group.second.weight);

    sum_weighted_mrr += weighted_mrr;

    // Precision @ 1
    const auto precision_at_1 =
        GreatestPredictionIndex(group.second.pred_and_label_relevance) == 0;
    sum_weighted_precision_at_1 += group.second.weight * precision_at_1;

    sum_weights += group.second.weight;
    min_num_items_in_group = std::min(
        min_num_items_in_group, group.second.pred_and_label_relevance.size());
    max_num_items_in_group = std::max(
        max_num_items_in_group, group.second.pred_and_label_relevance.size());
  }

  ranking.mutable_ndcg()->set_value(sum_weighted_ndcg / sum_weights);
  ranking.set_default_ndcg(sum_default_weighted_ndcg / sum_weights);

  ranking.mutable_mrr()->set_value(sum_weighted_mrr / sum_weights);
  ranking.set_mrr_truncation(option.ranking().mrr_truncation());

  ranking.mutable_precision_at_1()->set_value(sum_weighted_precision_at_1 /
                                              sum_weights);

  ranking.set_num_groups(grouped_examples.size());
  ranking.set_min_num_items_in_group(min_num_items_in_group);
  ranking.set_max_num_items_in_group(max_num_items_in_group);
  ranking.set_mean_num_items_in_group(
      num_preds / static_cast<double>(grouped_examples.size()));

  ranking.set_ndcg_truncation(option.ranking().ndcg_truncation());

  // Performs bootstrapping.
  if (option.bootstrapping_samples() > 0) {
    LOG(INFO) << "Computing ranking confidence intervals of evaluation "
                 "metrics with bootstrapping.";

    BootstrapMetricEstimate(individual_ndcgs, option.bootstrapping_samples(),
                            eval->mutable_ranking()->mutable_ndcg());

    BootstrapMetricEstimate(individual_mrrs, option.bootstrapping_samples(),
                            eval->mutable_ranking()->mutable_mrr());
  }
}

// Specialization of "MergeEvaluation" to specific task type.
// Don't call any these methods directly as "MergeEvaluation" implements part of
// the merging logic itself.
void MergeEvaluationClassification(
    const proto::EvaluationResults::Classification& src,
    proto::EvaluationResults::Classification* dst) {
  utils::AddToConfusionMatrixProto(src.confusion(), dst->mutable_confusion());
  dst->set_sum_log_loss(dst->sum_log_loss() + src.sum_log_loss());
}

void MergeEvaluationRegression(const proto::EvaluationResults::Regression& src,
                               proto::EvaluationResults::Regression* dst) {
  dst->set_sum_square_error(dst->sum_square_error() + src.sum_square_error());
  dst->set_sum_label(dst->sum_label() + src.sum_label());
  dst->set_sum_square_label(dst->sum_square_label() + src.sum_square_label());
}

void MergeEvaluationRanking(const proto::EvaluationResults::Ranking& src,
                            proto::EvaluationResults::Ranking* dst) {
  // No merging to be done.
}

void MergeEvaluationUplift(const proto::EvaluationResults::Uplift& src,
                           proto::EvaluationResults::Uplift* dst) {
  // No merging to be done.
}

}  // namespace

float PValueMeanIsGreaterThanZero(const std::vector<float>& sample) {
  if (sample.size() < 2) {
    // We need at least two observations to compute the variance and run a
    // t-test.
    return std::numeric_limits<float>::quiet_NaN();
  }

  // Compute mean and variance.
  double sum_squared_values = 0;
  double sum_values = 0;

  for (const auto value : sample) {
    sum_squared_values += value * value;
    sum_values += value;
  }
  const double mean = sum_values / sample.size();

  double standard_deviation =
      std::sqrt(sum_squared_values / sample.size() - mean * mean);
  if (standard_deviation == 0) {
    if (mean > 0.) {
      return 0.f;
    } else {
      return 1.f;
    }
  }

  const double statistic = mean * std::sqrt(sample.size()) / standard_deviation;
  const auto students_t =
      boost::math::students_t_distribution<double>(sample.size() - 1);
  return 1. - boost::math::cdf(students_t, statistic);
}

float ComputeThresholdForMaxAccuracy(
    const google::protobuf::RepeatedPtrField<proto::Roc::Point>& curve) {
  double max_accuracy = 0.0;
  float threshold_for_max_accuracy = 0.0;
  for (const auto& point : curve) {
    if (max_accuracy < internal::RocAccuracy(point)) {
      max_accuracy = internal::RocAccuracy(point);
      threshold_for_max_accuracy = point.threshold();
    }
  }
  return threshold_for_max_accuracy;
}

std::string GetPerClassComparisonMetricLabel(
    const proto::EvaluationResults& eval, const int label_value,
    absl::string_view metric) {
  return absl::StrCat(
      metric, " (",
      dataset::CategoricalIdxToRepresentation(eval.label_column(), label_value),
      " vs others)");
}

std::vector<MetricDefinition> DefaultMetrics(
    model::proto::Task task, const dataset::proto::Column& label) {
  switch (task) {
    case model::proto::Task::CLASSIFICATION: {
      proto::MetricAccessor accuracy;
      accuracy.mutable_classification()->mutable_accuracy();
      std::vector<MetricDefinition> metrics{{/*.name =*/"ACCURACY",
                                             /*.accessor =*/accuracy,
                                             /*.higher_is_better =*/true}};

      // The value 0 is reserved for missing values.
      // In case of binary classification, only present metrics about the
      // positive class.
      const int start_label_value =
          (label.categorical().number_of_unique_values() == 3) ? 2 : 1;

      for (int label_value = start_label_value;
           label_value < label.categorical().number_of_unique_values();
           label_value++) {
        const auto label_value_key =
            dataset::CategoricalIdxToRepresentation(label, label_value);

        // AUC
        proto::MetricAccessor auc;
        auc.mutable_classification()
            ->mutable_one_vs_other()
            ->set_positive_class(label_value_key);
        auc.mutable_classification()->mutable_one_vs_other()->mutable_auc();
        metrics.push_back(
            {/*.name =*/absl::Substitute("AUC_$0_VS_OTHERS", label_value_key),
             /*.accessor =*/auc,
             /*.higher_is_better =*/true,
             /*.require_sampling =*/true});

        // PR-AUC
        proto::MetricAccessor pr_auc;
        pr_auc.mutable_classification()
            ->mutable_one_vs_other()
            ->set_positive_class(label_value_key);
        pr_auc.mutable_classification()
            ->mutable_one_vs_other()
            ->mutable_pr_auc();
        metrics.push_back(
            {/*.name =*/absl::Substitute("PRAUC_$0_VS_OTHERS", label_value_key),
             /*.accessor =*/pr_auc,
             /*.higher_is_better =*/true,
             /*.require_sampling =*/true});

        // AP
        proto::MetricAccessor ap;
        ap.mutable_classification()->mutable_one_vs_other()->set_positive_class(
            label_value_key);
        ap.mutable_classification()->mutable_one_vs_other()->mutable_ap();
        metrics.push_back(
            {/*.name =*/absl::Substitute("AP_$0_VS_OTHERS", label_value_key),
             /*.accessor =*/ap,
             /*.higher_is_better =*/true,
             /*.require_sampling =*/true});
      }

      return metrics;
    }

    case model::proto::Task::REGRESSION: {
      proto::MetricAccessor rmse;
      rmse.mutable_regression()->mutable_rmse();
      return {{/*.name = */ "RMSE", /*.accessor =*/
               rmse, /*.higher_is_better = */ false}};
    }

    case model::proto::Task::RANKING: {
      proto::MetricAccessor ndcg;
      ndcg.mutable_ranking()->mutable_ndcg();
      return {{/*.name =*/"NDCG",
               /*.accessor = */ ndcg,
               /*.higher_is_better =*/true,
               /*.require_sampling =*/
               true}};
    }
    default:
      LOG(FATAL) << "Not implemented task.";
  }
}

namespace internal {

double RocSum(const proto::Roc::Point& point) {
  return point.tp() + point.fp() + point.tn() + point.fn();
}

double RocFPR(const proto::Roc::Point& point) {
  const double n = point.fp() + point.tn();
  if (n == 0.0) {
    return 1.0;
  }
  return static_cast<double>(point.fp()) / n;
}

double RocTPR(const proto::Roc::Point& point) {
  const double n = point.tp() + point.fn();
  if (n == 0.0) {
    return 0.0;
  }
  return static_cast<double>(point.tp()) / n;
}

double RocAccuracy(const proto::Roc::Point& point) {
  const double n = RocSum(point);
  if (n == 0.0) {
    return 0.0;
  }
  return static_cast<double>(point.tp() + point.tn()) / n;
}

double RocPrecision(const proto::Roc::Point& point) {
  const double n = point.tp() + point.fp();
  if (n == 0.0) {
    return 1.0;
  }
  return static_cast<double>(point.tp()) / n;
}

double RocPositiveRatio(const proto::Roc::Point& point) {
  const double n = RocSum(point);
  if (n == 0.0) {
    return 0.0;
  }
  const double p = point.tp() + point.fp();
  return p / n;
}

std::pair<double, double> GetQuantiles(
    const std::vector<proto::Roc>& samples,
    const std::function<double(const proto::Roc&)>& getter,
    const float quantile_1, const float quantile_2) {
  const auto n = samples.size();
  CHECK_GT(n, 0);
  std::vector<double> extracted_samples(n);
  for (size_t sample_idx = 0; sample_idx < n; sample_idx++) {
    extracted_samples[sample_idx] = getter(samples[sample_idx]);
  }
  std::sort(extracted_samples.begin(), extracted_samples.end());
  const auto quantile_1_idx =
      std::min(static_cast<size_t>(quantile_1 * n), n - 1);
  const auto quantile_2_idx =
      std::min(static_cast<size_t>(quantile_2 * n), n - 1);
  return {extracted_samples[quantile_1_idx], extracted_samples[quantile_2_idx]};
}

void UpdateRMSEConfidenceIntervalUsingBootstrapping(
    const proto::EvaluationOptions& option, proto::EvaluationResults* eval) {
  // Bootstrap samples of RMSEes.
  std::vector<float> rmse_samples(option.bootstrapping_samples());
  const auto num_preds = eval->sampled_predictions_size();
  CHECK_GT(num_preds, 0);
  // Random generator for the selection of predictions.
  utils::RandomEngine rnd;
  std::uniform_int_distribution<int64_t> prediction_idx_dist(0, num_preds - 1);
  // Bootstrapping.
  for (int64_t sample_idx = 0; sample_idx < option.bootstrapping_samples();
       sample_idx++) {
    double sum_square_error = 0;
    double sum_weights = 0;
    for (int64_t item_idx = 0; item_idx < num_preds; item_idx++) {
      const int64_t selected_prediction_idx = prediction_idx_dist(rnd);
      const auto& pred = eval->sampled_predictions(selected_prediction_idx);
      const float residual =
          pred.regression().value() - pred.regression().ground_truth();
      sum_square_error += residual * residual * pred.weight();
      sum_weights += pred.weight();
    }
    double rmse = 0;
    if (sum_weights > 0) {
      rmse = std::sqrt(sum_square_error / sum_weights);
    }
    rmse_samples[sample_idx] = rmse;
  }
  // Estimate confidence intervals using samples.
  std::sort(rmse_samples.begin(), rmse_samples.end());
  const auto quantile_1_idx =
      std::min(static_cast<size_t>(0.025f * rmse_samples.size()),
               rmse_samples.size() - 1);
  const auto quantile_2_idx =
      std::min(static_cast<size_t>(0.975f * rmse_samples.size()),
               rmse_samples.size() - 1);
  eval->mutable_regression()->set_bootstrap_rmse_lower_bounds_95p(
      rmse_samples[quantile_1_idx]);
  eval->mutable_regression()->set_bootstrap_rmse_upper_bounds_95p(
      rmse_samples[quantile_2_idx]);
}

void ComputeRocConfidenceIntervalsUsingBootstrapping(
    const proto::EvaluationOptions& option,
    const std::vector<BinaryPrediction>& sorted_predictions, proto::Roc* roc) {
  const auto num_preds = sorted_predictions.size();
  // Working buffer containing the number of time each prediction is selected.
  std::vector<int> selected_count(sorted_predictions.size());
  // Working buffer used to store temporarily the roc curves.
  google::protobuf::RepeatedPtrField<proto::Roc::Point> temporary_roc;
  // Random generator for the selection of predictions.
  utils::RandomEngine rnd;
  std::uniform_int_distribution<int64_t> prediction_idx_dist(
      0, sorted_predictions.size() - 1);
  // The samples.
  std::vector<proto::Roc> samples(option.bootstrapping_samples());

  // Bootstrapping.
  for (int64_t sample_idx = 0; sample_idx < option.bootstrapping_samples();
       sample_idx++) {
    auto& sample_roc = samples[sample_idx];

    // Create a bootstrap of predictions.
    std::fill(selected_count.begin(), selected_count.end(), 0);
    // "sampled_sum_positive_label" can be different than "sum_positive_label"
    // because all the examples don't have the same weights.
    double sampled_sum_positive_label = 0;
    double sampled_sum_negative_label = 0;
    for (int64_t item_idx = 0; item_idx < num_preds; item_idx++) {
      const int64_t selected_prediction_idx = prediction_idx_dist(rnd);
      selected_count[selected_prediction_idx]++;
      sorted_predictions[selected_prediction_idx].label
          ? (sampled_sum_positive_label +=
             sorted_predictions[selected_prediction_idx].weight)
          : (sampled_sum_negative_label +=
             sorted_predictions[selected_prediction_idx].weight);
    }

    // Compute the ROC
    temporary_roc.Clear();
    BuildROCCurveFromSortedPredictions(
        sorted_predictions, selected_count, sampled_sum_positive_label,
        sampled_sum_negative_label, &temporary_roc);

    // Evaluate metrics on the bootstrap.
    sample_roc.set_auc(computeAUC(temporary_roc));
    sample_roc.set_pr_auc(computePrAuc(temporary_roc));
    sample_roc.set_ap(computeAP(temporary_roc));
    ComputeXAtYMetrics(option, temporary_roc, &sample_roc);
  }

  // Estimate confidence intervals using samples.

  // AUC
  SetLowerAndUpperBounds(
      samples, [](const proto::Roc& item) { return item.auc(); },
      [](const double value, proto::Roc* item) { return item->set_auc(value); },
      roc);

  // PR AUC
  SetLowerAndUpperBounds(
      samples, [](const proto::Roc& item) { return item.pr_auc(); },
      [](const double value, proto::Roc* item) {
        return item->set_pr_auc(value);
      },
      roc);

  // AP
  SetLowerAndUpperBounds(
      samples, [](const proto::Roc& item) { return item.ap(); },
      [](const double value, proto::Roc* item) { return item->set_ap(value); },
      roc);

  // X@Y Metrics
  CHECK(!samples.empty());
  for (const auto& x_at_y_accessor : XAtYMetricsAccessors()) {
    const auto num_conditions = x_at_y_accessor.const_access(samples[0]).size();
    for (int constraint_idx = 0; constraint_idx < num_conditions;
         constraint_idx++) {
      const auto getter = [&](const proto::Roc& item) {
        const auto& x_at_y = x_at_y_accessor.const_access(item);
        CHECK_LT(constraint_idx, x_at_y.size());
        return x_at_y[constraint_idx].x_metric_value();
      };
      const auto setter = [&](const double value, proto::Roc* item) {
        auto& x_at_y = *x_at_y_accessor.mutable_access(item);
        CHECK_EQ(constraint_idx, x_at_y.size());
        auto* constraint_value = x_at_y.Add();
        constraint_value->set_x_metric_value(value);
      };
      SetLowerAndUpperBounds(samples, getter, setter, roc);
    }
  }
}
}  // namespace internal

std::vector<XAtYAccessor> XAtYMetricsAccessors() {
  std::vector<XAtYAccessor> accessors;
  accessors.push_back(
      {"Precision", "Recall",
       [](const proto::Roc& roc)
           -> const google::protobuf::RepeatedPtrField<proto::Roc::XAtYMetric>& {
         return roc.precision_at_recall();
       },
       [](proto::Roc* roc) { return roc->mutable_precision_at_recall(); }});

  accessors.push_back(
      {"Recall", "Precision",
       [](const proto::Roc& roc)
           -> const google::protobuf::RepeatedPtrField<proto::Roc::XAtYMetric>& {
         return roc.recall_at_precision();
       },
       [](proto::Roc* roc) { return roc->mutable_recall_at_precision(); }});

  accessors.push_back(
      {"Precision", "Volume",
       [](const proto::Roc& roc)
           -> const google::protobuf::RepeatedPtrField<proto::Roc::XAtYMetric>& {
         return roc.precision_at_volume();
       },
       [](proto::Roc* roc) { return roc->mutable_precision_at_volume(); }});

  accessors.push_back(
      {"Recall", "False Positive Rate",
       [](const proto::Roc& roc)
           -> const google::protobuf::RepeatedPtrField<proto::Roc::XAtYMetric>& {
         return roc.recall_at_false_positive_rate();
       },
       [](proto::Roc* roc) {
         return roc->mutable_recall_at_false_positive_rate();
       }});

  accessors.push_back(
      {"False Positive Rate", "Recall",
       [](const proto::Roc& roc)
           -> const google::protobuf::RepeatedPtrField<proto::Roc::XAtYMetric>& {
         return roc.false_positive_rate_at_recall();
       },
       [](proto::Roc* roc) {
         return roc->mutable_false_positive_rate_at_recall();
       }});
  return accessors;
}

void InitializeEvaluation(const proto::EvaluationOptions& option,
                          const dataset::proto::Column& label_column,
                          proto::EvaluationResults* eval) {
  switch (option.task()) {
    case model::proto::Task::CLASSIFICATION: {
      if (label_column.type() != dataset::proto::ColumnType::CATEGORICAL) {
        LOG(FATAL) << "Classification requires a categorical label.";
      }
      // Allocate and zero the confusion matrix.
      const int32_t num_classes =
          label_column.categorical().number_of_unique_values();
      utils::InitializeConfusionMatrixProto(
          num_classes, num_classes,
          eval->mutable_classification()->mutable_confusion());
    } break;
    case model::proto::Task::REGRESSION:
      if (label_column.type() != dataset::proto::ColumnType::NUMERICAL) {
        LOG(FATAL) << "Regression requires a numerical label.";
      }
      eval->mutable_regression();
      break;
    case model::proto::Task::RANKING:
      if (label_column.type() != dataset::proto::ColumnType::NUMERICAL) {
        LOG(FATAL) << "Ranking requires a numerical label.";
      }
      eval->mutable_ranking();
      break;
    case model::proto::Task::CATEGORICAL_UPLIFT:
      CHECK_OK(uplift::InitializeCategoricalUpliftEvaluation(
          option, label_column, eval));
      break;
    case model::proto::Task::NUMERICAL_UPLIFT:
      CHECK_OK(uplift::InitializeNumericalUpliftEvaluation(option, label_column,
                                                           eval));
      break;
    default:
      CHECK(false) << "Non supported task type: "
                   << model::proto::Task_Name(option.task());
      break;
  }
}

void AddPrediction(const proto::EvaluationOptions& option,
                   const model::proto::Prediction& pred,
                   utils::RandomEngine* rnd, proto::EvaluationResults* eval) {
  CHECK_EQ(option.has_weights(), pred.has_weight());
  // Count the number of predictions.
  eval->set_count_predictions(eval->count_predictions() + pred.weight());
  eval->set_count_predictions_no_weight(eval->count_predictions_no_weight() +
                                        1);

  // If "need_prediction_sampling=true" examples (or a sample of the examples)
  // are saved in the "sampled_predictions" field of the evaluation result
  // "eval". If "need_prediction_sampling=false", "sampled_predictions" is not
  // populated.
  bool need_prediction_sampling = option.bootstrapping_samples() > 0;

  switch (option.task()) {
    case model::proto::Task::CLASSIFICATION: {
      CHECK(pred.has_classification());
      auto* eval_cls = eval->mutable_classification();
      const auto& pred_cls = pred.classification();
      CHECK(pred_cls.has_ground_truth());
      // Confusion matrix.
      utils::AddToConfusionMatrixProto(pred_cls.ground_truth(),
                                       pred_cls.value(), pred.weight(),
                                       eval_cls->mutable_confusion());
      // Log-Loss
      if (pred_cls.has_distribution()) {
        auto pred_prob_true_class =
            pred_cls.distribution().counts(pred_cls.ground_truth());
        if (pred_cls.distribution().sum() > 0) {
          pred_prob_true_class /= pred_cls.distribution().sum();
        }
        if (pred_prob_true_class == 0)
          pred_prob_true_class = std::numeric_limits<double>::epsilon();
        DCHECK_GE(pred_prob_true_class, 0);
        DCHECK_LE(pred_prob_true_class, 1);
        const double logloss = -std::log(pred_prob_true_class);
        eval_cls->set_sum_log_loss(eval_cls->sum_log_loss() +
                                   logloss * pred.weight());
      }
      // ROC requires prediction sampling.
      need_prediction_sampling |= option.classification().roc_enable();
    } break;

    case model::proto::Task::REGRESSION: {
      CHECK(pred.has_regression());
      auto* eval_reg = eval->mutable_regression();
      const auto& pred_reg = pred.regression();
      CHECK(pred_reg.has_ground_truth());
      // MSE
      const float error = pred_reg.value() - pred_reg.ground_truth();
      eval_reg->set_sum_square_error(eval_reg->sum_square_error() +
                                     error * error * pred.weight());
      eval_reg->set_sum_label(eval_reg->sum_label() +
                              pred_reg.ground_truth() * pred.weight());
      eval_reg->set_sum_square_label(
          eval_reg->sum_square_label() +
          pred_reg.ground_truth() * pred_reg.ground_truth() * pred.weight());
      // Calibration plots require prediction sampling.
      need_prediction_sampling |= option.regression().enable_regression_plots();
    } break;

    case model::proto::Task::RANKING:
      CHECK(pred.has_ranking());
      need_prediction_sampling = true;
      break;
    case model::proto::Task::CATEGORICAL_UPLIFT:
    case model::proto::Task::NUMERICAL_UPLIFT:
      CHECK_OK(uplift::AddUpliftPredictionImp(option, pred, rnd, eval));
      need_prediction_sampling = true;
      break;

    default:
      break;
  }
  std::uniform_real_distribution<float> dist;
  if (need_prediction_sampling && dist(*rnd) <= option.prediction_sampling()) {
    *eval->mutable_sampled_predictions()->Add() = pred;
    eval->set_count_sampled_predictions(eval->count_sampled_predictions() +
                                        pred.weight());
  }
}

void FinalizeEvaluation(const proto::EvaluationOptions& option,
                        const dataset::proto::Column& label_column,
                        proto::EvaluationResults* eval) {
  CHECK(!eval->has_task()) << "The evaluation is finalized twice.";
  eval->set_task(option.task());
  *(eval->mutable_label_column()) = label_column;
  switch (option.task()) {
    case model::proto::Task::CLASSIFICATION: {
      auto* eval_cls = eval->mutable_classification();
      const int32_t num_classes =
          label_column.categorical().number_of_unique_values();
      if (option.classification().roc_enable()) {
        // Compute the ROCs.
        for (int label_value = 0; label_value < num_classes; label_value++) {
          auto* roc = eval_cls->mutable_rocs()->Add();
          BuildROCCurve(option, label_column, *eval, label_value, roc);
        }
      }
    } break;
    case model::proto::Task::REGRESSION: {
      // Performs bootstrapping.
      if (option.bootstrapping_samples() > 0) {
        LOG(INFO) << "Computing rmse intervals of evaluation metrics with "
                     "bootstrapping.";
        internal::UpdateRMSEConfidenceIntervalUsingBootstrapping(option, eval);
      }
    } break;
    case model::proto::Task::RANKING:
      FinalizeRankingMetricsFromSampledPredictions(option, eval);
      break;
    case model::proto::Task::CATEGORICAL_UPLIFT:
    case model::proto::Task::NUMERICAL_UPLIFT:
      CHECK_OK(uplift::FinalizeUpliftMetricsFromSampledPredictions(
          option, label_column, eval));
      break;
    default:
      break;
  }
  if (eval->num_folds() == 0) {
    eval->set_num_folds(1);
  }
}

float Accuracy(const proto::EvaluationResults& eval) {
  if (eval.classification().has_confusion()) {
    if (eval.count_predictions() == 0) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    const double diagonal =
        utils::ConfusionMatrixProtoTrace(eval.classification().confusion());
    return diagonal / eval.count_predictions();
  }
  if (eval.classification().has_accuracy()) {
    return eval.classification().accuracy();
  }
  return std::numeric_limits<float>::quiet_NaN();
}

float ErrorRate(const proto::EvaluationResults& eval) {
  return 1 - Accuracy(eval);
}

float LogLoss(const proto::EvaluationResults& eval) {
  if (eval.count_predictions() == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return eval.classification().sum_log_loss() / eval.count_predictions();
}

float Loss(const proto::EvaluationResults& eval) {
  if (eval.has_loss_value()) {
    return eval.loss_value();
  } else if (eval.classification().has_sum_log_loss()) {
    if (eval.count_predictions() == 0) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return eval.classification().sum_log_loss() / eval.count_predictions();
  } else {
    return std::numeric_limits<float>::quiet_NaN();
  }
}

float RMSE(const proto::EvaluationResults& eval) {
  if (eval.count_predictions() == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return sqrt(eval.regression().sum_square_error() / eval.count_predictions());
}

float NDCG(const proto::EvaluationResults& eval) {
  return eval.ranking().ndcg().value();
}

float MRR(const proto::EvaluationResults& eval) {
  return eval.ranking().mrr().value();
}

float PrecisionAt1(const proto::EvaluationResults& eval) {
  return eval.ranking().precision_at_1().value();
}

double AUUC(const proto::EvaluationResults& eval) {
  return eval.uplift().auuc();
}

double Qini(const proto::EvaluationResults& eval) {
  return eval.uplift().qini();
}

float DefaultRMSE(const proto::EvaluationResults& eval) {
  if (eval.count_predictions() == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const double mean_label =
      eval.regression().sum_label() / eval.count_predictions();
  return std::sqrt(eval.regression().sum_square_label() /
                       eval.count_predictions() -
                   mean_label * mean_label);
}

float DefaultAccuracy(const proto::EvaluationResults& eval) {
  if (eval.count_predictions() == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const auto& confusion = eval.classification().confusion();
  double top_row_sum = 0;
  for (int row = 0; row < confusion.nrow(); row++) {
    const double sum_col =
        utils::ConfusionMatrixProtoSumColumns(confusion, row);
    if (sum_col > top_row_sum) {
      top_row_sum = sum_col;
    }
  }
  return top_row_sum / eval.count_predictions();
}

float DefaultLogLoss(const proto::EvaluationResults& eval) {
  if (eval.count_predictions() == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const auto& confusion = eval.classification().confusion();
  double log_loss = 0.0;
  for (int row = 0; row < confusion.nrow(); row++) {
    const double sum_col =
        utils::ConfusionMatrixProtoSumColumns(confusion, row);
    double ratio = sum_col / confusion.sum();
    if (ratio == 0) {
      ratio = std::numeric_limits<double>::epsilon();
    }
    log_loss -= ratio * log(ratio);
  }
  return log_loss;
}

float DefaultErrorRate(const proto::EvaluationResults& eval) {
  return 1 - DefaultAccuracy(eval);
}

float DefaultNDCG(const proto::EvaluationResults& eval) {
  return eval.ranking().default_ndcg();
}

void ComputeXAtYMetrics(
    const proto::EvaluationOptions& option,
    const google::protobuf::RepeatedPtrField<proto::Roc::Point>& curve, proto::Roc* roc) {
  // Returns the first [if reversed_iteration==false] or last [if
  // reversed_iteration==true] "y" value such that the corresponding "x" value
  // is greater (non strict) [if upper_bound_constraint==false], or lower (non
  // strict) [if upper_bound_constraint==true].
  const auto get_x_at_y =
      [&](std::function<double(const proto::Roc::Point& point)> xs,
          std::function<double(const proto::Roc::Point& point)> ys,
          const double x_constraint_value, const bool reversed_iteration,
          const bool upper_bound_constraint, proto::Roc::XAtYMetric* metric) {
        const auto constraint = [&](const double x, const double t) {
          if (upper_bound_constraint) {
            return x <= t;
          } else {
            return x >= t;
          }
        };

        metric->set_y_metric_constraint(x_constraint_value);

        if (reversed_iteration) {
          for (int point_idx = curve.size() - 1; point_idx >= 0; point_idx--) {
            const auto& point = curve[point_idx];
            if (constraint(xs(point), x_constraint_value)) {
              metric->set_x_metric_value(ys(point));
              metric->set_threshold(point.threshold());
              return;
            }
          }
        } else {
          for (int point_idx = 0; point_idx < curve.size(); point_idx++) {
            const auto& point = curve[point_idx];
            if (constraint(xs(point), x_constraint_value)) {
              metric->set_x_metric_value(ys(point));
              metric->set_threshold(point.threshold());
              return;
            }
          }
        }
        metric->set_x_metric_value(std::numeric_limits<double>::quiet_NaN());
        metric->set_threshold(std::numeric_limits<double>::quiet_NaN());
      };

  // Precision @ fixed Recall
  for (const double recall : option.classification().precision_at_recall()) {
    get_x_at_y(internal::RocTPR, internal::RocPrecision, recall, true, false,
               roc->add_precision_at_recall());
  }

  // Recall @ fixed Precision
  for (const double precision : option.classification().recall_at_precision()) {
    get_x_at_y(internal::RocPrecision, internal::RocTPR, precision, false,
               false, roc->add_recall_at_precision());
  }

  // Precision @ fixed Volume
  for (const double volume : option.classification().precision_at_volume()) {
    get_x_at_y(internal::RocPositiveRatio, internal::RocPrecision, volume, true,
               false, roc->add_precision_at_volume());
  }

  // Recall @ fixed False positive rate
  for (const double fpr :
       option.classification().recall_at_false_positive_rate()) {
    get_x_at_y(internal::RocFPR, internal::RocTPR, fpr, false, true,
               roc->add_recall_at_false_positive_rate());
  }

  // False positive rate @ fixed Recall
  for (const double recall :
       option.classification().false_positive_rate_at_recall()) {
    get_x_at_y(internal::RocTPR, internal::RocFPR, recall, true, false,
               roc->add_false_positive_rate_at_recall());
  }
}

void BuildROCCurve(const proto::EvaluationOptions& option,
                   const dataset::proto::Column& label_column,
                   const proto::EvaluationResults& eval, const int label_value,
                   proto::Roc* roc) {
  // Retrieve, binarize and sort the predictions.
  std::vector<BinaryPrediction> sorted_predictions;
  sorted_predictions.reserve(eval.sampled_predictions().size());
  double sum_positive_label = 0.0;
  double sum_negative_label = 0.0;
  for (const auto& pred : eval.sampled_predictions()) {
    // The predicted probability for label=true.
    float predict_true = 0.0;
    const float sum = pred.classification().distribution().sum();
    if (sum > 0) {
      predict_true =
          pred.classification().distribution().counts(label_value) / sum;
    }
    const bool label_is_true =
        pred.classification().ground_truth() == label_value;
    sorted_predictions.push_back(
        BinaryPrediction{predict_true, label_is_true, pred.weight()});
    (label_is_true ? sum_positive_label : sum_negative_label) += pred.weight();
  }
  if (sum_positive_label == 0 || sum_negative_label == 0) {
    // Pure one-vs-other label i.e. all the labels are positive or negative.
    // This situation can happen when one of the label value is not represented
    // in the test dataset. This is generally the case for the Out-Of-Vocabulary
    // item.
    return;
  }
  if (sorted_predictions.empty()) {
    LOG(WARNING) << "No sampled prediction found. Computation of the ROC "
                    "curve skipped.";
    return;
  }
  std::sort(sorted_predictions.begin(), sorted_predictions.end(),
            [](const BinaryPrediction& a, const BinaryPrediction& b) {
              return a.predict_true < b.predict_true;
            });

  BuildROCCurveFromSortedPredictions(sorted_predictions, std::vector<int>(),
                                     sum_positive_label, sum_negative_label,
                                     roc->mutable_curve());

  roc->set_auc(computeAUC(roc->curve()));
  roc->set_pr_auc(computePrAuc(roc->curve()));
  roc->set_ap(computeAP(roc->curve()));

  ComputeXAtYMetrics(option, roc->curve(), roc);

  // Performs bootstrapping.
  if (option.bootstrapping_samples() > 0) {
    LOG(INFO) << "Computing confidence intervals of evaluation metrics with "
                 "bootstrapping for label #"
              << label_value << ".";
    internal::ComputeRocConfidenceIntervalsUsingBootstrapping(
        option, sorted_predictions, roc);
  }

  // Remove excedent points in the ROC.
  const auto max_roc_points = option.classification().max_roc_samples();
  if (max_roc_points > 0 && roc->curve_size() > max_roc_points) {
    std::mt19937 rnd;
    std::shuffle(roc->mutable_curve()->pointer_begin(),
                 roc->mutable_curve()->pointer_end(), rnd);
    utils::Truncate(roc->mutable_curve(), max_roc_points);
    std::sort(roc->mutable_curve()->begin(), roc->mutable_curve()->end(),
              [&](const proto::Roc::Point& a, const proto::Roc::Point& b) {
                return a.threshold() < b.threshold();
              });
  }
}

void MergeEvaluation(const proto::EvaluationOptions& option,
                     const proto::EvaluationResults& src,
                     proto::EvaluationResults* dst) {
  // Merging generic fields.
  dst->set_count_predictions(dst->count_predictions() +
                             src.count_predictions());
  dst->set_count_predictions_no_weight(dst->count_predictions_no_weight() +
                                       src.count_predictions_no_weight());
  dst->mutable_sampled_predictions()->MergeFrom(src.sampled_predictions());
  dst->set_count_sampled_predictions(dst->count_sampled_predictions() +
                                     src.count_sampled_predictions());
  dst->set_training_duration_in_seconds(dst->training_duration_in_seconds() +
                                        src.training_duration_in_seconds());
  dst->set_num_folds(dst->num_folds() + src.num_folds());
  CHECK_EQ(src.task(), dst->task());

  // Merging of task specific fields.
  CHECK_EQ(src.type_case(), dst->type_case());
  switch (src.type_case()) {
    case proto::EvaluationResults::kClassification:
      MergeEvaluationClassification(src.classification(),
                                    dst->mutable_classification());
      break;
    case proto::EvaluationResults::kRegression:
      MergeEvaluationRegression(src.regression(), dst->mutable_regression());
      break;
    case proto::EvaluationResults::kRanking:
      MergeEvaluationRanking(src.ranking(), dst->mutable_ranking());
      break;
    case proto::EvaluationResults::kUplift:
      MergeEvaluationUplift(src.uplift(), dst->mutable_uplift());
      break;
    case proto::EvaluationResults::TYPE_NOT_SET:
      LOG(FATAL) << "Non initialized evaluation";
      break;
  }
}

std::unordered_map<std::string, std::string> ExtractFlatMetrics(
    const absl::string_view model_name,
    const proto::EvaluationResults& evaluation) {
  std::unordered_map<std::string, std::string> flat_metrics;
  flat_metrics[kLabelModel] = std::string(model_name);

  if (evaluation.has_training_duration_in_seconds()) {
    flat_metrics[kLabelTrainingDuration] =
        absl::StrCat(evaluation.training_duration_in_seconds());
  }

  switch (evaluation.task()) {
    case model::proto::Task::CLASSIFICATION: {
      // Accuracy
      flat_metrics[kLabelAccuracy] = absl::StrCat(Accuracy(evaluation));
      const auto accuracy_ci_95 = AccuracyConfidenceInterval(evaluation, 0.95);
      flat_metrics[kLabelAccuracyConfidenceBounds95p] = absl::Substitute(
          "$0 $1", accuracy_ci_95.first, accuracy_ci_95.second);

      const auto num_label_values =
          evaluation.label_column().categorical().number_of_unique_values();
      for (int one_vs_other_label_value = 0;
           one_vs_other_label_value < num_label_values;
           one_vs_other_label_value++) {
        CHECK_LT(one_vs_other_label_value,
                 evaluation.classification().rocs_size());
        const auto& roc =
            evaluation.classification().rocs(one_vs_other_label_value);
        if (!roc.has_auc()) {
          continue;
        }

        const bool has_bootstrap_confidence_intervals =
            roc.has_bootstrap_lower_bounds_95p();
        // AUC
        flat_metrics[GetPerClassComparisonMetricLabel(
            evaluation, one_vs_other_label_value, kLabelAUC)] =
            absl::StrCat(roc.auc());
        const auto auc_ci_95 = AUCConfidenceInterval(roc, 0.95);
        flat_metrics[GetPerClassComparisonMetricLabel(
            evaluation, one_vs_other_label_value,
            kLabelAUCConfidenceBounds95p)] =
            absl::Substitute("$0 $1", auc_ci_95.first, auc_ci_95.second);
        if (has_bootstrap_confidence_intervals) {
          flat_metrics[GetPerClassComparisonMetricLabel(
              evaluation, one_vs_other_label_value,
              kLabelAUCConfidenceBoundsBootstrap95p)] =
              absl::Substitute("$0 $1", roc.bootstrap_lower_bounds_95p().auc(),
                               roc.bootstrap_upper_bounds_95p().auc());
        }
        // PR AUC
        flat_metrics[GetPerClassComparisonMetricLabel(
            evaluation, one_vs_other_label_value, kLabelPRAUC)] =
            absl::StrCat(roc.pr_auc());
        const auto pr_auc_ci_95 = PRAUCConfidenceInterval(roc, 0.95);
        flat_metrics[GetPerClassComparisonMetricLabel(
            evaluation, one_vs_other_label_value,
            kLabelPRAUCConfidenceBounds95p)] =
            absl::Substitute("$0 $1", pr_auc_ci_95.first, pr_auc_ci_95.second);
        if (has_bootstrap_confidence_intervals) {
          flat_metrics[GetPerClassComparisonMetricLabel(
              evaluation, one_vs_other_label_value,
              kLabelPRAUCConfidenceBoundsBootstrap95p)] =
              absl::Substitute("$0 $1",
                               roc.bootstrap_lower_bounds_95p().pr_auc(),
                               roc.bootstrap_upper_bounds_95p().pr_auc());
        }
        // AP
        flat_metrics[GetPerClassComparisonMetricLabel(
            evaluation, one_vs_other_label_value, kLabelAP)] =
            absl::StrCat(roc.ap());
        if (has_bootstrap_confidence_intervals) {
          flat_metrics[GetPerClassComparisonMetricLabel(
              evaluation, one_vs_other_label_value,
              kLabelAPConfidenceBoundsBootstrap95p)] =
              absl::Substitute("$0 $1", roc.bootstrap_lower_bounds_95p().ap(),
                               roc.bootstrap_upper_bounds_95p().ap());
        }
        // X@Y metrics.
        for (const auto& x_at_y_accessor : XAtYMetricsAccessors()) {
          const auto& x_at_ys = x_at_y_accessor.const_access(roc);
          for (int idx = 0; idx < x_at_ys.size(); idx++) {
            const auto& x_at_y = x_at_ys[idx];
            const std::string column_label = absl::Substitute(
                "$0@$1=$2", x_at_y_accessor.x_name, x_at_y_accessor.y_name,
                x_at_y.y_metric_constraint());
            // Main value.
            flat_metrics[GetPerClassComparisonMetricLabel(
                evaluation, one_vs_other_label_value, column_label)] =
                absl::StrCat(x_at_y.x_metric_value());
            // Confidence interval.
            if (has_bootstrap_confidence_intervals) {
              const auto& x_at_ys_lower_bound = x_at_y_accessor.const_access(
                  roc.bootstrap_lower_bounds_95p());
              const auto& x_at_ys_upper_bounds = x_at_y_accessor.const_access(
                  roc.bootstrap_upper_bounds_95p());
              CHECK_EQ(x_at_ys.size(), x_at_ys_upper_bounds.size());
              CHECK_EQ(x_at_ys_lower_bound.size(), x_at_ys_upper_bounds.size());
              const auto& x_at_y_lower_bound = x_at_ys_lower_bound[idx];
              const auto& x_at_y_upper_bound = x_at_ys_upper_bounds[idx];
              flat_metrics[GetPerClassComparisonMetricLabel(
                  evaluation, one_vs_other_label_value,
                  absl::StrCat(column_label, " CI95[B]"))] =
                  absl::Substitute("$0 $1", x_at_y_lower_bound.x_metric_value(),
                                   x_at_y_upper_bound.x_metric_value());
            }
          }
        }
      }
    } break;
    case model::proto::Task::REGRESSION: {
      flat_metrics[kLabelRmse] = absl::StrCat(RMSE(evaluation));

      const auto closed_ci = RMSEConfidenceInterval(evaluation);
      flat_metrics[kLabelRmseConfidenceBoundsChi95p] =
          absl::Substitute("$0 $1", closed_ci.first, closed_ci.second);

      if (evaluation.regression().has_bootstrap_rmse_lower_bounds_95p()) {
        flat_metrics[kLabelRmseConfidenceBoundsBootstrap95p] = absl::Substitute(
            "$0 $1", evaluation.regression().bootstrap_rmse_lower_bounds_95p(),
            evaluation.regression().bootstrap_rmse_upper_bounds_95p());
      }
    } break;

    case model::proto::Task::RANKING: {
      flat_metrics[absl::StrCat(kLabelNdcg, "@",
                                evaluation.ranking().ndcg_truncation())] =
          absl::StrCat(NDCG(evaluation));
      if (evaluation.ranking().ndcg().has_bootstrap_based_95p()) {
        flat_metrics[kLabelNdcgConfidenceBoundsBootstrap95p] = absl::Substitute(
            "$0 $1", evaluation.ranking().ndcg().bootstrap_based_95p().lower(),
            evaluation.ranking().ndcg().bootstrap_based_95p().upper());
      }

      flat_metrics[absl::StrCat(kLabelMrr, "@",
                                evaluation.ranking().mrr_truncation())] =
          absl::StrCat(MRR(evaluation));
      if (evaluation.ranking().mrr().has_bootstrap_based_95p()) {
        flat_metrics[kLabelMrrConfidenceBoundsBootstrap95p] = absl::Substitute(
            "$0 $1", evaluation.ranking().mrr().bootstrap_based_95p().lower(),
            evaluation.ranking().mrr().bootstrap_based_95p().upper());
      }
    } break;
    default:
      CHECK(false);
      break;
  }
  return flat_metrics;
}

utils::StatusOr<std::unordered_map<std::string, std::string>>
ExtractFlatMetrics(absl::string_view model_name,
                   absl::string_view evaluation_file) {
  ASSIGN_OR_RETURN(auto serialized_content, file::GetContent(evaluation_file));
  proto::EvaluationResults evaluation;
  evaluation.ParsePartialFromString(std::move(serialized_content));
  return ExtractFlatMetrics(model_name, evaluation);
}

std::pair<float, float> AccuracyConfidenceInterval(
    const proto::EvaluationResults& eval, const float confidence_level) {
  double lower = boost::math::binomial::find_lower_bound_on_p(
      eval.count_predictions_no_weight(),
      Accuracy(eval) * eval.count_predictions_no_weight(),
      1.f - confidence_level);
  double upper = boost::math::binomial::find_upper_bound_on_p(
      eval.count_predictions_no_weight(),
      Accuracy(eval) * eval.count_predictions_no_weight(),
      1.f - confidence_level);
  return {lower, upper};
}

std::pair<float, float> AUCConfidenceInterval(const proto::Roc& roc,
                                              const float confidence_level) {
  const double auc = roc.auc();
  const double auc_square = auc * auc;
  const double q1 = auc / (2 - auc);
  const double q2 = (2 * auc_square) / (1 + auc);
  // Number of positives.
  const double n1 = roc.curve(0).tp() + roc.curve(0).fn();
  // Number of negatives.
  const double n2 = roc.curve(0).fp() + roc.curve(0).tn();
  if (n1 == 0 || n2 == 0) {
    return {0, 1};
  }
  const double quantile = 1 - (1 - confidence_level) / 2;
  const double z = boost::math::quantile(boost::math::normal(), quantile);
  const double term1 = auc * (1 - auc);
  const double term2 = (n1 - 1) * (q1 - auc_square);
  const double term3 = (n2 - 1) * (q2 - auc_square);
  const double se = std::sqrt((term1 + term2 + term3) / (n1 * n2));
  return {auc - z * se, auc + z * se};
}

std::pair<float, float> PRAUCConfidenceInterval(const proto::Roc& roc,
                                                const float confidence_level) {
  const double pr_auc = roc.pr_auc();
  if (pr_auc == 1) {
    return {pr_auc, pr_auc};
  }
  // Number of positive examples.
  const int32_t n = roc.curve(0).tp() + roc.curve(0).fn();
  const double eta = log(pr_auc / (1 - pr_auc));
  const double tau = 1. / std::sqrt(n * pr_auc * (1 - pr_auc));
  const double quantile = 1 - (1 - confidence_level) / 2;
  const double z = boost::math::quantile(boost::math::normal(), quantile);
  const double term1 = std::exp(eta - z * tau);
  const double term2 = std::exp(eta + z * tau);
  return {term1 / (1 + term1), term2 / (1 + term2)};
}

namespace {

// Specialization of GetMetric.

double GetMetricClassificationOneVsOthers(
    const proto::EvaluationResults& evaluation,
    const proto::MetricAccessor::Classification::OneVsOther& metric) {
  const int num_label_classes =
      evaluation.label_column().categorical().number_of_unique_values();

  int positive_class_idx;
  if (!metric.has_positive_class()) {
    // If "metric.positive_class()" is not set, the last class (i.e. the class
    // with the higher index) is considered the positive class.
    positive_class_idx = num_label_classes - 1;
    if (num_label_classes > 3) {
      LOG(WARNING) << "The \"positive_class\" was not provided. Using "
                      "positive_class_idx="
                   << positive_class_idx << "=\""
                   << dataset::CategoricalIdxToRepresentation(
                          evaluation.label_column(), positive_class_idx)
                   << "\" instead.";
    }
  } else {
    positive_class_idx = dataset::CategoricalStringToValue(
        metric.positive_class(), evaluation.label_column());
  }

  CHECK_LT(positive_class_idx, evaluation.classification().rocs_size())
      << "The evaluation does not contains the requested metric. Make sure "
         "that the component that made this evaluation generated the request "
         "metric, or use another metric.\nEvaluation:\n"
      << evaluation.DebugString() << "\nRequested metric:\n"
      << metric.DebugString();

  const auto roc = evaluation.classification().rocs(positive_class_idx);
  switch (metric.Type_case()) {
    case proto::MetricAccessor::Classification::OneVsOther::kAuc:
      return roc.auc();
    case proto::MetricAccessor::Classification::OneVsOther::kPrAuc:
      return roc.pr_auc();
    case proto::MetricAccessor::Classification::OneVsOther::kAp:
      return roc.ap();

    case proto::MetricAccessor::Classification::OneVsOther::
        kPrecisionAtRecall: {
      const auto metric_idx = XAtYMetricIndexFromConstraint(
          roc.precision_at_recall(), metric.precision_at_recall().recall());
      return roc.precision_at_recall(metric_idx.value()).x_metric_value();
    }

    case proto::MetricAccessor::Classification::OneVsOther::
        kRecallAtPrecision: {
      const auto metric_idx = XAtYMetricIndexFromConstraint(
          roc.recall_at_precision(), metric.recall_at_precision().precision());
      return roc.recall_at_precision(metric_idx.value()).x_metric_value();
    }

    case proto::MetricAccessor::Classification::OneVsOther::
        kPrecisionAtVolume: {
      const auto metric_idx = XAtYMetricIndexFromConstraint(
          roc.precision_at_volume(), metric.precision_at_volume().volume());
      return roc.precision_at_volume(metric_idx.value()).x_metric_value();
    }

    case proto::MetricAccessor::Classification::OneVsOther::
        kRecallAtFalsePositiveRate: {
      const auto metric_idx = XAtYMetricIndexFromConstraint(
          roc.recall_at_false_positive_rate(),
          metric.recall_at_false_positive_rate().false_positive_rate());
      return roc.recall_at_false_positive_rate(metric_idx.value())
          .x_metric_value();
    }

    case proto::MetricAccessor::Classification::OneVsOther::
        kFalsePositiveRateAtRecall: {
      const auto metric_idx = XAtYMetricIndexFromConstraint(
          roc.false_positive_rate_at_recall(),
          metric.false_positive_rate_at_recall().recall());
      return roc.false_positive_rate_at_recall(metric_idx.value())
          .x_metric_value();
    }

    default:
      LOG(FATAL) << "Not implemented";
  }
}

double GetMetricClassification(
    const proto::EvaluationResults& evaluation,
    const proto::MetricAccessor::Classification& metric) {
  switch (metric.Type_case()) {
    case proto::MetricAccessor::Classification::kAccuracy:
      return Accuracy(evaluation);
    case proto::MetricAccessor::Classification::kLogloss:
      return LogLoss(evaluation);
    case proto::MetricAccessor::Classification::kOneVsOther:
      return GetMetricClassificationOneVsOthers(evaluation,
                                                metric.one_vs_other());
    default:
      LOG(FATAL) << "Not implemented";
  }
}

double GetMetricRegression(const proto::EvaluationResults& evaluation,
                           const proto::MetricAccessor::Regression& metric) {
  switch (metric.Type_case()) {
    case proto::MetricAccessor::Regression::kRmse:
      return RMSE(evaluation);
    default:
      LOG(FATAL) << "Not implemented";
  }
}

double GetMetricRanking(const proto::EvaluationResults& evaluation,
                        const proto::MetricAccessor::Ranking& metric) {
  switch (metric.Type_case()) {
    case proto::MetricAccessor::Ranking::kNdcg:
      return NDCG(evaluation);
    case proto::MetricAccessor::Ranking::kMrr:
      return MRR(evaluation);
    default:
      LOG(FATAL) << "Not implemented";
  }
}

double GetMetricUplift(const proto::EvaluationResults& evaluation,
                       const proto::MetricAccessor::Uplift& metric) {
  switch (metric.type_case()) {
    case proto::MetricAccessor::Uplift::kQini:
      return Qini(evaluation);
    default:
      LOG(FATAL) << "Not implemented";
  }
}

// Raises a LOG(FATAL) with an error message about the non-availability of a
// metric in a given evaluation.
void GetMetricFatalMissing(const absl::string_view required,
                           const proto::EvaluationResults& evaluation,
                           const proto::MetricAccessor& metric) {
  LOG(FATAL)
      << "The metric does not have " << required
      << " information. Make sure "
         "that the component that generates the evaluation generate this "
         "metric, or use another metric.\nevaluation:\n"
      << evaluation.DebugString() << "\nmetric:\n"
      << metric.DebugString();
}

}  // namespace

double GetMetric(const proto::EvaluationResults& evaluation,
                 const proto::MetricAccessor& metric) {
  switch (metric.Task_case()) {
    case proto::MetricAccessor::kClassification:
      if (!evaluation.has_classification()) {
        GetMetricFatalMissing("classification", evaluation, metric);
      }
      return GetMetricClassification(evaluation, metric.classification());
    case proto::MetricAccessor::kRegression:
      if (!evaluation.has_regression()) {
        GetMetricFatalMissing("regression", evaluation, metric);
      }
      return GetMetricRegression(evaluation, metric.regression());
    case proto::MetricAccessor::kLoss:
      if (!evaluation.has_loss_value()) {
        GetMetricFatalMissing("loss", evaluation, metric);
      }
      return evaluation.loss_value();
    case proto::MetricAccessor::kRanking:
      if (!evaluation.has_ranking()) {
        GetMetricFatalMissing("ranking", evaluation, metric);
      }
      return GetMetricRanking(evaluation, metric.ranking());
    case proto::MetricAccessor::kUplift:
      if (!evaluation.has_uplift()) {
        GetMetricFatalMissing("uplift", evaluation, metric);
      }
      return GetMetricUplift(evaluation, metric.uplift());
    case proto::MetricAccessor::TASK_NOT_SET:
      LOG(FATAL) << "Non set metric accessor proto";
  }
}

utils::StatusOr<bool> HigherIsBetter(const proto::MetricAccessor& metric) {
  switch (metric.Task_case()) {
    case proto::MetricAccessor::kClassification:
      switch (metric.classification().Type_case()) {
        case proto::MetricAccessor::Classification::kAccuracy:
          return true;
        case proto::MetricAccessor::Classification::kLogloss:
          return false;
        case proto::MetricAccessor::Classification::kOneVsOther:
          return true;
        default:
          break;
      }
      break;

    case proto::MetricAccessor::kRegression:
      switch (metric.regression().Type_case()) {
        case proto::MetricAccessor::Regression::kRmse:
          return false;
        default:
          break;
      }
      break;

    case proto::MetricAccessor::kLoss:
      return false;

    case proto::MetricAccessor::kRanking:
      switch (metric.ranking().Type_case()) {
        case proto::MetricAccessor::Uplift::kQini:
          return true;
        default:
          break;
      }
      break;

    case proto::MetricAccessor::kUplift:
      switch (metric.uplift().type_case()) {
        case proto::MetricAccessor::Uplift::kQini:
          return true;
        default:
          break;
      }
      break;

    default:
      break;
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Unknown if the metric should be maximized or minimized: ",
                   metric.DebugString()));
}

std::pair<float, float> RMSEConfidenceInterval(
    const proto::EvaluationResults& eval, const float confidence_level) {
  const double sampled_sd = RMSE(eval);
  const auto n = eval.count_predictions_no_weight();
  const auto chi_square = boost::math::chi_squared_distribution<double>(n);
  const double q1 = quantile(chi_square, 1 - (1. - confidence_level) / 2);
  const double q2 = quantile(chi_square, (1. - confidence_level) / 2);
  const double c1 = std::sqrt(n / q1) * sampled_sd;
  const double c2 = std::sqrt(n / q2) * sampled_sd;
  return {c1, c2};
}

proto::EvaluationResults BinaryClassificationEvaluationHelper(
    const std::vector<bool>& labels, const std::vector<float>& predictions,
    const proto::EvaluationOptions& option, utils::RandomEngine* rnd,
    const absl::string_view positive_label,
    const absl::string_view negative_label) {
  proto::EvaluationResults eval;

  // Categorical value of the "positive" and "negative" classes;
  const int negative_value = 1;
  const int positive_value = 2;

  // Definition of the label column.
  dataset::proto::Column label_column;
  label_column.set_name("Label");
  label_column.set_type(dataset::proto::ColumnType::CATEGORICAL);
  auto& categorical_label = *label_column.mutable_categorical();
  categorical_label.set_number_of_unique_values(3);
  (*categorical_label.mutable_items())["<OOD>"].set_index(0);
  (*categorical_label.mutable_items())[std::string(negative_label)].set_index(
      negative_value);
  (*categorical_label.mutable_items())[std::string(positive_label)].set_index(
      positive_value);

  InitializeEvaluation(option, label_column, &eval);

  CHECK_EQ(labels.size(), predictions.size());
  model::proto::Prediction prediction_proto;
  auto& prediction_distribution =
      *prediction_proto.mutable_classification()->mutable_distribution();
  prediction_distribution.mutable_counts()->Resize(3, 0);

  for (int example_idx = 0; example_idx < labels.size(); example_idx++) {
    prediction_proto.mutable_classification()->set_ground_truth(
        labels[example_idx] ? positive_value : negative_value);

    const float prediction = predictions[example_idx];
    CHECK_GE(prediction, 0.f);
    CHECK_LE(prediction, 1.f);

    prediction_proto.mutable_classification()->set_value(
        prediction >= 0.5f ? positive_value : negative_value);
    prediction_distribution.set_sum(1.f);
    prediction_distribution.set_counts(negative_value, 1 - prediction);
    prediction_distribution.set_counts(positive_value, prediction);

    AddPrediction(option, prediction_proto, rnd, &eval);
  }

  FinalizeEvaluation(option, label_column, &eval);
  return eval;
}

double RMSE(const std::vector<float>& labels,
            const std::vector<float>& predictions,
            const std::vector<float>& weights) {
  CHECK_EQ(labels.size(), predictions.size());
  CHECK_EQ(labels.size(), weights.size());
  double sum_loss = 0;
  double sum_weights = 0;
  for (size_t example_idx = 0; example_idx < labels.size(); example_idx++) {
    const float label = labels[example_idx];
    const float prediction = predictions[example_idx];
    const float weight = weights[example_idx];
    sum_weights += weight;
    // Loss:
    //   (label - prediction)^2
    sum_loss += weight * (label - prediction) * (label - prediction);
  }
  if (sum_weights > 0) {
    return sqrt(sum_loss / sum_weights);
  } else {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

double RMSE(const std::vector<float>& labels,
            const std::vector<float>& predictions) {
  CHECK_EQ(labels.size(), predictions.size());

  double sum_loss = 0;
  for (size_t example_idx = 0; example_idx < labels.size(); example_idx++) {
    const float label = labels[example_idx];
    const float prediction = predictions[example_idx];
    // Loss:
    //   (label - prediction)^2
    sum_loss += (label - prediction) * (label - prediction);
  }
  const auto sum_weights = labels.size();

  if (sum_weights > 0) {
    return sqrt(sum_loss / sum_weights);
  } else {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

}  // namespace metric
}  // namespace yggdrasil_decision_forests
