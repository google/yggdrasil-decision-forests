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

#include "yggdrasil_decision_forests/metric/comparison.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "boost/math/distributions/binomial.hpp"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/metric/labels.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/ranking_ndcg.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace metric {
using ::yggdrasil_decision_forests::model::proto::CLASSIFICATION;

dataset::proto::DataSpecification CreateDataSpecForComparisonTable(
    const proto::EvaluationOptions& option,
    const proto::EvaluationResults& example_of_evaluation) {
  dataset::proto::DataSpecification data_spec;
  dataset::AddColumn(kLabelModel, dataset::proto::ColumnType::STRING,
                     &data_spec);
  dataset::AddColumn(kLabelTrainingDuration,
                     dataset::proto::ColumnType::NUMERICAL, &data_spec);
  switch (option.task()) {
    case model::proto::Task::CLASSIFICATION: {
      dataset::AddColumn(kLabelAccuracy, dataset::proto::ColumnType::NUMERICAL,
                         &data_spec);
      dataset::AddColumn(kLabelAccuracyConfidenceBounds95p,
                         dataset::proto::ColumnType::NUMERICAL_SET, &data_spec);
      const auto num_label_values = example_of_evaluation.label_column()
                                        .categorical()
                                        .number_of_unique_values();
      for (int one_vs_other_label_value = 0;
           one_vs_other_label_value < num_label_values;
           one_vs_other_label_value++) {
        if (!example_of_evaluation.classification()
                 .rocs(one_vs_other_label_value)
                 .has_auc()) {
          continue;
        }
        const auto& roc = example_of_evaluation.classification().rocs(
            one_vs_other_label_value);
        const bool has_bootstrap_confidence_intervals =
            roc.has_bootstrap_lower_bounds_95p();

        struct AddColumnParams {
          const std::string label_name;
          dataset::proto::ColumnType column_type;
        };
        std::vector<AddColumnParams> add_column_params;

        // AUC
        add_column_params.push_back(
            {kLabelAUC, dataset::proto::ColumnType::NUMERICAL});
        add_column_params.push_back(
            {kLabelAUCConfidenceBounds95p,
             dataset::proto::ColumnType::NUMERICAL_SET});

        if (has_bootstrap_confidence_intervals) {
          add_column_params.push_back(
              {kLabelAUCConfidenceBoundsBootstrap95p,
               dataset::proto::ColumnType::NUMERICAL_SET});
        }

        // PR AUC
        add_column_params.push_back(
            {kLabelPRAUC, dataset::proto::ColumnType::NUMERICAL});
        add_column_params.push_back(
            {kLabelPRAUCConfidenceBounds95p,
             dataset::proto::ColumnType::NUMERICAL_SET});
        if (has_bootstrap_confidence_intervals) {
          add_column_params.push_back(
              {kLabelPRAUCConfidenceBoundsBootstrap95p,
               dataset::proto::ColumnType::NUMERICAL_SET});
        }
        // AP
        add_column_params.push_back(
            {kLabelAP, dataset::proto::ColumnType::NUMERICAL});
        if (has_bootstrap_confidence_intervals) {
          add_column_params.push_back(
              {kLabelAPConfidenceBoundsBootstrap95p,
               dataset::proto::ColumnType::NUMERICAL_SET});
        }
        // X@Y metrics.
        for (const auto& x_at_y_accessor : XAtYMetricsAccessors()) {
          const auto& x_at_ys = x_at_y_accessor.const_access(roc);
          for (int idx = 0; idx < x_at_ys.size(); idx++) {
            const auto& x_at_y = x_at_ys[idx];
            const std::string column_label = absl::Substitute(
                "$0@$1=$2", x_at_y_accessor.x_name, x_at_y_accessor.y_name,
                x_at_y.y_metric_constraint());
            add_column_params.push_back(
                {column_label, dataset::proto::ColumnType::NUMERICAL});
            if (has_bootstrap_confidence_intervals) {
              add_column_params.push_back(
                  {absl::StrCat(column_label, " CI95[B]"),
                   dataset::proto::ColumnType::NUMERICAL_SET});
            }
          }
        }

        for (const auto& params : add_column_params) {
          dataset::AddColumn(GetPerClassComparisonMetricLabel(
                                 example_of_evaluation,
                                 one_vs_other_label_value, params.label_name),
                             params.column_type, &data_spec);
        }
      }
    } break;
    case model::proto::Task::REGRESSION:
      dataset::AddColumn(kLabelRmse, dataset::proto::ColumnType::NUMERICAL,
                         &data_spec);
      dataset::AddColumn(kLabelRmseConfidenceBoundsChi95p,
                         dataset::proto::ColumnType::NUMERICAL_SET, &data_spec);
      if (example_of_evaluation.regression()
              .has_bootstrap_rmse_lower_bounds_95p()) {
        dataset::AddColumn(kLabelRmseConfidenceBoundsBootstrap95p,
                           dataset::proto::ColumnType::NUMERICAL_SET,
                           &data_spec);
      }
      break;
    case model::proto::Task::RANKING:
      dataset::AddColumn(
          absl::StrCat(kLabelNdcg, "@", option.ranking().ndcg_truncation()),
          dataset::proto::ColumnType::NUMERICAL, &data_spec);
      if (example_of_evaluation.ranking().ndcg().has_bootstrap_based_95p()) {
        dataset::AddColumn(kLabelNdcgConfidenceBoundsBootstrap95p,
                           dataset::proto::ColumnType::NUMERICAL_SET,
                           &data_spec);
      }
      dataset::AddColumn(
          absl::StrCat(kLabelMrr, "@", option.ranking().mrr_truncation()),
          dataset::proto::ColumnType::NUMERICAL, &data_spec);
      if (example_of_evaluation.ranking().mrr().has_bootstrap_based_95p()) {
        dataset::AddColumn(kLabelMrrConfidenceBoundsBootstrap95p,
                           dataset::proto::ColumnType::NUMERICAL_SET,
                           &data_spec);
      }
      break;
    default:
      CHECK(false);
      break;
  }

  return data_spec;
}

std::vector<std::pair<std::string, float>> OneSidedMcNemarTest(
    const proto::EvaluationResults& eval_results1,
    const proto::EvaluationResults& eval_results2) {
  std::vector<std::pair<std::string, float>> labels_and_p_values;
  if (!eval_results1.has_classification() ||
      !eval_results2.has_classification()) {
    return labels_and_p_values;
  }

  // For each roc value.
  for (int roc_idx = 0; roc_idx < eval_results1.classification().rocs_size();
       ++roc_idx) {
    const proto::Roc& roc_1 = eval_results1.classification().rocs(roc_idx);
    CHECK_LT(roc_idx, eval_results2.classification().rocs_size());
    const proto::Roc& roc_2 = eval_results2.classification().rocs(roc_idx);

    // Max accuracy.
    const float max_accuracy_threshold1 =
        ComputeThresholdForMaxAccuracy(roc_1.curve());
    const float max_accuracy_threshold2 =
        ComputeThresholdForMaxAccuracy(roc_2.curve());
    const std::string label =
        absl::StrCat(dataset::CategoricalIdxToRepresentation(
                         eval_results1.label_column(), roc_idx),
                     "_vs_the_others", "@MaxAccuracy");
    const float pvalue =
        OneSidedMcNemarTest(eval_results1, eval_results2, roc_idx,
                            max_accuracy_threshold1, max_accuracy_threshold2);
    labels_and_p_values.push_back(std::make_pair(label, pvalue));

    for (const auto& x_at_y_accessor : XAtYMetricsAccessors()) {
      const auto& x_at_ys = x_at_y_accessor.const_access(roc_1);
      const auto& x_at_ys2 = x_at_y_accessor.const_access(roc_2);
      CHECK_EQ(x_at_ys.size(), x_at_ys2.size());
      for (int x_at_y_idx = 0; x_at_y_idx < x_at_ys.size(); ++x_at_y_idx) {
        const std::string xy_label =
            absl::StrCat(dataset::CategoricalIdxToRepresentation(
                             eval_results1.label_column(), roc_idx),
                         "_vs_the_others@", x_at_y_accessor.y_name, "=",
                         x_at_ys[x_at_y_idx].y_metric_constraint());
        const float xy_pvalue = OneSidedMcNemarTest(
            eval_results1, eval_results2, roc_idx,
            x_at_ys[x_at_y_idx].threshold(), x_at_ys2[x_at_y_idx].threshold());
        labels_and_p_values.push_back(std::make_pair(xy_label, xy_pvalue));
      }
    }
  }
  return labels_and_p_values;
}

// This is calculated using McNemar Tests (Description at
// https://www.mathworks.com/help/stats/testcholdout.html#bup0p8g-1)
float OneSidedMcNemarTest(const proto::EvaluationResults& eval_results1,
                          const proto::EvaluationResults& eval_results2,
                          const int roc_idx, const float threshold1,
                          const float threshold2) {
  // Compute n12 and n21.
  double n12 = 0.0;
  double n21 = 0.0;
  for (size_t i = 0; i < eval_results1.sampled_predictions_size(); ++i) {
    const int ground_truth =
        eval_results1.sampled_predictions(i).classification().ground_truth();
    CHECK_EQ(
        ground_truth,
        eval_results2.sampled_predictions(i).classification().ground_truth());
    CHECK_GT(eval_results1.sampled_predictions(i)
                 .classification()
                 .distribution()
                 .sum(),
             0);
    CHECK_GT(eval_results2.sampled_predictions(i)
                 .classification()
                 .distribution()
                 .sum(),
             0);

    const float prediction1 = eval_results1.sampled_predictions(i)
                                  .classification()
                                  .distribution()
                                  .counts(roc_idx) /
                              eval_results1.sampled_predictions(i)
                                  .classification()
                                  .distribution()
                                  .sum();
    const float prediction2 = eval_results2.sampled_predictions(i)
                                  .classification()
                                  .distribution()
                                  .counts(roc_idx) /
                              eval_results2.sampled_predictions(i)
                                  .classification()
                                  .distribution()
                                  .sum();

    const bool correct1 =
        (prediction1 >= threshold1) == (ground_truth == roc_idx);
    const bool correct2 =
        (prediction2 >= threshold2) == (ground_truth == roc_idx);

    // The metric is not perfectly designed to take weights as input. If the
    // weights are imbalanced (for e.g if the weights are much larger than 1),
    // then this test does not work correctly. However, if enough samples are
    // collected, and the weights are not very imbalanced, this test works fine
    // (although a bit optimistically).
    if (correct1 && !correct2) {
      n12 += eval_results1.sampled_predictions(i).weight();
    }
    if (!correct1 && correct2) {
      n21 += eval_results1.sampled_predictions(i).weight();
    }
  }

  if (n12 + n21 == 0.0) {
    return 1.0;
  }

  const boost::math::binomial_distribution<double> distribution(n12 + n21, 0.5);
  const float pvalue = 1 - boost::math::cdf(distribution, n12);

  return static_cast<float>(pvalue);
}

float PairwiseRegressiveResidualTest(
    const proto::EvaluationResults& eval_baseline,
    const proto::EvaluationResults& eval_candidate) {
  CHECK_EQ(eval_baseline.sampled_predictions_size(),
           eval_candidate.sampled_predictions_size());
  const auto num_examples = eval_baseline.sampled_predictions_size();

  std::vector<float> sample;
  sample.reserve(num_examples);

  for (size_t i = 0; i < eval_baseline.sampled_predictions_size(); ++i) {
    const float label =
        eval_baseline.sampled_predictions(i).regression().ground_truth();
    CHECK_EQ(label,
             eval_candidate.sampled_predictions(i).regression().ground_truth());
    const float pred_1 =
        eval_baseline.sampled_predictions(i).regression().value();
    const float pred_2 =
        eval_candidate.sampled_predictions(i).regression().value();
    const float residual_1 = std::abs(label - pred_1);
    const float residual_2 = std::abs(label - pred_2);
    sample.push_back(residual_1 - residual_2);
  }

  return PValueMeanIsGreaterThanZero(sample);
}

float PairwiseRankingNDCG5Test(const proto::EvaluationResults& eval_baseline,
                               const proto::EvaluationResults& eval_candidate) {
  CHECK_EQ(eval_baseline.sampled_predictions_size(),
           eval_candidate.sampled_predictions_size());

  NDCGCalculator ndcg_calculator(5);
  struct Group {
    std::vector<RankingLabelAndPrediction> model_1, model_2;
  };
  absl::flat_hash_map<uint64_t, Group> groups;

  // Group examples.
  for (size_t i = 0; i < eval_baseline.sampled_predictions_size(); ++i) {
    const auto label =
        eval_baseline.sampled_predictions(i).ranking().ground_truth_relevance();
    CHECK_EQ(label, eval_candidate.sampled_predictions(i)
                        .ranking()
                        .ground_truth_relevance());
    const auto pred_1 =
        eval_baseline.sampled_predictions(i).ranking().relevance();
    const auto pred_2 =
        eval_candidate.sampled_predictions(i).ranking().relevance();
    const auto group_id =
        eval_baseline.sampled_predictions(i).ranking().group_id();
    CHECK_EQ(group_id,
             eval_candidate.sampled_predictions(i).ranking().group_id());
    auto& group = groups[group_id];
    group.model_1.push_back({/*.prediction =*/pred_1, /*.label =*/label});
    group.model_2.push_back({/*.prediction =*/pred_2, /*.label =*/label});
  }

  // Compute ndcg difference samples.
  std::vector<float> sample;
  sample.reserve(groups.size());
  for (auto& group : groups) {
    const float ndcg_1 = ndcg_calculator.NDCGForUnordered(group.second.model_1);
    const float ndcg_2 = ndcg_calculator.NDCGForUnordered(group.second.model_2);
    sample.push_back(ndcg_2 - ndcg_1);
  }

  return PValueMeanIsGreaterThanZero(sample);
}

}  // namespace metric
}  // namespace yggdrasil_decision_forests
