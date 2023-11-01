/*
 * Copyright 2022 Google LLC.
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

#include "yggdrasil_decision_forests/metric/report.h"

#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/histogram.h"
#include "yggdrasil_decision_forests/utils/html.h"
#include "yggdrasil_decision_forests/utils/plot.h"

namespace yggdrasil_decision_forests {
namespace metric {

namespace {

// Combine the two elements of a pair into a string.
template <typename T>
std::string PairToString(const std::pair<T, T>& p) {
  return absl::StrCat(p.first, " ", p.second);
}

// Adds the string "<key>: <value>\n" is "value is not Nan.
void AppendKeyValueIfNotNan(std::string* dst, absl::string_view key,
                            float value) {
  if (!std::isnan(value)) {
    absl::StrAppend(dst, key, ": ", value, "\n");
  }
}

absl::Status PlotClassificationCurves(const proto::Roc& roc,
                                      const absl::string_view label,
                                      utils::plot::Plot* roc_plot,
                                      utils::plot::Plot* pr_plot,
                                      utils::plot::Plot* tv_plot,
                                      utils::plot::Plot* ta_plot) {
  auto roc_curve = absl::make_unique<utils::plot::Curve>();
  auto pr_curve = absl::make_unique<utils::plot::Curve>();
  auto tv_curve = absl::make_unique<utils::plot::Curve>();
  auto ta_curve = absl::make_unique<utils::plot::Curve>();

  roc_curve->label = std::string(label);
  pr_curve->label = std::string(label);
  tv_curve->label = std::string(label);
  ta_curve->label = std::string(label);

  for (const auto& point : roc.curve()) {
    roc_curve->xs.push_back(internal::RocFPR(point));
    roc_curve->ys.push_back(internal::RocTPR(point));

    pr_curve->xs.push_back(internal::RocTPR(point));
    pr_curve->ys.push_back(internal::RocPrecision(point));

    tv_curve->xs.push_back(point.threshold());
    tv_curve->ys.push_back(internal::RocPositiveRatio(point));

    ta_curve->xs.push_back(point.threshold());
    ta_curve->ys.push_back(internal::RocAccuracy(point));
  }

  roc_plot->items.push_back(std::move(roc_curve));
  pr_plot->items.push_back(std::move(pr_curve));
  tv_plot->items.push_back(std::move(tv_curve));
  ta_plot->items.push_back(std::move(ta_curve));
  return absl::OkStatus();
}

// Creates the HTML report for a classication evaluation.
absl::Status AppendHtmlReportClassiciation(const proto::EvaluationResults& eval,
                                           const HtmlReportOptions& options,
                                           utils::html::Html* html) {
  if (eval.classification().rocs().empty()) {
    return absl::OkStatus();
  }

  utils::plot::MultiPlot multiplot;
  ASSIGN_OR_RETURN(auto placer,
                   utils::plot::PlotPlacer::Create(
                       4, options.num_plots_per_columns, &multiplot));

  ASSIGN_OR_RETURN(auto* roc_plot, placer.NewPlot());
  roc_plot->title = "ROC";
  roc_plot->x_axis.label = "False positive rate";
  roc_plot->y_axis.label = "True positive rate (Recall)";

  ASSIGN_OR_RETURN(auto* pr_plot, placer.NewPlot());
  pr_plot->title = "Precision Recall";
  pr_plot->x_axis.label = "Recall";
  pr_plot->y_axis.label = "Precision";

  ASSIGN_OR_RETURN(auto* tv_plot, placer.NewPlot());
  tv_plot->title = "Threshold / Volume";
  tv_plot->x_axis.label = "Threshold";
  tv_plot->y_axis.label = "Volume";

  ASSIGN_OR_RETURN(auto* ta_plot, placer.NewPlot());
  ta_plot->title = "Threshold / Accuracy";
  ta_plot->x_axis.label = "Threshold";
  ta_plot->y_axis.label = "Accuracy";

  RETURN_IF_ERROR(placer.Finalize());

  // Note: We start at roc_idx=1 as roc_idx=0 correspond to the "OOV vs others".

  for (int roc_idx = 0; roc_idx < eval.classification().rocs().size();
       roc_idx++) {
    const auto& roc = eval.classification().rocs(roc_idx);
    if (!roc.has_auc()) {
      continue;
    }
    const auto positive_label_value =
        dataset::CategoricalIdxToRepresentation(eval.label_column(), roc_idx);
    const auto label = absl::StrCat(positive_label_value, " vs others");
    RETURN_IF_ERROR(
        PlotClassificationCurves(eval.classification().rocs(roc_idx), label,
                                 roc_plot, pr_plot, tv_plot, ta_plot));
  }

  utils::plot::ExportOptions plot_options;
  plot_options.width = options.plot_width;
  plot_options.height = options.plot_height;
  ASSIGN_OR_RETURN(const auto multiplot_html,
                   utils::plot::ExportToHtml(multiplot, plot_options));
  html->AppendRaw(multiplot_html);
  return absl::OkStatus();
}

// Plot the distribution between two continuous variables.
void PlotConditionalVariables(const std::vector<float>& var_1,
                              const std::vector<float>& var_2,
                              const std::vector<float>& weights,
                              const float var_1_min, const float var_1_max,
                              const int num_bins, utils::plot::Plot* plot) {
  CHECK_EQ(var_1.size(), var_2.size());
  CHECK_EQ(var_1.size(), weights.size());

  // Compute the distribution of var_2 for a set of non-overlapping contiguous
  // segments of var1.
  struct ValuesAndWeights {
    double sum_value;
    double sum_weight;

    void add(const float value, const float weight) {
      sum_value += value;
      sum_weight += weight;
    }

    float mean() const { return sum_value / sum_weight; }
  };
  utils::histogram::BucketizedContainer<float, ValuesAndWeights> buckets(
      var_1_min, var_1_max, num_bins);
  for (int idx = 0; idx < var_1.size(); idx++) {
    buckets[var_1[idx]].add(var_2[idx], weights[idx]);
  }

  // Plot the mean of var2 for each segment of var1.
  auto curve_mean = absl::make_unique<utils::plot::Curve>();

  for (int bin_idx = 0; bin_idx < num_bins; bin_idx++) {
    const auto& bucket = buckets.ContentArray()[bin_idx];
    if (bucket.sum_weight == 0) {
      continue;
    }
    const float center = buckets.BinCenter(bin_idx);
    curve_mean->xs.push_back(center);
    curve_mean->ys.push_back(bucket.mean());
  }

  plot->items.push_back(std::move(curve_mean));
}

// Creates the HTML report for a regression evaluation.
absl::Status AppendHtmlReportRegression(const proto::EvaluationResults& eval,
                                        const HtmlReportOptions& options,
                                        utils::html::Html* html) {
  if (eval.sampled_predictions_size() == 0) {
    html->Append(utils::html::P("No predictions"));
    return absl::OkStatus();
  }

  utils::plot::MultiPlot multiplot;
  ASSIGN_OR_RETURN(auto placer,
                   utils::plot::PlotPlacer::Create(
                       6, options.num_plots_per_columns, &multiplot));

  ASSIGN_OR_RETURN(auto* res_plot, placer.NewPlot());
  res_plot->title = "Residual Histogram";
  res_plot->x_axis.label = "False positive rate";
  res_plot->y_axis.label = "True positive rate (Recall)";
  res_plot->show_legend = false;

  ASSIGN_OR_RETURN(auto* gt_plot, placer.NewPlot());
  gt_plot->title = "Ground Truth Histogram";
  gt_plot->x_axis.label = "Recall";
  gt_plot->y_axis.label = "Precision";
  gt_plot->show_legend = false;

  ASSIGN_OR_RETURN(auto* pred_plot, placer.NewPlot());
  pred_plot->title = "Prediction Histogram";
  pred_plot->x_axis.label = "Threshold";
  pred_plot->y_axis.label = "Volume";
  pred_plot->show_legend = false;

  ASSIGN_OR_RETURN(auto* gt_pred_plot, placer.NewPlot());
  gt_pred_plot->title = "Ground Truth vs Predictions";
  gt_pred_plot->x_axis.label = "Ground truth";
  gt_pred_plot->y_axis.label = "Prediction";
  gt_pred_plot->show_legend = false;

  ASSIGN_OR_RETURN(auto* pred_res_plot, placer.NewPlot());
  pred_res_plot->title = "Predictions vs Residual";
  pred_res_plot->x_axis.label = "Prediction";
  pred_res_plot->y_axis.label = "Residual";
  pred_res_plot->show_legend = false;

  ASSIGN_OR_RETURN(auto* gt_res_plot, placer.NewPlot());
  gt_res_plot->title = "Ground Truth vs Residual";
  gt_res_plot->x_axis.label = "Ground truth";
  gt_res_plot->y_axis.label = "Residual";
  gt_res_plot->show_legend = false;

  RETURN_IF_ERROR(placer.Finalize());

  // Determine the minimum and maximum values of specific variables. For
  // example, "prediction_bounds" will be the minimum and maximum prediction
  // value.
  MinMaxStream<float> prediction_bounds;
  MinMaxStream<float> ground_truth_bounds;
  MinMaxStream<float> residual_bounds;

  for (const auto& pred : eval.sampled_predictions()) {
    const float prediction = pred.regression().value();
    const float ground_truth = pred.regression().ground_truth();
    const float residual = ground_truth - prediction;

    prediction_bounds.visit(prediction);
    ground_truth_bounds.visit(ground_truth);
    residual_bounds.visit(residual);
  }

  // Conditional plots.
  std::vector<float> weights, predictions, ground_truths, residuals;
  weights.reserve(eval.sampled_predictions_size());
  predictions.reserve(eval.sampled_predictions_size());
  ground_truths.reserve(eval.sampled_predictions_size());
  residuals.reserve(eval.sampled_predictions_size());

  for (const auto& pred : eval.sampled_predictions()) {
    const float prediction = pred.regression().value();
    const float ground_truth = pred.regression().ground_truth();
    const float residual = ground_truth - prediction;
    const float weight = pred.weight();

    ground_truths.push_back(ground_truth);
    predictions.push_back(prediction);
    residuals.push_back(residual);
    weights.push_back(weight);
  }

  // Add diagonals to gt vs pred plot.
  {
    const auto minimum_model_output = ground_truth_bounds.min();
    const auto maximum_model_output = ground_truth_bounds.max();
    auto diagonal = absl::make_unique<utils::plot::Curve>();
    diagonal->xs = {minimum_model_output, maximum_model_output};
    diagonal->ys = {minimum_model_output, maximum_model_output};
    diagonal->style = utils::plot::LineStyle::DOTTED;
    gt_pred_plot->items.push_back(std::move(diagonal));
  }

  // Number of bins of the histograms and the calibration plots.
  //
  // If the predictions are uniformly distributed, we expect for each bin of the
  // residual histogram to contains approximately 2.5% of the observations.
  const int num_bins = 40;

  // Plot the conditional plots
  PlotConditionalVariables(ground_truths, predictions, weights,
                           ground_truth_bounds.min(), ground_truth_bounds.max(),
                           num_bins, gt_pred_plot);
  PlotConditionalVariables(ground_truths, residuals, weights,
                           ground_truth_bounds.min(), ground_truth_bounds.max(),
                           num_bins, pred_res_plot);
  PlotConditionalVariables(predictions, residuals, weights,
                           prediction_bounds.min(), prediction_bounds.max(),
                           num_bins, gt_res_plot);

  // Plot the histograms
  // Capturing `num_bins` is required for Windows compilation.
  // NOLINTNEXTLINE(clang-diagnostic-unused-lambda-capture)
  const auto add_histogram = [&weights, &num_bins](
                                 const std::vector<float>& values,
                                 utils::plot::Plot* plot) -> absl::Status {
    auto bars = absl::make_unique<utils::plot::Bars>();
    const auto hist = utils::histogram::Histogram<float>::MakeUniform(
        values, static_cast<size_t>(num_bins), weights);
    RETURN_IF_ERROR(bars->FromHistogram(hist));
    plot->items.push_back(std::move(bars));
    return absl::OkStatus();
  };

  RETURN_IF_ERROR(add_histogram(residuals, res_plot));
  RETURN_IF_ERROR(add_histogram(predictions, pred_plot));
  RETURN_IF_ERROR(add_histogram(ground_truths, gt_plot));

  utils::plot::ExportOptions plot_options;
  plot_options.width = options.plot_width;
  plot_options.height = options.plot_height;
  ASSIGN_OR_RETURN(const auto multiplot_html,
                   utils::plot::ExportToHtml(multiplot, plot_options));
  html->AppendRaw(multiplot_html);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::string> TextReport(const proto::EvaluationResults& eval) {
  std::string report;
  RETURN_IF_ERROR(AppendTextReportWithStatus(eval, &report));
  return report;
}

void AppendTextReport(const proto::EvaluationResults& eval,
                      std::string* report) {
  CHECK_OK(AppendTextReportWithStatus(eval, report));
}

absl::Status AppendTextReportWithStatus(const proto::EvaluationResults& eval,
                                        std::string* report) {
  if (eval.has_count_predictions_no_weight()) {
    absl::StrAppend(report, "Number of predictions (without weights): ",
                    eval.count_predictions_no_weight(), "\n");
  }
  if (eval.has_count_predictions()) {
    absl::StrAppend(report, "Number of predictions (with weights): ",
                    eval.count_predictions(), "\n");
  }
  if (eval.has_task()) {
    absl::StrAppend(report, "Task: ", model::proto::Task_Name(eval.task()),
                    "\n");
  }
  if (eval.has_label_column()) {
    absl::StrAppend(report, "Label: ", eval.label_column().name(), "\n");
  }
  if (eval.has_loss_value()) {
    absl::StrAppend(report, "Loss (", eval.loss_name(),
                    "): ", eval.loss_value(), "\n");
  }
  absl::StrAppend(report, "\n");

  switch (eval.type_case()) {
    case proto::EvaluationResults::TypeCase::kClassification:
      RETURN_IF_ERROR(AppendTextReportClassification(eval, report));
      break;
    case proto::EvaluationResults::TypeCase::kRegression:
      RETURN_IF_ERROR(AppendTextReportRegression(eval, report));
      break;
    case proto::EvaluationResults::TypeCase::kRanking:
      RETURN_IF_ERROR(AppendTextReportRanking(eval, report));
      break;
    case proto::EvaluationResults::TypeCase::kUplift:
      RETURN_IF_ERROR(AppendTextReportUplift(eval, report));
      break;
    default:
      STATUS_FATAL("This model does not support evaluation reports.");
  }
  return absl::OkStatus();
}

absl::Status AppendTextReportClassification(
    const proto::EvaluationResults& eval, std::string* report) {
  absl::SubstituteAndAppend(report, "Accuracy: $0  CI95[W][$1]\n",
                            Accuracy(eval),
                            PairToString(AccuracyConfidenceInterval(eval)));

  AppendKeyValueIfNotNan(report, "LogLoss: ", LogLoss(eval));
  AppendKeyValueIfNotNan(report, "ErrorRate: ", ErrorRate(eval));
  absl::StrAppend(report, "\n");

  AppendKeyValueIfNotNan(report, "Default Accuracy: ", DefaultAccuracy(eval));
  AppendKeyValueIfNotNan(report, "Default LogLoss: ", DefaultLogLoss(eval));
  AppendKeyValueIfNotNan(report, "Default ErrorRate: ", DefaultErrorRate(eval));
  absl::StrAppend(report, "\n");

  if (eval.classification().has_confusion()) {
    absl::StrAppend(report, "Confusion Table:\n");
    utils::IntegersConfusionMatrixDouble confusion;
    confusion.Load(eval.classification().confusion());
    RETURN_IF_ERROR(
        confusion.AppendTextReport(eval.label_column(), report, 1, 1));
    absl::StrAppend(report, "\n");
  }

  // Print a description of a X@Y metric. For example:
  // " Recall: 0.1 -> Precision: 0.2 CI95[0.1 0.3] [threshold: 0.3]";
  using ConfidenceIntervalSideAccessor = std::reference_wrapper<
      const google::protobuf::RepeatedPtrField<proto::Roc::XAtYMetric>>;
  using ConfidenceIntervalAccessor =
      std::pair<ConfidenceIntervalSideAccessor, ConfidenceIntervalSideAccessor>;
  const auto append_x_at_y_metric =
      [&](absl::string_view x_label, absl::string_view y_label,
          const google::protobuf::RepeatedPtrField<proto::Roc::XAtYMetric>& x_at_ys,
          absl::optional<ConfidenceIntervalAccessor> confidence_intervals) {
        if (!x_at_ys.empty()) {
          absl::SubstituteAndAppend(report, "    $0 @ fixed $1\n", x_label,
                                    y_label);
          for (int idx = 0; idx < x_at_ys.size(); idx++) {
            const auto& x_at_y = x_at_ys[idx];
            // Confidence interval.
            std::string ci_text;
            if (confidence_intervals.has_value()) {
              ci_text = absl::StrCat("CI95[B][",
                                     confidence_intervals.value()
                                         .first.get()[idx]
                                         .x_metric_value(),
                                     " ",
                                     confidence_intervals.value()
                                         .second.get()[idx]
                                         .x_metric_value(),
                                     "] ");
            }
            absl::SubstituteAndAppend(
                report, "      $0: $1 -> $2: $3 $4[threshold: $5]\n", y_label,
                x_at_y.y_metric_constraint(), x_label, x_at_y.x_metric_value(),
                ci_text, x_at_y.threshold());
          }
        }
      };

  for (int roc_idx = 0; roc_idx < eval.classification().rocs_size();
       roc_idx++) {
    const auto& roc = eval.classification().rocs(roc_idx);
    if (!roc.has_auc()) {
      continue;
    }
    const std::string selected_value =
        dataset::CategoricalIdxToRepresentation(eval.label_column(), roc_idx);
    absl::StrAppend(report, "  \"", selected_value, "\" vs. the others\n");
    // AUC
    absl::SubstituteAndAppend(report, "    auc: $0  CI95[H][$1]", roc.auc(),
                              PairToString(AUCConfidenceInterval(roc)));
    if (roc.has_bootstrap_lower_bounds_95p()) {
      absl::SubstituteAndAppend(report, " CI95[B][$0 $1]",
                                roc.bootstrap_lower_bounds_95p().auc(),
                                roc.bootstrap_upper_bounds_95p().auc());
    }
    absl::StrAppend(report, "\n");
    // PR AUC
    absl::SubstituteAndAppend(report, "    p/r-auc: $0  CI95[L][$1]",
                              roc.pr_auc(),
                              PairToString(PRAUCConfidenceInterval(roc)));
    if (roc.has_bootstrap_lower_bounds_95p()) {
      absl::SubstituteAndAppend(report, " CI95[B][$0 $1]",
                                roc.bootstrap_lower_bounds_95p().pr_auc(),
                                roc.bootstrap_upper_bounds_95p().pr_auc());
    }
    absl::StrAppend(report, "\n");
    // AP
    absl::SubstituteAndAppend(report, "    ap: $0  ", roc.ap());
    if (roc.has_bootstrap_lower_bounds_95p()) {
      absl::SubstituteAndAppend(report, " CI95[B][$0 $1]",
                                roc.bootstrap_lower_bounds_95p().ap(),
                                roc.bootstrap_upper_bounds_95p().ap());
    }
    absl::StrAppend(report, "\n");
    // X@Y
    for (const auto& x_at_y_accessor : XAtYMetricsAccessors()) {
      absl::optional<ConfidenceIntervalAccessor> confidence_interval_accessors;
      const bool has_confidence_intervals =
          roc.has_bootstrap_lower_bounds_95p();
      if (has_confidence_intervals) {
        confidence_interval_accessors = ConfidenceIntervalAccessor(
            x_at_y_accessor.const_access(roc.bootstrap_lower_bounds_95p()),
            x_at_y_accessor.const_access(roc.bootstrap_upper_bounds_95p()));
      }
      append_x_at_y_metric(x_at_y_accessor.x_name, x_at_y_accessor.y_name,
                           x_at_y_accessor.const_access(roc),
                           confidence_interval_accessors);
    }
    absl::StrAppend(report, "\n");
  }
  return absl::OkStatus();
}

absl::Status AppendTextReportRegression(const proto::EvaluationResults& eval,
                                        std::string* report) {
  absl::StrAppend(report, "RMSE: ", RMSE(eval));

  if (eval.count_predictions_no_weight() > 0) {
    const auto closed_ci = RMSEConfidenceInterval(eval);
    absl::SubstituteAndAppend(report, " CI95[X2][$0 $1]", closed_ci.first,
                              closed_ci.second);
  }

  if (eval.regression().has_bootstrap_rmse_lower_bounds_95p()) {
    absl::SubstituteAndAppend(
        report, " CI95[B][$0 $1]",
        eval.regression().bootstrap_rmse_lower_bounds_95p(),
        eval.regression().bootstrap_rmse_upper_bounds_95p());
  }
  absl::StrAppend(report, "\n");

  AppendKeyValueIfNotNan(report, "Default RMSE: ", DefaultRMSE(eval));
  return absl::OkStatus();
}

absl::Status AppendTextReportRanking(const proto::EvaluationResults& eval,
                                     std::string* report) {
  absl::StrAppend(report, "NDCG@", eval.ranking().ndcg_truncation(), ": ",
                  NDCG(eval));
  if (eval.ranking().ndcg().has_bootstrap_based_95p()) {
    absl::SubstituteAndAppend(
        report, " CI95[B][$0 $1]",
        eval.ranking().ndcg().bootstrap_based_95p().lower(),
        eval.ranking().ndcg().bootstrap_based_95p().upper());
  }
  absl::StrAppend(report, "\n");

  absl::StrAppend(report, "MRR@", eval.ranking().mrr_truncation(), ": ",
                  MRR(eval));
  if (eval.ranking().mrr().has_bootstrap_based_95p()) {
    absl::SubstituteAndAppend(
        report, " CI95[B][$0 $1]",
        eval.ranking().mrr().bootstrap_based_95p().lower(),
        eval.ranking().mrr().bootstrap_based_95p().upper());
  }
  absl::StrAppend(report, "\n");

  absl::StrAppend(report, "Precision@1: ", PrecisionAt1(eval));
  if (eval.ranking().precision_at_1().has_bootstrap_based_95p()) {
    absl::SubstituteAndAppend(
        report, " CI95[B][$0 $1]",
        eval.ranking().precision_at_1().bootstrap_based_95p().lower(),
        eval.ranking().precision_at_1().bootstrap_based_95p().upper());
  }
  absl::StrAppend(report, "\n");

  absl::StrAppend(report, "Default NDCG@", eval.ranking().ndcg_truncation(),
                  ": ", DefaultNDCG(eval), "\n");
  absl::StrAppend(report, "Number of groups: ", eval.ranking().num_groups(),
                  "\n");
  absl::StrAppend(report, "Number of items in groups: mean:",
                  eval.ranking().mean_num_items_in_group(),
                  " min:", eval.ranking().min_num_items_in_group(),
                  " max:", eval.ranking().max_num_items_in_group(), "\n");
  return absl::OkStatus();
}

absl::Status AppendTextReportUplift(const proto::EvaluationResults& eval,
                                    std::string* report) {
  absl::StrAppend(
      report, "Number of treatments: ", eval.uplift().num_treatments(), "\n");
  absl::StrAppend(report, "AUUC: ", AUUC(eval), "\n");
  absl::StrAppend(report, "Qini: ", Qini(eval), "\n");
  return absl::OkStatus();
}

absl::Status AppendHtmlReport(const proto::EvaluationResults& eval,
                              std::string* html_report,
                              const HtmlReportOptions& options) {
  namespace h = utils::html;

  h::Html html;

  if (options.include_title) {
    html.Append(h::H1("Evaluation report"));
  }

  if (options.include_text_report) {
    ASSIGN_OR_RETURN(const auto text_report, TextReport(eval));
    html.Append(h::Pre(text_report));
  }

  switch (eval.type_case()) {
    case proto::EvaluationResults::TypeCase::kClassification:
      RETURN_IF_ERROR(AppendHtmlReportClassiciation(eval, options, &html));
      break;

    case proto::EvaluationResults::TypeCase::kRegression:
      RETURN_IF_ERROR(AppendHtmlReportRegression(eval, options, &html));
      break;

    default:
      break;
  }

  absl::StrAppend(html_report, std::string(html.content()));
  return absl::OkStatus();
}

}  // namespace metric
}  // namespace yggdrasil_decision_forests
