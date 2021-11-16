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

#include "yggdrasil_decision_forests/metric/report.h"

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/distribution.h"

namespace yggdrasil_decision_forests {
namespace metric {

namespace {

// Combine the two elements of a pair into a string.
template <typename T>
std::string PairToString(const std::pair<T, T>& p) {
  return absl::StrCat(p.first, " ", p.second);
}

}  // namespace

std::string TextReport(const proto::EvaluationResults& eval) {
  std::string report;
  AppendTextReport(eval, &report);
  return report;
}

void AppendTextReport(const proto::EvaluationResults& eval,
                      std::string* report) {
  absl::StrAppend(report, "Number of predictions (without weights): ",
                  eval.count_predictions_no_weight(), "\n");
  absl::StrAppend(report, "Number of predictions (with weights): ",
                  eval.count_predictions(), "\n");
  absl::StrAppend(report, "Task: ", model::proto::Task_Name(eval.task()), "\n");
  absl::StrAppend(report, "Label: ", eval.label_column().name(), "\n");
  if (eval.has_loss_value()) {
    absl::StrAppend(report, "Loss (", eval.loss_name(),
                    "): ", eval.loss_value(), "\n");
  }
  absl::StrAppend(report, "\n");

  switch (eval.type_case()) {
    case proto::EvaluationResults::TypeCase::kClassification:
      AppendTextReportClassification(eval, report);
      break;
    case proto::EvaluationResults::TypeCase::kRegression:
      AppendTextReportRegression(eval, report);
      break;
    case proto::EvaluationResults::TypeCase::kRanking:
      AppendTextReportRanking(eval, report);
      break;
    case proto::EvaluationResults::TypeCase::kUplift:
      AppendTextReportUplift(eval, report);
      break;
    default:
      LOG(FATAL) << "Not implemented";
  }
}

void AppendTextReportClassification(const proto::EvaluationResults& eval,
                                    std::string* report) {
  absl::SubstituteAndAppend(report, "Accuracy: $0  CI95[W][$1]\n",
                            Accuracy(eval),
                            PairToString(AccuracyConfidenceInterval(eval)));
  absl::StrAppend(report, "LogLoss: ", LogLoss(eval), "\n");
  absl::StrAppend(report, "ErrorRate: ", ErrorRate(eval), "\n");
  absl::StrAppend(report, "\n");

  absl::StrAppend(report, "Default Accuracy: ", DefaultAccuracy(eval), "\n");
  absl::StrAppend(report, "Default LogLoss: ", DefaultLogLoss(eval), "\n");
  absl::StrAppend(report, "Default ErrorRate: ", DefaultErrorRate(eval), "\n");
  absl::StrAppend(report, "\n");

  absl::StrAppend(report, "Confusion Table:\n");
  utils::IntegersConfusionMatrixDouble confusion;
  confusion.Load(eval.classification().confusion());
  confusion.AppendTextReport(eval.label_column(), report);
  absl::StrAppend(report, "\n");

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

  absl::StrAppend(report, "One vs other classes:\n");
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
}

void AppendTextReportRegression(const proto::EvaluationResults& eval,
                                std::string* report) {
  absl::StrAppend(report, "RMSE: ", RMSE(eval));

  const auto closed_ci = RMSEConfidenceInterval(eval);
  absl::SubstituteAndAppend(report, " CI95[X2][$0 $1]", closed_ci.first,
                            closed_ci.second);

  if (eval.regression().has_bootstrap_rmse_lower_bounds_95p()) {
    absl::SubstituteAndAppend(
        report, " CI95[B][$0 $1]",
        eval.regression().bootstrap_rmse_lower_bounds_95p(),
        eval.regression().bootstrap_rmse_upper_bounds_95p());
  }
  absl::StrAppend(report, "\n");

  absl::StrAppend(report, "Default RMSE: ", DefaultRMSE(eval), "\n");
}

void AppendTextReportRanking(const proto::EvaluationResults& eval,
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
  absl::StrAppend(report, "Numer of items in groups: mean:",
                  eval.ranking().mean_num_items_in_group(),
                  " min:", eval.ranking().min_num_items_in_group(),
                  " max:", eval.ranking().max_num_items_in_group(), "\n");
}

void AppendTextReportUplift(const proto::EvaluationResults& eval,
                            std::string* report) {
  absl::StrAppend(
      report, "Number of treatments: ", eval.uplift().num_treatments(), "\n");
  absl::StrAppend(report, "AUUC: ", AUUC(eval), "\n");
  absl::StrAppend(report, "Qini: ", Qini(eval), "\n");
}

}  // namespace metric
}  // namespace yggdrasil_decision_forests
