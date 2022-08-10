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

#ifndef YGGDRASIL_DECISION_FORESTS_METRIC_LABELS_H_
#define YGGDRASIL_DECISION_FORESTS_METRIC_LABELS_H_

namespace yggdrasil_decision_forests {
namespace metric {

// Label when exporting metrics into tables and plots. These metrics are defined
// in "documentation/metrics.md".
constexpr char kLabelModel[] = "Model";
constexpr char kLabelTrainingDuration[] = "Training duration (s)";
constexpr char kLabelAccuracy[] = "Accuracy";
// Note: [W] stands for "Wilson Score Interval".
constexpr char kLabelAccuracyConfidenceBounds95p[] = "Accuracy CI95% [W]";
constexpr char kLabelAUC[] = "AUC";
// Note: [H] stands for "Hanley definition".
constexpr char kLabelAUCConfidenceBounds95p[] = "AUC CI95% [H]";
// Note: [B] stands for "Bootstrapping".
constexpr char kLabelAUCConfidenceBoundsBootstrap95p[] = "AUC CI95% [B]";
constexpr char kLabelPRAUC[] = "P/R AUC";
// Note: [L] stands for "logistic interval".
constexpr char kLabelPRAUCConfidenceBounds95p[] = "P/R AUC CI95% [L]";
constexpr char kLabelPRAUCConfidenceBoundsBootstrap95p[] = "P/R AUC CI95% [B]";
constexpr char kLabelAP[] = "AP";
constexpr char kLabelAPConfidenceBoundsBootstrap95p[] = "AP CI95% [B]";
constexpr char kLabelRmse[] = "Rmse";
constexpr char kLabelRmseConfidenceBoundsBootstrap95p[] = "Rmse CI95% [B]";
constexpr char kLabelRmseConfidenceBoundsChi95p[] = "Rmse CI95% [X2]";
constexpr char kLabelNdcg[] = "NDCG";
constexpr char kLabelNdcgConfidenceBoundsBootstrap95p[] = "NDCG CI95% [B]";
constexpr char kLabelMrr[] = "MRR";
constexpr char kLabelMrrConfidenceBoundsBootstrap95p[] = "MRR CI95% [B]";

}  // namespace metric
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_METRIC_LABELS_H_
