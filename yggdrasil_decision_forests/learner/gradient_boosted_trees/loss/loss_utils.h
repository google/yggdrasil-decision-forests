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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_UTILS_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_UTILS_H_

#include <math.h>
#include <stdint.h>

#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

using row_t = dataset::VerticalDataset::row_t;

// Maximum number of items in a ranking group (e.g. maximum number of queries
// for a document). While possible, it is very unlikely that a user would exceed
// this value. A most likely scenario would be a
// configuration/dataset-preparation error.
constexpr int64_t kMaximumItemsInRankingGroup = 2000;

constexpr int kNDCG5Truncation = 5;

// Index of the secondary metrics according to the type of loss.
constexpr int kBinomialLossSecondaryMetricClassificationIdx = 0;

// Minimum length of the hessian (i.e. denominator) in the Newton step
// optimization.
constexpr float kMinHessianForNewtonStep = 0.001f;

// Ensures that the value is finite i.e. not NaN and not infinite.
// This is a no-op in release mode.
template <typename T>
void DCheckIsFinite(T v) {
  DCHECK(!std::isnan(v) && !std::isinf(v));
}

// Set a leaf's value using one stop of the Newton–Raphson method.
// The label statistics should contain gradients + hessian values.
absl::Status SetLeafValueWithNewtonRaphsonStep(
    const proto::GradientBoostedTreesTrainingConfig& gbt_config_,
    const decision_tree::proto::LabelStatistics& label_statistics,
    decision_tree::proto::Node* node);

void UpdatePredictionWithMultipleUnivariateTrees(
    const dataset::VerticalDataset& dataset,
    const std::vector<const decision_tree::DecisionTree*>& trees,
    std::vector<float>* predictions, double* mean_abs_prediction);

void UpdatePredictionWithSingleUnivariateTree(
    const dataset::VerticalDataset& dataset,
    const decision_tree::DecisionTree& tree, std::vector<float>* predictions,
    double* mean_abs_prediction);

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_UTILS_H_
