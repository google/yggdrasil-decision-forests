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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_UTILS_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_UTILS_H_

#include <math.h>
#include <stdint.h>

#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

// Maximum number of items in a ranking group (e.g. maximum number of documents
// for a query). While possible, it is very unlikely that a user would exceed
// this value. The most likely scenario would be a
// configuration/dataset-preparation error. Since the running time is quadratic
// with the number of documents in a group, increasing this value further might
// allow very slow configurations.
//
// Since there exist a few valid use cases for large ranking groups, violating
// this maximum only triggers a stern warning.
constexpr int64_t kMaximumItemsInRankingGroup = 2048;

constexpr int kNDCG5Truncation = 5;

// Index of the secondary metrics according to the type of loss.
constexpr int kBinomialLossSecondaryMetricClassificationIdx = 0;

// Minimum length of the hessian (i.e. denominator) in the Newton step
// optimization.
constexpr double kMinHessianForNewtonStep = 0.001;

// Ensures that the value is finite i.e. not NaN and not infinite.
// This is a no-op in release mode.
template <typename T>
void DCheckIsFinite(T v) {
  DCHECK(!std::isnan(v) && !std::isinf(v));
}

// Set a leaf's value using one step of the Newton–Raphson method by using
// pre-computed and PRE-AGGREGATED gradient and hessians. The label statistics
// should contain gradients + hessian values.
absl::Status SetLeafValueWithNewtonRaphsonStep(
    const proto::GradientBoostedTreesTrainingConfig& gbt_config_,
    const decision_tree::proto::LabelStatistics& label_statistics,
    decision_tree::proto::Node* node);

// Creates a function to set leaf's values using one step of the Newton–Raphson
// method by using pre-computed (but not pre-aggregated) gradient and hessians.
decision_tree::CreateSetLeafValueFunctor
SetLeafValueWithNewtonRaphsonStepFunctor(
    const proto::GradientBoostedTreesTrainingConfig& gbt_config,
    const GradientData& gradients);

template <bool weighted>
absl::Status SetLeafValueWithNewtonRaphsonStep(
    const proto::GradientBoostedTreesTrainingConfig& gbt_config,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const GradientData& gradients,
    decision_tree::NodeWithChildren* node);

absl::Status UpdatePredictions(
    const std::vector<const decision_tree::DecisionTree*>& trees,
    const dataset::VerticalDataset& dataset, std::vector<float>* predictions,
    double* mean_abs_prediction);

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_UTILS_H_
