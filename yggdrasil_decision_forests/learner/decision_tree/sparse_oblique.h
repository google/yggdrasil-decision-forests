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

// Implementation and extension of the "Sparse Projection Oblique Random
// Forests" 2020 JMLR paper by Tomita et al.
// https://www.jmlr.org/papers/volume21/18-664/18-664.pdf
//
// Algorithm:
// During training, at each node, the algorithm samples multiple random sparse
// linear projections of the numerical features, and evaluate then as classical
// numerical features (i.e. looking for a split projection >= threshold).
//
// Experiments in Tomita's paper indicates that this split algorithm used with
// Random Forest can leads to improvements over classical (RF) and other
// random-projection-oblique (RR-RF, CCF) random forest algorithms. These
// results have been confirmed experimentally using this implementation.
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPARSE_OBLIQUE_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPARSE_OBLIQUE_H_

#include <random>
#include <vector>

#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

// The following three "FindBestConditionSparseOblique" functions are searching
// for the best sparse oblique split for different objectives / loss functions.
// These methods only differ by the type of the "label_stats" argument.

// Classification.
utils::StatusOr<bool> FindBestConditionSparseOblique(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const ClassificationLabelStats& label_stats,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache);

// Regression with hessian term.
utils::StatusOr<bool> FindBestConditionSparseOblique(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionHessianLabelStats& label_stats,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache);

// Regression.
utils::StatusOr<bool> FindBestConditionSparseOblique(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionLabelStats& label_stats,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache);

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPARSE_OBLIQUE_H_
