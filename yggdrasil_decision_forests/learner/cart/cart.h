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

// Classification And Regression Tree (CART) is a family of learning algorithm
// for single decision trees.
//
// CART models are less accurate but more easily interpretable than Random
// Forests or Gradient Boosted Decision Trees.
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_CART_CART_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_CART_CART_H_

#include <memory>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/cart/cart.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace cart {

class CartLearner : public AbstractLearner {
 public:
  explicit CartLearner(const model::proto::TrainingConfig& training_config);

  // Unique identifier of the learning algorithm.
  static constexpr char kRegisteredName[] = "CART";

  // Generic hyper parameter names.
  static constexpr char kHParamValidationRatio[] = "validation_ratio";

  utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset = {}) const override;

  // Sets the hyper-parameters of the learning algorithm from "generic hparams".
  absl::Status SetHyperParametersImpl(
      utils::GenericHyperParameterConsumer* generic_hyper_params) override;

  // Gets the available generic hparams.
  utils::StatusOr<model::proto::GenericHyperParameterSpecification>
  GetGenericHyperParameterSpecification() const override;

  // Pre-defined hyper-parameter space for hyper-parameter optimization.
  utils::StatusOr<model::proto::HyperParameterSpace>
  PredefinedHyperParameterSpace() const override;

  // Gets a description of what the learning algorithm can do.
  model::proto::LearnerCapabilities Capabilities() const override {
    model::proto::LearnerCapabilities capabilities;
    capabilities.set_support_max_training_duration(true);
    return capabilities;
  }
};

REGISTER_AbstractLearner(CartLearner, CartLearner::kRegisteredName);

namespace internal {

// Prune a decision tree using a validation dataset.
//
// For each non-leaf node, test if the validation metric (accuracy or rmse)
// would be better if the node was a leaf. If true, the node is turned into a
// leaf and the children nodes are pruned.
absl::Status PruneTree(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<float>& weights,
    const std::vector<dataset::VerticalDataset::row_t>& example_idxs,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    model::decision_tree::DecisionTree* tree);

}  // namespace internal

}  // namespace cart
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_CART_CART_H_
