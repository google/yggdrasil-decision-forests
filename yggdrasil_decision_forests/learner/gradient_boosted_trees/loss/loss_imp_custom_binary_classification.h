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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_CUSTOM_BINARY_CLASSIFICATION_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_CUSTOM_BINARY_CLASSIFICATION_H_

#include <stddef.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

// Functionals for computing a custom binary classification loss.
struct CustomBinaryClassificationLossFunctions {
  // Functional to return the initial predictions.
  std::function<absl::StatusOr<float>(
      const absl::Span<const int32_t> /*labels*/,
      const absl::Span<const float> /*weights*/)>
      initial_predictions;
  // Functional to return the loss of the current predictions.
  std::function<absl::StatusOr<float>(
      const absl::Span<const int32_t> /*labels*/,
      const absl::Span<const float> /*predictions*/,
      const absl::Span<const float> /*weights*/)>
      loss;
  // Functional to compute the gradient and the hessian of the current
  // predictions.
  std::function<absl::Status(const absl::Span<const int32_t> /*labels*/,
                             const absl::Span<const float> /*predictions*/,
                             absl::Span<float> /*gradients*/,
                             absl::Span<float> /*hessian*/)>
      gradient_and_hessian;
};

// Custom loss implementation suited for binary classification.
// The loss function is specified by the user through the constructor.
// See the tests for a sample loss.
// See `AbstractLoss` for the method documentation.
class CustomBinaryClassificationLoss : public AbstractLoss {
 public:
  // For unit testing.
  using AbstractLoss::Loss;
  using AbstractLoss::UpdateGradients;

  CustomBinaryClassificationLoss(
      const proto::GradientBoostedTreesTrainingConfig& gbt_config,
      model::proto::Task task, const dataset::proto::Column& label_column,
      const CustomBinaryClassificationLossFunctions& custom_loss_functions)
      : AbstractLoss(gbt_config, task, label_column),
        custom_loss_functions_(custom_loss_functions) {}

  absl::Status Status() const override;

  LossShape Shape() const override {
    return LossShape{.gradient_dim = 1, .prediction_dim = 1};
  }

  absl::StatusOr<std::vector<float>> InitialPredictions(
      const dataset::VerticalDataset& dataset, int label_col_idx,
      const std::vector<float>& weights) const override;

  absl::StatusOr<std::vector<float>> InitialPredictions(
      const decision_tree::proto::LabelStatistics& label_statistics)
      const override;

  absl::Status UpdateGradients(
      const std::vector<int32_t>& labels, const std::vector<float>& predictions,
      const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
      utils::RandomEngine* random,
      utils::concurrency::ThreadPool* thread_pool) const override;

  std::vector<std::string> SecondaryMetricNames() const override;

  absl::StatusOr<LossResults> Loss(
      const std::vector<int32_t>& labels, const std::vector<float>& predictions,
      const std::vector<float>& weights,
      const RankingGroupsIndices* ranking_index,
      utils::concurrency::ThreadPool* thread_pool) const override;

 private:
  CustomBinaryClassificationLossFunctions custom_loss_functions_;
};

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_CUSTOM_BINARY_CLASSIFICATION_H_
