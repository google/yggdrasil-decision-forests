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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_CUSTOM_MULTI_CLASSIFICATION_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_CUSTOM_MULTI_CLASSIFICATION_H_

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

// Functionals for computing a custom multi-class classification loss.
struct CustomMultiClassificationLossFunctions {
  // Functional to compute the initial predictions, one per class.
  std::function<absl::Status(const absl::Span<const int32_t> /*labels*/,
                             const absl::Span<const float> /*weights*/,
                             absl::Span<float> /*initial_predictions*/)>
      initial_predictions;
  // Functional to return the loss of the current predictions.
  std::function<absl::StatusOr<float>(
      const absl::Span<const int32_t> /*labels*/,
      const absl::Span<const float> /*predictions*/,
      const absl::Span<const float> /*weights*/)>
      loss;
  // Functional to compute the gradient and the hessian of the current
  // predictions. The function must one gradient per class.
  std::function<absl::Status(const absl::Span<const int32_t> /*labels*/,
                             const absl::Span<const float> /*predictions*/,
                             absl::Span<const absl::Span<float>> /*gradients*/,
                             absl::Span<const absl::Span<float>> /*hessian*/)>
      gradient_and_hessian;
};

// Custom loss implementation suited for multi-class classification.
// The loss function is specified by the user through the constructor.
// See the tests for a sample loss.
// See `AbstractLoss` for the method documentation.
class CustomMultiClassificationLoss : public AbstractLoss {
 public:
  // For unit testing.
  using AbstractLoss::Loss;
  using AbstractLoss::UpdateGradients;

  CustomMultiClassificationLoss(
      const proto::GradientBoostedTreesTrainingConfig& gbt_config,
      model::proto::Task task, const dataset::proto::Column& label_column,
      const CustomMultiClassificationLossFunctions& custom_loss_functions)
      : AbstractLoss(gbt_config, task, label_column),
        custom_loss_functions_(custom_loss_functions) {
    dimension_ = label_column_.categorical().number_of_unique_values() - 1;
  }

  absl::Status Status() const override;

  LossShape Shape() const override;

  absl::StatusOr<std::vector<float>> InitialPredictions(
      const dataset::VerticalDataset& dataset, int label_col_idx,
      const absl::Span<const float> weights) const override;

  absl::StatusOr<std::vector<float>> InitialPredictions(
      const decision_tree::proto::LabelStatistics& label_statistics)
      const override;

  absl::Status UpdateGradients(
      const absl::Span<const int32_t> labels,
      const absl::Span<const float> predictions,
      const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
      utils::RandomEngine* random,
      utils::concurrency::ThreadPool* thread_pool) const override;

  std::vector<std::string> SecondaryMetricNames() const override;

  absl::StatusOr<LossResults> Loss(
      const absl::Span<const int32_t> labels,
      const absl::Span<const float> predictions,
      const absl::Span<const float> weights,
      const RankingGroupsIndices* ranking_index,
      utils::concurrency::ThreadPool* thread_pool) const override;

 private:
  int dimension_;
  CustomMultiClassificationLossFunctions custom_loss_functions_;
};

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_CUSTOM_MULTI_CLASSIFICATION_H_
