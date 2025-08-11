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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_MEAN_SQUARE_ERROR_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_MEAN_SQUARE_ERROR_H_

#include <stddef.h>

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

// Mean squared Error loss.
// Suited for univariate regression.
// See "AbstractLoss" for the method documentation.
class MeanSquaredErrorLoss : public AbstractLoss {
 public:
  // For unit testing.
  using AbstractLoss::Loss;
  using AbstractLoss::UpdateGradients;

  MeanSquaredErrorLoss(const ConstructorArgs& args) : AbstractLoss(args) {}

  class Cache : public AbstractLossCache {
   public:
    // Only used to evaluate a ranking model
    std::unique_ptr<RankingGroupsIndices> ranking_index;

    virtual ~Cache() {}

    virtual absl::StatusOr<const RankingGroupsIndices*> ranking_indices()
        const override {
      if (!ranking_index) {
        return absl::InvalidArgumentError("Not ranking indices available");
      }
      return ranking_index.get();
    }
  };

  absl::StatusOr<std::unique_ptr<AbstractLossCache>> CreateLossCache(
      const dataset::VerticalDataset& dataset) const override;

  absl::StatusOr<std::unique_ptr<AbstractLossCache>> CreateRankingLossCache(
      absl::Span<const float> labels,
      absl::Span<const uint64_t> groups) const override;

  static absl::StatusOr<std::unique_ptr<AbstractLoss>> RegistrationCreate(
      const ConstructorArgs& args);

  LossShape Shape() const override {
    return LossShape{.gradient_dim = 1, .prediction_dim = 1};
  };

  absl::StatusOr<std::vector<float>> InitialPredictions(
      const dataset::VerticalDataset& dataset, int label_col_idx,
      const absl::Span<const float> weights) const override;

  absl::StatusOr<std::vector<float>> InitialPredictions(
      const decision_tree::proto::LabelStatistics& label_statistics)
      const override;

  absl::Status UpdateGradients(
      const absl::Span<const float> labels,
      const absl::Span<const float> predictions, const AbstractLossCache* cache,
      GradientDataRef* gradients, utils::RandomEngine* random,
      utils::concurrency::ThreadPool* thread_pool) const override;

  std::vector<std::string> SecondaryMetricNames() const override;

  absl::StatusOr<LossResults> Loss(
      const absl::Span<const float> labels,
      const absl::Span<const float> predictions,
      const absl::Span<const float> weights, const AbstractLossCache* cache,
      utils::concurrency::ThreadPool* thread_pool) const override;
};

REGISTER_AbstractGradientBoostedTreeLoss(MeanSquaredErrorLoss, "SQUARED_ERROR");

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_MEAN_SQUARE_ERROR_H_
