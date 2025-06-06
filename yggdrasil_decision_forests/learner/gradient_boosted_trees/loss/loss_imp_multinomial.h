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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_MULTINOMIAL_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_MULTINOMIAL_H_

#include <stddef.h>
#include <stdint.h>

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
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

// Multinomial log likelihood loss.
// Suited for binary and multi-class classification.
// See "AbstractLoss" for the method documentation.
class MultinomialLogLikelihoodLoss : public AbstractLoss {
 public:
  // For unit testing.
  using AbstractLoss::Loss;
  using AbstractLoss::UpdateGradients;

  MultinomialLogLikelihoodLoss(const ConstructorArgs& args)
      : AbstractLoss(args) {
    dimension_ = label_column_.categorical().number_of_unique_values() - 1;
  }

  static absl::StatusOr<std::unique_ptr<AbstractLoss>> RegistrationCreate(
      const ConstructorArgs& args);

  LossShape Shape() const override {
    return LossShape{.gradient_dim = dimension_, .prediction_dim = dimension_};
  };

  // Returns the initial predictions on the dataset.
  //
  // `weights` may be empty, which is interpreted as unit weights.
  absl::StatusOr<std::vector<float>> InitialPredictions(
      const dataset::VerticalDataset& dataset, int label_col_idx,
      const absl::Span<const float> weights) const override;

  absl::StatusOr<std::vector<float>> InitialPredictions(
      const decision_tree::proto::LabelStatistics& label_statistics)
      const override;

  template <typename T>
  absl::Status TemplatedUpdateGradients(
      const absl::Span<T> labels, const absl::Span<const float> predictions,
      const AbstractLossCache* cache, GradientDataRef* gradients,
      utils::RandomEngine* random,
      utils::concurrency::ThreadPool* thread_pool) const;

  absl::Status UpdateGradients(
      const absl::Span<const int16_t> labels,
      const absl::Span<const float> predictions, const AbstractLossCache* cache,
      GradientDataRef* gradients, utils::RandomEngine* random,
      utils::concurrency::ThreadPool* thread_pool) const override;

  absl::Status UpdateGradients(
      const absl::Span<const int32_t> labels,
      const absl::Span<const float> predictions, const AbstractLossCache* cache,
      GradientDataRef* gradients, utils::RandomEngine* random,
      utils::concurrency::ThreadPool* thread_pool) const override;

  std::vector<std::string> SecondaryMetricNames() const override;

  // Returns the loss of the given predictions.
  //
  // `weights` may be empty, which is interpreted as unit weights.
  template <typename T>
  absl::StatusOr<LossResults> TemplatedLoss(
      const absl::Span<T> labels, const absl::Span<const float> predictions,
      const absl::Span<const float> weights, const AbstractLossCache* cache,
      utils::concurrency::ThreadPool* thread_pool) const;

  template <bool use_weights, typename T>
  static void TemplatedLossImp(
      const absl::Span<T> labels, const absl::Span<const float> predictions,
      const absl::Span<const float> weights, size_t begin_example_idx,
      size_t end_example_idx, double* __restrict sum_loss,
      utils::IntegersConfusionMatrixDouble* confusion_matrix);

  // Returns the loss of the given predictions.
  //
  // `weights` may be empty, which is interpreted as unit weights.
  absl::StatusOr<LossResults> Loss(
      const absl::Span<const int32_t> labels,
      const absl::Span<const float> predictions,
      const absl::Span<const float> weights, const AbstractLossCache* cache,
      utils::concurrency::ThreadPool* thread_pool) const override;
  // Returns the loss of the given predictions.
  //
  // `weights` may be empty, which is interpreted as unit weights.
  absl::StatusOr<LossResults> Loss(
      const absl::Span<const int16_t> labels,
      const absl::Span<const float> predictions,
      const absl::Span<const float> weights, const AbstractLossCache* cache,
      utils::concurrency::ThreadPool* thread_pool) const override;

 private:
  int dimension_;
};

REGISTER_AbstractGradientBoostedTreeLoss(MultinomialLogLikelihoodLoss,
                                         "MULTINOMIAL_LOG_LIKELIHOOD");

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_MULTINOMIAL_H_
