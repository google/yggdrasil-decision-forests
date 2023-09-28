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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_BINOMIAL_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_BINOMIAL_H_

#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_utils.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

// Binomial log-likelihood loss.
// Suited for binary classification.
// See "AbstractLoss" for the method documentation.
class BinomialLogLikelihoodLoss : public AbstractLoss {
 public:
  // For unit testing.
  using AbstractLoss::Loss;
  using AbstractLoss::UpdateGradients;

  BinomialLogLikelihoodLoss(
      const proto::GradientBoostedTreesTrainingConfig& gbt_config,
      model::proto::Task task, const dataset::proto::Column& label_column)
      : AbstractLoss(gbt_config, task, label_column) {}

  absl::Status Status() const override;

  LossShape Shape() const override {
    return LossShape{.gradient_dim = 1, .prediction_dim = 1};
  };

  // Returns the initial predictions on the dataset.
  //
  // `weights` may be empty, which is interpreted as unit weights.
  // Returns log(y/(1-y)) with y the weighted ratio of positive labels.
  absl::StatusOr<std::vector<float>> InitialPredictions(
      const dataset::VerticalDataset& dataset, int label_col_idx,
      const std::vector<float>& weights) const override;

  virtual absl::StatusOr<std::vector<float>> InitialPredictions(
      const decision_tree::proto::LabelStatistics& label_statistics)
      const override;

  template <typename T>
  absl::Status TemplatedUpdateGradients(
      const std::vector<T>& labels, const std::vector<float>& predictions,
      const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
      utils::RandomEngine* random,
      utils::concurrency::ThreadPool* thread_pool) const;

  template <typename T>
  static void TemplatedUpdateGradientsImp(const std::vector<T>& labels,
                                          const std::vector<float>& predictions,
                                          size_t begin_example_idx,
                                          size_t end_example_idx,
                                          std::vector<float>* gradient_data,
                                          std::vector<float>* hessian_data);

  absl::Status UpdateGradients(
      const std::vector<int16_t>& labels, const std::vector<float>& predictions,
      const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
      utils::RandomEngine* random,
      utils::concurrency::ThreadPool* thread_pool) const override;

  absl::Status UpdateGradients(
      const std::vector<int32_t>& labels, const std::vector<float>& predictions,
      const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
      utils::RandomEngine* random,
      utils::concurrency::ThreadPool* thread_pool) const override;


  std::vector<std::string> SecondaryMetricNames() const override;

  template <typename T>
  absl::StatusOr<LossResults> TemplatedLoss(
      const std::vector<T>& labels, const std::vector<float>& predictions,
      const std::vector<float>& weights,
      const RankingGroupsIndices* ranking_index,
      utils::concurrency::ThreadPool* thread_pool) const;

  template <bool use_weights, typename T>
  static void TemplatedLossImp(
      const std::vector<T>& labels, const std::vector<float>& predictions,
      const std::vector<float>& weights, size_t begin_example_idx,
      size_t end_example_idx, double* __restrict sum_loss,
      utils::IntegersConfusionMatrixDouble* confusion_matrix);

  // Returns the loss of the given predictions.
  //
  // `weights` may be empty, which is interpreted as unit weights.
  absl::StatusOr<LossResults> Loss(
      const std::vector<int32_t>& labels, const std::vector<float>& predictions,
      const std::vector<float>& weights,
      const RankingGroupsIndices* ranking_index,
      utils::concurrency::ThreadPool* thread_pool) const override;

  // Returns the loss of the given predictions.
  //
  // `weights` may be empty, which is interpreted as unit weights.
  absl::StatusOr<LossResults> Loss(
      const std::vector<int16_t>& labels, const std::vector<float>& predictions,
      const std::vector<float>& weights,
      const RankingGroupsIndices* ranking_index,
      utils::concurrency::ThreadPool* thread_pool) const override;
};

REGISTER_AbstractGradientBoostedTreeLoss(BinomialLogLikelihoodLoss,
                                         "BINOMIAL_LOG_LIKELIHOOD");

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_BINOMIAL_H_
