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

// Implementation of binary focal loss following
// https://arxiv.org/pdf/1708.02002.pdf.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_BINARY_FOCAL_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_BINARY_FOCAL_H_

#include <stddef.h>
#include <stdint.h>

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
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_binomial.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_utils.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

// Local helper functions for calculating the (slightly more involved)
// focal loss gradients and hessians at a common place to avoid code duplication
struct FocalLossBasicData {
  float y;      // Label as from {-1, 1} set to follow paper's notation.
  float label;  // Label as from {0, 1} set.
  float pt;     // Probability of "being right".
  // Log of prob of being right. Calculated directly from log odds.
  float log_pt;
  float mispred;  // Probability of miss-prediction.
  float at;       // Sample weight (alpha_t, depends on ground truth label)
};

struct FocalLossGradientData {
  FocalLossBasicData basic;
  float gradient;
  float term1;  // The first derivative term of the first derivative
  float term2;  // The second derivative term of the first derivative
};

FocalLossGradientData CalculateFocalLossGradient(bool is_positive,
                                                 float prediction, float gamma,
                                                 float alpha);

float CalculateFocalLossHessian(FocalLossGradientData gradient_data,
                                float gamma, float alpha);

// Focal loss.
// Suited for binary classification, implementation based on log loss
// For implementation details, see paper: https://arxiv.org/pdf/1708.02002.pdf
// Note: uses the label set of {-1, 1} in formula derivation
class BinaryFocalLoss : public BinomialLogLikelihoodLoss {
 public:
  // For unit testing.
  using AbstractLoss::Loss;
  using AbstractLoss::UpdateGradients;

  BinaryFocalLoss(const proto::GradientBoostedTreesTrainingConfig& gbt_config,
                  model::proto::Task task,
                  const dataset::proto::Column& label_column);

  absl::Status Status() const override;

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
                                          size_t end_example_idx, float gamma,
                                          float alpha,
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
  static void TemplatedLossImp(const std::vector<T>& labels,
                               const std::vector<float>& predictions,
                               const std::vector<float>& weights,
                               size_t begin_example_idx, size_t end_example_idx,
                               float gamma, float alpha,
                               double* __restrict sum_loss,
                               double* __restrict count_correct_predictions,
                               double* __restrict sum_weights);

  absl::StatusOr<LossResults> Loss(
      const std::vector<int32_t>& labels, const std::vector<float>& predictions,
      const std::vector<float>& weights,
      const RankingGroupsIndices* ranking_index,
      utils::concurrency::ThreadPool* thread_pool) const override;

  absl::StatusOr<LossResults> Loss(
      const std::vector<int16_t>& labels, const std::vector<float>& predictions,
      const std::vector<float>& weights,
      const RankingGroupsIndices* ranking_index,
      utils::concurrency::ThreadPool* thread_pool) const override;

 private:
  // Focusing parameter, when 0.0, then falls back to log loss.
  float gamma_;
  // Class weighting parameter, positives with alpha,
  // negatives with (1-alpha) weight. Usually tuned together with gamma.
  float alpha_;
};

REGISTER_AbstractGradientBoostedTreeLoss(BinaryFocalLoss, "BINARY_FOCAL_LOSS");

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_BINARY_FOCAL_H_
