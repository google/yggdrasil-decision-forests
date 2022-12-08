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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_NDCG_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_NDCG_H_

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

// Normalized Discounted Cumulative Gain loss.
// Suited for ranking.
// See "AbstractLoss" for the method documentation.
class NDCGLoss : public AbstractLoss {
 public:
  // For unit testing.
  using AbstractLoss::Loss;
  using AbstractLoss::UpdateGradients;

  NDCGLoss(const proto::GradientBoostedTreesTrainingConfig& gbt_config,
           model::proto::Task task, const dataset::proto::Column& label_column)
      : AbstractLoss(gbt_config, task, label_column) {}

  absl::Status Status() const override;

  bool RequireGroupingAttribute() const override { return true; }

  LossShape Shape() const override {
    return LossShape{/*.gradient_dim =*/1, /*.prediction_dim =*/1,
                     /*.has_hessian =*/true};
  };

  absl::StatusOr<std::vector<float>> InitialPredictions(
      const dataset::VerticalDataset& dataset, int label_col_idx,
      const std::vector<float>& weights) const override;

  virtual absl::StatusOr<std::vector<float>> InitialPredictions(
      const decision_tree::proto::LabelStatistics& label_statistics)
      const override;

  absl::Status UpdateGradients(
      const std::vector<float>& labels, const std::vector<float>& predictions,
      const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
      utils::RandomEngine* random,
      utils::concurrency::ThreadPool* thread_pool) const override;

  decision_tree::CreateSetLeafValueFunctor SetLeafFunctor(
      const std::vector<float>& predictions,
      const std::vector<GradientData>& gradients,
      int label_col_idx) const override;

  absl::Status UpdatePredictions(
      const std::vector<const decision_tree::DecisionTree*>& new_trees,
      const dataset::VerticalDataset& dataset, std::vector<float>* predictions,
      double* mean_abs_prediction) const override;

  std::vector<std::string> SecondaryMetricNames() const override;

  absl::StatusOr<LossResults> Loss(
      const std::vector<float>& labels, const std::vector<float>& predictions,
      const std::vector<float>& weights,
      const RankingGroupsIndices* ranking_index,
      utils::concurrency::ThreadPool* thread_pool) const override;
};

REGISTER_AbstractGradientBoostedTreeLoss(NDCGLoss, "LAMBDA_MART_NDCG5");

template <bool weighted>
absl::Status SetLeafNDCG(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const std::vector<float>& predictions,
    const proto::GradientBoostedTreesTrainingConfig& gbt_config,
    const std::vector<GradientData>& gradients, const int label_col_idx,
    decision_tree::NodeWithChildren* node) {
  if constexpr (weighted) DCHECK_LE(selected_examples.size(), weights.size());
  if constexpr (!weighted) DCHECK(weights.empty());
  if (!gbt_config.use_hessian_gain()) {
    RETURN_IF_ERROR(decision_tree::SetRegressionLabelDistribution(
        train_dataset, selected_examples, weights, config_link,
        node->mutable_node()));
  }

  const auto& gradient_data = gradients.front().gradient;
  const auto& second_order_derivative_data = *(gradients.front().hessian);

  double sum_weighted_gradient = 0;
  double sum_weighted_second_order_derivative = 0;
  double sum_weights = 0;
  if constexpr (!weighted) {
    sum_weights = selected_examples.size();
  }
  for (const auto example_idx : selected_examples) {
    if constexpr (weighted) {
      const float weight = weights[example_idx];
      sum_weighted_gradient += weight * gradient_data[example_idx];
      sum_weighted_second_order_derivative +=
          weight * second_order_derivative_data[example_idx];
      sum_weights += weight;
    } else {
      sum_weighted_gradient += gradient_data[example_idx];
      sum_weighted_second_order_derivative +=
          second_order_derivative_data[example_idx];
    }
  }
  DCheckIsFinite(sum_weighted_gradient);
  DCheckIsFinite(sum_weighted_second_order_derivative);

  if (sum_weighted_second_order_derivative <= kMinHessianForNewtonStep) {
    sum_weighted_second_order_derivative = kMinHessianForNewtonStep;
  }

  if (gbt_config.use_hessian_gain()) {
    auto* reg = node->mutable_node()->mutable_regressor();
    reg->set_sum_gradients(sum_weighted_gradient);
    reg->set_sum_hessians(sum_weighted_second_order_derivative);
    reg->set_sum_weights(sum_weights);
  }

  node->mutable_node()->mutable_regressor()->set_top_value(
      gbt_config.shrinkage() *
      decision_tree::l1_threshold(sum_weighted_gradient,
                                  gbt_config.l1_regularization()) /
      (sum_weighted_second_order_derivative + gbt_config.l2_regularization()));
  return absl::OkStatus();
}
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_NDCG_H_
