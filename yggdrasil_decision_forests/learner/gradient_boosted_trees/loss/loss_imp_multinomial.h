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

// Multinomial log likelihood loss.
// Suited for binary and multi-class classification.
// See "AbstractLoss" for the method documentation.
class MultinomialLogLikelihoodLoss : public AbstractLoss {
 public:
  // For unit testing.
  using AbstractLoss::Loss;
  using AbstractLoss::UpdateGradients;

  MultinomialLogLikelihoodLoss(
      const proto::GradientBoostedTreesTrainingConfig& gbt_config,
      model::proto::Task task, const dataset::proto::Column& label_column)
      : AbstractLoss(gbt_config, task, label_column) {
    dimension_ = label_column_.categorical().number_of_unique_values() - 1;
  }

  absl::Status Status() const override;

  LossShape Shape() const override {
    return LossShape{/*.gradient_dim =*/dimension_,
                     /*.prediction_dim =*/dimension_,
                     /*.has_hessian =*/gbt_config_.use_hessian_gain()};
  };

  // Returns the initial predictions on the dataset.
  //
  // `weights` may be empty, which is interpreted as unit weights.
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

  decision_tree::CreateSetLeafValueFunctor SetLeafFunctor(
      const std::vector<float>& predictions,
      const std::vector<GradientData>& gradients,
      int label_col_idx) const override;

  // Sets the gain at leaf `node`.
  //
  // `weights` may be empty, which is interpreted as unit weights.
  template <bool weighted>
  absl::Status SetLeaf(const dataset::VerticalDataset& train_dataset,
                       const std::vector<UnsignedExampleIdx>& selected_examples,
                       const std::vector<float>& weights,
                       const model::proto::TrainingConfig& config,
                       const model::proto::TrainingConfigLinking& config_link,
                       const std::vector<float>& predictions,
                       const int label_col_idx,
                       decision_tree::NodeWithChildren* node) const {
    if constexpr (weighted) {
      STATUS_CHECK(weights.size() == train_dataset.nrow());
    } else {
      STATUS_CHECK(weights.empty());
    }
    // Initialize the distribution (as the "top_value" is overridden right
    // after.
    if (!gbt_config_.use_hessian_gain()) {
      RETURN_IF_ERROR(decision_tree::SetRegressionLabelDistribution(
          train_dataset, selected_examples, weights, config_link,
          node->mutable_node()));
    }

    // Set the value of the leaf to:
    //  (dim-1) / dim * ( \sum_i weight[i] grad[i] ) / (\sum_i |grad[i]| *
    //  (1-|grad[i]|))
    //
    // Note: The leaf value does not depend on the label value (directly).
    ASSIGN_OR_RETURN(
        const auto& column,
        train_dataset.ColumnWithCastWithStatus<
            dataset::VerticalDataset::NumericalColumn>(config_link.label()));
    const auto& grad = column->values();

    double numerator = 0;
    double denominator = 0;
    double sum_weights = 0;
    if constexpr (!weighted) {
      sum_weights = selected_examples.size();
    }
    for (const auto example_idx : selected_examples) {
      const float abs_grad = std::abs(grad[example_idx]);
      if constexpr (weighted) {
        const float weight = weights[example_idx];
        numerator += weight * grad[example_idx];
        denominator += weight * abs_grad * (1 - abs_grad);
        sum_weights += weight;
      } else {
        numerator += grad[example_idx];
        denominator += abs_grad * (1 - abs_grad);
      }
      DCheckIsFinite(numerator);
      DCheckIsFinite(denominator);
    }
    if (!std::isfinite(numerator) || !std::isfinite(denominator)) {
      return absl::InternalError("SetLeaf found invalid predictions");
    }

    if (denominator <= kMinHessianForNewtonStep) {
      denominator = kMinHessianForNewtonStep;
    }

    if (gbt_config_.use_hessian_gain()) {
      auto* reg = node->mutable_node()->mutable_regressor();
      reg->set_sum_gradients(numerator);
      reg->set_sum_hessians(denominator);
      reg->set_sum_weights(sum_weights);
    }

    numerator *= dimension_ - 1;
    denominator *= dimension_;
    const auto leaf_value =
        gbt_config_.shrinkage() *
        static_cast<float>(decision_tree::l1_threshold(
                               numerator, gbt_config_.l1_regularization()) /
                           (denominator + gbt_config_.l2_regularization()));
    DCheckIsFinite(leaf_value);

    node->mutable_node()->mutable_regressor()->set_top_value(
        utils::clamp(leaf_value, -gbt_config_.clamp_leaf_logit(),
                     gbt_config_.clamp_leaf_logit()));
    return absl::OkStatus();
  }

  absl::StatusOr<decision_tree::SetLeafValueFromLabelStatsFunctor>
  SetLeafFunctorFromLabelStatistics() const override {
    return [&](const decision_tree::proto::LabelStatistics& label_stats,
               decision_tree::proto::Node* node) {
      return SetLeafValueWithNewtonRaphsonStep(gbt_config_, label_stats, node);
    };
  }

  absl::Status UpdatePredictions(
      const std::vector<const decision_tree::DecisionTree*>& new_trees,
      const dataset::VerticalDataset& dataset, std::vector<float>* predictions,
      double* mean_abs_prediction) const override;

  std::vector<std::string> SecondaryMetricNames() const override;

  // Returns the loss of the given predictions.
  //
  // `weights` may be empty, which is interpreted as unit weights.
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

 private:
  int dimension_;
};

REGISTER_AbstractGradientBoostedTreeLoss(MultinomialLogLikelihoodLoss,
                                         "MULTINOMIAL_LOG_LIKELIHOOD");

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_MULTINOMIAL_H_
