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

  decision_tree::CreateSetLeafValueFunctor SetLeafFunctor(
      const std::vector<float>& predictions,
      const std::vector<GradientData>& gradients,
      int label_col_idx) const override;

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
      DCHECK_EQ(train_dataset.nrow(), weights.size());
    } else {
      DCHECK(weights.empty());
    }
    if (!gbt_config_.use_hessian_gain()) {
      if (weights.empty()) {
        RETURN_IF_ERROR(
            decision_tree::SetRegressionLabelDistribution(
                train_dataset, selected_examples, weights, config_link,
                node->mutable_node()));
      } else {
        RETURN_IF_ERROR(
            decision_tree::SetRegressionLabelDistribution(
                train_dataset, selected_examples, weights, config_link,
                node->mutable_node()));
      }
      // Even if "use_hessian_gain" is not enabled for the splits. We use a
      // Newton step in the leaves i.e. if "use_hessian_gain" is false, we need
      // all the information.
    }

    // `labels` is not owning.
    ASSIGN_OR_RETURN(
        const dataset::VerticalDataset::CategoricalColumn* labels,
        train_dataset.ColumnWithCastWithStatus<
            dataset::VerticalDataset::CategoricalColumn>(label_col_idx));

    double numerator = 0;
    double denominator = 0;
    double sum_weights = 0;
    if constexpr (!weighted) {
      sum_weights = selected_examples.size();
    }
    for (const auto example_idx : selected_examples) {
      const bool is_positive = labels->values()[example_idx] == 2;
      const float prediction = predictions[example_idx];
      const FocalLossGradientData gradient_data =
          CalculateFocalLossGradient(is_positive, prediction, gamma_, alpha_);
      DCheckIsFinite(gradient_data.gradient);

      const double hessian =
          CalculateFocalLossHessian(gradient_data, gamma_, alpha_);
      DCheckIsFinite(hessian);
      if constexpr (weighted) {
        const float weight = weights[example_idx];
        numerator += weight * gradient_data.gradient;
        denominator += weight * hessian;
        sum_weights += weight;
      } else {
        numerator += gradient_data.gradient;
        denominator += hessian;
      }
      DCheckIsFinite(numerator);
      DCheckIsFinite(denominator);
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

    const auto leaf_value =
        gbt_config_.shrinkage() *
        static_cast<float>(decision_tree::l1_threshold(
                               numerator, gbt_config_.l1_regularization()) /
                           (denominator + gbt_config_.l2_regularization()));

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
