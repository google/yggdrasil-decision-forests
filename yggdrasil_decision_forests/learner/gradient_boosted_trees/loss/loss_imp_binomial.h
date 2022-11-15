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
    return LossShape{/*.gradient_dim =*/1,
                     /*.prediction_dim =*/1,
                     /*.has_hessian =*/gbt_config_.use_hessian_gain()};
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
      DCHECK_LE(selected_examples.size(), weights.size());
    } else {
      DCHECK(weights.empty());
    }
    DCHECK_GE(gbt_config_.shrinkage(), 0);

    if (!gbt_config_.use_hessian_gain()) {
      RETURN_IF_ERROR(decision_tree::SetRegressionLabelDistribution(
          train_dataset, selected_examples, weights, config_link,
          node->mutable_node()));
      // Even if "use_hessian_gain" is not enabled for the splits, we use a
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
    // Set the value of the leaf to:
    //   (\sum_i weight[i] * (label[i] - p[i]) ) / (\sum_i weight[i] * p[i] *
    //   (1-p[i]))
    // with: p[i] = 1/(1+exp(-prediction)
    for (const auto example_idx : selected_examples) {
      // For binary classification, the positive examples correspond to class 2.
      const float label = labels->values()[example_idx] == 2;
      const float prediction = predictions[example_idx];
      const float p = 1.f / (1.f + std::exp(-prediction));
      if constexpr (weighted) {
        const float weight = weights[example_idx];
        numerator += weight * (label - p);
        denominator += weight * p * (1.f - p);
        sum_weights += weight;
      } else {
        numerator += (label - p);
        denominator += p * (1.f - p);
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
      auto* regressor = node->mutable_node()->mutable_regressor();
      regressor->set_sum_gradients(numerator);
      regressor->set_sum_hessians(denominator);
      regressor->set_sum_weights(sum_weights);
    }

    const float leaf_value =
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
