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

// Mean squared Error loss.
// Suited for univariate regression.
// See "AbstractLoss" for the method documentation.
class MeanSquaredErrorLoss : public AbstractLoss {
 public:
  // For unit testing.
  using AbstractLoss::Loss;
  using AbstractLoss::UpdateGradients;

  MeanSquaredErrorLoss(
      const proto::GradientBoostedTreesTrainingConfig& gbt_config,
      model::proto::Task task, const dataset::proto::Column& label_column)
      : AbstractLoss(gbt_config, task, label_column) {}

  absl::Status Status() const override;

  LossShape Shape() const override {
    return LossShape{/*.gradient_dim =*/1, /*.prediction_dim =*/1,
                     /*.has_hessian =*/false};
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
    RETURN_IF_ERROR(decision_tree::SetRegressionLabelDistribution(
        train_dataset, selected_examples, weights, config_link,
        node->mutable_node()));

    // Set the value of the leaf to be the residual:
    //   label[i] - prediction
    ASSIGN_OR_RETURN(
        const auto* labels,
        train_dataset.ColumnWithCastWithStatus<
            dataset::VerticalDataset::NumericalColumn>(label_col_idx));
    double sum_weighted_values = 0;
    double sum_weights = 0;
    if constexpr (!weighted) {
      sum_weights = selected_examples.size();
    }
    for (const auto example_idx : selected_examples) {
      const float label = labels->values()[example_idx];
      const float prediction = predictions[example_idx];
      if constexpr (weighted) {
        sum_weighted_values += weights[example_idx] * (label - prediction);
        sum_weights += weights[example_idx];
      } else {
        sum_weighted_values += label - prediction;
      }
    }
    if (sum_weights <= 0) {
      LOG(WARNING) << "Zero or negative weights in node";
      sum_weights = 1.0;
    }
    // Note: The "sum_weights" terms carries an implicit 2x factor that is
    // integrated in the shrinkage. We don't integrate this factor here not to
    // change the behavior of existing training configurations.
    node->mutable_node()->mutable_regressor()->set_top_value(
        gbt_config_.shrinkage() * sum_weighted_values /
        (sum_weights + gbt_config_.l2_regularization() / 2));
    return absl::OkStatus();
  }

  absl::StatusOr<decision_tree::SetLeafValueFromLabelStatsFunctor>
  SetLeafFunctorFromLabelStatistics() const override;

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

REGISTER_AbstractGradientBoostedTreeLoss(MeanSquaredErrorLoss, "SQUARED_ERROR");

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_MEAN_SQUARE_ERROR_H_
