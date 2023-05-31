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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_POISSON_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_POISSON_H_

#include <cmath>
#include <cstddef>
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
#include "yggdrasil_decision_forests/learner/types.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

// This class implements the Poisson log loss. This loss is suitable for
// regression problem where the label follows a Poisson distribution.
class PoissonLoss : public AbstractLoss {
 public:
  // For unit testing.
  using AbstractLoss::Loss;
  using AbstractLoss::UpdateGradients;

  PoissonLoss(const proto::GradientBoostedTreesTrainingConfig& gbt_config,
              model::proto::Task task,
              const dataset::proto::Column& label_column)
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

  static void UpdateGradientsImp(const std::vector<float>& labels,
                                 const std::vector<float>& predictions,
                                 size_t begin_example_idx,
                                 size_t end_example_idx,
                                 std::vector<float>* gradient_data);

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
    RETURN_IF_ERROR(decision_tree::SetRegressionLabelDistribution<weighted>(
        train_dataset, selected_examples, weights, config_link,
        node->mutable_node()));

    // Set the value of the leaf to be the residual:
    //   log(\sum w_i * label_i) - log(\sum w_i exp(pred_i))
    ASSIGN_OR_RETURN(
        const auto* labels,
        train_dataset.ColumnWithCastWithStatus<
            dataset::VerticalDataset::NumericalColumn>(label_col_idx));
    double sum_labels = 0;
    double sum_exp_predictions = 0;
    for (const auto example_idx : selected_examples) {
      const float label = labels->values()[example_idx];
      const float prediction = predictions[example_idx];
      if constexpr (weighted) {
        const float weight = weights[example_idx];
        sum_labels += weight * label;
        sum_exp_predictions += weight * std::exp(prediction);
      } else {
        sum_labels += label;
        sum_exp_predictions += std::exp(prediction);
      }
    }
    // TODO: Revise clamping. Note: R implements an e+19 / e-19
    // clamping for poisson loss.
    STATUS_CHECK_GE(sum_labels, 0);
    double top_value;
    if (sum_labels == 0) {
      top_value = -19;
    } else {
      top_value = std::log(sum_labels) - std::log(sum_exp_predictions);
    }
    STATUS_CHECK_GT(sum_exp_predictions, 0);

    node->mutable_node()->mutable_regressor()->set_top_value(
        gbt_config_.shrinkage() * top_value);
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

  template <bool use_weights>
  static void LossImp(const std::vector<float>& labels,
                      const std::vector<float>& predictions,
                      const std::vector<float>& weights,
                      size_t begin_example_idx, size_t end_example_idx,
                      double* __restrict sum_loss,
                      double* __restrict sum_square_error,
                      double* __restrict total_example_weight);
};
REGISTER_AbstractGradientBoostedTreeLoss(PoissonLoss, "POISSON");

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_POISSON_H_
