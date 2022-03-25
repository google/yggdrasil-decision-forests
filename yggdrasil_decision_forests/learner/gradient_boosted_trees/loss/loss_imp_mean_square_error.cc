/*
 * Copyright 2021 Google LLC.
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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_mean_square_error.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_utils.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {


absl::Status MeanSquaredErrorLoss::Status() const {
  if (task_ != model::proto::Task::REGRESSION &&
      task_ != model::proto::Task::RANKING) {
    return absl::InvalidArgumentError(
        "Mean squared error loss is only compatible with a "
        "regression or ranking task");
  }
  return absl::OkStatus();
}

utils::StatusOr<std::vector<float>> MeanSquaredErrorLoss::InitialPredictions(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const std::vector<float>& weights) const {
  // Note: The initial value is the weighted mean of the labels.
  double weighted_sum_values = 0;
  double sum_weights = 0;
  const auto* labels =
      dataset.ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
          label_col_idx);
  for (row_t example_idx = 0; example_idx < dataset.nrow(); example_idx++) {
    sum_weights += weights[example_idx];
    weighted_sum_values += weights[example_idx] * labels->values()[example_idx];
  }
  // Note: Null and negative weights are detected by the dataspec
  // computation.
  if (sum_weights <= 0) {
    return absl::InvalidArgumentError(
        "The sum of weights are null. The dataset is "
        "either empty or contains null weights.");
  }
  return std::vector<float>{
      static_cast<float>(weighted_sum_values / sum_weights)};
}

utils::StatusOr<std::vector<float>> MeanSquaredErrorLoss::InitialPredictions(
    const decision_tree::proto::LabelStatistics& label_statistics) const {
  const auto stats = label_statistics.regression().labels();
  return std::vector<float>{static_cast<float>(stats.sum() / stats.count())};
}

absl::Status MeanSquaredErrorLoss::UpdateGradients(
    const std::vector<float>& labels, const std::vector<float>& predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  // TODO(gbm): Implement thread_pool.

  // Set the gradient to:
  //   label - prediction
  if (gradients->size() != 1) {
    return absl::InternalError("Wrong gradient shape");
  }
  const auto num_examples = labels.size();
  std::vector<float>& gradient_data = *(*gradients)[0].gradient;
  for (size_t example_idx = 0; example_idx < num_examples; example_idx++) {
    const float label = labels[example_idx];
    const float prediction = predictions[example_idx];
    gradient_data[example_idx] = label - prediction;
  }
  return absl::OkStatus();
}

absl::Status MeanSquaredErrorLoss::UpdatePredictions(
    const std::vector<const decision_tree::DecisionTree*>& new_trees,
    const dataset::VerticalDataset& dataset, std::vector<float>* predictions,
    double* mean_abs_prediction) const {
  if (new_trees.size() != 1) {
    return absl::InternalError("Wrong number of trees");
  }
  UpdatePredictionWithSingleUnivariateTree(dataset, *new_trees.front(),
                                           predictions, mean_abs_prediction);
  return absl::OkStatus();
}

decision_tree::CreateSetLeafValueFunctor MeanSquaredErrorLoss::SetLeafFunctor(
    const std::vector<float>& predictions,
    const std::vector<GradientData>& gradients, const int label_col_idx) const {
  return
      [this, &predictions, label_col_idx](
          const dataset::VerticalDataset& train_dataset,
          const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
          const std::vector<float>& weights,
          const model::proto::TrainingConfig& config,
          const model::proto::TrainingConfigLinking& config_link,
          decision_tree::NodeWithChildren* node) {
        return SetLeaf(train_dataset, selected_examples, weights, config,
                       config_link, predictions, label_col_idx, node);
      };
}

void MeanSquaredErrorLoss::SetLeaf(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const std::vector<float>& predictions, const int label_col_idx,
    decision_tree::NodeWithChildren* node) const {
  decision_tree::SetRegressionLabelDistribution(
      train_dataset, selected_examples, weights, config_link,
      node->mutable_node());

  // Set the value of the leaf to be the residual:
  //   label[i] - prediction
  const auto* labels =
      train_dataset.ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
          label_col_idx);
  double sum_weighted_values = 0;
  double sum_weights = 0;
  for (const auto example_idx : selected_examples) {
    const float label = labels->values()[example_idx];
    const float prediction = predictions[example_idx];
    sum_weighted_values += weights[example_idx] * (label - prediction);
    sum_weights += weights[example_idx];
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
}

utils::StatusOr<decision_tree::SetLeafValueFromLabelStatsFunctor>
MeanSquaredErrorLoss::SetLeafFunctorFromLabelStatistics() const {
  return [&](const decision_tree::proto::LabelStatistics& label_stats,
             decision_tree::proto::Node* node) {
    if (!label_stats.has_regression()) {
      return absl::InternalError("No regression data available");
    }

    double denominator = label_stats.regression().labels().count();
    if (denominator <= 0) {
      LOG(WARNING) << "Zero or negative weights in node";
      denominator = 1.0;
    }

    const float leaf_value =
        gbt_config_.shrinkage() *
        (label_stats.regression().labels().sum() /
         (denominator + gbt_config_.l2_regularization() / 2));

    node->mutable_regressor()->set_top_value(leaf_value);
    return absl::OkStatus();
  };
}

std::vector<std::string> MeanSquaredErrorLoss::SecondaryMetricNames() const {
  if (task_ == model::proto::Task::RANKING) {
    return {"rmse", "NDCG@5"};
  } else {
    return {"rmse"};
  }
}

absl::Status MeanSquaredErrorLoss::Loss(
    const std::vector<float>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index, float* loss_value,
    std::vector<float>* secondary_metric,
    utils::concurrency::ThreadPool* thread_pool) const {
  // The RMSE is also the loss.
  if (weights.empty()) {
    *loss_value = metric::RMSE(labels, predictions);
  } else {
    *loss_value = metric::RMSE(labels, predictions, weights);
  }

  if (task_ == model::proto::Task::RANKING) {
    secondary_metric->resize(2);
    (*secondary_metric)[0] = *loss_value;
    (*secondary_metric)[1] =
        ranking_index->NDCG(predictions, weights, kNDCG5Truncation);
  } else {
    secondary_metric->resize(1);
    (*secondary_metric)[0] = *loss_value;
  }
  return absl::OkStatus();
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
