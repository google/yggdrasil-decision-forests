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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_mean_square_error.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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

absl::StatusOr<std::vector<float>> MeanSquaredErrorLoss::InitialPredictions(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const std::vector<float>& weights) const {
  // Note: The initial value is the weighted mean of the labels.
  double weighted_sum_values = 0;
  double sum_weights = 0;
  ASSIGN_OR_RETURN(
      const auto* labels,
      dataset
          .ColumnWithCastWithStatus<dataset::VerticalDataset::NumericalColumn>(
              label_col_idx));
  if (weights.empty()) {
    sum_weights = dataset.nrow();
    weighted_sum_values =
        utils::accumulate(labels->values().begin(), labels->values().end(), 0.);
  } else {
    for (UnsignedExampleIdx example_idx = 0; example_idx < dataset.nrow();
         example_idx++) {
      sum_weights += weights[example_idx];
      weighted_sum_values +=
          weights[example_idx] * labels->values()[example_idx];
    }
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

absl::StatusOr<std::vector<float>> MeanSquaredErrorLoss::InitialPredictions(
    const decision_tree::proto::LabelStatistics& label_statistics) const {
  const auto stats = label_statistics.regression().labels();
  return std::vector<float>{static_cast<float>(stats.sum() / stats.count())};
}

absl::Status MeanSquaredErrorLoss::UpdateGradients(
    const std::vector<float>& labels, const std::vector<float>& predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  // TODO: Implement thread_pool.

  // Set the gradient to:
  //   label - prediction
  if (gradients->size() != 1) {
    return absl::InternalError("Wrong gradient shape");
  }
  const auto num_examples = labels.size();
  std::vector<float>& gradient_data = *(*gradients)[0].gradient;
  std::vector<float>& hessian_data = *(*gradients)[0].hessian;
  DCHECK_EQ(gradient_data.size(), hessian_data.size());

  for (size_t example_idx = 0; example_idx < num_examples; example_idx++) {
    const float label = labels[example_idx];
    const float prediction = predictions[example_idx];
    gradient_data[example_idx] = label - prediction;
    hessian_data[example_idx] = 1.f;
  }
  return absl::OkStatus();
}


std::vector<std::string> MeanSquaredErrorLoss::SecondaryMetricNames() const {
  if (task_ == model::proto::Task::RANKING) {
    return {"rmse", "NDCG@5"};
  } else {
    return {"rmse"};
  }
}

absl::StatusOr<LossResults> MeanSquaredErrorLoss::Loss(
    const std::vector<float>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index,
    utils::concurrency::ThreadPool* thread_pool) const {
  float loss_value;
  // The RMSE is also the loss.
  ASSIGN_OR_RETURN(loss_value, metric::RMSE(labels, predictions, weights));

  std::vector<float> secondary_metrics = {loss_value};
  if (task_ == model::proto::Task::RANKING) {
    secondary_metrics.push_back(
        ranking_index->NDCG(predictions, weights, kNDCG5Truncation));
  }
  return LossResults{/*.loss =*/loss_value,
                     /*.secondary_metrics =*/std::move(secondary_metrics)};
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
