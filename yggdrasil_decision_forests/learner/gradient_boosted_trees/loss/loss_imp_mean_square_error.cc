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

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

absl::StatusOr<std::unique_ptr<AbstractLoss>>
MeanSquaredErrorLoss::RegistrationCreate(const ConstructorArgs& args) {
  if (args.task != model::proto::Task::REGRESSION &&
      args.task != model::proto::Task::RANKING) {
    return absl::InvalidArgumentError(
        "Mean squared error loss is only compatible with a "
        "regression or ranking task");
  }
  return absl::make_unique<MeanSquaredErrorLoss>(args);
}

absl::StatusOr<std::vector<float>> MeanSquaredErrorLoss::InitialPredictions(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const absl::Span<const float> weights) const {
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
    const absl::Span<const float> labels,
    const absl::Span<const float> predictions, const AbstractLossCache* cache,
    GradientDataRef* gradients, utils::RandomEngine* random,
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
    const absl::Span<const float> labels,
    const absl::Span<const float> predictions,
    const absl::Span<const float> weights, const AbstractLossCache* cache,
    utils::concurrency::ThreadPool* thread_pool) const {
  constexpr int kNDCG5Truncation = 5;
  float loss_value;
  // The RMSE is also the loss.
  ASSIGN_OR_RETURN(loss_value, metric::RMSE(labels, predictions, weights));

  std::vector<float> secondary_metrics = {loss_value};
  if (task_ == model::proto::Task::RANKING) {
    STATUS_CHECK(cache);
    auto* ranking_index = static_cast<const Cache*>(cache)->ranking_index.get();
    secondary_metrics.push_back(
        ranking_index->NDCG(predictions, weights, kNDCG5Truncation));
  }
  return LossResults{/*.loss =*/loss_value,
                     /*.secondary_metrics =*/std::move(secondary_metrics)};
}

absl::StatusOr<std::unique_ptr<AbstractLossCache>>
MeanSquaredErrorLoss::CreateLossCache(
    const dataset::VerticalDataset& dataset) const {
  if (task_ != model::proto::Task::RANKING) {
    return std::unique_ptr<AbstractLossCache>();
  }
  // Only create a ranking index if the model is a ranker.
  auto cache = absl::make_unique<MeanSquaredErrorLoss::Cache>();
  cache->ranking_index = absl::make_unique<RankingGroupsIndices>();
  RETURN_IF_ERROR(cache->ranking_index->Initialize(
      dataset, train_config_link_.label(), train_config_link_.ranking_group()));
  return cache;
}

absl::StatusOr<std::unique_ptr<AbstractLossCache>>
MeanSquaredErrorLoss::CreateRankingLossCache(
    absl::Span<const float> labels, absl::Span<const uint64_t> groups) const {
  if (task_ != model::proto::Task::RANKING) {
    return std::unique_ptr<AbstractLossCache>();
  }
  // Only create a ranking index if the model is a ranker.
  auto cache = absl::make_unique<MeanSquaredErrorLoss::Cache>();
  cache->ranking_index = absl::make_unique<RankingGroupsIndices>();
  RETURN_IF_ERROR(cache->ranking_index->Initialize(labels, groups));
  return cache;
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
