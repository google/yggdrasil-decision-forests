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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_poisson.h"

#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

absl::StatusOr<std::vector<float>> PoissonLoss::InitialPredictions(
    const dataset::VerticalDataset &dataset, int label_col_idx,
    const std::vector<float> &weights) const {
  // The initial value is the logarithm of the weighted mean of the labels.
  double weighted_sum_values = 0;
  double sum_weights = 0;
  STATUS_CHECK_EQ(dataset.data_spec().columns(label_col_idx).type(),
                  dataset::proto::ColumnType::NUMERICAL);
  STATUS_CHECK_GE(
      dataset.data_spec().columns(label_col_idx).numerical().min_value(), 0);
  ASSIGN_OR_RETURN(
      const auto *labels,
      dataset
          .ColumnWithCastWithStatus<dataset::VerticalDataset::NumericalColumn>(
              label_col_idx));
  if (weights.empty()) {
    sum_weights = dataset.nrow();
    weighted_sum_values =
        utils::accumulate(labels->values().begin(), labels->values().end(), 0.);
  } else {
    for (decision_tree::row_t example_idx = 0; example_idx < dataset.nrow();
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
  return std::vector<float>{static_cast<float>(std::log(weighted_sum_values) -
                                               std::log(sum_weights))};
}

absl::StatusOr<std::vector<float>> PoissonLoss::InitialPredictions(
    const decision_tree::proto::LabelStatistics &label_statistics) const {
  const auto stats = label_statistics.regression().labels();
  return std::vector<float>{
      static_cast<float>(std::log(stats.sum() / stats.count()))};
}

absl::Status PoissonLoss::Status() const {
  if (task_ != model::proto::Task::REGRESSION) {
    return absl::InvalidArgumentError(
        "Poisson loss is only compatible with a regression task");
  }
  return absl::OkStatus();
}

decision_tree::CreateSetLeafValueFunctor PoissonLoss::SetLeafFunctor(
    const std::vector<float> &predictions,
    const std::vector<GradientData> &gradients, int label_col_idx) const {
  return [](const dataset::VerticalDataset &,
            const std::vector<UnsignedExampleIdx> &, const std::vector<float> &,
            const model::proto::TrainingConfig &,
            const model::proto::TrainingConfigLinking &,
            decision_tree::NodeWithChildren *) {
    return absl::UnimplementedError("Not implemented");
  };
}

absl::StatusOr<decision_tree::SetLeafValueFromLabelStatsFunctor>
PoissonLoss::SetLeafFunctorFromLabelStatistics() const {
  return absl::UnimplementedError("Not implemented");
}

absl::Status PoissonLoss::UpdatePredictions(
    const std::vector<const decision_tree::DecisionTree *> &new_trees,
    const dataset::VerticalDataset &dataset, std::vector<float> *predictions,
    double *mean_abs_prediction) const {
  return absl::UnimplementedError("Not implemented");
}

absl::Status PoissonLoss::UpdateGradients(
    const std::vector<float> &labels, const std::vector<float> &predictions,
    const RankingGroupsIndices *ranking_index, GradientDataRef *gradients,
    utils::RandomEngine *random,
    utils::concurrency::ThreadPool *thread_pool) const {
  return absl::UnimplementedError("Not implemented");
}

std::vector<std::string> PoissonLoss::SecondaryMetricNames() const {
  return {};
}

absl::StatusOr<LossResults> PoissonLoss::Loss(
    const std::vector<float> &labels, const std::vector<float> &predictions,
    const std::vector<float> &weights,
    const RankingGroupsIndices *ranking_index,
    utils::concurrency::ThreadPool *thread_pool) const {
  return absl::UnimplementedError("Not implemented");
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
