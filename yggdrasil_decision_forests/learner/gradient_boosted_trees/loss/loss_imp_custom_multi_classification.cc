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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_custom_multi_classification.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

LossShape CustomMultiClassificationLoss::Shape() const {
  return LossShape{.gradient_dim = dimension_, .prediction_dim = dimension_};
};

absl::Status CustomMultiClassificationLoss::Status() const {
  if (task_ != model::proto::Task::CLASSIFICATION) {
    return absl::InvalidArgumentError(
        "This custom loss is only compatible with a classification task.");
  }
  if (dimension_ == 2) {
    return absl::InvalidArgumentError(
        "The dataset is a binary classification dataset. Please use a binary "
        "classification loss.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>>
CustomMultiClassificationLoss::InitialPredictions(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const absl::Span<const float> weights) const {
  ASSIGN_OR_RETURN(
      const auto* labels,
      dataset.ColumnWithCastWithStatus<
          dataset::VerticalDataset::CategoricalColumn>(label_col_idx));
  DCHECK_EQ(weights.size(), labels->nrows());

  auto labels_span = absl::MakeConstSpan(labels->values());
  auto weights_span = absl::MakeConstSpan(weights);
  std::vector<float> initial_predictions(dimension_);

  RETURN_IF_ERROR(custom_loss_functions_.initial_predictions(
      labels_span, weights_span, absl::MakeSpan(initial_predictions)));
  return initial_predictions;
}

absl::StatusOr<std::vector<float>>
CustomMultiClassificationLoss::InitialPredictions(
    const decision_tree::proto::LabelStatistics& label_statistics) const {
  return absl::UnimplementedError(
      "Loss with LabelStatistics (e.g. for distributed training) not supported "
      "for custom loss.");
}

absl::Status CustomMultiClassificationLoss::UpdateGradients(
    const absl::Span<const int32_t> labels,
    const absl::Span<const float> predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  auto labels_span = absl::MakeConstSpan(labels);
  auto predictions_span = absl::MakeConstSpan(predictions);
  std::vector<absl::Span<float>> gradient_data(dimension_);
  std::vector<absl::Span<float>> hessian_data(dimension_);
  for (int grad_idx = 0; grad_idx < dimension_; grad_idx++) {
    gradient_data[grad_idx] = absl::MakeSpan(*(*gradients)[grad_idx].gradient);
    hessian_data[grad_idx] = absl::MakeSpan(*(*gradients)[grad_idx].hessian);
  }

  RETURN_IF_ERROR(custom_loss_functions_.gradient_and_hessian(
      labels_span, predictions_span, absl::MakeConstSpan(gradient_data),
      absl::MakeConstSpan(hessian_data)));
  return absl::OkStatus();
}

std::vector<std::string> CustomMultiClassificationLoss::SecondaryMetricNames()
    const {
  return {};
}

absl::StatusOr<LossResults> CustomMultiClassificationLoss::Loss(
    const absl::Span<const int32_t> labels,
    const absl::Span<const float> predictions,
    const absl::Span<const float> weights,
    const RankingGroupsIndices* ranking_index,
    utils::concurrency::ThreadPool* thread_pool) const {
  DCHECK_EQ(weights.size(), labels.size());
  DCHECK_EQ(weights.size() * dimension_, predictions.size());

  auto labels_span = absl::MakeConstSpan(labels);
  auto predictions_span = absl::MakeConstSpan(predictions);
  auto weights_span = absl::MakeConstSpan(weights);
  ASSIGN_OR_RETURN(
      float loss_value,
      custom_loss_functions_.loss(labels_span, predictions_span, weights_span));

  return LossResults{/*.loss =*/loss_value,
                     /*.secondary_metrics =*/{}};
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
