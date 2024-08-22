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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_custom_binary_classification.h"

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

absl::Status CustomBinaryClassificationLoss::Status() const {
  if (task_ != model::proto::Task::CLASSIFICATION) {
    return absl::InvalidArgumentError(
        "This custom loss is only compatible with a classification task.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>>
CustomBinaryClassificationLoss::InitialPredictions(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const absl::Span<const float> weights) const {
  ASSIGN_OR_RETURN(
      const auto* labels,
      dataset.ColumnWithCastWithStatus<
          dataset::VerticalDataset::CategoricalColumn>(label_col_idx));
  DCHECK_EQ(weights.size(), labels->nrows());
  DCHECK_EQ(label_column_.categorical().number_of_unique_values(), 3);

  auto labels_span = absl::MakeConstSpan(labels->values());
  auto weights_span = absl::MakeConstSpan(weights);

  ASSIGN_OR_RETURN(
      float initial_prediction,
      custom_loss_functions_.initial_predictions(labels_span, weights_span));
  return std::vector<float>{initial_prediction};
}

absl::StatusOr<std::vector<float>>
CustomBinaryClassificationLoss::InitialPredictions(
    const decision_tree::proto::LabelStatistics& label_statistics) const {
  return absl::UnimplementedError(
      "Loss with LabelStatistics (e.g. for distributed training) not supported "
      "for custom loss.");
}

absl::Status CustomBinaryClassificationLoss::UpdateGradients(
    const absl::Span<const int32_t> labels,
    const absl::Span<const float> predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  auto labels_span = absl::MakeConstSpan(labels);
  auto predictions_span = absl::MakeConstSpan(predictions);
  auto gradient_data = absl::MakeSpan(*(gradients->front().gradient));
  auto hessian_data = absl::MakeSpan(*(gradients->front().hessian));
  DCHECK_EQ(gradient_data.size(), hessian_data.size());

  RETURN_IF_ERROR(custom_loss_functions_.gradient_and_hessian(
      labels_span, predictions_span, gradient_data, hessian_data));
  return absl::OkStatus();
}

std::vector<std::string> CustomBinaryClassificationLoss::SecondaryMetricNames()
    const {
  return {};
}

absl::StatusOr<LossResults> CustomBinaryClassificationLoss::Loss(
    const absl::Span<const int32_t> labels,
    const absl::Span<const float> predictions,
    const absl::Span<const float> weights,
    const RankingGroupsIndices* ranking_index,
    utils::concurrency::ThreadPool* thread_pool) const {
  DCHECK_EQ(weights.size(), labels.size());
  DCHECK_EQ(weights.size(), predictions.size());

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
