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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_mean_average_error.h"

#include <stddef.h>

#include <algorithm>
#include <string>
#include <vector>

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
#include "yggdrasil_decision_forests/utils/concurrency.h"  // IWYU pragma: keep
#include "yggdrasil_decision_forests/utils/logging.h"  // IWYU pragma: keep
#include "yggdrasil_decision_forests/utils/math.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {
void UpdateGradientsSingleThread(const absl::Span<const float> labels,
                                 const absl::Span<const float> predictions,
                                 absl::Span<float> gradient_data,
                                 absl::Span<float> hessian_data) {
  DCHECK_EQ(labels.size(), predictions.size());
  DCHECK_EQ(labels.size(), gradient_data.size());
  DCHECK_EQ(labels.size(), hessian_data.size());

  // We use "table" to avoid a branch in the following loop.
  // This optimization was found to improve the code speed. This should be
  // revisited as new compilers are likely to do this optimization
  // automatically one day.
  static float table[] = {-1.f, 1.f};

  for (size_t example_idx = 0; example_idx < labels.size(); ++example_idx) {
    const float label = labels[example_idx];
    const float prediction = predictions[example_idx];
    gradient_data[example_idx] = table[label >= prediction];
    hessian_data[example_idx] = 1.f;
  }
}

}  // namespace

absl::Status MeanAverageErrorLoss::Status() const {
  if (task_ != model::proto::Task::REGRESSION) {
    return absl::InvalidArgumentError(
        "Mean average error loss is only compatible with regression");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> MeanAverageErrorLoss::InitialPredictions(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const std::vector<float>& weights) const {
  // The initial value is the weighted median value.

  ASSIGN_OR_RETURN(
      const auto* label_col,
      dataset
          .ColumnWithCastWithStatus<dataset::VerticalDataset::NumericalColumn>(
              label_col_idx));
  const std::vector<float>& labels = label_col->values();
  STATUS_CHECK_GT(labels.size(), 0);

  float initial_prediction;
  if (weights.empty()) {
    initial_prediction = utils::Median(labels);
  } else {
    struct Item {
      float label;
      float weight;
    };
    std::vector<Item> items;
    items.reserve(labels.size());
    const UnsignedExampleIdx num_examples = labels.size();
    double sum_weights = 0;
    for (UnsignedExampleIdx example_idx = 0; example_idx < num_examples;
         example_idx++) {
      sum_weights += weights[example_idx];
      items.push_back({
          .label = labels[example_idx],
          .weight = weights[example_idx],
      });
    }
    std::sort(items.begin(), items.end(),
              [](const Item& a, const Item& b) { return a.label < b.label; });

    const auto mid_weight = sum_weights / 2;
    double cur_sum_weights = 0;
    for (const auto& item : items) {
      cur_sum_weights += item.weight;
      if (cur_sum_weights > mid_weight) {
        initial_prediction = item.label;
      }
    }
  }

  return std::vector<float>(1, initial_prediction);
}

absl::StatusOr<std::vector<float>> MeanAverageErrorLoss::InitialPredictions(
    const decision_tree::proto::LabelStatistics& label_statistics) const {
  return absl::InvalidArgumentError(
      "Mean Average Error (MAE) is not available for distributed training.");
}

absl::Status MeanAverageErrorLoss::UpdateGradients(
    const std::vector<float>& labels, const std::vector<float>& predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  STATUS_CHECK_EQ(gradients->size(), 1);
  std::vector<float>& gradient_data = *(*gradients)[0].gradient;
  std::vector<float>& hessian_data = *(*gradients)[0].hessian;
  STATUS_CHECK_EQ(gradient_data.size(), hessian_data.size());

  if (thread_pool == nullptr) {
    UpdateGradientsSingleThread(labels, predictions,
                                absl::Span<float>(gradient_data),
                                absl::Span<float>(hessian_data));
  } else {
    utils::concurrency::ConcurrentForLoop(
        thread_pool->num_threads(), thread_pool, labels.size(),
        [&labels, &predictions, &gradient_data, &hessian_data](
            const size_t block_idx, const size_t begin_idx,
            const size_t end_idx) -> void {
          UpdateGradientsSingleThread(
              absl::Span<const float>(labels).subspan(begin_idx,
                                                      end_idx - begin_idx),
              absl::Span<const float>(predictions)
                  .subspan(begin_idx, end_idx - begin_idx),
              absl::Span<float>(gradient_data)
                  .subspan(begin_idx, end_idx - begin_idx),
              absl::Span<float>(hessian_data)
                  .subspan(begin_idx, end_idx - begin_idx));
        });
  }

  return absl::OkStatus();
}

std::vector<std::string> MeanAverageErrorLoss::SecondaryMetricNames() const {
  return {"mae", "rmse"};
}

absl::StatusOr<LossResults> MeanAverageErrorLoss::Loss(
    const std::vector<float>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index,
    utils::concurrency::ThreadPool* thread_pool) const {
  ASSIGN_OR_RETURN(float mae,
                   metric::MAE(labels, predictions, weights, thread_pool));
  ASSIGN_OR_RETURN(float rmse,
                   metric::RMSE(labels, predictions, weights, thread_pool));
  return LossResults{.loss = mae, .secondary_metrics = {mae, rmse}};
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
