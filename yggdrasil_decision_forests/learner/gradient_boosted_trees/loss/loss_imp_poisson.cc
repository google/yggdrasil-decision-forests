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

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_utils.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

absl::StatusOr<std::vector<float>> PoissonLoss::InitialPredictions(
    const dataset::VerticalDataset &dataset, int label_col_idx,
    const std::vector<float> &weights) const {
  // The initial value is the logarithm of the weighted mean of the labels.
  double weighted_sum_labels = 0;
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
    weighted_sum_labels =
        utils::accumulate(labels->values().begin(), labels->values().end(), 0.);
  } else {
    for (decision_tree::row_t example_idx = 0; example_idx < dataset.nrow();
         example_idx++) {
      sum_weights += weights[example_idx];
      weighted_sum_labels +=
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
  // The initial prediction is the expected alpha coefficient of the poisson
  // distribution i.e. the log of the label mean.
  return std::vector<float>{static_cast<float>(std::log(weighted_sum_labels) -
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

absl::Status PoissonLoss::UpdateGradients(
    const std::vector<float> &labels, const std::vector<float> &predictions,
    const RankingGroupsIndices *ranking_index, GradientDataRef *gradients,
    utils::RandomEngine *random,
    utils::concurrency::ThreadPool *thread_pool) const {
  if (gradients->size() != 1) {
    return absl::InternalError("Wrong gradient shape");
  }
  const size_t num_examples = labels.size();
  std::vector<float> &gradient_data = *(*gradients)[0].gradient;
  std::vector<float> &hessian_data = *(*gradients)[0].hessian;
  DCHECK_EQ(gradient_data.size(), hessian_data.size());

  if (thread_pool == nullptr) {
    UpdateGradientsImp(labels, predictions, 0, num_examples, &gradient_data,
                       &hessian_data);
  } else {
    utils::concurrency::ConcurrentForLoop(
        thread_pool->num_threads(), thread_pool, num_examples,
        [&labels, &predictions, &gradient_data, &hessian_data](
            size_t block_idx, size_t begin_idx, size_t end_idx) -> void {
          UpdateGradientsImp(labels, predictions, begin_idx, end_idx,
                             &gradient_data, &hessian_data);
        });
  }
  return absl::OkStatus();
}

void PoissonLoss::UpdateGradientsImp(const std::vector<float> &labels,
                                     const std::vector<float> &predictions,
                                     size_t begin_example_idx,
                                     size_t end_example_idx,
                                     std::vector<float> *gradient_data,
                                     std::vector<float> *hessian_data) {
  // loss = exp(prediction) - label * prediction
  // -gradient = label - exp(prediction)
  // hessian = exp(prediction)
  for (size_t example_idx = begin_example_idx; example_idx < end_example_idx;
       example_idx++) {
    const float label = labels[example_idx];
    const float prediction = predictions[example_idx];
    const float exp_pred = std::exp(prediction);
    DCheckIsFinite(prediction);
    DCheckIsFinite(exp_pred);
    (*gradient_data)[example_idx] = label - exp_pred;
    (*hessian_data)[example_idx] = exp_pred;
  }
}


std::vector<std::string> PoissonLoss::SecondaryMetricNames() const {
  return {"RMSE"};
}

absl::StatusOr<LossResults> PoissonLoss::Loss(
    const std::vector<float> &labels, const std::vector<float> &predictions,
    const std::vector<float> &weights,
    const RankingGroupsIndices *ranking_index,
    utils::concurrency::ThreadPool *thread_pool) const {
  double sum_loss = 0;
  double total_example_weight = 0;
  double sum_square_error = 0;
  if (thread_pool == nullptr) {
    if (weights.empty()) {
      LossImp<false>(labels, predictions, weights, 0, labels.size(), &sum_loss,
                     &sum_square_error, &total_example_weight);
    } else {
      LossImp<true>(labels, predictions, weights, 0, labels.size(), &sum_loss,
                    &sum_square_error, &total_example_weight);
    }
  } else {
    const auto num_threads = thread_pool->num_threads();

    struct PerThread {
      double sum_loss = 0;
      double sum_square_error = 0;
      double total_example_weight = 0;
    };
    std::vector<PerThread> per_threads(num_threads);

    utils::concurrency::ConcurrentForLoop(
        num_threads, thread_pool, labels.size(),
        [&labels, &predictions, &per_threads, &weights](
            size_t block_idx, size_t begin_idx, size_t end_idx) -> void {
          auto &block = per_threads[block_idx];

          if (weights.empty()) {
            LossImp<false>(labels, predictions, weights, begin_idx, end_idx,
                           &block.sum_loss, &block.sum_square_error,
                           &block.total_example_weight);
          } else {
            LossImp<true>(labels, predictions, weights, begin_idx, end_idx,
                          &block.sum_loss, &block.sum_square_error,
                          &block.total_example_weight);
          }
        });

    for (const auto &block : per_threads) {
      sum_loss += block.sum_loss;
      sum_square_error += block.sum_square_error;
      total_example_weight += block.total_example_weight;
    }
  }
  auto float_poisson_loss = static_cast<float>(sum_loss / total_example_weight);
  auto float_rmse = static_cast<float>(sum_square_error / total_example_weight);
  return LossResults{float_poisson_loss, {float_rmse}};
}

template <bool use_weights>
void PoissonLoss::LossImp(const std::vector<float> &labels,
                          const std::vector<float> &predictions,
                          const std::vector<float> &weights,
                          size_t begin_example_idx, size_t end_example_idx,
                          double *__restrict sum_loss,
                          double *__restrict sum_square_error,
                          double *__restrict total_example_weight) {
  if constexpr (!use_weights) {
    *total_example_weight = end_example_idx - begin_example_idx;
  }
  for (size_t example_idx = begin_example_idx; example_idx < end_example_idx;
       example_idx++) {
    const float label = labels[example_idx];
    const float prediction = predictions[example_idx];
    const float exp_pred = std::exp(prediction);
    // The loss is the log-likelihood of the poisson distribution without the
    // term "-\sum log(factorial label)" (since this term is independent of the
    // predictions).
    //
    // loss = exp(prediction) - label * prediction
    //
    // TODO: Figure what a 2x factor was added.
    if constexpr (use_weights) {
      const float weight = weights[example_idx];
      *total_example_weight += weight;
      *sum_loss += 2.f * weight * (exp_pred - label * prediction);
      *sum_square_error += weight * (label - exp_pred) * (label - exp_pred);
    } else {
      *sum_loss += 2.f * (exp_pred - label * prediction);
      *sum_square_error += (label - exp_pred) * (label - exp_pred);
    }
    DCheckIsFinite(*sum_loss);
    DCheckIsFinite(*sum_square_error);
  }
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
