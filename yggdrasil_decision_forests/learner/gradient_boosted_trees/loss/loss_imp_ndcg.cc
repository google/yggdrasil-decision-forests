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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_ndcg.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_utils.h"
#include "yggdrasil_decision_forests/metric/ranking_ndcg.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

void UpdateGradientsIndicatorLabels(
    const absl::Span<const float> predictions,
    const RankingGroupsIndices::Group& group,
    const metric::NDCGCalculator& ndcg_calculator, const float lambda_loss,
    const absl::Span<const std::pair<float, int>> pred_and_in_group_idx,
    absl::Span<float> gradient_data, absl::Span<float> hessian_data) {
  const float lambda_loss_squared = lambda_loss * lambda_loss;
  const size_t group_size = group.items.size();
  const int ndcg_truncation = ndcg_calculator.truncation();
  // The group items are sorted by ground truth, so the first item must have
  // relevance 1, all other examples relevance 0.
  const auto& item_pos = group.items[0];
  DCHECK_EQ(item_pos.relevance, 1.f);
  const auto example_pos_idx = item_pos.example_idx;
  const auto pos_pred = predictions[example_pos_idx];
  int pos_item_idx = -1;
  for (int i = 0; i < group_size; i++) {
    // We need the prediction of the first item when ordered by ground
    // truth.
    if (pred_and_in_group_idx[i].second == 0) {
      pos_item_idx = i;
      break;
    }
  }
  DCHECK_NE(pos_item_idx, -1);

  float& grad_1 = gradient_data[example_pos_idx];
  float& second_order_1 = hessian_data[example_pos_idx];

  // "delta_utility" corresponds to "Z_{i,j}" in the paper.
  // Recall that item_1_idx < ndcg_truncation.
  //
  double pos_delta_utility = 0.;
  if (pos_item_idx < ndcg_truncation) {
    pos_delta_utility = ndcg_calculator.InvLogRank(pos_item_idx);
  }
  // Iteration over all other elements, all with relevance 0.
  for (int item_2_idx = 0; item_2_idx < group_size; item_2_idx++) {
    // Skip the positive item itself.
    if (item_2_idx == pos_item_idx) {
      continue;
    }
    const float pred_2 = pred_and_in_group_idx[item_2_idx].first;
    const int in_group_idx_2 = pred_and_in_group_idx[item_2_idx].second;
    const auto example_2_idx = group.items[in_group_idx_2].example_idx;
    DCHECK_EQ(group.items[in_group_idx_2].relevance, 0.f);

    // "delta_utility" corresponds to "Z_{i,j}" in the paper.
    // Recall that item_1_idx < ndcg_truncation.
    //
    double delta_utility = pos_delta_utility;

    if (item_2_idx < ndcg_truncation) {
      delta_utility =
          std::abs(delta_utility - ndcg_calculator.InvLogRank(item_2_idx));
    }

    // If ordered by ground truth, pos_item is before item_2.
    // "sigmoid" corresponds to "rho_{i,j}" in the paper.
    const float sigmoid =
        1.f / (1.f + std::exp(lambda_loss * (pos_pred - pred_2)));
    const float unit_grad = lambda_loss * sigmoid * delta_utility;
    const float unit_second_order =
        delta_utility * sigmoid * (1.f - sigmoid) * lambda_loss_squared;

    // Update the gradients of item_1.
    grad_1 += unit_grad;
    second_order_1 += unit_second_order;

    DCheckIsFinite(grad_1);
    DCheckIsFinite(second_order_1);

    // Update the gradients of item_2.
    gradient_data[example_2_idx] -= unit_grad;
    hessian_data[example_2_idx] += unit_second_order;
    DCheckIsFinite(gradient_data[example_2_idx]);
    DCheckIsFinite(hessian_data[example_2_idx]);
  }
}

void UpdateGradientsArbitraryLabels(
    const absl::Span<const float> predictions,
    const RankingGroupsIndices::Group& group,
    const metric::NDCGCalculator& ndcg_calculator, const float lambda_loss,
    const absl::Span<const std::pair<float, int>> pred_and_in_group_idx,
    const double utility_normalization_factor, absl::Span<float> gradient_data,
    absl::Span<float> hessian_data) {
  const float lambda_loss_squared = lambda_loss * lambda_loss;
  const int group_size = group.items.size();
  const int ndcg_truncation = ndcg_calculator.truncation();
  // Two items whose ground truth is above the truncation threshold will not
  // impact the gradient if swapped.
  const size_t maximum_relevant_item_idx =
      std::min(group_size, ndcg_truncation);

  // Compute the gradients and hessians: For every item, compute the force
  // (i.e. lambdas) to apply on the other items.
  for (int item_1_idx = 0; item_1_idx < maximum_relevant_item_idx;
       item_1_idx++) {
    const float pred_1 = pred_and_in_group_idx[item_1_idx].first;
    const int in_group_idx_1 = pred_and_in_group_idx[item_1_idx].second;
    const float relevance_1 = group.items[in_group_idx_1].relevance;
    const auto example_1_idx = group.items[in_group_idx_1].example_idx;

    // Accumulator for the gradient and second order derivative of the
    // example `example_1_idx`.
    float& grad_1 = gradient_data[example_1_idx];
    float& second_order_1 = hessian_data[example_1_idx];

    // Iterate over all items that should be ranked lower than item_1. In
    // the notation from the paper, item_1 ▷ item_2.
    for (int item_2_idx = item_1_idx + 1; item_2_idx < group_size;
         item_2_idx++) {
      const float pred_2 = pred_and_in_group_idx[item_2_idx].first;
      const int in_group_idx_2 = pred_and_in_group_idx[item_2_idx].second;
      const float relevance_2 = group.items[in_group_idx_2].relevance;
      const auto example_2_idx = group.items[in_group_idx_2].example_idx;

      // Skip examples with the same relevance value.
      if (relevance_1 == relevance_2) {
        continue;
      }

      // "delta_utility" corresponds to "Z_{i,j}" in the paper.
      // Recall that item_1_idx < ndcg_truncation.
      //
      double delta_utility = ndcg_calculator.Term(relevance_2, item_1_idx) -
                             ndcg_calculator.Term(relevance_1, item_1_idx);

      if (item_2_idx < ndcg_truncation) {
        delta_utility += ndcg_calculator.Term(relevance_1, item_2_idx) -
                         ndcg_calculator.Term(relevance_2, item_2_idx);
      }
      delta_utility = std::abs(delta_utility) * utility_normalization_factor;

      // If the current prediction has item_1 and item_2 ordered correctly,
      // i.e. in_group_idx_1 >= in_group_idx_2, apply them positively,
      // otherwise, multiply with -1.
      const float sign = 2.f * (in_group_idx_1 >= in_group_idx_2) - 1.f;
      // "sigmoid" corresponds to "rho_{i,j}" in the paper.
      // We use `sign` to negate the subtraction `pred_1 - pred_2`.
      const float sigmoid =
          1.f / (1.f + std::exp(-lambda_loss * sign * (pred_1 - pred_2)));
      const float unit_grad = sign * (-lambda_loss) * sigmoid * delta_utility;
      const float unit_second_order =
          delta_utility * sigmoid * (1.f - sigmoid) * lambda_loss_squared;

      // Update the gradients of item_1.
      grad_1 += unit_grad;
      second_order_1 += unit_second_order;

      DCheckIsFinite(grad_1);
      DCheckIsFinite(second_order_1);

      // Update the gradients of item_2.
      gradient_data[example_2_idx] -= unit_grad;
      hessian_data[example_2_idx] += unit_second_order;
      DCheckIsFinite(gradient_data[example_2_idx]);
      DCheckIsFinite(hessian_data[example_2_idx]);
    }
  }
}

absl::Status UpdateGradientsSingleThread(
    const absl::Span<const float> labels,
    const absl::Span<const float> predictions,
    const absl::Span<const RankingGroupsIndices::Group>& groups,
    const int ndcg_truncation, const float lambda_loss,
    const bool gradient_use_non_normalized_dcg,
    bool enable_indicator_labels_optimization, const int64_t seed,
    absl::Span<float> gradient_data, absl::Span<float> hessian_data) {
  utils::RandomEngine local_random(seed);
  DCHECK_EQ(gradient_data.size(), hessian_data.size());

  metric::NDCGCalculator ndcg_calculator(ndcg_truncation);

  // "pred_and_in_group_idx[j].first" is the prediction for the example
  // "group[pred_and_in_group_idx[j].second].example_idx".
  std::vector<std::pair<float, int>> pred_and_in_group_idx;
  for (const auto& group : groups) {
    // Number of items with relevance 0.
    int relevance_zero_count = 0;
    // Number of items with relevance 1.
    int relevance_one_count = 0;

    // For each group item, extract its prediction and its original example_idx.
    // Groups are sorted by relevance (ground truth).
    // Also check if the labels of this group are indicator labels.
    const int group_size = group.items.size();
    pred_and_in_group_idx.resize(group_size);
    for (int item_idx = 0; item_idx < group_size; item_idx++) {
      const auto& item = group.items[item_idx];
      pred_and_in_group_idx[item_idx] = {predictions[item.example_idx],
                                         item_idx};

      const int is_one = (item.relevance == 1.0f);
      const int is_zero = (item.relevance == 0.0f);

      relevance_one_count += is_one;
      relevance_zero_count += is_zero;
    }

    // Use the special path for indicator labels if and only if
    // - The optimization is enabled, and
    // - There is exactly on item of relevance 1, and
    // - All other items have relevance 0.

    const bool indicator_labels =
        enable_indicator_labels_optimization & (relevance_one_count == 1) &
        (relevance_zero_count + relevance_one_count == group_size);

    // NDCG normalization term.
    float utility_normalization_factor = 1.;
    if (!gradient_use_non_normalized_dcg && !indicator_labels) {
      const int max_rank = std::min(ndcg_truncation, group_size);
      float max_ndcg = 0;
      for (int rank = 0; rank < max_rank; rank++) {
        max_ndcg += ndcg_calculator.Term(group.items[rank].relevance, rank);
      }
      utility_normalization_factor = 1.f / max_ndcg;
    }

    // Sort pred_and_in_group_idx by decreasing predicted value.
    // Note: We shuffle the predictions so that the expected gradient value is
    // aligned with the metric value with ties taken into account (which is
    // too expensive to do here).
    std::sort(pred_and_in_group_idx.begin(), pred_and_in_group_idx.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    auto it = pred_and_in_group_idx.begin();
    while (it != pred_and_in_group_idx.end()) {
      auto next_it = std::next(it);
      while (next_it != pred_and_in_group_idx.end() &&
             it->first == next_it->first) {
        next_it++;
      }
      std::shuffle(it, next_it, local_random);
      it = next_it;
    }
    // For indicator labels, i.e. labels with a single 1 and 0 otherwise, we can
    // skip the double loop.
    if (indicator_labels) {
      UpdateGradientsIndicatorLabels(predictions, group, ndcg_calculator,
                                     lambda_loss, pred_and_in_group_idx,
                                     gradient_data, hessian_data);
    } else {
      UpdateGradientsArbitraryLabels(predictions, group, ndcg_calculator,
                                     lambda_loss, pred_and_in_group_idx,
                                     utility_normalization_factor,
                                     gradient_data, hessian_data);
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status NDCGLoss::Status() const {
  if (task_ != model::proto::Task::RANKING) {
    return absl::InvalidArgumentError(
        "NDCG loss is only compatible with a ranking task.");
  }
  if (ndcg_truncation_ < 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("The NDCG truncation must be set to a positive integer, "
                     "currently found: ",
                     ndcg_truncation_));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> NDCGLoss::InitialPredictions(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const absl::Span<const float> weights) const {
  return std::vector<float>{0.f};
}

absl::StatusOr<std::vector<float>> NDCGLoss::InitialPredictions(
    const decision_tree::proto::LabelStatistics& label_statistics) const {
  return std::vector<float>{0.f};
}

absl::Status NDCGLoss::UpdateGradients(
    const absl::Span<const float> labels,
    const absl::Span<const float> predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  STATUS_CHECK_EQ(gradients->size(), 1);
  std::vector<float>& gradient_data = *(*gradients)[0].gradient;
  std::vector<float>& hessian_data = *(*gradients)[0].hessian;
  STATUS_CHECK_EQ(gradient_data.size(), hessian_data.size());
  // Reset gradient accumulators.
  std::fill(gradient_data.begin(), gradient_data.end(), 0.f);
  std::fill(hessian_data.begin(), hessian_data.end(), 0.f);

  if (thread_pool == nullptr) {
    RETURN_IF_ERROR(UpdateGradientsSingleThread(
        labels, predictions, absl::MakeConstSpan(ranking_index->groups()),
        ndcg_truncation_, gbt_config_.lambda_loss(),
        gbt_config_.lambda_mart_ndcg().gradient_use_non_normalized_dcg(),
        gbt_config_.internal().enable_ndcg_indicator_labels_optimization(),
        (*random)(), absl::Span<float>(gradient_data),
        absl::Span<float>(hessian_data)));
    return absl::OkStatus();
  } else {
    absl::Status global_status;
    utils::concurrency::Mutex global_mutex;
    std::vector<int64_t> random_seeds(thread_pool->num_threads());
    for (size_t i = 0; i < random_seeds.size(); i++) {
      random_seeds[i] = (*random)();
    }
    utils::concurrency::ConcurrentForLoop(
        thread_pool->num_threads(), thread_pool, ranking_index->groups().size(),
        [&labels, &predictions, ranking_index, &gradient_data, &hessian_data,
         &random_seeds, lambda_loss = gbt_config_.lambda_loss(),
         gradient_use_non_normalized_dcg =
             gbt_config_.lambda_mart_ndcg().gradient_use_non_normalized_dcg(),
         enable_indicator_labels_optimization =
             gbt_config_.internal().enable_ndcg_indicator_labels_optimization(),
         ndcg_truncation = this->ndcg_truncation_, &global_status,
         &global_mutex](const size_t block_idx, const size_t begin_idx,
                        const size_t end_idx) -> void {
          {
            utils::concurrency::MutexLock lock(&global_mutex);
            if (!global_status.ok()) {
              return;
            }
          }
          absl::Status thread_status = UpdateGradientsSingleThread(
              absl::MakeConstSpan(labels), absl::MakeConstSpan(predictions),
              absl::MakeConstSpan(ranking_index->groups())
                  .subspan(begin_idx, end_idx - begin_idx),
              ndcg_truncation, lambda_loss, gradient_use_non_normalized_dcg,
              enable_indicator_labels_optimization, random_seeds[block_idx],
              absl::MakeSpan(gradient_data), absl::MakeSpan(hessian_data));
          if (!thread_status.ok()) {
            utils::concurrency::MutexLock lock(&global_mutex);
            global_status.Update(thread_status);
            return;
          }
        });
    return global_status;
  }
}

std::vector<std::string> NDCGLoss::SecondaryMetricNames() const {
  return {absl::StrCat("NDCG@", ndcg_truncation_)};
}

absl::StatusOr<LossResults> NDCGLoss::Loss(
    const absl::Span<const float> labels,
    const absl::Span<const float> predictions,
    const absl::Span<const float> weights,
    const RankingGroupsIndices* ranking_index,
    utils::concurrency::ThreadPool* thread_pool) const {
  if (ranking_index == nullptr) {
    return absl::InternalError("Missing ranking index");
  }

  const float ndcg =
      ranking_index->NDCG(predictions, weights, ndcg_truncation_);
  return LossResults{/*.loss =*/-ndcg, /*.secondary_metrics =*/{ndcg}};
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
