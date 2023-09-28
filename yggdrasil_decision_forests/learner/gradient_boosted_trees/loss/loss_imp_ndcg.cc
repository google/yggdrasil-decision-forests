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
#include "yggdrasil_decision_forests/metric/ranking_ndcg.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

absl::Status NDCGLoss::Status() const {
  if (task_ != model::proto::Task::RANKING) {
    return absl::InvalidArgumentError(
        "NDCG loss is only compatible with a ranking task.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> NDCGLoss::InitialPredictions(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const std::vector<float>& weights) const {
  return std::vector<float>{0.f};
}

absl::StatusOr<std::vector<float>> NDCGLoss::InitialPredictions(
    const decision_tree::proto::LabelStatistics& label_statistics) const {
  return std::vector<float>{0.f};
}

absl::Status NDCGLoss::UpdateGradients(
    const std::vector<float>& labels, const std::vector<float>& predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  // TODO: Implement thread_pool.

  std::vector<float>& gradient_data = *(*gradients)[0].gradient;
  std::vector<float>& hessian_data = *(*gradients)[0].hessian;
  DCHECK_EQ(gradient_data.size(), hessian_data.size());

  metric::NDCGCalculator ndcg_calculator(kNDCG5Truncation);

  const float lambda_loss = gbt_config_.lambda_loss();
  const float lambda_loss_squared = lambda_loss * lambda_loss;

  // Reset gradient accumulators.
  std::fill(gradient_data.begin(), gradient_data.end(), 0.f);
  std::fill(hessian_data.begin(), hessian_data.end(), 0.f);

  // "pred_and_in_ground_idx[j].first" is the prediction for the example
  // "group[pred_and_in_ground_idx[j].second].example_idx".
  std::vector<std::pair<float, int>> pred_and_in_ground_idx;
  for (const auto& group : ranking_index->groups()) {
    // Extract predictions.
    const int group_size = group.items.size();
    pred_and_in_ground_idx.resize(group_size);
    for (int item_idx = 0; item_idx < group_size; item_idx++) {
      pred_and_in_ground_idx[item_idx] = {
          predictions[group.items[item_idx].example_idx], item_idx};
    }

    // NDCG normalization term.
    // Note: At this point, "pred_and_in_ground_idx" is sorted by relevance
    // i.e. ground truth.
    float utility_norm_factor = 1.;
    if (!gbt_config_.lambda_mart_ndcg().gradient_use_non_normalized_dcg()) {
      const int max_rank = std::min(kNDCG5Truncation, group_size);
      float max_ndcg = 0;
      for (int rank = 0; rank < max_rank; rank++) {
        max_ndcg += ndcg_calculator.Term(group.items[rank].relevance, rank);
      }
      utility_norm_factor = 1.f / max_ndcg;
    }

    // Sort by decreasing predicted value.
    // Note: We shuffle the predictions so that the expected gradient value is
    // aligned with the metric value with ties taken into account (which is
    // too expensive to do here).
    std::shuffle(pred_and_in_ground_idx.begin(), pred_and_in_ground_idx.end(),
                 *random);
    std::sort(pred_and_in_ground_idx.begin(), pred_and_in_ground_idx.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    const int num_pred_and_in_ground = pred_and_in_ground_idx.size();

    // Compute the "force" that each item apply on each other items.
    for (int item_1_idx = 0; item_1_idx < num_pred_and_in_ground;
         item_1_idx++) {
      const float pred_1 = pred_and_in_ground_idx[item_1_idx].first;
      const int in_ground_idx_1 = pred_and_in_ground_idx[item_1_idx].second;
      const float relevance_1 = group.items[in_ground_idx_1].relevance;
      const auto example_1_idx = group.items[in_ground_idx_1].example_idx;

      // Accumulator for the gradient and second order derivative of the
      // example
      // "group[pred_and_in_ground_idx[item_1_idx].second].example_idx".
      float& grad_1 = gradient_data[example_1_idx];
      float& second_order_1 = hessian_data[example_1_idx];

      for (int item_2_idx = item_1_idx + 1; item_2_idx < num_pred_and_in_ground;
           item_2_idx++) {
        const float pred_2 = pred_and_in_ground_idx[item_2_idx].first;
        const int in_ground_idx_2 = pred_and_in_ground_idx[item_2_idx].second;
        const float relevance_2 = group.items[in_ground_idx_2].relevance;
        const auto example_2_idx = group.items[in_ground_idx_2].example_idx;

        // Skip examples with the same relevance value.
        if (relevance_1 == relevance_2) {
          continue;
        }

        // "delta_utility" corresponds to "Z_{i,j}" in the paper.
        float delta_utility = 0;
        if (item_1_idx < kNDCG5Truncation) {
          delta_utility += ndcg_calculator.Term(relevance_2, item_1_idx) -
                           ndcg_calculator.Term(relevance_1, item_1_idx);
        }
        if (item_2_idx < kNDCG5Truncation) {
          delta_utility += ndcg_calculator.Term(relevance_1, item_2_idx) -
                           ndcg_calculator.Term(relevance_2, item_2_idx);
        }
        delta_utility = std::abs(delta_utility) * utility_norm_factor;

        // "sign" correspond to the sign in front of the lambda_{i,j} terms
        // in the equation defining lambda_i, in section 7 of "From RankNet
        // to LambdaRank to LambdaMART: An Overview".
        // The "sign" is also used to reverse the {i,j} or {j,i} in the
        // "lambda" term i.e. "s_i" and "s_j" in the sigmoid.

        // sign = in_ground_idx_1 < in_ground_idx_2 ? +1.f : -1.f;
        // signed_lambda_loss = sign * lambda_loss;

        const float signed_lambda_loss =
            lambda_loss -
            2.f * lambda_loss * (in_ground_idx_1 >= in_ground_idx_2);

        // "sigmoid" corresponds to "rho_{i,j}" in the paper.
        const float sigmoid =
            1.f / (1.f + std::exp(signed_lambda_loss * (pred_1 - pred_2)));

        // "unit_grad" corresponds to "lambda_{i,j}" in the paper.
        // Note: We want to minimize the loss function i.e. go in opposite
        // side of the gradient.
        const float unit_grad = signed_lambda_loss * sigmoid * delta_utility;
        const float unit_second_order =
            delta_utility * sigmoid * (1.f - sigmoid) * lambda_loss_squared;

        grad_1 += unit_grad;
        second_order_1 += unit_second_order;

        DCheckIsFinite(grad_1);
        DCheckIsFinite(second_order_1);

        gradient_data[example_2_idx] -= unit_grad;
        hessian_data[example_2_idx] += unit_second_order;
      }
    }
  }
  return absl::OkStatus();
}


std::vector<std::string> NDCGLoss::SecondaryMetricNames() const {
  return {"NDCG@5"};
}

absl::StatusOr<LossResults> NDCGLoss::Loss(
    const std::vector<float>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index,
    utils::concurrency::ThreadPool* thread_pool) const {
  if (ranking_index == nullptr) {
    return absl::InternalError("Missing ranking index");
  }

  const float ndcg =
      ranking_index->NDCG(predictions, weights, kNDCG5Truncation);
  return LossResults{/*.loss =*/-ndcg, /*.secondary_metrics =*/{ndcg}};
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
