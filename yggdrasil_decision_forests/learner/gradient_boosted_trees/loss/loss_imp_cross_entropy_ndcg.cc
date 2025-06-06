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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_cross_entropy_ndcg.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_ndcg.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_utils.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

absl::StatusOr<std::unique_ptr<AbstractLoss>>
CrossEntropyNDCGLoss::RegistrationCreate(const ConstructorArgs& args) {
  if (args.task != model::proto::Task::RANKING) {
    return absl::InvalidArgumentError(
        "Cross Entropy NDCG loss is only compatible with a ranking task.");
  }
  if (args.gbt_config.xe_ndcg().ndcg_truncation() < 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("The NDCG truncation must be set to a positive integer, "
                     "currently found: ",
                     args.gbt_config.xe_ndcg().ndcg_truncation()));
  }
  return absl::make_unique<CrossEntropyNDCGLoss>(args);
}

absl::StatusOr<std::vector<float>> CrossEntropyNDCGLoss::InitialPredictions(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const absl::Span<const float> weights) const {
  return std::vector<float>{0.f};
}

absl::StatusOr<std::vector<float>> CrossEntropyNDCGLoss::InitialPredictions(
    const decision_tree::proto::LabelStatistics& label_statistics) const {
  return std::vector<float>{0.f};
}

absl::Status CrossEntropyNDCGLoss::UpdateGradients(
    const absl::Span<const float> labels,
    const absl::Span<const float> predictions, const AbstractLossCache* cache,
    GradientDataRef* gradients, utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  STATUS_CHECK(cache);
  auto& ranking_index = static_cast<const Cache*>(cache)->ranking_index;

  // TODO: Implement thread_pool.

  std::vector<float>& gradient_data = *(*gradients)[0].gradient;
  std::vector<float>& hessian_data = *((*gradients)[0].hessian);
  DCHECK_EQ(gradient_data.size(), hessian_data.size());

  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  // Reset gradient accumulators.
  std::fill(gradient_data.begin(), gradient_data.end(), 0.f);
  std::fill(hessian_data.begin(), hessian_data.end(), 0.f);

  // A vector of predictions for items in a group.
  std::vector<float> preds;
  // An auxiliary buffer of parameters used to form the ground-truth
  // distribution and compute the loss.
  std::vector<float> params;

  for (const auto& group : ranking_index.groups()) {
    const size_t group_size = group.items.size();

    // Skip groups with too few items.
    if (group_size <= 1) {
      continue;
    }

    // Extract predictions.
    preds.resize(group_size);
    params.resize(group_size);

    switch (gbt_config_.xe_ndcg().gamma()) {
      case proto::LossConfiguration::XeNdcg::ONE:
        std::fill(params.begin(), params.end(), 1.f);
        break;
      case proto::LossConfiguration::XeNdcg::AUTO:
      case proto::LossConfiguration::XeNdcg::UNIFORM:
        for (int item_idx = 0; item_idx < group_size; item_idx++) {
          params[item_idx] = distribution(*random);
        }
        break;
    }
    for (int item_idx = 0; item_idx < group_size; item_idx++) {
      preds[item_idx] = predictions[group.items[item_idx].example_idx];
    }

    // Turn scores into a probability distribution with Softmax.
    const float max_pred = *std::max_element(preds.begin(), preds.end());
    float sum_exp = 0.0f;
    for (int idx = 0; idx < group_size; idx++) {
      sum_exp += std::exp(preds[idx] - max_pred);
    }
    float log_sum_exp = max_pred + std::log(sum_exp + 1e-20f);
    for (int idx = 0; idx < group_size; idx++) {
      float probability = std::exp(preds[idx] - log_sum_exp);
      preds[idx] = std::clamp(probability, 1e-5f, .99999f);
    }

    // Approximate Newton's step.
    // First-order terms.
    float inv_denominator = 0;
    for (int idx = 0; idx < group_size; idx++) {
      // Params is currently a \gamma but becomes the numerator of the
      // first-order approximation terms.
      params[idx] = std::exp2f(group.items[idx].relevance) - params[idx];
      inv_denominator += params[idx];
    }
    if (inv_denominator == 0.f) {
      continue;
    }
    inv_denominator = 1.f / inv_denominator;

    float sum_l1 = 0.f;
    for (int idx = 0; idx < group_size; idx++) {
      const auto example_idx = group.items[idx].example_idx;
      const auto term = -params[idx] * inv_denominator + preds[idx];
      gradient_data[example_idx] = -term;

      // Params will now store terms needed to compute second-order terms.
      params[idx] = term / (1.f - preds[idx]);
      sum_l1 += params[idx];
    }
    // Second-order terms.
    float sum_l2 = 0.f;
    for (int idx = 0; idx < group_size; idx++) {
      const auto example_idx = group.items[idx].example_idx;
      const auto term = preds[idx] * (sum_l1 - params[idx]);
      gradient_data[example_idx] -= term;

      // Params will now store terms needed to compute third-order terms.
      params[idx] = term / (1.f - preds[idx]);
      sum_l2 += params[idx];
    }

    // Third-order terms and the Hessian.
    for (int idx = 0; idx < group_size; idx++) {
      const auto example_idx = group.items[idx].example_idx;
      gradient_data[example_idx] -= preds[idx] * (sum_l2 - params[idx]);
      hessian_data[example_idx] = preds[idx] * (1.f - preds[idx]);
    }
  }
  return absl::OkStatus();
}

std::vector<std::string> CrossEntropyNDCGLoss::SecondaryMetricNames() const {
  return {};
}

absl::StatusOr<LossResults> CrossEntropyNDCGLoss::Loss(
    const absl::Span<const float> labels,
    const absl::Span<const float> predictions,
    const absl::Span<const float> weights, const AbstractLossCache* cache,
    utils::concurrency::ThreadPool* thread_pool) const {
  STATUS_CHECK(cache);
  auto& ranking_index = static_cast<const Cache*>(cache)->ranking_index;
  float loss_value =
      -ranking_index.NDCG(predictions, weights, ndcg_truncation_);
  return LossResults{/*.loss =*/loss_value, /*.secondary_metrics =*/{}};
}

absl::StatusOr<std::unique_ptr<AbstractLossCache>>
CrossEntropyNDCGLoss::CreateLossCache(
    const dataset::VerticalDataset& dataset) const {
  auto cache = absl::make_unique<CrossEntropyNDCGLoss::Cache>();
  RETURN_IF_ERROR(cache->ranking_index.Initialize(
      dataset, train_config_link_.label(), train_config_link_.ranking_group()));
  return cache;
}

absl::StatusOr<std::unique_ptr<AbstractLossCache>>
CrossEntropyNDCGLoss::CreateRankingLossCache(
    absl::Span<const float> labels, absl::Span<const uint64_t> groups) const {
  auto cache = absl::make_unique<CrossEntropyNDCGLoss::Cache>();
  RETURN_IF_ERROR(cache->ranking_index.Initialize(labels, groups));
  return cache;
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
