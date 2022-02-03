/*
 * Copyright 2021 Google LLC.
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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"

#include <stdint.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_utils.h"
#include "yggdrasil_decision_forests/metric/ranking_ndcg.h"
#include "yggdrasil_decision_forests/metric/ranking_utils.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

absl::Status AbstractLoss::UpdateGradients(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const std::vector<float>& predictions,
    const RankingGroupsIndices* ranking_index,
    std::vector<GradientData>* gradients, utils::RandomEngine* random) const {
  GradientDataRef compact_gradient(gradients->size());
  for (int i = 0; i < gradients->size(); i++) {
    compact_gradient[i] = {&(*gradients)[i].gradient, (*gradients)[i].hessian};
  }

  const auto* categorical_labels =
      dataset.ColumnWithCastOrNull<dataset::VerticalDataset::CategoricalColumn>(
          label_col_idx);
  if (categorical_labels) {
    return UpdateGradients(categorical_labels->values(), predictions,
                           ranking_index, &compact_gradient, random, nullptr);
  }

  const auto* numerical_labels =
      dataset.ColumnWithCastOrNull<dataset::VerticalDataset::NumericalColumn>(
          label_col_idx);
  if (numerical_labels) {
    return UpdateGradients(numerical_labels->values(), predictions,
                           ranking_index, &compact_gradient, random, nullptr);
  }

  return absl::InternalError(
      absl::Substitute("Non supported label type for column \"$0\" ($1)",
                       dataset.column(label_col_idx)->name(), label_col_idx));
}

absl::Status AbstractLoss::Loss(const dataset::VerticalDataset& dataset,
                                int label_col_idx,
                                const std::vector<float>& predictions,
                                const std::vector<float>& weights,
                                const RankingGroupsIndices* ranking_index,
                                float* loss_value,
                                std::vector<float>* secondary_metric) const {
  const auto* categorical_labels =
      dataset.ColumnWithCastOrNull<dataset::VerticalDataset::CategoricalColumn>(
          label_col_idx);
  if (categorical_labels) {
    return Loss(categorical_labels->values(), predictions, weights,
                ranking_index, loss_value, secondary_metric, nullptr);
  }

  const auto* numerical_labels =
      dataset.ColumnWithCastOrNull<dataset::VerticalDataset::NumericalColumn>(
          label_col_idx);
  if (numerical_labels) {
    return Loss(numerical_labels->values(), predictions, weights, ranking_index,
                loss_value, secondary_metric, nullptr);
  }

  return absl::InternalError("Unknown label type");
}

void RankingGroupsIndices::Initialize(const dataset::VerticalDataset& dataset,
                                      int label_col_idx, int group_col_idx) {
  // Access to raw label and group values.
  const auto& label_values =
      dataset
          .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
              label_col_idx)
          ->values();

  const auto* group_categorical_values =
      dataset.ColumnWithCastOrNull<dataset::VerticalDataset::CategoricalColumn>(
          group_col_idx);
  const auto* group_hash_values =
      dataset.ColumnWithCastOrNull<dataset::VerticalDataset::HashColumn>(
          group_col_idx);

  // Fill index.
  absl::flat_hash_map<uint64_t, std::vector<Item>> tmp_groups;
  for (row_t example_idx = 0; example_idx < dataset.nrow(); example_idx++) {
    // Get the value of the group.
    uint64_t group_value;
    if (group_categorical_values) {
      group_value = group_categorical_values->values()[example_idx];
    } else if (group_hash_values) {
      group_value = group_hash_values->values()[example_idx];
    } else {
      LOG(FATAL) << "Invalid group type";
    }

    tmp_groups[group_value].push_back(
        {/*.relevance =*/label_values[example_idx],
         /*.example_idx =*/example_idx});
  }
  num_items_ = dataset.nrow();

  // Sort the group items by decreasing ground truth relevance.
  groups_.reserve(tmp_groups.size());
  for (auto& group : tmp_groups) {
    std::sort(group.second.begin(), group.second.end(),
              [](const Item& a, const Item& b) {
                if (a.relevance == b.relevance) {
                  return a.example_idx > b.example_idx;
                }
                return a.relevance > b.relevance;
              });

    if (group.second.size() > kMaximumItemsInRankingGroup) {
      LOG(FATAL) << "The number of items in the group \"" << group.first
                 << "\" is " << group.second.size()
                 << " and is greater than kMaximumItemsInRankingGroup="
                 << kMaximumItemsInRankingGroup
                 << ". This is likely a mistake in the generation of the "
                    "configuration of the group column.";
    }

    groups_.push_back(
        {/*.group_idx =*/group.first, /*.items =*/std::move(group.second)});
  }

  // Sort the group by example index to improve the data locality.
  std::sort(groups_.begin(), groups_.end(), [](const Group& a, const Group& b) {
    if (a.items.front().example_idx == b.items.front().example_idx) {
      return a.group_idx < b.group_idx;
    }
    return a.items.front().example_idx < b.items.front().example_idx;
  });
  LOG(INFO) << "Found " << groups_.size() << " groups in " << dataset.nrow()
            << " examples.";
}

double RankingGroupsIndices::NDCG(const std::vector<float>& predictions,
                                  const std::vector<float>& weights,
                                  const int truncation) const {
  DCHECK_EQ(predictions.size(), num_items_);
  DCHECK_EQ(weights.size(), num_items_);

  metric::NDCGCalculator ndcg_calculator(truncation);
  std::vector<metric::RankingLabelAndPrediction> pred_and_label_relevance;

  double sum_weighted_ndcg = 0;
  double sum_weights = 0;

  if (weights.empty()) {
    for (auto& group : groups_) {
      DCHECK(!group.items.empty());
      ExtractPredAndLabelRelevance(group.items, predictions,
                                   &pred_and_label_relevance);

      sum_weighted_ndcg += ndcg_calculator.NDCG(pred_and_label_relevance);
    }
    sum_weights += groups_.size();
  } else {
    for (auto& group : groups_) {
      DCHECK(!group.items.empty());
      const float weight = weights[group.items.front().example_idx];

      ExtractPredAndLabelRelevance(group.items, predictions,
                                   &pred_and_label_relevance);

      sum_weighted_ndcg +=
          weight * ndcg_calculator.NDCG(pred_and_label_relevance);
      sum_weights += weight;
    }
  }

  return sum_weighted_ndcg / sum_weights;
}

void RankingGroupsIndices::ExtractPredAndLabelRelevance(
    const std::vector<Item>& group, const std::vector<float>& predictions,
    std::vector<metric::RankingLabelAndPrediction>* pred_and_label_relevance) {
  pred_and_label_relevance->resize(group.size());
  for (int item_idx = 0; item_idx < group.size(); item_idx++) {
    (*pred_and_label_relevance)[item_idx] = {
        /*.prediction =*/predictions[group[item_idx].example_idx],
        /*.label =*/group[item_idx].relevance};
  }
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
