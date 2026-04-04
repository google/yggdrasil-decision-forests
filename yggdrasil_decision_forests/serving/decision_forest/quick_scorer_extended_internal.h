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

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_QUICK_SCORER_EXTENDED_HWY_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_QUICK_SCORER_EXTENDED_HWY_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/numeric/bits.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/serving/decision_forest/quick_scorer_extended.h"
#include "yggdrasil_decision_forests/serving/example_set.h"

namespace yggdrasil_decision_forests::serving::decision_forest {
namespace internal {

ABSL_ATTRIBUTE_ALWAYS_INLINE inline float ActivationBinomialLogLikelihood(
    const float value) {
  return std::clamp(1.f / (1.f + std::exp(-value)), 0.f, 1.f);
}

ABSL_ATTRIBUTE_ALWAYS_INLINE inline float ActivationPoisson(const float value) {
  return std::exp(
      std::clamp(value,
                 -model::gradient_boosted_trees::GradientBoostedTreesModel::
                     kPoissonLossClampBounds,
                 model::gradient_boosted_trees::GradientBoostedTreesModel::
                     kPoissonLossClampBounds));
}

ABSL_ATTRIBUTE_ALWAYS_INLINE inline float ActivationIdentity(
    const float value) {
  return value;
}

// Tree inference without SIMD i.e. one example at a time.
// This method is used for the examples outside of the SIMD batch.
//
// "active_leaf_buffer" is a pre-allocated buffer of at least "num-trees"
// elements.
template <typename Model, float (*Activation)(float)>
ABSL_ATTRIBUTE_ALWAYS_INLINE void PredictQuickScorerSequential(
    const Model& model,
    const std::vector<NumericalOrCategoricalValue>& fixed_length_features,
    const std::vector<Rangei32>& categorical_set_begins_and_ends,
    const std::vector<int32_t>& categorical_item_buffer,
    const int begin_example_idx, const int end_example_idx,
    const int major_feature_offset, std::vector<float>* predictions,
    internal::QuickScorerExtendedModel::LeafMask* active_leaf_buffer) {
  const size_t active_leaf_buffer_size =
      model.num_trees * sizeof(internal::QuickScorerExtendedModel::LeafMask);

  const auto index = [&major_feature_offset](const int feature_idx,
                                             const int example_idx) -> int {
    return feature_idx * major_feature_offset + example_idx;
  };

  for (int example_idx = begin_example_idx; example_idx < end_example_idx;
       ++example_idx) {
    // Reset active node buffer.
    std::memset(active_leaf_buffer, 0xFF, active_leaf_buffer_size);

    // Is higher conditions.
    for (const auto& is_higher_condition : model.is_higher_conditions) {
      const auto feature_value =
          fixed_length_features[index(is_higher_condition.internal_feature_idx,
                                      example_idx)]
              .numerical_value;

      if (model.global_imputation_optimization || !std::isnan(feature_value)) {
        for (const auto& item : is_higher_condition.items) {
          if (item.threshold > feature_value) {
            break;
          }
          active_leaf_buffer[item.tree_idx] &= item.leaf_mask;
        }

      } else {
        for (const auto& item : is_higher_condition.missing_value_items) {
          active_leaf_buffer[item.tree_idx] &= item.leaf_mask;
        }
      }
    }

    // Dense contains conditions.
    for (const auto& contains_condition :
         model.categorical_contains_conditions) {
      const auto feature_value =
          fixed_length_features[index(contains_condition.internal_feature_idx,
                                      example_idx)]
              .categorical_value;
      DCHECK_LE(model.num_trees * (feature_value + 1),
                contains_condition.items.size());
      const auto* leaf_mask_stream =
          &contains_condition.items[model.num_trees * feature_value];
      for (int tree_idx = 0; tree_idx < model.num_trees; ++tree_idx) {
        active_leaf_buffer[tree_idx] &= *(leaf_mask_stream++);
      }
    }

    // Sparse contains conditions.
    for (const auto& contains_condition :
         model.categoricalset_contains_conditions) {
      const auto& range_values = categorical_set_begins_and_ends
          [contains_condition.internal_feature_idx * major_feature_offset +
           example_idx];
      for (int value_idx = range_values.begin; value_idx < range_values.end;
           value_idx++) {
        const auto value = categorical_item_buffer[value_idx] + 1;
        const auto& range_masks = contains_condition.value_to_mask_range[value];
        for (int mask_idx = range_masks.first; mask_idx < range_masks.second;
             mask_idx++) {
          const auto& mask = contains_condition.mask_buffer[mask_idx];
          active_leaf_buffer[mask.first] &= mask.second;
        }
      }
    }

    // Get the active leaf.
    auto* leaf_reader = model.leaf_values.data();
    float output = model.initial_prediction;
    for (int tree_idx = 0; tree_idx < model.num_trees; ++tree_idx) {
      const auto shift_mask = active_leaf_buffer[tree_idx];
      const auto node_idx = absl::countr_zero(shift_mask);
      output += leaf_reader[node_idx];
      leaf_reader += model.max_num_leafs_per_tree;
    }

    (*predictions)[example_idx] = Activation(output);
  }
}

}  // namespace internal

template <typename Model, float (*Activation)(float)>
void PredictQuickScorerHighway(
    const Model& model,
    const std::vector<NumericalOrCategoricalValue>& fixed_length_features,
    const std::vector<Rangei32>& categorical_set_begins_and_ends,
    const std::vector<int32_t>& categorical_item_buffer, int num_examples,
    int major_feature_offset, std::vector<float>* predictions);
}  // namespace yggdrasil_decision_forests::serving::decision_forest

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_QUICK_SCORER_EXTENDED_HWY_H_
