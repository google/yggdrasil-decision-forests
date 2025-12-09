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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/log.h"
#include "yggdrasil_decision_forests/serving/decision_forest/quick_scorer_extended.h"
#include "yggdrasil_decision_forests/serving/decision_forest/quick_scorer_extended_internal.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/utils/usage.h"

// clang-format off

#ifdef YDF_USE_DYNAMIC_DISPATCH
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "yggdrasil_decision_forests/serving/decision_forest/quick_scorer_extended_hwy.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#endif
// clang-format on

// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"
#include "hwy/per_target.h"  // IWYU pragma: keep
#ifdef YDF_USE_DYNAMIC_DISPATCH
HWY_BEFORE_NAMESPACE();
namespace yggdrasil_decision_forests::serving::decision_forest {
namespace HWY_NAMESPACE {
#else
namespace yggdrasil_decision_forests::serving::decision_forest {
#endif

namespace hn = hwy::HWY_NAMESPACE;

using LeafMask = internal::QuickScorerExtendedModel::LeafMask;

// Highway implementation of the QuickScorer algorithm.
template <typename Model, float (*Activation)(float)>
ABSL_ATTRIBUTE_ALWAYS_INLINE void PredictQuickScorerHighwayImpl(
    const Model& model,
    const std::vector<NumericalOrCategoricalValue>& fixed_length_features,
    const std::vector<Rangei32>& categorical_set_begins_and_ends,
    const std::vector<int32_t>& categorical_item_buffer, const int num_examples,
    const int major_feature_offset, std::vector<float>* predictions) {
  utils::usage::OnInference(num_examples, model.metadata);
  predictions->resize(num_examples);

  // "kNumParallelExamples" examples are treated in parallel using SIMD
  // instructions. If the number of examples is not a multiple of
  // "kNumParallelExamples", the remaining examples are treated with
  // "PredictQuickScorerSequential".
  const hn::ScalableTag<LeafMask> d;
  const int kNumParallelExamples = hn::Lanes(d);
  const hn::Rebind<float, decltype(d)> df;
  const hn::Rebind<int, decltype(d)> di;

  const size_t num_leaf_masks = model.num_trees * kNumParallelExamples;
  const size_t bytes_needed = num_leaf_masks * sizeof(LeafMask);
  const size_t required_stack_allocation = bytes_needed + HWY_ALIGNMENT;

  // Working pointer used by the algorithm (points to either stack or heap
  // memory)
  LeafMask* active_leaf_buffer = nullptr;

  // RAII holder for the heap case. Keeps memory alive until function returns.
  hwy::AlignedFreeUniquePtr<LeafMask[]> heap_guard;
  // Older AMD machines (e.g. Milan) are ~10% slower on heap.
  if (required_stack_allocation <= internal::kMaxStackUsageInBytes) {
#if defined(_WIN32)
    void* ptr = _alloca(required_stack_allocation);
#else
    void* ptr = __builtin_alloca(required_stack_allocation);
#endif

    // Manually align the pointer
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_addr = (addr + HWY_ALIGNMENT - 1) & ~(HWY_ALIGNMENT - 1);
    active_leaf_buffer = reinterpret_cast<LeafMask*>(aligned_addr);
  } else {
    heap_guard = hwy::AllocateAligned<LeafMask>(num_leaf_masks);
    active_leaf_buffer = heap_guard.get();
  }

  int example_idx = 0;

  {
    auto* sample_reader = fixed_length_features.data();
    auto* prediction_reader = predictions->data();

    // First run on sub-batches of kNumParallelExamples at a time. The
    // remaining will be done sequentially below.
    int num_remaining_iters = num_examples / kNumParallelExamples;
    while (num_remaining_iters--) {
      // Reset active node buffer.
      std::memset(active_leaf_buffer, 0xFF, bytes_needed);

      // Is higher conditions.
      for (const auto& is_higher_condition : model.is_higher_conditions) {
        const float* begin_example =
            &sample_reader[0].numerical_value +
            is_higher_condition.internal_feature_idx * major_feature_offset;
        const auto feature_values = hn::LoadU(df, begin_example);

        // Note: model.global_imputation_optimization = true for nearly all
        // models, so this branch is almost never taken.
        if (!model.global_imputation_optimization) {
          // If any feature value is Nan
          //   Create NaN mask
          //   Iterate over examples and apply leaf mask * nan mask
          //   Replace value as - infinity in next loop

          // Test for the existence of at least one missing value.
          // mask_no_nan_128 is a bitmask of the non-missing values.
          const auto nan_mask = hn::IsNaN(feature_values);
          if (!hn::AllFalse(df, nan_mask)) {
            // At least one of the feature contains a missing value.
            const auto no_nan_mask_d =
                hn::VecFromMask(d, hn::Not(hn::PromoteMaskTo(d, df, nan_mask)));
            // Apply all the masks
            for (const auto& item : is_higher_condition.missing_value_items) {
              // For now, use unaligned load for slightly better performance.
              const auto active = hn::LoadU(
                  d, &active_leaf_buffer[item.tree_idx * kNumParallelExamples]);
              const auto leaf_mask = hn::Set(d, item.leaf_mask);
              const auto new_active =
                  hn::And(active, hn::Or(leaf_mask, no_nan_mask_d));
              hn::Store(
                  new_active, d,
                  &active_leaf_buffer[item.tree_idx * kNumParallelExamples]);
            }

            // Missing values are represented as Nan. They will fail at the
            // first comparison "value >= threshold" in the next loop.
          }
        }

        for (const auto& item : is_higher_condition.items) {
          const auto threshold = hn::Set(df, item.threshold);

          const auto comparison = hn::Ge(feature_values, threshold);
          // Note: "comparison" is either 0x00000000 or 0xFFFFFFFF depending on
          // the node condition value.
          if (!hn::AllFalse(df, comparison)) {
            auto active_leaf_buffer_ptr =
                &active_leaf_buffer[item.tree_idx * kNumParallelExamples];
            // For now, use unaligned load for slightly better performance.
            auto active = hn::LoadU(d, active_leaf_buffer_ptr);
            // The mask attached to the condition i.e. the mask to apply on the
            // active node bitmap iif. the condition is true.
            const auto mask = hn::Set(d, item.leaf_mask);
            const auto new_active =
                hn::IfThenElse(hn::PromoteMaskTo(d, df, comparison),
                               hn::And(active, mask), active);

            hn::StoreU(new_active, d, active_leaf_buffer_ptr);
          } else {
            break;
          }
        }
      }
      for (const auto& contains_condition :
           model.categorical_contains_conditions) {
        const int* begin_example =
            &sample_reader[0].categorical_value +
            contains_condition.internal_feature_idx * major_feature_offset;

        alignas(64) int32_t feature_vals[kNumParallelExamples];
        const auto feature_values_vec = hn::LoadU(di, begin_example);
        hn::Store(feature_values_vec, di, feature_vals);

        const LeafMask* leaf_mask_streams[kNumParallelExamples];
        for (int sub_example_idx = 0; sub_example_idx < kNumParallelExamples;
             ++sub_example_idx) {
          leaf_mask_streams[sub_example_idx] =
              &contains_condition
                   .items[model.num_trees * feature_vals[sub_example_idx]];
        }

        alignas(64) LeafMask tmp_masks[kNumParallelExamples];
        for (int tree_idx = 0; tree_idx < model.num_trees; ++tree_idx) {
          for (size_t sub_example_idx = 0;
               sub_example_idx < kNumParallelExamples; ++sub_example_idx) {
            tmp_masks[sub_example_idx] =
                *(leaf_mask_streams[sub_example_idx]++);
          }
          const auto leaf_masks = hn::Load(d, tmp_masks);
          auto* active_ptr =
              &active_leaf_buffer[tree_idx * kNumParallelExamples];
          const auto current_active = hn::Load(d, active_ptr);
          hn::StoreU(hn::And(current_active, leaf_masks), d, active_ptr);
        }
      }

      // Sparse contains conditions.
      for (int sub_example_idx = 0; sub_example_idx < kNumParallelExamples;
           ++sub_example_idx) {
        for (const auto& contains_condition :
             model.categoricalset_contains_conditions) {
          const auto& range_values = categorical_set_begins_and_ends
              [contains_condition.internal_feature_idx * major_feature_offset +
               sub_example_idx + example_idx];
          for (int value_idx = range_values.begin; value_idx < range_values.end;
               value_idx++) {
            const auto value = categorical_item_buffer[value_idx] + 1;
            const auto& range_masks =
                contains_condition.value_to_mask_range[value];
            for (int mask_idx = range_masks.first;
                 mask_idx < range_masks.second; mask_idx++) {
              const auto& mask = contains_condition.mask_buffer[mask_idx];
              active_leaf_buffer[mask.first * kNumParallelExamples +
                                 sub_example_idx] &= mask.second;
            }
          }
        }
      }

#pragma loop unroll(full)
      for (int sub_example_idx = 0; sub_example_idx < kNumParallelExamples;
           ++sub_example_idx) {
        prediction_reader[sub_example_idx] = model.initial_prediction;
      }

      auto* leaf_reader = model.leaf_values.data();
      for (int tree_idx = 0; tree_idx < model.num_trees; ++tree_idx) {
#pragma loop unroll(full)
        for (int sub_example_idx = 0; sub_example_idx < kNumParallelExamples;
             ++sub_example_idx) {
          const auto shift_mask =
              active_leaf_buffer[tree_idx * kNumParallelExamples +
                                 sub_example_idx];
          const auto node_idx = absl::countr_zero(shift_mask);
          prediction_reader[sub_example_idx] += leaf_reader[node_idx];
        }
        leaf_reader += model.max_num_leafs_per_tree;
      }

// Note: The compiler should be able to remove the following loop when
// Activation == Identity. Tested with gcc9 and clang9.
#pragma loop unroll(full)
      for (int sub_example_idx = 0; sub_example_idx < kNumParallelExamples;
           ++sub_example_idx) {
        prediction_reader[sub_example_idx] =
            Activation(prediction_reader[sub_example_idx]);
      }

      sample_reader += kNumParallelExamples;
      prediction_reader += kNumParallelExamples;
      example_idx += kNumParallelExamples;
    }
  }

  internal::PredictQuickScorerSequential<Model, Activation>(
      model, fixed_length_features, categorical_set_begins_and_ends,
      categorical_item_buffer, example_idx, num_examples, major_feature_offset,
      predictions, active_leaf_buffer);
}

// Highway currently doesn't support templated functions with two templates for
// dynamic dispatch, this is a workaround used for dynamic dispatch.
template <typename Model>
ABSL_ATTRIBUTE_ALWAYS_INLINE void PredictQuickScorerHighwayIdentity(
    const Model& model,
    const std::vector<NumericalOrCategoricalValue>& fixed_length_features,
    const std::vector<Rangei32>& categorical_set_begins_and_ends,
    const std::vector<int32_t>& categorical_item_buffer, const int num_examples,
    const int major_feature_offset, std::vector<float>* predictions) {
  PredictQuickScorerHighwayImpl<Model, internal::ActivationIdentity>(
      model, fixed_length_features, categorical_set_begins_and_ends,
      categorical_item_buffer, num_examples, major_feature_offset, predictions);
}

template <typename Model>
ABSL_ATTRIBUTE_ALWAYS_INLINE void PredictQuickScorerHighwayLogLikelihood(
    const Model& model,
    const std::vector<NumericalOrCategoricalValue>& fixed_length_features,
    const std::vector<Rangei32>& categorical_set_begins_and_ends,
    const std::vector<int32_t>& categorical_item_buffer, const int num_examples,
    const int major_feature_offset, std::vector<float>* predictions) {
  PredictQuickScorerHighwayImpl<Model,
                                internal::ActivationBinomialLogLikelihood>(
      model, fixed_length_features, categorical_set_begins_and_ends,
      categorical_item_buffer, num_examples, major_feature_offset, predictions);
}

template <typename Model>
ABSL_ATTRIBUTE_ALWAYS_INLINE void PredictQuickScorerHighwayPoisson(
    const Model& model,
    const std::vector<NumericalOrCategoricalValue>& fixed_length_features,
    const std::vector<Rangei32>& categorical_set_begins_and_ends,
    const std::vector<int32_t>& categorical_item_buffer, const int num_examples,
    const int major_feature_offset, std::vector<float>* predictions) {
  PredictQuickScorerHighwayImpl<Model, internal::ActivationPoisson>(
      model, fixed_length_features, categorical_set_begins_and_ends,
      categorical_item_buffer, num_examples, major_feature_offset, predictions);
}

#ifdef YDF_USE_DYNAMIC_DISPATCH
}  // namespace HWY_NAMESPACE
}  // namespace yggdrasil_decision_forests::serving::decision_forest
HWY_AFTER_NAMESPACE();
#else

}  // namespace yggdrasil_decision_forests::serving::decision_forest  // NOLINT
#endif  // YDF_USE_DYNAMIC_DISPATCH

#if HWY_ONCE
namespace yggdrasil_decision_forests::serving::decision_forest {

template <typename Model, float (*Activation)(float)>
void PredictQuickScorerHighway(
    const Model& model,
    const std::vector<NumericalOrCategoricalValue>& fixed_length_features,
    const std::vector<Rangei32>& categorical_set_begins_and_ends,
    const std::vector<int32_t>& categorical_item_buffer, int num_examples,
    int major_feature_offset, std::vector<float>* predictions) {
#ifdef YDF_USE_DYNAMIC_DISPATCH
  if constexpr (Activation == internal::ActivationBinomialLogLikelihood) {
    HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(
        PredictQuickScorerHighwayLogLikelihood<Model>)(
        model, fixed_length_features, categorical_set_begins_and_ends,
        categorical_item_buffer, num_examples, major_feature_offset,
        predictions);
  } else if constexpr (Activation == internal::ActivationPoisson) {
    HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(PredictQuickScorerHighwayPoisson<Model>)(
        model, fixed_length_features, categorical_set_begins_and_ends,
        categorical_item_buffer, num_examples, major_feature_offset,
        predictions);
  } else if constexpr (Activation == internal::ActivationIdentity) {
    HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(PredictQuickScorerHighwayIdentity<Model>)(
        model, fixed_length_features, categorical_set_begins_and_ends,
        categorical_item_buffer, num_examples, major_feature_offset,
        predictions);
  } else {
    LOG(ERROR) << "Unknown Activation function";
  }

#else
  PredictQuickScorerHighwayImpl<Model, Activation>(
      model, fixed_length_features, categorical_set_begins_and_ends,
      categorical_item_buffer, num_examples, major_feature_offset, predictions);
#endif  // YDF_USE_DYNAMIC_DISPATCH
}

#define INSTANTIATE_HIGHWAY(ModelType, Activation)                         \
  template void PredictQuickScorerHighway<ModelType, Activation>(          \
      const ModelType&, const std::vector<NumericalOrCategoricalValue>&,   \
      const std::vector<Rangei32>&, const std::vector<int32_t>&, int, int, \
      std::vector<float>*);

INSTANTIATE_HIGHWAY(GradientBoostedTreesRegressionQuickScorerExtendedHighway,
                    internal::ActivationIdentity);
INSTANTIATE_HIGHWAY(
    GradientBoostedTreesBinaryClassificationQuickScorerExtendedHighway,
    internal::ActivationBinomialLogLikelihood);
INSTANTIATE_HIGHWAY(
    GradientBoostedTreesBinaryClassificationQuickScorerExtendedHighway,
    internal::ActivationIdentity);
INSTANTIATE_HIGHWAY(GradientBoostedTreesRankingQuickScorerExtendedHighway,
                    internal::ActivationIdentity);
INSTANTIATE_HIGHWAY(
    GradientBoostedTreesPoissonRegressionQuickScorerExtendedHighway,
    internal::ActivationPoisson);
INSTANTIATE_HIGHWAY(
    GradientBoostedTreesPoissonRegressionQuickScorerExtendedHighway,
    internal::ActivationIdentity);

}  // namespace yggdrasil_decision_forests::serving::decision_forest
#endif  // HWY_ONCE