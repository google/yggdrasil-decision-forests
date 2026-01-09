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

#include "yggdrasil_decision_forests/serving/decision_forest/quick_scorer_extended.h"

#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "yggdrasil_decision_forests/serving/decision_forest/quick_scorer_extended_internal.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "absl/base/attributes.h"
#include "absl/base/config.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "hwy/targets.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/serving/decision_forest/utils.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/usage.h"

#ifdef YDF_USE_DYNAMIC_DISPATCH
#include "hwy/per_target.h"
#endif

namespace yggdrasil_decision_forests::serving::decision_forest {
using dataset::proto::ColumnType;
using LeafMask = internal::QuickScorerExtendedModel::LeafMask;
using model::gradient_boosted_trees::proto::Loss;
using ::yggdrasil_decision_forests::utils::bitmap::ToStringBit;

template <typename Model,
          float (*Activation)(float) = internal::ActivationIdentity>
ABSL_ATTRIBUTE_ALWAYS_INLINE void PredictQuickScorerMajorFeatureOffsetLegacy(
    const Model& model,
    const std::vector<NumericalOrCategoricalValue>& fixed_length_features,
    const std::vector<Rangei32>& categorical_set_begins_and_ends,
    const std::vector<int32_t>& categorical_item_buffer, int num_examples,
    int major_feature_offset, std::vector<float>* predictions);

// Apply the quick scorer algorithm.
//
// The examples are represented in the arguments "fixed_length_features",
// "categorical_item_buffer" and "categorical_set_begins_and_ands". These fields
// are made to be contained in the "ExampleSet" class. Refer to this class for
// their definition.
//
// "major_feature_offset" is the number of elements in between features blocks
// i.e. the j-th features of the i-th examples is "index = i + j *
// major_feature_offset".
//
template <typename Model,
          float (*Activation)(float) = internal::ActivationIdentity>
void PredictQuickScorerMajorFeatureOffset(
    const Model& model,
    const std::vector<NumericalOrCategoricalValue>& fixed_length_features,
    const std::vector<Rangei32>& categorical_set_begins_and_ends,
    const std::vector<int32_t>& categorical_item_buffer, const int num_examples,
    const int major_feature_offset, std::vector<float>* predictions) {
  if constexpr (std::is_base_of_v<QuickScorerExtendedModelLegacy, Model>) {
    PredictQuickScorerMajorFeatureOffsetLegacy<Model, Activation>(
        model, fixed_length_features, categorical_set_begins_and_ends,
        categorical_item_buffer, num_examples, major_feature_offset,
        predictions);
  } else if constexpr (std::is_base_of_v<QuickScorerExtendedModelHighway,
                                         Model>) {
    PredictQuickScorerHighway<Model, Activation>(
        model, fixed_length_features, categorical_set_begins_and_ends,
        categorical_item_buffer, num_examples, major_feature_offset,
        predictions);
  } else {
    LOG(ERROR) << "Unknown model type";
  }
}

// Apply the quick scorer algorithm.
//
// The examples are represented in the arguments "fixed_length_features",
// "categorical_item_buffer" and "categorical_set_begins_and_ands". These fields
// are made to be contained in the "ExampleSet" class. Refer to this class for
// their definition.
//
// "major_feature_offset" is the number of elements in between features blocks
// i.e. the j-th features of the i-th examples is "index = i + j *
// major_feature_offset".
//
template <typename Model, float (*Activation)(float)>
void PredictQuickScorerMajorFeatureOffsetLegacy(
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
  constexpr int kNumParallelExamples = 4;

  const size_t active_leaf_buffer_size =
      model.num_trees * kNumParallelExamples * sizeof(LeafMask);
  const size_t alignment = 32 * 8;

  // Make sure the allocated chunk of memory is a multiple of "alignment".
  size_t rounded_up_active_leaf_buffer_size = active_leaf_buffer_size;
  if ((rounded_up_active_leaf_buffer_size % alignment) != 0) {
    rounded_up_active_leaf_buffer_size +=
        alignment - rounded_up_active_leaf_buffer_size % alignment;
  }

  // Note: Alloca was measured to be faster and more consistent (in terms of
  // speed) than malloc or pre-allocated caches.
  //
  // The buffer must be aligned on a 32-byte boundary to work with _mm256
  // class of SIMD instructions (intrinsics).
  LeafMask* active_leaf_buffer;
  const bool active_leaf_buffer_uses_stack =
      active_leaf_buffer_size <= internal::kMaxStackUsageInBytes;

  if (active_leaf_buffer_uses_stack) {
#ifdef __AVX2__

#if defined(_WIN32)
    void* non_aligned = alloca(rounded_up_active_leaf_buffer_size + alignment);
    std::size_t space = rounded_up_active_leaf_buffer_size + alignment;
    void* aligned = std::align(alignment, 1, non_aligned, space);
#else
    void* aligned = __builtin_alloca_with_align(
        rounded_up_active_leaf_buffer_size, alignment);
#endif
    active_leaf_buffer = reinterpret_cast<LeafMask*>(aligned);

#else
    active_leaf_buffer =
        reinterpret_cast<LeafMask*>(alloca(rounded_up_active_leaf_buffer_size));
#endif
  } else {
#ifdef __AVX2__
    active_leaf_buffer = reinterpret_cast<LeafMask*>(
        ::aligned_alloc(alignment, rounded_up_active_leaf_buffer_size));
#else
    active_leaf_buffer = reinterpret_cast<LeafMask*>(
        std::malloc(rounded_up_active_leaf_buffer_size));
#endif
  }

  int example_idx = 0;

#ifdef __AVX2__
  if (model.cpu_supports_avx2) {
    // `fixed_length_features` can be empty, in which case the data pointer is
    // nullptr. Performing arithmetic on the address avoids UB and branches.
    auto sample_reader_addr =
        reinterpret_cast<uintptr_t>(fixed_length_features.data());
    auto* prediction_reader = predictions->data();

    // First run on sub-batches of kNumParallelExamples at a time. The
    // remaining will be done sequentially below.
    int num_remaining_iters = num_examples / kNumParallelExamples;
    while (num_remaining_iters--) {
      auto* sample_reader =
          reinterpret_cast<const NumericalOrCategoricalValue*>(
              sample_reader_addr);

      // Reset active node buffer.
      std::memset(active_leaf_buffer, 0xFF, active_leaf_buffer_size);

      // Is higher conditions.
      for (const auto& is_higher_condition : model.is_higher_conditions) {
        const float* begin_example =
            &sample_reader[0].numerical_value +
            is_higher_condition.internal_feature_idx * major_feature_offset;

        const auto feature_values = _mm_loadu_ps(begin_example);

        if (!model.global_imputation_optimization) {
          // If any feature value is Nan
          //   Create NaN mask
          //   Iterate over examples and apply leaf mask * nan mask
          //   Replace value as - infinity in next loop

          // Test for the existence of at least one missing value.
          // mask_no_nan_128 is a bitmask of the non-missing values.
          __m128i mask_no_nan_128 =
              _mm_castps_si128(_mm_cmpeq_ps(feature_values, feature_values));
          int has_nan = !_mm_test_all_ones(mask_no_nan_128);
          if (has_nan) {
            // At least one of the feature contains a missing value.

            // Nan mask in 256 bits
            __m256i mask_no_nan_256 = _mm256_cvtepi32_epi64(mask_no_nan_128);

            // Apply all the masks
            for (const auto& item : is_higher_condition.missing_value_items) {
              // Update the active node
              auto* active_si256 = reinterpret_cast<__m256i*>(
                  &active_leaf_buffer[item.tree_idx * kNumParallelExamples]);

              const auto active = _mm256_load_si256(active_si256);
              // new_active = active & ( mask_split | mask_no_nan )
              const auto new_active = _mm256_and_si256(
                  active, _mm256_or_si256(_mm256_set1_epi64x(item.leaf_mask),
                                          mask_no_nan_256));
              _mm256_store_si256(active_si256, new_active);
            }

            // Missing values are represented as Nan. They will fail at the
            // first comparison "value >= threshold" in the next loop.
          }
        }

        for (const auto& item : is_higher_condition.items) {
          const auto threshold = _mm_set1_ps(item.threshold);

          const auto comparison =
              _mm_castps_si128(_mm_cmpge_ps(feature_values, threshold));
          // Note: "comparison" is either 0x00000000 or 0xFFFFFFFF depending on
          // the node condition value.
          if (!_mm_test_all_zeros(comparison, comparison)) {
            // The mask attached to the condition i.e. the mask to apply on the
            // active node bitmap iif. the condition is true.
            const auto mask = _mm256_set1_epi64x(item.leaf_mask);
            auto* active_si256 = reinterpret_cast<__m256i*>(
                &active_leaf_buffer[item.tree_idx * kNumParallelExamples]);
            const auto active = _mm256_load_si256(active_si256);

            // Expand the comparison to 8 bytes.
            const auto pd_comparison = _mm256_cvtepi32_epi64(comparison);
            const auto mask_update = _mm256_andnot_si256(mask, pd_comparison);
            const auto new_active = _mm256_andnot_si256(mask_update, active);
            // new_active = (mask v not comparison) ^ active
            // is equivalent to:
            // new_active = not (not mask ^ comparison) ^ active

            _mm256_store_si256(active_si256, new_active);
          } else {
            break;
          }
        }
      }

      // Dense contains conditions.
      for (int sub_example_idx = 0; sub_example_idx < kNumParallelExamples;
           ++sub_example_idx) {
        for (const auto& contains_condition :
             model.categorical_contains_conditions) {
          const auto feature_value =
              sample_reader[contains_condition.internal_feature_idx *
                                major_feature_offset +
                            sub_example_idx]
                  .categorical_value;
          const auto* leaf_mask_stream =
              &contains_condition.items[model.num_trees * feature_value];
          for (int tree_idx = 0; tree_idx < model.num_trees; ++tree_idx) {
            active_leaf_buffer[tree_idx * kNumParallelExamples +
                               sub_example_idx] &= *(leaf_mask_stream++);
          }
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

      sample_reader_addr +=
          kNumParallelExamples * sizeof(NumericalOrCategoricalValue);
      prediction_reader += kNumParallelExamples;
      example_idx += kNumParallelExamples;
    }
  }
#endif

  internal::PredictQuickScorerSequential<Model, Activation>(
      model, fixed_length_features, categorical_set_begins_and_ends,
      categorical_item_buffer, example_idx, num_examples, major_feature_offset,
      predictions, active_leaf_buffer);

  if (!active_leaf_buffer_uses_stack) {
    free(active_leaf_buffer);
  }
}

template <typename Model>
void PredictQuickScorer(
    const Model& model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictQuickScorerMajorFeatureOffset(model, examples, {}, {}, num_examples,
                                       num_examples, predictions);
}

// Version of Predict compatible with the ExampleSet signature.
template <typename Model>
void Predict(const Model& model, const typename Model::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictQuickScorerMajorFeatureOffset(
      model, examples.InternalCategoricalAndNumericalValues(),
      examples.InternalCategoricalSetBeginAndEnds(),
      examples.InternalCategoricalItemBuffer(), num_examples,
      examples.NumberOfExamples(), predictions);
}

template void
PredictQuickScorer<GradientBoostedTreesRegressionQuickScorerExtended>(
    const GradientBoostedTreesRegressionQuickScorerExtended& model,
    const std::vector<NumericalOrCategoricalValue>& examples,
    const int num_examples, std::vector<float>* predictions);

template void Predict<GradientBoostedTreesRegressionQuickScorerExtended>(
    const GradientBoostedTreesRegressionQuickScorerExtended& model,
    const GradientBoostedTreesRegressionQuickScorerExtended::ExampleSet&
        examples,
    const int num_examples, std::vector<float>* predictions);

template void Predict<GradientBoostedTreesRankingQuickScorerExtended>(
    const GradientBoostedTreesRankingQuickScorerExtended& model,
    const GradientBoostedTreesRankingQuickScorerExtended::ExampleSet& examples,
    const int num_examples, std::vector<float>* predictions);

template <>
void Predict(
    const GradientBoostedTreesPoissonRegressionQuickScorerExtended& model,
    const GradientBoostedTreesPoissonRegressionQuickScorerExtended::ExampleSet&
        examples,
    const int num_examples, std::vector<float>* predictions) {
  if (model.output_logits) {
    PredictQuickScorerMajorFeatureOffset<
        GradientBoostedTreesPoissonRegressionQuickScorerExtended>(
        model, examples.InternalCategoricalAndNumericalValues(),
        examples.InternalCategoricalSetBeginAndEnds(),
        examples.InternalCategoricalItemBuffer(), num_examples,
        examples.NumberOfExamples(), predictions);
  } else {
    PredictQuickScorerMajorFeatureOffset<
        GradientBoostedTreesPoissonRegressionQuickScorerExtended,
        internal::ActivationPoisson>(
        model, examples.InternalCategoricalAndNumericalValues(),
        examples.InternalCategoricalSetBeginAndEnds(),
        examples.InternalCategoricalItemBuffer(), num_examples,
        examples.NumberOfExamples(), predictions);
  }
}

template <>
void Predict(
    const GradientBoostedTreesBinaryClassificationQuickScorerExtended& model,
    const GradientBoostedTreesBinaryClassificationQuickScorerExtended::
        ExampleSet& examples,
    const int num_examples, std::vector<float>* predictions) {
  if (model.output_logits) {
    PredictQuickScorerMajorFeatureOffset<
        GradientBoostedTreesBinaryClassificationQuickScorerExtended>(
        model, examples.InternalCategoricalAndNumericalValues(),
        examples.InternalCategoricalSetBeginAndEnds(),
        examples.InternalCategoricalItemBuffer(), num_examples,
        examples.NumberOfExamples(), predictions);
  } else {
    PredictQuickScorerMajorFeatureOffset<
        GradientBoostedTreesBinaryClassificationQuickScorerExtended,
        internal::ActivationBinomialLogLikelihood>(
        model, examples.InternalCategoricalAndNumericalValues(),
        examples.InternalCategoricalSetBeginAndEnds(),
        examples.InternalCategoricalItemBuffer(), num_examples,
        examples.NumberOfExamples(), predictions);
  }
}

template void
PredictQuickScorer<GradientBoostedTreesRegressionQuickScorerExtendedHighway>(
    const GradientBoostedTreesRegressionQuickScorerExtendedHighway& model,
    const std::vector<NumericalOrCategoricalValue>& examples,
    const int num_examples, std::vector<float>* predictions);

template void Predict<GradientBoostedTreesRegressionQuickScorerExtendedHighway>(
    const GradientBoostedTreesRegressionQuickScorerExtendedHighway& model,
    const GradientBoostedTreesRegressionQuickScorerExtendedHighway::ExampleSet&
        examples,
    const int num_examples, std::vector<float>* predictions);

template void Predict<GradientBoostedTreesRankingQuickScorerExtendedHighway>(
    const GradientBoostedTreesRankingQuickScorerExtendedHighway& model,
    const GradientBoostedTreesRankingQuickScorerExtendedHighway::ExampleSet&
        examples,
    const int num_examples, std::vector<float>* predictions);

template <>
void Predict(
    const GradientBoostedTreesPoissonRegressionQuickScorerExtendedHighway&
        model,
    const GradientBoostedTreesPoissonRegressionQuickScorerExtendedHighway::
        ExampleSet& examples,
    const int num_examples, std::vector<float>* predictions) {
  if (model.output_logits) {
    PredictQuickScorerMajorFeatureOffset<
        GradientBoostedTreesPoissonRegressionQuickScorerExtendedHighway>(
        model, examples.InternalCategoricalAndNumericalValues(),
        examples.InternalCategoricalSetBeginAndEnds(),
        examples.InternalCategoricalItemBuffer(), num_examples,
        examples.NumberOfExamples(), predictions);
  } else {
    PredictQuickScorerMajorFeatureOffset<
        GradientBoostedTreesPoissonRegressionQuickScorerExtendedHighway,
        internal::ActivationPoisson>(
        model, examples.InternalCategoricalAndNumericalValues(),
        examples.InternalCategoricalSetBeginAndEnds(),
        examples.InternalCategoricalItemBuffer(), num_examples,
        examples.NumberOfExamples(), predictions);
  }
}

template <>
void Predict(
    const GradientBoostedTreesBinaryClassificationQuickScorerExtendedHighway&
        model,
    const GradientBoostedTreesBinaryClassificationQuickScorerExtendedHighway::
        ExampleSet& examples,
    const int num_examples, std::vector<float>* predictions) {
  if (model.output_logits) {
    PredictQuickScorerMajorFeatureOffset<
        GradientBoostedTreesBinaryClassificationQuickScorerExtendedHighway>(
        model, examples.InternalCategoricalAndNumericalValues(),
        examples.InternalCategoricalSetBeginAndEnds(),
        examples.InternalCategoricalItemBuffer(), num_examples,
        examples.NumberOfExamples(), predictions);
  } else {
    PredictQuickScorerMajorFeatureOffset<
        GradientBoostedTreesBinaryClassificationQuickScorerExtendedHighway,
        internal::ActivationBinomialLogLikelihood>(
        model, examples.InternalCategoricalAndNumericalValues(),
        examples.InternalCategoricalSetBeginAndEnds(),
        examples.InternalCategoricalItemBuffer(), num_examples,
        examples.NumberOfExamples(), predictions);
  }
}

template <typename AbstractModel, typename CompiledModel>
absl::Status BaseGenericToSpecializedModel(const AbstractModel& src,
                                           CompiledModel* dst,
                                           const bool use_highway) {
  if (use_highway) {
#ifdef YDF_USE_DYNAMIC_DISPATCH
    LOG_FIRST_N(INFO, 1) << "Highway dynamic dispatch to CPU Target: "
                         << hwy::TargetName(hwy::DispatchedTarget());
#else
    LOG_FIRST_N(INFO, 1) << "Highway static dispatch to CPU Target: "
                         << hwy::TargetName(HWY_TARGET);
#endif  // YDF_USE_DYNAMIC_DISPATCH
  } else {
    LOG_FIRST_N(INFO, 1)
        << "Using legacy Quickscorer intrinsics. Consider using the Highway "
           "engine for better performance by updating your build flags.";
#ifdef __AVX2__
#if ABSL_HAVE_BUILTIN(__builtin_cpu_supports)
  dst->cpu_supports_avx2 = __builtin_cpu_supports("avx2");
#else
  // We cannot detect if the CPU supports AVX2 instructions. If it does not,
  // a fatal error will be raised.
  dst->cpu_supports_avx2 = true;
#endif
// We need the platform check before calling '__builtin_cpu_supports("avx2")'
// due to a known issue in 'clang' which will trigger a compilation error if the
// checked CPU feature is not valid for the platform. Hence, the call to
// '__builtin_cpu_supports("avx2")' would emit a compilation error when compiled
// for ARM.
// This is a temporary fix until this issue, tracked under
// https://github.com/llvm/llvm-project/issues/83407 is fixed.
#elif ABSL_HAVE_BUILTIN(__builtin_cpu_supports) && \
    (defined(__x86_64__) || defined(__i386__))
  if (__builtin_cpu_supports("avx2")) {
    LOG_EVERY_N_SEC(INFO, 30)
        << "The binary was compiled without AVX2 support, but your CPU "
           "supports it. Enable it for faster model inference.";
  }
#endif
  }

  if (src.task() != CompiledModel::kTask) {
    return absl::InvalidArgumentError("Wrong model class.");
  }

  src.metadata().Export(&dst->metadata);

  typename CompiledModel::BuildingAccumulator accumulator;

  // List the model input features.
  std::vector<int> all_input_features;
  RETURN_IF_ERROR(GetInputFeatures(src, &all_input_features, nullptr));

  dst->global_imputation_optimization =
      src.CheckStructure({/*.global_imputation_is_higher =*/true});

  RETURN_IF_ERROR(dst->mutable_features()->Initialize(
      all_input_features, src.data_spec(),
      /*missing_numerical_is_na=*/!dst->global_imputation_optimization));

  // Compile the model.
  RETURN_IF_ERROR(FillQuickScorer(src, dst, &accumulator));

  return absl::OkStatus();
}

template <>
absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesRegressionQuickScorerExtended* dst) {
  if (src.loss() != Loss::SQUARED_ERROR &&
      src.loss() != Loss::MEAN_AVERAGE_ERROR) {
    return absl::InvalidArgumentError(
        "The GBDT is not trained for regression with squared error loss.");
  }
  return BaseGenericToSpecializedModel(src, dst, false);
}

template <>
absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesPoissonRegressionQuickScorerExtended* dst) {
  if (src.loss() != Loss::POISSON) {
    return absl::InvalidArgumentError(
        "The GBDT is not trained for regression with poisson loss.");
  }
  return BaseGenericToSpecializedModel(src, dst, false);
}

template <>
absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesRankingQuickScorerExtended* dst) {
  if (src.loss() != Loss::LAMBDA_MART_NDCG5 &&
      src.loss() != Loss::LAMBDA_MART_NDCG &&
      src.loss() != Loss::XE_NDCG_MART) {
    return absl::InvalidArgumentError(
        "The GBDT is not trained for ranking with ranking loss.");
  }
  return BaseGenericToSpecializedModel(src, dst, false);
}

template <>
absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesBinaryClassificationQuickScorerExtended* dst) {
  if ((src.loss() != Loss::BINOMIAL_LOG_LIKELIHOOD &&
       src.loss() != Loss::BINARY_FOCAL_LOSS) ||
      src.initial_predictions().size() != 1) {
    return absl::InvalidArgumentError(
        "The GBDT is not trained for binary classification with binomial log "
        "likelihood or binary focal loss.");
  }
  return BaseGenericToSpecializedModel(src, dst, false);
}

template <>
absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesRegressionQuickScorerExtendedHighway* dst) {
  if (src.loss() != Loss::SQUARED_ERROR &&
      src.loss() != Loss::MEAN_AVERAGE_ERROR) {
    return absl::InvalidArgumentError(
        "The GBDT is not trained for regression with squared error loss.");
  }
  return BaseGenericToSpecializedModel(src, dst, true);
}

template <>
absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesPoissonRegressionQuickScorerExtendedHighway* dst) {
  if (src.loss() != Loss::POISSON) {
    return absl::InvalidArgumentError(
        "The GBDT is not trained for regression with poisson loss.");
  }
  return BaseGenericToSpecializedModel(src, dst, true);
}

template <>
absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesRankingQuickScorerExtendedHighway* dst) {
  if (src.loss() != Loss::LAMBDA_MART_NDCG5 &&
      src.loss() != Loss::LAMBDA_MART_NDCG &&
      src.loss() != Loss::XE_NDCG_MART) {
    return absl::InvalidArgumentError(
        "The GBDT is not trained for ranking with ranking loss.");
  }
  return BaseGenericToSpecializedModel(src, dst, true);
}

template <>
absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesBinaryClassificationQuickScorerExtendedHighway* dst) {
  if ((src.loss() != Loss::BINOMIAL_LOG_LIKELIHOOD &&
       src.loss() != Loss::BINARY_FOCAL_LOSS) ||
      src.initial_predictions().size() != 1) {
    return absl::InvalidArgumentError(
        "The GBDT is not trained for binary classification with binomial log "
        "likelihood or binary focal loss.");
  }
  return BaseGenericToSpecializedModel(src, dst, true);
}

template <typename CompiledModel>
absl::Status CreateEmptyModel(const std::vector<int>& input_features,
                              const dataset::proto::DataSpecification& dataspec,
                              CompiledModel* dst) {
  return dst->mutable_features()->Initialize(input_features, dataspec);
}

template absl::Status
CreateEmptyModel<GradientBoostedTreesRegressionQuickScorerExtended>(
    const std::vector<int>& input_features,
    const dataset::proto::DataSpecification& dataspec,
    GradientBoostedTreesRegressionQuickScorerExtended* dst);

template <typename Model>
std::string DescribeQuickScorer(const Model& model, const bool detailed) {
  std::string structure;

  // Global data.
  absl::SubstituteAndAppend(&structure,
                            "Maximum number of leafs per trees: $0\n",
                            model.max_num_leafs_per_tree);
  absl::SubstituteAndAppend(&structure, "Number of trees: $0\n",
                            model.num_trees);
  absl::SubstituteAndAppend(&structure, "Initial prediction: $0\n",
                            model.initial_prediction);

  // List of input features.
  absl::StrAppend(&structure, "Features (and missing replacement value):\n");
  for (const auto& feature : model.features().fixed_length_features()) {
    absl::SubstituteAndAppend(&structure, "\t$0 [$1]", feature.name,
                              dataset::proto::ColumnType_Name(feature.type));
    switch (feature.type) {
      case ColumnType::NUMERICAL:
      case ColumnType::DISCRETIZED_NUMERICAL:
        absl::SubstituteAndAppend(
            &structure, "($0)\n",
            model.features()
                .fixed_length_na_replacement_values()[feature.internal_idx]
                .numerical_value);
        break;
      case ColumnType::CATEGORICAL:
        absl::SubstituteAndAppend(
            &structure, "($0)\n",
            model.features()
                .fixed_length_na_replacement_values()[feature.internal_idx]
                .categorical_value);
        break;
      default:
        absl::StrAppend(&structure, "\n");
        break;
    }
  }
  for (const auto& feature : model.features().categorical_set_features()) {
    absl::SubstituteAndAppend(&structure, "\t$0 [CATEGORICAL_SET] (none)\n",
                              feature.name);
  }
  absl::StrAppend(&structure, "\n");

  // Leafs.
  absl::SubstituteAndAppend(&structure, "Output leaf values ($0):\n",
                            model.leaf_values.size());
  if (detailed) {
    for (const auto& leaf_value : model.leaf_values) {
      absl::SubstituteAndAppend(&structure, " $0", leaf_value);
    }
    absl::StrAppend(&structure, "\n\n");
  }

  // Condition "contains" for categorical features.
  absl::SubstituteAndAppend(&structure,
                            "Conditions [categorical contains] ($0):\n",
                            model.categorical_contains_conditions.size());
  for (const auto& item : model.categorical_contains_conditions) {
    absl::SubstituteAndAppend(
        &structure, "\tfeature: $0 ($1) (num=$2)\n", item.internal_feature_idx,
        model.features()
            .fixed_length_features()[item.internal_feature_idx]
            .name,
        item.items.size());
    if (detailed) {
      for (int item_idx = 0; item_idx < item.items.size(); ++item_idx) {
        const auto bitmap_representation = ToStringBit(
            std::string(
                reinterpret_cast<const char* const>(&item.items[item_idx]),
                sizeof(LeafMask)),
            internal::QuickScorerExtendedModel::kMaxLeafs);
        absl::SubstituteAndAppend(
            &structure, "\t\ttree:$0 value:$1 mask : $2\n",
            item_idx % model.num_trees, item_idx / model.num_trees,
            bitmap_representation);
      }
    }
  }
  absl::StrAppend(&structure, "\n");

  // Condition "contains" for Categorical Set features.
  absl::SubstituteAndAppend(&structure,
                            "Conditions [categorical set contains] ($0):\n",
                            model.categoricalset_contains_conditions.size());
  for (const auto& item : model.categoricalset_contains_conditions) {
    absl::SubstituteAndAppend(
        &structure,
        "\tfeature: $0 ($1) (ranges=$2 masks=$3 mask/range=$4/$5)\n",
        item.internal_feature_idx,
        model.features()
            .categorical_set_features()[item.internal_feature_idx]
            .name,
        item.value_to_mask_range.size(), item.mask_buffer.size(),
        static_cast<float>(item.mask_buffer.size()) /
            item.value_to_mask_range.size(),
        model.num_trees);
    if (detailed) {
      for (int value = 0; value < item.value_to_mask_range.size(); value++) {
        absl::SubstituteAndAppend(&structure, "\tValue: $0:\n", value);
        const auto& range = item.value_to_mask_range[value];
        for (int mask_idx = range.first; mask_idx < range.second; mask_idx++) {
          const auto& mask = item.mask_buffer[mask_idx];
          const auto bitmap_representation = ToStringBit(
              std::string(reinterpret_cast<const char* const>(&mask.second),
                          sizeof(LeafMask)),
              internal::QuickScorerExtendedModel::kMaxLeafs);
          absl::SubstituteAndAppend(&structure, "\t\ttree:$0 mask : $1\n",
                                    mask.first, bitmap_representation);
        }
      }
    }
  }
  absl::StrAppend(&structure, "\n");

  // Conditions "is higher".
  absl::SubstituteAndAppend(&structure, "Conditions [is_higher] ($0):\n",
                            model.is_higher_conditions.size());
  for (const auto& item : model.is_higher_conditions) {
    int num_duplicates = 0;
    for (int sub_item_idx = 0; sub_item_idx < item.items.size() - 1;
         ++sub_item_idx) {
      if (item.items[sub_item_idx].threshold ==
          item.items[sub_item_idx + 1].threshold) {
        ++num_duplicates;
      }
    }
    float duplicate_ratio = -1.f;
    if (!item.items.empty()) {
      duplicate_ratio = static_cast<float>(num_duplicates) / item.items.size();
    }

    absl::SubstituteAndAppend(
        &structure, "\tfeature: $0 ($1) (num=$2; duplicate=$3)\n",
        item.internal_feature_idx,
        model.features()
            .fixed_length_features()[item.internal_feature_idx]
            .name,
        item.items.size(), duplicate_ratio);
    if (detailed) {
      for (const auto& sub_item : item.items) {
        const auto bitmap_representation = ToStringBit(
            std::string(
                reinterpret_cast<const char* const>(&sub_item.leaf_mask),
                sizeof(LeafMask)),
            internal::QuickScorerExtendedModel::kMaxLeafs);
        absl::SubstituteAndAppend(&structure,
                                  "\t\tmask:$0 = $1 threshold:$2 tree:$3\n",
                                  sub_item.leaf_mask, bitmap_representation,
                                  sub_item.threshold, sub_item.tree_idx);
      }

      for (const auto& sub_item : item.missing_value_items) {
        const auto bitmap_representation = ToStringBit(
            std::string(
                reinterpret_cast<const char* const>(&sub_item.leaf_mask),
                sizeof(LeafMask)),
            internal::QuickScorerExtendedModel::kMaxLeafs);
        absl::SubstituteAndAppend(
            &structure, "\t\tmask:$0 = $1 threshold:MISSING tree:$2\n",
            sub_item.leaf_mask, bitmap_representation, sub_item.tree_idx);
      }
    }
  }
  absl::StrAppend(&structure, "\n");

  return structure;
}

template std::string
DescribeQuickScorer<GradientBoostedTreesRegressionQuickScorerExtended>(
    const GradientBoostedTreesRegressionQuickScorerExtended& model,
    bool detailed);

template std::string DescribeQuickScorer<
    GradientBoostedTreesBinaryClassificationQuickScorerExtended>(
    const GradientBoostedTreesBinaryClassificationQuickScorerExtended& model,
    bool detailed);

template std::string
DescribeQuickScorer<GradientBoostedTreesRegressionQuickScorerExtendedHighway>(
    const GradientBoostedTreesRegressionQuickScorerExtendedHighway& model,
    bool detailed);

template std::string DescribeQuickScorer<
    GradientBoostedTreesBinaryClassificationQuickScorerExtendedHighway>(
    const GradientBoostedTreesBinaryClassificationQuickScorerExtendedHighway&
        model,
    bool detailed);

namespace internal {

template <typename Item>
void MergeAdjacent(const std::vector<Item>& src, std::vector<Item>* dst) {
  const size_t n = src.size();
  dst->clear();
  dst->reserve(n);

  auto it_begin = src.begin();
  while (it_begin != src.end()) {
    // Iterate and merge the elements equivalent to "it_begin".
    auto merged_item = *it_begin;
    auto it_end = it_begin + 1;
    while (it_end != src.end() && it_begin->CanMerge(*it_end)) {
      merged_item.leaf_mask &= it_end->leaf_mask;
      it_end++;
    }

    // Add the merged items [it_begin, it_end) to "dst".
    dst->push_back(merged_item);

    // Continue
    it_begin = it_end;
  }

  dst->shrink_to_fit();
}

void FinalizeConditionItems(
    std::vector<QuickScorerExtendedModel::ConditionItem>* items) {
  std::sort(items->begin(), items->end());
  const auto save = std::move(*items);
  MergeAdjacent(save, items);
}

void FinalizeIsHigherConditionItems(
    std::vector<QuickScorerExtendedModel::IsHigherConditionItem>* items) {
  std::sort(items->begin(), items->end());
  const auto save = std::move(*items);
  MergeAdjacent(save, items);
}

// Finalize the model. To be run once all the trees have been integrated to the
// quick scorer representation with the "FillQuickScorer" method.
absl::Status FinalizeModel(
    const internal::QuickScorerExtendedModel::BuildingAccumulator& accumulator,
    internal::QuickScorerExtendedModel* dst) {
  // Copy the conditions from the accumulator index to the optimized model.

  // For "is_higher" conditions.
  for (const auto& it_is_higher_condition : accumulator.is_higher_conditions) {
    dst->is_higher_conditions.push_back(it_is_higher_condition.second);
    auto& condition = dst->is_higher_conditions.back();
    FinalizeIsHigherConditionItems(&condition.items);
    FinalizeConditionItems(&condition.missing_value_items);
  }
  // Sort the condition by increasing feature index (for better locality when
  // querying the examples).
  std::sort(dst->is_higher_conditions.begin(), dst->is_higher_conditions.end(),
            [](const auto& a, const auto& b) {
              return a.internal_feature_idx < b.internal_feature_idx;
            });

  // For dense "contains" conditions.
  for (const auto& it_contains_condition :
       accumulator.categorical_contains_conditions) {
    dst->categorical_contains_conditions.push_back(
        it_contains_condition.second);
  }

  // For sparse "contains" conditions.
  for (const auto& it_contains_condition :
       accumulator.categoricalset_contains_conditions) {
    internal::QuickScorerExtendedModel::SparseContainsConditions condition;
    condition.internal_feature_idx =
        it_contains_condition.second.internal_feature_idx;
    const auto& src_masks = it_contains_condition.second.masks;
    condition.value_to_mask_range.reserve(src_masks.size());
    for (const auto& mask : src_masks) {
      condition.value_to_mask_range.emplace_back();
      condition.value_to_mask_range.back().first = condition.mask_buffer.size();
      for (const auto& tree_mask : mask) {
        if (tree_mask.second ==
            ~internal::QuickScorerExtendedModel::kZeroLeafMask) {
          continue;
        }
        condition.mask_buffer.push_back(tree_mask);
      }
      condition.value_to_mask_range.back().second =
          condition.mask_buffer.size();
    }
    dst->categoricalset_contains_conditions.push_back(std::move(condition));
  }

  return absl::OkStatus();
}

}  // namespace internal
}  // namespace yggdrasil_decision_forests::serving::decision_forest
