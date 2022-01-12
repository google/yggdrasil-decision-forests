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

#include "yggdrasil_decision_forests/serving/decision_forest/quick_scorer_extended.h"

#include <stdlib.h>

#include "absl/status/status.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "absl/base/config.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/usage.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace decision_forest {

using dataset::proto::ColumnType;
using LeafMask = internal::QuickScorerExtendedModel::LeafMask;
using model::decision_tree::NodeWithChildren;
using model::decision_tree::proto::Condition;
using model::gradient_boosted_trees::proto::Loss;
using ::yggdrasil_decision_forests::utils::bitmap::ToStringBit;

namespace {

// Maximum stack size used by the model during inference
constexpr size_t kMaxStackUsageInBytes = 16 * 1024;

namespace portable {
#ifdef __AVX2__
void* aligned_alloc(std::size_t alignment, std::size_t size) {
#if defined(_WIN32)
  // Visual Studio
  return _aligned_malloc(/*size=*/size, /*alignment=*/alignment);
#else
  return ::aligned_alloc(/*alignment=*/alignment, /*size=*/size);
#endif
}

void aligned_free(void* mem) {
#if defined(_WIN32)
  _aligned_free(mem);
#else
  free(mem);
#endif
}
#endif
}  // namespace portable

// Returns the number of trailing 0-bits in x, starting at the least significant
// bit position. If x is 0, the result is undefined.
int FindLSBSetNonZero64(uint64_t n) {
  return utils::CountTrailingZeroesNonzero64(n);
}

// Activation function for binary classification GBDT trained with Binomial
// LogLikelihood loss.
float ActivationBinomialLogLikelihood(const float value) {
  return utils::clamp(1.f / (1.f + std::exp(-value)), 0.f, 1.f);
}

// Identity activation function.
float ActivationIdentity(const float value) { return value; }

// Initialize the accumulator used to construct the quick scorer model
// representation.
//
// Note: This accumulator is discarded at the end of the model generation.
template <typename AbstractModel>
absl::Status InitializeAccumulator(
    const AbstractModel& src, const internal::QuickScorerExtendedModel& dst,
    internal::QuickScorerExtendedModel::BuildingAccumulator* accumulator) {
  for (const auto& feature : dst.features().fixed_length_features()) {
    const auto& feature_spec = src.data_spec().columns(feature.spec_idx);

    switch (feature.type) {
      case ColumnType::CATEGORICAL: {
        // Note: Initially, the bitmap is initially filled with 1s i.e. no leaf
        // is filtered.
        auto& feature_acc =
            accumulator->categorical_contains_conditions[feature.spec_idx];
        feature_acc.internal_feature_idx = feature.internal_idx;
        feature_acc.items.assign(
            src.NumTrees() *
                feature_spec.categorical().number_of_unique_values(),
            ~internal::QuickScorerExtendedModel::kZeroLeafMask);
      } break;

      case ColumnType::NUMERICAL:
      case ColumnType::DISCRETIZED_NUMERICAL:
      case ColumnType::BOOLEAN: {
        // Note: Initially, the bitmap is initially filled with 1s i.e. no leaf
        // is filtered.
        auto& feature_acc = accumulator->is_higher_conditions[feature.spec_idx];
        feature_acc.internal_feature_idx = feature.internal_idx;
      } break;

      default:
        return absl::InternalError("Unexpected feature type");
    }
  }

  for (const auto& feature : dst.features().categorical_set_features()) {
    const auto& feature_spec = src.data_spec().columns(feature.spec_idx);
    if (feature.type == ColumnType::CATEGORICAL_SET) {
      auto& feature_acc =
          accumulator->categoricalset_contains_conditions[feature.spec_idx];
      feature_acc.internal_feature_idx = feature.internal_idx;
      feature_acc.masks.resize(
          feature_spec.categorical().number_of_unique_values() + 1);
    } else {
      return absl::InternalError("Unexpected feature type");
    }
  }

  return absl::OkStatus();
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
    // Sort in increasing threshold value.
    std::sort(
        condition.items.begin(), condition.items.end(),
        [](const auto& a, const auto& b) { return a.threshold < b.threshold; });
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

// Adds the content of a node (and its children i.e. recursive visit) to the
// quick scorer tree structure.
template <typename AbstractModel>
absl::Status FillQuickScorerNode(
    const AbstractModel& src,
    const internal::QuickScorerExtendedModel::TreeIdx tree_idx,
    const NodeWithChildren& src_node, internal::QuickScorerExtendedModel* dst,
    int* leaf_idx, int* non_leaf_idx,
    internal::QuickScorerExtendedModel::BuildingAccumulator* accumulator) {
  if (src_node.IsLeaf()) {
    // Store the lead value.
    if (*leaf_idx >= internal::QuickScorerExtendedModel::kMaxLeafs) {
      return absl::InternalError("Leaf idx too large");
    }
    if (*leaf_idx >= dst->max_num_leafs_per_tree) {
      return absl::InternalError("Leaf idx too large");
    }
    const auto leaf_value_idx =
        *leaf_idx + tree_idx * dst->max_num_leafs_per_tree;
    if (leaf_value_idx >= dst->leaf_values.size()) {
      return absl::InternalError("Leaf value idx too large");
    }
    dst->leaf_values[leaf_value_idx] = src_node.node().regressor().top_value();
    (*leaf_idx)++;
  } else {
    // Index of the first leaf in the negative branch.
    const auto begin_neg_leaf_idx = *leaf_idx;

    // Parse the negative branch.
    RETURN_IF_ERROR(FillQuickScorerNode(src, tree_idx, *src_node.neg_child(),
                                        dst, leaf_idx, non_leaf_idx,
                                        accumulator));

    // Index of the feature used by the node.
    const int spec_feature_idx = src_node.node().condition().attribute();

    // Compute the bitmap mask i.e. the bitmap that hide the leafs of the
    // negative branch.
    //
    // Example:
    // If begin_neg_leaf_idx=2 and end_neg_leaf_idx = 5, the mask will be:
    //   "1100011111" + 54 * "1" (lower bit on the left).
    const auto end_neg_leaf_idx = *leaf_idx;
    const auto start_leaf_mask =
        (internal::QuickScorerExtendedModel::kOneLeafMask
         << begin_neg_leaf_idx) -
        1;
    const auto after_neg_mask =
        (internal::QuickScorerExtendedModel::kOneLeafMask << end_neg_leaf_idx) -
        1;
    internal::QuickScorerExtendedModel::LeafMask mask =
        ~(after_neg_mask ^ start_leaf_mask);

    const auto& condition = src_node.node().condition().condition();
    // Branch to take is case of missing value. Can be ignored in the case of
    // numerical and categorical features as the use "feature_missing_values"
    // produce an equivalent (but more efficient) behavior.
    const bool na_value = src_node.node().condition().na_value();
    const auto& attribute_spec =
        src.data_spec().columns(src_node.node().condition().attribute());

    auto set_numerical_higher = [&]() {
      const auto threshold = condition.higher_condition().threshold();
      accumulator->is_higher_conditions[spec_feature_idx].items.push_back(
          {/*.threshold =*/threshold, /*.tree_idx =*/tree_idx,
           /*.leaf_mask =*/mask});
    };

    auto set_boolean_is_true = [&]() {
      accumulator->is_higher_conditions[spec_feature_idx].items.push_back(
          {/*.threshold =*/0.5f, /*.tree_idx =*/tree_idx,
           /*.leaf_mask =*/mask});
    };

    auto set_discretized_numerical_higher = [&]() {
      const auto discretized_threshold =
          condition.discretized_higher_condition().threshold();
      const float threshold = attribute_spec.discretized_numerical().boundaries(
          discretized_threshold - 1);
      accumulator->is_higher_conditions[spec_feature_idx].items.push_back(
          {/*.threshold = */ threshold, /*.tree_idx =*/tree_idx,
           /*.leaf_mask =*/mask});
    };

    auto set_categorical_contains = [&]() {
      const auto elements = condition.contains_condition().elements();
      for (const auto feature_value : elements) {
        accumulator->categorical_contains_conditions[spec_feature_idx]
            .items[tree_idx + feature_value * dst->num_trees] &= mask;
      }
    };

    auto set_categorical_bitmap_contains = [&]() {
      const auto bitmap =
          condition.contains_bitmap_condition().elements_bitmap();
      const int num_unique_values =
          attribute_spec.categorical().number_of_unique_values();
      for (int feature_value = 0; feature_value < num_unique_values;
           ++feature_value) {
        if (utils::bitmap::GetValueBit(bitmap, feature_value)) {
          accumulator->categorical_contains_conditions[spec_feature_idx]
              .items[tree_idx + feature_value * dst->num_trees] &= mask;
        }
      }
    };

    auto set_categoricalset_contains = [&]() {
      const auto elements = condition.contains_condition().elements();
      if (na_value) {
        internal::AndMaskMap(
            tree_idx, mask,
            &accumulator->categoricalset_contains_conditions[spec_feature_idx]
                 .masks[0]);
      }
      for (const auto feature_value : elements) {
        internal::AndMaskMap(
            tree_idx, mask,
            &accumulator->categoricalset_contains_conditions[spec_feature_idx]
                 .masks[feature_value + 1]);
      }
    };

    auto set_categoricalset_bitmap_contains = [&]() {
      if (na_value) {
        internal::AndMaskMap(
            tree_idx, mask,
            &accumulator->categoricalset_contains_conditions[spec_feature_idx]
                 .masks[0]);
      }
      const auto bitmap =
          condition.contains_bitmap_condition().elements_bitmap();
      const int num_unique_values =
          attribute_spec.categorical().number_of_unique_values();
      for (int feature_value = 0; feature_value < num_unique_values;
           ++feature_value) {
        if (utils::bitmap::GetValueBit(bitmap, feature_value)) {
          internal::AndMaskMap(
              tree_idx, mask,
              &accumulator->categoricalset_contains_conditions[spec_feature_idx]
                   .masks[feature_value + 1]);
        }
      }
    };

    // Process the node's condition.
    switch (condition.type_case()) {
      case Condition::TypeCase::kHigherCondition:
        DCHECK_EQ(attribute_spec.type(), ColumnType::NUMERICAL);
        set_numerical_higher();
        break;

      case Condition::TypeCase::kDiscretizedHigherCondition:
        DCHECK_EQ(attribute_spec.type(), ColumnType::DISCRETIZED_NUMERICAL);
        set_discretized_numerical_higher();
        break;

      case Condition::TypeCase::kTrueValueCondition:
        DCHECK_EQ(attribute_spec.type(), ColumnType::BOOLEAN);
        set_boolean_is_true();
        break;

      case Condition::TypeCase::kContainsCondition:
        if (attribute_spec.type() == ColumnType::CATEGORICAL) {
          set_categorical_contains();
        } else if (attribute_spec.type() == ColumnType::CATEGORICAL_SET) {
          set_categoricalset_contains();
        } else {
          return absl::InternalError("Unexpected type");
        }
        break;

      case Condition::TypeCase::kContainsBitmapCondition:
        if (attribute_spec.type() == ColumnType::CATEGORICAL) {
          set_categorical_bitmap_contains();
        } else if (attribute_spec.type() == ColumnType::CATEGORICAL_SET) {
          set_categoricalset_bitmap_contains();
        } else {
          return absl::InternalError("Unexpected type");
        }
        break;

      default:
        return absl::InvalidArgumentError("Unsupported condition type.");
    }

    ++(*non_leaf_idx);

    RETURN_IF_ERROR(FillQuickScorerNode(src, tree_idx, *src_node.pos_child(),
                                        dst, leaf_idx, non_leaf_idx,
                                        accumulator));
  }
  return absl::OkStatus();
}

// Adds the content of the tree structures to the quick scorer structure.
template <typename AbstractModel>
absl::Status FillQuickScorer(
    const AbstractModel& src, internal::QuickScorerExtendedModel* dst,
    internal::QuickScorerExtendedModel::BuildingAccumulator* accumulator) {
  RETURN_IF_ERROR(InitializeAccumulator(src, *dst, accumulator));

  dst->initial_prediction = src.initial_predictions()[0];
  dst->output_logits = src.output_logits();
  dst->num_trees = src.NumTrees();
  if (dst->num_trees > internal::QuickScorerExtendedModel::kMaxTrees) {
    return absl::InvalidArgumentError(
        absl::Substitute("The model contains trees with more than $0 trees",
                         internal::QuickScorerExtendedModel::kMaxTrees));
  }

  // Get the maximum number of leafs per trees.
  dst->max_num_leafs_per_tree = 0;
  int num_leafs = 0;
  for (const auto& src_tree : src.decision_trees()) {
    const auto num_leafs_in_tree = src_tree->NumLeafs();
    num_leafs += num_leafs_in_tree;
    if (num_leafs_in_tree > dst->max_num_leafs_per_tree) {
      dst->max_num_leafs_per_tree = num_leafs_in_tree;
    }
  }

  if (dst->max_num_leafs_per_tree >
      internal::QuickScorerExtendedModel::kMaxLeafs) {
    return absl::InvalidArgumentError(
        absl::Substitute("The model contains trees with more than $0 leafs",
                         internal::QuickScorerExtendedModel::kMaxLeafs));
  }

  dst->leaf_values.assign(dst->max_num_leafs_per_tree * dst->num_trees, 0.f);

  for (internal::QuickScorerExtendedModel::TreeIdx tree_idx = 0;
       tree_idx < src.decision_trees().size(); ++tree_idx) {
    const auto& src_tree = src.decision_trees()[tree_idx];
    int leaf_idx = 0;
    int non_leaf_idx = 0;
    RETURN_IF_ERROR(FillQuickScorerNode(src, tree_idx, src_tree->root(), dst,
                                        &leaf_idx, &non_leaf_idx, accumulator));
  }

  RETURN_IF_ERROR(FinalizeModel(*accumulator, dst));
  return absl::OkStatus();
}

// Tree inference without SIMD i.e. one example at a time.
// This method is used for the examples outside of the SIMD batch.
//
// "active_leaf_buffer" is a pre-allocated buffer of at least "num-trees"
// elements.
template <typename Model, float (*Activation)(float)>
void PredictQuickScorerSequential(
    const Model& model,
    const std::vector<NumericalOrCategoricalValue>& fixed_length_features,
    const std::vector<Rangei32>& categorical_set_begins_and_ends,
    const std::vector<int32_t>& categorical_item_buffer,
    const int begin_example_idx, const int end_example_idx,
    const int major_feature_offset, std::vector<float>* predictions,
    internal::QuickScorerExtendedModel::LeafMask* active_leaf_buffer) {
  const size_t active_leaf_buffer_size = model.num_trees * sizeof(LeafMask);

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

      for (const auto& item : is_higher_condition.items) {
        if (item.threshold > feature_value) {
          break;
        }
        active_leaf_buffer[item.tree_idx] &= item.leaf_mask;
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
      const auto node_idx = FindLSBSetNonZero64(shift_mask);
      output += leaf_reader[node_idx];
      leaf_reader += model.max_num_leafs_per_tree;
    }

    (*predictions)[example_idx] = Activation(output);
  }
}

}  // namespace

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
template <typename Model, float (*Activation)(float) = ActivationIdentity>
void PredictQuickScorerMajorFeatureOffset(
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
      active_leaf_buffer_size <= kMaxStackUsageInBytes;

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
        portable::aligned_alloc(alignment, rounded_up_active_leaf_buffer_size));
#else
    active_leaf_buffer = reinterpret_cast<LeafMask*>(
        std::malloc(rounded_up_active_leaf_buffer_size));
#endif
  }

  int example_idx = 0;

#ifdef __AVX2__
  if (model.cpu_supports_avx2) {
    auto* sample_reader = fixed_length_features.data();
    auto* prediction_reader = predictions->data();

    // First run on sub-batches of kNumParallelExamples at a time. The
    // remaining will be done sequentially below.
    int num_remaining_iters = num_examples / kNumParallelExamples;
    while (num_remaining_iters--) {
      // Reset active node buffer.
      std::memset(active_leaf_buffer, 0xFF, active_leaf_buffer_size);

      // Is higher conditions.
      for (const auto& is_higher_condition : model.is_higher_conditions) {
        const float* begin_example =
            &sample_reader[0].numerical_value +
            is_higher_condition.internal_feature_idx * major_feature_offset;

        const auto feature_values = _mm_loadu_ps(begin_example);
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
          const auto node_idx = FindLSBSetNonZero64(shift_mask);
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
#endif

  PredictQuickScorerSequential<Model, Activation>(
      model, fixed_length_features, categorical_set_begins_and_ends,
      categorical_item_buffer, example_idx, num_examples, major_feature_offset,
      predictions, active_leaf_buffer);

  if (!active_leaf_buffer_uses_stack) {
#ifdef __AVX2__
    portable::aligned_free(active_leaf_buffer);
#else
    free(active_leaf_buffer);
#endif
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
        ActivationBinomialLogLikelihood>(
        model, examples.InternalCategoricalAndNumericalValues(),
        examples.InternalCategoricalSetBeginAndEnds(),
        examples.InternalCategoricalItemBuffer(), num_examples,
        examples.NumberOfExamples(), predictions);
  }
}

template <typename AbstractModel, typename CompiledModel>
absl::Status BaseGenericToSpecializedModel(const AbstractModel& src,
                                           CompiledModel* dst) {
#ifdef __AVX2__
#if ABSL_HAVE_BUILTIN(__builtin_cpu_supports)
  dst->cpu_supports_avx2 = __builtin_cpu_supports("avx2");
#else
  // We cannot detect if the CPU supports AVX2 instructions. If it does not,
  // a fatal error will be raised.
  dst->cpu_supports_avx2 = true;
#endif
#elif ABSL_HAVE_BUILTIN(__builtin_cpu_supports)
  if (__builtin_cpu_supports("avx2")) {
    LOG_INFO_EVERY_N_SEC(
        30, _ << "The binary was compiled without AVX2 support, but your CPU "
                 "supports it. Enable it for faster model inference.");
  }
#endif

  if (src.task() != CompiledModel::kTask) {
    return absl::InvalidArgumentError("Wrong model class.");
  }

  src.metadata().Export(&dst->metadata);

  typename CompiledModel::BuildingAccumulator accumulator;

  // List the model input features.
  std::vector<int> all_input_features;
  RETURN_IF_ERROR(GetInputFeatures(src, &all_input_features, nullptr));

  RETURN_IF_ERROR(
      dst->mutable_features()->Initialize(all_input_features, src.data_spec()));

  // Compile the model.
  RETURN_IF_ERROR(FillQuickScorer(src, dst, &accumulator));

  return absl::OkStatus();
}

template <>
absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesRegressionQuickScorerExtended* dst) {
  if (src.loss() != Loss::SQUARED_ERROR) {
    return absl::InvalidArgumentError(
        "The GBDT is not trained for regression with squared error loss.");
  }
  return BaseGenericToSpecializedModel(src, dst);
}

template <>
absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesRankingQuickScorerExtended* dst) {
  if (src.loss() != Loss::LAMBDA_MART_NDCG5 &&
      src.loss() != Loss::XE_NDCG_MART) {
    return absl::InvalidArgumentError(
        "The GBDT is not trained for ranking with ranking loss.");
  }
  return BaseGenericToSpecializedModel(src, dst);
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
  return BaseGenericToSpecializedModel(src, dst);
}

template <typename CompiledModel>
absl::Status CreateEmptyModel(const std::vector<int>& input_features,
                              const DataSpecification& dataspec,
                              CompiledModel* dst) {
  return dst->mutable_features()->Initialize(input_features, dataspec);
}

template absl::Status
CreateEmptyModel<GradientBoostedTreesRegressionQuickScorerExtended>(
    const std::vector<int>& input_features, const DataSpecification& dataspec,
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
    int leaf_idx = 0;
    for (const auto& leaf_value : model.leaf_values) {
      absl::SubstituteAndAppend(&structure, " $0", leaf_value);
      ++leaf_idx;
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

  // Condition "contains" for categoricalset features.
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
                                  "\t\tmask:$0 = $1 thre:$2 tree:$3\n",
                                  sub_item.leaf_mask, bitmap_representation,
                                  sub_item.threshold, sub_item.tree_idx);
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

}  // namespace decision_forest
}  // namespace serving
}  // namespace yggdrasil_decision_forests
