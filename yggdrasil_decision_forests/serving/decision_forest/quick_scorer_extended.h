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

// QuickScorer is inference algorithm for decision trees.
// This implementation extends the QuickScorer algorithms to categorical and
// categorical-set features.
//
// The central idea is to run the model inference per feature and per condition,
// instead of running it per tree and per node. While more complex in
// appearance, the algorithm generates less CPU branch misses and should overall
// run faster on modern CPUs.
//
// At its code, the algorithm manages a bitmap over the model leaves (called
// "active leaf bitmap" in the code). Each condition is attached to a mask
// (called "mask" in the code) of the same size which is applied (conjunction)
// over the leaves if the condition is true. After all the condition have been
// applied, the "first" active leaf (i.e. the leaf corresponding to the first
// non-zero bitmap value) is returned.
//
// The SIMD instructions are used to process multiple examples at the sametime.
//
// Important: This library works faster if AVX2 is enabled at computation:
//   Add "--copt=-mavx2" to the build call.
//   Add "requirements = {constraints = cpu_features.require(['avx2'])}" to your
//     borgcfg.
//   Add a "tricorder > builder > copt: '-mavx2'", in your METADATA for your
//     forge tests.
//
// Note: Adding 'copts = ["-mavx2"],' to your binary configuration wont work as
// it will only be used to compile your main target.
//
// The current implementation supports Gradient Boosted Trees with the following
// constraints:
//   - Maximum of 65k trees.
//   - Maximum of 128 nodes per trees (e.g. max depth = 6).
//   - Maximum of 65k unique input features.
//   - No oblique splits.
//
// Unlike the other YDF optimized inference engines, categorical features are
// not restricted to have a maximum of 32 unique values.
//
// The algorithm are described in the following papers:
//
// Original paper:
//   http://ecmlpkdd2017.ijs.si/papers/paperID718.pdf
// Extension with SMID instructions:
//   http://pages.di.unipi.it/rossano/wp-content/uploads/sites/7/2016/07/SIGIR16a.pdf
// Categorical-Set Extension:
//   https://arxiv.org/abs/2009.09991
//
//
// In the code "feature_idx" refers to features indexed according to the
// "dataspec" (i.e. the training dataset). "internal_feature_idx" refers to a
// internal feature indices from the point of view of the model (i.e. where the
// features used by the model are densely packaged).
//
// QuickScorer has two implementations:
// 1. Legacy: Hand-written AVX2 intrinsics.
// 2. Highway: Uses the Highway library (https://github.com/google/highway).
//
// The Highway implementation is benchmarked to be at least as fast as the
// legacy AVX2 implementation and is recommended. The AVX2 implementation will
// be removed in the future.
//
// By default, the Highway implementation uses static dispatch, meaning the
// SIMD intrinsics are selected at compile time. To enable dynamic dispatch
// (i.e., selecting the best intrinsics available on the current CPU at
// runtime), define the preprocessor flag YDF_USE_DYNAMIC_DISPATCH.

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_QUICK_SCORER_EXTENDED_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_QUICK_SCORER_EXTENDED_H_

#include <stdlib.h>

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::decision_forest {

namespace internal {

// Base model representation compatible with the QuickScorer algorithm.
struct QuickScorerExtendedModel {
  using ExampleSet =
      ExampleSetNumericalOrCategoricalFlat<QuickScorerExtendedModel,
                                           ExampleFormat::FORMAT_FEATURE_MAJOR>;
  // Backward compatibility.
  using ValueType = NumericalOrCategoricalValue;

  // Definition of the input features of the model, and how they are represented
  // in an input example set.
  const ExampleSet::FeaturesDefinition& features() const {
    return intern_features;
  }

  ExampleSet::FeaturesDefinition* mutable_features() {
    return &intern_features;
  }

  ExampleSet::FeaturesDefinition intern_features;

  // Note: The following four fields will be integrated as template parameters
  // in a future cl.
  // Index of a tree in the model. Limits the number of trees of the model.
  using TreeIdx = uint32_t;
  // Bitmap over the leafs in a tree. Limits the number of leafs in a tree.
  using LeafMask = uint64_t;
  // The value of a leaf.
  using LeafOutput = float;

  // Helper values.
  static constexpr LeafMask kZeroLeafMask = static_cast<LeafMask>(0);
  static constexpr LeafMask kOneLeafMask = static_cast<LeafMask>(1);

  // Maximum number of trees and number of leafs per trees.
  static constexpr size_t kMaxTrees = std::numeric_limits<TreeIdx>::max();
  static constexpr size_t kMaxLeafs = sizeof(LeafMask) * 8;

  // If true, the engine inference runs with the global imputation optimization.
  // That is, missing values are replaced with global imputation.
  bool global_imputation_optimization;

  // Maximum number of leafs in each tree.
  int max_num_leafs_per_tree;

  // Value (i.e. prediction) of each leaf.
  // "leaf_values[i + j * max_num_leafs_per_tree]" is the value of the "i-th"
  // leaf in the "j-th" tree.
  std::vector<LeafOutput> leaf_values;

  // Number of trees in the model.
  int num_trees;

  // Initial prediction / bias of the model.
  float initial_prediction = 0.f;

  // If true, do not apply the activation function of the model (if any).
  bool output_logits = false;

  // Support for N/A conditions has not been implemented.
  static constexpr bool uses_na_conditions = false;

#ifdef __AVX2__
  // This flag is set during the compilation of the model and indicates if the
  // CPU supports AVX2 instructions
  bool cpu_supports_avx2 = true;
#endif

  struct ConditionItem {
    TreeIdx tree_idx;
    LeafMask leaf_mask;

    bool operator<(const ConditionItem& e) const {
      return tree_idx < e.tree_idx;
    }

    // Indicates that two items can be merged without impact on the inference
    // logic.
    bool CanMerge(const ConditionItem& e) const {
      return tree_idx == e.tree_idx;
    }
  };

  // Data for "IsHigher" conditions i.e. condition of the form "feature >= t".
  struct IsHigherConditionItem {
    float threshold;
    TreeIdx tree_idx;
    LeafMask leaf_mask;

    bool operator<(const IsHigherConditionItem& e) const {
      if (threshold != e.threshold) {
        return threshold < e.threshold;
      }
      return tree_idx < e.tree_idx;
    }

    // Indicates that two items can be merged without impact on the inference
    // logic.
    bool CanMerge(const IsHigherConditionItem& e) const {
      return tree_idx == e.tree_idx && threshold == e.threshold;
    }
  };

  struct IsHigherConditions {
    // Index of the feature in "model.features".
    // See the definition of "internal_feature_idx" in the head comment.
    int internal_feature_idx;

    // Thresholds ordered in ascending order.
    std::vector<IsHigherConditionItem> items;

    // Items to consider in the case of a missing value.
    std::vector<ConditionItem> missing_value_items;
  };

  // Data for "Contains" conditions i.e. condition of the form "feature \in
  // set".
  struct ContainsConditions {
    // Internal index of the feature.
    int internal_feature_idx;

    // "Contains" type condition for each feature value.
    // items[tree_idx + feature_value * num_trees] is the mask to apply on tree
    // "tree_idx" when the feature value is "feature_value".
    std::vector<LeafMask> items;
  };

  // Similar to "ContainsConditions", but only index the trees impacted by each
  // feature value.
  struct SparseContainsConditions {
    // Internal index of the feature.
    int internal_feature_idx;

    // The "i-th" feature value maps to the masks "mask_buffer[j]" for "j" in
    // "[value_to_mask_range[i).first, value_to_mask_range[i].second[".
    std::vector<std::pair<int, int>> value_to_mask_range;
    std::vector<std::pair<TreeIdx, LeafMask>> mask_buffer;
  };

  std::vector<IsHigherConditions> is_higher_conditions;
  std::vector<ContainsConditions> categorical_contains_conditions;
  std::vector<SparseContainsConditions> categoricalset_contains_conditions;

  // Structure used during the compilation of the model and discarded at the
  // end.
  struct BuildingAccumulator {
    struct SparseContainsConditions {
      // Internal index of the feature.
      int internal_feature_idx;

      // "masks[i][j]" is the mask for the "i-th" feature value on the "j-th"
      // tree;
      std::vector<std::unordered_map<TreeIdx, LeafMask>> masks;
    };

    // Similar to the fields of the same name above, but indexed by the dataspec
    // feature index.
    //
    // Note: Absl hash map does not check at compile time the availability of
    // AVX2.
    std::unordered_map<int, IsHigherConditions> is_higher_conditions;
    std::unordered_map<int, ContainsConditions> categorical_contains_conditions;
    std::unordered_map<int, SparseContainsConditions>
        categoricalset_contains_conditions;
  };

  model::proto::Metadata metadata;
};

// ANDs a "mask" on a value contained in a map (specified by a key) i.e.
// "map[key] &= mask". If the map does not contain the key, set it to the
// "mask" value i.e. "map[key] = mask".
template <typename Map>
void AndMaskMap(const typename Map::key_type& key,
                QuickScorerExtendedModel::LeafMask mask, Map* map) {
  const auto insertion = map->insert({key, mask});
  if (!insertion.second) {
    insertion.first->second &= mask;
  }
}

// Finalize a set of condition from the "BuildingAccumulator" into the final
// model.
void FinalizeConditionItems(
    std::vector<QuickScorerExtendedModel::ConditionItem>* items);

void FinalizeIsHigherConditionItems(
    std::vector<QuickScorerExtendedModel::IsHigherConditionItem>* items);

// Maximum stack size used by the model during inference
constexpr size_t kMaxStackUsageInBytes = 16 * 1024;

// Returns the number of trailing 0-bits in x, starting at the least significant
// bit position. If x is 0, the result is undefined.
int FindLSBSetNonZero64(uint64_t n);

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
      case dataset::proto::ColumnType::CATEGORICAL: {
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

      case dataset::proto::ColumnType::NUMERICAL:
      case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL:
      case dataset::proto::ColumnType::BOOLEAN: {
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
    if (feature.type == dataset::proto::ColumnType::CATEGORICAL_SET) {
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
    internal::QuickScorerExtendedModel* dst);

// Adds the content of a node (and its children i.e. recursive visit) to the
// quick scorer tree structure.
template <typename AbstractModel>
absl::Status FillQuickScorerNode(
    const AbstractModel& src,
    const internal::QuickScorerExtendedModel::TreeIdx tree_idx,
    const model::decision_tree::NodeWithChildren& src_node,
    internal::QuickScorerExtendedModel* dst, int* leaf_idx, int* non_leaf_idx,
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

      if (src_node.node().condition().na_value()) {
        // The condition evaluates to true when the attribute is missing.
        accumulator->is_higher_conditions[spec_feature_idx]
            .missing_value_items.push_back({/*.tree_idx =*/tree_idx,
                                            /*.leaf_mask =*/mask});
      }
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
      case model::decision_tree::proto::Condition::TypeCase::kHigherCondition:
        DCHECK_EQ(attribute_spec.type(), dataset::proto::ColumnType::NUMERICAL);
        set_numerical_higher();
        break;

      case model::decision_tree::proto::Condition::TypeCase::
          kDiscretizedHigherCondition:
        DCHECK_EQ(attribute_spec.type(),
                  dataset::proto::ColumnType::DISCRETIZED_NUMERICAL);
        set_discretized_numerical_higher();
        break;

      case model::decision_tree::proto::Condition::TypeCase::
          kTrueValueCondition:
        DCHECK_EQ(attribute_spec.type(), dataset::proto::ColumnType::BOOLEAN);
        set_boolean_is_true();
        break;

      case model::decision_tree::proto::Condition::TypeCase::kContainsCondition:
        if (attribute_spec.type() == dataset::proto::ColumnType::CATEGORICAL) {
          set_categorical_contains();
        } else if (attribute_spec.type() ==
                   dataset::proto::ColumnType::CATEGORICAL_SET) {
          set_categoricalset_contains();
        } else {
          return absl::InternalError("Unexpected type");
        }
        break;

      case model::decision_tree::proto::Condition::TypeCase::
          kContainsBitmapCondition:
        if (attribute_spec.type() == dataset::proto::ColumnType::CATEGORICAL) {
          set_categorical_bitmap_contains();
        } else if (attribute_spec.type() ==
                   dataset::proto::ColumnType::CATEGORICAL_SET) {
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
  RETURN_IF_ERROR(internal::InitializeAccumulator(src, *dst, accumulator));

  dst->initial_prediction = src.initial_predictions()[0];
  dst->output_logits = src.output_logits();
  dst->num_trees = src.NumTrees();
  if (dst->num_trees > internal::QuickScorerExtendedModel::kMaxTrees) {
    return absl::InvalidArgumentError(
        absl::Substitute("The model contains more than $0 trees",
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
    RETURN_IF_ERROR(
        internal::FillQuickScorerNode(src, tree_idx, src_tree->root(), dst,
                                      &leaf_idx, &non_leaf_idx, accumulator));
  }

  RETURN_IF_ERROR(internal::FinalizeModel(*accumulator, dst));
  return absl::OkStatus();
}

}  // namespace internal

// Legacy implementation of the QuickScorer algorithm.
struct QuickScorerExtendedModelLegacy : internal::QuickScorerExtendedModel {};

// Highway implementation of the QuickScorer algorithm.
struct QuickScorerExtendedModelHighway : internal::QuickScorerExtendedModel {};

// Specialization of quick scorer for GBDT regression model with MSE loss.
struct GradientBoostedTreesRegressionQuickScorerExtended
    : QuickScorerExtendedModelLegacy {
  static constexpr model::proto::Task kTask = model::proto::Task::REGRESSION;
};

// Specialization of quick scorer for GBDT regression model with poisson loss.
struct GradientBoostedTreesPoissonRegressionQuickScorerExtended
    : QuickScorerExtendedModelLegacy {
  static constexpr model::proto::Task kTask = model::proto::Task::REGRESSION;
};

// Specialization of quick scorer for GBDT binary classification model.
struct GradientBoostedTreesBinaryClassificationQuickScorerExtended
    : QuickScorerExtendedModelLegacy {
  static constexpr model::proto::Task kTask =
      model::proto::Task::CLASSIFICATION;
};

// Specialization of quick scorer for GBDT ranking model.
struct GradientBoostedTreesRankingQuickScorerExtended
    : QuickScorerExtendedModelLegacy {
  static constexpr model::proto::Task kTask = model::proto::Task::RANKING;
};

// Specialization of quick scorer for GBDT regression model with MSE loss.
struct GradientBoostedTreesRegressionQuickScorerExtendedHighway
    : QuickScorerExtendedModelHighway {
  static constexpr model::proto::Task kTask = model::proto::Task::REGRESSION;
};

// Specialization of quick scorer for GBDT regression model with poisson loss.
struct GradientBoostedTreesPoissonRegressionQuickScorerExtendedHighway
    : QuickScorerExtendedModelHighway {
  static constexpr model::proto::Task kTask = model::proto::Task::REGRESSION;
};

// Specialization of quick scorer for GBDT binary classification model.
struct GradientBoostedTreesBinaryClassificationQuickScorerExtendedHighway
    : QuickScorerExtendedModelHighway {
  static constexpr model::proto::Task kTask =
      model::proto::Task::CLASSIFICATION;
};

// Specialization of quick scorer for GBDT ranking model.
struct GradientBoostedTreesRankingQuickScorerExtendedHighway
    : QuickScorerExtendedModelHighway {
  static constexpr model::proto::Task kTask = model::proto::Task::RANKING;
};

// Computes the model's prediction on a batch of examples.
//
// This method is thread safe.
//
// This method uses a significant amount of stack size. See
// "GenericToSpecializedModel" for more details.
//
// Args:
//   - model: A quick scorer model (e.g.
//     GradientBoostedTreesRegressionQuickScorer) initialized with
//     "GenericToSpecializedModel".
//   - examples: A batch of examples. The examples are stored FEATURE-WISE.
//   - num_examples: Number of examples in the batch.
//   - predictions: Output predictions. Does not need to be pre-allocated.
//

template <typename Model>
void PredictQuickScorer(
    const Model& model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions);

// Version of PredictQuickScorer compatible with the ExampleSet signature.
template <typename Model>
void Predict(const Model& model, const typename Model::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions);

// Converts a generic GradientBoostedTreesModel with regression loss into a
// quick scorer compatible model.
//
// This method checks that the model inference (i.e. PredictQuickScorer) won't
// take more than 16kb of stack size. The stack size usage is
// defined by the number of trees and number of leafs of the model. For
// reference, the ML AOI model is taking 1.5kb of stack size. If your model is
// too large, contact us (gbm@) for the heap version of this method (~10%
// slower) or use the inference code in "decision_forest.h" (>2x slower, no
// model limit).
template <typename AbstractModel, typename CompiledModel>
absl::Status GenericToSpecializedModel(const AbstractModel& src,
                                       CompiledModel* dst);

// Creates an empty model that returns a constant value (e.g. 0 for regression)
// but which consumes (and ignores) the input features specified at
// construction.
//
// This function can be used to create fake models to unit test the generation
// of ExampleSets.
template <typename CompiledModel>
absl::Status CreateEmptyModel(const std::vector<int>& input_features,
                              const dataset::proto::DataSpecification& dataspec,
                              CompiledModel* dst);

// Generates a human readable text describing the internal of the quick scorer
// model.
//
// This description is intended for debugging or optimization purpose. For a ML
// development intended description of the model, use the "describe" method on
// the non-compiled model.
template <typename Model>
std::string DescribeQuickScorer(const Model& model, bool detailed = true);

}  // namespace yggdrasil_decision_forests::serving::decision_forest

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_QUICK_SCORER_EXTENDED_H_
