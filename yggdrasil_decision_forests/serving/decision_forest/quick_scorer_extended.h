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
// The current implementation support:
//   - Regressive GBDTs.
//
// With the following constraints:
//   - Maximum of 65k trees.
//   - Maximum of 128 nodes per trees (e.g. max depth = 6).
//   - Maximum of 65k unique input features.
//   - Support categorical and numerical features.
//
// Unlike the other simpleML optimized inference engine, categorical are not
// restricted to have a maximum of 32 unique values.
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
// The code can be benchmarked using the following command:
//
//   bazel run -c opt --copt=-mavx2 :benchmark_nogpu -- \
//    --alsologtostderr \
//    --mtc=GRADIENT_BOOSTED_TREES_REGRESSION_NUMERICAL_AND_CATEGORICAL_32_ATTRIBUTES
//
// On 9.10.2019, this command returned the following results:
//
//   ...
//   Average serving time per example:
//   gbdt_regression_num_and_cat32_attributes_single_thread_cpu_quick_scorer :
//   647.5ns gbdt_regression_num_and_cat32_attributes_single_thread_cpu_opt_v1
//   : 1.3857us gbdt_regression_num_and_cat32_attributes_single_thread_cpu
//   : 1.4537us
//   ...
//
// In the code "feature_idx" refers to features indexed according to the
// "dataspec" (i.e. the training dataset). "internal_feature_idx" refers to a
// internal feature indices from the point of view of the model (i.e. where the
// features used by the model are densely packaged).
//
// See the "Intrinsics Guide" for a definition of the used SIMD instructions
// (https://software.intel.com/sites/landingpage/IntrinsicsGuide/).

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_QUICK_SCORER_EXTENDED_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_QUICK_SCORER_EXTENDED_H_

#include <stdlib.h>

#include <unordered_map>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/serving/decision_forest/utils.h"
#include "yggdrasil_decision_forests/serving/example_set.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace decision_forest {

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

#ifdef __AVX2__
  // This flag is set during the compilation of the model and indicates if the
  // CPU supports AVX2 instructions
  bool cpu_supports_avx2 = true;
#endif

  // Data for "IsHigher" conditions i.e. condition of the form "feature >= t".
  struct IsHigherConditionItem {
    float threshold;
    TreeIdx tree_idx;
    LeafMask leaf_mask;
  };

  struct IsHigherConditions {
    // Index of the feature in "model.features".
    // See the definition of "internal_feature_idx" in the head comment.
    int internal_feature_idx;

    // Thresholds ordered in ascending order.
    std::vector<IsHigherConditionItem> items;
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

}  // namespace internal

// Specialization of quick scorer for GBDT regression model.
struct GradientBoostedTreesRegressionQuickScorerExtended
    : internal::QuickScorerExtendedModel {
  static constexpr model::proto::Task kTask = model::proto::Task::REGRESSION;
};

// Specialization of quick scorer for GBDT binary classification model.
struct GradientBoostedTreesBinaryClassificationQuickScorerExtended
    : internal::QuickScorerExtendedModel {
  static constexpr model::proto::Task kTask =
      model::proto::Task::CLASSIFICATION;
};

// Specialization of quick scorer for GBDT ranking model.
struct GradientBoostedTreesRankingQuickScorerExtended
    : internal::QuickScorerExtendedModel {
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
                              const DataSpecification& dataspec,
                              CompiledModel* dst);

// Generates a human readable text describing the internal of the quick scorer
// model.
//
// This description is intended for debugging or optimization purpose. For a ML
// development intended description of the model, use the "describe" method on
// the non-compiled model.
template <typename Model>
std::string DescribeQuickScorer(const Model& model, bool detailed = true);

}  // namespace decision_forest
}  // namespace serving
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_QUICK_SCORER_EXTENDED_H_
