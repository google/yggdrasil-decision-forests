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

// This file contains a fast but limited inference engine for models with uint8
// numerical features.
//
// Limitations:
//   - For binary classification, regression and ranking GBT models
//   - Only for uint8 numerical features represented as follows:
//     - With only discretized numerical features with integers buckets starting
//       at 0.
//     - The number of buckets to encode the discretized numerical features
//       should be less or equal than 256.
//     - The bucket boundaries should be [0.5, 1.5, 2.5, ....].
//   - No support for missing values.
//   - The input features are fed as an array of uint8 values. The order of the
//     features is determined by the dataspec.
//   - Maximum of 64 leaves per trees.
//
// This engine is faster than other engines for very small models (e.g. a few
// 10s of trees). For larger models, the QuickScorer engine (when compatible)
// is faster.
//
// This engine is not selected automatically during model inference. The only
// way to use this engine is to call it directly. See
// examples/fast_8bits_numerical.cc for an example
//
// This engine works with a leaf masking algorithm similar to quick scorer.
// However, instead of looking for the mask corresponding to a feature value
// using a linear search (like QuickScorer), this engine directly uses the
// feature value as an index in a short array. This is the reason this engine
// only supports uint8 numerical features.

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_8BITS_NUMERICAL_FEATURES_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_8BITS_NUMERICAL_FEATURES_H_

#include <limits>
#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace decision_forest {
namespace num_8bits {

// Bitmap over the leafs in a tree. Limits the number of leafs in a tree.
using LeafMask = uint64_t;

// The value of a leaf.
using LeafOutput = float;

// Index of a tree.
using TreeIdx = uint32_t;

// Maximum number of trees and number of leafs per tree.
static constexpr size_t kMaxTrees = std::numeric_limits<TreeIdx>::max();
static constexpr size_t kMaxLeafs = sizeof(LeafMask) * 8;

// Helper values.
static constexpr LeafMask kZeroLeafMask = static_cast<LeafMask>(0);
static constexpr LeafMask kOneLeafMask = static_cast<LeafMask>(1);
static constexpr LeafMask kFullLeafMask = std::numeric_limits<LeafMask>::max();

struct RawModel {
  // Column idx (in the dataspec) of the input features of the model.
  const std::vector<int>& get_features() const { return features; }

  int num_trees;
  int num_features;

  // Bias of the model.
  float initial_prediction;

  // Node mask according to the feature values.
  //
  // The masks "masks_v2[begin..begin+num_trees]"
  // with begin=feature_value_to_mask_list[feature_to_feature_value[f]+v] , is
  // the mask to apply when the feature "f" is observed with value "v".
  std::vector<uint32_t> feature_to_feature_value;
  std::vector<uint32_t> feature_value_to_mask_list;
  std::vector<LeafMask> masks_v2;

  // Number of buckets for each feature.
  // Used only for debugging.
  //
  // The value should be in [0, 256].
  std::vector<uint16_t> num_buckets;

  // Leaf values.
  //
  // "leaves[leaves_tree_index[t]]..leaves[leaves_tree_index[t+1]]-1" are the
  // leaf values of tree "t".
  std::vector<LeafOutput> leaves;
  std::vector<uint32_t> leaves_tree_index;

  // Column idx (in the dataspec) of the input features of the model.
  std::vector<int> features;
};

// Data used during the compilation of the model.
struct BuildingWorkMemory {
  struct Mask {
    // Tree to mask.
    int tree_idx;
    // Bitmap mask on the leaves.
    LeafMask mask;
  };

  struct Masks {
    // The set of masks.
    std::vector<Mask> masks;
  };

  struct Feature {
    // Name of the feature. Similar as the dataspec feature name.
    std::string name;

    // Index of the feature from the point of view of the engine.
    uint32_t local_idx;

    // "value_to_masks[v]" contains the information observed for the feature
    // value "v".
    std::vector<Masks> value_to_masks;
  };

  // Mapping from a column index (in the dataspec) to feature data.
  std::map<int, Feature> features;

  // Total number of feature values. "sum_num_buckets" is the sum of the
  // "num_buckets" of each individual features.
  uint32_t sum_num_buckets;
};

struct GradientBoostedTreesBinaryClassificationModel : RawModel {};
struct GradientBoostedTreesBinaryRegressiveModel : RawModel {};

// Compiles a model into a compatible engine.
absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesBinaryClassificationModel* dst);

absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesBinaryRegressiveModel* dst);

// Run the engine on a set of examples.
//
// Args:
//   model: A compiled model.
//   examples: Input features in row-major, feature-minor format. The order of
//     the features is defined by "model.get_features()". For example,
//     "examples[num_examples * i + j]" is the value of the feature
//     "k = model.get_features()[j]" for the "i-th" example. The string name of
//     the feature is available using the dataspec i.e.
//     "pre_compiled_model.data_spec().columns(k).name()"
//   num_examples: Number of examples. predictions: Output predictions. Will be
//     resized to "num_examples".

absl::Status Predict(const GradientBoostedTreesBinaryClassificationModel& model,
                     const std::vector<uint8_t>& examples,
                     uint32_t num_examples, std::vector<float>* predictions);

absl::Status Predict(const GradientBoostedTreesBinaryRegressiveModel& model,
                     const std::vector<uint8_t>& examples,
                     uint32_t num_examples, std::vector<float>* predictions);

// Human readable string with information about the engine.
std::string EngineDetails(
    const GradientBoostedTreesBinaryClassificationModel& model);

std::string EngineDetails(
    const GradientBoostedTreesBinaryRegressiveModel& model);

}  // namespace num_8bits
}  // namespace decision_forest
}  // namespace serving
}  // namespace yggdrasil_decision_forests

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_8BITS_NUMERICAL_FEATURES_H_
