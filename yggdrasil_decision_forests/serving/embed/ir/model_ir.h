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

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_IR_MODEL_IR_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_IR_MODEL_IR_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/serving/embed/utils.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

using FeatureIdx = int32_t;
using NodeIdx = int64_t;

struct FeatureInfo {
  // The name in the data_spec.
  std::string original_name;

  enum class Type {
    kBoolean,
    kNumerical,
    kCategorical,
    kIntegerizedCategorical
  } type;

  bool is_float = false;
  bool is_label = false;

  // Replacement value for NaNs / missing values.
  std::optional<DoubleOrInt64> na_replacement = {};

  // Used for integer categoricals in the NaN-replacement phase.
  std::optional<int64_t> maximum_value = {};

  std::vector<std::string> vocabulary = {};
};

enum class ConditionType {
  HIGHER_CONDITION = 0,
  CONTAINS_CONDITION_BUFFER_BITMAP = 1,
  OBLIQUE_CONDITION = 2,
  TRUE_CONDITION = 3,
  NUM_ROUTING_CONDITION_TYPES = 4,
};

struct Node {
  enum class Type { kCondition, kLeaf } type = Type::kLeaf;

  ConditionType condition_type;

  FeatureIdx feature_idx = -1;

  DoubleOrInt64 threshold_or_offset = 0.0;

  // Offset to the positive child in the nodes array.
  // Note that the offset to the negative child is 1.
  NodeIdx next_pos_node_idx = -1;

  FeatureIdx num_oblique_features = 0;

  NodeIdx tree_idx = -1;
};

struct ModelIR {
  enum class ModelType { kGradientBoostedTrees, kRandomForest } model_type;
  NodeIdx num_trees = 0;
  NodeIdx num_leaves = 0;
  int32_t num_output_classes = 0;

  proto::DType::Enum leaf_value_dtype = proto::DType::UNDEFINED;
  int leaf_value_dims = 0;

  bool winner_takes_all = false;
  std::vector<DoubleOrInt64> accumulator_initialization;

  enum class Task {
    kRegression,
    kBinaryClassification,
    kMulticlassClassification,
  } task;

  enum class Activation {
    kEquality,
    kSigmoid,
    kSoftmax,
  } activation = Activation::kEquality;

  std::vector<FeatureInfo> features;
  FeatureIdx num_features;
  // All features are currently stored with the same number of bytes.
  int32_t feature_value_bytes = 0;

  std::vector<Node> nodes;
  std::vector<NodeIdx> tree_start_offsets;
  int32_t node_offset_bytes;

  std::vector<ConditionType> active_condition_types;

  std::vector<bool> bitset_bank;
  // If a leaf has multiple outputs, those are (currently) float only.
  std::vector<float> leaf_value_bank;

  std::vector<float> oblique_weights;  // Always stored as float in the model.
  std::vector<FeatureIdx> oblique_features;
};
}  // namespace yggdrasil_decision_forests::serving::embed::internal
#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_IR_MODEL_IR_H_
