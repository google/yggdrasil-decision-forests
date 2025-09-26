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

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_JAVA_NODE_DATA_BANK_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_JAVA_NODE_DATA_BANK_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

// A union type that can hold either an int64_t or a float.
using Int64OrFloat = std::variant<int64_t, float>;

struct NodeDataArray {
  absl::string_view java_name;
  std::vector<Int64OrFloat> data = {};
  std::string java_type;

  // Returns a String with the data properly serialized to Java. Returns an
  // empty string if the data is empty.
  absl::StatusOr<std::string> SerializeToString() const;
  // Returns the Java read method for an array of this type.
  absl::StatusOr<std::string> GetJavaReadMethod() const;
};

// Manages the various data banks of a model. For Java exports, all data banks
// are written to a resource file to bypass code size limitations.
struct ModelDataBank {
  // The model statistics are used to determine which node data arrays are
  // necessary.
  ModelDataBank(const BaseInternalOptions& internal_options,
                const ModelStatistics& stats,
                const SpecializedConversion& specialized_conversion);

  struct AddNodeOptions {
    std::optional<int64_t> pos;
    std::optional<Int64OrFloat> val;
    std::optional<int64_t> feat;
    std::optional<Int64OrFloat> thr;
    std::optional<int64_t> cat;
    std::optional<int64_t> obl;
    Int64OrFloat sentinel = int64_t{0};
    std::vector<float> oblique_weights;
    std::vector<size_t> oblique_features;
    std::vector<float> leaf_values;
  };

  absl::Status AddNode(const AddNodeOptions& options);

  absl::Status AddRootDelta(int64_t new_root_delta);

  absl::Status AddConditionTypes(
      const std::vector<uint8_t>& new_condition_types);

  absl::StatusOr<size_t> GetObliqueFeaturesSize() const;

  absl::StatusOr<size_t> GetLeafValuesSize() const;

  // Set the Java types of the arrays. This function must be called after the
  // nodes have been added.
  absl::Status FinalizeJavaTypes();

  absl::StatusOr<std::string> GenerateJavaCode(
      const BaseInternalOptions& internal_options, absl::string_view class_name,
      absl::string_view resource_name) const;

  absl::StatusOr<std::string> SerializeData(
      const BaseInternalOptions& internal_options) const;

 public:
  // Values for categorical conditions.
  std::vector<bool> categorical;

  // Number of conditions for each condition implementations.
  std::array<int, static_cast<int>(
                      RoutingConditionType::NUM_ROUTING_CONDITION_TYPES)>
      num_conditions{0};

 private:
  // Returns a vector of pointers to the optional NodeDataArray members in the
  // order they should be serialized and deserialized.
  std::vector<const std::optional<NodeDataArray>*> GetOrderedNodeDataArrays()
      const;

  // Offset of the positive child -1, or 0 if the node is a leaf.
  std::optional<NodeDataArray> node_pos;
  // Value of the node or an unused sentinel if the node is not a leaf.
  std::optional<NodeDataArray> node_val;
  // Feature of the node condition or an unused sentinel if the node is a leaf.
  std::optional<NodeDataArray> node_feat;
  // Threshold of the is_higher condition or an unused sentinel if the node
  // does not have an is_higher condition or is a leaf.
  //
  // This array is not serialized if the model does not contain any is_higher
  // conditions.
  std::optional<NodeDataArray> node_thr;
  // Index of the node's contains bitmap in the aggregated contains bitmap
  // or an unused sentinel if the node does not have a contains condition or is
  // a leaf.
  //
  // This array is not serialized if the model does not contain any contains
  // conditions.
  std::optional<NodeDataArray> node_cat;
  // Index of the node's oblique weights and oblique features in the
  // corresponding vectors or an unused sentinel if the node does not have a
  // oblique condition or is a leaf.
  //
  // This array is not serialized if the model does not contain any oblique
  // conditions.
  std::optional<NodeDataArray> node_obl;
  // Weights to encode oblique conditions. Also contains the threshold.
  //
  // An oblique condition for a node with a given "obl(ique")" is defined as
  // follow:
  // num_oblique_projections := oblique_features[obl]
  // threshold := oblique_weights[obl]
  // acc := 0
  // for i in 0..num_oblique_projections:
  //   offset := obl + i + 1
  //   acc += feature_value[oblique_features[offset]] *
  //     oblique_weights[offset];
  // eval := acc >= threshold
  //
  // This array is not serialized if the model does not contain any oblique
  // conditions.
  std::optional<NodeDataArray> oblique_weights;
  // Feature index to encode oblique conditions. Also contain the number of
  // features in an oblique split. See definition of "oblique_weights".
  //
  // This array is not serialized if the model does not contain any oblique
  // conditions.
  std::optional<NodeDataArray> oblique_features;
  // The "root_deltas" contains the number of nodes in each tree. The node
  // index of the root of each tree can be computed by running a cumulative
  // sum.
  std::optional<NodeDataArray> root_deltas;
  // For each node, the type of condition used in the node.
  //
  // This array is not serialized if the model contains only a single type of
  // condition.
  std::optional<NodeDataArray> condition_types;
  // For multi-output leaves, the bank storing all leaf outputs.
  //
  // This array is not serialized if the model uses single-output leaves.
  std::optional<NodeDataArray> leaf_values;
};

}  // namespace yggdrasil_decision_forests::serving::embed::internal
#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_JAVA_NODE_DATA_BANK_H_
