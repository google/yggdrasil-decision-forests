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

#include "yggdrasil_decision_forests/serving/embed/java/model_data_bank.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"
#include "yggdrasil_decision_forests/serving/embed/utils.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

namespace {

// Helper functions to append data in Big-Endian format.

void AppendByte(std::string& dest, int8_t value) { dest.push_back(value); }

void AppendShort(std::string& dest, uint16_t value) {
  dest.push_back(static_cast<char>((value >> 010) & 0xFF));
  dest.push_back(static_cast<char>(value & 0xFF));
}

void AppendInt(std::string& dest, uint32_t value) {
  dest.push_back(static_cast<char>((value >> 030) & 0xFF));
  dest.push_back(static_cast<char>((value >> 020) & 0xFF));
  dest.push_back(static_cast<char>((value >> 010) & 0xFF));
  dest.push_back(static_cast<char>(value & 0xFF));
}

void AppendLong(std::string& dest, uint64_t value) {
  dest.push_back(static_cast<char>((value >> 070) & 0xFF));
  dest.push_back(static_cast<char>((value >> 060) & 0xFF));
  dest.push_back(static_cast<char>((value >> 050) & 0xFF));
  dest.push_back(static_cast<char>((value >> 040) & 0xFF));
  dest.push_back(static_cast<char>((value >> 030) & 0xFF));
  dest.push_back(static_cast<char>((value >> 020) & 0xFF));
  dest.push_back(static_cast<char>((value >> 010) & 0xFF));
  dest.push_back(static_cast<char>(value & 0xFF));
}

void AppendFloat(std::string& dest, float value) {
  uint32_t as_int = absl::bit_cast<uint32_t>(value);
  AppendInt(dest, as_int);
}

}  // namespace

absl::StatusOr<std::string> NodeDataArray::GetJavaReadMethod() const {
  if (java_type == "float") {
    return "readFloat()";
  } else if (java_type == "int") {
    return "readInt()";
  } else if (java_type == "short") {
    return "readShort()";
  } else if (java_type == "byte") {
    return "readByte()";
  } else {
    return absl::InternalError(absl::StrCat("Unknown java type for array ",
                                            java_name, ": ", java_type));
  }
}

absl::StatusOr<std::string> NodeDataArray::SerializeToString() const {
  std::string serialized_data;
  for (const auto& value : data) {
    if (java_type == "float") {
      if (std::holds_alternative<float>(value)) {
        AppendFloat(serialized_data, std::get<float>(value));
      } else {
        // Sentinel value
        STATUS_CHECK_EQ(std::get<int64_t>(value), 0);
        AppendFloat(serialized_data, 0.0);
      }
    } else if (java_type == "int") {
      STATUS_CHECK(std::holds_alternative<int64_t>(value));
      AppendInt(serialized_data,
                static_cast<int32_t>(std::get<int64_t>(value)));
    } else if (java_type == "short") {
      STATUS_CHECK(std::holds_alternative<int64_t>(value));
      AppendShort(serialized_data,
                  static_cast<int16_t>(std::get<int64_t>(value)));
    } else if (java_type == "byte") {
      STATUS_CHECK(std::holds_alternative<int64_t>(value));
      AppendByte(serialized_data,
                 static_cast<int8_t>(std::get<int64_t>(value)));
    } else {
      return absl::InternalError(absl::StrCat("Unknown java type for array ",
                                              java_name, ": ", java_type));
    }
  }
  return serialized_data;
}

ModelDataBank::ModelDataBank(
    const BaseInternalOptions& internal_options, const ModelStatistics& stats,
    const SpecializedConversion& specialized_conversion) {
  node_pos.emplace(NodeDataArray{
      .java_name = "nodePos",
      .java_type = JavaInteger(internal_options.node_offset_bytes)});
  if (specialized_conversion.leaf_value_spec.dims == 1) {
    node_val.emplace(NodeDataArray{
        .java_name = "nodeVal",
        .java_type =
            DTypeToJavaType(specialized_conversion.leaf_value_spec.dtype)});
  } else {
    node_val.emplace(NodeDataArray{
        .java_name = "nodeVal",
        .java_type = JavaInteger(MaxUnsignedValueToNumBytes(
            stats.num_leaves / specialized_conversion.leaf_value_spec.dims))});
  }
  node_feat.emplace(NodeDataArray{
      .java_name = "nodeFeat",
      .java_type = JavaInteger(internal_options.feature_index_bytes)});
  root_deltas.emplace(NodeDataArray{
      .java_name = "rootDeltas",
      .java_type = JavaInteger(internal_options.node_offset_bytes)});

  if (stats.has_conditions
          [model::decision_tree::proto::Condition::kHigherCondition]) {
    node_thr.emplace(NodeDataArray{.java_name = "nodeThr"});
    if (internal_options.numerical_feature_is_float) {
      DCHECK_EQ(internal_options.feature_value_bytes, 4);
      node_thr->java_type = "float";
    } else {
      node_thr->java_type = JavaInteger(internal_options.feature_value_bytes);
    }
  }
  if (stats.has_conditions
          [model::decision_tree::proto::Condition::kContainsBitmapCondition] ||
      stats.has_conditions
          [model::decision_tree::proto::Condition::kContainsCondition]) {
    DCHECK_GT(internal_options.categorical_idx_bytes, 0);
    node_cat.emplace(NodeDataArray{
        .java_name = "nodeCat",
        .java_type = JavaInteger(internal_options.categorical_idx_bytes)});
  }
  if (stats.has_conditions
          [model::decision_tree::proto::Condition::kObliqueCondition]) {
    node_obl.emplace(NodeDataArray{.java_name = "nodeObl"});
    oblique_weights.emplace(
        NodeDataArray{.java_name = "obliqueWeights", .java_type = "float"});
    oblique_features.emplace(NodeDataArray{
        .java_name = "obliqueFeatures",
        .java_type = JavaInteger(internal_options.feature_index_bytes)});
  }
  if (specialized_conversion.leaf_value_spec.dims > 1) {
    leaf_values.emplace(
        NodeDataArray{.java_name = "leafValues", .java_type = "float"});
  }
}

absl::StatusOr<size_t> ModelDataBank::GetObliqueFeaturesSize() const {
  if (!oblique_features.has_value()) {
    return absl::InternalError("No oblique features used.");
  }
  return oblique_features->data.size();
}

absl::StatusOr<size_t> ModelDataBank::GetLeafValuesSize() const {
  if (!leaf_values.has_value()) {
    return absl::InternalError("No external leaf values used.");
  }
  return leaf_values->data.size();
}

absl::Status ModelDataBank::AddNode(const AddNodeOptions& options) {
  auto append_value = [&](std::optional<NodeDataArray>& array,
                          const std::optional<Int64OrFloat>& value) {
    if (array.has_value()) {
      if (value.has_value()) {
        array->data.push_back(*value);
      } else {
        array->data.push_back(options.sentinel);
      }
    } else {
      if (value.has_value()) {
        return absl::InternalError(
            "Expected array to push node to, but no array was found.");
      }
    }
    return absl::OkStatus();
  };

  RETURN_IF_ERROR(append_value(node_pos, options.pos));
  RETURN_IF_ERROR(append_value(node_val, options.val));
  RETURN_IF_ERROR(append_value(node_feat, options.feat));
  RETURN_IF_ERROR(append_value(node_thr, options.thr));
  RETURN_IF_ERROR(append_value(node_cat, options.cat));
  RETURN_IF_ERROR(append_value(node_obl, options.obl));

  for (const float val : options.oblique_weights) {
    if (!oblique_weights.has_value()) {
      return absl::InternalError(
          "Expected array oblique_weights to push node to, but no array was "
          "found.");
    }
    oblique_weights->data.push_back(val);
  }
  for (const size_t val : options.oblique_features) {
    if (!oblique_features.has_value()) {
      return absl::InternalError(
          "Expected array oblique_features to push node to, but no array was "
          "found.");
    }
    oblique_features->data.push_back(static_cast<int64_t>(val));
  }
  for (const float val : options.leaf_values) {
    if (!leaf_values.has_value()) {
      return absl::InternalError(
          "Expected array leaf_values to push node to, but no array was "
          "found.");
    }
    leaf_values->data.push_back(val);
  }

  return absl::OkStatus();
}

absl::Status ModelDataBank::FinalizeJavaTypes() {
  if (oblique_features.has_value()) {
    STATUS_CHECK(!oblique_features->data.empty());
    node_obl->java_type =
        JavaInteger(MaxSignedValueToNumBytes(oblique_features->data.size()));
  };

  return absl::OkStatus();
}

absl::Status ModelDataBank::AddRootDelta(int64_t root_delta) {
  STATUS_CHECK(root_deltas.has_value());
  root_deltas->data.push_back(root_delta);
  return absl::OkStatus();
}

absl::Status ModelDataBank::AddConditionTypes(
    const std::vector<uint8_t>& new_condition_types) {
  if (condition_types.has_value()) {
    return absl::InternalError("The condition types have already been set.");
  }

  std::vector<Int64OrFloat> condition_types_data;
  condition_types_data.reserve(new_condition_types.size());
  for (const uint8_t val : new_condition_types) {
    condition_types_data.push_back(static_cast<int64_t>(val));
  }

  condition_types.emplace(NodeDataArray{.java_name = "conditionTypes",
                                        .data = condition_types_data,
                                        .java_type = "byte"});
  return absl::OkStatus();
}

std::vector<const std::optional<NodeDataArray>*>
ModelDataBank::GetOrderedNodeDataArrays() const {
  return {&node_pos,        &node_val,         &node_feat,   &node_thr,
          &node_cat,        &node_obl,         &root_deltas, &condition_types,
          &oblique_weights, &oblique_features, &leaf_values};
}

absl::StatusOr<std::string> ModelDataBank::GenerateJavaCode(
    const BaseInternalOptions& internal_options, absl::string_view class_name,
    absl::string_view resource_name) const {
  std::string declarations;
  std::string static_block;

  // Helper to generate loader for NodeDataArray
  auto generate_node_array_loader =
      [&](const std::optional<NodeDataArray>& array) -> absl::Status {
    if (!array.has_value()) {
      return absl::OkStatus();
    }
    const auto& data = array->data;
    const auto& java_type = array->java_type;
    const absl::string_view java_name = array->java_name;
    if (java_type.empty()) {
      if (!data.empty()) {
        return absl::InternalError(
            absl::StrCat("Array ", java_name, " has data but no java type."));
      }
      return absl::OkStatus();
    }
    if (data.empty()) {
      return absl::InternalError(
          absl::StrCat("Array ", java_name, " is unexpectedly empty."));
    }
    absl::SubstituteAndAppend(&declarations,
                              "  private static final $0[] $1;\n", java_type,
                              java_name);

    absl::SubstituteAndAppend(&static_block,
                              "    int $0Length = dis.readInt();\n"
                              "    $1 = new $2[$0Length];\n"
                              "    for (int i = 0; i < $0Length; i++) {\n",
                              java_name, java_name, java_type);

    ASSIGN_OR_RETURN(const std::string read_method, array->GetJavaReadMethod());
    absl::SubstituteAndAppend(&static_block, "      $0[i] = dis.$1;\n",
                              java_name, read_method);
    absl::StrAppend(&static_block, "    }\n");
    return absl::OkStatus();
  };

  // The order of reads in the Java resource must match the order in the binary
  // resource.

  // 1. NodeDataArrays.
  for (const auto* array_ptr : GetOrderedNodeDataArrays()) {
    RETURN_IF_ERROR(generate_node_array_loader(*array_ptr));
  }

  // 2. Categorical condition bank if categorical conditions exist.
  if (node_cat.has_value()) {
    absl::StrAppend(&declarations,
                    "  private static final BitSet categoricalBank;\n");
    absl::StrAppend(
        &static_block,
        "    int categoricalBankNumLongs = dis.readInt();\n"
        "    if (categoricalBankNumLongs > 0) {\n"
        "      long[] longs = new long[categoricalBankNumLongs];\n"
        "      for (int i = 0; i < categoricalBankNumLongs; i++) {\n"
        "        longs[i] = dis.readLong();\n"
        "      }\n"
        "      categoricalBank = BitSet.valueOf(longs);\n"
        "    } else {\n"
        "      categoricalBank = new BitSet();\n"
        "    }\n");
  }

  std::string content = declarations;
  absl::StrAppend(&content, "\n  static {\n");
  absl::SubstituteAndAppend(&content,
                            "    try (InputStream is = "
                            "$0.class.getResourceAsStream(\"$1\");\n"
                            "         DataInputStream dis = new "
                            "DataInputStream(new "
                            "BufferedInputStream(is))) {\n",
                            class_name, resource_name);
  absl::StrAppend(&content, static_block);
  absl::StrAppend(&content,
                  "    } catch (IOException e) {\n"
                  "      throw new RuntimeException(\"Failed to load model "
                  "data resource: \" + e.getMessage(), e);\n"
                  "    }\n"
                  "  }\n");

  return content;
}

absl::StatusOr<std::string> ModelDataBank::SerializeData(
    const BaseInternalOptions& internal_options) const {
  if (node_obl.has_value() && node_obl->java_type.empty()) {
    return absl::InternalError(
        "ModelDataBank is not finalized (oblique type not set).");
  }

  std::string binary_data;

  // Helper to serialize NodeDataArray
  auto serialize_node_array =
      [&](const std::optional<NodeDataArray>& array) -> absl::Status {
    if (!array.has_value()) {
      // No data, don't serialize.
      return absl::OkStatus();
    }
    const auto& data = array->data;
    const auto& java_type = array->java_type;
    const auto& java_name = array->java_name;
    if (java_type.empty() && !data.empty()) {
      return absl::InternalError(
          absl::StrCat("Array ", java_name, " has data but no java type."));
    }
    AppendInt(binary_data, data.size());
    ASSIGN_OR_RETURN(const auto serialized_array, array->SerializeToString());
    absl::StrAppend(&binary_data, serialized_array);

    return absl::OkStatus();
  };

  // The order of appends must match the order of reads in the Java resource
  // loader.

  // 1. NodeDataArrays.
  for (const auto* array_ptr : GetOrderedNodeDataArrays()) {
    RETURN_IF_ERROR(serialize_node_array(*array_ptr));
  }

  // 2. Categorical condition bank if categorical conditions exist.
  if (node_cat.has_value()) {
    size_t num_longs = (categorical.size() + 63) / 64;
    std::vector<uint64_t> longs(num_longs, 0);
    for (size_t i = 0; i < categorical.size(); ++i) {
      if (categorical[i]) {
        longs[i / 64] |= (1ULL << (i % 64));
      }
    }
    AppendInt(binary_data, longs.size());
    for (uint64_t val : longs) {
      AppendLong(binary_data, val);
    }
  } else {
    AppendInt(binary_data, 0);  // Zero longs
  }

  return binary_data;
}

}  // namespace yggdrasil_decision_forests::serving::embed::internal
