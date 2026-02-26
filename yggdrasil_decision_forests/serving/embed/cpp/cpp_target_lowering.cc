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

#include "yggdrasil_decision_forests/serving/embed/cpp/cpp_target_lowering.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/cpp_ir.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/serving/embed/ir/model_ir.h"
#include "yggdrasil_decision_forests/serving/embed/utils.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed::internal {
namespace {
// Number of feature indexes to reserve.
// The precision of feature indexes should be sufficient to encode the index
// of any features + kReservedFeatureIndexes. For example, if the model has
// 254 features, but kReservedFeatureIndexes=2, the maximum feature index is
// 254+2=256, and feature indexes are encoded as uint16 (instead of uint8).
static constexpr int kReservedFeatureIndexes = 1;

std::string DTypeToCCType(const proto::DType::Enum value) {
  switch (value) {
    case proto::DType::UNDEFINED:
      return "UNDEFINED";

    case proto::DType::INT8:
      return "int8_t";
    case proto::DType::INT16:
      return "int16_t";
    case proto::DType::INT32:
      return "int32_t";

    case proto::DType::UINT8:
      return "uint8_t";
    case proto::DType::UINT16:
      return "uint16_t";
    case proto::DType::UINT32:
      return "uint32_t";

    case proto::DType::FLOAT32:
      return "float";

    case proto::DType::BOOL:
      return "bool";
  }
}

bool IsZero(const DoubleOrInt64 val) {
  if (IsDouble(val)) {
    return AsDouble(val) == 0.0;
  } else {
    return AsInt(val) == 0;
  }
}

}  // namespace

absl::StatusOr<CppIR> CppTargetLowering::Lower(const ModelIR& model_ir,
                                               const proto::Options& options) {
  CppTargetLowering lowerer(model_ir, options);
  ASSIGN_OR_RETURN(CppIR cpp_ir, lowerer.Run());
  return cpp_ir;
}

absl::StatusOr<CppIR> CppTargetLowering::Run() {
  if (model_ir_.leaf_value_dims <= 0) {
    return absl::InvalidArgumentError("leaf_value_dims must be positive");
  }

  RETURN_IF_ERROR(LowerGlobalFormatting());

  // Step 2: Translate Variables & Types
  RETURN_IF_ERROR(LowerEnumsAndFeatures());

  // Step 3: Accumulator & Activation
  RETURN_IF_ERROR(LowerInferenceEngine());

  // Step 4: Tree Nodes
  RETURN_IF_ERROR(LowerNodes());

  // Step 5: Routing Data (Conditional)
  if (options_.algorithm() == proto::Algorithm::ROUTING) {
    RETURN_IF_ERROR(LowerRoutingData());
  }

  return std::move(cpp_ir_);
}

absl::Status CppTargetLowering::LowerGlobalFormatting() {
  cpp_ir_.namespace_name = StringToVariableSymbol(options_.name());
  cpp_ir_.header_guard = absl::Substitute(
      "YDF_MODEL_$0_H_", StringToConstantSymbol(options_.name()));
  cpp_ir_.num_trees = model_ir_.num_trees;
  // For integer types.
  cpp_ir_.includes.insert("<stdint.h>");
  // For memcpy.
  cpp_ir_.includes.insert("<cstring>");
  // For assert().
  cpp_ir_.includes.insert("<cassert>");

  if (model_ir_.activation == ModelIR::Activation::kSigmoid ||
      model_ir_.activation == ModelIR::Activation::kSoftmax) {
    cpp_ir_.includes.insert("<cmath>");
  }
  cpp_ir_.node_offset_type = UnsignedInteger(model_ir_.node_offset_bytes);
  cpp_ir_.root_deltas_type = UnsignedInteger(model_ir_.node_offset_bytes);

  return absl::OkStatus();
}

absl::Status CppTargetLowering::LowerEnumsAndFeatures() {
  absl::flat_hash_set<std::string> sanitized_feature_names;
  absl::flat_hash_set<std::string> sanitized_categorical_type_names;

  const size_t num_features = model_ir_.features.size();

  cpp_ir_.unsigned_integer_type =
      UnsignedInteger(model_ir_.feature_value_bytes);
  cpp_ir_.signed_integer_type = SignedInteger(model_ir_.feature_value_bytes);

  std::vector<int> condition_types_bank;
  condition_types_bank.reserve(num_features);

  // Generate the Features.
  cpp_ir_.features.reserve(num_features);
  for (size_t i = 0; i < num_features; ++i) {
    RETURN_IF_ERROR(LowerFeature(i, sanitized_feature_names,
                                 sanitized_categorical_type_names,
                                 condition_types_bank));
  }

  // Generate the Enums.
  for (size_t i = 0; i < num_features; ++i) {
    RETURN_IF_ERROR(
        LowerEnum(i, model_ir_.features[i], sanitized_categorical_type_names));
  }

  cpp_ir_.condition_types_content = absl::StrJoin(condition_types_bank, ",");
  if (condition_types_bank.size() > std::numeric_limits<uint8_t>::max()) {
    return absl::InvalidArgumentError(
        "The number of condition types exceeds the maximum value of uint8_t.");
  }
  cpp_ir_.condition_types_type = "uint8_t";

  return absl::OkStatus();
}

absl::Status CppTargetLowering::LowerFeature(
    const FeatureIdx feature_idx,
    absl::flat_hash_set<std::string>& sanitized_feature_names,
    absl::flat_hash_set<std::string>& sanitized_categorical_type_names,
    std::vector<int>& condition_types_bank) {
  const auto& feature = model_ir_.features[feature_idx];
  // Labels are not features.
  if (feature.is_label) {
    return absl::OkStatus();
  }

  const std::string var_name = StringToVariableSymbol(feature.original_name);
  if (!sanitized_feature_names.insert(var_name).second) {
    std::vector<std::string> model_ir_features;
    model_ir_features.reserve(model_ir_.features.size());
    for (const auto& f : model_ir_.features) {
      model_ir_features.push_back(f.original_name);
    }
    return absl::InvalidArgumentError(
        absl::Substitute("Feature name clash on feature $0, consider "
                         "renaming your features!",
                         var_name));
  }

  CppFeature cpp_feature;
  cpp_feature.var_name = var_name;

  // Track the mapping from the IR index to the C++ variable name
  feat_idx_to_var_name_[feature_idx] = var_name;

  switch (feature.type) {
    case FeatureInfo::Type::kNumerical: {
      cpp_feature.cpp_type = "Numerical";
      if (feature.is_float) {
        // If we have a float feature, we need at least 4 bytes.
        STATUS_CHECK_EQ(model_ir_.feature_value_bytes, 4);
        cpp_ir_.numerical_type = "float";
        STATUS_CHECK(feature.na_replacement.has_value());
        ASSIGN_OR_RETURN(const std::string un,
                         FormatLiteral(*feature.na_replacement,
                                       /*is_float=*/true));
        cpp_feature.na_sanitization = {absl::Substitute(
            "instance.$0 = std::isnan(instance.$0) ? $1 : instance.$0;",
            var_name, un)};
        cpp_ir_.includes.insert("<cmath>");
        cpp_ir_.needs_predict_unsafe = true;
      } else if (cpp_ir_.numerical_type.empty()) {
        cpp_ir_.numerical_type = SignedInteger(model_ir_.feature_value_bytes);
      }
      condition_types_bank.push_back(
          static_cast<int>(ConditionType::HIGHER_CONDITION));
    } break;
    case FeatureInfo::Type::kCategorical: {
      const std::string categorical_type =
          absl::StrCat("Feature", StringToStructSymbol(feature.original_name));
      if (!sanitized_categorical_type_names.insert(categorical_type).second) {
        return absl::InvalidArgumentError(absl::Substitute(
            "Feature name clash on categorical feature $0, consider "
            "renaming your features.",
            categorical_type));
      }
      cpp_feature.cpp_type = categorical_type;
      condition_types_bank.push_back(
          static_cast<int>(ConditionType::CONTAINS_CONDITION_BUFFER_BITMAP));
    } break;
    case FeatureInfo::Type::kIntegerizedCategorical: {
      cpp_ir_.has_integerized_numerical_features = true;
      cpp_feature.cpp_type = "IntegerizedCategorical";
      STATUS_CHECK(feature.na_replacement.has_value());
      STATUS_CHECK(feature.maximum_value.has_value());
      cpp_feature.na_sanitization.push_back(absl::Substitute(
          "instance.$0 = instance.$0 == -1 ? $1 : instance.$0;", var_name,
          std::get<int64_t>(*feature.na_replacement)));
      cpp_feature.na_sanitization.push_back(absl::Substitute(
          "instance.$0 = (instance.$0 < -1 || instance.$0 > $1) ? 0 : "
          "instance.$0;",
          var_name, *feature.maximum_value));
      cpp_ir_.needs_predict_unsafe = true;
      condition_types_bank.push_back(
          static_cast<int>(ConditionType::CONTAINS_CONDITION_BUFFER_BITMAP));
    } break;
    case FeatureInfo::Type::kBoolean: {
      cpp_feature.cpp_type = "bool";
      condition_types_bank.push_back(
          static_cast<int>(ConditionType::TRUE_CONDITION));
    } break;
  }
  cpp_ir_.features.push_back(cpp_feature);
  return absl::OkStatus();
}

absl::Status CppTargetLowering::LowerEnum(
    const FeatureIdx feature_idx, const FeatureInfo& feature,
    absl::flat_hash_set<std::string>& sanitized_categorical_type_names) {
  if (feature.type != FeatureInfo::Type::kCategorical) {
    return absl::OkStatus();
  }
  CppEnum cpp_enum;

  // Set of all sanitized_items for this specific enum.
  // Used to detect and resolve duplications exactly like common.cc
  absl::flat_hash_set<std::string> sanitized_items;

  if (feature.is_label) {
    cpp_enum.name = "Label";
    // Omit the OOV item for the label
    for (int i = 1; i < feature.vocabulary.size(); ++i) {
      std::string item_symbol = StringToStructSymbol(
          feature.vocabulary[i], /*ensure_letter_first=*/false);

      if (sanitized_items.contains(item_symbol)) {
        int local_index = 1;
        std::string extended_item_symbol;
        do {
          extended_item_symbol = absl::StrCat(item_symbol, "_", local_index++);
        } while (sanitized_items.contains(extended_item_symbol));
        item_symbol = extended_item_symbol;
      }
      sanitized_items.insert(item_symbol);

      cpp_enum.items.push_back({.name = absl::StrCat("k", item_symbol),
                                .value = i - 1,
                                .original = feature.vocabulary[i]});
    }
    // Labels are at the top of file.
    cpp_ir_.enums.insert(cpp_ir_.enums.begin(), cpp_enum);
  } else {
    const std::string feature_struct_name =
        StringToStructSymbol(feature.original_name);
    cpp_enum.name = absl::StrCat("Feature", feature_struct_name);

    // Track the mapping from the IR index to the Enum class name
    feat_idx_to_enum_name_[feature_idx] = cpp_enum.name;

    if (sanitized_categorical_type_names.insert(cpp_enum.name).second) {
      // Collision check already handled in previous loop for unique names,
      // but double check here just to be safe.
    }

    for (int i = 0; i < feature.vocabulary.size(); ++i) {
      if (i == 0) {
        cpp_enum.items.push_back(
            {.name = "kOutOfVocabulary", .value = 0, .original = ""});
        // Protect "OutOfVocabulary" from being overwritten by a weird user
        // category
        sanitized_items.insert("OutOfVocabulary");
      } else {
        std::string item_symbol = StringToStructSymbol(
            feature.vocabulary[i], /*ensure_letter_first=*/false);

        // Exact collision resolution logic from common.cc
        if (sanitized_items.contains(item_symbol)) {
          int local_index = 1;
          std::string extended_item_symbol;
          do {
            extended_item_symbol =
                absl::StrCat(item_symbol, "_", local_index++);
          } while (sanitized_items.contains(extended_item_symbol));
          item_symbol = extended_item_symbol;
        }
        sanitized_items.insert(item_symbol);

        cpp_enum.items.push_back({.name = absl::StrCat("k", item_symbol),
                                  .value = i,
                                  .original = feature.vocabulary[i]});
      }
    }
    if (options_.categorical_from_string()) {
      cpp_enum.generate_from_string_method = true;
      cpp_ir_.includes.insert("<string_view>");
      cpp_ir_.includes.insert("<unordered_map>");
    }
    // Other feature enums are in arbitrary order.
    cpp_ir_.enums.push_back(cpp_enum);
  }
  return absl::OkStatus();
}

absl::Status CppTargetLowering::LowerInferenceEngine() {
  RETURN_IF_ERROR(LowerAccumulator());
  RETURN_IF_ERROR(LowerActivation());

  return absl::OkStatus();
}

absl::Status CppTargetLowering::LowerAccumulator() {
  const std::string base_type =
      model_ir_.winner_takes_all
          ? UnsignedInteger(MaxUnsignedValueToNumBytes(model_ir_.num_trees))
          : "float";

  if (model_ir_.num_output_classes == 1) {
    cpp_ir_.accumulator_sum_statement = "accumulator += node->leaf.val;";
  } else {
    if (model_ir_.winner_takes_all) {
      cpp_ir_.accumulator_sum_statement = "accumulator[node->leaf.val]++;";
    } else {
      if (model_ir_.leaf_value_dims > 1) {
        cpp_ir_.accumulator_sum_statement = absl::Substitute(
            R"(const size_t offset = node->leaf.val * $0;
    for(int dim=0; dim!=$0; dim++) {
      accumulator[dim] += leaf_value_bank[offset + dim];
    })",
            model_ir_.num_output_classes);
      } else {
        cpp_ir_.accumulator_sum_statement =
            absl::Substitute("accumulator[tree_idx % $0] += node->leaf.val;",
                             model_ir_.num_output_classes);
      }
    }
  }
  ASSIGN_OR_RETURN(
      cpp_ir_.num_trees_type,
      StorageToCppType(MaxUnsignedValueToNumBytes(model_ir_.num_trees), false,
                       false));

  // 2. Initialize Accumulator
  if (model_ir_.num_output_classes == 1) {
    cpp_ir_.accumulator_type = base_type;
    ASSIGN_OR_RETURN(const std::string init_val,
                     FormatLiteral(model_ir_.accumulator_initialization[0],
                                   /*is_float=*/!model_ir_.winner_takes_all));
    cpp_ir_.accumulator_init_statement =
        absl::Substitute("$0 accumulator {$1};", base_type, init_val);
  } else {
    cpp_ir_.accumulator_type = absl::Substitute("std::array<$0, $1>", base_type,
                                                model_ir_.num_output_classes);
    cpp_ir_.output_type =
        absl::Substitute("std::array<float, $0>", model_ir_.num_output_classes);
    cpp_ir_.includes.insert("<array>");

    std::vector<std::string> init_vals;
    for (const auto& val : model_ir_.accumulator_initialization) {
      ASSIGN_OR_RETURN(
          const std::string str_val,
          FormatLiteral(val, /*is_float=*/!model_ir_.winner_takes_all));
      init_vals.push_back(str_val);
    }
    cpp_ir_.accumulator_init_statement =
        absl::Substitute("$0 accumulator {$1};", cpp_ir_.accumulator_type,
                         absl::StrJoin(init_vals, ", "));
  }

  if (model_ir_.task == ModelIR::Task::kRegression) {
    // Regression always returns the accumulated float score.
    cpp_ir_.output_type = "float";
  }
  return absl::OkStatus();
}

absl::Status CppTargetLowering::LowerActivation() {
  if (model_ir_.task == ModelIR::Task::kRegression) {
    cpp_ir_.activation_statement = "return accumulator;";
    return absl::OkStatus();
  }

  // For Classification, format the return statement based on the user's
  // requested output format.
  switch (options_.classification_output()) {
    case proto::ClassificationOutput::CLASS:
      cpp_ir_.output_type = "Label";
      if (model_ir_.num_output_classes == 1) {
        if (model_ir_.winner_takes_all) {
          cpp_ir_.activation_statement =
              absl::Substitute("return static_cast<Label>(accumulator > $0);",
                               model_ir_.num_trees / 2);
        } else {
          if (model_ir_.model_type == ModelIR::ModelType::kRandomForest) {
            cpp_ir_.activation_statement =
                "return static_cast<Label>(accumulator > 1);";

          } else if (model_ir_.model_type ==
                     ModelIR::ModelType::kGradientBoostedTrees) {
            cpp_ir_.activation_statement =
                "return static_cast<Label>(accumulator >= 0.0f);";
          } else {
            return absl::InvalidArgumentError("Invalid model type");
          }
        }
      } else {
        cpp_ir_.includes.insert("<algorithm>");
        cpp_ir_.includes.insert("<iterator>");
        cpp_ir_.activation_statement =
            "return static_cast<Label>(std::distance(accumulator.begin(), "
            "std::max_element(accumulator.begin(), accumulator.end())));";
      }
      break;

    case proto::ClassificationOutput::SCORE:
      cpp_ir_.activation_statement = "return accumulator;";
      cpp_ir_.output_type = cpp_ir_.accumulator_type;
      break;

    case proto::ClassificationOutput::PROBABILITY:
      if (model_ir_.winner_takes_all) {
        // Normalize integer votes into float probabilities.
        if (model_ir_.num_output_classes == 1) {
          cpp_ir_.activation_statement =
              absl::Substitute("return static_cast<float>(accumulator) / $0;",
                               model_ir_.num_trees);
          cpp_ir_.output_type = "float";
        } else {
          cpp_ir_.activation_statement = absl::Substitute(
              R"(  std::array<float, $0> probas;
  for (int i = 0; i < $0; ++i) { 
    probas[i] = static_cast<float>(accumulator[i]) / $1.0f; 
  }
  return probas;)",
              model_ir_.num_output_classes, model_ir_.num_trees);

          cpp_ir_.output_type = absl::Substitute("std::array<float, $0>",
                                                 model_ir_.num_output_classes);
        }
      } else {
        if (model_ir_.num_output_classes == 1) {
          cpp_ir_.output_type = "float";
        } else {
          cpp_ir_.output_type = absl::StrCat("std::array<float, ",
                                             model_ir_.num_output_classes, ">");
        }
        // Apply continuous activation functions.
        if (model_ir_.activation == ModelIR::Activation::kSigmoid) {
          cpp_ir_.includes.insert("<cmath>");
          cpp_ir_.activation_statement =
              "// Sigmoid\n  return 1.f / (1.f + std::exp(-accumulator));\n";
        } else if (model_ir_.activation == ModelIR::Activation::kSoftmax) {
          cpp_ir_.includes.insert("<cmath>");
          cpp_ir_.includes.insert("<algorithm>");
          cpp_ir_.activation_statement = absl::Substitute(
              R"(  std::array<float, $0> probas;
  const float max_logit = *std::max_element(accumulator.begin(), accumulator.end());
  float sum_exps = 0.f;
  for (int i = 0; i < $0; ++i) { 
    probas[i] = std::exp(accumulator[i] - max_logit); 
    sum_exps += probas[i];
  }
  for (int i = 0; i < $0; ++i) { 
    probas[i] /= sum_exps; 
  }
  return probas;)",
              model_ir_.num_output_classes);

        } else {
          // ModelIR::Activation::kEquality
          cpp_ir_.activation_statement = "return accumulator;";
        }
      }
      break;

    default:
      return absl::InvalidArgumentError("Unknown classification output type.");
  }
  return absl::OkStatus();
}

absl::Status CppTargetLowering::LowerNodes() {
  cpp_ir_.nodes.reserve(model_ir_.nodes.size());
  // This value is used as a marker in the feature index field to indicate an
  // oblique split. It is chosen to be the largest representable value for the
  // feature index type.
  const uint32_t oblique_feature_idx = NumBytesToMaxUnsignedValue(
      MaxUnsignedValueToNumBytes(static_cast<int64_t>(model_ir_.num_features) +
                                 kReservedFeatureIndexes));
  cpp_ir_.oblique_feature_idx = oblique_feature_idx;

  for (NodeIdx i = 0; i < model_ir_.nodes.size(); ++i) {
    const auto& ir_node = model_ir_.nodes[i];
    CppNode cpp_node;
    cpp_node.is_leaf = (ir_node.type == Node::Type::kLeaf);
    // The True branch is implicitly +1. The False branch jumps over the True
    // branch's subtree.
    cpp_node.jump_offset_false = ir_node.next_pos_node_idx;

    if (cpp_node.is_leaf) {
      RETURN_IF_ERROR(LowerLeafNode(ir_node, &cpp_node));
    } else {
      RETURN_IF_ERROR(LowerConditionNode(ir_node, &cpp_node));
    }
    cpp_ir_.nodes.push_back(cpp_node);
  }

  // Count the number of condition types used.
  for (const auto& cond : cpp_ir_.used_condition_types) {
    cpp_ir_.num_used_conditions += cond;
  }

  return absl::OkStatus();
}

absl::Status CppTargetLowering::LowerLeafNode(const Node& ir_node,
                                              CppNode* cpp_node) {
  if (model_ir_.leaf_value_dims == 1) {
    // Scalar Leaf
    // Note: Using threshold_or_offset because builder.cc populated it here
    // instead of leaf_value
    ASSIGN_OR_RETURN(const std::string val_str,
                     FormatLiteral(ir_node.threshold_or_offset,
                                   /*is_float=*/!model_ir_.winner_takes_all));
    if (model_ir_.winner_takes_all) {
      if (model_ir_.num_output_classes == 1) {
        if (!IsZero(ir_node.threshold_or_offset)) {
          cpp_node->if_else_leaf = "accumulator++;";
        }
      } else {
        cpp_node->if_else_leaf =
            absl::Substitute("accumulator[$0]++;", val_str);
      }
    } else {
      if (!IsZero(ir_node.threshold_or_offset)) {
        if (model_ir_.num_output_classes == 1) {
          cpp_node->if_else_leaf =
              absl::Substitute("accumulator += $0;", val_str);
        } else {
          const int32_t accumulator_idx =
              ir_node.tree_idx % model_ir_.num_output_classes;
          cpp_node->if_else_leaf = absl::Substitute("accumulator[$0] += $1;",
                                                    accumulator_idx, val_str);
        }
      }
    }
    cpp_node->routing_def = absl::Substitute("{.leaf={.val=$0}}", val_str);
  } else {
    // Vector Leaf (Multi-class Classification)
    const int offset = AsInt(ir_node.threshold_or_offset);

    std::vector<std::string> unrolled;
    unrolled.reserve(model_ir_.num_output_classes);
    for (int j = 0; j < model_ir_.num_output_classes; ++j) {
      const auto& val = model_ir_.leaf_value_bank[offset + j];
      if (!IsZero(val)) {
        ASSIGN_OR_RETURN(const std::string val_str,
                         FormatLiteral(val, !model_ir_.winner_takes_all));
        unrolled.push_back(
            absl::Substitute("accumulator[$0] += $1;", j, val_str));
      }
    }
    cpp_node->if_else_leaf = absl::StrJoin(unrolled, "\n");

    // Routing optimizes binary size by dividing the offset by the
    // dimension.
    const int encoded_leaf_value = offset / model_ir_.num_output_classes;
    cpp_node->routing_def =
        absl::Substitute("{.leaf={.val=$0}}", encoded_leaf_value);
  }
  return absl::OkStatus();
}

absl::Status CppTargetLowering::LowerConditionNode(const Node& ir_node,
                                                   CppNode* cpp_node) {
  STATUS_CHECK_GE(ir_node.feature_idx, 0);

  bool is_float = true;
  std::string var_name = "";

  FeatureInfo feat;

  // GUARD: If it's Oblique, feature_idx is a bank offset, NOT a dense index!
  if (ir_node.condition_type != ConditionType::OBLIQUE_CONDITION) {
    feat = model_ir_.features[ir_node.feature_idx];
    is_float = feat.is_float;
    var_name =
        absl::StrCat("instance.", feat_idx_to_var_name_[ir_node.feature_idx]);
  }

  switch (ir_node.condition_type) {
    case ConditionType::HIGHER_CONDITION: {
      cpp_ir_.used_condition_types[static_cast<int>(
          ConditionType::HIGHER_CONDITION)] = true;
      DoubleOrInt64 val = ir_node.threshold_or_offset;
      STATUS_CHECK(IsDouble(val));
      // For if-else, the legacy implementation always uses
      // floating-points.
      cpp_node->if_else_condition =
          absl::Substitute("$0 >= $1", var_name, AsDouble(val));

      // For routing, convert thresholds to the nearest integer, rounded up.
      if (!is_float) {
        val = static_cast<int64_t>(std::ceil(AsDouble(val)));
      }
      ASSIGN_OR_RETURN(const std::string routing_thresh_str,
                       FormatLiteral(val, is_float));

      cpp_node->routing_def = absl::Substitute(
          "{.pos=$0,.cond={.feat=$1,.thr=$2}}", cpp_node->jump_offset_false,
          ir_node.feature_idx, routing_thresh_str);
      break;
    }

    case ConditionType::CONTAINS_CONDITION_BUFFER_BITMAP: {
      cpp_ir_.used_condition_types[static_cast<int>(
          ConditionType::CONTAINS_CONDITION_BUFFER_BITMAP)] = true;
      const int offset = AsInt(ir_node.threshold_or_offset);

      if (options_.algorithm() == proto::Algorithm::IF_ELSE) {
        ASSIGN_OR_RETURN(cpp_node->if_else_condition,
                         FormatContainsCondition(ir_node, feat, var_name));
      }

      // Routing format is straightforward: point to the start index of the
      // boolean slice
      cpp_node->routing_def = absl::Substitute(
          "{.pos=$0,.cond={.feat=$1,.cat=$2}}", cpp_node->jump_offset_false,
          ir_node.feature_idx, offset);
      break;
    }

    case ConditionType::OBLIQUE_CONDITION: {
      cpp_ir_.used_condition_types[static_cast<int>(
          ConditionType::OBLIQUE_CONDITION)] = true;
      // Original cc_embed.cc explicitly does not support IF_ELSE for oblique
      // splits.
      if (options_.algorithm() == proto::Algorithm::IF_ELSE) {
        return absl::InvalidArgumentError(
            "Oblique conditions are not supported in IF_ELSE algorithm. "
            "Please use ROUTING.");
      }
      // Your builder stored the oblique bank index directly in feature_idx
      const int oblique_bank_idx = ir_node.feature_idx;

      cpp_node->routing_def = absl::Substitute(
          "{.pos=$0,.cond={.feat=$1,.obl=$2}}", cpp_node->jump_offset_false,
          cpp_ir_.oblique_feature_idx, oblique_bank_idx);
      break;
    }

    case ConditionType::TRUE_CONDITION: {
      cpp_ir_.used_condition_types[static_cast<int>(
          ConditionType::TRUE_CONDITION)] = true;
      // Evaluate the boolean feature directly
      cpp_node->if_else_condition = var_name;

      // In CC_embed routing, true conditions are generally treated as
      // numerical thresholds >= 0.5f if they must be mapped to the routing
      // engine constraints.
      cpp_node->routing_def =
          absl::Substitute("{.pos=$0,.cond={.feat=$1,.thr=0.5f}}",
                           cpp_node->jump_offset_false, ir_node.feature_idx);
      break;
    }

    default:
      return absl::InternalError(
          "Unknown ConditionType encountered in LowerNodes.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> CppTargetLowering::FormatContainsCondition(
    const Node& ir_node, const FeatureInfo& feat, const std::string& var_name) {
  const int offset = AsInt(ir_node.threshold_or_offset);

  // Reconstruct the actual categorical subset from the dense boolean bank.
  int num_unique = 0;
  if (feat.type == FeatureInfo::Type::kCategorical) {
    num_unique = feat.vocabulary.size();
  } else if (feat.type == FeatureInfo::Type::kIntegerizedCategorical) {
    STATUS_CHECK(feat.maximum_value.has_value());
    num_unique = *feat.maximum_value + 1;
  }

  std::vector<int32_t> items;
  for (int j = 0; j < num_unique; ++j) {
    // Extract the True values from the bank segment.
    if (model_ir_.bitset_bank.at(offset + j)) items.push_back(j);
  }

  // Lambda to resolve string categorical items vs integerized items.
  auto get_element_str = [&](int32_t element) {
    if (feat.type == FeatureInfo::Type::kIntegerizedCategorical) {
      return std::to_string(element);
    } else {
      if (element == 0)
        return absl::StrCat(feat_idx_to_enum_name_[ir_node.feature_idx],
                            "::kOutOfVocabulary");
      return absl::StrCat(
          feat_idx_to_enum_name_[ir_node.feature_idx], "::k",
          StringToStructSymbol(feat.vocabulary[element], false));
    }
  };

  constexpr int kSmallSetThreshold = 8;
  if (items.size() < kSmallSetThreshold) {
    // Small Sets: Unroll directly with ||
    std::vector<std::string> checks;
    checks.reserve(items.size());
    for (auto item : items) {
      checks.push_back(
          absl::Substitute("$0 == $1", var_name, get_element_str(item)));
    }
    // Indentation matching cc_embed.cc
    return absl::StrJoin(checks, " ||\n    ");
  } else {
    // Large Sets: Use array and binary_search
    cpp_ir_.includes.insert("<array>");
    cpp_ir_.includes.insert("<algorithm>");

    std::vector<std::string> str_elements;
    str_elements.reserve(items.size());
    for (auto item : items) str_elements.push_back(get_element_str(item));
    const std::string mask = absl::StrJoin(str_elements, ", ");

    const std::string array_type =
        (feat.type == FeatureInfo::Type::kIntegerizedCategorical)
            ? "Instance::IntegerizedCategorical"
            : feat_idx_to_enum_name_[ir_node.feature_idx];

    // Emits standard C++17 `if (init; condition)` statement syntax
    return absl::Substitute(
        "std::array<$1,$0> mask = {$2};\n"
        "std::binary_search(mask.begin(), mask.end(),  $3)",
        items.size(), array_type, mask, var_name);
  }
}

absl::Status CppTargetLowering::LowerRoutingData() {
  cpp_ir_.root_deltas_content =
      absl::StrJoin(model_ir_.tree_start_offsets, ",");
  const NodeIdx max_root_delta = model_ir_.tree_start_offsets.empty()
                                     ? 0
                                     : model_ir_.tree_start_offsets.back();
  ASSIGN_OR_RETURN(cpp_ir_.root_deltas_type,
                   StorageToCppType(MaxUnsignedValueToNumBytes(max_root_delta),
                                    false, false));

  if (!model_ir_.bitset_bank.empty()) {
    // Determine the size of the bitset and format the string.
    cpp_ir_.categorical_bank_size = model_ir_.bitset_bank.size();
    cpp_ir_.includes.insert("<bitset>");

    // In YDF, the bitset is printed in reverse order as a string of '0's and
    // '1's.
    std::string bitset_str;
    bitset_str.reserve(model_ir_.bitset_bank.size());
    for (auto it = model_ir_.bitset_bank.rbegin();
         it != model_ir_.bitset_bank.rend(); ++it) {
      bitset_str.push_back(*it ? '1' : '0');
    }
    cpp_ir_.categorical_bank_content = bitset_str;

    // Type used to index into the categorical bank
    ASSIGN_OR_RETURN(cpp_ir_.cat_idx_type,
                     StorageToCppType(MaxUnsignedValueToNumBytes(
                                          model_ir_.bitset_bank.size()),
                                      false, false));
  }
  ASSIGN_OR_RETURN(
      cpp_ir_.pos_type,
      StorageToCppType(MaxUnsignedValueToNumBytes(model_ir_.nodes.size()),
                       false, false));

  if (!model_ir_.oblique_weights.empty()) {
    cpp_ir_.oblique_weights_type = "float";
    cpp_ir_.oblique_weights_content =
        absl::StrJoin(model_ir_.oblique_weights, ",");

    // Oblique Features (Integer)
    int max_val = 0;
    for (int f : model_ir_.oblique_features) {
      if (f > max_val) max_val = f;
    }
    ASSIGN_OR_RETURN(
        cpp_ir_.oblique_features_type,
        StorageToCppType(MaxUnsignedValueToNumBytes(max_val), false, false));
    cpp_ir_.oblique_features_content =
        absl::StrJoin(model_ir_.oblique_features, ",");

    // Index into Oblique Bank
    if (model_ir_.oblique_weights.size() >
        std::numeric_limits<uint32_t>::max()) {
      return absl::InternalError(
          "Oblique weights bank size exceeds 32-bit limit.");
    }
    ASSIGN_OR_RETURN(cpp_ir_.obl_idx_type,
                     StorageToCppType(MaxUnsignedValueToNumBytes(
                                          model_ir_.oblique_weights.size()),
                                      false, false));
  }

  ASSIGN_OR_RETURN(
      cpp_ir_.feature_idx_type,
      StorageToCppType(MaxUnsignedValueToNumBytes(model_ir_.features.size()),
                       false, false));
  cpp_ir_.leaf_val_type = model_ir_.winner_takes_all ? "int32_t" : "float";
  STATUS_CHECK_GE(model_ir_.leaf_value_dims, 1);
  if (model_ir_.leaf_value_dims == 1) {
    // The leaf value is stored in the node struct.
    cpp_ir_.leaf_val_type = DTypeToCCType(model_ir_.leaf_value_dtype);
  } else {
    // The leaf values are stored in a separate buffer. The node struct contains
    // an index to this buffer.
    cpp_ir_.leaf_val_type = UnsignedInteger(MaxUnsignedValueToNumBytes(
        model_ir_.num_leaves / model_ir_.leaf_value_dims));
  }

  for (const auto& cond_type : model_ir_.active_condition_types) {
    if (cond_type == ConditionType::HIGHER_CONDITION) {
      STATUS_CHECK(!cpp_ir_.numerical_type.empty());
      cpp_ir_.routing_condition_eval_blocks.push_back(
          std::make_pair(cond_type, absl::Substitute(
                                        R"(        $0 numerical_feature;
        std::memcpy(&numerical_feature, raw_instance + node->cond.feat * sizeof($0), sizeof($0));
        eval = numerical_feature >= node->cond.thr;)",
                                        cpp_ir_.numerical_type)));
    } else if (cond_type == ConditionType::OBLIQUE_CONDITION) {
      STATUS_CHECK(!cpp_ir_.oblique_features_type.empty());
      STATUS_CHECK(!cpp_ir_.numerical_type.empty());
      cpp_ir_.routing_condition_eval_blocks.insert(
          cpp_ir_.routing_condition_eval_blocks.begin(),
          std::make_pair(
              cond_type,
              absl::Substitute(
                  R"(        const $0 num_projs = oblique_features[node->cond.obl];
        float obl_acc = -oblique_weights[node->cond.obl];
        for ($0 proj_idx=0; proj_idx<num_projs; proj_idx++){
          const auto off = node->cond.obl + proj_idx + 1;
          $1 numerical_feature;
          std::memcpy(&numerical_feature, raw_instance + oblique_features[off] * sizeof($1), sizeof($1));
          obl_acc += numerical_feature * oblique_weights[off];
        }
        eval = obl_acc >= 0;)",
                  cpp_ir_.oblique_features_type, cpp_ir_.numerical_type)));
    } else if (cond_type == ConditionType::CONTAINS_CONDITION_BUFFER_BITMAP) {
      STATUS_CHECK(!cpp_ir_.unsigned_integer_type.empty());
      cpp_ir_.routing_condition_eval_blocks.push_back(
          std::make_pair(cond_type, absl::Substitute(
                                        R"(        $0 categorical_feature;
        std::memcpy(&categorical_feature, raw_instance + node->cond.feat * sizeof($0), sizeof($0));
        eval = categorical_bank[categorical_feature + node->cond.cat];)",
                                        cpp_ir_.unsigned_integer_type)));
    } else if (cond_type == ConditionType::TRUE_CONDITION) {
      // Find what type categorical features use
      cpp_ir_.routing_condition_eval_blocks.push_back(
          std::make_pair(cond_type,
                         R"(        bool boolean_feature;
        std::memcpy(&boolean_feature, raw_instance + node->cond.feat * sizeof(bool), sizeof(bool));
        eval = boolean_feature;)"));
    }
  }

  if (!model_ir_.leaf_value_bank.empty()) {
    cpp_ir_.leaf_value_bank_content =
        absl::StrJoin(model_ir_.leaf_value_bank, ",");
  }

  return absl::OkStatus();
}

// --- Helpers ---

absl::StatusOr<std::string> CppTargetLowering::FormatLiteral(
    const DoubleOrInt64& val, const bool is_float) const {
  if (is_float) {
    if (IsDouble(val)) {
      // Ensure it always prints with a decimal to be treated as a float in C++
      std::string s = absl::StrCat(AsDouble(val));
      // TODO: Update this and all golden tests.
      // if (!absl::StrContains(s, ".") && !absl::StrContains(s, "e")) {
      //   s += ".0";
      // }
      return s;
    } else {
      return absl::InternalError(
          "Expected float value but found integer in variant.");
    }
  } else {
    if (IsInt(val)) {
      return absl::StrCat(AsInt(val));
    } else {
      return absl::InternalError(
          "Expected integer value but found float in variant.");
    }
  }
}

absl::StatusOr<std::string> CppTargetLowering::StorageToCppType(
    const int bytes, const bool is_float, const bool is_signed) const {
  if (is_float) {
    if (bytes == 4) return "float";
    if (bytes == 8) return "double";
  } else {
    if (is_signed) {
      if (bytes == 1) return "int8_t";
      if (bytes == 2) return "int16_t";
      if (bytes == 4) return "int32_t";
      if (bytes == 8) return "int64_t";
    } else {
      if (bytes == 1) return "uint8_t";
      if (bytes == 2) return "uint16_t";
      if (bytes == 4) return "uint32_t";
      if (bytes == 8) return "uint64_t";
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported storage type: bytes=", bytes,
                   " float=", is_float, " signed=", is_signed));
}

}  // namespace yggdrasil_decision_forests::serving::embed::internal
