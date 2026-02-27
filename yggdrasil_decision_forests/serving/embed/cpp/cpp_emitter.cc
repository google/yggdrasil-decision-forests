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

#include "yggdrasil_decision_forests/serving/embed/cpp/cpp_emitter.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/cpp_ir.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/serving/embed/ir/model_ir.h"
#include "yggdrasil_decision_forests/serving/embed/utils.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

absl::StatusOr<std::string> CppEmitter::Emit(const CppIR& ir,
                                             const proto::Options& options) {
  CppEmitter emitter(ir, options);
  return emitter.Run();
}

absl::StatusOr<std::string> CppEmitter::Run() const {
  std::string out;

  absl::SubstituteAndAppend(&out, "#ifndef $0\n#define $0\n\n",
                            ir_.header_guard);

  EmitIncludes(&out);

  absl::SubstituteAndAppend(&out, "\nnamespace $0 {\n\n", ir_.namespace_name);

  EmitEnums(&out);
  EmitInstanceStruct(&out);

  if (options_.algorithm() == proto::Algorithm::ROUTING) {
    RETURN_IF_ERROR(EmitRoutingData(&out));
    RETURN_IF_ERROR(EmitPredictUnsafeRouting(&out));
  } else if (options_.algorithm() == proto::Algorithm::IF_ELSE) {
    RETURN_IF_ERROR(EmitPredictUnsafeIfElse(&out));
  } else {
    return absl::InvalidArgumentError("Unknown Algorithm");
  }

  if (ir_.needs_predict_unsafe) {
    EmitPredictSanitized(&out);
  }

  absl::SubstituteAndAppend(&out, "}  // namespace $0\n#endif\n",
                            ir_.namespace_name);

  return out;
}

void CppEmitter::EmitIncludes(std::string* out) const {
  std::vector<std::string> sorted_includes(ir_.includes.begin(),
                                           ir_.includes.end());
  std::sort(sorted_includes.begin(), sorted_includes.end());
  for (const std::string& inc : sorted_includes) {
    absl::StrAppend(out, "#include ", inc, "\n");
  }
}

void CppEmitter::EmitEnums(std::string* out) const {
  for (const auto& feature_enum : ir_.enums) {
    absl::SubstituteAndAppend(out, "enum class $0 : $1 {\n", feature_enum.name,
                              ir_.types.categorical_feature);
    for (const auto& item : feature_enum.items) {
      absl::SubstituteAndAppend(out, "  $0 = $1,\n", item.name, item.value);
    }
    absl::StrAppend(out, "};\n\n");

    if (feature_enum.generate_from_string_method) {
      absl::SubstituteAndAppend(out, R"(
$0 $0FromString(const std::string_view name) {
  using F = $0;
  static const std::unordered_map<std::string_view, $0>
      k$0Map = {
)",
                                feature_enum.name);  // $0

      for (auto& item : feature_enum.items) {
        // Skip OOV item with index 0 (value 0)
        if (item.value == 0) {
          continue;
        }
        absl::SubstituteAndAppend(out, "          {$0, F::$1},\n",
                                  QuoteString(item.original), item.name);
      }
      absl::SubstituteAndAppend(out, R"(      };
  auto it = k$0Map.find(name);
  if (it == k$0Map.end()) {
    return F::kOutOfVocabulary;
  }
  return it->second;
}
)",
                                feature_enum.name);  // $0
    }
  }
}

void CppEmitter::EmitInstanceStruct(std::string* out) const {
  absl::SubstituteAndAppend(out, "constexpr const int kNumFeatures = $0;\n",
                            ir_.features.size());
  absl::SubstituteAndAppend(out, "constexpr const $1 kNumTrees = $0;\n\n",
                            ir_.num_trees, ir_.types.num_trees);

  absl::StrAppend(out, "struct Instance {\n");
  // 4. Constants and Typedefs
  if (!ir_.types.numerical_feature.empty()) {
    absl::SubstituteAndAppend(out, "  typedef $0 Numerical;\n",
                              ir_.types.numerical_feature);
  }
  if (ir_.has_integerized_categorical_features) {
    absl::SubstituteAndAppend(
        out, R"(  // Integerized categorical features require special care.
  // The PredictUnsafe() function has undefined behavior if an integerized
  // categorical feature value is outside the range [0, max_value], where
  // max_value is the maximum value seen during training. To ensure
  // safe predictions, use the Predict() function, which automatically
  // sanitizes the input values.
  typedef $0 IntegerizedCategorical;
)",
        ir_.types.integerized_categorical_feature);
  }
  absl::StrAppend(out, "\n");

  for (const auto& feat : ir_.features) {
    absl::SubstituteAndAppend(out, "  $0 $1;\n", feat.cpp_type, feat.var_name);
  }
  absl::StrAppend(out, "};\n\n");
}

absl::Status CppEmitter::EmitRoutingData(std::string* out) const {
  // 1. Define Node Struct
  absl::StrAppend(out, "struct __attribute__((packed)) Node {\n");
  absl::SubstituteAndAppend(out, "  $0 pos = 0;\n", ir_.types.pos);
  absl::StrAppend(out, "  union {\n");
  absl::StrAppend(out, "    struct __attribute__((packed)) {\n");
  absl::SubstituteAndAppend(out, "      $0 feat;\n", ir_.types.feature_idx);
  absl::StrAppend(out, "      union {\n");
  if (!ir_.types.numerical_feature.empty()) {
    absl::SubstituteAndAppend(out, "        $0 thr;\n",
                              ir_.types.numerical_feature);
  }
  if (!ir_.categorical_bank_content.empty()) {
    absl::SubstituteAndAppend(out, "        $0 cat;\n", ir_.types.cat_bank_idx);
  }
  if (!ir_.oblique_features_content.empty()) {
    absl::SubstituteAndAppend(out, "        $0 obl;\n", ir_.types.obl_bank_idx);
  }
  absl::StrAppend(out, "      };\n");
  absl::StrAppend(out, "    } cond;\n");
  absl::StrAppend(out, "    struct __attribute__((packed)) {\n");
  absl::SubstituteAndAppend(out, "      $0 val;\n", ir_.types.leaf_value);
  absl::StrAppend(out, "    } leaf;\n");
  absl::StrAppend(out, "  };\n");
  absl::StrAppend(out, "};\n");

  // 2. Nodes Array
  absl::StrAppend(out, "static const Node nodes[] = {\n");
  for (const auto& node : ir_.nodes) {
    absl::StrAppend(out, node.routing_def, ",\n");
  }
  absl::StrAppend(out, "};\n\n");

  // 3. Auxiliary Arrays
  if (ir_.num_used_conditions > 1) {
    STATUS_CHECK(!ir_.condition_types_content.empty());
    absl::SubstituteAndAppend(
        out, "static const $0 condition_types[] = {$1};\n\n",
        ir_.types.condition_types, ir_.condition_types_content);
  }

  if (!ir_.root_deltas_content.empty()) {
    absl::SubstituteAndAppend(out,
                              "\nstatic const $0 root_deltas[] = {$1};\n\n",
                              ir_.types.root_deltas, ir_.root_deltas_content);
  }

  if (ir_.categorical_bank_size > 0) {
    absl::SubstituteAndAppend(
        out, "\nstatic const std::bitset<$0> categorical_bank {\"$1\"};\n\n",
        ir_.categorical_bank_size, ir_.categorical_bank_content);
  }

  if (!ir_.leaf_value_bank_content.empty()) {
    absl::SubstituteAndAppend(
        out, "\nstatic const $0 leaf_value_bank[] = {$1};\n\n",
        ir_.types.leaf_value_bank, ir_.leaf_value_bank_content);
  }

  if (!ir_.oblique_weights_content.empty()) {
    absl::SubstituteAndAppend(
        out, "static const $0 oblique_weights[] = {$1};\n\n",
        ir_.types.oblique_weights, ir_.oblique_weights_content);
    absl::SubstituteAndAppend(
        out, "static const $0 oblique_features[] = {$1};\n\n",
        ir_.types.oblique_features, ir_.oblique_features_content);
  }
  return absl::OkStatus();
}

absl::Status CppEmitter::EmitPredictUnsafeIfElse(std::string* out) const {
  // The signature changes slightly if needs_predict_unsafe is false,
  // but let's assume it always outputs PredictUnsafe for consistency,
  // or alias PredictUnsafe to Predict if no sanitization is needed.
  if (ir_.needs_predict_unsafe) {
    absl::SubstituteAndAppend(
        out, R"(// Predicts on an instance without any safety checks.
//
// The caller must ensure that the instance meets the following conditions:
// - Numerical features must not be NaN.
// - Integerized categorical features must be within the range [0, max_value],
//   where max_value is the maximum value observed during training.
//
// Failure to meet these conditions may result in undefined behavior.
//
// It is recommended to use `Predict()` instead, unless the instance has
// already been sanitized.
//
// This function is called by `Predict()`.
inline $0 PredictUnsafe(const Instance& instance) {
)",
        ir_.full_output_type);
  } else {
    absl::SubstituteAndAppend(out,
                              "inline $0 Predict(const Instance& instance) {\n",
                              ir_.full_output_type);
  }

  // Accumulator init
  absl::SubstituteAndAppend(out, "  $0\n\n", ir_.accumulator_init_statement);

  // Iterate over trees
  int current_node_idx = 0;
  for (int t = 0; t < ir_.num_trees; ++t) {
    absl::SubstituteAndAppend(out, "  // Tree #$0\n", t);
    ASSIGN_OR_RETURN(current_node_idx,
                     PrintIfElseNode(current_node_idx, 1, out));
    absl::StrAppend(out, "\n");
  }

  // Activation & Return
  absl::SubstituteAndAppend(out, "  $0\n}\n\n", ir_.activation_statement);

  return absl::OkStatus();
}

absl::StatusOr<int> CppEmitter::PrintIfElseNode(int node_idx, int depth,
                                                std::string* out) const {
  const std::string indent(depth * 2, ' ');
  const auto& node = ir_.nodes[node_idx];

  if (node.is_leaf) {
    if (!node.if_else_leaf.empty()) {
      const auto leaf_statement = absl::StrReplaceAll(
          node.if_else_leaf,
          {{"\naccumulator[", absl::StrCat("\n", indent, "accumulator[")}});
      absl::SubstituteAndAppend(out, "$0$1\n", indent, leaf_statement);
    }
    return node_idx + 1;  // Return index of next element in the flat array
  }

  // It's a condition.
  // Format the condition with the indent.
  std::string condition = absl::StrReplaceAll(
      node.if_else_condition, {{"||\n", absl::StrCat("||\n", indent)}});
  condition = absl::StrReplaceAll(
      condition, {{"\nstd::binary_search",
                   absl::StrCat("\n", indent, "    std::binary_search")}});
  absl::SubstituteAndAppend(out, "$0if ($1) {\n", indent, condition);

  // Recursively process the positive branch.
  ASSIGN_OR_RETURN(
      const int next_idx,
      PrintIfElseNode(node_idx + 1 + node.jump_offset_false, depth + 1, out));

  absl::SubstituteAndAppend(out, "$0} else {\n", indent);

  // Recursively process negative branch
  ASSIGN_OR_RETURN(const int pos_branch_node_idx,
                   PrintIfElseNode(node_idx + 1, depth + 1, out));
  STATUS_CHECK_EQ(pos_branch_node_idx, node_idx + 1 + node.jump_offset_false);

  absl::SubstituteAndAppend(out, "$0}\n", indent);
  return next_idx;
}

absl::Status CppEmitter::EmitPredictUnsafeRouting(std::string* out) const {
  if (ir_.needs_predict_unsafe) {
    absl::SubstituteAndAppend(
        out, R"(// Predicts on an instance without any safety checks.
//
// The caller must ensure that the instance meets the following conditions:
// - Numerical features must not be NaN.
// - Integerized categorical features must be within the range [0, max_value],
//   where max_value is the maximum value observed during training.
//
// Failure to meet these conditions may result in undefined behavior.
//
// It is recommended to use `Predict()` instead, unless the instance has
// already been sanitized.
//
// This function is called by `Predict()`.
inline $0 PredictUnsafe(const Instance& instance) {
)",
        ir_.full_output_type);
  } else {
    absl::SubstituteAndAppend(out,
                              "inline $0 Predict(const Instance& instance) {\n",
                              ir_.full_output_type);
  }

  // Accumulator init
  absl::SubstituteAndAppend(out, "  $0\n\n", ir_.accumulator_init_statement);

  // Loop vars
  absl::StrAppend(out, "  const Node* root = nodes;\n");
  absl::StrAppend(out, "  const Node* node;\n");
  absl::StrAppend(out,
                  "  const char* raw_instance = (const char*)(&instance);\n");
  absl::SubstituteAndAppend(out, "  $0 eval;\n", ir_.types.eval);

  // Tree loop
  absl::SubstituteAndAppend(out,
                            "  for ($0 tree_idx = 0; tree_idx != kNumTrees; "
                            "tree_idx++) {\n",
                            ir_.types.num_trees);
  absl::StrAppend(out, "    node = root;\n");
  absl::StrAppend(out, "    while(node->pos) {\n");

  if (ir_.routing_condition_eval_blocks.empty()) {
    // The model does not contain any condition.
    absl::StrAppend(out, "      assert(false);\n");
  }
  if (ir_.routing_condition_eval_blocks.size() == 1) {
    absl::StrAppend(out, ir_.routing_condition_eval_blocks.begin()->second,
                    "\n\n");
  }
  if (ir_.routing_condition_eval_blocks.size() > 1) {
    // Condition evaluation blocks
    for (size_t i = 0; i < ir_.routing_condition_eval_blocks.size(); ++i) {
      const auto& [cond_type, eval_block] =
          ir_.routing_condition_eval_blocks[i];
      if (i == 0) {
        if (cond_type == ConditionType::OBLIQUE_CONDITION) {
          absl::SubstituteAndAppend(out, "      if (node->cond.feat == $0) {\n",
                                    ir_.oblique_feature_idx);
        } else {
          absl::SubstituteAndAppend(
              out, "      if (condition_types[node->cond.feat] == $0) {\n",
              static_cast<int>(cond_type));
        }
      } else {
        STATUS_CHECK_NE(cond_type, ConditionType::OBLIQUE_CONDITION);
        absl::SubstituteAndAppend(
            out, "      } else if (condition_types[node->cond.feat] == $0) {\n",
            static_cast<int>(cond_type));
      }
      absl::StrAppend(out, eval_block, "\n");
    }
    absl::StrAppend(out, "      } else {\n");
    absl::StrAppend(out, "        assert(false);\n");
    absl::StrAppend(out, "      }\n");
  }

  absl::StrAppend(out, "      node += (node->pos & -eval) + 1;\n");
  absl::StrAppend(out, "    }\n");

  absl::StrAppend(out, "    ", ir_.accumulator_sum_statement, "\n");

  absl::StrAppend(out, "    root += root_deltas[tree_idx];\n");
  absl::StrAppend(out, "  }\n\n");

  // Activation & Return
  absl::SubstituteAndAppend(out, "  $0\n}\n\n", ir_.activation_statement);
  return absl::OkStatus();
}

void CppEmitter::EmitPredictSanitized(std::string* out) const {
  absl::SubstituteAndAppend(out,
                            R"(// Prediction function.
//
// Sanitizes the given instance, then calls `PredictUnsafe()`.
inline $0 Predict(Instance instance) {
)",
                            ir_.full_output_type);

  for (const auto& feat : ir_.features) {
    for (const auto& sanitization : feat.na_sanitization) {
      absl::StrAppend(out, "  ", sanitization, "\n");
    }
  }

  absl::StrAppend(out, "\n  return PredictUnsafe(instance);\n}\n\n");
}

}  // namespace yggdrasil_decision_forests::serving::embed::internal
