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

#include "yggdrasil_decision_forests/serving/embed/c/c_emitter.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/serving/embed/c/c_ir.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/serving/embed/ir/model_ir.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

absl::StatusOr<absl::node_hash_map<std::string, std::string>> CEmitter::Emit(
    const CIR& ir, const proto::Options& options) {
  CEmitter emitter(ir, options);
  return emitter.Run();
}

absl::StatusOr<absl::node_hash_map<std::string, std::string>> CEmitter::Run()
    const {
  absl::node_hash_map<std::string, std::string> files;

  ASSIGN_OR_RETURN(files[ir_.header_filename], CreateSTBHeader());
  ASSIGN_OR_RETURN(files[ir_.source_filename], CreatePseudoSource());
  return files;
}
absl::StatusOr<std::string> CEmitter::CreatePseudoSource() const {
  return absl::Substitute(R"(#define $0
// TODO: Replace with the header location
#include "$1"
)",
                          ir_.impl_guard, ir_.header_filename);
}

absl::StatusOr<std::string> CEmitter::CreateSTBHeader() const {
  std::string out;
  EmitActualHeader(&out);
  RETURN_IF_ERROR(EmitImplementation(&out));
  return out;
}
// Creates the header part of the STB-style file.
void CEmitter::EmitActualHeader(std::string* out) const {
  absl::SubstituteAndAppend(out, R"(/*
   $0 - Single-Header C Library for YDF Model Prediction
   
   USAGE:
   
   1. In EXACTLY ONE .c file, define the implementation before including:
      #define $1
      #include "$0"

   2. In all other .c files, just include it normally:
      #include "$0"
*/

#ifndef $2
#define $2

)",
                            ir_.header_filename, ir_.impl_guard,
                            ir_.header_guard);

  EmitIncludes(ir_.header_includes, out);

  absl::StrAppend(out, R"(
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

)");

  EmitEnums(out);
  EmitInstanceStruct(out);

  std::string predict_signature = BuildPredictSignature();
  std::string output_parameter_comment = "";
  if (!ir_.output_parameter.empty()) {
    output_parameter_comment =
        "\n// The results are written into the provided `out` array.";
  }
  if (ir_.needs_predict_unsafe) {
    absl::SubstituteAndAppend(out, R"(// Prediction function.$2
//
// Sanitizes the given instance, then calls `$1PredictUnsafe()`.
$0;

)",
                              predict_signature, ir_.pseudo_namespace_name,
                              output_parameter_comment);
    std::string predict_unsafe_signature = BuildPredictUnsafeSignature();
    absl::SubstituteAndAppend(
        out,
        R"(// Predicts on an instance without any safety checks.$2
//
// The caller must ensure that the instance meets the following conditions:
// - Numerical features must not be NaN.
// - Integerized categorical features must be within the range [0, max_value],
//   where max_value is the maximum value observed during training.
//
// Failure to meet these conditions may result in undefined behavior.
//
// It is recommended to use `$1Predict()` instead, unless the instance has
// already been sanitized.
//
// This function is called by `$1Predict()`.
$0;
)",
        predict_unsafe_signature, ir_.pseudo_namespace_name,
        output_parameter_comment);
  } else {
    absl::SubstituteAndAppend(out, R"(// Prediction function.$1
$0;
)",
                              predict_signature, output_parameter_comment);
  }

  absl::SubstituteAndAppend(out, R"(
#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // $0

)",
                            ir_.header_guard);
}

// Creates the implementation part of the STB-style file.
absl::Status CEmitter::EmitImplementation(std::string* out) const {
  absl::SubstituteAndAppend(out,
                            R"(// This code is compiled ONLY when $0 is defined.
#ifdef $0

)",
                            ir_.impl_guard);

  EmitIncludes(ir_.impl_includes, out);

  absl::StrAppend(out, R"(
// Portable packing macros.
#if defined(__KERNEL__) || defined(MODULE)
    #define YDF_MODEL_PACK_PUSH
    #define YDF_MODEL_PACK_POP
    #define YDF_MODEL_PACKED_ATTR __packed
#elif defined(_MSC_VER)
    #define YDF_MODEL_PACK_PUSH __pragma(pack(push, 1))
    #define YDF_MODEL_PACK_POP __pragma(pack(pop))
    #define YDF_MODEL_PACKED_ATTR
#elif defined(__GNUC__) || defined(__clang__)
    #define YDF_MODEL_PACK_PUSH
    #define YDF_MODEL_PACK_POP
    #define YDF_MODEL_PACKED_ATTR __attribute__((packed))
#else
    #define YDF_MODEL_PACK_PUSH
    #define YDF_MODEL_PACK_POP
    #define YDF_MODEL_PACKED_ATTR
#endif

)");

  if (!ir_.categorical_bank_content.empty()) {
    absl::SubstituteAndAppend(
        out,
        R"(// Helper to test a bit in a byte array (for categorical byte arrays)
static inline bool $0BitTest(const $1* array, const $2 bit_index) {
  return array[bit_index / 8] & (1 << (bit_index % 8));
}

)",
        ir_.pseudo_namespace_name, ir_.types.categorical_bank,
        ir_.types.categorical_feature);
  }

  absl::SubstituteAndAppend(out, "static const $0 $1NumTrees = $2;\n\n",
                            ir_.types.num_trees, ir_.pseudo_namespace_name,
                            ir_.num_trees);

  RETURN_IF_ERROR(EmitRoutingData(out));
  RETURN_IF_ERROR(EmitPredictUnsafeRouting(out));

  if (ir_.needs_predict_unsafe) {
    EmitPredictSanitized(out);
  }

  absl::SubstituteAndAppend(out, "#endif  // $0\n", ir_.impl_guard);
  return absl::OkStatus();
}

void CEmitter::EmitIncludes(const absl::flat_hash_set<std::string>& includes,
                            std::string* out) const {
  std::vector<std::string> sorted_includes(includes.begin(), includes.end());
  absl::c_sort(sorted_includes);
  for (const std::string& inc : sorted_includes) {
    absl::StrAppend(out, "#include ", inc, "\n");
  }
}

void CEmitter::EmitEnums(std::string* out) const {
  for (const auto& feature_enum : ir_.enums) {
    absl::StrAppend(out, "typedef enum {\n");
    for (const auto& item : feature_enum.items) {
      absl::SubstituteAndAppend(out, "  $0 = $1,\n", item.name, item.value);
    }
    absl::SubstituteAndAppend(out, "} $0;\n", feature_enum.name);
    absl::SubstituteAndAppend(out, "typedef $0 $1;\n\n",
                              ir_.types.categorical_feature,
                              feature_enum.type_name);
  }
}

void CEmitter::EmitInstanceStruct(std::string* out) const {
  if (ir_.has_numercial_feature) {
    absl::SubstituteAndAppend(out, "typedef $0 $1Numerical;\n\n",
                              ir_.types.numerical_feature,
                              ir_.pseudo_namespace_name);
  }

  absl::StrAppend(out, "typedef struct {\n");
  for (const auto& feat : ir_.features) {
    absl::SubstituteAndAppend(out, "  $0 $1;\n", feat.c_type, feat.var_name);
  }
  absl::SubstituteAndAppend(out, "} $0Instance;\n\n",
                            ir_.pseudo_namespace_name);
}

absl::Status CEmitter::EmitRoutingData(std::string* out) const {
  absl::SubstituteAndAppend(out, R"(YDF_MODEL_PACK_PUSH
typedef struct YDF_MODEL_PACKED_ATTR {
  $0 pos;
  union {
    struct YDF_MODEL_PACKED_ATTR {
      $1 feat;
      union {
)",
                            ir_.types.pos, ir_.types.feature_idx);
  if (ir_.has_numercial_feature) {
    absl::SubstituteAndAppend(out, "        $0 thr;\n",
                              ir_.types.numerical_feature);
  }
  if (!ir_.categorical_bank_content.empty()) {
    absl::SubstituteAndAppend(out, "        $0 cat;\n", ir_.types.cat_bank_idx);
  }
  if (!ir_.oblique_weights_content.empty()) {
    absl::SubstituteAndAppend(out, "        $0 obl;\n", ir_.types.obl_bank_idx);
  }
  absl::SubstituteAndAppend(out, R"(      };
    } cond;
    struct YDF_MODEL_PACKED_ATTR {
      $0 val;
    } leaf;
  };
} $1Node;
YDF_MODEL_PACK_POP

)",
                            ir_.types.leaf_value, ir_.pseudo_namespace_name);

  absl::SubstituteAndAppend(out, "static const $0Node $0nodes[] = {\n",
                            ir_.pseudo_namespace_name);
  for (const auto& node_def : ir_.nodes) {
    absl::StrAppend(out, node_def, ",\n");
  }
  absl::StrAppend(out, "};\n\n");

  // 3. Auxiliary Arrays
  if (ir_.num_used_conditions > 1) {
    STATUS_CHECK(!ir_.condition_types_content.empty());
    absl::SubstituteAndAppend(
        out, "static const $0 $1condition_types[] = {$2};\n\n",
        ir_.types.condition_types, ir_.pseudo_namespace_name,
        ir_.condition_types_content);
  }

  if (!ir_.root_deltas_content.empty()) {
    absl::SubstituteAndAppend(out,
                              "\nstatic const $0 $1root_deltas[] = {$2};\n\n",
                              ir_.types.root_deltas, ir_.pseudo_namespace_name,
                              ir_.root_deltas_content);
  }

  if (!ir_.categorical_bank_content.empty()) {
    absl::SubstituteAndAppend(
        out, "\nstatic const $0 $1categorical_bank[] = {$2};\n\n",
        ir_.types.categorical_bank, ir_.pseudo_namespace_name,
        ir_.categorical_bank_content);
  }

  if (!ir_.leaf_value_bank_content.empty()) {
    absl::SubstituteAndAppend(
        out, "\nstatic const $0 $1leaf_value_bank[] = {$2};\n\n",
        ir_.types.leaf_value_bank, ir_.pseudo_namespace_name,
        ir_.leaf_value_bank_content);
  }

  if (!ir_.oblique_weights_content.empty()) {
    absl::SubstituteAndAppend(
        out, "static const $1 $0oblique_weights[] = {$2};\n\n",
        ir_.pseudo_namespace_name, ir_.types.oblique_weights,
        ir_.oblique_weights_content);
    absl::SubstituteAndAppend(
        out, "static const $1 $0oblique_features[] = {$2};\n\n",
        ir_.pseudo_namespace_name, ir_.types.oblique_features,
        ir_.oblique_features_content);
  }

  EmitFeatureOffsets(out);
  return absl::OkStatus();
}

void CEmitter::EmitFeatureOffsets(std::string* out) const {
  absl::SubstituteAndAppend(out, "static const size_t $0FeatureOffsets[] = {\n",
                            ir_.pseudo_namespace_name);
  for (const auto& feat : ir_.features) {
    absl::SubstituteAndAppend(out, "    offsetof($0Instance, $1),\n",
                              ir_.pseudo_namespace_name, feat.var_name);
  }
  absl::StrAppend(out, "};\n\n");
}

absl::Status CEmitter::EmitPredictUnsafeRouting(std::string* out) const {
  if (ir_.needs_predict_unsafe) {
    std::string predict_unsafe_signature = BuildPredictUnsafeSignature();
    absl::SubstituteAndAppend(out, "$0 {\n", predict_unsafe_signature);
  } else {
    std::string predict_signature = BuildPredictSignature();
    absl::SubstituteAndAppend(out, "$0 {\n", predict_signature);
  }

  // Accumulator init
  absl::SubstituteAndAppend(out, R"(  $0

  const $1Node* root = $1nodes;
  const $1Node* node;
  const char* raw_instance = (const char*)(instance);
  $2 eval;
  for ($3 tree_idx = 0; tree_idx != $1NumTrees; tree_idx++) {
    node = root;
    while(node->pos) {
)",
                            ir_.accumulator_init_statement,
                            ir_.pseudo_namespace_name, ir_.types.eval,
                            ir_.types.num_trees);

  if (ir_.routing_condition_eval_blocks.empty()) {
    // The model does not contain any condition.
    if (options_.c().linux_kernel_compatible()) {
      absl::StrAppend(out, "      BUG();\n");
    } else {
      absl::StrAppend(out, "      assert(false);\n");
    }
  }
  if (ir_.routing_condition_eval_blocks.size() == 1) {
    if (ir_.routing_condition_eval_blocks[0].first !=
        ConditionType::OBLIQUE_CONDITION) {
      absl::SubstituteAndAppend(
          out, "      const size_t off = $0FeatureOffsets[node->cond.feat];\n",
          ir_.pseudo_namespace_name);
    }
    absl::StrAppend(out, ir_.routing_condition_eval_blocks[0].second, "\n\n");
  }
  if (ir_.routing_condition_eval_blocks.size() > 1) {
    // Condition evaluation blocks
    for (size_t i = 0; i < ir_.routing_condition_eval_blocks.size(); ++i) {
      const auto& [cond_type, eval_block] =
          ir_.routing_condition_eval_blocks[i];
      std::string if_prefix = (i == 0) ? "      if" : "      } else if";
      if (cond_type == ConditionType::OBLIQUE_CONDITION) {
        absl::SubstituteAndAppend(out, "$0 (node->cond.feat == $1) {\n",
                                  if_prefix, ir_.oblique_feature_idx);
      } else {
        absl::SubstituteAndAppend(
            out, "$0 ($1condition_types[node->cond.feat] == $2) {\n", if_prefix,
            ir_.pseudo_namespace_name, static_cast<int>(cond_type));
        absl::SubstituteAndAppend(
            out,
            "        const size_t off = $0FeatureOffsets[node->cond.feat];\n",
            ir_.pseudo_namespace_name);
      }
      absl::StrAppend(out, eval_block, "\n");
    }
    absl::StrAppend(out, "      } else {\n");
    if (options_.c().linux_kernel_compatible()) {
      absl::StrAppend(out, "        BUG();\n");
    } else {
      absl::StrAppend(out, "        assert(false);\n");
    }
    absl::StrAppend(out, "      }\n");
  }

  absl::SubstituteAndAppend(out, R"(      node += (node->pos & -eval) + 1;
    }
    $0
    root += $1root_deltas[tree_idx];
  }

  $2
}

)",
                            ir_.accumulator_sum_statement,
                            ir_.pseudo_namespace_name,
                            ir_.activation_statement);
  return absl::OkStatus();
}

void CEmitter::EmitPredictSanitized(std::string* out) const {
  std::string predict_signature = BuildPredictSignature();
  absl::SubstituteAndAppend(out, "$0 {\n", predict_signature);
  absl::SubstituteAndAppend(out,
                            "  $0Instance sanitized_instance = *instance;\n",
                            ir_.pseudo_namespace_name);

  for (const auto& feat : ir_.features) {
    for (const auto& sanitization : feat.na_sanitization) {
      absl::StrAppend(out, "  ", sanitization, "\n");
    }
  }

  if (ir_.output_parameter.empty()) {
    absl::SubstituteAndAppend(
        out, "\n  return $0PredictUnsafe(&sanitized_instance);\n}\n\n",
        ir_.pseudo_namespace_name);
  } else {
    absl::SubstituteAndAppend(
        out, "\n  $0PredictUnsafe(&sanitized_instance, out);\n}\n\n",
        ir_.pseudo_namespace_name);
  }
}
std::string CEmitter::BuildPredictSignature() const {
  if (ir_.output_parameter.empty()) {
    return absl::Substitute("$0 $1Predict(const $1Instance* instance)",
                            ir_.types.output, ir_.pseudo_namespace_name);
  } else {
    return absl::Substitute("void $0Predict(const $0Instance* instance, $1)",
                            ir_.pseudo_namespace_name, ir_.output_parameter);
  }
}
std::string CEmitter::BuildPredictUnsafeSignature() const {
  if (ir_.output_parameter.empty()) {
    return absl::Substitute("$0 $1PredictUnsafe(const $1Instance* instance)",
                            ir_.types.output, ir_.pseudo_namespace_name);
  } else {
    return absl::Substitute(
        "void $0PredictUnsafe(const $0Instance* instance, $1)",
        ir_.pseudo_namespace_name, ir_.output_parameter);
  }
}

}  // namespace yggdrasil_decision_forests::serving::embed::internal
