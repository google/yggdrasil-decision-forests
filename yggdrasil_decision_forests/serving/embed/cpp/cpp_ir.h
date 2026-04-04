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

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_CPP_CPP_IR_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_CPP_CPP_IR_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"
#include "yggdrasil_decision_forests/serving/embed/ir/model_ir.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

struct CppFeature {
  // The de-duplicated, sanitized C++ name for the feature
  // (e.g., "age_2").
  std::string var_name;

  // The concrete C++ type of the feature (e.g., "float", "int16_t").
  std::string cpp_type;

  // The statements to sanitize this feature's NA values before passing it to
  // the PredictUnsafe function.
  // e.g., "instance.age = std::isnan(instance.age) ? 35.5f : instance.age;"
  // If empty, no sanitization is needed for this feature.
  std::vector<std::string> na_sanitization;
};

struct CppEnum {
  std::string name;
  struct Item {
    std::string name;      // e.g. "kRed"
    int value;             // e.g. 0
    std::string original;  // e.g. "Red"
  };
  std::vector<Item> items;
  bool generate_from_string_method = false;
};

struct CppNode {
  bool is_leaf;
  // --- If-Else specific fields ---
  // Formatted condition for if-else generation (e.g., "instance.age >= 40.5f").
  std::string if_else_condition;
  // Formatted leaf statement (e.g., "accumulator += 0.5f;").
  std::string if_else_leaf;
  // The offset to the False branch, necessary for the Emitter's recursive
  // tree reconstruction. (The True branch is implicitly offset 1).
  NodeIdx jump_offset_false;

  // --- Routing specific fields ---
  // The pre-formatted struct initializer for the nodes[] array.
  // e.g., "{.pos=3, .cond={.feat=2, .thr=40.5f}}"
  std::string routing_def;
};

struct CppIR {
  // The header define guard (e.g., "YDF_MODEL_MY_MODEL_H_").
  std::string header_guard;
  // The includes required for this model.
  // Must be sorted to ensure deterministic code generation.
  absl::flat_hash_set<std::string> includes;
  // The namespace enclosing the generated code.
  std::string namespace_name;

  // Global model constants
  NodeIdx num_trees = 0;

  // The custom enums required for categorical string features.
  std::vector<CppEnum> enums;
  std::vector<CppFeature> features;
  bool has_integerized_categorical_features = false;

  // --- Inference Engine Shared ---
  std::string
      full_output_type;  // Output type including, potentially std::array<>.
  std::string accumulator_init_statement;  // e.g., "float accumulator {1.5f};"
  std::string accumulator_sum_statement;
  std::string activation_statement;  // e.g., "return 1.f / (1.f +
                                     // std::exp(-accumulator));"

  // If true, the Emitter will output both PredictUnsafe() and Predict()
  // functions. If false, it only outputs Predict().
  bool needs_predict_unsafe = false;

  std::vector<CppNode> nodes;

  // =========================================================================
  // --- Routing Specific Data ---
  // The fields below are only populated and used if the selected algorithm
  // is ROUTING. They represent the type mappings and bank contents.
  // =========================================================================

  // Formatted snippets for the condition evaluation blocks in the routing loop.
  // E.g., "float f; std::memcpy(&f, raw + ..., 4); eval = f >= node->cond.thr;"
  std::vector<std::pair<ConditionType, std::string>>
      routing_condition_eval_blocks;
  // Comma-separated list of integers representing the condition types for each
  // node.
  std::string condition_types_content;
  // The indexes of the condition types used in the model
  std::array<bool, static_cast<int>(ConditionType::NUM_ROUTING_CONDITION_TYPES)>
      used_condition_types{};
  // The number of condition types used by the model.
  int num_used_conditions = 0;
  // Comma-separated list of jump offsets to the root of each tree.
  std::string root_deltas_content;

  size_t categorical_bank_size = 0;      // Needed to instantiate std::bitset<N>
  std::string categorical_bank_content;  // E.g., "101110110011"

  // Oblique split variables. Empty if no oblique conditions are used.
  uint32_t oblique_feature_idx = 0;
  std::string oblique_weights_content;   // Comma separated list of floats
  std::string oblique_features_content;  // Comma separated list of indices

  // Multi-class leaf value variables. Empty if scalar leaves are used.
  std::string leaf_value_bank_content;

  BaseTypes types;
};
}  // namespace yggdrasil_decision_forests::serving::embed::internal

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_CPP_CPP_IR_H_
