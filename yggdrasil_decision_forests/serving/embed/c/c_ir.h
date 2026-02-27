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

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_C_C_IR_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_C_C_IR_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"
#include "yggdrasil_decision_forests/serving/embed/ir/model_ir.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

struct CFeature {
  // The de-duplicated, sanitized C++ name for the feature
  // (e.g., "age_2").
  std::string var_name;

  // The concrete C++ type of the feature (e.g., "float", "int16_t").
  std::string c_type;

  // The statements to sanitize this feature's NA values before passing it to
  // the PredictUnsafe function.
  // e.g., "instance.age = std::isnan(instance.age) ? 35.5f : instance.age;"
  // If empty, no sanitization is needed for this feature.
  std::vector<std::string> na_sanitization;
};

struct CEnum {
  std::string name;
  std::string type_name;
  struct Item {
    std::string name;  // e.g. "kRed"
    int value;         // e.g. 0
  };
  std::vector<Item> items;
};

struct CIR {
  // The name of the header file.
  std::string header_filename;
  // The name of the .c file.
  std::string source_filename;
  // The header define guard (e.g., "YDF_MODEL_MY_MODEL_H_").
  std::string header_guard;
  // The implementation guard (e.g., "YDF_MODEL_MY_MODEL_IMPL_").
  std::string impl_guard;
  // The pseudo-namespace before any global names.
  std::string pseudo_namespace_name;
  // The includes required for the model header.
  absl::flat_hash_set<std::string> header_includes;
  // The includes required for the model header.
  absl::flat_hash_set<std::string> impl_includes;

  // Global model constants
  NodeIdx num_trees = 0;

  // The C++ type for numerical values (float or some integer type). Will be
  // specified as a typedef.
  bool has_numercial_feature = false;

  // The custom enums required for categorical string features.
  std::vector<CEnum> enums;
  std::vector<CFeature> features;

  // --- Inference Engine Shared ---
  std::string output_parameter;  // If the prediction functions have an output
                                 // parameter, it's stored here.
  std::string accumulator_init_statement;  // e.g., "float accumulator {1.5f};"
  std::string accumulator_sum_statement;
  std::string activation_statement;  // e.g., "return 1.f / (1.f +
                                     // std::exp(-accumulator));"

  // If true, the Emitter will output both PredictUnsafe() and Predict()
  // functions. If false, it only outputs Predict().
  bool needs_predict_unsafe = false;

  std::vector<std::string> nodes;

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

  // The number of condition types used by the model.
  int num_used_conditions = 0;
  // Comma-separated list of jump offsets to the root of each tree.
  std::string root_deltas_content;

  // Categorical mask variables. Empty if no categorical conditions are used.
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

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_C_C_IR_H_
