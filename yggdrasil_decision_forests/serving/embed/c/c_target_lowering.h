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

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_C_C_TARGET_LOWERING_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_C_C_TARGET_LOWERING_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/serving/embed/c/c_ir.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/serving/embed/ir/model_ir.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

// Performs Target Lowering from the language-agnostic ModelIR to the C specific
// CIR.
class CTargetLowering {
 public:
  // Main entry point. Translates the ModelIR into a printable CIR.
  static absl::StatusOr<CIR> Lower(const ModelIR& model_ir,
                                   const proto::Options& options);

 private:
  CTargetLowering(const ModelIR& model_ir, const proto::Options& options)
      : model_ir_(model_ir), options_(options) {}

  // Orchestrates the lowering steps.
  absl::StatusOr<CIR> Run();

  // --- Lowering Steps ---

  // Populates header_guard, namespace_name, and the includes list.
  absl::Status LowerGlobalFormatting();

  // Iterates features: maps storage_bytes to C types (e.g. int16_t),
  // sanitizes/deduplicates variable names, builds enum definitions, and
  // generates na_sanitization statements.
  absl::Status LowerEnumsAndFeatures();

  absl::Status LowerFeature(
      FeatureIdx feature_idx,
      absl::flat_hash_set<std::string>& sanitized_feature_names,
      absl::flat_hash_set<std::string>& sanitized_categorical_type_names,
      std::vector<int>& condition_types_bank);

  absl::Status LowerEnum(
      FeatureIdx feature_idx, const FeatureInfo& feature,
      absl::flat_hash_set<std::string>& sanitized_categorical_type_names);

  // Sets up the accumulator_type (scalar vs std::array), the initialization
  // statement, and the activation return statement.
  absl::Status LowerInferenceEngine();
  absl::Status LowerAccumulator();
  absl::Status LowerActivation();

  // Flattens IR nodes into CNodes.
  // Pre-formats the conditions (e.g. "instance.age >= 40.5f") and leaf logic.
  absl::Status LowerNodes();

  absl::StatusOr<std::string> LowerLeafNode(const Node& ir_node);
  absl::StatusOr<std::string> LowerConditionNode(const Node& ir_node);

  // Helper for generating the complex condition strings for categorical sets.
  absl::StatusOr<std::string> FormatContainsCondition(
      const Node& ir_node, const FeatureInfo& feat,
      const std::string& var_name);

  absl::Status LowerRoutingData();

  // --- Helpers ---

  std::string AddPseudoNamespace(absl::string_view name) const;
  const ModelIR& model_ir_;
  const proto::Options& options_;

  // The output object being built.
  CIR c_ir_;

  // State maps used during lowering to resolve references.
  // Maps IR Feature ID -> Sanizited C struct member name.
  absl::flat_hash_map<FeatureIdx, std::string> feat_idx_to_var_name_;

  // Maps IR Feature ID -> Generated Enum Class name.
  absl::flat_hash_map<FeatureIdx, std::string> feat_idx_to_enum_name_;
};

}  // namespace yggdrasil_decision_forests::serving::embed::internal

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_C_C_TARGET_LOWERING_H_
