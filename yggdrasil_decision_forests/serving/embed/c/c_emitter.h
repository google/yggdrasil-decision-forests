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

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_C_C_EMITTER_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_C_C_EMITTER_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/serving/embed/c/c_ir.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

// Takes the Target-Specific CIR and formats it into the final C source code.
class CEmitter {
 public:
  // Main entry point. Returns the complete contents of the generated .h file.
  static absl::StatusOr<absl::node_hash_map<std::string, std::string>> Emit(
      const CIR& ir, const proto::Options& options);

 private:
  CEmitter(const CIR& ir, const proto::Options& options)
      : ir_(ir), options_(options) {}

  absl::StatusOr<absl::node_hash_map<std::string, std::string>> Run() const;

  // Creates a very small .c file to activate the implementation header.
  absl::StatusOr<std::string> CreatePseudoSource() const;

  // Creates as STB-style single-header library.
  absl::StatusOr<std::string> CreateSTBHeader() const;

  // Creates the header part of the STB-style file.
  void EmitActualHeader(std::string* out) const;

  // Creates the implementation part of the STB-style file.
  absl::Status EmitImplementation(std::string* out) const;

  // --- Printing Helpers ---

  // Prints the `#include <...>` block.
  void EmitIncludes(const absl::flat_hash_set<std::string>& includes,
                    std::string* out) const;

  // Prints the `enum class` definitions and optional string-conversion
  // functions.
  void EmitEnums(std::string* out) const;

  // Prints the `struct Instance { ... }` block containing the typed features.
  void EmitInstanceStruct(std::string* out) const;

  // Prints the `nodes[]` array, data banks, and condition eval blocks
  // required by the ROUTING algorithm.
  absl::Status EmitRoutingData(std::string* out) const;

  // Generates the `PredictUnsafe()` function using the routing while-loop.
  absl::Status EmitPredictUnsafeRouting(std::string* out) const;

  // Generates the `Predict()` wrapper that contains NaN sanitization logic.
  void EmitPredictSanitized(std::string* out) const;

  // Emits the FeatureOffsets array.
  void EmitFeatureOffsets(std::string* out) const;

  // Helper to generate the signatures of functions needed both in the header
  // and the implementation.
  std::string BuildPredictSignature() const;
  std::string BuildPredictUnsafeSignature() const;

  const CIR& ir_;
  const proto::Options& options_;
};

}  // namespace yggdrasil_decision_forests::serving::embed::internal

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_C_C_EMITTER_H_
