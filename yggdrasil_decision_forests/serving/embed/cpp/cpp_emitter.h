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

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_CPP_CPP_EMITTER_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_CPP_CPP_EMITTER_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/serving/embed/cpp/cpp_ir.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

// Phase 3: Target Emitter.
// Takes the Target-Specific CppIR and formats it into the final C++ source
// code. This class contains NO logic for type inference, variable
// deduplication, or model math. It simply prints what it is told to print using
// text templates.
class CppEmitter {
 public:
  // Main entry point. Returns the complete contents of the generated .h file.
  static absl::StatusOr<std::string> Emit(const CppIR& ir,
                                          const proto::Options& options);

 private:
  CppEmitter(const CppIR& ir, const proto::Options& options)
      : ir_(ir), options_(options) {}

  absl::StatusOr<std::string> Run() const;

  // --- Printing Helpers ---

  // Prints the `#include <...>` block.
  void EmitIncludes(std::string* out) const;

  // Prints the `enum class` definitions and optional string-conversion
  // functions.
  void EmitEnums(std::string* out) const;

  // Prints the `struct Instance { ... }` block containing the typed features.
  void EmitInstanceStruct(std::string* out) const;

  // Prints the `nodes[]` array, data banks, and condition eval blocks
  // required by the ROUTING algorithm.
  absl::Status EmitRoutingData(std::string* out) const;

  // Generates the `PredictUnsafe()` function using nested if-else statements.
  absl::Status EmitPredictUnsafeIfElse(std::string* out) const;

  // Generates the `PredictUnsafe()` function using the routing while-loop.
  absl::Status EmitPredictUnsafeRouting(std::string* out) const;

  // Generates the `Predict()` wrapper that contains NaN sanitization logic.
  void EmitPredictSanitized(std::string* out) const;

  // Recursive helper for `EmitPredictUnsafeIfElse`.
  // Reconstructs the tree structure from the flattened array.
  // Returns the index of the next node to process.
  absl::StatusOr<int> PrintIfElseNode(int node_idx, int depth,
                                      std::string* out) const;

  const CppIR& ir_;
  const proto::Options& options_;
};

}  // namespace yggdrasil_decision_forests::serving::embed::internal

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_CPP_CPP_EMITTER_H_