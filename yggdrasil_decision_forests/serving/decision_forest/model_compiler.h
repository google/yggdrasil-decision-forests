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

// Compile a YDF model into a small C++ header for very efficient serving.
//
// This library converts a YDF model into a C++ header that can be served with
// minimal dependencies. Doing so is only necessary for very specific use cases
// where binary size is very important. The majority of users should use the
// classic path, see predict.cc for details.
//
// An example of model header generated with this tool is available at
// examples/model_compiler/generated_model.h
//
// Supported models:
// - Ranking GBT with numerical only splits

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_MODEL_COMPILER_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_MODEL_COMPILER_H_

#include <string>

#include "absl/status/statusor.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace decision_forest {

absl::StatusOr<std::string> CompileRankingNumericalOnly(
    const std::string& model_path, const std::string& name_space);

}  // namespace decision_forest
}  // namespace serving
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_MODEL_COMPILER_H_
