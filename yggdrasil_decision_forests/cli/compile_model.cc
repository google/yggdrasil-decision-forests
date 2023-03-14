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

// Compile a YDF model into a small C++ header for very efficient serving and
// print this .h model on the standard output.
//
// This tool converts a YDF model into a C++ header that can be served with
// minimal dependencies. Doing so is only necessary for very specific use cases
// where binary size is very important. The majority of users should use the
// classic path, see predict.cc for details.
//
// An example of model header generated with this tool is available at
// examples/model_compiler/generated_model.h
//
// Supported models:
// - Ranking GBT with numerical only splits

#include <iostream>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/serving/decision_forest/model_compiler.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

ABSL_FLAG(std::string, model, "", "Model directory (required).");

ABSL_FLAG(std::string, namespace, "",
          "Innermost namespace for the model (required), e.g. my_model. The "
          "model will then be available as under "
          "yggdrasil_decision_forests::compiled_model::my_model::GetModel()");

constexpr char kUsageMessage[] = "Compile a model into a C++ include.";
namespace yggdrasil_decision_forests {
namespace cli {
absl::StatusOr<std::string> CompileModel() {
  // Check required flags.
  STATUS_CHECK(!absl::GetFlag(FLAGS_model).empty());
  STATUS_CHECK(!absl::GetFlag(FLAGS_namespace).empty());

  return serving::decision_forest::CompileRankingNumericalOnly(
      absl::GetFlag(FLAGS_model), absl::GetFlag(FLAGS_namespace));
}
}  // namespace cli
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  const auto model_file = yggdrasil_decision_forests::cli::CompileModel();
  QCHECK_OK(model_file.status());
  std::cout << model_file.value();
  return 0;
}
