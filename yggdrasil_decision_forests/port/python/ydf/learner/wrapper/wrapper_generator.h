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

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_LEARNER_WRAPPER_WRAPPER_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_LEARNER_WRAPPER_WRAPPER_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"

namespace yggdrasil_decision_forests {

namespace internal {

// Configuration data for each individual learner.
struct LearnerConfig {
  // Name of the python class of the model.
  std::string model_class_name = "generic_model.GenericModel";

  // Default value of the "task" learner constructor argument.
  std::string default_task = "CLASSIFICATION";

  // Default value of the "task" learner constructor argument.
  bool support_distributed_training = false;
};

// Create the configurations for all learners.
absl::flat_hash_map<std::string, LearnerConfig> LearnerConfigs();

// Creates the header of the file header
std::string GenHeader();

// Creates and appends the capability-based parameters of the learner.
absl::Status AppendCapabilityParameters(
    const LearnerConfig& learner_config,
    const model::proto::LearnerCapabilities& capabilities,
    std::string* fields_documentation, std::string* fields_constructor,
    std::string* deployment_config_constructor);

// Creates the Python code source of the class wrapping a single learner.
absl::StatusOr<std::string> GenSingleLearnerWrapper(
    absl::string_view learner_key, LearnerConfig learner_config);

// Returns Python class name from the learner key.
std::string LearnerKeyToClassName(absl::string_view key);

// Formats a text into python documentation.
//
// Do the following transformations:
//   - Wraps lines to "max_char_per_lines" characters (while carrying leading
//   space).
//   - Add leading spaces (leading_spaces_first_line and
//     leading_spaces_next_lines).
//   - Detect and format bullet lists.
std::string FormatDocumentation(absl::string_view raw,
                                int leading_spaces_first_line,
                                int leading_spaces_next_lines,
                                int max_char_per_lines = 80);

// Gets the number of leading spaces of a string.
int NumLeadingSpaces(absl::string_view text);
}  // namespace internal

// Creates the Python code source that contains wrapping classes for all linked
// learners.
absl::StatusOr<std::string> GenAllLearnersWrapper();
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_LEARNER_WRAPPER_WRAPPER_H_
