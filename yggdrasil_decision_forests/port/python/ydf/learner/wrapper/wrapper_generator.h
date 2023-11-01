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

#include "absl/status/statusor.h"

namespace yggdrasil_decision_forests {

// Creates the python code source that contains wrapping classes for all linked
// learners.
absl::StatusOr<std::string> GenLearnerWrapper();

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

}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_LEARNER_WRAPPER_WRAPPER_H_
