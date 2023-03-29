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

// Utility for the manipulation of regular expressions.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_REGEX_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_REGEX_H_

#include "absl/strings/string_view.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Escapes a string for an exact match with std::regex_match for the ECMAScript
// syntax (default of std::regex).
//
// Args:
//   raw: String to escape.
//   full_match: If false, build a search regex (i.e., look for "raw" in a
//   string). If true, build a match regexp (i.e., look for the a string equal
//   to "raw").
std::string QuoteRegex(absl::string_view raw, bool full_match);

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_REGEX_H_
