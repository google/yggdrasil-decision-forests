/*
 * Copyright 2021 Google LLC.
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

#include "yggdrasil_decision_forests/utils/regex.h"

#include <array>

namespace yggdrasil_decision_forests {
namespace utils {

std::string QuoteRegex(absl::string_view raw) {
  std::string result;
  result.reserve(raw.size() * 2);
  const std::array<char, 14> escape_characters = {
      '^', '$', '\\', '.', '*', '+', '?', '(', ')', '[', ']', '{', '}', '|'};
  for (const char c : raw) {
    if (std::find(escape_characters.begin(), escape_characters.end(), c) !=
        escape_characters.end()) {
      result += '\\';
    }
    result += c;
  }
  return result;
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
