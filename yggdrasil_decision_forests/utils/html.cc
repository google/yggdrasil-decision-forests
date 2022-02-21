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

#include "yggdrasil_decision_forests/utils/html.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace html {

std::string Escape(absl::string_view text) {
  return absl::StrReplaceAll(text, {{"&", "&amp;"},
                                    {"<", "&lt;"},
                                    {">", "&gt;"},
                                    {"\"", "&quot;"},
                                    {"'", "&#39;"}});
}

std::string Style::content() const { return std::string(content_); }

void Style::BackgroundColorHSL(float h, float s, float l) {
  AddRaw("background-color",
         absl::StrFormat("hsl(%d, %d%%, %d%%)", static_cast<int>(h * 360),
                         static_cast<int>(s * 100), static_cast<int>(l * 100)));
}

void Style::AddRaw(absl::string_view key, absl::string_view value) {
  content_.Append(key);
  content_.Append(":");
  content_.Append(value);
  content_.Append(";");
}

}  // namespace html
}  // namespace utils
}  // namespace yggdrasil_decision_forests
