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

#include "yggdrasil_decision_forests/utils/html.h"

#include <cstddef>
#include <string>

#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"

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

namespace internal {

std::string SanitizeUrl(absl::string_view url) {
  size_t start = 0;
  while (start < url.size() &&
         (absl::ascii_isspace(url[start]) || absl::ascii_iscntrl(url[start]))) {
    start++;
  }

  absl::string_view trimmed_url = url.substr(start);

  size_t colon_pos = trimmed_url.find(':');

  // Relative URLs are allowed.
  if (colon_pos == absl::string_view::npos) {
    return std::string(url);
  }

  absl::string_view scheme_view = trimmed_url.substr(0, colon_pos);
  std::string scheme;
  scheme.reserve(scheme_view.size());

  for (char c : scheme_view) {
    if (!absl::ascii_isspace(c) && !absl::ascii_iscntrl(c)) {
      scheme += absl::ascii_tolower(c);
    }
  }

  if (scheme == "http" || scheme == "https" || scheme == "mailto") {
    return std::string(url);
  }

  // If it has a scheme but isn't explicitly allowed, nuke it.
  return "#";
}

}  // namespace internal

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
