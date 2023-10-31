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

// Assets to display Html pages, and utilities to access them.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_HTML_CONTENT_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_HTML_CONTENT_H_

#include <string>

#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/html.h"

namespace yggdrasil_decision_forests::utils {

std::string CssCommon();
std::string JsCommon();

// Combines CssCommon and JsCommon.
std::string HeaderCommon();

// Creates a tab bar.
class TabBarBuilder {
 public:
  TabBarBuilder(absl::string_view block_id = {});

  // Adds a new tab.
  void AddTab(absl::string_view key, absl::string_view title,
              const utils::html::Html& content);

  // Html displaying the tab bar and content.
  utils::html::Html Html() const;

 private:
  std::string block_id_;
  utils::html::Html tab_header_;
  utils::html::Html tab_content_;
  bool first_tab_ = true;
};

}  // namespace yggdrasil_decision_forests::utils

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_HTML_CONTENT_H_
