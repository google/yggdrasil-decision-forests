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

#include "yggdrasil_decision_forests/utils/html_content.h"

#include <string>

#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace yggdrasil_decision_forests::utils {

std::string CssCommon() {
  return R"(
.tab_block .header {
    flex-direction: row;
    display: flex;
}

.tab_block .header .tab {
    cursor: pointer;
    background-color: #F6F5F5;
    text-decoration: none;
    text-align: center;
    padding: 4px 12px;
    color: black;
}

.tab_block .header .tab.selected {
    border-bottom: 2px solid #2F80ED;
}

.tab_block .header .tab:hover {
    text-decoration: none;
    background-color: #DCDCDC;
}

.tab_block .body .tab_content {
    display: none;
    padding: 5px;
}

.tab_block .body .tab_content.selected {
    display: block;
}

.ydf_pre {
    font-size: medium;
}

)";
}

std::string JsCommon() {
  return R"(
function ydfShowTab(block_id, item) {
    const block = document.getElementById(block_id);
    
    
    console.log("HIDE first of:",block.getElementsByClassName("tab selected"));
    console.log("HIDE first of:",block.getElementsByClassName("tab_content selected"));
    
    block.getElementsByClassName("tab selected")[0].classList.remove("selected");
    block.getElementsByClassName("tab_content selected")[0].classList.remove("selected");
    document.getElementById(block_id + "_" + item).classList.add("selected");
    document.getElementById(block_id + "_body_" + item).classList.add("selected");
}
  )";
}

std::string HeaderCommon() {
  return absl::Substitute(R"(
<style>
$0
</style>
<script>
$1
</script>
  )",
                          CssCommon(), JsCommon());
}

TabBarBuilder::TabBarBuilder(const absl::string_view block_id) {
  if (block_id.empty()) {
    block_id_ = utils::GenUniqueId();
  } else {
    block_id_ = std::string(block_id);
  }
}

void TabBarBuilder::AddTab(const absl::string_view key,
                           const absl::string_view title,
                           const utils::html::Html& content) {
  namespace h = utils::html;

  const absl::string_view maybe_selected = first_tab_ ? " selected" : "";
  first_tab_ = false;

  const auto onclick =
      absl::Substitute("ydfShowTab('$0', '$1')", block_id_, key);

  tab_header_.Append(h::A(h::Id(absl::StrCat(block_id_, "_", key)),
                          h::Class(absl::StrCat("tab", maybe_selected)),
                          h::OnClick(onclick), title));
  tab_content_.Append(
      h::Div(h::Id(absl::StrCat(block_id_, "_body_", key)),
             h::Class(absl::StrCat("tab_content", maybe_selected)), content));

  first_tab_ = false;
}

utils::html::Html TabBarBuilder::Html() const {
  namespace h = utils::html;
  return h::Div(h::Class("tab_block"), h::Id(block_id_),
                h::Div(h::Class("header"), tab_header_),
                h::Div(h::Class("body"), tab_content_));
}

}  // namespace yggdrasil_decision_forests::utils
