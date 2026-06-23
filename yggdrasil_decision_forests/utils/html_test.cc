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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace html {
namespace {

TEST(Html, Escape) {
  EXPECT_EQ(Escape("Hello world"), "Hello world");
  EXPECT_EQ(Escape("<script>hi</script>"), "&lt;script&gt;hi&lt;/script&gt;");
}

TEST(Html, Composition) {
  auto page = Th(Th("<b>a</b>"), "b", "c");
  EXPECT_EQ(page.content(), "<th><th>&lt;b&gt;a&lt;/b&gt;</th>bc</th>");
}

TEST(Html, Attribute) {
  auto page = Th(Type("a"), "b", "c");
  EXPECT_EQ(page.content(), "<th type=\"a\">bc</th>");
}

TEST(Html, Style) {
  class Style style;
  style.BackgroundColorHSL(0.5f, 0.5f, 0.5f);
  style.AddRaw("a", "b");
  EXPECT_EQ(style.content(), "background-color:hsl(180, 50%, 50%);a:b;");
}

TEST(Html, OnClick) {
  auto page = A(OnClick("f('hello');"), "world");
  // Note: Surprisingly to me, escaping strings this was in js code works fine.
  EXPECT_EQ(page.content(), "<a onclick=\"f('hello');\">world</a>");
}

TEST(Html, HRefSanitization) {
  auto page_safe = A(HRef("https://my.corp.url/123"), "study");
  EXPECT_EQ(page_safe.content(),
            "<a href=\"https://my.corp.url/123\">study</a>");

  auto go_link = A(HRef("go/mylink"), "study");
  EXPECT_EQ(go_link.content(), "<a href=\"go/mylink\">study</a>");

  auto page_js = A(HRef("javascript:alert(1)"), "alert");
  EXPECT_EQ(page_js.content(), "<a href=\"#\">alert</a>");

  auto page_js_spaced = A(HRef(" j a v a s c r i p t : alert(1)"), "alert");
  EXPECT_EQ(page_js_spaced.content(), "<a href=\"#\">alert</a>");

  auto page_js_mixed = A(HRef("JaVaScRiPt:alert(1)"), "alert");
  EXPECT_EQ(page_js_mixed.content(), "<a href=\"#\">alert</a>");

  auto page_vb = A(HRef("vbscript:msgbox(1)"), "alert");
  EXPECT_EQ(page_vb.content(), "<a href=\"#\">alert</a>");

  auto page_data = A(HRef("data:text/html,<script>alert(1)</script>"), "data");
  EXPECT_EQ(page_data.content(), "<a href=\"#\">data</a>");
}

}  // namespace
}  // namespace html
}  // namespace utils
}  // namespace yggdrasil_decision_forests
