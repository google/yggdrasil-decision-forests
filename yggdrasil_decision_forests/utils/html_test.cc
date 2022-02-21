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

}  // namespace
}  // namespace html
}  // namespace utils
}  // namespace yggdrasil_decision_forests
