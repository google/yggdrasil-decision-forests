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

// Utility to generate html code.
//
// Usage example:
//   // Escaping.
//   Escape("<script>hi</script>")
//
//   // Building a page
//   Input(Type("checkbox"),  Checked(""), "Name of my box").html()
//
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_HTML_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_HTML_H_

#include <string>

#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace html {

// Escapes HTML markup into html text. Escape the characters '&', '<', '>', '"'
// and '\'').
std::string Escape(absl::string_view text);

namespace internal {

// Chunk of html.
class Html {
 public:
  void Append(Html text) { AppendRaw(text.content()); }

  void Append(absl::string_view text) { AppendRaw(Escape(text)); }

  void AppendRaw(absl::string_view text) { content_.Append(text); }

  void AppendRaw(const absl::Cord& text) { content_.Append(text); }

  const absl::Cord& content() const { return content_; }

 private:
  absl::Cord content_;
};

// Attribute.
class Attr {
 public:
  Attr(absl::string_view key, absl::string_view value, const bool escape = true)
      : key_(key), value_(value), escape_(escape) {}

  absl::string_view key() { return key_; }

  absl::string_view value() { return value_; }

  bool escape() { return escape_; }

 private:
  std::string key_;
  std::string value_;
  bool escape_;
};

// Append tag body.
inline void Append(Html* dst, const absl::Cord& src) {
  dst->AppendRaw(Escape(std::string(src)));
}

inline void Append(Html* dst, absl::string_view src) {
  dst->AppendRaw(Escape(src));
}

inline void Append(Html* dst, const char* const src) {
  dst->AppendRaw(Escape(src));
}

inline void Append(Html* dst, const Html& src) {
  dst->AppendRaw(src.content());
}

inline void Append(Html* dst, const std::string src) {
  dst->AppendRaw(Escape(src));
}

template <typename T, typename... Args>
void Append(Html* dst, T src, Args... remain) {
  Append(dst, src);
  Append(dst, remain...);
}

// Append tag header and attributes.
inline void AppendWithAttr(Html* dst, absl::string_view tag_key,
                           absl::Cord* attr_accumulator) {
  dst->AppendRaw("<");
  dst->AppendRaw(tag_key);
  dst->AppendRaw(*attr_accumulator);
  dst->AppendRaw(">");
}

template <typename... Args>
void AppendWithAttr(Html* dst, absl::string_view tag_key,
                    absl::Cord* attr_accumulator, Attr attr, Args... remain) {
  attr_accumulator->Append(" ");
  attr_accumulator->Append(attr.key());
  attr_accumulator->Append("=\"");
  if (attr.escape()) {
    attr_accumulator->Append(Escape(attr.value()));
  } else {
    attr_accumulator->Append(attr.value());
  }
  attr_accumulator->Append("\"");

  AppendWithAttr(dst, tag_key, attr_accumulator, remain...);
}

template <typename... Args>
void AppendWithAttr(Html* dst, absl::string_view tag_key,
                    absl::Cord* attr_accumulator, Args... remain) {
  dst->AppendRaw("<");
  dst->AppendRaw(tag_key);
  dst->AppendRaw(*attr_accumulator);
  dst->AppendRaw(">");

  Append(dst, remain...);

  dst->AppendRaw("</");
  dst->AppendRaw(tag_key);
  dst->AppendRaw(">");
}

// Tag creation utility.
template <typename... Args>
Html Tag(absl::string_view tag_key, Args... args) {
  absl::Cord attr_accumulator;
  Html dst;
  AppendWithAttr(&dst, tag_key, &attr_accumulator, args...);
  return dst;
}

}  // namespace internal

using Html = internal::Html;

// Tags.

template <typename... Args>
Html H1(Args... args) {
  return internal::Tag("h1", args...);
}

template <typename... Args>
Html H2(Args... args) {
  return internal::Tag("h2", args...);
}

template <typename... Args>
Html H3(Args... args) {
  return internal::Tag("h3", args...);
}

template <typename... Args>
Html H4(Args... args) {
  return internal::Tag("h4", args...);
}

template <typename... Args>
Html Pre(Args... args) {
  return internal::Tag("pre", args...);
}

template <typename... Args>
Html Th(Args... args) {
  return internal::Tag("th", args...);
}

template <typename... Args>
Html Tr(Args... args) {
  return internal::Tag("tr", args...);
}

template <typename... Args>
Html Td(Args... args) {
  return internal::Tag("td", args...);
}

template <typename... Args>
Html Div(Args... args) {
  return internal::Tag("div", args...);
}

template <typename... Args>
Html P(Args... args) {
  return internal::Tag("p", args...);
}

template <typename... Args>
Html Li(Args... args) {
  return internal::Tag("li", args...);
}

template <typename... Args>
Html Ul(Args... args) {
  return internal::Tag("ul", args...);
}

template <typename... Args>
Html Strong(Args... args) {
  return internal::Tag("strong", args...);
}

template <typename... Args>
Html Table(Args... args) {
  return internal::Tag("table", args...);
}

template <typename... Args>
Html Input(Args... args) {
  return internal::Tag("input", args...);
}

template <typename... Args>
Html Button(Args... args) {
  return internal::Tag("button", args...);
}

template <typename... Args>
Html A(Args... args) {
  return internal::Tag("a", args...);
}

template <typename... Args>
Html B(Args... args) {
  return internal::Tag("b", args...);
}

template <typename... Args>
Html Select(Args... args) {
  return internal::Tag("select", args...);
}

template <typename... Args>
Html Option(Args... args) {
  return internal::Tag("option", args...);
}

inline Html Br() {
  Html content;
  content.AppendRaw("<br>");
  return content;
}

// Style

class Style {
 public:
  std::string content() const;

  // HSL color with "h", "s" and "l" in [0,1].
  void BackgroundColorHSL(float h, float s, float l);

  void AddRaw(absl::string_view key, absl::string_view value);

 private:
  absl::Cord content_;
};

// Attributes.

inline internal::Attr Type(absl::string_view value) {
  return internal::Attr("type", value);
}

inline internal::Attr Class(absl::string_view value) {
  return internal::Attr("class", value);
}

inline internal::Attr OnClick(absl::string_view value) {
  return internal::Attr("onclick", value, false);
}

inline internal::Attr OnChange(absl::string_view value) {
  return internal::Attr("onchange", value, false);
}

inline internal::Attr Id(absl::string_view value) {
  return internal::Attr("id", value);
}

inline internal::Attr Align(absl::string_view value) {
  return internal::Attr("align", value);
}

inline internal::Attr DataAttr(absl::string_view key, absl::string_view value) {
  return internal::Attr(absl::StrCat("data-", key), value);
}

inline internal::Attr Checked(absl::string_view value) {
  return internal::Attr("checked", value);
}

inline internal::Attr Style(Style value) {
  return internal::Attr("style", value.content());
}

inline internal::Attr HRef(absl::string_view value) {
  return internal::Attr("href", value);
}

inline internal::Attr Value(absl::string_view value) {
  return internal::Attr("value", value, false);
}

inline internal::Attr Target(absl::string_view value) {
  return internal::Attr("target", value);
}

}  // namespace html
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_HTML_H_
