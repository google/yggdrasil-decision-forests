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

// Utilities for unit testing.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_TEST_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_TEST_H_

#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "src/google/protobuf/text_format.h"
#include "src/google/protobuf/util/message_differencer.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace test {

// Tests that a absl::Status is of specific type, and contains a specific text.
//
// Usage example:
//    EXPECT_THAT(..., StatusIs(absl::StatusCode::kInvalidArgument, "ABC"));
MATCHER_P2(StatusIs, type, message_contains,
           "Status is " + testing::PrintToString(type) +
               " and contains message \"" +
               testing::PrintToString(message_contains) + "\"") {
  return arg.code() == type &&
         absl::StrContains(arg.message(), message_contains);
}

MATCHER_P(StatusIs, type, "Status is " + testing::PrintToString(type)) {
  return arg.code() == type;
}

// Ensures that two protos are equal.
MATCHER_P(EqualsProto, expected,
          "Proto is equal to:\n" + expected.DebugString()) {
  return google::protobuf::util::MessageDifferencer::Equivalent(arg, expected);
}

MATCHER_P(ApproximatelyEqualsProto, expected,
          "Proto is approximately equal to:\n" + expected.DebugString()) {
  return google::protobuf::util::MessageDifferencer::ApproximatelyEquivalent(arg,
                                                                   expected);
}

MATCHER(StatusIsOk, "Status is OK") { return arg.ok(); }

#ifndef EXPECT_OK
#define EXPECT_OK(expr) \
  EXPECT_THAT(expr, ::yggdrasil_decision_forests::test::StatusIsOk())
#endif
#ifndef ASSERT_OK
#define ASSERT_OK(expr) \
  ASSERT_THAT(expr, ::yggdrasil_decision_forests::test::StatusIsOk())
#endif

// Gets the root directory for data dependency files.
std::string DataRootDirectory();

// Temporary directory.
std::string TmpDirectory();

// Utility to create parameterized test easily.
#define SIMPLE_PARAMETERIZED_TEST(NAME, CASE_CLASS, ...)                \
  using NAME##Test = ::testing::TestWithParam<CASE_CLASS>;              \
  INSTANTIATE_TEST_SUITE_P(NAME##Instantiation, NAME##Test,             \
                           testing::ValuesIn<CASE_CLASS>(__VA_ARGS__)); \
  TEST_P(NAME##Test, Simple)

// Parse a proto from its text representation. Infers the proto message from
// the destination variable.
//
// Usage example:
// proto::MyMessage a = ParseProtoHelper("field_1: 5");
class ParseProtoHelper {
 public:
  ParseProtoHelper(absl::string_view text_proto) : text_proto_(text_proto) {}

  template <typename T>
  T call() {
    T message;
    if (!google::protobuf::TextFormat::ParseFromString(text_proto_, &message)) {
      LOG(FATAL) << "Cannot parse proto:\n" << text_proto_;
    }
    return message;
  }

  template <typename T>
  operator T() {  // NOLINT(runtime/explicit)
    return call<T>();
  }

 private:
  std::string text_proto_;
};

#define PARSE_TEST_PROTO(str) test::ParseProtoHelper(str)

#define PARSE_TEST_PROTO_WITH_TYPE(T, str) \
  test::ParseProtoHelper(str).call<typename T>()

// Returns a free TCP port.
int PickUnusedPortOrDie();

// Checks if "content" is equal to the content of "path".
// The content is exported to a temporary file in case of failure. This function
// is suited for large content. For small content, use "EXPECT_EQ" instead.
//
// To reduce flakiness, it is possible to replace substrings of the form
// ${TOKEN_NAME} in the golden string.
void ExpectEqualGolden(absl::string_view content, absl::string_view path,
                       const std::vector<std::pair<std::string, std::string>>&
                           tokens_to_replace = {});

}  // namespace test
}  // namespace yggdrasil_decision_forests

namespace yggdrasil_decision_forests {
namespace test {
using yggdrasil_decision_forests::test::DataRootDirectory;
using yggdrasil_decision_forests::test::TmpDirectory;
}  // namespace test
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_TEST_H_
