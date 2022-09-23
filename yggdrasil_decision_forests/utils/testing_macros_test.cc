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

#include "yggdrasil_decision_forests/utils/testing_macros.h"

#include "gmock/gmock.h"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace testing_macros {
namespace {

void FailsForFalse(const bool a) {
  const auto f = [](const bool a) -> absl::StatusOr<int> {
    if (!a) {
      return absl::InvalidArgumentError("a is false");
    }
    return 1;
  };

  int b;
  ASSERT_OK_AND_ASSIGN(b, f(a));
}

TEST(TestingMacros, ASSERT_OK_AND_ASSIGN_ok) {
  FailsForFalse(true);  // Does not fail
}

TEST(TestingMacros, ASSERT_NOT_OK_AND_ASSIGN_error) {
  EXPECT_FATAL_FAILURE(FailsForFalse(false), "a is false");
}

}  // namespace
}  // namespace testing_macros
}  // namespace utils
}  // namespace yggdrasil_decision_forests
