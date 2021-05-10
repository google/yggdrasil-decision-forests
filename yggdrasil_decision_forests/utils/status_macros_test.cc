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

#include "yggdrasil_decision_forests/utils/status_macros.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace status_macros {
namespace {

TEST(StatusMacros, RETURN_IF_ERROR) {
  const auto check_positive = [](const int a) -> absl::Status {
    if (a < 0) {
      return absl::InvalidArgumentError("A is lower than zero.");
    }
    return absl::OkStatus();
  };

  const auto check_two_positive = [&](const int a,
                                      const int b) -> absl::Status {
    RETURN_IF_ERROR(check_positive(a));
    RETURN_IF_ERROR(check_positive(b));
    return absl::OkStatus();
  };

  EXPECT_TRUE(check_two_positive(1, 2).ok());
  EXPECT_FALSE(check_two_positive(-1, 2).ok());
  EXPECT_FALSE(check_two_positive(1, -2).ok());
}

TEST(StatusMacros, ASSIGN_OR_RETURN_2ARGS) {
  const auto f = [](const int a) -> utils::StatusOr<int> {
    if (a < 0) {
      return absl::InvalidArgumentError("A is lower than zero.");
    }
    return a;
  };

  const auto g = [&](const int a) -> absl::Status {
    ASSIGN_OR_RETURN(const int b, f(a));
    LOG(INFO) << "b:" << b;
    return absl::OkStatus();
  };

  EXPECT_TRUE(g(1).ok());
  EXPECT_FALSE(g(-1).ok());
}

TEST(StatusMacros, ASSIGN_OR_RETURN_3ARGS) {
  const auto f = [](const int a) -> utils::StatusOr<int> {
    if (a < 0) {
      return absl::InvalidArgumentError("A is lower than zero.");
    }
    return a;
  };

  const auto g = [&](const int a) -> absl::Status {
    ASSIGN_OR_RETURN(const int b, f(a), _ << "a:" << a);
    LOG(INFO) << "b:" << b;
    return absl::OkStatus();
  };

  EXPECT_TRUE(g(1).ok());
  EXPECT_FALSE(g(-1).ok());
}

}  // namespace
}  // namespace status_macros
}  // namespace utils
}  // namespace yggdrasil_decision_forests
