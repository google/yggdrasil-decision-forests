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

#include "yggdrasil_decision_forests/utils/time.h"

#include "gtest/gtest.h"
#include "absl/time/time.h"

namespace yggdrasil_decision_forests::utils {
namespace {

TEST(FormatDurationForLogs, Sime) {
  EXPECT_EQ(FormatDurationForLogs(absl::Seconds(0)), "0s");
  EXPECT_EQ(FormatDurationForLogs(absl::Seconds(0.5)), "0.5s");
  EXPECT_EQ(FormatDurationForLogs(absl::Seconds(0.12345)), "0.12s");
  EXPECT_EQ(FormatDurationForLogs(absl::Seconds(1.5)), "1.5s");
  EXPECT_EQ(FormatDurationForLogs(absl::Minutes(5)), "5m");
  EXPECT_EQ(FormatDurationForLogs(absl::Minutes(5) + absl::Seconds(30.5)),
            "5m30.5s");
  EXPECT_EQ(FormatDurationForLogs(absl::Hours(2) + absl::Seconds(30.5)),
            "2h0m30.5s");
  EXPECT_EQ(FormatDurationForLogs(absl::Hours(2) + absl::Minutes(5) +
                                  absl::Seconds(30.5)),
            "2h5m30.5s");
}

}  // namespace
}  // namespace yggdrasil_decision_forests::utils
