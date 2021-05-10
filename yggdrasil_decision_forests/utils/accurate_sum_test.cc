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

#include "yggdrasil_decision_forests/utils/accurate_sum.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

TEST(AccurateSum, Empty) {
  AccurateSum s;
  EXPECT_EQ(s.Sum(), 0);
  EXPECT_EQ(s.ErrorSum(), 0);
}

TEST(AccurateSum, Simple) {
  AccurateSum s;
  s.Add(2.0);
  EXPECT_EQ(s.Sum(), 2.0);
  EXPECT_EQ(s.ErrorSum(), 0.0);
}

TEST(AccurateSum, SimpleNeg) {
  AccurateSum s;
  s.Add(-2.0);
  EXPECT_EQ(s.Sum(), -2.0);
  EXPECT_EQ(s.ErrorSum(), 0.0);
}

TEST(AccurateSum, SumOfSquares) {
  double basic_sum = 0;
  AccurateSum s;
  const int64_t n = 1e6;
  for (int64_t i = 1; i <= n; i++) {
    s.Add(i * i);
    basic_sum += i * i;
  }
  const int64_t expected_sum = n * (n + 1) * (2 * n + 1) / 6;
  const double expected_sum_double = static_cast<double>(expected_sum);
  EXPECT_EQ(s.Sum(), expected_sum_double);
  EXPECT_NE(basic_sum, expected_sum_double);
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
