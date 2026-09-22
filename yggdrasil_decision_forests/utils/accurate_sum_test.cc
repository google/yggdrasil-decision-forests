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

#include "yggdrasil_decision_forests/utils/accurate_sum.h"

#include <cmath>
#include <cstdint>
#include <vector>

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

TEST(NeumaierSum, Empty) {
  NeumaierSum s;
  EXPECT_EQ(s.Value(), 0);
  EXPECT_EQ(s.Compensation(), 0);
}

TEST(NeumaierSum, Simple) {
  NeumaierSum s;
  s.Add(2.0);
  EXPECT_EQ(s.Value(), 2.0);
}

TEST(NeumaierSum, SimpleNeg) {
  NeumaierSum s;
  s.Add(-2.0);
  EXPECT_EQ(s.Value(), -2.0);
}

TEST(NeumaierSum, Reset) {
  NeumaierSum s;
  s.Add(1e100);
  s.Add(1.0);
  s.Reset();
  EXPECT_EQ(s.Value(), 0.0);
  EXPECT_EQ(s.Compensation(), 0.0);
}

TEST(NeumaierSum, SumOfSquares) {
  double basic_sum = 0;
  NeumaierSum s;
  const int64_t n = 1e6;
  for (int64_t i = 1; i <= n; i++) {
    s.Add(i * i);
    basic_sum += i * i;
  }
  const int64_t expected_sum = n * (n + 1) * (2 * n + 1) / 6;
  const double expected_sum_double = static_cast<double>(expected_sum);
  EXPECT_EQ(s.Value(), expected_sum_double);
  EXPECT_NE(basic_sum, expected_sum_double);
}

TEST(NeumaierSum, SmallTermsSurviveRemovalOfLargeTerm) {
  const double big = std::exp(40);

  NeumaierSum compensated;
  AccurateSum kahan;
  double naive = 0.;
  for (const double value : {big, 1., 1., 1., -big}) {
    compensated.Add(value);
    kahan.Add(value);
    naive += value;
  }

  EXPECT_EQ(compensated.Value(), 3.0);
  // Documents the limitation of the two other approaches.
  EXPECT_EQ(kahan.Sum(), 0.0);
  EXPECT_EQ(naive, 0.0);
}

TEST(NeumaierSum, BuildUpAndTearDownIsAlmostNull) {
  std::vector<double> values;
  double sum_of_magnitudes = 0.;
  for (int i = 0; i < 1000; i++) {
    values.push_back(std::exp(i % 100));
    sum_of_magnitudes += 2 * values.back();
  }

  NeumaierSum compensated;
  double naive = 0.;
  for (const double value : values) {
    compensated.Add(value);
    naive += value;
  }
  for (const double value : values) {
    compensated.Add(-value);
    naive -= value;
  }

  EXPECT_LT(std::abs(compensated.Value()), 1e-28 * sum_of_magnitudes);
  EXPECT_GT(std::abs(naive), 1e-20 * sum_of_magnitudes);
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
