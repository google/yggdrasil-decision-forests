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

#include "yggdrasil_decision_forests/utils/math.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "absl/random/random.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

TEST(CeilDiV, Base) {
  EXPECT_EQ(CeilDiV(0, 5), 0);

  EXPECT_EQ(CeilDiV(1, 5), 1);
  EXPECT_EQ(CeilDiV(2, 5), 1);

  EXPECT_EQ(CeilDiV(10, 5), 2);

  EXPECT_EQ(CeilDiV(11, 5), 3);
  EXPECT_EQ(CeilDiV(14, 5), 3);
  EXPECT_EQ(CeilDiV(15, 5), 3);

  EXPECT_EQ(CeilDiV(16, 5), 4);
}

TEST(MidThreshold, Base) {
  const auto test = [](const float a, const float b) {
    SCOPED_TRACE(testing::Message() << "a=" << a << " b=" << b);
    ASSERT_GT(b, a);
    const float r = MidThreshold(a, b);
    EXPECT_TRUE(std::isfinite(r));
    EXPECT_GT(r, a);
    EXPECT_LE(r, b);
  };
  using nl = std::numeric_limits<float>;

  test(1.f, 2.f);
  test(-10.f, -5.f);

  test(0.f, nl::max());
  test(nl::min(), nl::max());

  test(1.f, std::nextafter(1.f, 2.f));
  test(std::nextafter(1.f, 0.f), 1.f);
  test(std::nextafter(1.f, 0.f), std::nextafter(1.f, 2.f));

  test(std::nextafter(nl::max(), 0.f), nl::max());
  test(nl::lowest(), std::nextafter(nl::lowest(), 0.f));

  // "b - a" overflows.
  test(nl::lowest(), nl::max());
  test(-2e38f, nl::max());
  test(nl::lowest(), 2e38f);
}

TEST(MidThreshold, Values) {
  using nl = std::numeric_limits<float>;
  EXPECT_EQ(MidThreshold(1.f, 2.f), 1.5f);
  EXPECT_EQ(MidThreshold(-2.f, -1.f), -1.5f);
  EXPECT_EQ(MidThreshold(nl::lowest(), nl::max()), 0.f);

  // No float in between "a" and "b".
  const float a = 1.f;
  const float b = std::nextafter(a, 2.f);
  EXPECT_EQ(MidThreshold(a, b), b);
}

TEST(Median, Empty) { EXPECT_TRUE(std::isnan(Median({}))); }

TEST(Median, Base) {
  EXPECT_EQ(Median({1.f}), 1.f);

  EXPECT_EQ(Median({1.f, 2.f}), 1.5f);
  EXPECT_EQ(Median({2.f, 1.f}), 1.5f);

  EXPECT_EQ(Median({1.f, 2.f, 3.f}), 2.f);
  EXPECT_EQ(Median({2.f, 3.f, 1.f}), 2.f);
  EXPECT_EQ(Median({3.f, 1.f, 2.f}), 2.f);

  EXPECT_EQ(Median({3.f, 4.f, 1.f, 2.f}), 2.5f);
}

TEST(Median, Duplicates) {
  EXPECT_EQ(Median({2.f, 2.f}), 2.f);
  EXPECT_EQ(Median({2.f, 2.f, 2.f}), 2.f);
  EXPECT_EQ(Median({2.f, 2.f, 1.f}), 2.f);
  EXPECT_EQ(Median({3.f, 2.f, 2.f}), 2.f);
}

class MedianRandomTest : public testing::TestWithParam<int> {};

TEST_P(MedianRandomTest, Base) {
  const int n = GetParam();
  EXPECT_GE(n, 1);

  // Generate some data
  absl::BitGen rnd;
  std::vector<float> values(n);
  std::generate(values.begin(), values.end(), [&rnd]() {
    return std::uniform_real_distribution<float>()(rnd);
  });

  const float median = Median(values);

  // Check median results against n log n algorithm.
  std::sort(values.begin(), values.end());
  if ((n % 2) == 0) {
    // Event
    EXPECT_EQ(median, (values[n / 2] + values[n / 2 - 1]) / 2);
  } else {
    // Odd
    EXPECT_EQ(median, values[n / 2]);
  }
}

INSTANTIATE_TEST_SUITE_P(Event, MedianRandomTest, testing::Values(2, 10, 50));
INSTANTIATE_TEST_SUITE_P(Odd, MedianRandomTest, testing::Values(1, 51, 101));
}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
