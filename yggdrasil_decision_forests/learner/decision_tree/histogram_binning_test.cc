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

#include "yggdrasil_decision_forests/learner/decision_tree/histogram_binning.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "hwy/targets.h"

namespace yggdrasil_decision_forests::model::decision_tree::internal {
namespace {

int32_t ReferenceIndex(const std::vector<float>& thresholds, const float x) {
  return static_cast<int32_t>(
             std::upper_bound(thresholds.begin(), thresholds.end(), x) -
             thresholds.begin()) -
         1;
}

// Values around every threshold plus the special cases.
std::vector<float> ProbeValues(const std::vector<float>& thresholds,
                               std::mt19937* rng) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  std::vector<float> values = {-kInf, kInf, 0.f, -0.f,
                               std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::lowest(),
                               std::numeric_limits<float>::max()};
  for (const float t : thresholds) {
    values.push_back(t);
    values.push_back(std::nextafter(t, -kInf));
    values.push_back(std::nextafter(t, kInf));
  }
  std::uniform_real_distribution<float> dist(-2.f, 2.f);
  for (int i = 0; i < 1000; ++i) values.push_back(dist(*rng));
  return values;
}

void CheckAllTargets(const std::vector<float>& thresholds,
                     const bool expect_simd_if_wide) {
  std::mt19937 rng(thresholds.size());
  const std::vector<float> values = ProbeValues(thresholds, &rng);
  for (const int64_t target : hwy::SupportedAndGeneratedTargets()) {
    hwy::SetSupportedTargetsForTest(target);
    const HistogramBinner binner(thresholds);
    std::vector<int32_t> got(values.size());
    binner.Index(values, absl::MakeSpan(got));
    for (size_t i = 0; i < values.size(); ++i) {
      ASSERT_EQ(got[i], ReferenceIndex(thresholds, values[i]))
          << "target=" << hwy::TargetName(target)
          << " num_thresholds=" << thresholds.size() << " x=" << values[i];
    }
    if (!expect_simd_if_wide) {
      EXPECT_FALSE(binner.simd()) << hwy::TargetName(target);
    }
  }
  hwy::SetSupportedTargetsForTest(0);
}

TEST(HistogramBinner, MatchesUpperBound) {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (const int n : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64,
                      65, 100, 127, 128, 129, 255, 256, 257, 511, 512, 1000,
                      1024}) {
    std::vector<float> thresholds(n);
    for (auto& t : thresholds) t = dist(rng);
    std::sort(thresholds.begin(), thresholds.end());
    CheckAllTargets(thresholds, /*expect_simd_if_wide=*/true);
  }
}

TEST(HistogramBinner, DuplicatesAndInfiniteThresholds) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  std::vector<float> thresholds;
  for (int i = 0; i < 70; ++i) thresholds.push_back(static_cast<float>(i / 9));
  CheckAllTargets(thresholds, /*expect_simd_if_wide=*/true);
  thresholds = std::vector<float>(255, 0.5f);
  CheckAllTargets(thresholds, /*expect_simd_if_wide=*/true);
  thresholds = {-kInf, -1.f, 0.f, 0.f, 1.f, kInf, kInf};
  CheckAllTargets(thresholds, /*expect_simd_if_wide=*/true);
}

TEST(HistogramBinner, TooManyThresholdsFallsBackToScalar) {
  std::vector<float> thresholds(4 * 16 * 16 + 1);
  for (size_t i = 0; i < thresholds.size(); ++i) thresholds[i] = i;
  CheckAllTargets(thresholds, /*expect_simd_if_wide=*/false);
}

TEST(HistogramBinner, Empty) {
  const std::vector<float> thresholds;
  const HistogramBinner binner(thresholds);
  EXPECT_FALSE(binner.simd());
  const std::vector<float> values = {0.f, 1.f};
  std::vector<int32_t> got(values.size());
  binner.Index(values, absl::MakeSpan(got));
  EXPECT_THAT(got, testing::ElementsAre(-1, -1));
}

}  // namespace
}  // namespace yggdrasil_decision_forests::model::decision_tree::internal
