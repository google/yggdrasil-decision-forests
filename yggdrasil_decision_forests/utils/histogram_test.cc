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

#include "yggdrasil_decision_forests/utils/histogram.h"

#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace histogram {
namespace {

using testing::ElementsAre;

TEST(Histogram, UniformInt) {
  EXPECT_EQ(Histogram<int>::MakeUniform({}).ToString(),
            R"(Count: 0 Average: 0 StdDev: 0
Min: 0 Max: 0 Ignored: 0
----------------------------------------------
)");

  EXPECT_EQ(Histogram<int>::MakeUniform({2}).ToString(),
            R"(Count: 1 Average: 2 StdDev: 0
Min: 2 Max: 2 Ignored: 0
----------------------------------------------
[ 2, 2] 1 100.00% 100.00% ##########
)");

  EXPECT_EQ(Histogram<int>::MakeUniform({2, 2}).ToString(),
            R"(Count: 2 Average: 2 StdDev: 0
Min: 2 Max: 2 Ignored: 0
----------------------------------------------
[ 2, 2] 2 100.00% 100.00% ##########
)");

  EXPECT_EQ(Histogram<int>::MakeUniform({2, 2, 3}).ToString(),
            R"(Count: 3 Average: 2.33333 StdDev: 0.471405
Min: 2 Max: 3 Ignored: 0
----------------------------------------------
[ 2, 3) 2  66.67%  66.67% ##########
[ 3, 3] 1  33.33% 100.00% #####
)");

  EXPECT_EQ(Histogram<int>::MakeUniform({2, 2, 3, 5}).ToString(),
            R"(Count: 4 Average: 3 StdDev: 1.22474
Min: 2 Max: 5 Ignored: 0
----------------------------------------------
[ 2, 3) 2  50.00%  50.00% ##########
[ 3, 4) 1  25.00%  75.00% #####
[ 4, 5) 0   0.00%  75.00%
[ 5, 5] 1  25.00% 100.00% #####
)");

  EXPECT_EQ(Histogram<int>::MakeUniform({2, 2, 3, 5, -4}).ToString(),
            R"(Count: 5 Average: 1.6 StdDev: 3.00666
Min: -4 Max: 5 Ignored: 0
----------------------------------------------
[ -4, -3) 1  20.00%  20.00% #####
[ -3, -2) 0   0.00%  20.00%
[ -2, -1) 0   0.00%  20.00%
[ -1,  0) 0   0.00%  20.00%
[  0,  1) 0   0.00%  20.00%
[  1,  2) 0   0.00%  20.00%
[  2,  3) 2  40.00%  60.00% ##########
[  3,  4) 1  20.00%  80.00% #####
[  4,  5) 0   0.00%  80.00%
[  5,  5] 1  20.00% 100.00% #####
)");

  EXPECT_EQ(Histogram<int>::MakeUniform({-50, 50}, /*max_bins=*/5).ToString(),
            R"(Count: 2 Average: 0 StdDev: 50
Min: -50 Max: 50 Ignored: 0
----------------------------------------------
[ -50, -30) 1  50.00%  50.00% ##########
[ -30, -10) 0   0.00%  50.00%
[ -10,  10) 0   0.00%  50.00%
[  10,  30) 0   0.00%  50.00%
[  30,  50] 1  50.00% 100.00% ##########
)");

  EXPECT_EQ(Histogram<int>::MakeUniform({-50, 50}, /*max_bins=*/1).ToString(),
            R"(Count: 2 Average: 0 StdDev: 50
Min: -50 Max: 50 Ignored: 0
----------------------------------------------
[ -50,  50] 2 100.00% 100.00% ##########
)");

  EXPECT_EQ(
      Histogram<int>::MakeUniform({-50, 50}, /*max_bins=*/5, /*weights=*/{4, 2})
          .ToString(),
      R"(Count: 6 Average: -16.6667 StdDev: 89.7527
Min: -50 Max: 50 Ignored: 0
----------------------------------------------
[ -50, -30) 4  66.67%  66.67% ##########
[ -30, -10) 0   0.00%  66.67%
[ -10,  10) 0   0.00%  66.67%
[  10,  30) 0   0.00%  66.67%
[  30,  50] 2  33.33% 100.00% #####
)");
}

TEST(Histogram, UniformFloat) {
  EXPECT_EQ(Histogram<float>::MakeUniform({-50, 20.5, 20.7, 50}, /*max_bins=*/5)
                .ToString(),
            R"(Count: 4 Average: 10.3 StdDev: 36.8252
Min: -50 Max: 50 Ignored: 0
----------------------------------------------
[   -50, -29.8) 1  25.00%  25.00% #####
[ -29.8,  -9.6) 0   0.00%  25.00%
[  -9.6,  10.6) 0   0.00%  25.00%
[  10.6,  30.8) 2  50.00%  75.00% ##########
[  30.8,    50] 1  25.00% 100.00% #####
)");
}

TEST(Histogram, UniformIntRnd) {
  std::vector<int> values;
  utils::RandomEngine rnd;
  std::normal_distribution<double> dist(20, 40);
  for (int i = 0; i < 10000; i++) {
    values.push_back(dist(rnd));
  }
  YDF_LOG(INFO) << Histogram<int>::MakeUniform(values).ToString();
}

TEST(Histogram, UniformFloatRnd) {
  std::vector<float> values;
  utils::RandomEngine rnd;
  std::normal_distribution<double> dist(20, 40);
  for (int i = 0; i < 10000; i++) {
    values.push_back(dist(rnd));
  }
  YDF_LOG(INFO) << Histogram<float>::MakeUniform(values).ToString();
}

TEST(Histogram, UniformFloatWithWeightsRnd) {
  utils::RandomEngine rnd;
  std::normal_distribution<double> dist(20, 40);
  std::uniform_real_distribution<double> dist2(0, 1);
  std::vector<float> values, weights;
  for (int i = 0; i < 10000; i++) {
    values.push_back(dist(rnd));
    weights.push_back(dist2(rnd));
  }
  YDF_LOG(INFO)
      << Histogram<float>::MakeUniform(values, 10, weights).ToString();
}

TEST(Metric, BucketizedContainer) {
  // The bin boundaries are [1, 1.5, 2, 2.5, 3].
  BucketizedContainer<float, int> container(1.f, 3.f, 4);
  // Bin 1
  container[1.f]++;
  container[1.2f]++;
  // Bin 2
  container[1.5f]++;
  // Bin 3
  container[2.2f]++;
  // Bin 3
  container[2.7f]++;
  container[3.f]++;
  EXPECT_THAT(container.ContentArray(), ElementsAre(2, 1, 1, 2));
}

TEST(Metric, BucketizedContainerCollapsed) {
  BucketizedContainer<float, int> container(1.f, 1.f, 3);
  container[1.f] = 5;
  EXPECT_THAT(container.ContentArray(), ElementsAre(0, 0, 5));
}

}  // namespace
}  // namespace histogram
}  // namespace utils
}  // namespace yggdrasil_decision_forests
