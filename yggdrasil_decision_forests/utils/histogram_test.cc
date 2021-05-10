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
}

TEST(Histogram, UniformIntRnd) {
  std::vector<int> values;
  utils::RandomEngine rnd;
  std::normal_distribution<double> dist(20, 40);
  for (int i = 0; i < 10000; i++) {
    values.push_back(dist(rnd));
  }
  LOG(INFO) << Histogram<int>::MakeUniform(values).ToString();
}

}  // namespace
}  // namespace histogram
}  // namespace utils
}  // namespace yggdrasil_decision_forests
