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

#include "yggdrasil_decision_forests/utils/own_or_borrow.h"

#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
namespace yggdrasil_decision_forests::utils {
using ::testing::ElementsAre;
namespace {

TEST(OwnOrBorrow, Vector) {
  std::vector<int> data1{1, 2, 3, 4};
  std::vector<int> data2{5, 6};

  VectorOwnOrBorrow<int> a;
  EXPECT_TRUE(a.owner());
  EXPECT_THAT(a.values(), ElementsAre());

  a.borrow(data1);
  EXPECT_FALSE(a.owner());
  EXPECT_THAT(a.values(), ElementsAre(1, 2, 3, 4));
  data1[1] = 5;
  EXPECT_THAT(a.values(), ElementsAre(1, 5, 3, 4));

  a.borrow(data2);
  EXPECT_FALSE(a.owner());
  EXPECT_THAT(a.values(), ElementsAre(5, 6));

  a.own(std::move(data1));
  EXPECT_TRUE(data1.empty());
  EXPECT_TRUE(a.owner());
  EXPECT_THAT(a.values(), ElementsAre(1, 5, 3, 4));
  EXPECT_THAT(a.values(), ElementsAre(1, 5, 3, 4));
}

}  // namespace

}  // namespace yggdrasil_decision_forests::utils
