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

#include "yggdrasil_decision_forests/utils/circular_buffer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

TEST(CircularBuffer, Base) {
  CircularBuffer<int> b(4);
  EXPECT_EQ(b.size(), 0);
  EXPECT_TRUE(b.empty());
  EXPECT_THAT(b.to_vector(), testing::ElementsAre());

  b.push_back(1);
  EXPECT_EQ(b.size(), 1);
  EXPECT_FALSE(b.empty());
  EXPECT_THAT(b.to_vector(), testing::ElementsAre(1));

  b.push_back(2);
  EXPECT_EQ(b.size(), 2);
  EXPECT_THAT(b.to_vector(), testing::ElementsAre(1, 2));

  b.push_front(3);
  EXPECT_EQ(b.size(), 3);
  EXPECT_THAT(b.to_vector(), testing::ElementsAre(3, 1, 2));

  b.push_front(4);
  EXPECT_EQ(b.size(), 4);
  EXPECT_THAT(b.to_vector(), testing::ElementsAre(4, 3, 1, 2));
  EXPECT_TRUE(b.full());
  EXPECT_EQ(b.front(), 4);
  EXPECT_EQ(b.back(), 2);

  b.pop_front();
  EXPECT_EQ(b.size(), 3);
  EXPECT_THAT(b.to_vector(), testing::ElementsAre(3, 1, 2));

  b.pop_front();
  b.pop_front();
  EXPECT_EQ(b.size(), 1);
  EXPECT_THAT(b.to_vector(), testing::ElementsAre(2));

  b.push_back(5);
  b.push_back(6);
  b.pop_front();  // 2
  b.pop_front();  // 5
  b.push_back(7);
  b.push_back(8);
  b.pop_front();  // 6
  b.pop_front();  // 7
  EXPECT_THAT(b.to_vector(), testing::ElementsAre(8));

  b.clear();
  EXPECT_EQ(b.size(), 0);
  EXPECT_TRUE(b.empty());
  EXPECT_THAT(b.to_vector(), testing::ElementsAre());

  b.push_back(1);
  EXPECT_EQ(b.size(), 1);
  EXPECT_FALSE(b.empty());
  EXPECT_THAT(b.to_vector(), testing::ElementsAre(1));
}

TEST(CircularBuffer, Empty) {
  CircularBuffer<int> b;
  EXPECT_EQ(b.size(), 0);
  EXPECT_TRUE(b.empty());
}

TEST(CircularBuffer, Resize) {
  CircularBuffer<int> b(1);
  EXPECT_EQ(b.size(), 0);
  EXPECT_TRUE(b.empty());

  b.clear_and_resize(3);
  EXPECT_EQ(b.size(), 0);
  EXPECT_TRUE(b.empty());

  b.push_back(2);
  b.push_back(3);
  b.push_front(1);
  EXPECT_THAT(b.to_vector(), testing::ElementsAre(1, 2, 3));
  EXPECT_TRUE(b.full());
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
