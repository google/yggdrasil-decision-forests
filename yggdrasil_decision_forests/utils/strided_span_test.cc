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

#include "yggdrasil_decision_forests/utils/strided_span.h"

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"

namespace yggdrasil_decision_forests::utils {
namespace {

static_assert(
    !std::is_constructible_v<StridedSpan<bool>, const std::vector<bool>&>);

template <typename T>
std::vector<T> ToVector(const StridedSpan<T> span) {
  std::vector<T> result;
  for (size_t i = 0; i < span.size(); i++) {
    result.push_back(span[i]);
  }
  return result;
}

TEST(StridedSpan, Default) {
  const StridedSpan<int> empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.size(), 0);
  EXPECT_EQ(empty.item_stride(), 1);
  EXPECT_EQ(empty.data(), nullptr);
  EXPECT_THAT(ToVector(empty), testing::IsEmpty());
}

TEST(StridedSpan, Contiguous) {
  std::vector<float> values = {1.f, 2.f, 3.f};

  const StridedSpan<float> from_vector = values;
  EXPECT_FALSE(from_vector.empty());
  EXPECT_EQ(from_vector.size(), 3);
  EXPECT_EQ(from_vector.item_stride(), 1);
  EXPECT_EQ(from_vector.data(), values.data());
  EXPECT_THAT(ToVector(from_vector), testing::ElementsAre(1.f, 2.f, 3.f));

  const std::array<float, 3> array_values = {1.f, 2.f, 3.f};
  const StridedSpan<float> from_array = array_values;
  EXPECT_THAT(ToVector(from_array), testing::ElementsAre(1.f, 2.f, 3.f));

  const StridedSpan<float> from_const_span = absl::MakeConstSpan(values);
  EXPECT_THAT(ToVector(from_const_span), testing::ElementsAre(1.f, 2.f, 3.f));

  const StridedSpan<float> from_mutable_span = absl::MakeSpan(values);
  EXPECT_THAT(ToVector(from_mutable_span), testing::ElementsAre(1.f, 2.f, 3.f));
}

TEST(StridedSpan, MatrixColumn) {
  // Row-major matrix with 3 rows and 2 columns.
  const std::vector<int> matrix = {1, 2,  //
                                   3, 4,  //
                                   5, 6};
  const StridedSpan<int> col_0(matrix.data(), /*size=*/3, /*item_stride=*/2);
  const StridedSpan<int> col_1(matrix.data() + 1, /*size=*/3,
                               /*item_stride=*/2);
  EXPECT_THAT(ToVector(col_0), testing::ElementsAre(1, 3, 5));
  EXPECT_THAT(ToVector(col_1), testing::ElementsAre(2, 4, 6));
  EXPECT_THAT(ToVector(col_1.subspan(3, 0)), testing::IsEmpty());
}

TEST(StridedSpan, NegativeStride) {
  const std::vector<int> values = {1, 2, 3, 4};
  const StridedSpan<int> reversed(values.data() + 3, /*size=*/4,
                                  /*item_stride=*/-1);
  EXPECT_THAT(ToVector(reversed), testing::ElementsAre(4, 3, 2, 1));
  EXPECT_THAT(ToVector(reversed.subspan(1, 2)), testing::ElementsAre(3, 2));
  EXPECT_THAT(ToVector(reversed.subspan(4, 0)), testing::IsEmpty());
}

TEST(StridedSpan, ZeroStride) {
  const int value = 7;
  const StridedSpan<int> repeated(&value, /*size=*/3, /*item_stride=*/0);
  EXPECT_THAT(ToVector(repeated), testing::ElementsAre(7, 7, 7));
  EXPECT_THAT(ToVector(repeated.subspan(1, 2)), testing::ElementsAre(7, 7));
}

TEST(StridedSpan, Subspan) {
  const std::vector<int> values = {0, 1, 2, 3, 4, 5, 6};
  const StridedSpan<int> even(values.data(), /*size=*/4, /*item_stride=*/2);
  EXPECT_THAT(ToVector(even.subspan(1, 2)), testing::ElementsAre(2, 4));
  EXPECT_THAT(ToVector(even.subspan(4, 0)), testing::IsEmpty());
  EXPECT_THAT(ToVector(even.subspan(0, 4)), testing::ElementsAre(0, 2, 4, 6));
}

}  // namespace
}  // namespace yggdrasil_decision_forests::utils
