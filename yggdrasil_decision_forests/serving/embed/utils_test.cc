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

// Test the utils

#include "yggdrasil_decision_forests/serving/embed/utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests::serving::embed {
namespace {

TEST(Embed, CheckModelName) {
  EXPECT_OK(CheckModelName("my_model_123"));
  EXPECT_THAT(CheckModelName("my-model"),
              test::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CheckModelName("my model"),
              test::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Embed, CheckFeatureName) {
  EXPECT_OK(CheckFeatureName("my_feature_123"));
  EXPECT_THAT(CheckFeatureName("my-feature"),
              test::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CheckFeatureName("my feature"),
              test::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_OK(CheckFeatureName("A<B"));
}

TEST(Embed, StringToUpperCaseVariable) {
  EXPECT_EQ(StringToConstantSymbol("A nice-and-cold_day 123!"),
            "A_NICE_AND_COLD_DAY_123");
  EXPECT_EQ(StringToConstantSymbol("123AAA!"), "V123AAA");
  EXPECT_EQ(StringToConstantSymbol(""), "");

  EXPECT_EQ(StringToConstantSymbol("A<B"), "ALtB");
}

TEST(Embed, StringToLowerCaseVariable) {
  EXPECT_EQ(StringToVariableSymbol("A nice-and-cold_day 123!"),
            "a_nice_and_cold_day_123");
  EXPECT_EQ(StringToVariableSymbol("123AAA!"), "v123aaa");
  EXPECT_EQ(StringToVariableSymbol(""), "");
  EXPECT_EQ(StringToVariableSymbol("A<B"), "aLtb");
}

TEST(Embed, StringToStructSymbol) {
  EXPECT_EQ(StringToStructSymbol("A nice-and-cold_day 123!"),
            "ANiceAndColdDay123");
  EXPECT_EQ(StringToStructSymbol("123AAA!"), "V123AAA");
  EXPECT_EQ(StringToStructSymbol(""), "");
  EXPECT_EQ(StringToStructSymbol("A<B"), "ALtB");
}

TEST(Embed, MaxUnsignedValueToNumBytes) {
  EXPECT_EQ(MaxUnsignedValueToNumBytes(0), 1);
  EXPECT_EQ(MaxUnsignedValueToNumBytes(255), 1);
  EXPECT_EQ(MaxUnsignedValueToNumBytes(256), 2);
  EXPECT_EQ(MaxUnsignedValueToNumBytes(65535), 2);
  EXPECT_EQ(MaxUnsignedValueToNumBytes(65536), 4);
  EXPECT_EQ(MaxUnsignedValueToNumBytes(4294967295), 4);
}

TEST(Embed, MaxSignedValueToNumBytes) {
  EXPECT_EQ(MaxSignedValueToNumBytes(0), 1);
  EXPECT_EQ(MaxSignedValueToNumBytes(127), 1);
  EXPECT_EQ(MaxSignedValueToNumBytes(-128), 1);
  EXPECT_EQ(MaxSignedValueToNumBytes(128), 2);
  EXPECT_EQ(MaxSignedValueToNumBytes(-129), 2);
  EXPECT_EQ(MaxSignedValueToNumBytes(32767), 2);
  EXPECT_EQ(MaxSignedValueToNumBytes(-32768), 2);
  EXPECT_EQ(MaxSignedValueToNumBytes(32768), 4);
  EXPECT_EQ(MaxSignedValueToNumBytes(-32769), 4);
  EXPECT_EQ(MaxSignedValueToNumBytes(2147483647), 4);
  EXPECT_EQ(MaxSignedValueToNumBytes(-2147483648), 4);
}

TEST(Embed, DTypeToCCType) {
  EXPECT_EQ(DTypeToCCType(proto::DType::INT8), "int8_t");
  EXPECT_EQ(DTypeToCCType(proto::DType::INT16), "int16_t");
  EXPECT_EQ(DTypeToCCType(proto::DType::INT32), "int32_t");
  EXPECT_EQ(DTypeToCCType(proto::DType::FLOAT32), "float");
}

TEST(Embed, NumLeavesToNumNodes) {
  EXPECT_EQ(NumLeavesToNumNodes(0), 0);
  EXPECT_EQ(NumLeavesToNumNodes(1), 1);
  EXPECT_EQ(NumLeavesToNumNodes(2), 3);
  EXPECT_EQ(NumLeavesToNumNodes(3), 5);
}

TEST(Embed, UnsignedIntegerToDtype) {
  EXPECT_EQ(UnsignedIntegerToDtype(1), proto::DType::UINT8);
  EXPECT_EQ(UnsignedIntegerToDtype(2), proto::DType::UINT16);
  EXPECT_EQ(UnsignedIntegerToDtype(4), proto::DType::UINT32);
}

TEST(NumBytesToMaxUnsignedValue, Basic) {
  EXPECT_EQ(NumBytesToMaxUnsignedValue(1), 0xff);
  EXPECT_EQ(NumBytesToMaxUnsignedValue(2), 0xffff);
  EXPECT_EQ(NumBytesToMaxUnsignedValue(4), 0xffffffff);
}

TEST(QuoteString, Basic) {
  EXPECT_EQ(QuoteString("hello"), "\"hello\"");
  EXPECT_EQ(QuoteString("hello\"world"), "\"hello\\\"world\"");
  EXPECT_EQ(QuoteString("hello\nworld"), "\"hello\\nworld\"");
}

}  // namespace
}  // namespace yggdrasil_decision_forests::serving::embed
