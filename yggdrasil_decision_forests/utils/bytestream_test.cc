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

#include "yggdrasil_decision_forests/utils/bytestream.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

TEST(Bytestream, StringInputByteStream) {
  StringInputByteStream stream("Hello world");
  char buffer[128];
  EXPECT_EQ(stream.ReadUpTo(buffer, 5).value(), 5);
  EXPECT_EQ(stream.ReadUpTo(buffer, 1).value(), 1);
  EXPECT_EQ(stream.ReadUpTo(buffer, 5).value(), 5);
  EXPECT_EQ(stream.ReadUpTo(buffer, 50).value(), 0);
}

TEST(Bytestream, StringOutputByteStream) {
  StringOutputByteStream stream;
  EXPECT_OK(stream.Write("ABC"));
  EXPECT_OK(stream.Write(""));
  EXPECT_OK(stream.Write("DEF"));
  EXPECT_EQ(stream.ToString(), "ABCDEF");
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
