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

#include "yggdrasil_decision_forests/utils/endian.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

TEST(EndianTest, LittleEndian16) {
  uint16_t value = 0x1234;
  uint16_t le = HostToLittleEndian16(value);
  EXPECT_EQ(LittleEndianToHost16(le), value);

  // Check manual byte pattern for 0x1234.
  // In little endian, this is 0x34 0x12.
  uint8_t bytes[2] = {0x34, 0x12};
  uint16_t from_bytes;
  std::memcpy(&from_bytes, bytes, 2);
  EXPECT_EQ(LittleEndianToHost16(from_bytes), 0x1234);
}

TEST(EndianTest, LittleEndian32) {
  uint32_t value = 0x12345678;
  uint32_t le = HostToLittleEndian32(value);
  EXPECT_EQ(LittleEndianToHost32(le), value);

  // Check manual byte pattern for 0x12345678.
  // In little endian, this is 0x78 0x56 0x34 0x12.
  uint8_t bytes[4] = {0x78, 0x56, 0x34, 0x12};
  uint32_t from_bytes;
  std::memcpy(&from_bytes, bytes, 4);
  EXPECT_EQ(LittleEndianToHost32(from_bytes), 0x12345678);
}

TEST(EndianTest, LittleEndian64) {
  uint64_t value = 0x123456789ABCDEF0ULL;
  uint64_t le = HostToLittleEndian64(value);
  EXPECT_EQ(LittleEndianToHost64(le), value);

  // Check manual byte pattern for 0x123456789ABCDEF0.
  // In little endian, this is 0xF0 0xDE 0xBC 0x9A 0x78 0x56 0x34 0x12.
  uint8_t bytes[8] = {0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12};
  uint64_t from_bytes;
  std::memcpy(&from_bytes, bytes, 8);
  EXPECT_EQ(LittleEndianToHost64(from_bytes), 0x123456789ABCDEF0ULL);
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
