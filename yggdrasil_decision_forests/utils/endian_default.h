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

// IWYU pragma: private, include "yggdrasil_decision_forests/utils/endian.h"
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_ENDIAN_DEFAULT_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_ENDIAN_DEFAULT_H_

#include <cstdint>
#include <cstring>

namespace yggdrasil_decision_forests {
namespace utils {

inline uint16_t LittleEndianToHost16(uint16_t v) {
  uint8_t p[2];
  std::memcpy(p, &v, sizeof(v));
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

inline uint32_t LittleEndianToHost32(uint32_t v) {
  uint8_t p[4];
  std::memcpy(p, &v, sizeof(v));
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

inline uint64_t LittleEndianToHost64(uint64_t v) {
  uint8_t p[8];
  std::memcpy(p, &v, sizeof(v));
  return static_cast<uint64_t>(p[0]) | (static_cast<uint64_t>(p[1]) << 8) |
         (static_cast<uint64_t>(p[2]) << 16) |
         (static_cast<uint64_t>(p[3]) << 24) |
         (static_cast<uint64_t>(p[4]) << 32) |
         (static_cast<uint64_t>(p[5]) << 40) |
         (static_cast<uint64_t>(p[6]) << 48) |
         (static_cast<uint64_t>(p[7]) << 56);
}

inline uint16_t HostToLittleEndian16(uint16_t v) {
  uint8_t p[2];
  p[0] = static_cast<uint8_t>(v & 0xFF);
  p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
  uint16_t result;
  std::memcpy(&result, p, sizeof(result));
  return result;
}

inline uint32_t HostToLittleEndian32(uint32_t v) {
  uint8_t p[4];
  p[0] = static_cast<uint8_t>(v & 0xFF);
  p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
  p[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
  p[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
  uint32_t result;
  std::memcpy(&result, p, sizeof(result));
  return result;
}

inline uint64_t HostToLittleEndian64(uint64_t v) {
  uint8_t p[8];
  p[0] = static_cast<uint8_t>(v & 0xFF);
  p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
  p[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
  p[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
  p[4] = static_cast<uint8_t>((v >> 32) & 0xFF);
  p[5] = static_cast<uint8_t>((v >> 40) & 0xFF);
  p[6] = static_cast<uint8_t>((v >> 48) & 0xFF);
  p[7] = static_cast<uint8_t>((v >> 56) & 0xFF);
  uint64_t result;
  std::memcpy(&result, p, sizeof(result));
  return result;
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_ENDIAN_DEFAULT_H_
