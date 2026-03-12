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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_ENDIAN_GOOGLE_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_ENDIAN_GOOGLE_H_

#include <cstdint>

#include "util/endian/endian.h"

namespace yggdrasil_decision_forests {
namespace utils {

inline uint16_t LittleEndianToHost16(uint16_t v) {
  return LittleEndian::ToHost16(v);
}
inline uint32_t LittleEndianToHost32(uint32_t v) {
  return LittleEndian::ToHost32(v);
}
inline uint64_t LittleEndianToHost64(uint64_t v) {
  return LittleEndian::ToHost64(v);
}

inline uint16_t HostToLittleEndian16(uint16_t v) {
  return LittleEndian::FromHost16(v);
}
inline uint32_t HostToLittleEndian32(uint32_t v) {
  return LittleEndian::FromHost32(v);
}
inline uint64_t HostToLittleEndian64(uint64_t v) {
  return LittleEndian::FromHost64(v);
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_ENDIAN_GOOGLE_H_
