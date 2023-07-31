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

// Generates uids.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_UID_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_UID_H_

#include <random>

#include "absl/random/random.h"
#include "absl/strings/str_format.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Generates a low quality unique id.
inline std::string GenUniqueId() {
  absl::BitGen bitgen;
  return absl::StrFormat(
      "%04x-%04x-%04x-%04x", absl::Uniform(bitgen, 0, 0x10000),
      absl::Uniform(bitgen, 0, 0x10000), absl::Uniform(bitgen, 0, 0x10000),
      absl::Uniform(bitgen, 0, 0x10000));
}

// Generates a low quality unique id.
inline uint64_t GenUniqueIdUint64() {
  absl::BitGen bitgen;
  using nl = std::numeric_limits<uint64_t>;
  return absl::Uniform(bitgen, nl::min(), nl::max());
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_UID_H_
