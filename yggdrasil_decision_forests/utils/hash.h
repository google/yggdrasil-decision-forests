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

// Hashing methods.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_HASH_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_HASH_H_

#include <cstdint>

#include "absl/strings/string_view.h"

#include "farmhash.h"
namespace farmhash_namespace = ::util;

namespace yggdrasil_decision_forests {
namespace utils {
namespace hash {

inline uint64_t HashStringViewToUint64(const absl::string_view value) {
  return ::farmhash_namespace::Fingerprint64(value);
}

inline absl::uint128 HashStringViewToUint128(const absl::string_view value) {
  const auto sign_fh = ::farmhash_namespace::Fingerprint128(value);
  return absl::MakeUint128(::farmhash_namespace::Uint128Low64(sign_fh),
                           ::farmhash_namespace::Uint128High64(sign_fh));
}

inline uint64_t HashInt64ToUint64(int64_t value) {
  return ::farmhash_namespace::Fingerprint(value);
}

}  // namespace hash
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_HASH_H_
