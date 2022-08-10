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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_CAST_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_CAST_H_

#include <type_traits>

namespace yggdrasil_decision_forests {
namespace utils {

// Casts a parent class to one of its child class. If compiled in debug mode and
// if RTTI is active, checks that the case if valid at runtime.
template <typename To, typename From>  // use like this: down_cast<T*>(foo);
inline To down_cast(From* f) {         // so we only accept pointers
  static_assert((std::is_base_of<From, std::remove_pointer_t<To>>::value),
                "target type not derived from source type");

  // We skip the assert and hence the dynamic_cast if RTTI is disabled.
#if !defined(__GNUC__) || defined(__GXX_RTTI)
  // Uses RTTI in dbg and fastbuild. asserts are disabled in opt builds.
  assert(f == nullptr || dynamic_cast<To>(f) != nullptr);
#endif  // !defined(__GNUC__) || defined(__GXX_RTTI)

  return static_cast<To>(f);
}

// Similar to "down_cast" above, but for references (instead of pointers).
template <typename To, typename From>
inline To down_cast(From& f) {
  static_assert(std::is_lvalue_reference<To>::value,
                "target type not a reference");
  static_assert((std::is_base_of<From, std::remove_reference_t<To>>::value),
                "target type not derived from source type");

  // We skip the assert and hence the dynamic_cast if RTTI is disabled.
#if !defined(__GNUC__) || defined(__GXX_RTTI)
  // RTTI: debug mode only
  assert(dynamic_cast<std::remove_reference_t<To>*>(&f) != nullptr);
#endif  // !defined(__GNUC__) || defined(__GXX_RTTI)

  return static_cast<To>(f);
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_CAST_H_
