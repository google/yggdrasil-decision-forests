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

// Mathematical operations.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_MATH_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_MATH_H_

#include <type_traits>

#include "absl/types/span.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Compute "ceil((double)x, (double)y)" for x and y being positive integers
// values, and without relying on floating point operations.
template <typename T>
T CeilDiV(T x, T y) {
  static_assert(std::is_integral<T>::value, "Integral required.");
  return (x + y - 1) / y;
}

// Computes the median of "values".
//
// Uses the Quick Select algorithm: The average time and space complexity is
// linear. If the number of values is event, return the average of the two
// median values. If empty, returns NaN. "values" should not contain NaNs or
// Infs.
float Median(absl::Span<const float> values);

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_MATH_H_
