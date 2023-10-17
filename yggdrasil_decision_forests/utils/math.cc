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

#include "yggdrasil_decision_forests/utils/math.h"

#include <stddef.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "absl/types/span.h"
#include "yggdrasil_decision_forests/utils/logging.h"  // IWYU pragma: keep

namespace yggdrasil_decision_forests::utils {
namespace {

// Returns the "target_idx"-th smallest value in "values".
// "values" is a non-sorted array containing non infinite or nan values.
// The content of "values" is reordered during the computation.
// "values" cannot be empty.
float QuickSelect(std::vector<float>& values, size_t target_idx) {
  DCHECK_GT(values.size(), 0);
  // Boundaries of the search window.
  size_t left = 0;
  // Using a "right" instead of an "end" simplifies the code.
  size_t right = values.size() - 1;

  while (true) {
    DCHECK_LE(right, values.size());
    DCHECK_LE(left, values.size());
    DCHECK_LE(left, right) << "The left index cannot move past the right index";
    DCHECK_GE(target_idx, left) << "target_idx should be in [left, right]";
    DCHECK_LE(target_idx, right) << "target_idx should be in [left, right]";

    if (left == right) {
      return values[left];
    }

    // Pivot the values around "pivot_selector".
    // std::partition cannot be used as it does not guarantee that the pivoted
    // value will be located at the output pivot index.
    //
    // Note: This code can be sped up using the Hoare algorithm.
    using std::swap;
    const float pivot_value = values[target_idx];
    size_t pivot_idx = left;
    swap(values[target_idx], values[right]);
    for (size_t i = left; i < right; ++i) {
      if (values[i] < pivot_value) {
        swap(values[i], values[pivot_idx]);
        pivot_idx++;
      }
    }
    swap(values[pivot_idx], values[right]);

    // Select the side containing the target index.
    if (pivot_idx == target_idx) {
      return values[pivot_idx];
    } else if (target_idx < pivot_idx) {
      right = pivot_idx - 1;
    } else {
      left = pivot_idx + 1;
    }
  }
}

}  // namespace

float Median(const absl::Span<const float> values) {
  if (values.empty()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<float> working_values = {values.begin(), values.end()};
  const size_t half_size = working_values.size() / 2;
  if (values.size() % 2 == 1) {
    return QuickSelect(working_values, half_size);
  } else {
    return (QuickSelect(working_values, half_size) +
            QuickSelect(working_values, half_size - 1)) /
           2;
  }
}

}  // namespace yggdrasil_decision_forests::utils
