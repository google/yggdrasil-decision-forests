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

#include "yggdrasil_decision_forests/utils/shap.h"

#include <cmath>
#include <vector>

#include "absl/log/check.h"

namespace yggdrasil_decision_forests::utils::shap {

namespace internal {

bool operator==(const PathItem& a, const PathItem& b) {
  return a.column_idx == b.column_idx && a.zero_fraction == b.zero_fraction &&
         a.one_fraction == b.one_fraction && a.weight == b.weight;
}

void extend(const double zero_fraction,  // "p_z" in paper.
            const double one_fraction,   // "p_o" in paper.
            const int attribute_idx,     // "p_i" in paper.
            Path& path                   // "m" in paper.
) {
  DCHECK_GT(zero_fraction, 0);
  const int n = path.size();
  path.push_back({
      .column_idx = attribute_idx,
      .zero_fraction = zero_fraction,
      .one_fraction = one_fraction,
      .weight = path.empty() ? 1.f : 0.f,
  });
  for (int i = n - 1; i >= 0; i--) {
    // The paper is 1-indexed.
    path[i + 1].weight += one_fraction * path[i].weight * (i + 1) / (n + 1);
    DCHECK(!std::isnan(path[i + 1].weight));
    path[i].weight *= zero_fraction * (n - i) / (n + 1);
    DCHECK(!std::isnan(path[i].weight));
  }
}

void unwind(const int path_idx, Path& path) {
  const int n = path.size() - 1;  // Size of the path after the "pop" back.
  const double one_fraction = path[path_idx].one_fraction;
  const double zero_fraction = path[path_idx].zero_fraction;
  DCHECK_GT(zero_fraction, 0);
  double save_weight = path.back().weight;

  for (int j = n - 1; j >= 0; j--) {
    if (one_fraction != 0) {
      const double tmp_weight = path[j].weight;
      path[j].weight = save_weight * (n + 1) / ((j + 1) * one_fraction);
      DCHECK(!std::isnan(path[j].weight));
      save_weight =
          tmp_weight - path[j].weight * zero_fraction * (n - j) / (n + 1);
    } else {
      path[j].weight = path[j].weight * (n + 1) / (zero_fraction * (n - j));
      DCHECK(!std::isnan(path[j].weight));
    }
  }

  for (int j = path_idx; j < n; j++) {
    path[j].column_idx = path[j + 1].column_idx;
    path[j].zero_fraction = path[j + 1].zero_fraction;
    path[j].one_fraction = path[j + 1].one_fraction;
  }

  // Note: The paper contains an error. The pop should be done at the end
  // (instead of at the start). Otherwise, the loop above can go out of bounds.
  // The SHAP python package tracks the size of the array, but does not pop the
  // value. So this bug does not manifest.
  path.pop_back();
}

double unwound_sum(const int path_idx, Path& path) {
  const int unwound_path_length = path.size() - 1;
  const double one_fraction = path[path_idx].one_fraction;
  const double zero_fraction = path[path_idx].zero_fraction;
  DCHECK_GT(zero_fraction, 0);
  double next_one_portion = path.back().weight;
  double total = 0;

  if (one_fraction != 0) {
    for (int i = unwound_path_length - 1; i >= 0; i--) {
      const double tmp = next_one_portion / (one_fraction * (i + 1));
      total += tmp;
      next_one_portion =
          path[i].weight - tmp * zero_fraction * (unwound_path_length - i);
    }
  } else {
    for (int i = unwound_path_length - 1; i >= 0; i--) {
      total += path[i].weight / (zero_fraction * (unwound_path_length - i));
    }
  }
  return total * (unwound_path_length + 1);
}
}  // namespace internal
}  // namespace yggdrasil_decision_forests::utils::shap
