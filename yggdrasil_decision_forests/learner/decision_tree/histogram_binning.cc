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

#include "yggdrasil_decision_forests/learner/decision_tree/histogram_binning.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "absl/types/span.h"
#include "yggdrasil_decision_forests/utils/logging.h"

// Compiles the kernel once per Highway target, picked at runtime.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "yggdrasil_decision_forests/learner/decision_tree/histogram_binning.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace yggdrasil_decision_forests::decision_tree::internal {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

using DF = hn::CappedTag<float, HistogramBinner::kMaxLanes>;

size_t NumLanes() { return hn::Lanes(DF()); }

// Count of thresholds <= x = L - CountTrue(x < thr) per vector. Lt (not Ge)
// keeps std::upper_bound's NaN behavior. The +inf padding never counts for
// finite x.
void BinIndices(const float* HWY_RESTRICT padded_thresholds,
                const float* HWY_RESTRICT group_maxima,
                const size_t num_coarse_vectors, const size_t num_groups,
                const size_t num_thresholds, const float* HWY_RESTRICT values,
                const size_t num_values, int32_t* HWY_RESTRICT bin_idxs) {
  const DF d;
  const size_t num_lanes = hn::Lanes(d);
  for (size_t i = 0; i < num_values; ++i) {
    const auto x = hn::Set(d, values[i]);
    // Number of groups whose thresholds are all <= x.
    size_t group = 0;
    for (size_t c = 0; c < num_coarse_vectors; ++c) {
      group += num_lanes - hn::CountTrue(d, hn::Lt(x, hn::LoadU(d, group_maxima +
                                                               c * num_lanes)));
    }
    if (group >= num_groups) {
      bin_idxs[i] = static_cast<int32_t>(num_thresholds) - 1;
      continue;
    }
    const size_t offset = group * num_lanes;
    const size_t count =
        offset + num_lanes -
        hn::CountTrue(d, hn::Lt(x, hn::LoadU(d, padded_thresholds + offset)));
    bin_idxs[i] = static_cast<int32_t>(std::min(count, num_thresholds)) - 1;
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace yggdrasil_decision_forests::decision_tree::internal
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace yggdrasil_decision_forests::decision_tree::internal {

HWY_EXPORT(NumLanes);
HWY_EXPORT(BinIndices);

HistogramBinner::HistogramBinner(absl::Span<const float> sorted_thresholds)
    : thresholds_(sorted_thresholds) {
  const size_t num_lanes = HWY_DYNAMIC_DISPATCH(NumLanes)();
  const size_t n = thresholds_.size();
  if (num_lanes < 4 || n == 0) return;  // No SIMD gain on 1-2 lane targets.
  const size_t num_groups = (n + num_lanes - 1) / num_lanes;
  const size_t num_coarse_vectors = (num_groups + num_lanes - 1) / num_lanes;
  if (num_coarse_vectors > kMaxCoarseVectors) return;

  constexpr float kInf = std::numeric_limits<float>::infinity();
  std::copy(thresholds_.begin(), thresholds_.end(), padded_thresholds_);
  std::fill(padded_thresholds_ + n, padded_thresholds_ + num_groups * num_lanes,
            kInf);
  for (size_t g = 0; g < num_groups; ++g) {
    group_maxima_[g] = thresholds_[std::min((g + 1) * num_lanes, n) - 1];
  }
  std::fill(group_maxima_ + num_groups,
            group_maxima_ + num_coarse_vectors * num_lanes, kInf);
  num_lanes_ = num_lanes;
  num_groups_ = num_groups;
  num_coarse_vectors_ = num_coarse_vectors;
}

int32_t HistogramBinner::ScalarIndex(const float value) const {
  const auto it =
      std::upper_bound(thresholds_.begin(), thresholds_.end(), value);
  return static_cast<int32_t>(it - thresholds_.begin()) - 1;
}

void HistogramBinner::Index(absl::Span<const float> values,
                            absl::Span<int32_t> bin_idxs) const {
  DCHECK_EQ(values.size(), bin_idxs.size());
  if (!simd()) {
    for (size_t i = 0; i < values.size(); ++i) {
      bin_idxs[i] = ScalarIndex(values[i]);
    }
    return;
  }
  HWY_DYNAMIC_DISPATCH(BinIndices)
  (padded_thresholds_, group_maxima_, num_coarse_vectors_, num_groups_,
   thresholds_.size(), values.data(), values.size(), bin_idxs.data());
#ifndef NDEBUG
  for (size_t i = 0; i < values.size(); ++i) {
    DCHECK_EQ(bin_idxs[i], ScalarIndex(values[i]))
        << "SIMD binning disagrees with std::upper_bound for " << values[i];
  }
#endif
}

}  // namespace yggdrasil_decision_forests::decision_tree::internal
#endif  // HWY_ONCE
