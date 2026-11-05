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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_HISTOGRAM_BINNING_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_HISTOGRAM_BINNING_H_

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"

namespace yggdrasil_decision_forests::decision_tree::internal {

// Maps values to histogram bins: the index of the last threshold <= value, or
// -1 if none. Same result as std::upper_bound(thresholds, value) - begin - 1.
// Two-level SIMD search (Highway dynamic dispatch), scalar fallback otherwise.
class HistogramBinner {
 public:
  // Lane cap: larger vectors (e.g. wide SVE) would only grow the tables.
  static constexpr size_t kMaxLanes = 16;
  // The group maxima table spans at most this many vectors.
  static constexpr size_t kMaxCoarseVectors = 4;

  // `sorted_thresholds` must be sorted ascending and outlive the binner.
  explicit HistogramBinner(absl::Span<const float> sorted_thresholds);

  // bin_idxs[i] = bin of values[i]. Both spans have the same size.
  void Index(absl::Span<const float> values, absl::Span<int32_t> bin_idxs) const;

  // Reference implementation (std::upper_bound).
  int32_t ScalarIndex(float value) const;

  // True iff Index() runs the SIMD kernel.
  bool simd() const { return num_lanes_ > 0; }

 private:
  absl::Span<const float> thresholds_;
  size_t num_lanes_ = 0;  // 0 = scalar fallback.
  size_t num_groups_ = 0;
  size_t num_coarse_vectors_ = 0;
  // Thresholds padded with +inf to num_groups_ * num_lanes_. Inline (no heap
  // allocation per split search), only the used prefix is written.
  alignas(64) float padded_thresholds_[kMaxCoarseVectors * kMaxLanes * kMaxLanes];
  // Last real threshold of each group, padded with +inf.
  alignas(64) float group_maxima_[kMaxCoarseVectors * kMaxLanes];
};

}  // namespace yggdrasil_decision_forests::decision_tree::internal

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_HISTOGRAM_BINNING_H_
