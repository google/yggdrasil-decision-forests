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

// Implementation of a reliability diagram.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_RELIABILITY_DIAGRAM_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_RELIABILITY_DIAGRAM_H_

#include <cstddef>
#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/utils/smoothed_pav_calibration_fit.h"

namespace yggdrasil_decision_forests::utils::reliability_diagram {

// Wraps BinAccumulator/accumulate_bins for incremental (streaming) updates,
// and exposes per-bin diagnostics (mean predicted, mean observed, count) plus
// summary statistics (ECE, MCE), e.g. for plotting or logging.
class ReliabilityDiagram {
 public:
  explicit ReliabilityDiagram(std::size_t n_bins)
      : n_bins_(n_bins), bins_(n_bins) {}

  explicit ReliabilityDiagram(const std::vector<BinAccumulator>& bins)
      : n_bins_(bins.size()), bins_(bins) {}

  // Incrementally folds in one chunk of (p, y) pairs. Safe to call
  // repeatedly across chunks or streamed points -- a thin wrapper
  // around accumulate_bins, so it never drops empty bins mid-stream.
  absl::Status update(const std::vector<BinAccumulator::AccumulatorType>& p,
                      const std::vector<BinAccumulator::AccumulatorType>& y);

  // Single-point convenience overload, for genuinely one-at-a-time streaming.
  absl::Status update(BinAccumulator::AccumulatorType p,
                      BinAccumulator::AccumulatorType y);

  // Discards all accumulated data, keeping the same bin count.
  void reset() { bins_.assign(n_bins_, BinAccumulator{}); }

  // Number of bins configured (including any still empty).
  std::size_t n_bins() const noexcept { return n_bins_; }

  // Mean predicted probability in each non-empty bin, in bin order.
  std::vector<float> bin_mean_predicted() const;

  // Mean observed frequency (empirical positive rate) in each non-empty bin.
  std::vector<float> bin_mean_observed() const;

  // Sample count in each non-empty bin (e.g. for bubble-size weighting in a
  // plot).
  std::vector<float> bin_counts() const;

  // Expected Calibration Error: count-weighted mean absolute gap
  // between predicted and observed, over non-empty bins. 0.0 if no
  // data has been accumulated yet.
  float ece() const;

  // Maximum Calibration Error: largest single-bin gap between
  // predicted and observed, over non-empty bins. 0.0 if no data has
  // been accumulated yet.
  float mce() const;

 private:
  std::size_t n_bins_;
  std::vector<BinAccumulator> bins_;
};

}  // namespace yggdrasil_decision_forests::utils::reliability_diagram

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_RELIABILITY_DIAGRAM_H_
