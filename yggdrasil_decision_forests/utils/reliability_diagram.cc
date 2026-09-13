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

#include "yggdrasil_decision_forests/utils/reliability_diagram.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/utils/smoothed_pav_calibration_fit.h"

namespace yggdrasil_decision_forests::utils::reliability_diagram {

absl::Status ReliabilityDiagram::update(
    const std::vector<BinAccumulator::AccumulatorType>& p,
    const std::vector<BinAccumulator::AccumulatorType>& y) {
  return accumulate_bins(bins_, p, y, n_bins_);
}

absl::Status ReliabilityDiagram::update(BinAccumulator::AccumulatorType p,
                                        BinAccumulator::AccumulatorType y) {
  const std::vector<BinAccumulator::AccumulatorType> single_p{p};
  const std::vector<BinAccumulator::AccumulatorType> single_y{y};
  return accumulate_bins(bins_, single_p, single_y, n_bins_);
}

std::vector<float> ReliabilityDiagram::bin_mean_predicted() const {
  std::vector<float> out;
  out.reserve(bins_.size());
  for (const auto& b : bins_) {
    if (b.count > 0.0) {
      out.push_back(b.prob_pred());
    }
  }
  return out;
}

std::vector<float> ReliabilityDiagram::bin_mean_observed() const {
  std::vector<float> out;
  out.reserve(bins_.size());
  for (const auto& b : bins_) {
    if (b.count > 0.0) {
      out.push_back(b.prob_true());
    }
  }
  return out;
}

std::vector<float> ReliabilityDiagram::bin_counts() const {
  std::vector<float> out;
  out.reserve(bins_.size());
  for (const auto& b : bins_) {
    if (b.count > 0.0) {
      out.push_back(b.count);
    }
  }
  return out;
}

float ReliabilityDiagram::ece() const {
  float total = 0.0;
  for (const auto& b : bins_) total += b.count;
  if (total == 0.0) return 0.0;
  float sum = 0.0;
  for (const auto& b : bins_) {
    if (b.count > 0.0) {
      sum += b.count * std::abs(b.prob_pred() - b.prob_true());
    }
  }
  return sum / total;
}

float ReliabilityDiagram::mce() const {
  float worst = 0.0;
  for (const auto& b : bins_) {
    if (b.count > 0.0) {
      worst = std::max(worst, std::abs(b.prob_pred() - b.prob_true()));
    }
  }
  return worst;
}

}  // namespace yggdrasil_decision_forests::utils::reliability_diagram
