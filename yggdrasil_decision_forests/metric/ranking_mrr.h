/*
 * Copyright 2021 Google LLC.
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

// Compute the Mean Reciprocal Rank (MRR) ranking metric.
//
// The relevance values are expected to be binary with 0 indicating non-relevant
// and >0.5 indicating relevant examples.
//
// MRR currently break score ties according to the training examples' order.
//
#ifndef YGGDRASIL_DECISION_FORESTS_METRIC_RANKING_MRR_H_
#define YGGDRASIL_DECISION_FORESTS_METRIC_RANKING_MRR_H_

#include <vector>

#include "yggdrasil_decision_forests/metric/ranking_utils.h"

namespace yggdrasil_decision_forests {
namespace metric {

class MRRCalculator {
 public:
  explicit MRRCalculator(int truncation);

  // Computes the MRR@truncation of a set of examples.
  double MRR(const std::vector<RankingLabelAndPrediction>& group) const;

 private:
  int truncation_;
};

}  // namespace metric
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_METRIC_RANKING_MRR_H_
