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

#include "yggdrasil_decision_forests/metric/ranking_mrr.h"

#include <algorithm>
#include <vector>

namespace yggdrasil_decision_forests {
namespace metric {

MRRCalculator::MRRCalculator(const int truncation) : truncation_(truncation) {}

double MRRCalculator::MRR(
    const std::vector<RankingLabelAndPrediction>& group) const {
  auto mutable_group = group;
  std::sort(mutable_group.begin(), mutable_group.end(),
            OrderDecreasingPrediction);
  const int max_rank = std::min(truncation_, static_cast<int>(group.size()));
  for (int rank = 0; rank < max_rank; rank++) {
    if (mutable_group[rank].label > 0.5) {
      return 1.0 / (rank + 1.0);
    }
  }
  return 0.0;
}

}  // namespace metric
}  // namespace yggdrasil_decision_forests
