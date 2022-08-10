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

// Utilities for ranking metrics.

#ifndef YGGDRASIL_DECISION_FORESTS_METRIC_RANKING_UTILS_H_
#define YGGDRASIL_DECISION_FORESTS_METRIC_RANKING_UTILS_H_

namespace yggdrasil_decision_forests {
namespace metric {

// Maximum value of the relevance label in a ranking problem.
constexpr float kMaxRankingRelevance = 4.0;

struct RankingLabelAndPrediction {
  float prediction;
  float label;
};

inline bool OrderDecreasingLabel(const RankingLabelAndPrediction& a,
                                 const RankingLabelAndPrediction& b) {
  return a.label > b.label;
}

inline bool OrderDecreasingPrediction(const RankingLabelAndPrediction& a,
                                      const RankingLabelAndPrediction& b) {
  return a.prediction > b.prediction;
}

}  // namespace metric
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_METRIC_RANKING_UTILS_H_
