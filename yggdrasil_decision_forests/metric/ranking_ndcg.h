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

// Compute the Normalized Discounted Cumulative Gain (NDCG) ranking metric.
//
// The relevance values are expected to be in [0, 4] range.
//
// Ties in the prediction will have their label gains averaged.
//
#ifndef YGGDRASIL_DECISION_FORESTS_METRIC_RANKING_NDCG_H_
#define YGGDRASIL_DECISION_FORESTS_METRIC_RANKING_NDCG_H_

#include <cmath>
#include <vector>

#include "yggdrasil_decision_forests/metric/ranking_utils.h"

namespace yggdrasil_decision_forests {
namespace metric {

class NDCGCalculator {
 public:
  explicit NDCGCalculator(int truncation);

  // Computes the NDCG@truncation of a set of examples. "group" is a list of
  // <predicted relevance, ground truth relevance> ordered by ground truth
  // relevance.
  double NDCG(const std::vector<RankingLabelAndPrediction>& group) const;

  // Computes the NDCG@truncation of a set of examples. "group" is a list of
  // <predicted relevance, ground truth relevance> that can be in any order.
  double NDCGForUnordered(
      const std::vector<RankingLabelAndPrediction>& group) const;

  // Default NDCG i.e. NDCG with a model predicting always the same value.
  //
  // The default is computed by averaging the gain of all the items.
  double DefaultNDCG(const std::vector<RankingLabelAndPrediction>& group) const;

  // Computes "(2^relevance-1)/log2(rank+1)" with "rank<truncation".
  inline double Term(const double relevance, const int rank) const {
    return TermFromGain(std::exp2(relevance) - 1.0, rank);
  }

  inline double Term(const float relevance, const int rank) const {
    return TermFromGain(std::exp2f(relevance) - 1.f, rank);
  }

  inline double TermFromGain(const double gain, const int rank) const {
    return gain * inv_log_rank_[rank];
  }

 private:
  int truncation_;

  // "inv_log_rank_[i] := 1. / log2(2 + i)".
  std::vector<double> inv_log_rank_;
};

}  // namespace metric
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_METRIC_RANKING_NDCG_H_
