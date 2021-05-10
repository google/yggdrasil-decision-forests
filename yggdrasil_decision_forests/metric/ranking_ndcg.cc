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

#include "yggdrasil_decision_forests/metric/ranking_ndcg.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace metric {

NDCGCalculator::NDCGCalculator(const int truncation)
    : truncation_(truncation), inv_log_rank_(truncation) {
  for (int i = 0; i < inv_log_rank_.size(); i++) {
    inv_log_rank_[i] = 1. / std::log2(2 + i);
  }
}

double NDCGCalculator::NDCGForUnordered(
    const std::vector<RankingLabelAndPrediction>& group) const {
  auto sorted_group = group;
  std::sort(sorted_group.begin(), sorted_group.end(), OrderDecreasingLabel);
  return NDCG(sorted_group);
}

double NDCGCalculator::NDCG(
    const std::vector<RankingLabelAndPrediction>& group) const {
  const int max_rank = std::min(truncation_, static_cast<int>(group.size()));
  auto mutable_group = group;

  // DCG with ideal ordering.
  DCHECK(std::is_sorted(
      mutable_group.begin(), mutable_group.end(),
      [](const RankingLabelAndPrediction& a,
         const RankingLabelAndPrediction& b) { return a.label > b.label; }));

  double idcg = 0;
  for (int rank = 0; rank < max_rank; rank++) {
    idcg += Term(mutable_group[rank].label, rank);
  }

  // Observed DCG.
  std::sort(mutable_group.begin(), mutable_group.end(),
            OrderDecreasingPrediction);

  float last_value = -1.f;
  double sum_gains = 0.;
  int num_gains = 0;
  double dcg = 0.;
  double sum_discounts = 0.;

  for (int rank = 0; rank < group.size(); rank++) {
    const float value = mutable_group[rank].prediction;
    const bool inside_truncation = rank < max_rank;

    if (last_value != value && rank != 0) {
      if (!inside_truncation) {
        break;
      }
      dcg += sum_gains / num_gains * sum_discounts;
      sum_gains = 0.;
      sum_discounts = 0.;
      num_gains = 0;
    }
    last_value = value;

    sum_gains += std::pow(2, mutable_group[rank].label) - 1;
    num_gains++;

    if (inside_truncation) {
      sum_discounts += inv_log_rank_[rank];
    }
  }

  if (num_gains > 0) {
    dcg += sum_gains / num_gains * sum_discounts;
  }

  // "idcg" is independent of the predictions. This value will be skipped
  // for all the candidate models.
  if (idcg == 0) {
    return 0.;
  }
  return dcg / idcg;
}

double NDCGCalculator::DefaultNDCG(
    const std::vector<RankingLabelAndPrediction>& group) const {
  const int max_rank = std::min(truncation_, static_cast<int>(group.size()));

  // DCG with ideal ordering.
  DCHECK(std::is_sorted(group.begin(), group.end(), OrderDecreasingLabel));

  double idcg = 0;
  for (int rank = 0; rank < max_rank; rank++) {
    idcg += Term(group[rank].label, rank);
  }

  double mean_gain = 0;
  for (const auto& example : group) {
    mean_gain += (std::pow(2, example.label) - 1);
  }
  mean_gain /= group.size();

  // Default DCG.
  double dcg = 0;
  for (int rank = 0; rank < max_rank; rank++) {
    dcg += TermFromGain(mean_gain, rank);
  }

  if (idcg == 0) {
    return 0.;
  }
  return dcg / idcg;
}

}  // namespace metric
}  // namespace yggdrasil_decision_forests
