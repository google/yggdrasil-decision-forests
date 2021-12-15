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

#ifndef YGGDRASIL_DECISION_FORESTS_TOOL_FEATURE_IMPORTANCE_H_
#define YGGDRASIL_DECISION_FORESTS_TOOL_FEATURE_IMPORTANCE_H_

#include <functional>
#include <random>
#include <vector>

#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Computes and adds to the model permutation feature importances.
void ComputePermutationFeatureImportance(
    const metric::proto::EvaluationResults& base_evaluation,
    const std::function<
        absl::optional<metric::proto::EvaluationResults>(const int feature_idx)>
        get_permutation_evaluation,
    model::AbstractModel* model, int num_rounds = 1);

// Computes and adds to the model permutation feature importances.
absl::Status ComputePermutationFeatureImportance(
    const dataset::VerticalDataset& dataset, model::AbstractModel* model,
    int num_rounds = 1);

// Builds a copy of the dataset with the values of the columns in
// "shuffle_column_idxs" are shuffled randomly.
dataset::VerticalDataset ShuffleDatasetColumns(
    const dataset::VerticalDataset& dataset,
    const std::vector<int>& shuffle_column_idxs, utils::RandomEngine* rnd);

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_TOOL_FEATURE_IMPORTANCE_H_
