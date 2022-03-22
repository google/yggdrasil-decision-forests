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

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_WEIGHT_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_WEIGHT_H_

#include <vector>

#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/weight.pb.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// Obtains the "linked weight definition" from a "weight definition". A  "weight
// definition" is a user input. The "linked weight definition" is an equivalent
// representation, attached to a given dataspec, and optimized for fast code
// inference (e.g. categorical values stored as string a converted into
// indices).
absl::Status GetLinkedWeightDefinition(
    const proto::WeightDefinition& def,
    const proto::DataSpecification& data_spec,
    proto::LinkedWeightDefinition* linked_def);

// Reverses "GetLinkedWeightDefinition".
utils::StatusOr<proto::WeightDefinition> GetUnlinkedWeightDefinition(
    const proto::LinkedWeightDefinition& linked_def,
    const proto::DataSpecification& data_spec);

// Get the weight of an example from a vertical dataset.
float GetWeight(const VerticalDataset& dataset, VerticalDataset::row_t row,
                const proto::LinkedWeightDefinition& weight_definition);

// Get the weight of a proto::Example.
float GetWeight(const proto::Example& example,
                const proto::LinkedWeightDefinition& weight_definition);

// Get the weights of all the examples in a vertical dataset.
absl::Status GetWeights(const VerticalDataset& dataset,
                        const proto::LinkedWeightDefinition& weight_definition,
                        std::vector<float>* weights);

// Helper function on top of GetWeights(weight_definition). If
// "weight_definition" is not set, all the weights are set to 1.
//
// If `use_optimized_unit_weights` is set, `weights` becomes an empty vector if
// and only if no weight definition is given or all given weights are equal
// to 1.
absl::Status GetWeights(
    const VerticalDataset& dataset,
    const model::proto::TrainingConfigLinking& train_config_link,
    std::vector<float>* weights,
    bool use_optimized_unit_weights = false);

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_WEIGHT_H_
