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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_VECTOR_SEQUENCE_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_VECTOR_SEQUENCE_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/label.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests::model::decision_tree {

// Find the best possible split on a numerical vector sequence attribute for a
// classification label.
//
// Uses dynamic dispatching on the label type.
absl::StatusOr<SplitSearchResult>
FindSplitAnyLabelFeatureNumericalVectorSequence(
    model::proto::Task task,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::NumericalVectorSequenceColumn& attribute,
    const dataset::proto::Column& attribute_spec, const LabelStats& label_stats,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache);

// Find the best possible split on a numerical vector sequence attribute for a
// classification label.
//
// Uses template dispatching on the label type.
template <typename LabelType>
absl::StatusOr<SplitSearchResult>
FindSplitAnyLabelTemplateFeatureNumericalVectorSequence(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::NumericalVectorSequenceColumn& attribute,
    const dataset::proto::Column& attribute_spec, const LabelType& label_stats,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache);

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif
