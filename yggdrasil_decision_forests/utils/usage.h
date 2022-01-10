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

// Tracks learner and model usage for accounting.
//
// Those methods are called whenever the corresponding action occurs. For
// example, "OnTrainingStart" is called whenever the training of a new model
// starts.

#ifndef YGGDRASIL_DECISION_FORESTS_TOOL_USAGE_H_
#define YGGDRASIL_DECISION_FORESTS_TOOL_USAGE_H_

#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/metadata.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace usage {

// Start a new model training.
// Should be called at the start of the "Train" methods of learners.
void OnTrainingStart(
    const dataset::proto::DataSpecification& data_spec,
    const model::proto::TrainingConfig& train_config,
    const model::proto::TrainingConfigLinking& train_config_link,
    int64_t num_examples);

// Complete a model training.
// Should be called at the end of the "Train" methods of learners.
void OnTrainingEnd(const dataset::proto::DataSpecification& data_spec,
                   const model::proto::TrainingConfig& train_config,
                   const model::proto::TrainingConfigLinking& train_config_link,
                   int64_t num_examples, const model::AbstractModel& model,
                   absl::Duration training_duration);

// Inference of a model.
//
// The inference on model containing other models (e.g. the ensembler or the
// calibrator) might or might not be counted multiple times depending on the
// specific model implementation.
//
// TODO: Merge the two functions when model::Metadata is removed.
void OnInference(int64_t num_examples, const model::proto::Metadata& metadata);
void OnInference(int64_t num_examples, const model::MetaData& metadata);

}  // namespace usage
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_TOOL_USAGE_H_
