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

#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/utils/usage.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace usage {

void OnTrainingStart(const dataset::proto::DataSpecification& data_spec,
                     const model::proto::TrainingConfig& train_config,
                     const model::proto::Metadata& metadata,
                     int64_t num_examples) {
  // Add usage tracking here.
}

void OnTrainingEnd(const dataset::proto::DataSpecification& data_spec,
                   const model::proto::TrainingConfig& train_config,
                   int64_t num_examples, const model::AbstractModel& model,
                   absl::Duration training_duration) {
  // Add usage tracking here.
}

void OnInference(const int64_t num_examples,
                 const model::proto::Metadata& metadata) {
  // Add usage tracking here.
}

void OnInference(int64_t num_examples, const model::MetaData& metadata) {
  // Add usage tracking here.
}

void OnLoadDataset(absl::string_view path) {}

void OnSaveDataset(absl::string_view path) {}

void OnLoadModel(absl::string_view path) {}

void OnSaveModel(absl::string_view path) {}

void EnableUsage(bool usage) {}

}  // namespace usage
}  // namespace utils
}  // namespace yggdrasil_decision_forests
