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

// The multitask learner trains multiple models in parallel.
// The model trained by the multitask learner is a multitask model that contains
// all models it trained.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_MULTITASKER_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_MULTITASKER_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace multitasker {

class MultitaskerLearner : public AbstractLearner {
 public:
  explicit MultitaskerLearner(
      const model::proto::TrainingConfig& training_config);

  static constexpr char kRegisteredName[] = "MULTITASKER";

  absl::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset = {}) const override;

  absl::Status SetHyperParameters(
      const proto::GenericHyperParameters& generic_hyper_params) override;

  absl::StatusOr<model::proto::GenericHyperParameterSpecification>
  GetGenericHyperParameterSpecification() const override;

  model::proto::LearnerCapabilities Capabilities() const override {
    model::proto::LearnerCapabilities capabilities;
    capabilities.set_support_max_training_duration(true);
    capabilities.set_support_max_model_size_in_memory(true);
    return capabilities;
  }

 private:
  absl::StatusOr<std::unique_ptr<AbstractLearner>> BuildSubLearner(
      const int learner_idx) const;
  absl::StatusOr<model::proto::TrainingConfig> BuildSubTrainingConfig(
      const int learner_idx) const;

  // Hyper-parameters for the sub-learner.
  model::proto::GenericHyperParameters generic_hyper_params_;
};

REGISTER_AbstractLearner(MultitaskerLearner,
                         MultitaskerLearner::kRegisteredName);

}  // namespace multitasker
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_MULTITASKER_H_
