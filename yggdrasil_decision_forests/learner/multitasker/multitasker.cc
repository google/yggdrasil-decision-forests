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

#include "yggdrasil_decision_forests/learner/multitasker/multitasker.h"

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/multitasker/multitasker.pb.h"
#include "yggdrasil_decision_forests/model/multitasker/multitasker.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"
#include "yggdrasil_decision_forests/utils/usage.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace multitasker {

constexpr char MultitaskerLearner::kRegisteredName[];

MultitaskerLearner::MultitaskerLearner(
    const model::proto::TrainingConfig& training_config)
    : AbstractLearner(training_config) {}

absl::StatusOr<std::unique_ptr<AbstractModel>>
MultitaskerLearner::TrainWithStatus(
    const dataset::VerticalDataset& train_dataset,
    absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
        valid_dataset) const {
  const auto& mt_config =
      training_config().GetExtension(proto::multitasker_config);
  if (mt_config.subtasks_size() == 0) {
    return absl::InvalidArgumentError("At least one task required");
  }

  auto model = absl::make_unique<MultitaskerModel>();
  model->set_data_spec(train_dataset.data_spec());
  STATUS_CHECK_LE(model->models_.size(), mt_config.subtasks_size());
  model->models_.resize(mt_config.subtasks_size());

  ASSIGN_OR_RETURN(const auto first_subtraining_config,
                   BuildSubTrainingConfig(0));
  model::proto::TrainingConfigLinking first_subtraining_config_link;
  RETURN_IF_ERROR(AbstractLearner::LinkTrainingConfig(
      first_subtraining_config, train_dataset.data_spec(),
      &first_subtraining_config_link));
  InitializeModelWithAbstractTrainingConfig(
      first_subtraining_config, first_subtraining_config_link, model.get());

  utils::concurrency::Mutex mutex;
  absl::Status status;

  const auto train_subtask = [&](const int subtask_idx) {
    STATUS_CHECK_GE(subtask_idx, 0);
    ASSIGN_OR_RETURN(const auto sublearner, BuildSubLearner(subtask_idx));
    ASSIGN_OR_RETURN(auto submodel,
                     sublearner->TrainWithStatus(train_dataset, valid_dataset));
    utils::concurrency::MutexLock lock(&mutex);
    STATUS_CHECK_LT(subtask_idx, model->models_.size());
    model->models_[subtask_idx] = std::move(submodel);
    return absl::OkStatus();
  };

  const auto train_subtask_nostatus = [&](const int subtask_idx) {
    {
      utils::concurrency::MutexLock lock(&mutex);
      if (!status.ok()) {
        return;
      }
    }
    const auto substatus = train_subtask(subtask_idx);
    utils::concurrency::MutexLock lock(&mutex);
    if (!substatus.ok()) {
      status.Update(substatus);
    }
  };

  {
    utils::concurrency::ThreadPool pool("multitasker",
                                        deployment().num_threads());
    pool.StartWorkers();
    for (int subtask_idx = 0; subtask_idx < mt_config.subtasks_size();
         subtask_idx++) {
      pool.Schedule([train_subtask_nostatus, subtask_idx]() {
        train_subtask_nostatus(subtask_idx);
      });
    }
  }

  RETURN_IF_ERROR(status);

  return model;
}

absl::Status MultitaskerLearner::SetHyperParameters(
    const model::proto::GenericHyperParameters& generic_hyper_params) {
  generic_hyper_params_ = generic_hyper_params;
  return absl::OkStatus();
}

absl::StatusOr<model::proto::TrainingConfig>
MultitaskerLearner::BuildSubTrainingConfig(const int learner_idx) const {
  const auto& mt_config =
      training_config().GetExtension(proto::multitasker_config);

  model::proto::TrainingConfig sub_learner_config = mt_config.base_learner();
  sub_learner_config.MergeFrom(mt_config.subtasks(learner_idx).train_config());

  if (training_config().has_maximum_training_duration_seconds() &&
      !sub_learner_config.has_maximum_training_duration_seconds()) {
    sub_learner_config.set_maximum_training_duration_seconds(
        training_config().maximum_training_duration_seconds());
  }

  if (training_config().has_maximum_model_size_in_memory_in_bytes() &&
      !sub_learner_config.has_maximum_model_size_in_memory_in_bytes()) {
    sub_learner_config.set_maximum_model_size_in_memory_in_bytes(
        training_config().maximum_model_size_in_memory_in_bytes());
  }
  return sub_learner_config;
}

absl::StatusOr<std::unique_ptr<AbstractLearner>>
MultitaskerLearner::BuildSubLearner(const int learner_idx) const {
  const auto& mt_config =
      training_config().GetExtension(proto::multitasker_config);

  ASSIGN_OR_RETURN(const auto sub_learner_config,
                   BuildSubTrainingConfig(learner_idx));

  // Build sub-model learner.
  std::unique_ptr<AbstractLearner> sub_learner;
  RETURN_IF_ERROR(GetLearner(sub_learner_config, &sub_learner));
  *sub_learner->mutable_deployment() = mt_config.base_learner_deployment();
  RETURN_IF_ERROR(sub_learner->SetHyperParameters(generic_hyper_params_));
  return sub_learner;
}

absl::StatusOr<model::proto::GenericHyperParameterSpecification>
MultitaskerLearner::GetGenericHyperParameterSpecification() const {
  ASSIGN_OR_RETURN(auto sub_learner, BuildSubLearner(0));
  return sub_learner->GetGenericHyperParameterSpecification();
}

}  // namespace multitasker
}  // namespace model
}  // namespace yggdrasil_decision_forests
