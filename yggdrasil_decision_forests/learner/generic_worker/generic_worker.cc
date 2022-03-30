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

#include "yggdrasil_decision_forests/learner/generic_worker/generic_worker.h"

#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/weight.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace generic_worker {
namespace {
using Blob = distribute::Blob;
}

constexpr char GenericWorker::kWorkerKey[];

absl::Status GenericWorker::TrainModel(
    const proto::Request::TrainModel& request,
    proto::Result::TrainModel* result) {
  LOG(INFO) << "Training model with:";
  LOG(INFO) << "Configuration:\n" << request.train_config().DebugString();
  LOG(INFO) << "Deployment:\n" << request.deployment_config().DebugString();
  LOG(INFO) << "Dataset: " << request.dataset_path();
  LOG(INFO) << "Valid dataset: " << request.valid_dataset_path();

  result->set_model_path(
      file::JoinPath(request.model_base_path(), utils::GenUniqueId()));

  std::unique_ptr<model::AbstractLearner> learner;
  RETURN_IF_ERROR(model::GetLearner(request.train_config(), &learner));
  *learner->mutable_deployment() = request.deployment_config();
  if (request.has_log_directory()) {
    learner->set_log_directory(request.log_directory());
  }

  RETURN_IF_ERROR(
      learner->SetHyperParameters(request.generic_hyper_parameters()));

  // TODO: Interrupt the training if override_model=false and the model is
  // present.

  learner->set_stop_training_trigger(&done_was_called_);

  absl::optional<std::string> valid_dataset;
  if (request.has_valid_dataset_path()) {
    valid_dataset = request.valid_dataset_path();
  }

  LOG(INFO) << "Start training model";
  ASSIGN_OR_RETURN(auto model,
                   learner->TrainWithStatus(request.dataset_path(),
                                            request.dataspec(), valid_dataset));

  if (request.return_model_validation()) {
    *result->mutable_validation_evaluation() = model->ValidationEvaluation();
  }

  LOG(INFO) << "Save model to " << result->model_path();
  RETURN_IF_ERROR(model::SaveModel(result->model_path(), model.get()));

  return absl::OkStatus();
}

absl::Status GenericWorker::EvaluateModel(
    const proto::Request::EvaluateModel& request,
    proto::Result::EvaluateModel* result) {
  LOG(INFO) << "Evaluating model with:";
  LOG(INFO) << "Options:\n" << request.options().DebugString();
  LOG(INFO) << "Model: " << request.model_path();
  LOG(INFO) << "Dataset: " << request.dataset_path();

  std::unique_ptr<model::AbstractModel> model;
  RETURN_IF_ERROR(model::LoadModel(request.model_path(), &model));

  auto options = request.options();
  utils::RandomEngine rnd;
  metric::proto::EvaluationResults evaluation;

  // Evaluation weighting.
  if (model->weights().has_value() && !options.has_weights()) {
    ASSIGN_OR_RETURN(*options.mutable_weights(),
                     dataset::GetUnlinkedWeightDefinition(
                         model->weights().value(), model->data_spec()));
  }

  // Load dataset.
  dataset::VerticalDataset dataset;
  QCHECK_OK(LoadVerticalDataset(request.dataset_path(), model->data_spec(),
                                &dataset));

  if (!options.has_task()) {
    options.set_task(model->task());
  }
  // Evaluate model.
  *result->mutable_evaluation() = model->Evaluate(dataset, options, &rnd);

  return absl::OkStatus();
}

absl::Status GenericWorker::Setup(Blob serialized_welcome) {
  ASSIGN_OR_RETURN(welcome_,
                   utils::ParseBinaryProto<proto::Welcome>(serialized_welcome));
  return absl::OkStatus();
}

utils::StatusOr<Blob> GenericWorker::RunRequest(Blob serialized_request) {
  ASSIGN_OR_RETURN(auto request,
                   utils::ParseBinaryProto<proto::Request>(serialized_request));
  proto::Result result;
  if (request.has_request_id()) {
    result.set_request_id(request.request_id());
  }
  switch (request.type_case()) {
    case proto::Request::kTrainModel:
      RETURN_IF_ERROR(
          TrainModel(request.train_model(), result.mutable_train_model()));
      break;
    case proto::Request::kEvaluateModel:
      RETURN_IF_ERROR(EvaluateModel(request.evaluate_model(),
                                    result.mutable_evaluate_model()));
      break;
    case proto::Request::TYPE_NOT_SET:
      return absl::InvalidArgumentError("Request without type");
  }
  return result.SerializeAsString();
}

}  // namespace generic_worker
}  // namespace model
}  // namespace yggdrasil_decision_forests
