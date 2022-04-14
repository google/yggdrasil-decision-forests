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

#include "yggdrasil_decision_forests/learner/hyperparameters_optimizer/hyperparameters_optimizer.h"

#include <cmath>
#include <limits>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/hyperparameters_optimizer/hyperparameters_optimizer.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/concurrency_streamprocessor.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/usage.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace hyperparameters_optimizer_v2 {

constexpr char HyperParameterOptimizerLearner::kRegisteredName[];

namespace {
using row_t = dataset::VerticalDataset::row_t;

// Set the default values of the training configuration.
absl::Status SetTrainConfigDefaultValues(
    model::proto::TrainingConfig* effective_config) {
  auto* hparam_opt_config = effective_config->MutableExtension(
      proto::hyperparameters_optimizer_config);
  auto& sub_learner = *hparam_opt_config->mutable_base_learner();

  // The training duration of each individual trial should be less than the
  // overall tuning.
  if (effective_config->has_maximum_training_duration_seconds() &&
      !sub_learner.has_maximum_training_duration_seconds()) {
    sub_learner.set_maximum_training_duration_seconds(
        effective_config->maximum_training_duration_seconds());
  }

  RETURN_IF_ERROR(CopyProblemDefinition(*effective_config, &sub_learner));
  return absl::OkStatus();
}

}  // namespace

HyperParameterOptimizerLearner::HyperParameterOptimizerLearner(
    const model::proto::TrainingConfig& training_config)
    : AbstractLearner(training_config) {}

absl::Status HyperParameterOptimizerLearner::SetHyperParametersImpl(
    utils::GenericHyperParameterConsumer* generic_hyper_params) {
  auto& spe_config = *training_config_.MutableExtension(
      proto::hyperparameters_optimizer_config);

  if (!spe_config.has_base_learner()) {
    // The base learner is not set. This is possible during the automated
    // documentation generation.
    LOG(WARNING) << "Sub-learner not set. This is only expected during the "
                    "automatic documentation generation.";
    return AbstractLearner::SetHyperParametersImpl(generic_hyper_params);
  }

  auto base_learner_config = spe_config.base_learner();
  RETURN_IF_ERROR(
      CopyProblemDefinition(training_config_, &base_learner_config));
  std::unique_ptr<AbstractLearner> base_learner;

  RETURN_IF_ERROR(GetLearner(base_learner_config, &base_learner));
  // Apply the hyperparameters to the base learner.
  RETURN_IF_ERROR(base_learner->SetHyperParametersImpl(generic_hyper_params));
  // Copy-back the hyper-parameter of the base learner into the base learner
  // configuration in the hyperparameter optimizer configuration.
  *spe_config.mutable_base_learner() = base_learner->training_config();

  // Copy-back some of the generic hyperparameters into the optimizer
  // configuration.

  // The maximum training duration applies both to the optimizer and the base
  // learner.
  if (spe_config.base_learner().has_maximum_training_duration_seconds()) {
    training_config_.set_maximum_training_duration_seconds(
        spe_config.base_learner().maximum_training_duration_seconds());
  }
  return absl::OkStatus();
}

utils::StatusOr<model::proto::GenericHyperParameterSpecification>
HyperParameterOptimizerLearner::GetGenericHyperParameterSpecification() const {
  // Returns the hyper-parameters of the base learner.
  const auto& spe_config =
      training_config_.GetExtension(proto::hyperparameters_optimizer_config);

  if (!spe_config.has_base_learner()) {
    LOG(WARNING) << "Sub-learner not set. This is only expected during the "
                    "automatic documentation generation.";
    return AbstractLearner::GetGenericHyperParameterSpecification();
  }

  auto spe_config_with_pb_definition = spe_config;
  RETURN_IF_ERROR(CopyProblemDefinition(
      training_config_, spe_config_with_pb_definition.mutable_base_learner()));

  ASSIGN_OR_RETURN(
      auto base_learner,
      BuildBaseLearner(spe_config_with_pb_definition, /*for_tuning=*/true));
  return base_learner->GetGenericHyperParameterSpecification();
}

utils::StatusOr<std::unique_ptr<AbstractModel>>
HyperParameterOptimizerLearner::TrainFromFileOnMemoryDataset(
    const dataset::VerticalDataset& train_dataset,
    absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
        valid_dataset) const {
  LOG(INFO) << "Serialize memory dataset to disk. To skip this stage and a "
               "more efficient training, provide the dataset as a path instead "
               "of as a VerticalDataset";
  const auto& serialized_dataset_format =
      training_config_.GetExtension(proto::hyperparameters_optimizer_config)
          .serialized_dataset_format();

  RETURN_IF_ERROR(
      file::RecursivelyCreateDir(deployment().cache_path(), file::Defaults()));
  const auto train_dataset_path = absl::StrCat(
      serialized_dataset_format, ":",
      file::JoinPath(deployment().cache_path(), "train_dataset.tfe"));
  RETURN_IF_ERROR(
      dataset::SaveVerticalDataset(train_dataset, train_dataset_path));

  absl::optional<std::string> valid_dataset_path;
  if (valid_dataset.has_value()) {
    valid_dataset_path = absl::StrCat(
        serialized_dataset_format, ":",
        file::JoinPath(deployment().cache_path(), "valid_dataset.tfe"));
    RETURN_IF_ERROR(dataset::SaveVerticalDataset(valid_dataset.value(),
                                                 valid_dataset_path.value()));
  }

  return TrainWithStatus(train_dataset_path, train_dataset.data_spec(),
                         valid_dataset_path);
}

utils::StatusOr<std::unique_ptr<AbstractModel>>
HyperParameterOptimizerLearner::TrainWithStatus(
    const dataset::VerticalDataset& train_dataset,
    absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
        valid_dataset) const {
  if (deployment().execution_case() ==
      model::proto::DeploymentConfig::ExecutionCase::kDistribute) {
    // Export the dataset to file and run the training on file.
    return TrainFromFileOnMemoryDataset(train_dataset, valid_dataset);
  }

  if (deployment().execution_case() !=
          model::proto::DeploymentConfig::ExecutionCase::EXECUTION_NOT_SET &&
      deployment().execution_case() !=
          model::proto::DeploymentConfig::ExecutionCase::kLocal) {
    return absl::InvalidArgumentError(
        "The HyperParameterOptimizerLearner only support local or distributed "
        "deployment configs.");
  }

  const auto begin_training = absl::Now();

  // The effective configuration is the user configuration + the default value +
  // the automatic configuration (if enabled) + the copy of the non-specified
  // training configuration field from the learner to the sub-learner (e.g. copy
  // of the label name).
  model::proto::TrainingConfig effective_config;
  model::proto::TrainingConfigLinking config_link;
  RETURN_IF_ERROR(GetEffectiveConfiguration(train_dataset.data_spec(),
                                            &effective_config, &config_link));
  const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config =
      effective_config.GetExtension(proto::hyperparameters_optimizer_config);

  utils::usage::OnTrainingStart(train_dataset.data_spec(), effective_config,
                                config_link, train_dataset.nrow());

  // Initialize the learner with the base hyperparameters.
  ASSIGN_OR_RETURN(auto base_learner,
                   BuildBaseLearner(spe_config, /*for_tuning=*/true));

  // Hyperparameters of the base learner.TrainWithStatus().
  ASSIGN_OR_RETURN(const auto search_space_spec,
                   base_learner->GetGenericHyperParameterSpecification());

  // Build the effective space to optimize.
  ASSIGN_OR_RETURN(const auto search_space,
                   BuildSearchSpace(spe_config, *base_learner));
  LOG(INFO) << "Hyperparameter search space:\n" << search_space.DebugString();

  // Select the best hyperparameters.
  model::proto::HyperparametersOptimizerLogs logs;
  std::unique_ptr<AbstractModel> best_model;
  ASSIGN_OR_RETURN(const auto best_params,
                   SearchBestHyperparameterInProcess(
                       spe_config, config_link, search_space_spec, search_space,
                       train_dataset, valid_dataset, &best_model, &logs));
  LOG(INFO) << "Best hyperparameters:\n" << best_params.DebugString();

  // TODO(gbm): Record the logs.

  if (spe_config.retrain_final_model()) {
    // Train a model on the entire train dataset using the best hyperparameters.
    LOG(INFO) << "Training a model on the best hyper parameters.";
    RETURN_IF_ERROR(base_learner->SetHyperParameters(best_params));
    ASSIGN_OR_RETURN(
        auto mdl, base_learner->TrainWithStatus(train_dataset, valid_dataset));
    utils::usage::OnTrainingEnd(train_dataset.data_spec(), training_config(),
                                config_link, train_dataset.nrow(), *mdl,
                                absl::Now() - begin_training);
    *mdl->mutable_hyperparameter_optimizer_logs() = logs;
    return mdl;
  } else {
    if (!best_model) {
      return absl::InternalError("Missing model");
    }
    *best_model->mutable_hyperparameter_optimizer_logs() = logs;
    return best_model;
  }
}

absl::Status HyperParameterOptimizerLearner::GetEffectiveConfiguration(
    const dataset::proto::DataSpecification& data_spec,
    model::proto::TrainingConfig* effective_config,
    model::proto::TrainingConfigLinking* effective_config_link) const {
  // Apply the default values.
  *effective_config = training_config();

  // Apply the default values.
  RETURN_IF_ERROR(SetTrainConfigDefaultValues(effective_config));

  // Solve the symbols in the configuration.
  RETURN_IF_ERROR(AbstractLearner::LinkTrainingConfig(
      *effective_config, data_spec, effective_config_link));

  RETURN_IF_ERROR(CheckConfiguration(data_spec, *effective_config,
                                     *effective_config_link, deployment_));
  return absl::OkStatus();
}

utils::StatusOr<std::unique_ptr<AbstractModel>>
HyperParameterOptimizerLearner::TrainWithStatus(
    const absl::string_view typed_path,
    const dataset::proto::DataSpecification& data_spec,
    const absl::optional<std::string>& typed_valid_path) const {
  if (deployment().execution_case() ==
          model::proto::DeploymentConfig::ExecutionCase::EXECUTION_NOT_SET ||
      deployment().execution_case() ==
          model::proto::DeploymentConfig::ExecutionCase::kLocal) {
    // Load the dataset in memory and run the in-memory training.
    return AbstractLearner::TrainWithStatus(typed_path, data_spec,
                                            typed_valid_path);
  }

  if (!deployment().has_distribute()) {
    return absl::InvalidArgumentError(
        "The HyperParameterOptimizerLearner only support local or distributed "
        "deployment configs.");
  }

  const auto begin_training = absl::Now();

  // Initialize the remote workers.
  ASSIGN_OR_RETURN(auto manager, CreateDistributeManager());

  // The effective configuration is the user configuration + the default value +
  // the automatic configuration (if enabled) + the copy of the non-specified
  // training configuration field from the learner to the sub-learner (e.g. copy
  // of the label name).
  model::proto::TrainingConfig effective_config;
  model::proto::TrainingConfigLinking config_link;
  RETURN_IF_ERROR(
      GetEffectiveConfiguration(data_spec, &effective_config, &config_link));
  const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config =
      effective_config.GetExtension(proto::hyperparameters_optimizer_config);

  utils::usage::OnTrainingStart(data_spec, effective_config, config_link, -1);

  // Initialize the learner with the base hyperparameters.
  ASSIGN_OR_RETURN(auto base_learner,
                   BuildBaseLearner(spe_config, /*for_tuning=*/true));

  // Hyperparameters of the base learner.TrainWithStatus().
  ASSIGN_OR_RETURN(const auto search_space_spec,
                   base_learner->GetGenericHyperParameterSpecification());

  // Build the effective space to optimize.
  ASSIGN_OR_RETURN(const auto search_space,
                   BuildSearchSpace(spe_config, *base_learner));
  LOG(INFO) << "Hyperparameter search space:\n" << search_space.DebugString();

  // Select the best hyperparameters.
  model::proto::HyperparametersOptimizerLogs logs;
  std::unique_ptr<AbstractModel> best_model;
  ASSIGN_OR_RETURN(
      const auto best_params,
      SearchBestHyperparameterDistributed(
          spe_config, config_link, search_space_spec, search_space, typed_path,
          data_spec, typed_valid_path, &best_model, manager.get(), &logs));
  LOG(INFO) << "Best hyperparameters:\n" << best_params.DebugString();

  // TODO(gbm): Record the logs.

  if (spe_config.retrain_final_model()) {
    // Train a model on the entire train dataset using the best hyperparameters.
    LOG(INFO) << "Training a model on the best hyper parameters.";
    ASSIGN_OR_RETURN(auto model,
                     TrainRemoteModel(spe_config.base_learner(), config_link,
                                      spe_config.base_learner_deployment(),
                                      best_params, typed_path, data_spec,
                                      typed_valid_path, manager.get()));

    utils::usage::OnTrainingEnd(data_spec, training_config(), config_link, -1,
                                *model, absl::Now() - begin_training);
    *model->mutable_hyperparameter_optimizer_logs() = logs;

    RETURN_IF_ERROR(manager->Done());
    return model;
  } else {
    if (!best_model) {
      return absl::InternalError("Missing model");
    }
    *best_model->mutable_hyperparameter_optimizer_logs() = logs;
    RETURN_IF_ERROR(manager->Done());
    return best_model;
  }
}

utils::StatusOr<std::unique_ptr<AbstractModel>>
HyperParameterOptimizerLearner::TrainRemoteModel(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const model::proto::DeploymentConfig& deployment_config,
    const model::proto::GenericHyperParameters& generic_hyper_params,
    const absl::string_view typed_train_path,
    const dataset::proto::DataSpecification& data_spec,
    const absl::optional<std::string>& typed_valid_path,
    distribute::AbstractManager* manager) const {
  generic_worker::proto::Request generic_request;
  auto& train_request = *generic_request.mutable_train_model();

  *train_request.mutable_train_config() = config;
  *train_request.mutable_deployment_config() = deployment_config;
  *train_request.mutable_generic_hyper_parameters() = generic_hyper_params;

  train_request.set_dataset_path(std::string(typed_train_path));
  if (typed_valid_path.has_value()) {
    train_request.set_valid_dataset_path(typed_valid_path.value());
  }

  *train_request.mutable_dataspec() = data_spec;
  train_request.set_model_base_path(
      file::JoinPath(deployment().cache_path(), "models"));

  ASSIGN_OR_RETURN(auto result,
                   manager->BlockingProtoRequest<generic_worker::proto::Result>(
                       generic_request));

  std::unique_ptr<model::AbstractModel> model;
  RETURN_IF_ERROR(model::LoadModel(result.train_model().model_path(), &model));
  return model;
}

utils::StatusOr<std::unique_ptr<distribute::AbstractManager>>
HyperParameterOptimizerLearner::CreateDistributeManager() const {
  // Configure the working directory.
  if (deployment().cache_path().empty()) {
    return absl::InvalidArgumentError(
        "deployment.cache_path is empty. Please provide a cache directory with "
        "ensemble distributed training.");
  }

  if (!deployment().distribute().working_directory().empty()) {
    return absl::InvalidArgumentError(
        "deployment.distribute.working_directory should be empty. Use "
        "deployment.cache_path to specify the cache directory.");
  }

  // Initialize the distribute manager.
  auto distribute_config = deployment_.distribute();
  distribute_config.set_working_directory(
      file::JoinPath(deployment_.cache_path(), "distribute"));
  generic_worker::proto::Welcome welcome;
  welcome.set_temporary_directory(
      file::JoinPath(deployment_.cache_path(), "workers"));
  return distribute::CreateManager(distribute_config, "GENERIC_WORKER",
                                   welcome.SerializeAsString());
}

utils::StatusOr<std::unique_ptr<AbstractLearner>>
HyperParameterOptimizerLearner::HyperParameterOptimizerLearner::
    BuildBaseLearner(
        const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config,
        const bool for_tuning) const {
  std::unique_ptr<AbstractLearner> base_learner;
  RETURN_IF_ERROR(GetLearner(spe_config.base_learner(), &base_learner,
                             spe_config.base_learner_deployment()));
  return base_learner;
}

utils::StatusOr<model::proto::HyperParameterSpace>
HyperParameterOptimizerLearner::BuildSearchSpace(
    const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config,
    const AbstractLearner& base_learner) const {
  model::proto::HyperParameterSpace space = spe_config.search_space();
  return space;
}

utils::StatusOr<bool> HyperParameterOptimizerLearner::IsMaximization(
    const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config,
    const metric::proto::MetricAccessor& metric_accessor) const {
  if (spe_config.evaluation().has_maximize_metric()) {
    return spe_config.evaluation().maximize_metric();
  }
  return metric::HigherIsBetter(metric_accessor);
}

utils::StatusOr<model::proto::GenericHyperParameters>
HyperParameterOptimizerLearner::SearchBestHyperparameterInProcess(
    const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config,
    const model::proto::TrainingConfigLinking& config_link,
    const model::proto::GenericHyperParameterSpecification& search_space_spec,
    const model::proto::HyperParameterSpace& search_space,
    const dataset::VerticalDataset& train_dataset,
    absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
        valid_dataset,
    std::unique_ptr<AbstractModel>* best_model,
    model::proto::HyperparametersOptimizerLogs* logs) const {
  const auto begin_optimization = absl::Now();

  // Initialize the hyperparameter logs.
  *logs->mutable_space() = search_space;
  logs->set_hyperparameter_optimizer_key(
      spe_config.optimizer().optimizer_key());

  // Instantiate the optimizer.
  ASSIGN_OR_RETURN(auto optimizer, OptimizerInterfaceRegisterer::Create(
                                       spe_config.optimizer().optimizer_key(),
                                       spe_config.optimizer(), search_space));

  // The "async_evaluator" evaluates candidates in parallel using
  // multi-threading.
  struct Output {
    double score;
    model::proto::GenericHyperParameters candidate;
    std::unique_ptr<AbstractModel> model;
  };
  utils::concurrency::StreamProcessor<model::proto::GenericHyperParameters,
                                      utils::StatusOr<Output>>
      async_evaluator(
          "evaluator", deployment().num_threads(),
          [&](const model::proto::GenericHyperParameters& candidate)
              -> utils::StatusOr<Output> {
            std::unique_ptr<AbstractModel> model;
            ASSIGN_OR_RETURN(
                const auto score,
                EvaluateCandidateLocally(candidate, spe_config, config_link,
                                         train_dataset, valid_dataset, &model));
            return Output{score, candidate, std::move(model)};
          });

  LOG(INFO) << "Start local tuner with " << deployment().num_threads()
            << " thread(s)";
  async_evaluator.StartWorkers();

  // Number of candidate being evaluated.
  int pending_evaluation = 0;

  // Number of evaluated and processed candidates.
  int round_idx = 0;

  // True iff. the optimizer is done generating new candidates.
  bool exploration_is_done = false;

  double logging_best_score = std::numeric_limits<double>::quiet_NaN();

  while (true) {
    if (stop_training_trigger_ != nullptr && *stop_training_trigger_) {
      LOG(INFO) << "Training interrupted per the user";
      break;
    }

    // Generate as many candidates as possible.
    while (true) {
      model::proto::GenericHyperParameters candidate;
      ASSIGN_OR_RETURN(const auto optimizer_status,
                       optimizer->NextCandidate(&candidate));
      if (optimizer_status == NextCandidateStatus::kExplorationIsDone) {
        // The optimization can stop.
        exploration_is_done = true;
        if (pending_evaluation > 0) {
          return absl::InternalError(
              "The optimizer stopped the optimization while some evaluations "
              "are still running.");
        }
        async_evaluator.CloseSubmits();
        break;
      } else if (optimizer_status == NextCandidateStatus::kWaitForEvaluation) {
        // Wait for some evaluation to finish.
        if (pending_evaluation == 0) {
          return absl::InternalError(
              "The optimizer requested an evaluation while not evaluation is "
              "currently pending.");
        }
        break;
      } else if (optimizer_status ==
                 NextCandidateStatus::kNewCandidateAvailable) {
        // Start evaluating this new candidate.
        pending_evaluation++;
        async_evaluator.Submit(candidate);
      }
    }

    auto maybe_output = async_evaluator.GetResult();
    if (!maybe_output.has_value()) {
      // Stop the optimization.
      if (!exploration_is_done) {
        return absl::InternalError(
            "No more evaluation results while the exploration is not marked as "
            "done");
      }
      break;
    }
    if (!maybe_output.value().ok()) {
      return maybe_output.value().status();
    }
    auto& output = maybe_output.value().value();
    pending_evaluation--;

    RETURN_IF_ERROR(
        optimizer->ConsumeEvaluation(output.candidate, output.score));

    // Record the hyperparameter + evaluation.
    auto& log_entry = *logs->add_steps();
    log_entry.set_evaluation_time(
        absl::ToDoubleSeconds(absl::Now() - begin_optimization));
    *log_entry.mutable_hyperparameters() = output.candidate;
    log_entry.set_score(output.score);

    if (std::isnan(logging_best_score) || output.score > logging_best_score) {
      logging_best_score = output.score;
      *best_model = std::move(output.model);
    }
    LOG(INFO) << "[" << round_idx + 1 << "/" << optimizer->NumExpectedRounds()
              << "] Score: " << output.score << " / " << logging_best_score
              << " HParams: " << output.candidate.ShortDebugString();

    if (training_config().has_maximum_training_duration_seconds() &&
        (absl::Now() - begin_optimization) >
            absl::Seconds(
                training_config().maximum_training_duration_seconds())) {
      LOG(INFO)
          << "Stop optimization because of the maximum training duration.";
      break;
    }
    round_idx++;
  }
  async_evaluator.JoinAllAndStopThreads();

  model::proto::GenericHyperParameters best_params;
  double best_score;
  std::tie(best_params, best_score) = optimizer->BestParameters();

  logs->set_best_score(best_score);
  *logs->mutable_best_hyperparameters() = best_params;

  return best_params;
}

utils::StatusOr<model::proto::GenericHyperParameters>
HyperParameterOptimizerLearner::SearchBestHyperparameterDistributed(
    const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config,
    const model::proto::TrainingConfigLinking& config_link,
    const model::proto::GenericHyperParameterSpecification& search_space_spec,
    const model::proto::HyperParameterSpace& search_space,
    const absl::string_view typed_train_path,
    const dataset::proto::DataSpecification& data_spec,
    const absl::optional<std::string>& typed_valid_path,
    std::unique_ptr<AbstractModel>* best_model,
    distribute::AbstractManager* manager,
    model::proto::HyperparametersOptimizerLogs* logs) const {
  const auto begin_optimization = absl::Now();

  // Initializer the hyperparameter logs.
  *logs->mutable_space() = search_space;
  logs->set_hyperparameter_optimizer_key(
      spe_config.optimizer().optimizer_key());

  // Instantiate the optimizer.
  ASSIGN_OR_RETURN(auto optimizer, OptimizerInterfaceRegisterer::Create(
                                       spe_config.optimizer().optimizer_key(),
                                       spe_config.optimizer(), search_space));

  // Mapping between request_id and hyperparameter of the currently running
  // evaluations.
  absl::flat_hash_map<std::string, model::proto::GenericHyperParameters>
      pending_hyperparameters;

  // Number of evaluated and processed candidates.
  int round_idx = 0;

  // Id of the next request.
  int next_request_id = 0;

  // True iff. the optimizer is done generating new candidates.
  bool exploration_is_done = false;

  double logging_best_score = std::numeric_limits<double>::quiet_NaN();

  // Path to the best model so far.
  std::string best_model_path;

  while (true) {
    if (stop_training_trigger_ != nullptr && *stop_training_trigger_) {
      LOG(INFO) << "Training interrupted per the user";
      break;
    }

    // Generate as many candidates as possible.
    while (true) {
      model::proto::GenericHyperParameters candidate;
      ASSIGN_OR_RETURN(const auto optimizer_status,
                       optimizer->NextCandidate(&candidate));
      if (optimizer_status == NextCandidateStatus::kExplorationIsDone) {
        // The optimization can stop.
        exploration_is_done = true;
        if (!pending_hyperparameters.empty()) {
          return absl::InternalError(
              "The optimizer stopped the optimization while some evaluations "
              "are still running.");
        }
        break;
      } else if (optimizer_status == NextCandidateStatus::kWaitForEvaluation) {
        // Wait for some evaluation to finish.
        if (pending_hyperparameters.empty()) {
          return absl::InternalError(
              "The optimizer requested an evaluation while not evaluation is "
              "currently pending.");
        }
        break;
      } else if (optimizer_status ==
                 NextCandidateStatus::kNewCandidateAvailable) {
        // Start evaluating this new candidate.
        generic_worker::proto::Request generic_request;
        auto& train_request = *generic_request.mutable_train_model();
        train_request.set_return_model_validation(true);

        *train_request.mutable_train_config() = spe_config.base_learner();
        *train_request.mutable_deployment_config() =
            spe_config.base_learner_deployment();
        *train_request.mutable_generic_hyper_parameters() = candidate;

        train_request.set_dataset_path(std::string(typed_train_path));
        if (typed_valid_path.has_value()) {
          train_request.set_valid_dataset_path(typed_valid_path.value());
        }

        *train_request.mutable_dataspec() = data_spec;
        train_request.set_model_base_path(
            file::JoinPath(deployment().cache_path(), "models"));

        std::string request_id = absl::StrCat(next_request_id++);
        generic_request.set_request_id(request_id);
        pending_hyperparameters[request_id] = std::move(candidate);

        RETURN_IF_ERROR(manager->AsynchronousProtoRequest(generic_request));
      }
    }

    if (exploration_is_done && pending_hyperparameters.empty()) {
      break;
    }

    ASSIGN_OR_RETURN(
        auto evaluator_result,
        manager->NextAsynchronousProtoAnswer<generic_worker::proto::Result>());

    ASSIGN_OR_RETURN(
        const auto score,
        EvaluationToScore(
            spe_config,
            evaluator_result.train_model().validation_evaluation()));

    auto it_hyperparameter =
        pending_hyperparameters.find(evaluator_result.request_id());
    if (it_hyperparameter == pending_hyperparameters.end()) {
      return absl::InternalError("Unknown request id");
    }
    const auto candidate = std::move(it_hyperparameter->second);
    pending_hyperparameters.erase(it_hyperparameter);

    RETURN_IF_ERROR(optimizer->ConsumeEvaluation(candidate, score));

    // Record the hyperparameter + evaluation.
    auto& log_entry = *logs->add_steps();
    log_entry.set_evaluation_time(
        absl::ToDoubleSeconds(absl::Now() - begin_optimization));
    *log_entry.mutable_hyperparameters() = candidate;
    log_entry.set_score(score);

    if (std::isnan(logging_best_score) || score > logging_best_score) {
      logging_best_score = score;
      best_model_path = evaluator_result.train_model().model_path();
    }
    LOG(INFO) << "[" << round_idx + 1 << "/" << optimizer->NumExpectedRounds()
              << "] Score: " << score << " / " << logging_best_score
              << " HParams: " << candidate.ShortDebugString();

    if (training_config().has_maximum_training_duration_seconds() &&
        (absl::Now() - begin_optimization) >
            absl::Seconds(
                training_config().maximum_training_duration_seconds())) {
      LOG(INFO)
          << "Stop optimization because of the maximum training duration.";
      break;
    }
    round_idx++;
  }

  model::proto::GenericHyperParameters best_params;
  double best_score;
  std::tie(best_params, best_score) = optimizer->BestParameters();

  logs->set_best_score(best_score);
  *logs->mutable_best_hyperparameters() = best_params;

  if (!spe_config.retrain_final_model()) {
    // If the best model is needed, load it.
    RETURN_IF_ERROR(model::LoadModel(best_model_path, best_model));
  }

  return best_params;
}

utils::StatusOr<double>
HyperParameterOptimizerLearner::EvaluateCandidateLocally(
    const model::proto::GenericHyperParameters& candidate,
    const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config,
    const model::proto::TrainingConfigLinking& config_link,
    const dataset::VerticalDataset& train_dataset,
    absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
        valid_dataset,
    std::unique_ptr<AbstractModel>* model) const {
  ASSIGN_OR_RETURN(const auto base_learner, BuildBaseLearner(spe_config, true));
  RETURN_IF_ERROR(base_learner->SetHyperParameters(candidate));
  base_learner->set_stop_training_trigger(stop_training_trigger_);

  metric::proto::EvaluationResults evaluation;
  switch (spe_config.evaluation().source_case()) {
    case proto::Evaluation::SourceCase::SOURCE_NOT_SET:
    case proto::Evaluation::SourceCase::kSelfModelEvaluation:

      // Simple training + get the model self evaluation.
      ASSIGN_OR_RETURN(
          *model, base_learner->TrainWithStatus(train_dataset, valid_dataset));
      evaluation = (*model)->ValidationEvaluation();
      break;
  }

  ASSIGN_OR_RETURN(const auto score, EvaluationToScore(spe_config, evaluation));
  return score;
}

utils::StatusOr<double> HyperParameterOptimizerLearner::EvaluationToScore(
    const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config,
    const metric::proto::EvaluationResults& evaluation) const {
  // Extract the metric to optimize from the evaluation.
  metric::proto::MetricAccessor target_metric;
  if (spe_config.evaluation().has_metric()) {
    target_metric = spe_config.evaluation().metric();
  } else {
    ASSIGN_OR_RETURN(target_metric, internal::DefaultTargetMetric(evaluation));
  }

  auto metric_value = metric::GetMetric(evaluation, target_metric);
  if (!std::isfinite(metric_value)) {
    return absl::InvalidArgumentError("Non finite target metric value");
  }

  // Slip the metric value into a score.
  ASSIGN_OR_RETURN(const auto higher_is_better,
                   IsMaximization(spe_config, target_metric));
  if (!higher_is_better) {
    metric_value = -metric_value;
  }

  return metric_value;
}

namespace internal {

utils::StatusOr<metric::proto::MetricAccessor> DefaultTargetMetric(
    const metric::proto::EvaluationResults& evaluation) {
  if (evaluation.has_loss_value()) {
    metric::proto::MetricAccessor accessor;
    accessor.mutable_loss();
    return accessor;
  }

  if (evaluation.classification().rocs_size() == 3 &&
      evaluation.classification().rocs(2).has_auc()) {
    metric::proto::MetricAccessor accessor;
    accessor.mutable_classification()->mutable_one_vs_other()->mutable_auc();
    return accessor;
  }

  if (evaluation.classification().has_accuracy() ||
      evaluation.classification().has_confusion()) {
    metric::proto::MetricAccessor accessor;
    accessor.mutable_classification()->mutable_accuracy();
    return accessor;
  }

  if (evaluation.regression().has_sum_square_error()) {
    metric::proto::MetricAccessor accessor;
    accessor.mutable_regression()->mutable_rmse();
    return accessor;
  }

  if (evaluation.ranking().has_ndcg()) {
    metric::proto::MetricAccessor accessor;
    accessor.mutable_ranking()->mutable_ndcg();
    return accessor;
  }

  if (evaluation.uplift().has_qini()) {
    metric::proto::MetricAccessor accessor;
    accessor.mutable_uplift()->mutable_qini();
    return accessor;
  }

  return absl::InvalidArgumentError(
      "Not available default metric. Select the target metric manually.");
}

}  // namespace internal
}  // namespace hyperparameters_optimizer_v2
}  // namespace model
}  // namespace yggdrasil_decision_forests
