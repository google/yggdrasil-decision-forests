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

// Find the hyper-parameters that maximize/minimize a specific metric of a
// sub-learner. For example, find the hyper-parameters of the
// GradientBoostedTrees learner that maximize the accuracy of its model on a
// given dataset.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_HYPERPARAMETERS_OPTIMIZER_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_HYPERPARAMETERS_OPTIMIZER_H_

#include <memory>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/generic_worker/generic_worker.pb.h"
#include "yggdrasil_decision_forests/learner/hyperparameters_optimizer/hyperparameters_optimizer.pb.h"
#include "yggdrasil_decision_forests/learner/hyperparameters_optimizer/optimizer_interface.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace hyperparameters_optimizer_v2 {

class HyperParameterOptimizerLearner : public AbstractLearner {
 public:
  explicit HyperParameterOptimizerLearner(
      const model::proto::TrainingConfig& training_config);

  // Unique identifier of the learning algorithm.
  static constexpr char kRegisteredName[] = "HYPERPARAMETER_OPTIMIZER";

  utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset = {}) const override;

  utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      const absl::string_view typed_path,
      const dataset::proto::DataSpecification& data_spec,
      const absl::optional<std::string>& typed_valid_path = {}) const override;

  // Sets the hyper-parameters of the learning algorithm from "generic hparams".
  absl::Status SetHyperParametersImpl(
      utils::GenericHyperParameterConsumer* generic_hyper_params) override;

  // Gets the available generic hparams.
  utils::StatusOr<model::proto::GenericHyperParameterSpecification>
  GetGenericHyperParameterSpecification() const override;

  // Gets a description of what the learning algorithm can do.
  model::proto::LearnerCapabilities Capabilities() const override {
    model::proto::LearnerCapabilities capabilities;
    capabilities.set_support_max_training_duration(true);
    return capabilities;
  }

 private:
  // Trains from file (needed for distributed training) from a dataset in
  // memory.
  utils::StatusOr<std::unique_ptr<AbstractModel>> TrainFromFileOnMemoryDataset(
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset) const;

  // Aggregates the user inputs and automated logic output and returns the
  // effectively training configuration effectively used for training.
  absl::Status GetEffectiveConfiguration(
      const dataset::proto::DataSpecification& data_spec,
      model::proto::TrainingConfig* effective_config,
      model::proto::TrainingConfigLinking* effective_config_link) const;

  // Assembles the effective search space.
  utils::StatusOr<model::proto::HyperParameterSpace> BuildSearchSpace(
      const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config,
      const AbstractLearner& base_learner) const;

  // Instantiates a base learner.
  utils::StatusOr<std::unique_ptr<AbstractLearner>> BuildBaseLearner(
      const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config,
      const bool for_tuning) const;

  // Searches for the best hyperparameter in process from a dataset loaded in
  // memory. The dataset object is shared among the trials.
  utils::StatusOr<model::proto::GenericHyperParameters>
  SearchBestHyperparameterInProcess(
      const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config,
      const model::proto::TrainingConfigLinking& config_link,
      const model::proto::GenericHyperParameterSpecification& search_space_spec,
      const model::proto::HyperParameterSpace& search_space,
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset,
      std::unique_ptr<AbstractModel>* best_model,
      model::proto::HyperparametersOptimizerLogs* logs) const;

  // Searches for the best hyperparameter using distributed training from a disk
  // dataset.
  utils::StatusOr<model::proto::GenericHyperParameters>
  SearchBestHyperparameterDistributed(
      const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config,
      const model::proto::TrainingConfigLinking& config_link,
      const model::proto::GenericHyperParameterSpecification& search_space_spec,
      const model::proto::HyperParameterSpace& search_space,
      const absl::string_view typed_train_path,
      const dataset::proto::DataSpecification& data_spec,
      const absl::optional<std::string>& typed_valid_path,
      std::unique_ptr<AbstractModel>* best_model,
      distribute::AbstractManager* manager,
      model::proto::HyperparametersOptimizerLogs* logs) const;

  // If true, the metric needs to be maximized. If false, the metric needs to be
  // minimized.
  utils::StatusOr<bool> IsMaximization(
      const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config,
      const metric::proto::MetricAccessor& metric_accessor) const;

  // Evaluates the quality of a candidate locally i.e. train and evaluate the
  // model locally.
  utils::StatusOr<double> EvaluateCandidateLocally(
      const model::proto::GenericHyperParameters& candidate,
      const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config,
      const model::proto::TrainingConfigLinking& config_link,
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset,
      std::unique_ptr<AbstractModel>* model) const;

  // Creates an initialized distribute manager with "GENERIC_WORKER" workers.
  utils::StatusOr<std::unique_ptr<distribute::AbstractManager>>
  CreateDistributeManager() const;

  // Trains a model remotely.
  utils::StatusOr<std::unique_ptr<AbstractModel>> TrainRemoteModel(
      const model::proto::TrainingConfig& config,
      const model::proto::TrainingConfigLinking& config_link,
      const model::proto::DeploymentConfig& deployment_config,
      const model::proto::GenericHyperParameters& generic_hyper_params,
      const absl::string_view typed_train_path,
      const dataset::proto::DataSpecification& data_spec,
      const absl::optional<std::string>& typed_valid_path,
      distribute::AbstractManager* manager) const;

  // Extracts the score from an evaluation. For scores, larger is always better.
  // Scores can be negative.
  utils::StatusOr<double> EvaluationToScore(
      const proto::HyperParametersOptimizerLearnerTrainingConfig& spe_config,
      const metric::proto::EvaluationResults& evaluation) const;
};

REGISTER_AbstractLearner(HyperParameterOptimizerLearner,
                         HyperParameterOptimizerLearner::kRegisteredName);

namespace internal {

// Gets the default metric to evaluate in the list:
// loss > auc > accuracy > rmse ndcg > qini. Fails if none of those metrics
// are defined.
utils::StatusOr<metric::proto::MetricAccessor> DefaultTargetMetric(
    const metric::proto::EvaluationResults& evaluation);

}  // namespace internal

}  // namespace hyperparameters_optimizer_v2
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_HYPERPARAMETER_OPTIMIZER_RANDOM_H_
