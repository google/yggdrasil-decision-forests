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

// Abstract classes for learners a.k.a model trainers.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_ABSTRACT_LEARNER_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_ABSTRACT_LEARNER_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/fold_generator.pb.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"

namespace yggdrasil_decision_forests {
namespace model {

class AbstractLearner {
 public:
  explicit AbstractLearner(const proto::TrainingConfig& training_config)
      : training_config_(training_config) {}

  virtual ~AbstractLearner() = default;

  // Trains a model using the dataset stored on disk at the path "typed_path".
  //
  // A typed path is a dataset with a format prefix. prefix format. For example,
  // "csv:/tmp/dataset.csv". The path supports sharding, globbing and comma
  // separation. See the "Dataset path and format" section of the user manual
  // for more details: go/ydf_documentation/user_manual.md#dataset-path-and-format
  //
  // Algorithms with the "use_validation_dataset" capability use a validation
  // dataset for training. If "typed_valid_path" is provided, it will be used
  // for validation. If "typed_valid_path" is not provided, a validation dataset
  // will be extracted from the training dataset. If the algorithm does not have
  // the "use_validation_dataset" capability, "typed_valid_path" is ignored.
  virtual utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      const absl::string_view typed_path,
      const dataset::proto::DataSpecification& data_spec,
      const absl::optional<std::string>& typed_valid_path = {}) const;

  // Trains a model using the dataset stored on memory .
  //
  // Algorithms with the "use_validation_dataset" capability use a validation
  // dataset for training. If "valid_dataset" is provided, it will be used
  // for validation. If "valid_dataset" is not provided, a validation dataset
  // will be extracted from the training dataset. If the algorithm does not have
  // the "use_validation_dataset" capability, "valid_dataset" is ignored.
  virtual utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset = {}) const = 0;

  // Similar as TrainWithStatus, but crash in case of error.
  // [Deprecated]
  virtual std::unique_ptr<AbstractModel> Train(
      const absl::string_view typed_path,
      const dataset::proto::DataSpecification& data_spec) const;

  // Trains and returns a model from a training dataset stored on drive.
  // [Deprecated]
  virtual std::unique_ptr<AbstractModel> Train(
      const dataset::VerticalDataset& train_dataset) const;

  // Obtains the linked training configuration i.e. match feature names to
  // feature idxs from a training configuration and a dataspec.
  //
  // Input feature with only missing values are removed with a warning message.
  static absl::Status LinkTrainingConfig(
      const proto::TrainingConfig& training_config,
      const dataset::proto::DataSpecification& data_spec,
      proto::TrainingConfigLinking* config_link);

  // Accessor to the training configuration. Contains the definition of the task
  // (e.g. input features, label, weights) as well as the hyper parameters of
  // the learning.
  proto::TrainingConfig* mutable_training_config() { return &training_config_; }
  const proto::TrainingConfig& training_config() const {
    return training_config_;
  }

  // Update the training config hyper parameters with a generic hyper parameter
  // definition. Hyper parameter fields non defined in "generic_hyper_params"
  // are not modified.
  virtual absl::Status SetHyperParameters(
      const proto::GenericHyperParameters& generic_hyper_params);

  // The function for learners to override when setting hyper-parameters.
  virtual absl::Status SetHyperParametersImpl(
      utils::GenericHyperParameterConsumer* generic_hyper_params);

  // Get a description of the generic hyper-parameters supported by the learner.
  virtual utils::StatusOr<proto::GenericHyperParameterSpecification>
  GetGenericHyperParameterSpecification() const;

  // Returns a list of hyper-parameter sets that outperforms the default
  // hyper-parameters (either generally or in specific scenarios). Like default
  // hyper-parameters, existing pre-defined hyper-parameters cannot change.
  // However new versions can be added with a same name.
  virtual std::vector<proto::PredefinedHyperParameterTemplate>
  PredefinedHyperParameters() const {
    return {};
  }

  // Pre-defined space of hyper-parameters to be automatically optimized.
  // Returns a failing status if the learner does not provide a pre-defined
  // space of hyper-parameter to optimize.
  virtual utils::StatusOr<proto::HyperParameterSpace>
  PredefinedHyperParameterSpace() const;

  // Accessor to the deployment configuration.
  const proto::DeploymentConfig& deployment() const { return deployment_; }
  proto::DeploymentConfig* mutable_deployment() { return &deployment_; }

  const std::string& log_directory() const { return log_directory_; }
  void set_log_directory(absl::string_view log_path) {
    log_directory_ = std::string(log_path);
  }

  // Detects configuration errors and warnings.
  static absl::Status CheckConfiguration(
      const dataset::proto::DataSpecification& data_spec,
      const model::proto::TrainingConfig& config,
      const model::proto::TrainingConfigLinking& config_link,
      const model::proto::DeploymentConfig& deployment);

  // Gets the capabilities of the learning algorithm.
  virtual proto::LearnerCapabilities Capabilities() const { return {}; }

  // Checks if the training config is compatible with the learner capabilities.
  absl::Status CheckCapabilities() const;

  // Register a trigger to stop the training. Later if a trigger is set to true
  // at any time during training (e.g. during an interrupt caused by user
  // control+C one may want to set it to true), the training algorithm will
  // gracefully interrupt. This is done by polling, so expect a little latency
  // to respond to the trigger setting.
  // If the training is interrupted, the output model is valid but partially (or
  // not at all) trained. If trigger==nullptr (default behavior), the training
  // cannot be stopped, and will continue until finished.
  void set_stop_training_trigger(std::atomic<bool>* trigger) {
    stop_training_trigger_ = trigger;
  }

 protected:
  // Training configuration. Contains the hyper parameters of the learner.
  proto::TrainingConfig training_config_;

  // Deployment configuration. Defines the computing resources to use for the
  // training.
  proto::DeploymentConfig deployment_;

  // If non empty, this directory can be used by the learner to export any
  // training log data. Such data can include (non exhaustive) texts, tables and
  // plots. The learner is not guarantied to populate the log directory. It is
  // the responsibility of the learner to create this directory if it does not
  // exist.
  std::string log_directory_;

  // If set, the training should stop is "*stop_training_trigger_" is true.
  // If the training is interrupted, the output model is valid but partially (or
  // not at all) trained. If flag==nullptr (default behavior), the flag is
  // ignored.
  std::atomic<bool>* stop_training_trigger_ = nullptr;
};

REGISTRATION_CREATE_POOL(AbstractLearner, const proto::TrainingConfig&);

#define REGISTER_AbstractLearner(name, key) \
  REGISTRATION_REGISTER_CLASS(name, key, AbstractLearner);

// Generic hyper parameter names.
static constexpr char kHParamMaximumTrainingDurationSeconds[] =
    "maximum_training_duration_seconds";
static constexpr char kHParamMaximumModelSizeInMemoryInBytes[] =
    "maximum_model_size_in_memory_in_bytes";
static constexpr char kHParamRandomSeed[] = "random_seed";

// Check a set of hyper parameter against an hyper parameter specification.
absl::Status CheckGenericHyperParameterSpecification(
    const proto::GenericHyperParameters& params,
    const proto::GenericHyperParameterSpecification& spec);

// Evaluates a learner using a fold generator.
//
// Note: To evaluate a pre-trained model, see "AbstractModel::Evaluate".
//
// For each fold group, a model is trained and evaluate. The final evaluation is
// the average evaluation of all the models.
//
// If the fold generator contains multiple fold groups (e.g. cross-validation),
// the different models will be trained and evaluated in parallel using the
// "deployment_evaluation" specification. By default, 6 models will be evaluated
// in parallel on the local machine.
//
// "deployment_evaluation" specifies the computing resources for the evaluation.
// It does not impact the computing resources training (contained in
// "learner.deployment()").
//
// If your learner is already parallelized, training and evaluating a single
// models at a time might be best i.e. deployment_evaluation = { .num_threads =
// 1 }.
metric::proto::EvaluationResults EvaluateLearner(
    const AbstractLearner& learner, const dataset::VerticalDataset& dataset,
    const utils::proto::FoldGenerator& fold_generator,
    const metric::proto::EvaluationOptions& evaluation_options,
    const proto::DeploymentConfig& deployment_evaluation = {});

// Initialize the abstract model fields
void InitializeModelWithAbstractTrainingConfig(
    const proto::TrainingConfig& training_config,
    const proto::TrainingConfigLinking& training_config_linking,
    AbstractModel* model);

// Copies or set with the default values the metadata of the model.
void InitializeModelMetadataWithAbstractTrainingConfig(
    const proto::TrainingConfig& training_config, AbstractModel* model);

// Copies the part of the training configuration related to the problem
// definition. Fails if the "dst" already existing configuration is not
// compatible or contradictory.
absl::Status CopyProblemDefinition(const proto::TrainingConfig& src,
                                   proto::TrainingConfig* dst);

// Create a dataset loading configuration adapted to the training configuration
// link. Skill unused features and examples with zero weights.
dataset::LoadConfig OptimalDatasetLoadingConfig(
    const proto::TrainingConfigLinking& link_config);

}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_ABSTRACT_LEARNER_H_
