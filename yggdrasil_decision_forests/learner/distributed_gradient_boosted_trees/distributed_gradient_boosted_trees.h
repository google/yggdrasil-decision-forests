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

// Implementation of Gradient Boosted Trees learning algorithm using distributed
// training. Unless stated otherwise, the output model is the same as it would
// be if trained with the (non distributed) GradientBoostedTreesLearner.
//
// This algorithm is an extension of https://arxiv.org/abs/1804.06755
//
// DistributedGradientBoostedTreesLearner might only support a subset of the
// features (e.g. hyper-parameters) proposed in GradientBoostedTreesLearner. In
// this case, an error message will be raised during the
// DistributedGradientBoostedTreesLearner object construction.
//
// DistributedGradientBoostedTreesLearner only support trainig from a dataset
// path (i.e. training on in-memory dataset is not allowed).
//
// At the start of the training, the dataset is divided by columns and shards
// and then indexed.
//
// The learning algorithm support worker and manager interruption.
//

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_GRADIENT_BOOSTED_TREES_DISTRIBUTED_GRADIENT_BOOSTED_TREES_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_GRADIENT_BOOSTED_TREES_DISTRIBUTED_GRADIENT_BOOSTED_TREES_H_

#include <memory>

#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/load_balancer/load_balancer.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/splitter.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/worker.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_library.h"
#include "yggdrasil_decision_forests/learner/types.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_gradient_boosted_trees {

class DistributedGradientBoostedTreesLearner : public AbstractLearner {
 public:
  explicit DistributedGradientBoostedTreesLearner(
      const model::proto::TrainingConfig& training_config)
      : AbstractLearner(training_config) {}

  static constexpr char kRegisteredName[] =
      "DISTRIBUTED_GRADIENT_BOOSTED_TREES";

  static constexpr char kHParamWorkerLogs[] = "worker_logs";
  static constexpr char kHParamMaxUniqueValuesForDiscretizedNumerical[] =
      "max_unique_values_for_discretized_numerical";
  static constexpr char kHParamForceNumericalDiscretization[] =
      "force_numerical_discretization";

  utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset = {}) const override;

  utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      absl::string_view typed_path,
      const dataset::proto::DataSpecification& data_spec,
      const absl::optional<std::string>& typed_valid_path) const override;

  absl::Status SetHyperParametersImpl(
      utils::GenericHyperParameterConsumer* generic_hyper_params) override;

  utils::StatusOr<model::proto::GenericHyperParameterSpecification>
  GetGenericHyperParameterSpecification() const override;

  model::proto::LearnerCapabilities Capabilities() const override;
};

REGISTER_AbstractLearner(
    DistributedGradientBoostedTreesLearner,
    DistributedGradientBoostedTreesLearner::kRegisteredName);

namespace internal {

// One weak model being constructed.
struct WeakModel {
  std::unique_ptr<distributed_decision_tree::TreeBuilder> tree_builder;
};

// List of weak models being build.
typedef std::vector<WeakModel> WeakModels;

// List of worker indices.
typedef std::vector<int> WorkerIdxs;

// Map worker idx -> features.
typedef absl::flat_hash_map<int, std::vector<std::vector<int>>>
    WorkersToFeaturesMap;

// Loss and metric values. The metrics are controlled by the loss
// implementation.
struct Evaluation {
  float loss = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> metrics;
};

// Monitoring of training resources.
//
// Used to measure and display the time each stage last.
class Monitoring {
 public:
  // Those stage names match the stage name in worker.proto.
  enum Stages {
    kGetLabelStatistics,
    kSetInitialPredictions,
    kStartNewIter,
    kFindSplits,
    kEvaluateSplits,
    kShareSplits,
    kEndIter,
    kRestoreCheckpoint,
    kCreateCheckpoint,
    kStartTraining,
    kNumStages
  };

  void BeginTraining();
  void BeginDatasetCacheCreation();
  bool ShouldDisplayLogs();
  std::string InlineLogs();
  void BeginStage(Stages stage);
  void EndStage(Stages stage);
  void NewIter();
  void FindSplitWorkerReplyTime(int worker_idx, absl::Duration delay);
  absl::string_view StageName(Stages stage);

 private:
  // Index of the current stage. -1 if not stage is enabled.
  int current_stage_ = -1;

  // Starting time of the current stage.
  absl::Time begin_current_stage_;

  bool logs_already_displayed_ = false;

  // Last time the logs were displayed.
  absl::Time last_display_logs_;

  bool verbose_ = false;

  // Worker index and work duration of the last split stage.
  std::vector<std::pair<int, absl::Duration>> last_min_split_reply_times_;

  absl::Duration last_min_split_reply_time_;
  absl::Duration last_median_split_reply_time_;
  absl::Duration last_max_split_reply_time_;
  int last_fastest_worker_idx_ = -1;
  int last_slowest_worker_idx_ = -1;

  absl::Duration sum_min_split_reply_time_;
  absl::Duration sum_median_split_reply_time_;
  absl::Duration sum_max_split_reply_time_;
  int count_reply_times_ = 0;

  struct StageStats {
    absl::Duration sum_duration;
    size_t count{0};
  };

  // Statistics for each of the stages.
  StageStats stage_stats_[kNumStages];

  // Number of ran iterations so far.
  int num_iters_ = 0;

  // Starting time of the first iteration i.e. ~ start of the training.
  absl::Time time_first_iter_;
  absl::Time begin_current_iter_;
};

absl::Status SetDefaultHyperParameters(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const dataset::proto::DataSpecification& data_spec,
    proto::DistributedGradientBoostedTreesTrainingConfig* spe_config);

absl::Status CheckConfiguration(
    const model::proto::DeploymentConfig& deployment);

// Create the dataset cache (i.e. indexed dataset values) from a generic datast
// path.
absl::Status CreateDatasetCache(
    const model::proto::DeploymentConfig& deployment,
    absl::string_view cache_path,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DistributedGradientBoostedTreesTrainingConfig& spe_config,
    absl::string_view typed_path,
    const dataset::proto::DataSpecification& data_spec);

// Finalize the creation of a dataset cache from a partial dataset cache.
absl::Status CreateDatasetCacheFromPartialDatasetCache(
    const model::proto::DeploymentConfig& deployment,
    absl::string_view partial_cache_path, absl::string_view final_cache_path,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DistributedGradientBoostedTreesTrainingConfig& spe_config,
    const dataset::proto::DataSpecification& data_spec);

// Initiliaze the model for training.
utils::StatusOr<
    std::unique_ptr<gradient_boosted_trees::GradientBoostedTreesModel>>
InitializeModel(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DistributedGradientBoostedTreesTrainingConfig& spe_config,
    const dataset::proto::DataSpecification& data_spec,
    const gradient_boosted_trees::AbstractLoss& loss);

// Train the model from a dataset cache.
utils::StatusOr<
    std::unique_ptr<gradient_boosted_trees::GradientBoostedTreesModel>>
TrainWithCache(
    const model::proto::DeploymentConfig& deployment,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DistributedGradientBoostedTreesTrainingConfig& spe_config,
    absl::string_view cache_path, absl::string_view work_directory,
    const dataset::proto::DataSpecification& data_spec,
    const absl::string_view& log_directory, internal::Monitoring* monitoring);

// Run a single iteration of training.
absl::Status RunIteration(
    int iter_idx, const model::proto::TrainingConfigLinking& config_link,
    const proto::DistributedGradientBoostedTreesTrainingConfig& spe_config,
    const model::proto::TrainingConfig& weak_learner_train_config,
    const decision_tree::SetLeafValueFromLabelStatsFunctor& set_leaf_functor,
    distributed_decision_tree::LoadBalancer* load_balancer,
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<std::string>& metric_names,
    const std::vector<int>& features, const absl::string_view& log_directory,
    gradient_boosted_trees::GradientBoostedTreesModel* model,
    Evaluation* training_evaluation,
    distribute::AbstractManager* distribute_manager, utils::RandomEngine* rnd,
    internal::Monitoring* monitoring);

// Skips the next "num_skip" asynchronous answers from the manager.
absl::Status SkipAsyncAnswers(int num_skip,
                              distribute::AbstractManager* distribute_manager);

// If true, a checkpoint should be created as soon as possible.
bool ShouldCreateCheckpoint(
    int iter_idx, const absl::Time& time_last_checkpoint,
    const proto::DistributedGradientBoostedTreesTrainingConfig& spe_config);

// Restores the content of a checkpoint.
absl::Status RestoreManagerCheckpoint(
    int iter_idx, absl::string_view work_directory,
    std::unique_ptr<gradient_boosted_trees::GradientBoostedTreesModel>* model,
    decision_tree::proto::LabelStatistics* label_statistics,
    proto::Checkpoint* checkpoint);

// Creates a checkpoint of the model.
absl::Status CreateCheckpoint(
    int iter_idx,
    const gradient_boosted_trees::GradientBoostedTreesModel& model,
    absl::string_view work_directory,
    const decision_tree::proto::LabelStatistics& label_statistics,
    distribute::AbstractManager* distribute_manager,
    internal::Monitoring* monitoring);

// Console line showing the progress of training.
std::string TrainingLog(
    const gradient_boosted_trees::GradientBoostedTreesModel& model,
    const Evaluation& training_evaluation,
    const proto::DistributedGradientBoostedTreesTrainingConfig& spe_config,
    const std::vector<std::string>& metric_names,
    internal::Monitoring* monitoring,
    const distributed_decision_tree::LoadBalancer& load_balancer);

absl::Status InitializeDirectoryStructure(absl::string_view work_directory);

utils::StatusOr<std::unique_ptr<distribute::AbstractManager>>
InitializeDistributionManager(
    const model::proto::DeploymentConfig& deployment,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DistributedGradientBoostedTreesTrainingConfig& spe_config,
    absl::string_view cache_path, absl::string_view work_directory,
    const dataset::proto::DataSpecification& data_spec,
    const distributed_decision_tree::LoadBalancer& load_balancer);

// The following "Emit{stage_name}" function are calling the workers with the
// {stage_name} work. All these functions are blocking. See "worker.proto" for
// the definition of the stages.

utils::StatusOr<decision_tree::proto::LabelStatistics> EmitGetLabelStatistics(
    distribute::AbstractManager* distribute, internal::Monitoring* monitoring);

absl::Status EmitSetInitialPredictions(
    const decision_tree::proto::LabelStatistics& label_statistics,
    distribute::AbstractManager* distribute, internal::Monitoring* monitoring);

utils::StatusOr<std::vector<decision_tree::proto::LabelStatistics>>
EmitStartNewIter(int iter_idx, utils::RandomEngine::result_type seed,
                 distribute::AbstractManager* distribute,
                 internal::Monitoring* monitoring);

utils::StatusOr<std::vector<distributed_decision_tree::SplitPerOpenNode>>
EmitFindSplits(
    const proto::DistributedGradientBoostedTreesTrainingConfig& spe_config,
    const std::vector<int>& features,
    distributed_decision_tree::LoadBalancer* load_balancer,
    const WeakModels& weak_models, distribute::AbstractManager* distribute,
    utils::RandomEngine* rnd, internal::Monitoring* monitoring);

// Returns the list of workers that owns the split evaluation data.
absl::Status EmitEvaluateSplits(
    const std::vector<distributed_decision_tree::SplitPerOpenNode>&
        splits_per_weak_models,
    distributed_decision_tree::LoadBalancer* load_balancer,
    distribute::AbstractManager* distribute, utils::RandomEngine* rnd,
    internal::Monitoring* monitoring);

absl::Status EmitShareSplits(
    const std::vector<distributed_decision_tree::SplitPerOpenNode>&
        splits_per_weak_models,
    distributed_decision_tree::LoadBalancer* load_balancer,
    distribute::AbstractManager* distribute, internal::Monitoring* monitoring);

absl::Status EmitEndIter(int iter_idx, distribute::AbstractManager* distribute,
                         absl::optional<Evaluation*> training_evaluation,
                         internal::Monitoring* monitoring);

absl::Status EmitRestoreCheckpoint(int iter_idx, int num_shards,
                                   int num_weak_models,
                                   distribute::AbstractManager* distribute,
                                   internal::Monitoring* monitoring);

absl::Status EmitCreateCheckpoint(int iter_idx, size_t num_examples,
                                  int num_shards,
                                  absl::string_view work_directory,
                                  distribute::AbstractManager* distribute,
                                  internal::Monitoring* monitoring);

absl::Status EmitStartTraining(
    distribute::AbstractManager* distribute, internal::Monitoring* monitoring,
    distributed_decision_tree::LoadBalancer* load_balancer);

// Sets the load-balancing fields of in a worker request/result.
// The load balancing fields are only necessary for worker tasks that require
// access to the feature data.
absl::Status SetLoadBalancingRequest(
    int worker, distributed_decision_tree::LoadBalancer* load_balancer,
    proto::WorkerRequest* generic_request);

// Computes the list of active workers (i.e. workers having a job to do) and the
// features they have to check. Returns a mapping worker_idx -> weak_model_idx
// -> split_idx.
utils::StatusOr<WorkersToFeaturesMap> BuildActiveWorkers(
    const std::vector<distributed_decision_tree::SplitPerOpenNode>&
        splits_per_weak_models,
    const distributed_decision_tree::LoadBalancer& load_balancer,
    utils::RandomEngine* rnd);

// Lists the features used in a set of splits.
struct ActiveFeature {
  struct Item {
    int weak_model_idx;
    int split_idx;
  };
  std::vector<Item> splits;
};
typedef absl::flat_hash_map<int, ActiveFeature> ActiveFeaturesMap;
utils::StatusOr<ActiveFeaturesMap> ActiveFeatures(
    const std::vector<distributed_decision_tree::SplitPerOpenNode>&
        splits_per_weak_models);

// Sets the "weak learner" and "split idx" fields in a plan from the "feature"
// field.
absl::Status SetSplitsInPlan(
    const ActiveFeaturesMap& active_features,
    distributed_decision_tree::proto::SplitSharingPlan* plan);

// Samples the input features to send to a worker/weak model/node.
typedef std::vector<std::vector<std::vector<std::vector<int>>>>
    FeaturesPerWorkerWeakModelAndNode;
absl::Status SampleInputFeatures(
    const proto::DistributedGradientBoostedTreesTrainingConfig& spe_config,
    int num_workers, const std::vector<int>& features,
    const distributed_decision_tree::LoadBalancer& load_balancer,
    const WeakModels& weak_models, FeaturesPerWorkerWeakModelAndNode* samples,
    utils::RandomEngine* rnd);

// Randomly selects "num_sampled_features" features from "features".
absl::Status SampleFeatures(const std::vector<int>& features,
                            int num_sampled_features,
                            std::vector<int>* sampled_features,
                            utils::RandomEngine* rnd);

// Extracts the sampled features for a specific worker.
absl::Status ExactSampledFeaturesForWorker(
    const FeaturesPerWorkerWeakModelAndNode& sampled_features, int worker_idx,
    proto::WorkerRequest::FindSplits* request, int* num_selected_features);

}  // namespace internal

}  // namespace distributed_gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_GRADIENT_BOOSTED_TREES_DISTRIBUTED_GRADIENT_BOOSTED_TREES_H_
