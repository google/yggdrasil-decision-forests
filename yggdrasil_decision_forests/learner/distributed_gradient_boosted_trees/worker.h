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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_GRADIENT_BOOSTED_TREES_WORKER_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_GRADIENT_BOOSTED_TREES_WORKER_H_

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/worker.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_library.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_gradient_boosted_trees {

class DistributedGradientBoostedTreesWorker
    : public distribute::AbstractWorker {
 public:
  static constexpr char kWorkerKey[] = "DISTRIBUTED_GRADIENT_BOOSTED_TREES";

  absl::Status Setup(distribute::Blob serialized_welcome) override;

  utils::StatusOr<distribute::Blob> RunRequest(
      distribute::Blob serialized_request) override;

  absl::Status Done() override;

  virtual ~DistributedGradientBoostedTreesWorker();

 private:
  // Internal data related to one weak model e.g. a decision tree.
  struct WeakModel {
    // Gradient / hessian of the GBT i.e. pseudo response of the weak model.
    std::vector<float> gradients;
    std::vector<float> hessians;

    // Accessor to the pseudo response data.
    std::unique_ptr<distributed_decision_tree::AbstractLabelAccessor>
        label_accessor;

    // How to access the data in the "label_accessor".
    distributed_decision_tree::LabelAccessorType label_accessor_type;

    // Example to open node assignation.
    distributed_decision_tree::ExampleToNodeMap example_to_node;

    // Decision tree being build.
    std::unique_ptr<distributed_decision_tree::TreeBuilder> tree_builder;

    // Label statistics (from the pseudo response) for each open node.
    distributed_decision_tree::LabelStatsPerNode label_stats_per_node;

    // Last evaluation of the split value requested in "EvaluateSplits".
    distributed_decision_tree::SplitEvaluationPerOpenNode last_split_evaluation;

    // Split evaluation of the current iteration. Those contains both the split
    // evaluation computed by this worker (as the end of the "EvaluateSplits"
    // stage) as well as the split evaluation received from other workers (as
    // the end of the "ShareSplits" stage).
    distributed_decision_tree::SplitPerOpenNode last_splits;

    bool has_multiple_node_idxs;
  };

  // Internal data related to the training of one layer of a weak model.
  struct WeakModelLayer {
    // Split for each open nodes.
    distributed_decision_tree::SplitPerOpenNode splits;

    // Evaluation of the splits for each open nodes.
    distributed_decision_tree::SplitEvaluationPerOpenNode split_evaluations;
  };

  utils::StatusOr<distribute::Blob> RunRequestImp(
      distribute::Blob serialized_request);

  // Simulates worker failures and restart. A failure is artificially generated
  // for each worker (by index) and for each task exactly once. Failure are
  // generated according to the iteration index. For example, the failure on
  // request #4 and worker #2 might be generated on iteration #16 (not exact
  // values). Used for testing.
  void MaybeSimulateFailure(proto::WorkerRequest::TypeCase request_type);

  // The following methods with stage names are defined in "worker.proto".

  absl::Status GetLabelStatistics(
      const proto::WorkerRequest::GetLabelStatistics& request,
      proto::WorkerResult::GetLabelStatistics* answer);

  absl::Status SetInitialPredictions(
      const proto::WorkerRequest::SetInitialPredictions& request,
      proto::WorkerResult::SetInitialPredictions* answer);

  absl::Status StartNewIter(const proto::WorkerRequest::StartNewIter& request,
                            proto::WorkerResult::StartNewIter* answer);

  absl::Status FindSplits(const proto::WorkerRequest::FindSplits& request,
                          proto::WorkerResult::FindSplits* answer);

  absl::Status EvaluateSplits(
      const proto::WorkerRequest::EvaluateSplits& request,
      proto::WorkerResult::EvaluateSplits* answer);

  absl::Status ShareSplits(const proto::WorkerRequest::ShareSplits& request,
                           proto::WorkerResult::ShareSplits* answer,
                           proto::WorkerResult* generic_answer);

  absl::Status GetSplitValue(const proto::WorkerRequest::GetSplitValue& request,
                             proto::WorkerResult::GetSplitValue* answer);

  absl::Status EndIter(const proto::WorkerRequest::EndIter& request,
                       proto::WorkerResult::EndIter* answer);

  absl::Status RestoreCheckpoint(
      const proto::WorkerRequest::RestoreCheckpoint& request,
      proto::WorkerResult::RestoreCheckpoint* answer);

  absl::Status CreateCheckpoint(
      const proto::WorkerRequest::CreateCheckpoint& request,
      proto::WorkerResult::CreateCheckpoint* answer);

  absl::Status StartTraining(const proto::WorkerRequest::StartTraining& request,
                             proto::WorkerResult::StartTraining* answer);

  // End of stage names.

  // Change the features owned by the worker.
  absl::Status UpdateOwnedFeatures(std::vector<int> features);

  // Initiate the pre-loading of features for future usage.
  //
  // True true if any preloading work is in progress.
  utils::StatusOr<bool> PreloadFutureOwnedFeatures(
      const proto::WorkerRequest::FutureOwnedFeatures& future_owned_features);

  // Merges the split evaluation into the "last_split_evaluation" field of each
  // weak model. After this function call, "src_split_values" is invalid. This
  // method is thread safe (can be called at the same time from different
  // threads).
  absl::Status MergingSplitEvaluationToLastSplitEvaluation(
      proto::WorkerResult::GetSplitValue* src_split_values);

  // Loss of a set of predictions.
  absl::Status Loss(
      distributed_decision_tree::dataset_cache::DatasetCacheReader* dataset,
      const std::vector<float>& predictions, float* loss_value,
      std::vector<float>* secondary_metric);

  // Initialize the working memory of the worker. This stage requires the number
  // of weak models and cannot be done during the "setup" stage. This method is
  // called by the "SetInitialPredictions" message (i.e. the start of the
  // training) or "RestoreCheckpoint" message (i.e. resuming training from a
  // checkpoint).
  absl::Status InitializerWorkingMemory(int num_weak_models);

  // Skips the next "num_skip" answers on the worker-to-worker async channel.
  absl::Status SkipAsyncWorkerToWorkerAnswers(int num_skip);

  // Static configuration provided by the manager. Initialized during the
  // "setup" stage.
  proto::WorkerWelcome welcome_;

  // Training dataset. Initialized during the "setup" stage.
  std::unique_ptr<distributed_decision_tree::dataset_cache::DatasetCacheReader>
      dataset_;

  // Current training iteration. -1 before the training starts. Updated during
  // training.
  int iter_idx_ = -1;

  // Unique identifier of the current iteration. Updated during training.
  std::string iter_uid_;

  // Random seed for the current iteration. Updated during training.
  uint64_t seed_;

  // Training loss.  Initialized during the "setup" stage.
  std::unique_ptr<gradient_boosted_trees::AbstractLoss> loss_;

  // Accumulated predictions of the current model.
  // Array of size num_examples x num_output_dim. Initialized with the
  // "SetInitialPredictions" message.
  std::vector<float> predictions_;

  // Training data for each weak model in the current iteration. Initialized
  // with the "SetInitialPredictions" message.
  std::vector<WeakModel> weak_models_;

  // Accessor to the pseudo response. Initialized with the
  // "SetInitialPredictions" message.
  gradient_boosted_trees::GradientDataRef gradient_ref_;

  // The worker received the initial model predictions from the manager.
  // Initialized with the "SetInitialPredictions" message.
  std::atomic<bool> received_initial_predictions_{false};

  utils::RandomEngine random_;

  // List of worker request (i.e. proto::WorkerRequest::k*) where an failure was
  // already simulate. Only for unit testing.
  std::unordered_set<int> debug_forced_failure_;

  // Working threads. Initialized during the "setup" stage.
  std::unique_ptr<utils::concurrency::ThreadPool> thread_pool_;

  // True iff. the worker was asked to stop i.e. "Done" was called.
  std::atomic<bool> stop_{false};

  // Number of running "RunRequest".
  int num_running_requests_ = 0;
  utils::concurrency::Mutex mutex_num_running_requests_;

  // Time taken to load the features of the dataset in memory.
  absl::Duration dataset_feature_duration_;
  int dataset_num_features_loaded_ = 0;

  // Prints details about the computation with LOG(INFO).
  bool worker_logs_ = true;
};

// Extract the requested features in a FindSplits request.
utils::StatusOr<std::vector<std::vector<int>>> ExtractInputFeaturesPerNodes(
    const proto::WorkerRequest::FindSplits::FeaturePerNode& src);

// Update the "predictions_" of the examples in nodes closed in
// "node_remapping".
absl::Status UpdateClosingNodesPredictions(
    const distributed_decision_tree::ExampleToNodeMap& example_to_node,
    const distributed_decision_tree::LabelStatsPerNode& label_stats_per_node,
    const distributed_decision_tree::NodeRemapping& node_remapping,
    const distributed_decision_tree::SetLeafValueFromLabelStatsFunctor&
        set_leaf_functor,
    int weak_model_idx, int num_weak_models, std::vector<float>* predictions,
    utils::concurrency::ThreadPool* thread_pool);

}  // namespace distributed_gradient_boosted_trees
}  // namespace model

namespace distribute {
using DistributedGradientBoostedTreesWorker = model::
    distributed_gradient_boosted_trees::DistributedGradientBoostedTreesWorker;
REGISTER_Distribution_Worker(DistributedGradientBoostedTreesWorker,
                             DistributedGradientBoostedTreesWorker::kWorkerKey);
}  // namespace distribute

}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_GRADIENT_BOOSTED_TREES_WORKER_H_
