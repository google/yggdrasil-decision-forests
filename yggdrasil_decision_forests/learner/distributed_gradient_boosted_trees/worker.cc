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

#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/worker.h"

#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/common.h"
#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_library.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_gradient_boosted_trees {

constexpr char DistributedGradientBoostedTreesWorker::kWorkerKey[];

DistributedGradientBoostedTreesWorker::
    ~DistributedGradientBoostedTreesWorker() {
  if (worker_logs_) {
    LOG(INFO) << "Destroying DistributedGradientBoostedTreesWorker";
  }
}

absl::Status DistributedGradientBoostedTreesWorker::Setup(
    distribute::Blob serialized_welcome) {
  ASSIGN_OR_RETURN(welcome_, utils::ParseBinaryProto<proto::WorkerWelcome>(
                                 serialized_welcome));

  const auto& spe_config = welcome_.train_config().GetExtension(
      proto::distributed_gradient_boosted_trees_config);
  worker_logs_ = spe_config.worker_logs();

  if (worker_logs_) {
    LOG(INFO) << "Initializing DistributedGradientBoostedTreesWorker";
  }

  // Load the dataset.
  const auto& worker_features = welcome_.owned_features(WorkerIdx()).features();
  auto read_options = spe_config.dataset_reader_options();
  *read_options.mutable_features() = worker_features;
  ASSIGN_OR_RETURN(
      dataset_,
      distributed_decision_tree::dataset_cache::DatasetCacheReader::Create(
          welcome_.cache_path(), read_options));

  dataset_feature_duration_ = dataset_->load_in_memory_duration();
  dataset_num_features_loaded_ = dataset_->features().size();

  // Training loss.
  ASSIGN_OR_RETURN(
      loss_,
      gradient_boosted_trees::CreateLoss(
          spe_config.gbt().loss(), welcome_.train_config().task(),
          welcome_.dataspec().columns(welcome_.train_config_linking().label()),
          spe_config.gbt()));

  // Threadpool.
  if (worker_logs_) {
    LOG(INFO) << "Create thread pool with "
              << welcome_.deployment_config().num_threads() << " threads";
  }
  thread_pool_ = absl::make_unique<utils::concurrency::ThreadPool>(
      "generic", welcome_.deployment_config().num_threads());
  thread_pool_->StartWorkers();

  return absl::OkStatus();
}

utils::StatusOr<distribute::Blob>
DistributedGradientBoostedTreesWorker::RunRequest(
    distribute::Blob serialized_request) {
  {
    utils::concurrency::MutexLock l(&mutex_num_running_requests_);
    num_running_requests_++;
  }

  auto status_or = RunRequestImp(std::move(serialized_request));

  {
    utils::concurrency::MutexLock l(&mutex_num_running_requests_);
    num_running_requests_--;
    if (stop_) {
      if (num_running_requests_ == 0) {
        LOG(INFO) << "Clear the worker memory";
        dataset_.reset();
        loss_.reset();
        predictions_.clear();
        weak_models_.clear();
        thread_pool_.reset();
      } else {
        LOG(INFO) << "Will clear the worker memory when all requests are done ("
                  << num_running_requests_ << " requeres remaining)";
      }
    }
  }

  return status_or;
}

utils::StatusOr<distribute::Blob>
DistributedGradientBoostedTreesWorker::RunRequestImp(
    distribute::Blob serialized_request) {
  const auto begin = absl::Now();
  ASSIGN_OR_RETURN(auto request, utils::ParseBinaryProto<proto::WorkerRequest>(
                                     serialized_request));

  const auto& spe_config = welcome_.train_config().GetExtension(
      proto::distributed_gradient_boosted_trees_config);
  if (worker_logs_) {
    LOG(INFO) << "Worker #" << WorkerIdx() << " received request "
              << request.type_case();
  }
  proto::WorkerResult result;
  result.set_request_id(request.request_id());
  result.set_worker_idx(WorkerIdx());

  // [For unit testing only] Simulate failure of the workers.
  // Each message type (i.e. request.type_case()) will fail one on each worker.
  if (spe_config.internal().simulate_worker_failure()) {
    MaybeSimulateFailure(request.type_case());
  }

  if (request.type_case() != proto::WorkerRequest::kStartTraining &&
      request.type_case() != proto::WorkerRequest::kGetLabelStatistics &&
      request.type_case() != proto::WorkerRequest::kSetInitialPredictions &&
      request.type_case() != proto::WorkerRequest::kRestoreCheckpoint) {
    bool missing_data = false;
    if (!received_initial_predictions_) {
      missing_data = true;
    }
    if (request.type_case() != proto::WorkerRequest::kStartNewIter &&
        request.type_case() != proto::WorkerRequest::kCreateCheckpoint &&
        iter_idx_ == -1) {
      missing_data = true;
    }
    if (missing_data) {
      // The worker was restarted during the training of this tree. Tell the
      // manager to restart the training of this tree.
      LOG(WARNING) << "Incomplete information to run a request #"
                   << request.type_case() << " on worker #" << WorkerIdx()
                   << ". Ask manager to restart";
      result.set_request_restart_iter(true);
      return result.SerializeAsString();
    }
  }

  // Make sure the requested features are available.
  // Such change open append when the worker is restarted.
  if (request.has_owned_features()) {
    RETURN_IF_ERROR(
        UpdateOwnedFeatures({request.owned_features().features().begin(),
                             request.owned_features().features().end()}));
  }

  // Non-blocking pre-loading of the features that will be required in the
  // future.
  if (request.has_future_owned_features()) {
    RETURN_IF_ERROR(
        PreloadFutureOwnedFeatures(request.future_owned_features()).status());
  }

  switch (request.type_case()) {
    case proto::WorkerRequest::kGetLabelStatistics:
      RETURN_IF_ERROR(
          GetLabelStatistics(request.get_label_statistics(),
                             result.mutable_get_label_statistics()));
      break;

    case proto::WorkerRequest::kSetInitialPredictions:
      RETURN_IF_ERROR(
          SetInitialPredictions(request.set_initial_predictions(),
                                result.mutable_set_initial_predictions()));
      break;

    case proto::WorkerRequest::kStartNewIter:
      RETURN_IF_ERROR(StartNewIter(request.start_new_iter(),
                                   result.mutable_start_new_iter()));
      break;

    case proto::WorkerRequest::kFindSplits:
      RETURN_IF_ERROR(
          FindSplits(request.find_splits(), result.mutable_find_splits()));
      break;

    case proto::WorkerRequest::kEvaluateSplits:
      RETURN_IF_ERROR(EvaluateSplits(request.evaluate_splits(),
                                     result.mutable_evaluate_splits()));
      break;

    case proto::WorkerRequest::kShareSplits:
      RETURN_IF_ERROR(ShareSplits(request.share_splits(),
                                  result.mutable_share_splits(), &result));
      break;

    case proto::WorkerRequest::kGetSplitValue:
      // TODO(gbm): Since the answer is large and the same for all the workers,
      // can the answer be somehow not duplicated in memory?
      RETURN_IF_ERROR(GetSplitValue(request.get_split_value(),
                                    result.mutable_get_split_value()));
      break;

    case proto::WorkerRequest::kEndIter:
      RETURN_IF_ERROR(EndIter(request.end_iter(), result.mutable_end_iter()));
      break;

    case proto::WorkerRequest::kRestoreCheckpoint:
      RETURN_IF_ERROR(RestoreCheckpoint(request.restore_checkpoint(),
                                        result.mutable_restore_checkpoint()));
      break;

    case proto::WorkerRequest::kCreateCheckpoint:
      RETURN_IF_ERROR(CreateCheckpoint(request.create_checkpoint(),
                                       result.mutable_create_checkpoint()));
      break;

    case proto::WorkerRequest::kStartTraining:
      RETURN_IF_ERROR(StartTraining(request.start_training(),
                                    result.mutable_start_training()));
      break;

    case proto::WorkerRequest::TYPE_NOT_SET:
      return absl::InvalidArgumentError("Request without type");
  }

  const auto runtime = absl::Now() - begin;
  if (worker_logs_) {
    LOG(INFO) << "Worker #" << WorkerIdx() << " answered request "
              << request.type_case() << " in " << runtime;
  }
  result.set_runtime_seconds(absl::ToDoubleSeconds(runtime));

  // Update the manager about pre-loading status.
  result.set_preloading_work_in_progress(
      dataset_->IsNonBlockingLoadingInProgress());

  return result.SerializeAsString();
}

void DistributedGradientBoostedTreesWorker::MaybeSimulateFailure(
    const proto::WorkerRequest::TypeCase request_type) {
  if (iter_idx_ > 8) {
    if (((iter_idx_ * NumWorkers() + WorkerIdx()) % 12) == request_type) {
      if (debug_forced_failure_.find(request_type) ==
          debug_forced_failure_.end()) {
        debug_forced_failure_.insert(request_type);
        LOG(WARNING) << "[!!!!!] Simulate the failure and restart of worker #"
                     << WorkerIdx() << " on message " << request_type
                     << " and iteration " << iter_idx_;

        // Reset the worker to its initial state.
        received_initial_predictions_ = false;
        iter_idx_ = -1;
      }
    }
  }
}

absl::Status DistributedGradientBoostedTreesWorker::Done() {
  LOG(INFO) << "Done called on the worker (" << num_running_requests_
            << " running requests)";
  stop_ = true;
  return absl::OkStatus();
}

absl::Status DistributedGradientBoostedTreesWorker::GetLabelStatistics(
    const proto::WorkerRequest::GetLabelStatistics& request,
    proto::WorkerResult::GetLabelStatistics* const answer) {
  switch (welcome_.train_config().task()) {
    case model::proto::Task::CLASSIFICATION: {
      const auto num_classes =
          welcome_.dataspec()
              .columns(welcome_.train_config_linking().label())
              .categorical()
              .number_of_unique_values();
      distributed_decision_tree::ClassificationLabelAccessor label_accessor(
          dataset_->categorical_labels(), dataset_->weights(),
          /*num_classes=*/num_classes);

      RETURN_IF_ERROR(distributed_decision_tree::AggregateLabelStatistics(
          label_accessor, welcome_.train_config().task(),
          distributed_decision_tree::LabelAccessorType::kAutomatic,
          answer->mutable_label_statistics(), thread_pool_.get()));
    } break;
    case model::proto::Task::REGRESSION: {
      distributed_decision_tree::RegressionLabelAccessor label_accessor(
          dataset_->regression_labels(), dataset_->weights());

      RETURN_IF_ERROR(distributed_decision_tree::AggregateLabelStatistics(
          label_accessor, welcome_.train_config().task(),
          distributed_decision_tree::LabelAccessorType::kAutomatic,
          answer->mutable_label_statistics(), thread_pool_.get()));
    } break;
    default:
      return absl::InvalidArgumentError("Not supported task");
  }

  return absl::OkStatus();
}

absl::Status DistributedGradientBoostedTreesWorker::InitializerWorkingMemory(
    int num_weak_models) {
  const auto& spe_config = welcome_.train_config().GetExtension(
      proto::distributed_gradient_boosted_trees_config);

  if (worker_logs_) {
    LOG(INFO) << "Initialize worker memory";
  }

  // Allocate the memory for the gradient and hessian (if needed). Create a
  // "label_accessor" for each gradient+hessian buffer.
  weak_models_.resize(num_weak_models);
  for (int weak_model_idx = 0; weak_model_idx < weak_models_.size();
       weak_model_idx++) {
    auto& weak_model = weak_models_[weak_model_idx];

    weak_model.gradients.resize(dataset_->num_examples());
    if (welcome_.train_config().task() != model::proto::Task::REGRESSION) {
      weak_model.hessians.resize(dataset_->num_examples());

      if (spe_config.gbt().use_hessian_gain()) {
        return absl::InternalError("Use hessian gain not implemented.");
      } else {
        weak_model.label_accessor_type =
            distributed_decision_tree::LabelAccessorType::kNumericalWithHessian;
        weak_model.label_accessor = absl::make_unique<
            distributed_decision_tree::RegressionWithHessianLabelAccessor>(
            weak_model.gradients, weak_model.hessians, dataset_->weights());
      }
    } else {
      if (spe_config.gbt().use_hessian_gain()) {
        return absl::InternalError("Hessian gain not supported for regression");
      }
      weak_model.label_accessor_type =
          distributed_decision_tree::LabelAccessorType::kAutomatic;
      weak_model.hessians.clear();
      weak_model.label_accessor =
          absl::make_unique<distributed_decision_tree::RegressionLabelAccessor>(
              weak_model.gradients, dataset_->weights());
    }
  }

  // The "gradient_ref" is used by some methods to access the gradient+hessian
  // buffers.
  gradient_ref_.resize(weak_models_.size());
  for (int i = 0; i < weak_models_.size(); i++) {
    gradient_ref_[i] = {&weak_models_[i].gradients, &weak_models_[i].hessians};
  }

  return absl::OkStatus();
}

absl::Status DistributedGradientBoostedTreesWorker::SetInitialPredictions(
    const proto::WorkerRequest::SetInitialPredictions& request,
    proto::WorkerResult::SetInitialPredictions* const answer) {
  ASSIGN_OR_RETURN(const auto initial_predictions,
                   loss_->InitialPredictions(request.label_statistics()));

  if (worker_logs_) {
    LOG(INFO) << "Initialize initial predictions";
  }

  gradient_boosted_trees::internal::SetInitialPredictions(
      initial_predictions, dataset_->num_examples(), &predictions_);
  received_initial_predictions_ = true;

  return InitializerWorkingMemory(initial_predictions.size());
}

absl::Status DistributedGradientBoostedTreesWorker::StartNewIter(
    const proto::WorkerRequest::StartNewIter& request,
    proto::WorkerResult::StartNewIter* const answer) {
  if (request.iter_idx() != 0 && request.iter_idx() != iter_idx_ + 1) {
    return absl::InternalError(
        absl::Substitute("StartNewIter with unexpected iter_idx. Got "
                         "iter_idx=$0 while expecting $1",
                         request.iter_idx(), iter_idx_ + 1));
  }

  iter_idx_ = request.iter_idx();
  iter_uid_ = request.iter_uid();
  seed_ = request.seed();
  random_.seed(seed_);

  // Computes the initial gradient.
  if (welcome_.train_config().task() == model::proto::Task::CLASSIFICATION) {
    RETURN_IF_ERROR(loss_->UpdateGradients(
        dataset_->categorical_labels(), predictions_, nullptr, &gradient_ref_,
        &random_, thread_pool_.get()));
  } else {
    RETURN_IF_ERROR(loss_->UpdateGradients(
        dataset_->regression_labels(), predictions_, nullptr, &gradient_ref_,
        &random_, thread_pool_.get()));
  }

  // The weak learners are predicting the loss's gradient.
  auto weak_learner_train_config = welcome_.train_config();
  weak_learner_train_config.set_task(model::proto::Task::REGRESSION);

  ASSIGN_OR_RETURN(const auto set_leaf_functor,
                   loss_->SetLeafFunctorFromLabelStatistics());

  const auto& spe_config = welcome_.train_config().GetExtension(
      proto::distributed_gradient_boosted_trees_config);

  for (int weak_model_idx = 0; weak_model_idx < weak_models_.size();
       weak_model_idx++) {
    auto& weak_model = weak_models_[weak_model_idx];

    ASSIGN_OR_RETURN(
        weak_model.tree_builder,
        distributed_decision_tree::TreeBuilder::Create(
            weak_learner_train_config, welcome_.train_config_linking(),
            spe_config.gbt().decision_tree(), weak_model.label_accessor_type,
            set_leaf_functor));

    // Initialize the statistics of the root node (the only node in the tree).
    weak_model.label_stats_per_node.assign(1, {});
    RETURN_IF_ERROR(weak_model.tree_builder->AggregateLabelStatistics(
        *weak_model.label_accessor.get(),
        &weak_model.label_stats_per_node.front(), thread_pool_.get()));

    weak_model.has_multiple_node_idxs = false;

    // Label statistics for the manager.
    *answer->add_label_statistics() = weak_model.label_stats_per_node.front();

    RETURN_IF_ERROR(weak_model.tree_builder->SetRootValue(
        weak_model.label_stats_per_node.front()));

    weak_model.example_to_node =
        distributed_decision_tree::CreateExampleToNodeMap(
            dataset_->num_examples());
  }

  return absl::OkStatus();
}

absl::Status DistributedGradientBoostedTreesWorker::FindSplits(
    const proto::WorkerRequest::FindSplits& request,
    proto::WorkerResult::FindSplits* const answer) {
  if (request.features_per_weak_models().size() != weak_models_.size()) {
    return absl::InternalError("Unexpected number of weak models");
  }

  answer->mutable_split_per_weak_model()->Reserve(weak_models_.size());

  // Collect the active features i.e. the features on which to find splits.
  size_t num_unique_active_features_across_weak_models = 0;
  // Model idx -> node idx -> list of features.
  std::vector<std::vector<std::vector<int>>> active_features_per_weak_models(
      weak_models_.size());
  // Model idx -> list of features.
  std::vector<std::vector<int>> unique_active_features_per_weak_models(
      weak_models_.size());

  for (int weak_model_idx = 0; weak_model_idx < weak_models_.size();
       weak_model_idx++) {
    auto& weak_model = weak_models_[weak_model_idx];

    if (weak_model.label_stats_per_node.size() !=
        request.features_per_weak_models(weak_model_idx)
            .features_per_node_size()) {
      return absl::InternalError("Unexpected number of open nodes");
    }

    // List the features to test.
    auto& active_features = active_features_per_weak_models[weak_model_idx];
    ASSIGN_OR_RETURN(active_features,
                     ExtractInputFeaturesPerNodes(
                         request.features_per_weak_models(weak_model_idx)));
    if (active_features.size() != weak_model.label_stats_per_node.size()) {
      return absl::InternalError("Unexpected number of requested nodes");
    }

    // List all the features tested by at least one node.
    auto& unique_active_features =
        unique_active_features_per_weak_models[weak_model_idx];
    absl::flat_hash_set<int> unique_active_features_set;
    for (const auto& features : active_features) {
      unique_active_features_set.insert(features.begin(), features.end());
    }
    unique_active_features.insert(unique_active_features.end(),
                                  unique_active_features_set.begin(),
                                  unique_active_features_set.end());
    num_unique_active_features_across_weak_models +=
        unique_active_features.size();
  }

  std::vector<distributed_decision_tree::SplitPerOpenNode>
      splits_per_weak_models(weak_models_.size());

  utils::concurrency::Mutex mutex_splits_per_weak_models;
  utils::concurrency::BlockingCounter done_find_splits(
      num_unique_active_features_across_weak_models);
  absl::Status worker_status;

  // Find the best splits.
  for (int weak_model_idx = 0; weak_model_idx < weak_models_.size();
       weak_model_idx++) {
    auto& weak_model = weak_models_[weak_model_idx];

    // Look for the splits.
    RETURN_IF_ERROR(weak_model.tree_builder->FindBestSplitsWithThreadPool(
        {active_features_per_weak_models[weak_model_idx],
         weak_model.example_to_node, welcome_.dataspec(),
         *weak_model.label_accessor.get(), weak_model.label_stats_per_node,
         weak_model.has_multiple_node_idxs, dataset_.get(),
         &splits_per_weak_models[weak_model_idx]},
        unique_active_features_per_weak_models[weak_model_idx],
        thread_pool_.get(), &mutex_splits_per_weak_models, &done_find_splits,
        &worker_status));
  }

  done_find_splits.Wait();
  utils::concurrency::MutexLock l(&mutex_splits_per_weak_models);
  RETURN_IF_ERROR(worker_status);

  // Save the best splits into the reply.
  for (int weak_model_idx = 0; weak_model_idx < weak_models_.size();
       weak_model_idx++) {
    const auto& splits = splits_per_weak_models[weak_model_idx];
    distributed_decision_tree::ConvertToProto(
        splits, answer->add_split_per_weak_model());
  }
  return absl::OkStatus();
}

absl::Status DistributedGradientBoostedTreesWorker::EvaluateSplits(
    const proto::WorkerRequest::EvaluateSplits& request,
    proto::WorkerResult::EvaluateSplits* answer) {
  if (request.split_per_weak_model().size() != weak_models_.size()) {
    return absl::InternalError("Unexpected number of weak models");
  }

  for (int weak_model_idx = 0; weak_model_idx < weak_models_.size();
       weak_model_idx++) {
    auto& weak_model = weak_models_[weak_model_idx];

    // Clear the last split evaluation.
    weak_model.last_split_evaluation.clear();

    distributed_decision_tree::ConvertFromProto(
        request.split_per_weak_model(weak_model_idx), &weak_model.last_splits);

    RETURN_IF_ERROR(distributed_decision_tree::EvaluateSplits(
        weak_model.example_to_node, weak_model.last_splits,
        &weak_model.last_split_evaluation, dataset_.get(), thread_pool_.get()));
  }

  return absl::OkStatus();
}

absl::Status DistributedGradientBoostedTreesWorker::UpdateOwnedFeatures(
    std::vector<int> target_features) {
  const auto& initial_features = dataset_->features();
  std::sort(target_features.begin(), target_features.end());

  // New features to load i.e. target_features / initial_features
  std::vector<int> features_to_load;
  std::set_difference(target_features.begin(), target_features.end(),
                      initial_features.begin(), initial_features.end(),
                      std::back_inserter(features_to_load));

  // Features to unload i.e. initial_features / target_features.
  std::vector<int> features_to_unload;
  std::set_difference(initial_features.begin(), initial_features.end(),
                      target_features.begin(), target_features.end(),
                      std::back_inserter(features_to_unload));

  if (features_to_load.empty() && features_to_unload.empty()) {
    return absl::OkStatus();
  }

  if (dataset_->IsNonBlockingLoadingInProgress()) {
    // TODO(gbm): Just wait?
    return absl::InternalError(absl::StrCat(
        "Unexpected change of loaded features while a non-blocking loading is "
        "in progress on worker #",
        WorkerIdx()));
  }
  if (worker_logs_) {
    if (!features_to_load.empty()) {
      LOG(INFO)
          << "Blocking loading of " << features_to_load.size()
          << " features. This is less efficient that non-blocking feature "
             "loading and should open append when the manager or the "
             "worker get rescheduled.";
    }
  }

  return dataset_->LoadingAndUnloadingFeatures(features_to_load,
                                               features_to_unload);
}

utils::StatusOr<bool>
DistributedGradientBoostedTreesWorker::PreloadFutureOwnedFeatures(
    const proto::WorkerRequest::FutureOwnedFeatures& future_owned_features) {
  const std::vector<int> load_features = {
      future_owned_features.load_features().begin(),
      future_owned_features.load_features().end(),
  };
  // We ignore the unloading instructions.
  std::vector<int> unload_features;

  // Is the request similar at the already running process?
  const bool requested_equals_running =
      (dataset_->NonBlockingLoadingInProgressLoadedFeatures() ==
       load_features) &&
      (dataset_->NonBlockingLoadingInProgressUnloadedFeatures() ==
       unload_features);

  if (dataset_->IsNonBlockingLoadingInProgress()) {
    // Pre-loading is already running.

    // Check and update running status.
    ASSIGN_OR_RETURN(const auto preloading_running,
                     dataset_->CheckAndUpdateNonBlockingLoading());

    if (preloading_running) {
      // Still running.
      if (!requested_equals_running) {
        LOG(INFO) << "Requested future owned features are different from the "
                     "ones currently being loaded";
      }
      return true;
    } else {
      // Just done running.

      LOG(INFO) << "Feature pre-loading done on worker " << WorkerIdx();
      if (!requested_equals_running) {
        // Quickly start the pre-loading of the request (because it was
        // different from the execution).
        LOG(INFO) << "Immediate restart of non-blocking loading ("
                  << load_features.size() << ") and unloading ("
                  << unload_features.size()
                  << ") of features for future work on worker " << WorkerIdx();

        RETURN_IF_ERROR(dataset_->NonBlockingLoadingAndUnloadingFeatures(
            load_features, unload_features, /*num_threads=*/5));
        return true;
      } else {
        return false;
      }
    }

    return true;
  } else {
    if (!requested_equals_running) {
      LOG(INFO) << "Non-blocking loading (" << load_features.size()
                << ") and unloading (" << unload_features.size()
                << ") of features for future work on worker " << WorkerIdx();

      RETURN_IF_ERROR(dataset_->NonBlockingLoadingAndUnloadingFeatures(
          load_features, unload_features));
      return true;
    } else {
      return false;
    }
  }
}


absl::Status DistributedGradientBoostedTreesWorker::
    MergingSplitEvaluationToLastSplitEvaluation(
        proto::WorkerResult::GetSplitValue* src_split_values) {
  if (weak_models_.size() !=
      src_split_values->evaluation_per_weak_model_size()) {
    return absl::InternalError("Unexpected number of weak models");
  }
  for (int weak_model_idx = 0; weak_model_idx < weak_models_.size();
       weak_model_idx++) {
    auto* src_evaluations =
        src_split_values->mutable_evaluation_per_weak_model(weak_model_idx)
            ->mutable_evaluation_per_open_node();
    if (src_evaluations->size() !=
        weak_models_[weak_model_idx].last_split_evaluation.size()) {
      return absl::InternalError(absl::Substitute(
          "Wrong number of splits in MergingSplitEvaluation. $0 != $1",
          src_evaluations->size(),
          weak_models_[weak_model_idx].last_split_evaluation.size()));
    }
    for (int split_idx = 0; split_idx < src_evaluations->size(); split_idx++) {
      if (src_evaluations->Get(split_idx).empty()) {
        continue;
      }
      weak_models_[weak_model_idx].last_split_evaluation[split_idx] =
          std::move(*src_evaluations->Mutable(split_idx));
    }
  }

  return absl::OkStatus();
}

absl::Status DistributedGradientBoostedTreesWorker::ShareSplits(
    const proto::WorkerRequest::ShareSplits& request,
    proto::WorkerResult::ShareSplits* answer,
    proto::WorkerResult* generic_answer) {

  // Request the split evaluation from other workers.
  for (const auto& items : request.request().items()) {
    proto::WorkerRequest generic_other_request;
    *generic_other_request.mutable_get_split_value()->mutable_splits() =
        items.splits();
    RETURN_IF_ERROR(AsynchronousProtoRequestToOtherWorker(generic_other_request,
                                                          items.src_worker()));
  }
  const int num_requests = request.request().items_size();


  // Aggregate all the split evaluations.
  for (int reply_idx = 0; reply_idx < num_requests; reply_idx++) {
    auto reply_status =
        NextAsynchronousProtoAnswerFromOtherWorker<proto::WorkerResult>();
    if (!reply_status.ok()) {
      LOG(WARNING) << "Other replied with error: "
                   << reply_status.status().message()
                   << ". Answering the manager with missing data error";
      RETURN_IF_ERROR(
          SkipAsyncWorkerToWorkerAnswers(num_requests - reply_idx - 1));
      generic_answer->set_request_restart_iter(true);
      return absl::OkStatus();
    }
    auto generic_other_result = std::move(reply_status).value();

    if (generic_other_result.request_restart_iter()) {
      // The target worker does not have the required data.
      LOG(WARNING) << "Other worker responded to GetSplitValue request with "
                      "missing data error";
      RETURN_IF_ERROR(
          SkipAsyncWorkerToWorkerAnswers(num_requests - reply_idx - 1));
      generic_answer->set_request_restart_iter(true);
      return absl::OkStatus();
    }
    if (!generic_other_result.has_get_split_value()) {
      return absl::InternalError("Unexpected answer. Expecting GetSplitValue.");
    }
    auto* other_result = generic_other_result.mutable_get_split_value();
    RETURN_IF_ERROR(MergingSplitEvaluationToLastSplitEvaluation(other_result));
  }

  if (request.request().last_request_of_plan()) {
    for (int weak_model_idx = 0; weak_model_idx < weak_models_.size();
         weak_model_idx++) {
      auto& weak_model = weak_models_[weak_model_idx];
      // const auto& iter = layer_per_weak_models[weak_model_idx];

      ASSIGN_OR_RETURN(
          const auto node_remapping,
          weak_model.tree_builder->ApplySplitToTree(weak_model.last_splits));

      weak_model.has_multiple_node_idxs = true;

      RETURN_IF_ERROR(UpdateClosingNodesPredictions(
          weak_model.example_to_node, weak_model.label_stats_per_node,
          node_remapping, weak_model.tree_builder->set_leaf_functor(),
          weak_model_idx, weak_models_.size(), &predictions_,
          thread_pool_.get()));

      RETURN_IF_ERROR(UpdateExampleNodeMap(
          weak_model.last_splits, weak_model.last_split_evaluation,
          node_remapping, &weak_model.example_to_node, thread_pool_.get()));

      RETURN_IF_ERROR(UpdateLabelStatistics(weak_model.last_splits,
                                            node_remapping,
                                            &weak_model.label_stats_per_node));
    }
  }
  return absl::OkStatus();
}

absl::Status
DistributedGradientBoostedTreesWorker::SkipAsyncWorkerToWorkerAnswers(
    int num_skip) {
  for (int i = 0; i < num_skip; i++) {
    RETURN_IF_ERROR(NextAsynchronousAnswerFromOtherWorker().status());
  }
  return absl::OkStatus();
}

absl::Status DistributedGradientBoostedTreesWorker::GetSplitValue(
    const proto::WorkerRequest::GetSplitValue& request,
    proto::WorkerResult::GetSplitValue* answer) {
  answer->set_source_worker(WorkerIdx());

  // Allocate the answer.
  answer->mutable_evaluation_per_weak_model()->Reserve(weak_models_.size());
  for (int weak_model_idx = 0; weak_model_idx < weak_models_.size();
       weak_model_idx++) {
    const auto& weak_model = weak_models_[weak_model_idx];
    auto* dst_evaluation_per_open_node =
        answer->add_evaluation_per_weak_model()
            ->mutable_evaluation_per_open_node();
    dst_evaluation_per_open_node->Reserve(
        weak_model.last_split_evaluation.size());
    for (int split_idx = 0; split_idx < weak_model.last_split_evaluation.size();
         split_idx++) {
      dst_evaluation_per_open_node->Add();
    }
  }

  // Copy the split evaluations.
  for (const auto& split : request.splits()) {
    const auto& src = weak_models_[split.weak_model_idx()]
                          .last_split_evaluation[split.split_idx()];
    if (src.empty()) {
      return absl::InternalError("Split data not available");
    }
    auto& dst =
        *answer->mutable_evaluation_per_weak_model(split.weak_model_idx())
             ->mutable_evaluation_per_open_node(split.split_idx());
    dst = src;
  }

  return absl::OkStatus();
}

absl::Status DistributedGradientBoostedTreesWorker::EndIter(
    const proto::WorkerRequest::EndIter& request,
    proto::WorkerResult::EndIter* answer) {
  for (int weak_model_idx = 0; weak_model_idx < weak_models_.size();
       weak_model_idx++) {
    const auto& weak_model = weak_models_[weak_model_idx];

    // Closing all the remaining open nodes.
    RETURN_IF_ERROR(UpdateClosingNodesPredictions(
        weak_model.example_to_node, weak_model.label_stats_per_node,
        weak_model.tree_builder->CreateClosingNodeRemapping(),
        weak_model.tree_builder->set_leaf_functor(), weak_model_idx,
        weak_models_.size(), &predictions_, thread_pool_.get()));
  }

  if (request.compute_training_loss()) {
    float loss;
    std::vector<float> secondary_metric;
    RETURN_IF_ERROR(
        Loss(dataset_.get(), predictions_, &loss, &secondary_metric));
    answer->set_training_loss(loss);
    *answer->mutable_training_metrics() = {secondary_metric.begin(),
                                           secondary_metric.end()};
  }

  return absl::OkStatus();
}

absl::Status DistributedGradientBoostedTreesWorker::RestoreCheckpoint(
    const proto::WorkerRequest::RestoreCheckpoint& request,
    proto::WorkerResult::RestoreCheckpoint* answer) {
  LOG(INFO) << "Restore checkpoint to iter " << request.iter_idx() << " (was "
            << iter_idx_ << " before)";
  iter_idx_ = request.iter_idx();
  const auto path =
      file::JoinPath(welcome_.work_directory(), kFileNameCheckPoint,
                     absl::StrCat(iter_idx_), "predictions");
  predictions_.clear();
  // All the predictions are stored in a single shard.
  RETURN_IF_ERROR(
      distributed_decision_tree::dataset_cache::ShardedFloatColumnReader::
          ReadAndAppend(path, 0, request.num_shards(), &predictions_));
  received_initial_predictions_ = true;
  RETURN_IF_ERROR(InitializerWorkingMemory(request.num_weak_models()));

  // The worker will receive a "StartNewIter" message with "iter_idx=iter_idx_"
  // (before --) next.
  iter_idx_--;

  return absl::OkStatus();
}

absl::Status DistributedGradientBoostedTreesWorker::CreateCheckpoint(
    const proto::WorkerRequest::CreateCheckpoint& request,
    proto::WorkerResult::CreateCheckpoint* answer) {
  const auto path = file::JoinPath(welcome_.work_directory(), kFileNameTmp,
                                   utils::GenUniqueId());
  answer->set_shard_idx(request.shard_idx());
  answer->set_path(path);

  distributed_decision_tree::dataset_cache::FloatColumnWriter writer;
  RETURN_IF_ERROR(writer.Open(path));
  size_t begin_idx = request.begin_example_idx() * weak_models_.size();
  size_t end_idx = request.end_example_idx() * weak_models_.size();
  RETURN_IF_ERROR(writer.WriteValues(
      absl::MakeSpan(predictions_).subspan(begin_idx, end_idx - begin_idx)));
  RETURN_IF_ERROR(writer.Close());
  return absl::OkStatus();
}

absl::Status DistributedGradientBoostedTreesWorker::StartTraining(
    const proto::WorkerRequest::StartTraining& request,
    proto::WorkerResult::StartTraining* answer) {
  answer->set_num_loaded_features(dataset_num_features_loaded_);
  answer->set_feature_loading_time_seconds(
      absl::ToDoubleSeconds(dataset_feature_duration_));
  return absl::OkStatus();
}

absl::Status DistributedGradientBoostedTreesWorker::Loss(
    distributed_decision_tree::dataset_cache::DatasetCacheReader* dataset,
    const std::vector<float>& predictions, float* loss_value,
    std::vector<float>* secondary_metric) {
  switch (welcome_.train_config().task()) {
    case model::proto::Task::CLASSIFICATION:
      RETURN_IF_ERROR(loss_->Loss(dataset->categorical_labels(), predictions,
                                  dataset->weights(), nullptr, loss_value,
                                  secondary_metric, thread_pool_.get()));
      break;
    case model::proto::Task::REGRESSION:
      RETURN_IF_ERROR(loss_->Loss(dataset->regression_labels(), predictions,
                                  dataset->weights(), nullptr, loss_value,
                                  secondary_metric, thread_pool_.get()));
      break;
    default:
      return absl::InvalidArgumentError("Not supported task");
  }
  return absl::OkStatus();
}

absl::Status UpdateClosingNodesPredictions(
    const distributed_decision_tree::ExampleToNodeMap& example_to_node,
    const distributed_decision_tree::LabelStatsPerNode& label_stats_per_node,
    const distributed_decision_tree::NodeRemapping& node_remapping,
    const distributed_decision_tree::SetLeafValueFromLabelStatsFunctor&
        set_leaf_functor,
    const int weak_model_idx, const int num_weak_models,
    std::vector<float>* predictions,
    utils::concurrency::ThreadPool* thread_pool) {
  // Collect the increase of prediction value for each closing node.
  std::vector<float> prediction_offset_per_node(
      label_stats_per_node.size(), std::numeric_limits<float>::quiet_NaN());
  for (int node_idx = 0; node_idx < label_stats_per_node.size(); node_idx++) {
    decision_tree::proto::Node fake_node;
    RETURN_IF_ERROR(
        set_leaf_functor(label_stats_per_node[node_idx], &fake_node));
    if (!fake_node.has_regressor() || !fake_node.regressor().has_top_value()) {
      return absl::InternalError(
          "The set leaf functor did not create a regressive node");
    }
    prediction_offset_per_node[node_idx] = fake_node.regressor().top_value();
  }

  // Increase the prediction values.
  decision_tree::ConcurrentForLoop(
      thread_pool->num_threads(), thread_pool, example_to_node.size(),
      [&example_to_node, &label_stats_per_node, &node_remapping, predictions,
       num_weak_models, weak_model_idx, &prediction_offset_per_node](
          size_t block_idx, size_t begin_idx, size_t end_idx) -> void {
        for (size_t example_idx = begin_idx; example_idx < end_idx;
             example_idx++) {
          auto& node_idx = example_to_node[example_idx];
          if (node_idx == distributed_decision_tree::kClosedNode) {
            // The example is not is a closed node.
            continue;
          }
          DCHECK_GE(node_idx, 0);
          DCHECK_LT(node_idx, label_stats_per_node.size());

          if (node_remapping[node_idx].indices[0] !=
              distributed_decision_tree::kClosedNode) {
            // This example remains in an open node.
            continue;
          }

          // The example is in a node that is closed during this iteration.
          (*predictions)[example_idx * num_weak_models + weak_model_idx] +=
              prediction_offset_per_node[node_idx];
        }
      });

  return absl::OkStatus();
}

utils::StatusOr<std::vector<std::vector<int>>> ExtractInputFeaturesPerNodes(
    const proto::WorkerRequest::FindSplits::FeaturePerNode& src) {
  const auto& request_features_per_node = src.features_per_node();
  std::vector<std::vector<int>> dst;
  dst.reserve(request_features_per_node.size());
  for (const auto& features : request_features_per_node) {
    dst.push_back({features.features().begin(), features.features().end()});
  }
  return dst;
}

}  // namespace distributed_gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
