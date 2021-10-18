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

#include "yggdrasil_decision_forests/utils/distribute/implementations/multi_thread/multi_thread.h"

#include "yggdrasil_decision_forests/utils/concurrency_channel.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/multi_thread/multi_thread.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace distribute {

constexpr char MultiThreadManager::kKey[];

utils::StatusOr<Blob> MultiThreadManager::BlockingRequest(Blob blob,
                                                          int worker_idx) {
  if (verbosity_ >= 2) {
    LOG(INFO) << "Emitting blocking request of " << blob.size() << " bytes";
  }

  if (worker_idx < 0) {
    worker_idx = next_worker_.fetch_add(1) % workers_.size();
  }

  ASSIGN_OR_RETURN(Blob answer,
                   workers_[worker_idx]->worker_imp->RunRequest(blob),
                   _ << "Error emitted by worker #" << worker_idx);

  if (verbosity_ >= 2) {
    LOG(INFO) << "Completed blocking request with " << answer.size()
              << " bytes";
  }
  return answer;
}

absl::Status MultiThreadManager::AsynchronousRequest(Blob blob,
                                                     int worker_idx) {
  if (verbosity_ >= 2) {
    LOG(INFO) << "Emitting asynchronous request of " << blob.size() << " bytes";
  }
  if (worker_idx < 0) {
    pending_queries_.Push(std::move(blob));
  } else {
    workers_[worker_idx]->pending_queries.Push(std::move(blob));
  }
  return absl::OkStatus();
}

utils::StatusOr<Blob> MultiThreadManager::NextAsynchronousAnswer() {
  if (verbosity_ >= 2) {
    LOG(INFO) << "Wait for next result";
  }
  auto answer = pending_answers_.Pop();
  if (!answer.has_value()) {
    return absl::OutOfRangeError("No more results available");
  }
  if (verbosity_ >= 1 && !answer.value().ok()) {
    LOG(INFO) << "Return asynchronous result failure: "
              << answer.value().status();
  }
  if (verbosity_ >= 2 && answer.value().ok()) {
    LOG(INFO) << "Return asynchronous result with "
              << answer.value().value().size() << " bytes";
  }
  return std::move(answer.value());
}

void MultiThreadManager::ProcessGlobalQueries(Worker* worker) {
  while (true) {
    auto pending_blob_or = pending_queries_.Pop();
    if (!pending_blob_or.has_value()) {
      break;
    }
    auto answer =
        worker->worker_imp->RunRequest(std::move(pending_blob_or.value()));
    pending_answers_.Push(std::move(answer));
  }
}

void MultiThreadManager::ProcessLocalQueries(Worker* worker) {
  while (true) {
    auto pending_blob_or = worker->pending_queries.Pop();
    if (!pending_blob_or.has_value()) {
      break;
    }
    auto answer =
        worker->worker_imp->RunRequest(std::move(pending_blob_or.value()));
    pending_answers_.Push(std::move(answer));
  }
}

void MultiThreadManager::ProcessInterWorkersLocalQueries(Worker* worker) {
  while (true) {
    auto pending_blob_or = worker->pending_inter_workers_queries.Pop();
    if (!pending_blob_or.has_value()) {
      break;
    }
    auto answer = worker->worker_imp->RunRequest(
        std::move(pending_blob_or.value().second));
    workers_[pending_blob_or.value().first]->pending_inter_workers_answers.Push(
        std::move(answer));
  }
}

int MultiThreadManager::NumWorkers() { return workers_.size(); }

absl::Status MultiThreadManager::Done(
    absl::optional<bool> kill_worker_manager) {
  if (verbosity_ >= 1) {
    LOG(INFO) << "Release workers";
  }
  if (done_was_called_) {
    LOG(WARNING) << "Calling done twice";
    return absl::OkStatus();
  }
  done_was_called_ = true;
  pending_queries_.Close();
  pending_answers_.Close();

  for (auto& worker : workers_) {
    worker->pending_queries.Close();
    worker->pending_inter_workers_queries.Close();
    worker->pending_inter_workers_answers.Close();
    RETURN_IF_ERROR(worker->worker_imp->Done());
    worker->process_global_queries.JoinAndClear();
    worker->process_local_queries.JoinAndClear();
    worker->process_inter_workers_local_queries.JoinAndClear();
  }
  return absl::OkStatus();
}

absl::Status MultiThreadManager::SetParallelExecutionPerWorker(int num) {
  if (verbosity_) {
    LOG(INFO) << "Change the number of parallel execution per worker";
  }

  // Close the query channels.
  pending_queries_.Close();
  for (auto& worker : workers_) {
    worker->pending_queries.Close();
    worker->pending_inter_workers_queries.Close();
  }

  // Wait for the threads to join
  for (auto& worker : workers_) {
    worker->process_global_queries.JoinAndClear();
    worker->process_local_queries.JoinAndClear();
    worker->process_inter_workers_local_queries.JoinAndClear();
  }

  // Re-open the channels and restart the threads.
  pending_queries_.Reopen();
  for (auto& worker : workers_) {
    worker->pending_queries.Reopen();
    worker->pending_inter_workers_queries.Reopen();
    worker->StartThreads(num, this);
  }
  return absl::OkStatus();
}

absl::Status MultiThreadManager::Initialize(const proto::Config& config,
                                            const absl::string_view worker_name,
                                            Blob welcome_blob,
                                            int parallel_execution_per_worker) {
  const auto& imp_config = config.GetExtension(proto::multi_thread);
  const auto num_workers = imp_config.num_workers();
  verbosity_ = config.verbosity();
  if (verbosity_ >= 1) {
    LOG(INFO) << "Initialize manager with " << welcome_blob.size()
              << " bytes welcome blob and " << num_workers << " workers";
  }
  if (num_workers <= 0) {
    return absl::InvalidArgumentError(
        "The number of workers should greater than zero");
  }

  workers_.reserve(num_workers);
  for (int worker_idx = 0; worker_idx < num_workers; worker_idx++) {
    auto worker = absl::make_unique<Worker>();
    ASSIGN_OR_RETURN(worker->worker_imp,
                     AbstractWorkerRegisterer::Create(worker_name));
    RETURN_IF_ERROR(InternalInitializeWorker(worker_idx, num_workers,
                                             worker->worker_imp.get(), this));
    RETURN_IF_ERROR(worker->worker_imp->Setup(welcome_blob));
    worker->StartThreads(parallel_execution_per_worker, this);
    workers_.push_back(std::move(worker));
  }

  return absl::OkStatus();
}

utils::StatusOr<int> MultiThreadManager::NumWorkersInConfiguration(
    const proto::Config& config) const {
  const auto& imp_config = config.GetExtension(proto::multi_thread);
  return imp_config.num_workers();
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
