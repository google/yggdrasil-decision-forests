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

#include "yggdrasil_decision_forests/utils/distribute/implementations/single_thread/single_thread.h"

#include "yggdrasil_decision_forests/utils/concurrency_channel.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/single_thread/single_thread.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace distribute {

utils::StatusOr<Blob> SingleThreadManager::BlockingRequest(Blob blob,
                                                           int worker_idx) {
  if (verbose_) {
    LOG(INFO) << "Incoming blocking request with " << blob.size() << " bytes";
  }

  if (worker_idx < 0) {
    worker_idx = next_worker_.fetch_add(1) % workers_.size();
  }

  ASSIGN_OR_RETURN(Blob answer, workers_[worker_idx]->RunRequest(blob),
                   _ << "Error emitted by worker #" << worker_idx);

  if (verbose_) {
    LOG(INFO) << "Completed blocking request with " << answer.size()
              << " bytes";
  }
  return answer;
}

absl::Status SingleThreadManager::AsynchronousRequest(Blob blob,
                                                      int worker_idx) {
  if (verbose_) {
    LOG(INFO) << "Incoming asynchronous request with " << blob.size()
              << " bytes";
  }
  if (worker_idx < 0) {
    worker_idx = next_worker_.fetch_add(1) % workers_.size();
  }
  ASSIGN_OR_RETURN(Blob answer, workers_[worker_idx]->RunRequest(blob),
                   _ << "Error emitted by worker #" << worker_idx);
  async_pending_answers_.Push(std::move(answer));
  return absl::OkStatus();
}

utils::StatusOr<Blob> SingleThreadManager::NextAsynchronousAnswer() {
  if (verbose_) {
    LOG(INFO) << "Wait for next result";
  }
  auto answer = async_pending_answers_.Pop();
  if (!answer.has_value()) {
    return absl::InternalError("Closed channel");
  }
  if (verbose_) {
    LOG(INFO) << "Return asynchronous result with " << answer.value().size()
              << " bytes";
  }
  return std::move(answer.value());
}

int SingleThreadManager::NumWorkers() { return workers_.size(); }

absl::Status SingleThreadManager::Done(
    absl::optional<bool> kill_worker_manager) {
  if (verbose_) {
    LOG(INFO) << "Release workers";
  }
  for (auto& worker : workers_) {
    RETURN_IF_ERROR(worker->Done());
  }
  return absl::OkStatus();
}

absl::Status SingleThreadManager::Initialize(
    const proto::Config& config, const absl::string_view worker_name,
    Blob welcome_blob) {
  const auto& imp_config = config.GetExtension(proto::single_thread);
  const auto num_workers = imp_config.num_workers();

  verbose_ = config.verbose();
  if (verbose_) {
    LOG(INFO) << "Initialize manager with " << welcome_blob.size()
              << " bytes welcome blob and " << num_workers << " workers";
  }
  if (num_workers <= 0) {
    return absl::InvalidArgumentError(
        "The number of workers should greater than zero");
  }
  for (int worker_idx = 0; worker_idx < num_workers; worker_idx++) {
    ASSIGN_OR_RETURN(auto worker,
                     AbstractWorkerRegisterer::Create(worker_name));
    RETURN_IF_ERROR(InternalInitializeWorker(worker_idx, worker.get()));
    RETURN_IF_ERROR(worker->Setup(welcome_blob));
    workers_.push_back(std::move(worker));
  }
  return absl::OkStatus();
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
