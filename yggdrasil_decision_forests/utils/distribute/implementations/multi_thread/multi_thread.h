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

// In process implementation. For debugging and pipeline development.
//
// For efficient multi-threading, use a "ThreadPool" or a "StreamProcessor".
//

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_MULTI_THREAD_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_MULTI_THREAD_H_

#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/utils.h"

namespace yggdrasil_decision_forests {
namespace distribute {

class MultiThreadManager : public AbstractManager,
                           protected AbstractWorkerHook {
 public:
  static constexpr char kKey[] = "MULTI_THREAD";

  virtual ~MultiThreadManager() {
    if (!done_was_called_) {
      LOG(WARNING) << "Calling destructor on distribute manager before having "
                      "called \"Done\".";
      CHECK_OK(Done({}));
    }
  }

  utils::StatusOr<Blob> BlockingRequest(Blob blob, int worker_idx) override;

  absl::Status AsynchronousRequest(Blob blob, int worker_idx) override;

  utils::StatusOr<Blob> NextAsynchronousAnswer() override;

  int NumWorkers() override;

  absl::Status Done(absl::optional<bool> kill_worker_manager) override;

  utils::StatusOr<int> NumWorkersInConfiguration(
      const proto::Config& config) const override;

  absl::Status SetParallelExecutionPerWorker(int num) override;

 protected:
  virtual absl::Status AsynchronousRequestToOtherWorker(
      Blob blob, int target_worker_idx, AbstractWorker* emitter_worker) {
    workers_[target_worker_idx]->pending_inter_workers_queries.Push(
        std::make_pair(emitter_worker->WorkerIdx(), std::move(blob)));
    return absl::OkStatus();
  }

  virtual utils::StatusOr<Blob> NextAsynchronousAnswerFromOtherWorker(
      AbstractWorker* emitter_worker) {
    auto answer = workers_[emitter_worker->WorkerIdx()]
                      ->pending_inter_workers_answers.Pop();
    if (!answer.has_value()) {
      return absl::OutOfRangeError("No more results available");
    }
    return std::move(answer.value());
  }

 private:
  struct Worker {
    void StartThreads(int num_threads, MultiThreadManager* manager) {
      process_global_queries.Start(num_threads, [manager, this]() {
        manager->ProcessGlobalQueries(this);
      });
      process_local_queries.Start(num_threads, [manager, this]() {
        manager->ProcessLocalQueries(this);
      });
      process_inter_workers_local_queries.Start(num_threads, [manager, this]() {
        manager->ProcessInterWorkersLocalQueries(this);
      });
    }

    std::unique_ptr<AbstractWorker> worker_imp;
    utils::concurrency::Channel<Blob> pending_queries;
    // Requesting worker index and payload.
    utils::concurrency::Channel<std::pair<int, Blob>>
        pending_inter_workers_queries;
    utils::concurrency::Channel<utils::StatusOr<Blob>>
        pending_inter_workers_answers;

    ThreadVector process_global_queries;
    ThreadVector process_local_queries;
    ThreadVector process_inter_workers_local_queries;
  };

  // Thread loop to process the, worker-specific and inter-worker queries.
  void ProcessGlobalQueries(Worker* worker);
  void ProcessLocalQueries(Worker* worker);
  void ProcessInterWorkersLocalQueries(Worker* worker);

  absl::Status Initialize(const proto::Config& config,
                          absl::string_view worker_name, Blob welcome_blob,
                          int parallel_execution_per_worker) override;

  int verbosity_;
  std::vector<std::unique_ptr<Worker>> workers_;

  // Next worker that will solve the next request.
  std::atomic<int> next_worker_ = {0};

  utils::concurrency::Channel<Blob> pending_queries_;
  utils::concurrency::Channel<utils::StatusOr<Blob>> pending_answers_;

  std::atomic<bool> done_was_called_{false};
};

REGISTER_Distribution_Manager(MultiThreadManager, MultiThreadManager::kKey);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_MULTI_THREAD_H_
