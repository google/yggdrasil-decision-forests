/*
 * Copyright 2022 Google LLC.
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

// Distribution over a set of processed communicating through GRPC.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_GRPC_MANAGER_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_GRPC_MANAGER_H_

#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "absl/container/node_hash_map.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc.grpc.pb.h"
#include "yggdrasil_decision_forests/utils/distribute/utils.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests {
namespace distribute {

class GRPCManager : public AbstractManager {
 public:
  static constexpr char kKey[] = "GRPC";

  virtual ~GRPCManager() {
    if (!done_was_called_) {
      LOG(WARNING) << "Calling destructor on distribute manager before having "
                      "called \"Done\".";
      CHECK_OK(Done({}));
    }
  }

  absl::StatusOr<Blob> BlockingRequest(Blob blob, int worker_idx) override;

  absl::Status AsynchronousRequest(Blob blob, int worker_idx) override;

  absl::StatusOr<Blob> NextAsynchronousAnswer() override;

  int NumWorkers() override;

  absl::Status Done(absl::optional<bool> kill_worker_manager) override;

  absl::StatusOr<int> NumWorkersInConfiguration(
      const proto::Config& config) const override;

  absl::Status SetParallelExecutionPerWorker(int num) override;

  // Changes the address of a worker. The next requests emited by the manager or
  // the other workers to worke "worker_idx" will use this new address.
  absl::Status UpdateWorkerAddress(int worker_idx,
                                   absl::string_view new_address);

  // Shuts down a worker without updating the manager or any of the other
  // worker. This function simulates the interruption of the worker process. It
  // should only be used in unit testing.
  absl::Status DebugShutdownWorker(int worker_idx);

 private:
  struct Worker {
    int worker_idx;

    // Connection to the worker.
    std::unique_ptr<proto::Server::Stub> stub GUARDED_BY(mutex_address);

    // Address currently connected by the stub.
    std::string connected_address GUARDED_BY(mutex_address);

    // Address of the worker. "expected_address" and "connected_address" might
    // be different for a short time when a worker is re-located.
    std::string expected_address GUARDED_BY(mutex_address);

    // Disconnected worker stubs kept until releasing.
    // TODO: Release the discarded worker stubs.
    std::vector<std::unique_ptr<proto::Server::Stub>> discarded_stubs_
        GUARDED_BY(mutex_address);

    utils::concurrency::Mutex mutex_address;

    // Async query to execute specific to this worker.
    utils::concurrency::Channel<Blob> async_pending_queries_;

    ThreadVector process_local_queries;
    ThreadVector process_global_queries;

    void StartThreads(int parallel_execution_per_worker, GRPCManager* manager);
  };

  absl::Status Initialize(const proto::Config& config,
                          absl::string_view worker_name, Blob welcome_blob,
                          int parallel_execution_per_worker) override;

  absl::Status InitializeWorkers(const proto::Config& config,
                                 int parallel_execution_per_worker);

  absl::Status InitializeConfigFile(const proto::Config& config,
                                    absl::string_view worker_name,
                                    int parallel_execution_per_worker,
                                    Blob welcome_blob);

  // Thread loop to process the global and worker-specific queries.
  void ProcessGlobalQueries(Worker* worker);
  void ProcessLocalQueries(Worker* worker);

  // Processes a query and exports the result to the answer queue.
  void WorkerRun(Blob blob, Worker* worker);

  // Processes a query and returns the answer.
  absl::StatusOr<Blob> WorkerRunImp(Blob blob, Worker* worker);

  void JoinWorkers();

  // Checks and possibly update the effectively targeted worker.
  absl::StatusOr<proto::Server::Stub*> UpdateWorkerConnection(Worker* worker);

  // Path to serialized worker configuration accessible by all workers.
  proto::WorkerConfig worker_config_;
  int verbosity_;
  std::vector<std::unique_ptr<Worker>> workers_;

  // Manager UID.
  uint64_t manager_uid_;

  // Async query to execute by any worker.
  utils::concurrency::Channel<Blob> async_pending_queries_;

  // Async answers that should be returned, indexed by task.
  utils::concurrency::Channel<absl::StatusOr<Blob>> async_pending_answers_;

  // Idx of the next worker to receive a job if the worker idx is not specified
  // by the user.
  std::atomic<int> next_auto_worker_idx_ = {0};

  // Check if "Done" was called. "Done" will be called as the object destruction
  // if it was not called manually before.
  bool done_was_called_ = false;

  std::shared_ptr<grpc::ChannelCredentials> credential_;
};

REGISTER_Distribution_Manager(GRPCManager, GRPCManager::kKey);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_GRPC_MANAGER_H_
