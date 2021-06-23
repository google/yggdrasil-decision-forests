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

// Distribution over a set of processed communicating through GRPC.

#ifndef THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_GRPC_MANAGER_H_
#define THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_GRPC_MANAGER_H_

#include "grpcpp/channel.h"
#include "grpcpp/server.h"
#include "absl/container/node_hash_map.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc.grpc.pb.h"

namespace yggdrasil_decision_forests {
namespace distribute {

class GRPCManager : public AbstractManager {
 public:
  static constexpr char kKey[] = "GRPC";

  virtual ~GRPCManager() {
    if (!done_was_called_) {
      LOG(WARNING) << "Calling Done in distribution manager destructor";
      CHECK_OK(Done({}));
    }
  }

  utils::StatusOr<Blob> BlockingRequest(Blob blob, int worker_idx) override;

  absl::Status AsynchronousRequest(Blob blob, int worker_idx) override;

  utils::StatusOr<Blob> NextAsynchronousAnswer() override;

  int NumWorkers() override;

  absl::Status Done(absl::optional<bool> kill_worker_manager) override;

 private:
  struct Worker {
    int worker_idx;
    std::unique_ptr<proto::Server::Stub> stub;
    std::string address;
    std::shared_ptr<grpc::Channel> channel;

    // Async query to execute specific to this worker.
    utils::concurrency::Channel<Blob> async_pending_queries_;

    std::unique_ptr<utils::concurrency::Thread> main_thread_1;
    std::unique_ptr<utils::concurrency::Thread> main_thread_2;
  };

  absl::Status Initialize(const proto::Config& config,
                          const absl::string_view worker_name,
                          Blob welcome_blob) override;

  absl::Status InitializeWorkers(const proto::Config& config);

  absl::Status InitializeConfigFile(const proto::Config& config,
                                    const absl::string_view worker_name,
                                    Blob welcome_blob);

  void WorkerMain1(Worker* worker);
  void WorkerMain2(Worker* worker);

  void WorkerRun(Blob blob, Worker* worker);

  void JoinWorkers();

  // Path to serialized worker configuration accessible by all workers.
  std::string worker_config_path_;
  bool verbose_ = true;
  std::vector<std::unique_ptr<Worker>> workers_;

  // Manager UID.
  uint64_t manager_uid_;

  // Async query to execute by any worker.
  utils::concurrency::Channel<Blob> async_pending_queries_;

  // Async answers that should be returned, indexed by task.
  utils::concurrency::Channel<proto::Answer> async_pending_answers_;

  // Idx of the next worker to receive a job if the worker idx is not specified
  // by the user.
  std::atomic<int> next_auto_worker_idx_ = 0;

  // Check if "Done" was called. "Done" will be called as the object destruction
  // if it was not called manually before.
  bool done_was_called_ = false;
};

REGISTER_Distribution_Manager(GRPCManager, GRPCManager::kKey);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_GRPC_MANAGER_H_
