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

#ifndef THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_GRPC_WORKER_H_
#define THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_GRPC_WORKER_H_

#include "grpcpp/server.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc.grpc.pb.h"
#include "yggdrasil_decision_forests/utils/distribute/utils.h"

namespace yggdrasil_decision_forests {
namespace distribute {
namespace grpc_worker {

// Runs the a worker. The function is blocking until the manager stops the
// worker. If "port==0", automatically select a valid port.
absl::Status WorkerMain(int port, bool use_loas = false);

namespace internal {

class WorkerService;

// Inter worker communication hook.
class WorkerHook : public AbstractWorkerHook {
 public:
  WorkerHook(WorkerService* parent) : parent_(parent) {}

  absl::Status AsynchronousRequestToOtherWorker(
      Blob blob, int target_worker_idx,
      AbstractWorker* emitter_worker) override;

  absl::StatusOr<Blob> NextAsynchronousAnswerFromOtherWorker(
      AbstractWorker* emitter_worker) override;

 private:
  WorkerService* parent_;
};

// Implementation of the GRPC service for the worker.
class WorkerService final : public proto::Server::Service {
 public:
  WorkerService(utils::concurrency::Notification* stop_server, bool use_loas);

  void ShutDown();

 protected:
  // Implementation of worker->worker request.
  absl::Status AsynchronousRequestToOtherWorker(Blob blob,
                                                int target_worker_idx,
                                                AbstractWorker* emitter_worker);

  // Implementation of the worker->worker async reply.
  absl::StatusOr<Blob> NextAsynchronousAnswerFromOtherWorker(
      AbstractWorker* emitter_worker);

 private:
  // Execution of a query emitted by the manager.
  grpc::Status Run(grpc::ServerContext* context, const proto::Query* request,
                   proto::Answer* reply) override;

  // Execution of a query emitted by another worker.
  grpc::Status WorkerRun(grpc::ServerContext* context,
                         const proto::WorkerQuery* request,
                         proto::WorkerAnswer* reply) override;

  // Possibly update the inter worker connections if a worker was relocated.
  // Should be called before any communication in between workers.
  grpc::Status UpdateWorkerAddress(
      grpc::ServerContext* context,
      const proto::UpdateWorkerAddressQuery* request,
      proto::Empty* reply) override;

  grpc::Status Shutdown(grpc::ServerContext* context,
                        const proto::ShutdownQuery* request,
                        proto::Empty* reply) override;

  grpc::Status Ping(grpc::ServerContext* context, const proto::Empty* request,
                    proto::Empty* reply) override;

  // Calls "Done" on the worker, wait for all the pending operation to be done
  // or cancel, and destroy the "worker_" object.
  absl::Status BlockingDoneOnWorker(utils::concurrency::MutexLock* lock);

  // After a call to this method, the worker is ready to processed requests.
  // This method should be called before any request.
  absl::Status EnsureReadyWorker(uint64_t manager_uid,
                                 const proto::Query& request,
                                 const int worker_idx,
                                 utils::concurrency::MutexLock* lock);

  // Blocking inter worker request.
  absl::StatusOr<Blob> BlockingInterWorkerRequest(Blob blob,
                                                  const int target_worker);

  // Loop for a thread processing inter worker requests.
  void ProcessInterWorkerCommunication();

  // Initialize the connection and thread for the inter worker communication.
  // This method should be called before any inter worker communication.
  void InitializerInterWorkerCommunication(
      const proto::WorkerConfig& worker_config);

  // Ensures that the communication with another worker is ready.
  absl::StatusOr<proto::Server::Stub*> EnsureIntraWorkerStubIsReady(
      const int worker_idx);

  // Finalize the current worker communication.
  // No more inter worker communication should be done after this call, except
  // for "InitializerInterWorkerCommunication" to re-initialize it.
  void FinalizeIntraWorkerCommunication();

  // Non owning pointer to the notification that stops the server.
  utils::concurrency::Notification* stop_server_ = nullptr;

  // Active worker implementation.
  std::unique_ptr<AbstractWorker> worker_;

  // UID of the manager. Only valid if worker_ is set.
  uint64_t manager_uid_;

  // Fields related to the inter worker communication.
  struct InterWorkerCommunication {
    // List of target worker index and data emitted by this worker.
    utils::concurrency::Channel<std::pair<int, Blob>> pending_queries;

    // Answers to this worker queries.
    utils::concurrency::Channel<absl::StatusOr<Blob>> pending_answers;

    // Thread emitting and receiving intra-workers requests/answers.
    ThreadVector threads;

    struct Worker {
      std::unique_ptr<proto::Server::Stub> stub GUARDED_BY(mutex_address);

      // Address currently connected by the stub.
      std::string connected_address GUARDED_BY(mutex_address);

      // Address of the worker. "expected_address" and "connected_address" might
      // be different for a short time with a worker is re-located.
      std::string expected_address GUARDED_BY(mutex_address);

      // Disconnected worker stubs kept until releasing.
      // TODO: Release the discarded worker stubs.
      std::vector<std::unique_ptr<proto::Server::Stub>> discarded_stubs_
          GUARDED_BY(mutex_address);

      utils::concurrency::Mutex mutex_address;
    };

    // Communication channel to other workers for intra worker communication.
    std::vector<std::unique_ptr<Worker>> workers;
  };

  std::unique_ptr<InterWorkerCommunication> intra_worker_communication_;

  // utils::concurrency::Mutex protecting the initialization of the worker.
  utils::concurrency::Mutex mutex_ GUARDED_BY(mutex_);

  // True when the worker is being stopped (i.e. waiting for all the requests to
  // be completed) because the user called "Done" on the manager, or because the
  // manager have changed.
  bool stopping_worker_ GUARDED_BY(mutex_) = false;

  // Signal when the worker are done being stopped i.e. stopping_worker_ goes
  // from true to false.
  utils::concurrency::CondVar stopping_worker_done_cv_;

  // Signal are the end of a request execution when not other request are
  // running (i.e. num_active_requests_=0).
  utils::concurrency::CondVar request_done_cv_;

  // Number of requests currently being processed.
  int num_active_requests_ GUARDED_BY(mutex_) = 0;

  // Does the worker uses LOAS.
  bool use_loas_;

  // Callback to inter-worker communication.
  WorkerHook hook_;

  friend WorkerHook;
};

}  // namespace internal

// The class "GRPCWorkerServer", and the functions "StartGRPCWorker" and
// "WaitForGRPCWorkerToShutdown" are utilities to separate the creation and life
// of a server. When possible, use "WorkerMain" instead.

struct GRPCWorkerServer {
  utils::concurrency::Notification stop_server;
  std::unique_ptr<grpc::Server> grpc_server;
  std::unique_ptr<internal::WorkerService> service;
  std::unique_ptr<utils::concurrency::Thread> server_thread;
  int port;
};

// Non-blocking function to start a GRPC worker.
absl::StatusOr<std::unique_ptr<GRPCWorkerServer>> StartGRPCWorker(
    int port, bool use_loas = false);

// Blocking function for the execution of a GRPC worker.
void WaitForGRPCWorkerToShutdown(GRPCWorkerServer* server);

}  // namespace grpc_worker
}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_GRPC_WORKER_H_
