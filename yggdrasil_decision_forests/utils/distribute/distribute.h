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

// Abstraction for the distribution of computation over different systems (e.g.
// Multi-thread, multi-process, Borg+GRPC, TF Distributed).
//
// Usage example:
//
// // Worker code
// class MyWorker : public AbstractWorker {
//  public:
//   absl::Status Setup(Blob welcome_blob) override {
//     LOG(INFO) << "Initialization of the worker #" << WorkerIdx();
//     return absl::OkStatus();
//   }
//
//   absl::Status Done() override {
//     LOG(INFO) << "Termination of the worker #" << WorkerIdx();
//     return absl::OkStatus();
//   }
//
//   utils::StatusOr<Blob> RunRequest(Blob blob) override {
//     LOG(INFo) << "Processing " << blob;
//     return "MyAnswer";
//   }
// };
// REGISTER_Distribution_Worker(MyWorker, "MY_WORKER");
//
// // Create the manager code
// auto manager = CreateManager(config,"MY_WORKER", "welcome message");
// // Run a synchronous tasks on a random worker
// auto result = manager->BlockingRequest("data for task2").value();
// // Run an asynchronous tasks on a random worker
// manager->AsynchronousRequest("data for task2");
// result = manager->NextAsynchronousAnswer().value();
// // Stop the manager and the workers
// manager->Shutdown();
//
// Rules
//   - There can only be one manager per worker. However, it is possible for the
//     manager to die and re-connect to the workers.
//   - It is up to the client library to handle checkpoint and restarts. The
//     "temporary_directory()" available both for the manager and workers can be
//     used.
//   - Error in a worker job (worker->RunRequest()) is returned to the manager,
//     but does not stop the worker.
//   - If the manager changes (because a new one was initialized, or because the
//     manager job was restarted), a new worker object "AbstractWorker" will be
//     instantiated for the new manager.
//   - (Currently only for the GRPC distribution strategy) When a worker is
//     restarted (previous point), the previous "AbstractWorker" object is
//     destructed before the new one is created.
//
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_DISTRIBUTE_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_DISTRIBUTE_H_

#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.pb.h"

namespace yggdrasil_decision_forests {
namespace distribute {

// Instantiate a manager.
//
// Args:
//   config: Selection and configuration of the distribution implementation.
//   worker_name: Target registered worker class.
//   welcome_blob: Data received by each worker during setup and
//     restart-after-rescheduling.
//   parallel_execution_per_worker: Number of jobs that each worker will run in
//     parallel. For example, If parallel_execution_per_worker=1, each worker
//     will receive only one job at a time.
//
utils::StatusOr<std::unique_ptr<AbstractManager>> CreateManager(
    const proto::Config& config, absl::string_view worker_name,
    Blob welcome_blob, int parallel_execution_per_worker = 1);

// Gets the number of workers available in a distribute configuration without
// having to create the distribution manager. If a distribute manager is
// available, use the "distribute->NumWorkers()" instead.
utils::StatusOr<int> NumWorkers(const proto::Config& config);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_CORE_H_
