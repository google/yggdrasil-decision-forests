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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_CORE_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_CORE_H_

#include <string>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.pb.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/registration.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace distribute {

// A chunk of data exchanged between the manager and the workers. When possible,
// the blob should be moved (std::move).
typedef std::string Blob;

// Abstract worker class containing the custom worker logic.
//
// A separate worker is instantiated on each distribution unit (e.g. machine).
class AbstractWorker {
 public:
  virtual ~AbstractWorker() = default;

  // Callback at the worker creation. Called again if the worker stops (e.g.
  // rescheduled) and restart. Called before any other public method.
  virtual absl::Status Setup(Blob welcome_blob) { return absl::OkStatus(); }

  // Called when the manager calls "ShutDown", and before the worker object is
  // destroyed. Not called if the worker is killed or is reallocated. Not called
  // if the manager is killed and restarted. Called after "Setup".
  //
  // Can be called while a "RunRequest" is running. In this case, "RunRequest"
  // can return immediately and its result will be ignored.
  virtual absl::Status Done() { return absl::OkStatus(); }

  // Callback for the execution of a request.
  // Can be called by different threads at the same time. It is up to the
  // "RunRequest" implementation to handle thread safety.
  virtual utils::StatusOr<Blob> RunRequest(Blob blob) = 0;

  // Index of the worker.
  int WorkerIdx() const { return worker_idx_; }

 private:
  // Initialize the internal fields of the workers. Called before "Setup".
  absl::Status InternalInitialize(int worker_idx);

  int worker_idx_;

  friend absl::Status InternalInitializeWorker(int worker_idx,
                                               AbstractWorker* worker);
};

REGISTRATION_CREATE_POOL(AbstractWorker);

#define REGISTER_Distribution_Worker(implementation, worker_name) \
  REGISTRATION_REGISTER_CLASS(implementation, worker_name, AbstractWorker);

// Manages a remote collection of workers. Should be created with
// "CreateManager()". Unless specified otherwise, all the methods are thread
// safe.
class AbstractManager {
 public:
  virtual ~AbstractManager() = default;

  // Runs a task and wait for the result.
  // If the worker idx is <0, it is up to the manager to select a worker.
  virtual utils::StatusOr<Blob> BlockingRequest(Blob blob,
                                                int worker_idx = -1) = 0;

  // Same as "BlockingRequest", but serialize the blob from a proto.
  template <typename Result, typename Request>
  utils::StatusOr<Result> BlockingProtoRequest(Request request,
                                               int worker_idx = -1);

  // Runs a task but do not wait for the result. The result can be retrieved by
  // "NextAsynchronousAnswer". If the worker idx is <0, it is up to the manager
  // to select a worker.
  virtual absl::Status AsynchronousRequest(Blob blob, int worker_idx = -1) = 0;

  template <typename Request>
  absl::Status AsynchronousProtoRequest(Request request, int worker_idx = -1);

  // Waits and retrieves the next answer from an AsynchronousRequest or
  // AsynchronousProtoRequest.
  virtual utils::StatusOr<Blob> NextAsynchronousAnswer() = 0;

  // Same as "NextAsynchronousAnswer", but unserialize the answer into a proto.
  template <typename Result>
  utils::StatusOr<Result> NextAsynchronousProtoAnswer();

  // Number of workers.
  virtual int NumWorkers() = 0;

  // Calls "Done" and destroy on all the worker instances.
  //
  // If "kill_worker_manager" is true, the worker manager will be killed and not
  // reusable by another manager. If "kill_worker_manager" is false, the worker
  // manager will remain active and wait for possibly another manager and other
  // workers. If "kill_worker_manager" is not set, it is up to the
  // implementation to decide of the solution. In this case, and as a rule of
  // thumb, the worker manager will be killed iff. the implementation started
  // the worker manager itself.
  //
  // Requesting a new work (e.g. BlockingRequest) or waiting for a result (e.g.
  // NextAsynchronousAnswer) after Done is called results in an error.
  virtual absl::Status Done(absl::optional<bool> kill_worker_manager = {}) = 0;

 private:
  // Called one at creation with the distribution configuration.
  virtual absl::Status Initialize(const proto::Config& config,
                                  const absl::string_view worker_name,
                                  Blob welcome_blob) {
    return absl::OkStatus();
  }

  friend utils::StatusOr<std::unique_ptr<AbstractManager>> CreateManager(
      const proto::Config& config, absl::string_view worker_name,
      Blob welcome_blob);
};

REGISTRATION_CREATE_POOL(AbstractManager);

#define REGISTER_Distribution_Manager(implementation, name) \
  REGISTRATION_REGISTER_CLASS(implementation, name, AbstractManager);

// Calls worker->InternalInitialize().
absl::Status InternalInitializeWorker(int worker_idx, AbstractWorker* worker);

template <typename Result, typename Request>
utils::StatusOr<Result> AbstractManager::BlockingProtoRequest(
    Request request, const int worker_idx) {
  ASSIGN_OR_RETURN(auto serialized_result,
                   BlockingRequest(request.SerializeAsString(), worker_idx));
  return utils::ParseBinaryProto<Result>(serialized_result);
}

template <typename Request>
absl::Status AbstractManager::AsynchronousProtoRequest(Request request,
                                                       int worker_idx) {
  return AsynchronousRequest(request.SerializeAsString(), worker_idx);
}

template <typename Result>
utils::StatusOr<Result> AbstractManager::NextAsynchronousProtoAnswer() {
  ASSIGN_OR_RETURN(auto serialized_result, NextAsynchronousAnswer());
  return utils::ParseBinaryProto<Result>(serialized_result);
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_CORE_H_
