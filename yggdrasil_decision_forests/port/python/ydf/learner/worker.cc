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

#include "ydf/learner/worker.h"

#include <pybind11/pybind11.h>
#include <stdint.h>

#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ydf/utils/status_casters.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_worker.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {
namespace {

// Starts a worker and block until the worker is shutdown.
absl::Status StartWorkerBlocking(const int port) {
  YDF_LOG(INFO) << "Start YDF worker on port " << port;
  py::gil_scoped_release release;
  const auto status =
      distribute::grpc_worker::WorkerMain(port, /*use_loas=*/false);
  YDF_LOG(INFO) << "Stopping worker";
  return status;
}

// A single non blocking worker.
struct NonBlockingWorker {
  std::unique_ptr<distribute::grpc_worker::GRPCWorkerServer> server;
  std::unique_ptr<utils::concurrency::Thread> thread;
};

// All the non-blocking workers.
struct NonBlockingWorkers {
  absl::flat_hash_map<uint64_t, NonBlockingWorker> per_uid GUARDED_BY(mutex);
  utils::concurrency::Mutex mutex;
};

NonBlockingWorkers& GetNonBlockingWorkers() {
  static NonBlockingWorkers data;
  return data;
}

// Starts a worker and returns immediately a key to later stop the worker.
absl::StatusOr<uint64_t> StartWorkerNonBlocking(const int port) {
  YDF_LOG(INFO) << "Start YDF worker on port " << port;

  ASSIGN_OR_RETURN(auto server, distribute::grpc_worker::StartGRPCWorker(
                                    /*port=*/port,
                                    /*use_loas=*/false));
  auto thread =
      absl::make_unique<utils::concurrency::Thread>([server = server.get()]() {
        distribute::grpc_worker::WaitForGRPCWorkerToShutdown(server);
      });

  NonBlockingWorkers& workers = GetNonBlockingWorkers();
  utils::concurrency::MutexLock l(&workers.mutex);
  const uint64_t uid = utils::GenUniqueIdUint64();
  NonBlockingWorker& worker = workers.per_uid[uid];
  worker.server = std::move(server);
  worker.thread = std::move(thread);
  return uid;
}

absl::Status StopWorkerNonBlocking(const uint64_t uid) {
  YDF_LOG(INFO) << "Stop YDF worker";

  NonBlockingWorker worker;
  {
    NonBlockingWorkers& workers = GetNonBlockingWorkers();
    utils::concurrency::MutexLock l(&workers.mutex);

    auto it = workers.per_uid.find(uid);
    if (it == workers.per_uid.end()) {
      return absl::InvalidArgumentError(
          "Cannot stop non existing non-blocking worker");
    }
    worker = std::move(it->second);
    workers.per_uid.erase(it);
  }

  worker.server->stop_server.Notify();
  YDF_LOG(INFO) << "\tWaiting for worker to finish its work";
  worker.thread->Join();
  worker.thread.reset();
  YDF_LOG(INFO) << "\tWorker have been stopped";
  return absl::OkStatus();
}

}  // namespace

void init_worker(py::module_& m) {
  m.def("StartWorkerBlocking", WithStatus(StartWorkerBlocking),
        py::arg("port"));
  m.def("StartWorkerNonBlocking", WithStatusOr(StartWorkerNonBlocking),
        py::arg("port"));
  m.def("StopWorkerNonBlocking", WithStatus(StopWorkerNonBlocking),
        py::arg("uid"));
}

}  // namespace yggdrasil_decision_forests::port::python
