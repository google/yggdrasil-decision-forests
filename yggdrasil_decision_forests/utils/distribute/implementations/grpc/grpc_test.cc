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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc.pb.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_manager.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_worker.h"
#include "yggdrasil_decision_forests/utils/distribute/test_utils.h"
#include "yggdrasil_decision_forests/utils/distribute/toy_worker.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace distribute {
namespace grpc_worker {
namespace {

// Create a GRPC manager and its workers.
ManagerCreatorAndWorkers CreateGrpcManagerCreator(
    int parallel_execution_per_worker = 1, int num_workers = 5) {
  ManagerCreatorAndWorkers manager_and_workers;
  // Manager configuration.
  proto::Config config;
  config.set_implementation_key("GRPC");
  config.set_verbosity(1);
  config.set_working_directory(
      file::JoinPath(test::TmpDirectory(), "work_dir"));
  auto* addresses =
      config.MutableExtension(proto::grpc)->mutable_socket_addresses();
  for (int worker_idx = 0; worker_idx < num_workers; worker_idx++) {
    // Create address.
    auto* address = addresses->add_addresses();
    address->set_ip("localhost");
    std::string error;
    const int port = test::PickUnusedPortOrDie();
    CHECK_GT(port, 0);
    address->set_port(port);

    // Create worker thread.
    manager_and_workers.worker_threads.push_back(
        absl::make_unique<utils::concurrency::Thread>(
            [port]() { CHECK_OK(WorkerMain(port)); }));
    absl::SleepFor(absl::Seconds(0.2));
  }

  // Start manager.
  manager_and_workers.manager_creator = [config,
                                         parallel_execution_per_worker]() {
    return CreateManager(config, /*worker_name=*/kToyWorkerKey,
                         /*welcome_blob=*/"hello",
                         parallel_execution_per_worker)
        .value();
  };
  return manager_and_workers;
}

ManagerAndWorkers CreateGrpcManager(int parallel_execution_per_worker = 1) {
  auto creator = CreateGrpcManagerCreator(parallel_execution_per_worker);
  ManagerAndWorkers m_and_w;
  m_and_w.worker_threads = std::move(creator.worker_threads);
  m_and_w.manager = creator.manager_creator();
  return m_and_w;
}

TEST(GRPC, WorkerError) {
  auto all = CreateGrpcManager();
  TestWorkerError(all.manager.get());
  all.Join();
}

TEST(GRPC, BlockingRequest) {
  auto all = CreateGrpcManager();
  TestBlockingRequest(all.manager.get());
  all.Join();
}

TEST(GRPC, AsynchronousRequest) {
  auto all = CreateGrpcManager();
  TestAsynchronousRequest(all.manager.get());
  all.Join();
}

TEST(GRPC, BlockingRequestWithSpecificWorker) {
  auto all = CreateGrpcManager();
  TestBlockingRequestWithSpecificWorker(all.manager.get());
  all.Join();
}

TEST(GRPC, AsynchronousRequestWithSpecificWorker) {
  auto all = CreateGrpcManager();
  TestAsynchronousRequestWithSpecificWorker(all.manager.get());
  all.Join();
}

TEST(GRPC, AsynchronousIntraWorkerCommunication) {
  auto all = CreateGrpcManager();
  TestAsynchronousIntraWorkerCommunication(all.manager.get());
  all.Join();
}

TEST(GRPC, AsynchronousParallelWorkerExecution) {
  auto all = CreateGrpcManager(5);
  TestAsynchronousParallelWorkerExecution(all.manager.get());
  all.Join();
}

TEST(GRPC, TestChangeManagerNice) {
  auto all = CreateGrpcManagerCreator(5, 1);
  TestChangeManager(&all, /*nice=*/true);
  all.Join();
}

  TEST(GRPC, DISABLED_TestChangeManagerNotNice) {
  auto all = CreateGrpcManagerCreator(5, 1);
  TestChangeManager(&all, /*nice=*/false);
  all.Join();
}

TEST(GRPC, TestMessup) {
  auto all = CreateGrpcManager();
  TestMessup(all.manager.get(), [&]() {
    // This method creates a new worker #0, and tell the manager to replace the
    // old worker #0 by the new worker #0. After this change, any call to the
    // old worker #0 will trigger a YDF_LOG(FATAL).

    // Mark worker #0 as forbidden. New requests to this worker will trigger a
    // YDF_LOG(FATAL).
    CHECK_OK(all.manager->BlockingRequest("forbidden", 0).status());

    auto* grpc_manager = dynamic_cast<GRPCManager*>(all.manager.get());
    CHECK(grpc_manager != nullptr);
    CHECK_OK(grpc_manager->DebugShutdownWorker(0));

    // New port for worker #0.
    const int port = test::PickUnusedPortOrDie();
    CHECK_GT(port, 0);

    // Isolate the forbidden worker thread.
    all.discarded_worker_threads.push_back(
        std::move(all.worker_threads.front()));

    // Create worker thread.
    all.worker_threads.front() = absl::make_unique<utils::concurrency::Thread>(
        [port]() { CHECK_OK(WorkerMain(port)); });
    absl::SleepFor(absl::Seconds(0.2));

    CHECK_OK(grpc_manager->UpdateWorkerAddress(
        /*worker_idx=*/0, absl::StrCat("localhost:", port)));
  });
  all.Join();
}

}  // namespace
}  // namespace grpc_worker
}  // namespace distribute
}  // namespace yggdrasil_decision_forests
