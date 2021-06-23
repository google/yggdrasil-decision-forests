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

#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_manager.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_worker.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc.pb.h"
#include "yggdrasil_decision_forests/utils/distribute/test_utils.h"
#include "yggdrasil_decision_forests/utils/distribute/toy_worker.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace distribute {
namespace {

// Create a GRPC manager and its workers.
ManagerAndWorkers CreateGrpcManager() {
  ManagerAndWorkers manager_and_workers;
  // Manager configuration.
  proto::Config config;
  config.set_implementation_key("GRPC");
  config.set_verbose(true);
  config.set_working_directory(
      file::JoinPath(test::TmpDirectory(), "work_dir"));
  auto* addresses =
      config.MutableExtension(proto::grpc)->mutable_socket_addresses();
  for (int worker_idx = 0; worker_idx < 5; worker_idx++) {
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
            [port]() { CHECK_OK(GRPCWorkerMainWorkerMain(port)); }));
    absl::SleepFor(absl::Seconds(1));
  }

  // Start manager.
  manager_and_workers.manager =
      CreateManager(config, /*worker_name=*/kToyWorkerKey,
                    /*welcome_blob=*/"hello")
          .value();
  return manager_and_workers;
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

}  // namespace
}  // namespace distribute
}  // namespace yggdrasil_decision_forests
