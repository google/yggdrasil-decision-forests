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

#include "yggdrasil_decision_forests/utils/distribute/distribute.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/single_thread/single_thread.pb.h"
#include "yggdrasil_decision_forests/utils/distribute/test_utils.h"
#include "yggdrasil_decision_forests/utils/distribute/toy_worker.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace distribute {
namespace {

// Create a single thread manager with 5 workers.
ManagerAndWorkers CreateSingleThreadManager() {
  ManagerAndWorkers manager_and_workers;
  proto::Config config;
  config.set_implementation_key("SINGLE_THREAD");
  config.MutableExtension(proto::single_thread)->set_num_workers(5);
  config.set_verbose(false);
  manager_and_workers.manager =
      CreateManager(config, /*worker_name=*/kToyWorkerKey,
                    /*welcome_blob=*/"hello")
          .value();
  return manager_and_workers;
}

// Test invalid manager configurations.
TEST(MultiThread, InitializationError) {
  proto::Config config;
  config.set_implementation_key("unknown implementation");
  EXPECT_THAT(CreateManager(config, /*worker_name=*/kToyWorkerKey,
                            /*welcome_blob=*/"hello")
                  .status(),
              test::StatusIs(absl::StatusCode::kInvalidArgument));

  config.set_implementation_key("SINGLE_THREAD");
  EXPECT_THAT(CreateManager(config, /*worker_name=*/"unknown worker key",
                            /*welcome_blob=*/"hello")
                  .status(),
              test::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SingleThread, WorkerError) {
  auto all = CreateSingleThreadManager();
  TestWorkerError(all.manager.get());
  all.Join();
}

TEST(SingleThread, BlockingRequest) {
  auto all = CreateSingleThreadManager();
  TestBlockingRequest(all.manager.get());
  all.Join();
}

TEST(SingleThread, AsynchronousRequest) {
  auto all = CreateSingleThreadManager();
  TestAsynchronousRequest(all.manager.get());
  all.Join();
}

TEST(SingleThread, BlockingRequestWithSpecificWorker) {
  auto all = CreateSingleThreadManager();
  TestBlockingRequestWithSpecificWorker(all.manager.get());
  all.Join();
}

TEST(SingleThread, AsynchronousRequestWithSpecificWorker) {
  auto all = CreateSingleThreadManager();
  TestAsynchronousRequestWithSpecificWorker(all.manager.get());
  all.Join();
}

}  // namespace
}  // namespace distribute
}  // namespace yggdrasil_decision_forests
