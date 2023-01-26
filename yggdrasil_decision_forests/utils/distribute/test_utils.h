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

// Utility functions to test distribute implementations.

#ifndef THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_TEST_UTILS_H_
#define THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_TEST_UTILS_H_

#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace distribute {

// A manager and a set of thread containing running the worker binaries (if
// used).
struct ManagerAndWorkers {
  std::unique_ptr<AbstractManager> manager;
  std::vector<std::unique_ptr<utils::concurrency::Thread>> worker_threads;
  std::vector<std::unique_ptr<utils::concurrency::Thread>>
      discarded_worker_threads;

  void Join() {
    if (worker_threads.empty()) {
      return;
    }
    YDF_LOG(INFO) << "Waiting for workers to stop";
    for (auto& worker : worker_threads) {
      worker->Join();
    }
    YDF_LOG(INFO) << "Waiting for discarded workers to stop";
    for (auto& worker : discarded_worker_threads) {
      worker->Join();
    }
    worker_threads.clear();
  }
};

struct ManagerCreatorAndWorkers {
  std::function<std::unique_ptr<AbstractManager>()> manager_creator;
  std::vector<std::unique_ptr<utils::concurrency::Thread>> worker_threads;
  std::vector<std::unique_ptr<utils::concurrency::Thread>>
      discarded_worker_threads;

  void Join() {
    if (worker_threads.empty()) {
      return;
    }
    YDF_LOG(INFO) << "Waiting for workers to stop";
    for (auto& worker : worker_threads) {
      worker->Join();
    }
    worker_threads.clear();
  }
};

// Various tests.
void TestWorkerError(AbstractManager* manager, bool call_done = true);
void TestBlockingRequest(AbstractManager* manager, bool call_done = true);
void TestBlockingRequestWithSpecificWorker(AbstractManager* manager,
                                           bool call_done = true);
void TestAsynchronousRequest(AbstractManager* manager, bool call_done = true);
void TestAsynchronousRequestWithSpecificWorker(AbstractManager* manager,
                                               bool call_done = true);
void TestAsynchronousIntraWorkerCommunication(AbstractManager* manager,
                                              bool call_done = true);
void TestAsynchronousParallelWorkerExecution(AbstractManager* manager,
                                             bool call_done = true);
void TestChangeManager(ManagerCreatorAndWorkers* manager_creator, bool nice);

// Call various other tests (e.g., TestWorkerError) interrupted with calls to
// "messup". "messup" is expected to make some changes (e.g. relocate a worker)
// that might cause failure. The test checks if the manager can recover from
// such failure.
void TestMessup(AbstractManager* manager, std::function<void()> messup);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_TEST_UTILS_H_
