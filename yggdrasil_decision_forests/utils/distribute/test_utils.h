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

  void Join() {
    if (worker_threads.empty()) {
      return;
    }
    LOG(INFO) << "Waiting for workers to stop";
    for (auto& worker : worker_threads) {
      worker->Join();
    }
    worker_threads.clear();
  }
};

struct ManagerCreatorAndWorkers {
  std::function<std::unique_ptr<AbstractManager>()> manager_creator;
  std::vector<std::unique_ptr<utils::concurrency::Thread>> worker_threads;

  void Join() {
    if (worker_threads.empty()) {
      return;
    }
    LOG(INFO) << "Waiting for workers to stop";
    for (auto& worker : worker_threads) {
      worker->Join();
    }
    worker_threads.clear();
  }
};

// Various tests.
void TestWorkerError(AbstractManager* manager);
void TestBlockingRequest(AbstractManager* manager);
void TestBlockingRequestWithSpecificWorker(AbstractManager* manager);
void TestAsynchronousRequest(AbstractManager* manager);
void TestAsynchronousRequestWithSpecificWorker(AbstractManager* manager);
void TestAsynchronousIntraWorkerCommunication(AbstractManager* manager);
void TestAsynchronousParallelWorkerExecution(AbstractManager* manager);
void TestChangeManager(ManagerCreatorAndWorkers* manager_creator, bool nice);
}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_TEST_UTILS_H_
