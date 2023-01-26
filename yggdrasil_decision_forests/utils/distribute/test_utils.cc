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

#include "yggdrasil_decision_forests/utils/distribute/test_utils.h"

namespace yggdrasil_decision_forests {
namespace distribute {

void TestWorkerError(AbstractManager* manager, bool call_done) {
  EXPECT_FALSE(manager->BlockingRequest("gen_error").ok());
  if (call_done) {
    EXPECT_OK(manager->Done(true));
  }
}

void TestBlockingRequest(AbstractManager* manager, bool call_done) {
  for (int i = 0; i <= 100; i++) {
    auto result =
        manager->BlockingRequest(absl::StrCat("identity:", i)).value();
    EXPECT_EQ(result, absl::StrCat(i));
  }
  if (call_done) {
    EXPECT_OK(manager->Done(true));
  }
}

void TestBlockingRequestWithSpecificWorker(AbstractManager* manager,
                                           bool call_done) {
  for (int i = 0; i <= 100; i++) {
    const int worker_idx = i % manager->NumWorkers();
    auto result = manager->BlockingRequest("worker_idx", worker_idx).value();
    EXPECT_EQ(result, absl::StrCat(worker_idx));
  }
  if (call_done) {
    EXPECT_OK(manager->Done(true));
  }
}

void TestAsynchronousRequest(AbstractManager* manager, bool call_done) {
  const int n = 100;
  std::vector<bool> meet(n, false);
  for (int i = 0; i < n; i++) {
    EXPECT_OK(manager->AsynchronousRequest(absl::StrCat("identity:", i)));
  }
  for (int i = 0; i < n; i++) {
    auto result = manager->NextAsynchronousAnswer().value();
    int int_result = std::stoi(result);
    EXPECT_GE(int_result, 0);
    EXPECT_LT(int_result, n);
    EXPECT_FALSE(meet[int_result]);
    meet[int_result] = true;
  }
  if (call_done) {
    EXPECT_OK(manager->Done(true));
  }
}

void TestAsynchronousRequestWithSpecificWorker(AbstractManager* manager,
                                               bool call_done) {
  const int n = 100;
  for (int i = 0; i < n; i++) {
    const int worker_idx = i % manager->NumWorkers();
    EXPECT_OK(manager->AsynchronousRequest("worker_idx", worker_idx));
  }

  std::map<int, int> count;
  for (int i = 0; i < n; i++) {
    auto result = manager->NextAsynchronousAnswer().value();
    int int_result = std::stoi(result);
    count[int_result]++;
  }

  EXPECT_EQ(count.size(), manager->NumWorkers());
  for (const auto& value : count) {
    EXPECT_EQ(value.second, n / manager->NumWorkers());
  }

  if (call_done) {
    EXPECT_OK(manager->Done(true));
  }
}

void TestAsynchronousIntraWorkerCommunication(AbstractManager* manager,
                                              bool call_done) {
  for (int w = 0; w < manager->NumWorkers(); w++) {
    auto result =
        manager->BlockingRequest("worker_idx", /*worker_idx=*/w).value();
    EXPECT_EQ(result, absl::StrCat(w));
  }

  const int sum_worker_idxs =
      (manager->NumWorkers() - 1) * manager->NumWorkers() / 2;
  for (int w = 0; w < manager->NumWorkers(); w++) {
    auto result =
        manager->BlockingRequest("sum_other_worker_idxs", /*worker_idx=*/w)
            .value();
    EXPECT_EQ(result, absl::StrCat(sum_worker_idxs - w));
  }
  if (call_done) {
    EXPECT_OK(manager->Done(true));
  }
}

void TestAsynchronousParallelWorkerExecution(AbstractManager* manager,
                                             bool call_done) {
  const int n = 5;
  CHECK_OK(manager->BlockingRequest("create_5_barrier", 0).status());
  for (int i = 0; i < n; i++) {
    EXPECT_OK(manager->AsynchronousRequest("wait_barrier", 0));
  }
  for (int i = 0; i < n; i++) {
    EXPECT_OK(manager->NextAsynchronousAnswer().status());
  }
  if (call_done) {
    EXPECT_OK(manager->Done(true));
  }
}

void TestChangeManager(ManagerCreatorAndWorkers* manager_creator, bool nice) {
  auto m1 = manager_creator->manager_creator();

  EXPECT_EQ(m1->BlockingRequest("num_existing_toy_workers").value(), "1");

  EXPECT_EQ(m1->BlockingRequest("get").value(), "");
  EXPECT_OK(m1->BlockingRequest("set:1").status());
  EXPECT_EQ(m1->BlockingRequest("get").value(), "1");

  EXPECT_OK(m1->AsynchronousRequest("sleep", 0));
  EXPECT_OK(m1->AsynchronousRequest("sleep", 0));
  EXPECT_OK(m1->AsynchronousRequest("sleep", 0));

  // Do not check for the results.

  if (nice) {
    YDF_LOG(INFO) << "Nicely stop the manager";
    EXPECT_OK(m1->Done(false));
  }

  YDF_LOG(INFO) << "Create new manager";

  auto m2 = manager_creator->manager_creator();

  // Make sure the previous worker was removed before the new one as created.
  EXPECT_EQ(m2->BlockingRequest("num_existing_toy_workers").value(), "1");

  // Ensure there is never more than one worker initialized at any time.
  YDF_LOG(INFO) << "max_num_existing_toy_workers:"
                << m2->BlockingRequest("max_num_existing_toy_workers").value()
                << " (expecting 1 if the test is run in isolation)";

  EXPECT_EQ(m2->BlockingRequest("get").value(), "");
  EXPECT_OK(m2->BlockingRequest("set:2").status());
  EXPECT_EQ(m2->BlockingRequest("get").value(), "2");
  EXPECT_OK(m2->Done(true));
}

void TestMessup(AbstractManager* manager, std::function<void()> messup) {
  TestBlockingRequest(manager, false);

  YDF_LOG(INFO) << "Messup #1";
  messup();

  TestBlockingRequest(manager, false);

  YDF_LOG(INFO) << "Messup #2";
  messup();

  TestBlockingRequestWithSpecificWorker(manager, false);

  YDF_LOG(INFO) << "Messup #3";
  messup();

  TestAsynchronousRequest(manager, false);
  TestAsynchronousRequestWithSpecificWorker(manager, false);

  YDF_LOG(INFO) << "Messup #4";
  messup();

  TestAsynchronousIntraWorkerCommunication(manager, false);

  YDF_LOG(INFO) << "Messup #5";
  messup();

  TestAsynchronousIntraWorkerCommunication(manager, false);

  YDF_LOG(INFO) << "Done";
  EXPECT_OK(manager->Done(true));
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
