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

#include "yggdrasil_decision_forests/utils/distribute/test_utils.h"

namespace yggdrasil_decision_forests {
namespace distribute {

void TestWorkerError(AbstractManager* manager) {
  EXPECT_FALSE(manager->BlockingRequest("gen_error").ok());
  EXPECT_OK(manager->Done(true));
}

void TestBlockingRequest(AbstractManager* manager) {
  for (int i = 0; i <= 100; i++) {
    auto result =
        manager->BlockingRequest(absl::StrCat("identity:", i)).value();
    EXPECT_EQ(result, absl::StrCat(i));
  }
  EXPECT_OK(manager->Done(true));
}

void TestBlockingRequestWithSpecificWorker(AbstractManager* manager) {
  for (int i = 0; i <= 100; i++) {
    const int worker_idx = i % manager->NumWorkers();
    auto result = manager->BlockingRequest("worker_idx", worker_idx).value();
    EXPECT_EQ(result, absl::StrCat(worker_idx));
  }
  EXPECT_OK(manager->Done(true));
}

void TestAsynchronousRequest(AbstractManager* manager) {
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
  EXPECT_OK(manager->Done(true));
}

void TestAsynchronousRequestWithSpecificWorker(AbstractManager* manager) {
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

  EXPECT_OK(manager->Done(true));
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
