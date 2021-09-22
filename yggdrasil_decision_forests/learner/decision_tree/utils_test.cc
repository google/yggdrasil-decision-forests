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

#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"

#include <atomic>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {
namespace {

TEST(Utils, ConcurrentForLoop) {
  std::atomic<int> sum{0};
  std::vector<int> items(500, 2);
  {
    utils::concurrency::ThreadPool pool("", 5);
    pool.StartWorkers();
    ConcurrentForLoop(
        4, &pool, items.size(),
        [&sum, &items](size_t block_idx, size_t begin_idx, size_t end_idx) {
          int a = 0;
          for (int i = begin_idx; i < end_idx; i++) {
            a += items[i];
          }
          sum += a;
        });
  }
  EXPECT_EQ(sum, items.size() * 2);
}

}  // namespace
}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
