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

#include <stddef.h>

#include <atomic>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/optional.h"

#include "yggdrasil_decision_forests/utils/concurrency.h"  // IWYU pragma: keep

namespace yggdrasil_decision_forests::utils::concurrency {
namespace {

TEST(ThreadPool, Empty) {
  { ThreadPool pool("MyPool", 1); }
}

TEST(ThreadPool, Simple) {
  std::atomic<int> counter{0};
  const int n = 100;
  {
    ThreadPool pool("MyPool", 1);
    pool.StartWorkers();
    for (int i = 1; i <= n; i++) {
      pool.Schedule([&, i]() { counter += i; });
    }
  }
  EXPECT_EQ(counter, n * (n + 1) / 2);
}

TEST(StreamProcessor, Simple) {
  const int num_jobs = 1000;
  const int num_initially_planned_jobs = 10;

  StreamProcessor<int, int> processor(
      "MyPipe", /*num_threads=*/num_initially_planned_jobs,
      [](int x) { return x; });

  int sum = 0;
  processor.StartWorkers();

  // Start one job for each thread.
  for (int i = 0; i < num_initially_planned_jobs; i++) {
    processor.Submit(i);
  }

  // Continuously consume a result, and restart a new job.
  for (int i = 0; i < num_jobs; i++) {
    const absl::optional<int> result_or = processor.GetResult();
    ASSERT_TRUE(result_or.has_value());
    sum += *result_or;
    if (i < num_jobs - num_initially_planned_jobs) {
      processor.Submit(i + num_initially_planned_jobs);
    }
  }

  // Ensures that all jobs have be run exactly once.
  EXPECT_EQ(sum, (num_jobs - 1) * num_jobs / 2);
}

TEST(StreamProcessor, NonCopiableData) {
  using Question = std::unique_ptr<int>;
  using Answer = std::unique_ptr<int>;

  StreamProcessor<Question, Answer> processor("MyPipe", 5,
                                              [](Question x) { return x; });

  processor.StartWorkers();
  processor.Submit(std::make_unique<int>(10));
  const absl::optional<std::unique_ptr<int>> result_or = processor.GetResult();
  EXPECT_THAT(result_or, testing::Optional(testing::Pointee(10)));
}

TEST(StreamProcessor, InOrder) {
  const int num_jobs = 1000;
  const int num_initially_planned_jobs = 10;

  StreamProcessor<int, int> processor(
      "MyPipe", /*num_threads=*/num_initially_planned_jobs,
      [](int x) { return x; }, /*result_in_order=*/true);

  // Next query to send.
  int next_query = 0;
  // Extracted next result value.
  int next_expected_result = 0;

  processor.StartWorkers();

  // Start one job for each thread.
  for (int i = 0; i < num_initially_planned_jobs; i++) {
    processor.Submit(next_query++);
  }

  // Continuously consume a result, and restart a new job.
  for (int i = 0; i < num_jobs; i++) {
    const absl::optional<int> result = processor.GetResult();
    EXPECT_THAT(result, testing::Optional(next_expected_result++));
    processor.Submit(next_query++);
  }
}

TEST(StreamProcessor, EarlyClose) {
  StreamProcessor<int, int> processor(
      "MyPipe", 5, [](int x) { return x; }, /*result_in_order=*/true);

  processor.StartWorkers();
  processor.Submit(1);
  processor.Submit(2);
  processor.Submit(3);
  processor.CloseSubmits();

  absl::optional<int> result = processor.GetResult();
  EXPECT_THAT(result, testing::Optional(1));
  result = processor.GetResult();
  EXPECT_THAT(result, testing::Optional(2));
  result = processor.GetResult();
  EXPECT_THAT(result, testing::Optional(3));
  EXPECT_FALSE(processor.GetResult().has_value());

  processor.JoinAllAndStopThreads();
}

TEST(Utils, ConcurrentForLoop) {
  std::atomic<int> sum{0};
  std::vector<int> items(500, 2);
  {
    ThreadPool pool("", 5);
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
}  // namespace yggdrasil_decision_forests::utils::concurrency
