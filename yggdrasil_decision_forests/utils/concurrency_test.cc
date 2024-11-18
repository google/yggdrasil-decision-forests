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
#include <optional>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/utils/test.h"

#include "yggdrasil_decision_forests/utils/concurrency.h"  // IWYU pragma: keep

namespace yggdrasil_decision_forests::utils::concurrency {
namespace {

using ::testing::TestWithParam;

TEST(ThreadPool, Empty) {
  {
    ThreadPool pool(1, {.name_prefix = std::string("MyPool")});
  }
}

TEST(ThreadPool, Simple) {
  std::atomic<int> counter{0};
  const int n = 100;
  {
    ThreadPool pool(1, {.name_prefix = std::string("MyPool")});
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
    const std::optional<int> result_or = processor.GetResult();
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
  const std::optional<std::unique_ptr<int>> result_or = processor.GetResult();
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
    const std::optional<int> result = processor.GetResult();
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

  std::optional<int> result = processor.GetResult();
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
    ThreadPool pool(5, {.name_prefix = std::string("")});
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

struct ConcurrentForLoopWithWorkerTestCase {
  size_t max_num_threads;
};

using ConcurrentForLoopWithWorkerTest =
    TestWithParam<ConcurrentForLoopWithWorkerTestCase>;

INSTANTIATE_TEST_SUITE_P(
    ConcurrentForLoopWithWorkerTestSuiteInstantiation,
    ConcurrentForLoopWithWorkerTest,
    testing::ValuesIn<ConcurrentForLoopWithWorkerTestCase>({{10}, {1}}));

TEST_P(ConcurrentForLoopWithWorkerTest, Base) {
  const ConcurrentForLoopWithWorkerTestCase& test_case = GetParam();
  std::atomic<int> sum{0};

  struct Cache {
    size_t thread_idx;
  };

  const auto create_cache = [&](size_t thread_idx, size_t num_threads,
                                size_t block_size) -> Cache {
    return {.thread_idx = thread_idx};
  };

  const auto run = [&](size_t block_idx, size_t begin_item_idx,
                       size_t end_item_idx, Cache* cache) -> absl::Status {
    CHECK_GE(end_item_idx - begin_item_idx, 10);
    CHECK_GE(end_item_idx - begin_item_idx, 20);
    for (size_t i = begin_item_idx; i < end_item_idx; i++) {
      sum++;
    }
    return absl::OkStatus();
  };

  EXPECT_OK(utils::concurrency::ConcurrentForLoopWithWorker<Cache>(
      /*num_items=*/1000,
      /*max_num_threads=*/test_case.max_num_threads,
      /*min_block_size=*/10,
      /*max_block_size=*/20, create_cache, run));

  EXPECT_EQ(sum, 1000);
}

TEST(ConcurrentForLoopWithWorker, Error) {
  struct Cache {};

  const auto create_cache = [&](size_t thread_idx, size_t num_threads,
                                size_t block_size) -> Cache { return {}; };

  const auto run = [&](size_t block_idx, size_t begin_item_idx,
                       size_t end_item_idx, Cache* cache) -> absl::Status {
    if (begin_item_idx == 0) {
      return absl::InvalidArgumentError("error");
    }
    return absl::OkStatus();
  };

  EXPECT_THAT(utils::concurrency::ConcurrentForLoopWithWorker<Cache>(
                  /*num_items=*/1000,
                  /*max_num_threads=*/5,
                  /*min_block_size=*/10,
                  /*max_block_size=*/20, create_cache, run),
              test::StatusIs(absl::StatusCode::kInvalidArgument, "error"));
}

struct GetConfigTestCase {
  size_t num_items;
  size_t max_num_threads;
  size_t min_block_size;
  size_t max_block_size;
  size_t expected_block_size;
  size_t expected_num_threads;
};

using GetConfigTest = TestWithParam<GetConfigTestCase>;

INSTANTIATE_TEST_SUITE_P(GetConfigTestSuiteInstantiation, GetConfigTest,
                         testing::ValuesIn<GetConfigTestCase>({
                             {1000, 2000, 1, 20, 1, 1000},
                             {1000, 2000, 2, 20, 2, 500},
                             {1000, 50, 1, 20, 20, 50},
                             {1000, 50, 1, 10, 10, 50},
                         }));

TEST_P(GetConfigTest, Base) {
  const GetConfigTestCase& test_case = GetParam();
  size_t block_size;
  size_t num_threads;
  std::tie(block_size, num_threads) =
      internal::GetConfig(test_case.num_items, test_case.max_num_threads,
                          test_case.min_block_size, test_case.max_block_size);
  EXPECT_EQ(block_size, test_case.expected_block_size);
  EXPECT_EQ(num_threads, test_case.expected_num_threads);
}

}  // namespace
}  // namespace yggdrasil_decision_forests::utils::concurrency
