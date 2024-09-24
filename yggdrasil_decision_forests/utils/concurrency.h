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

// Various concurrency tools.
//
//   ThreadPool: Parallel execution of jobs (std::function<void(void)>) on a
//     pre-determined number of threads. Does not implement a maximum capacity.
//   StreamProcessor: Parallel processing of a stream of "Input" into a stream
//     of "Output" using a pre-determined number of threads. Does not implement
//     a maximum capacity.
//
// Usage examples:
//
//   # ThreadPool
//   {
//   ThreadPool pool("name", /*num_threads=*/10);
//   pool.StartWorkers();
//   pool.Schedule([](){...});
//   }
//
//   # StreamProcessor
//   StreamProcessor<int, int> processor("name", /*num_threads=*/10,
//     [](int x) { return x + 1; });
//   processor.StartWorkers();
//   while(...){
//     // Mix of:
//     processor.Submit(...);
//     // and
//     result = processor.GetResult();
//   }
//

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_H_

#include <stddef.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

#include "yggdrasil_decision_forests/utils/concurrency_default.h"

#include "yggdrasil_decision_forests/utils/concurrency_channel.h"
#include "yggdrasil_decision_forests/utils/concurrency_streamprocessor.h"

namespace yggdrasil_decision_forests::utils::concurrency {

// Applies "function" over a range of elements using multi-threading.
//
// Given "num_items" elements divided into "num_blocks" contiguous blocks of
// the same size (except possibly for the last one). This method calls
// "function" on each block in parallel using the thread-pool.
//
// The method is blocking until all the "function" call have returned.
//
// For example, support num_items=10 and num_blocks=3 defines the following
// blocks: [0,4), [4,8), [8,10). Then, "function" will be called in parallel on:
//   function(block_idx=0, begin_item_idx=0, end_item_idx=0)
//   function(block_idx=1, begin_item_idx=4, end_item_idx=8)
//   function(block_idx=2, begin_item_idx=8, end_item_idx=10)
//
// Args:
//   num_blocks: Number of range subsets.
//   thread_pool: Already started thread pool.
//   num_items: Number of items to process.
//   function: Function to call.
//
// function's signature:
//   block_idx: Index of the block in [0, num_blocks).
//   begin_item_idx: First item to process (inclusive).
//   end_item_idx: Last item to process (exclusive).
//
void ConcurrentForLoop(
    size_t num_blocks, ThreadPool* thread_pool, size_t num_items,
    const std::function<void(size_t block_idx, size_t begin_item_idx,
                             size_t end_item_idx)>& function);

// "ConcurrentForLoopWithWorker" applies "run" function over a range of elements
// using multi-threading.
//
// See "ConcurrentForLoop" for the definition of the arguments.
//
// "ConcurrentForLoopWithWorker" has the following advantages over
// "ConcurrentForLoop":
//   1. A per-thread working memory can be allocated once and reused.
//   2. Easy configuration of the resource allocations, especially to handle
//      edge cases with very small and very large workload.
//   3. Support for absl::Status.
//   4. Only use atomics to synchronize in between threads (unless there is a
//      status error).
//
// "ConcurrentForLoopWithWorker" has the following disadvantages over
// "ConcurrentForLoop":
//   1. Cannot use an existing thread-pool.
//
namespace internal {

// Computes the optimal block size and number of threads.
std::pair<size_t, size_t> GetConfig(size_t num_items, size_t max_num_threads,
                                    size_t min_block_size,
                                    size_t max_block_size);

}  // namespace internal

template <typename Cache>
absl::Status ConcurrentForLoopWithWorker(
    size_t num_items, size_t max_num_threads, size_t min_block_size,
    size_t max_block_size,
    const std::function<Cache(size_t thread_idx, size_t num_threads,
                              size_t block_size)>
        create_cache,
    const std::function<absl::Status(size_t block_idx, size_t begin_item_idx,
                                     size_t end_item_idx, Cache* cache)>
        run) {
  if (max_num_threads == 1) {
    // Execute all the runs sequentially.
    const auto block_size = std::min(max_block_size, num_items);
    auto cache = create_cache(0, 1, block_size);
    size_t num_blocks = (num_items + block_size - 1) / block_size;
    for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
      const size_t begin_item_idx = block_idx * block_size;
      const size_t end_item_idx =
          std::min(begin_item_idx + block_size, num_items);
      RETURN_IF_ERROR(run(block_idx, begin_item_idx, end_item_idx, &cache));
    }
    return absl::OkStatus();
  }

  // Find the optimal parameters.
  size_t block_size, num_threads;
  std::tie(block_size, num_threads) = internal::GetConfig(
      num_items, max_num_threads, min_block_size, max_block_size);
  size_t num_blocks = (num_items + block_size - 1) / block_size;

  // Next block to process.
  std::atomic<size_t> next_block_idx{0};

  // Aggregated failure status of the individual jobs.
  Mutex global_status_mutex;
  absl::Status global_status GUARDED_BY(global_status_mutex);

  // Does global_status contains a failure?
  std::atomic<bool> has_failure{false};

  // The main thread loop.
  const auto thread_loop = [&, block_size, num_threads, num_blocks,
                            num_items](int thread_idx) {
    // Allocate the cache.
    Cache cache = create_cache(thread_idx, num_threads, block_size);
    while (true) {
      // Stop if any of the other job has failed.
      if (has_failure) {
        break;
      }
      // Get a new job.
      size_t block_idx = next_block_idx++;
      if (block_idx >= num_blocks) {
        break;
      }
      // Execute job.
      const size_t begin_item_idx = block_idx * block_size;
      const size_t end_item_idx =
          std::min(begin_item_idx + block_size, num_items);
      const auto status = run(block_idx, begin_item_idx, end_item_idx, &cache);
      // Record job status.
      if (!status.ok()) {
        MutexLock l(&global_status_mutex);
        global_status.Update(status);
        has_failure = true;
      }
    }
  };

  // Initialize cache and start threads.
  std::vector<Thread> threads;
  threads.reserve(num_threads);
  for (size_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    threads.emplace_back([&, thread_idx]() { thread_loop(thread_idx); });
  }
  // Wait for the computation to be done.
  for (auto& thread : threads) {
    thread.Join();
  }
  return global_status;
}

}  // namespace yggdrasil_decision_forests::utils::concurrency

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_H_
