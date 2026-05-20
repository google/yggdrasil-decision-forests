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

#include "yggdrasil_decision_forests/utils/concurrency.h"

#include <stddef.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <utility>

#include "yggdrasil_decision_forests/utils/logging.h"  // IWYU pragma: keep
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests::utils::concurrency {

void ConcurrentForLoop(
    const size_t num_blocks, ThreadPool* thread_pool, const size_t num_items,
    const std::function<void(size_t block_idx, size_t begin_item_idx,
                             size_t end_item_idx)>& function) {
  DCHECK(thread_pool != nullptr);
  if (num_blocks <= 1) {
    function(0, 0, num_items);
    return;
  }
  const size_t effective_num_blocks = std::min(num_blocks, num_items);
  BlockingCounter blocker(effective_num_blocks);
  size_t begin_idx = 0;
  const size_t block_size =
      (num_items + effective_num_blocks - 1) / effective_num_blocks;
  for (size_t block_idx = 0; block_idx < effective_num_blocks; block_idx++) {
    const auto end_idx = std::min(begin_idx + block_size, num_items);
    if (begin_idx <= end_idx) {
      thread_pool->Schedule(
          [block_idx, begin_idx, end_idx, &blocker, &function]() -> void {
            function(block_idx, begin_idx, end_idx);
            blocker.DecrementCount();
          });
      begin_idx += block_size;
    } else {
      blocker.DecrementCount();
    }
  }
  blocker.Wait();
}

absl::Status ConcurrentForLoopWithStatus(
    size_t num_blocks, ThreadPool* thread_pool, size_t num_items,
    const std::function<absl::Status(size_t block_idx, size_t begin_item_idx,
                                     size_t end_item_idx)>& function) {
  DCHECK(thread_pool != nullptr);
  const size_t effective_num_blocks = std::min(num_blocks, num_items);

  if (effective_num_blocks <= 1) {
    return function(0, 0, num_items);
  }
  BlockingCounter blocker(effective_num_blocks);
  size_t begin_idx = 0;
  const size_t block_size =
      (num_items + effective_num_blocks - 1) / effective_num_blocks;

  Mutex status_mutex;
  absl::Status status GUARDED_BY(status_mutex);
  std::atomic<bool> has_failure{false};

  for (size_t block_idx = 0; block_idx < effective_num_blocks; block_idx++) {
    const auto end_idx = std::min(begin_idx + block_size, num_items);
    if (begin_idx <= end_idx) {
      thread_pool->Schedule([block_idx, begin_idx, end_idx, &blocker, &function,
                             &status_mutex, &status, &has_failure]() -> void {
        if (!has_failure.load(std::memory_order_relaxed)) {
          const auto sub_status = function(block_idx, begin_idx, end_idx);
          if (!sub_status.ok()) {
            MutexLock l(status_mutex);
            status.Update(sub_status);
            has_failure.store(true, std::memory_order_relaxed);
          }
        }
        blocker.DecrementCount();
      });
      begin_idx += block_size;
    } else {
      blocker.DecrementCount();
    }
  }
  blocker.Wait();
  return status;
}

namespace internal {

// Computes the optimal block size and number of threads.
std::pair<size_t, size_t> GetConfig(size_t num_items, size_t max_num_threads,
                                    size_t min_block_size,
                                    size_t max_block_size) {
  size_t block_size = (num_items + max_num_threads - 1) / max_num_threads;
  if (block_size < min_block_size) {
    block_size = min_block_size;
  } else if (block_size > max_block_size) {
    block_size = max_block_size;
  }

  size_t num_threads = (num_items + block_size - 1) / block_size;
  if (num_threads < 1) {
    num_threads = 1;
  } else if (num_threads > max_num_threads) {
    num_threads = max_num_threads;
  }
  return std::make_pair(block_size, num_threads);
}

}  // namespace internal
}  // namespace yggdrasil_decision_forests::utils::concurrency
