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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_UTILS_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_UTILS_H_

#include <functional>

#include "yggdrasil_decision_forests/utils/concurrency.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace concurrency {

// Utility to apply a function over a range of elements using multi-threading.
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
void ConcurrentForLoop(
    const size_t num_blocks, ThreadPool* thread_pool, const size_t num_items,
    const std::function<void(size_t block_idx, size_t begin_item_idx,
                             size_t end_item_idx)>& function) {
  DCHECK(thread_pool != nullptr);
  if (num_blocks <= 1) {
    function(0, 0, num_items);
    return;
  }
  BlockingCounter blocker(num_blocks);
  size_t begin_idx = 0;
  const size_t block_size = (num_items + num_blocks - 1) / num_blocks;
  for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
    const auto end_idx = std::min(begin_idx + block_size, num_items);
    thread_pool->Schedule(
        [block_idx, begin_idx, end_idx, &blocker, &function]() -> void {
          function(block_idx, begin_idx, end_idx);
          blocker.DecrementCount();
        });
    begin_idx += block_size;
  }
  blocker.Wait();
}

}  // namespace concurrency
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_UTILS_H_