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

#include <functional>

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

}  // namespace yggdrasil_decision_forests::utils::concurrency

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_H_
