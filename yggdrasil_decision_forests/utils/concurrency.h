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

#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

#include "yggdrasil_decision_forests/utils/concurrency_default.h"

#include "yggdrasil_decision_forests/utils/concurrency_channel.h"
#include "yggdrasil_decision_forests/utils/concurrency_streamprocessor.h"

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_H_
