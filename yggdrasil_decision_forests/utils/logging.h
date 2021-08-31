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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_LOGGING_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_LOGGING_H_

#include "yggdrasil_decision_forests/utils/logging_default.h"

// Prints something every INTERVAL seconds.
//
// Usage example:
//   LOG_INFO_EVERY_N_SEC(5, _ << "Hello world");
//
#define LOG_INFO_EVERY_N_SEC(INTERVAL, MESSAGE)               \
  {                                                           \
    static std::atomic<int64_t> next_log_time_atomic{0};      \
    const auto now_time = absl::GetCurrentTimeNanos();        \
    const auto next_log_time =                                \
        next_log_time_atomic.load(std::memory_order_relaxed); \
    if (now_time > next_log_time) {                           \
      next_log_time_atomic.store(now_time + INTERVAL * 1e9,   \
                                 std::memory_order_relaxed);  \
      std::string _;                                          \
      LOG(INFO) << MESSAGE;                                   \
    }                                                         \
  }

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_LOGGING_H_
