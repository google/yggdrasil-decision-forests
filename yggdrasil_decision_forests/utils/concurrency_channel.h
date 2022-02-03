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

// A synchronized queue with a bocking "Pop" operation. Does not  implements a
// maximum capacity.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_CHANNEL_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_CHANNEL_H_

#include <memory>
#include <queue>
#include <string>

#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace concurrency {

// Thread safe channel. The channel does not have a maximum capacity i.e. the
// input is not blocking.
template <typename Input>
class Channel {
 public:
  // Close the channel. No new items can be push in the channel.
  void Close() {
    MutexLock results_lock(&mutex_);
    close_channel_ = true;
    cond_var_.SignalAll();
  }

  // Clear the content of a channel.
  void Clear() {
    MutexLock results_lock(&mutex_);
    std::queue<Input>().swap(content_);
  }

  // Re-open a previously closed channel.
  void Reopen() {
    MutexLock results_lock(&mutex_);
    close_channel_ = false;
    cond_var_.SignalAll();
  }

  // Push an item in the channel.
  void Push(Input item) {
    if (close_channel_) {
      LOG(ERROR) << "Ignoring value added to closed channel.";
      return;
    }
    MutexLock results_lock(&mutex_);
    content_.push(std::move(item));
    cond_var_.Signal();
  }

  // Pops a value from the channel. If the channel is closed and empty, returns
  // {}. If the channel is empty but not closed, blocks. If the channel is not
  // empty, returns the first added element.
  //
  // If "num_pop" is specified, the pointed value will be set the number of
  // "Pop" results returned so far. The number of pops returned by
  // "PopAndNumPop" is guaranteed to be unique, dense and in order.
  absl::optional<Input> Pop(int64_t* get_num_pop = nullptr) {
    MutexLock results_lock(&mutex_);
    while (content_.empty() && !close_channel_) {
      cond_var_.Wait(&mutex_, &results_lock);
    }
    if (close_channel_ && content_.empty()) {
      return {};
    }
    Input input{std::move(content_.front())};
    content_.pop();
    if (get_num_pop) {
      *get_num_pop = num_pops_;
    }
    num_pops_++;
    return std::move(input);
  }

 private:
  std::queue<Input> content_ GUARDED_BY(mutex_);
  bool close_channel_ GUARDED_BY(mutex_) = false;
  uint64_t num_pops_ GUARDED_BY(mutex_) = 0;
  CondVar cond_var_;
  Mutex mutex_;
};

}  // namespace concurrency
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_CHANNEL_H_
