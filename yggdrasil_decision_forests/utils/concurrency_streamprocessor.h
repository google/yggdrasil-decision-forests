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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_COMMON_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_COMMON_H_

#include <memory>
#include <queue>
#include <string>


#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/concurrency_channel.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace concurrency {

template <typename Input, typename Output>
class StreamProcessor {
 public:
  // Creates the processor. Don't start the threads yet.
  //
  // Args:
  //   name: Name of the processor. For debug only.
  //   num_threads: Number of running threads.
  //   call: Job to be executed on each query.
  //   result_in_order: If false (default), the results will be returned in
  //     random order (depending on the job scheduling). If true, the results
  //     will be returned in the same order as the queries. A pending result
  //     will block the thread that executed it.
  StreamProcessor(std::string name, int num_threads,
                  std::function<Output(Input)> call,
                  bool result_in_order = false);

  ~StreamProcessor();

  // Starts the threads.
  void StartWorkers();

  // Adds a new job.
  void Submit(Input input);

  // Get the result of a job. Returns {} if the "JoinAllAndStopThreads" or
  // "CloseSubmits" has been called and all outputs have already been retrieved.
  absl::optional<Output> GetResult();

  // Indicates that no more request can be submitted.
  void CloseSubmits();

  // Ensure all the jobs are done and all the threads have been joined.
  // Called by the destructor.
  void JoinAllAndStopThreads();

 private:
  // Running loop for the threads.
  void ThreadLoop();

  // Number of threads.
  int num_threads_;

  // Name of the pool.
  std::string name_;

  // Active threads.
  std::vector<Thread> threads_;

  // Processing function.
  std::function<Output(Input)> call_;

  Channel<Input> input_channel_;
  Channel<Output> output_channel_;

  // Should the results be returned in order.
  const bool result_in_order_;
  // Id of the next query to return with "GetResult" if result_in_order_=true.
  int64_t next_query_id GUARDED_BY(mutex_) = 0;
  // Cond variable for the threads returning in order results.
  CondVar cond_var_;
  // Number of threads still running.
  int num_active_threads_ GUARDED_BY(mutex_) = 0;

  Mutex mutex_;
};

template <typename Input, typename Output>
StreamProcessor<Input, Output>::StreamProcessor(
    std::string name, int num_threads, std::function<Output(Input)> call,
    bool result_in_order)
    : name_(std::move(name)),
      num_threads_(num_threads),
      call_(std::move(call)),
      result_in_order_(result_in_order) {}

template <typename Input, typename Output>
StreamProcessor<Input, Output>::~StreamProcessor() {
  JoinAllAndStopThreads();
}

template <typename Input, typename Output>
void StreamProcessor<Input, Output>::StartWorkers() {
  {
    MutexLock results_lock(&mutex_);
    num_active_threads_ = num_threads_;
  }
  while (threads_.size() < num_threads_) {
    threads_.emplace_back([this]() mutable { ThreadLoop(); });
  }
}

template <typename Input, typename Output>
void StreamProcessor<Input, Output>::CloseSubmits() {
  input_channel_.Close();
}

template <typename Input, typename Output>
void StreamProcessor<Input, Output>::JoinAllAndStopThreads() {
  CloseSubmits();
  for (auto& thread : threads_) {
    thread.Join();
  }
  output_channel_.Close();
  threads_.clear();
}

template <typename Input, typename Output>
void StreamProcessor<Input, Output>::ThreadLoop() {
  while (true) {
    int64_t query_id;
    auto optional_input = input_channel_.Pop(&query_id);
    if (!optional_input.has_value()) {
      break;
    }

    // Run computation.
    Output result = call_(std::move(optional_input).value());

    if (result_in_order_) {
      // The results should be returned in order.
      MutexLock results_lock(&mutex_);
      while (query_id != next_query_id) {
        // Not my turn yet.
        cond_var_.Wait(&mutex_, &results_lock);
      }

      next_query_id++;
      cond_var_.SignalAll();

      // Note: "Push" is protected by "mutex_".
      output_channel_.Push(std::move(result));
    } else {
      output_channel_.Push(std::move(result));
    }
  }

  MutexLock results_lock(&mutex_);
  num_active_threads_--;
  if (num_active_threads_ == 0) {
    output_channel_.Close();
  }
}

template <typename Input, typename Output>
void StreamProcessor<Input, Output>::Submit(Input input) {
  input_channel_.Push(std::move(input));
}

template <typename Input, typename Output>
absl::optional<Output> StreamProcessor<Input, Output>::GetResult() {
  return output_channel_.Pop();
}

}  // namespace concurrency
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_COMMON_H_
