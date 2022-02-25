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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_DEFAULT_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_DEFAULT_H_

#include <queue>
#include <string>
#include <thread>

#include "yggdrasil_decision_forests/utils/concurrency_channel.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace concurrency {

class Thread {
 public:
  explicit Thread(std::function<void(void)> call);

  void Join();

 private:
  std::thread thread_;
};

class ThreadPool {
 public:
  // Creates the thread pool. Don't start any thread yet.
  // If "num_threads==0", callback are executed synchronously without threading
  // during "Schedule" calls.
  ThreadPool(std::string name, int num_threads);

  // Ensure all the jobs are done and all the threads have been joined.
  ~ThreadPool();

  // Starts the threads.
  void StartWorkers();

  // Schedules a new job. Returns immediately i.e. does not wait for the job to
  // be executed.
  // If "num_threads==0", execute "callback" synchronously.
  void Schedule(std::function<void()> callback);

  // Number of threads configured in the constructor.
  int num_threads() const { return threads_.size(); }

 private:
  // Ensure all the jobs are done and all the threads have been joined.
  void JoinAllAndStopThreads();

  // Running loop for the threads.
  void ThreadLoop();

  // Name of the pool.
  std::string name_;

  // Number of threads.
  int num_threads_;

  // Active threads.
  std::vector<std::thread> threads_;

  // Scheduled jobs.
  Channel<std::function<void()>> jobs_;
};

}  // namespace concurrency
}  // namespace utils
}  // namespace yggdrasil_decision_forests
#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_CONCURRENCY_DEFAULT_H_
