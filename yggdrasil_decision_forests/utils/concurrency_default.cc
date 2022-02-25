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

#include "yggdrasil_decision_forests/utils/concurrency_default.h"

#include <iostream>

namespace yggdrasil_decision_forests {
namespace utils {
namespace concurrency {

Thread::Thread(std::function<void(void)> call) : thread_(std::move(call)) {}

void Thread::Join() { thread_.join(); }

ThreadPool::ThreadPool(std::string name, int num_threads)
    : name_(std::move(name)), num_threads_(num_threads) {}

ThreadPool::~ThreadPool() { JoinAllAndStopThreads(); }

void ThreadPool::JoinAllAndStopThreads() {
  if (num_threads_ == 0) {
    return;
  }
  jobs_.Close();
  for (auto& thread : threads_) {
    thread.join();
  }
  threads_.clear();
}

void ThreadPool::StartWorkers() {
  while (threads_.size() < num_threads_) {
    threads_.emplace_back(&ThreadPool::ThreadLoop, this);
  }
}

void ThreadPool::Schedule(std::function<void()> callback) {
  if (num_threads_ == 0) {
    callback();
  } else {
    jobs_.Push(std::move(callback));
  }
}

void ThreadPool::ThreadLoop() {
  while (true) {
    // Get a job.
    auto optional_input = jobs_.Pop();
    if (!optional_input.has_value()) {
      break;
    }

    // Run the job.
    std::move(optional_input).value()();
  }
}

}  // namespace concurrency
}  // namespace utils
}  // namespace yggdrasil_decision_forests
