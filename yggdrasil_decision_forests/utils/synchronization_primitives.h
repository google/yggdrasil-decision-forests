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

// Synchronization primitives.
//
// Depending on the preprocessor definitions, used either std or absl
// implementations.
//
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_SYNCHRONIZATION_PRIMITIVES_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_SYNCHRONIZATION_PRIMITIVES_H_

#ifdef YGG_STD_MUTEX
#include <condition_variable>  // c++11
#include <mutex>               // c++11
#include <shared_mutex>        // c++14 and c++17
#else
#include "absl/synchronization/mutex.h"
#endif

#ifdef YGG_STD_BLOCKING_COUNTER_AND_NOTIFICATION
#include <latch>  // c++20
#else
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/notification.h"
#endif

namespace yggdrasil_decision_forests {
namespace utils {
namespace concurrency {

#ifdef YGG_STD_MUTEX

#if _LIBCPP_STD_VER >= 17
// shared_mutex was introduced in c++17.
using std_shared_mutex = std::shared_mutex;
#else
using std_shared_mutex = std::shared_timed_mutex;
#endif

class Mutex {
 public:
  std::mutex& std() { return mu_; }

 private:
  std::mutex mu_;
};

class SharedMutex {
 public:
  std_shared_mutex& std() { return mu_; }

 private:
  std_shared_mutex mu_;
};

class MutexLock {
 public:
  MutexLock(Mutex* mutex) : lock_(mutex->std()) {}
  std::unique_lock<std::mutex>& std() { return lock_; }

 private:
  std::unique_lock<std::mutex> lock_;
};

class WriterMutexLock {
 public:
  WriterMutexLock(SharedMutex* mutex) : lock_(mutex->std()) {}
  std::unique_lock<std_shared_mutex>& std() { return lock_; }

 private:
  std::unique_lock<std_shared_mutex> lock_;
};

class ReaderMutexLock {
 public:
  ReaderMutexLock(SharedMutex* mutex) : lock_(mutex->std()) {}
  std::shared_lock<std_shared_mutex>& std() { return lock_; }

 private:
  std::shared_lock<std_shared_mutex> lock_;
};

class CondVar {
 public:
  void Signal() { cv_.notify_one(); }
  void SignalAll() { cv_.notify_all(); }
  void Wait(Mutex* mutex, MutexLock* lock) { cv_.wait(lock->std()); }
  void WaitWithTimeout(Mutex* mutex, MutexLock* lock, double timeout_seconds) {
    cv_.wait_for(lock->std(), std::chrono::duration<double>(timeout_seconds));
  }

 private:
  std::condition_variable cv_;
};

#else

using Mutex = absl::Mutex;
using MutexLock = absl::MutexLock;
using WriterMutexLock = absl::WriterMutexLock;
using ReaderMutexLock = absl::ReaderMutexLock;
using SharedMutex = Mutex;

class CondVar {
 public:
  void Signal() { cv_.Signal(); }
  void SignalAll() { cv_.SignalAll(); }
  void Wait(Mutex* mutex, MutexLock* lock) { cv_.Wait(mutex); }
  void WaitWithTimeout(Mutex* mutex, MutexLock* lock, double timeout_seconds) {
    cv_.WaitWithTimeout(mutex, absl::Seconds(timeout_seconds));
  }

 private:
  absl::CondVar cv_;
};

#ifndef GUARDED_BY
#define GUARDED_BY(x) ABSL_GUARDED_BY(x)
#define LOCKS_EXCLUDED(x) ABSL_LOCKS_EXCLUDED(x)
#define GUARDED_BY(x) ABSL_GUARDED_BY(x)
#define EXCLUSIVE_LOCKS_REQUIRED(x) ABSL_EXCLUSIVE_LOCKS_REQUIRED(x)
#endif

#define GLOBAL_MUTEX(x) \
  ABSL_CONST_INIT utils::concurrency::Mutex x(absl::kConstInit);

#endif

#ifdef YGG_STD_BLOCKING_COUNTER_AND_NOTIFICATION
// C++20
class BlockingCounter {
 public:
  BlockingCounter(int count) : latch_(count) {}
  void DecrementCount() { latch_.count_down(); }
  void Wait() { latch_.wait(); }

 private:
  std::latch latch_;
};
// using BlockingCounter = absl::BlockingCounter;

// C++20
class Notification {
 public:
  Notification() : latch_(1) {}
  void Notify() { latch_.count_down(); }
  void WaitForNotification() { latch_.wait(); }
  bool HasBeenNotified() { return latch_.try_wait(); }

 private:
  std::latch latch_;
};
// using Notification = absl::Notification;

#define GUARDED_BY(x)
#define LOCKS_EXCLUDED(x)
#define GUARDED_BY(x)
#define EXCLUSIVE_LOCKS_REQUIRED(x)
#define GLOBAL_MUTEX(x) utils::concurrency::Mutex x
#else
using BlockingCounter = absl::BlockingCounter;
using Notification = absl::Notification;
#endif

}  // namespace concurrency
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_SYNCHRONIZATION_PRIMITIVES_H_
