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

#ifndef THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_TOY_WORKER_H_
#define THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_TOY_WORKER_H_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/barrier.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests {
namespace distribute {

// A toy worker
//
// Tasks:
//   Start with "identity:" -> return the value after "identity:";
//   "gen_error" -> Raise an error.
//   "worker_idx" -> Return the ID of the worker as a string.
//
class ToyWorker final : public AbstractWorker {
 public:
  ToyWorker() {
    num_existing_toy_workers_++;
    max_num_existing_toy_workers_.store(
        std::max(max_num_existing_toy_workers_, num_existing_toy_workers_));
  }

  virtual ~ToyWorker() { num_existing_toy_workers_--; }

  absl::Status Setup(Blob welcome_blob) override {
    LOG(INFO) << "Setup worker " << WorkerIdx();
    CHECK_EQ(welcome_blob, "hello");
    absl::SleepFor(absl::Milliseconds(setup_delay_ms_.load()));
    setup_done_ = true;
    return absl::OkStatus();
  }

  // Duration of the next calls to "Setup" (e.g. a worker loading its dataset).
  static void SetSetupDelay(absl::Duration delay) {
    setup_delay_ms_ = absl::ToInt64Milliseconds(delay);
  }

  absl::Status Done() override {
    LOG(INFO) << "Done worker " << WorkerIdx();
    return absl::OkStatus();
  }

  absl::StatusOr<Blob> RunRequest(Blob blob) override {
    LOG(INFO) << "RunRequest " << blob << " on worker " << WorkerIdx();

    if (forbidden) {
      LOG(FATAL) << "A forbidden worker was called!";
    }

    if (!setup_done_) {
      return absl::FailedPreconditionError("Request before the end of Setup");
    }

    if (absl::StartsWith(blob, "identity")) {
      std::pair<std::string, std::string> items = absl::StrSplit(blob, ':');
      return items.second;
    } else if (blob == "gen_error") {
      return absl::InvalidArgumentError("Some error");
    } else if (blob == "worker_idx") {
      return absl::StrCat(WorkerIdx());
    } else if (blob == "sum_other_worker_idxs") {
      // Request and sum the idx of all the other workers.
      int sum_other_idxs = 0;
      for (int w = 0; w < NumWorkers(); w++) {
        if (w != WorkerIdx()) {
          RETURN_IF_ERROR(AsynchronousRequestToOtherWorker("worker_idx", w));
        }
      }
      for (int w = 0; w < NumWorkers() - 1; w++) {
        ASSIGN_OR_RETURN(const auto str_other_worker_idx,
                         NextAsynchronousAnswerFromOtherWorker());
        int other_worker_idx;
        CHECK(absl::SimpleAtoi(str_other_worker_idx, &other_worker_idx));
        sum_other_idxs += other_worker_idx;
      }
      return absl::StrCat(sum_other_idxs);
    } else if (blob == "blocking_sum_other_worker_idxs") {
      // Same as "sum_other_worker_idxs" with blocking requests, all at the
      // same time (one thread per other worker).
      std::vector<absl::StatusOr<Blob>> answers(NumWorkers(), Blob());
      {
        std::vector<std::unique_ptr<utils::concurrency::Thread>> threads;
        for (int w = 0; w < NumWorkers(); w++) {
          if (w == WorkerIdx()) continue;
          threads.push_back(std::make_unique<utils::concurrency::Thread>(
              [this, w, &answers]() {
                answers[w] = BlockingRequestToOtherWorker("worker_idx", w);
              }));
        }
        for (auto& thread : threads) thread->Join();
      }
      int sum_other_idxs = 0;
      for (int w = 0; w < NumWorkers(); w++) {
        if (w == WorkerIdx()) continue;
        RETURN_IF_ERROR(answers[w].status());
        int other_worker_idx;
        CHECK(absl::SimpleAtoi(*answers[w], &other_worker_idx));
        sum_other_idxs += other_worker_idx;
      }
      return absl::StrCat(sum_other_idxs);
    } else if (blob == "next_worker_blocking_sum") {
      // The "blocking_sum_other_worker_idxs" of the next worker, requested
      // with a blocking request (requests between the workers while
      // answering a request of another worker).
      return BlockingRequestToOtherWorker("blocking_sum_other_worker_idxs",
                                          (WorkerIdx() + 1) % NumWorkers());
    } else if (blob == "create_5_barrier") {
      barrier_ = new absl::Barrier(5);
      return "";
    } else if (blob == "wait_barrier") {
      // Wait and block for 5 calls to this request.
      CHECK(barrier_);
      LOG(INFO) << "Worker " << WorkerIdx()
                << " is waiting for 5 other calls at barrier";
      if (barrier_->Block()) {
        delete barrier_;
        barrier_ = nullptr;
      }
      LOG(INFO) << "Worker #" << WorkerIdx() << " passed the barrier";
      return "";
    } else if (blob == "get") {
      return value_;
    } else if (absl::StartsWith(blob, "set:")) {
      std::pair<std::string, std::string> items = absl::StrSplit(blob, ':');
      value_ = items.second;
      return "";
    } else if (absl::StartsWith(blob, "sleep")) {
      num_sleeps_++;
      absl::SleepFor(absl::Seconds(5));
      return "";
    } else if (blob == "other_worker_sleep") {
      // Asks the next worker to sleep, and returns the status of its answer.
      RETURN_IF_ERROR(AsynchronousRequestToOtherWorker(
          "sleep", (WorkerIdx() + 1) % NumWorkers()));
      const auto answer = NextAsynchronousAnswerFromOtherWorker();
      return answer.ok() ? std::string("ok")
                         : std::string(absl::StatusCodeToString(
                               answer.status().code()));
    } else if (blob == "num_sleeps") {
      return absl::StrCat(num_sleeps_.load());
    } else if (absl::StartsWith(blob, "concurrent_requests:")) {
      // Sends this many "concurrent_sleep" requests to the other workers (in
      // turn) at once, and returns the largest number of them that ran at the
      // same time.
      int num_requests;
      if (!absl::SimpleAtoi(blob.substr(blob.find(':') + 1), &num_requests)) {
        return absl::InvalidArgumentError("Invalid number of requests");
      }
      max_concurrent_sleeps_ = 0;
      for (int i = 0; i < num_requests; i++) {
        const int target =
            (WorkerIdx() + 1 + i % (NumWorkers() - 1)) % NumWorkers();
        RETURN_IF_ERROR(
            AsynchronousRequestToOtherWorker("concurrent_sleep", target));
      }
      for (int i = 0; i < num_requests; i++) {
        RETURN_IF_ERROR(NextAsynchronousAnswerFromOtherWorker().status());
      }
      return absl::StrCat(max_concurrent_sleeps_.load());
    } else if (blob == "concurrent_sleep") {
      // Sleeps 1 second, counting the "concurrent_sleep" requests running at
      // the same time (on all the toy workers).
      const int running = ++concurrent_sleeps_;
      int max_running = max_concurrent_sleeps_.load();
      while (
          running > max_running &&
          !max_concurrent_sleeps_.compare_exchange_weak(max_running, running)) {
      }
      absl::SleepFor(absl::Seconds(1));
      concurrent_sleeps_--;
      return "";
    } else if (blob == "next_worker_idx_or_error") {
      // The index of the next worker, or the error of its answer.
      RETURN_IF_ERROR(AsynchronousRequestToOtherWorker(
          "worker_idx", (WorkerIdx() + 1) % NumWorkers()));
      const auto answer = NextAsynchronousAnswerFromOtherWorker();
      return answer.ok() ? answer.value() : answer.status().ToString();
    } else if (absl::StartsWith(blob, "short_sleep_and_error")) {
      absl::SleepFor(absl::Seconds(2));
      return absl::InvalidArgumentError("Some error");
    } else if (absl::StartsWith(blob, "num_existing_toy_workers")) {
      return absl::StrCat(num_existing_toy_workers_.load());
    } else if (absl::StartsWith(blob, "max_num_existing_toy_workers")) {
      return absl::StrCat(max_num_existing_toy_workers_.load());
    } else if (blob == "forbidden") {
      LOG(INFO) << "Set the worker as forbidden";
      forbidden = true;
      return "";
    }
    return absl::InvalidArgumentError("Unknown task");
  }

 private:
  // Number of "ToyWorker" objects initialized in memory.
  static std::atomic<int> num_existing_toy_workers_;

  // Maximum number of "ToyWorker" objects ever initialized in memory.
  static std::atomic<int> max_num_existing_toy_workers_;
  // Number of "sleep" requests run, by all the toy workers.
  static std::atomic<int> num_sleeps_;
  // Duration of "Setup", in milliseconds.
  static std::atomic<int> setup_delay_ms_;
  // "concurrent_sleep" requests running, and their maximum since the last
  // "concurrent_requests" (all the toy workers).
  static std::atomic<int> concurrent_sleeps_;
  static std::atomic<int> max_concurrent_sleeps_;

  // "Setup" is done.
  std::atomic<bool> setup_done_{false};

  std::string value_;  // For the "get/set" task.

  absl::Barrier* barrier_ = nullptr;

  std::atomic<bool> forbidden{false};
};

constexpr char kToyWorkerKey[] = "ToyWorker";
REGISTER_Distribution_Worker(ToyWorker, kToyWorkerKey);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif
