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

#ifndef THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_TOY_WORKER_H_
#define THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_TOY_WORKER_H_

#include "absl/synchronization/barrier.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/logging.h"
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
    return absl::OkStatus();
  }

  absl::Status Done() override {
    LOG(INFO) << "Done worker " << WorkerIdx();
    return absl::OkStatus();
  }

  utils::StatusOr<Blob> RunRequest(Blob blob) override {
    LOG(INFO) << "RunRequest " << blob << " on worker " << WorkerIdx();
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
      absl::SleepFor(absl::Seconds(5));
      return "";
    } else if (absl::StartsWith(blob, "num_existing_toy_workers")) {
      return absl::StrCat(num_existing_toy_workers_.load());
    } else if (absl::StartsWith(blob, "max_num_existing_toy_workers")) {
      return absl::StrCat(max_num_existing_toy_workers_.load());
    }
    return absl::InvalidArgumentError("Unknown task");
  }

 private:
  // Number of "ToyWorker" objects initialized in memory.
  static std::atomic<int> num_existing_toy_workers_;

  // Maximum number of "ToyWorker" objects ever initialized in memory.
  static std::atomic<int> max_num_existing_toy_workers_;

  std::string value_;  // For the "get/set" task.

  absl::Barrier *barrier_ = nullptr;
};

constexpr char kToyWorkerKey[] = "ToyWorker";
REGISTER_Distribution_Worker(ToyWorker, kToyWorkerKey);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif
