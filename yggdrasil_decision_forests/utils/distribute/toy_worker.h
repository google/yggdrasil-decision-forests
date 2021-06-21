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

#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/logging.h"

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
      std::pair<std::string, std::string> items = absl::StrSplit(blob, ":");
      return items.second;
    } else if (blob == "gen_error") {
      return absl::InvalidArgumentError("Some error");
    } else if (blob == "worker_idx") {
      return absl::StrCat(WorkerIdx());
    }
    return absl::InvalidArgumentError("Unknown task");
  }
};

constexpr char kToyWorkerKey[] = "ToyWorker";
REGISTER_Distribution_Worker(ToyWorker, kToyWorkerKey);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif
