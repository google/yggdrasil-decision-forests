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

// In process, non-parallelized implementation. For debugging and pipeline
// development.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_SINGLE_THREAD_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_SINGLE_THREAD_H_

#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"

namespace yggdrasil_decision_forests {
namespace distribute {

class SingleThreadManager : public AbstractManager {
 public:
  static constexpr char kKey[] = "SINGLE_THREAD";

  utils::StatusOr<Blob> BlockingRequest(Blob blob, int worker_idx) override;

  absl::Status AsynchronousRequest(Blob blob, int worker_idx) override;

  utils::StatusOr<Blob> NextAsynchronousAnswer() override;

  int NumWorkers() override;

  absl::Status Done(absl::optional<bool> kill_worker_manager) override;

 private:
  absl::Status Initialize(const proto::Config& config,
                          const absl::string_view worker_name,
                          Blob welcome_blob) override;

  bool verbose_ = true;
  std::vector<std::unique_ptr<AbstractWorker>> workers_;

  // Next worker that will solve the next request.
  std::atomic<int> next_worker_ = 0;

  utils::concurrency::Channel<Blob> async_pending_answers_;
};

REGISTER_Distribution_Manager(SingleThreadManager, SingleThreadManager::kKey);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_SINGLE_THREAD_H_
