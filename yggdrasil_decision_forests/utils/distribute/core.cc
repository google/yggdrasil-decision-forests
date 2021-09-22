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

#include "yggdrasil_decision_forests/utils/distribute/core.h"

namespace yggdrasil_decision_forests {
namespace distribute {

AbstractWorkerHook AbstractWorker::default_hook_;

absl::Status InternalInitializeWorker(
    const int worker_idx, const int num_workers, AbstractWorker* worker,
    AbstractWorkerHook* worker_implementation) {
  return worker->InternalInitialize(worker_idx, num_workers,
                                    worker_implementation);
}

absl::Status AbstractWorker::InternalInitialize(
    int worker_idx, const int num_workers,
    AbstractWorkerHook* worker_implementation) {
  worker_idx_ = worker_idx;
  num_workers_ = num_workers;
  if (worker_implementation) {
    hook_ = worker_implementation;
  }
  return absl::OkStatus();
}

absl::Status AbstractWorker::AsynchronousRequestToOtherWorker(Blob blob,
                                                              int worker_idx) {
  return hook_->AsynchronousRequestToOtherWorker(blob, worker_idx, this);
}

utils::StatusOr<Blob> AbstractWorker::NextAsynchronousAnswerFromOtherWorker() {
  return hook_->NextAsynchronousAnswerFromOtherWorker(this);
}

absl::Status AbstractWorkerHook::AsynchronousRequestToOtherWorker(
    Blob blob, int target_worker_idx, AbstractWorker* emitter_worker) {
  return absl::InternalError(
      "AsynchronousRequestToOtherWorker Not implemented");
}

utils::StatusOr<Blob> AbstractWorkerHook::NextAsynchronousAnswerFromOtherWorker(
    AbstractWorker* emitter_worker) {
  return absl::InternalError(
      "NextAsynchronousAnswerFromOtherWorker Not implemented");
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
