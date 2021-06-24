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

// Collection of generic workers for simple distributed training algorithms e.g.
// train a model, evaluate a model.
//
// See generic_worker.proto for the list of available works.
//
#ifndef THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_LEARNER_generic_worker_GENERIC_WORKER_H_
#define THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_LEARNER_generic_worker_GENERIC_WORKER_H_

#include "yggdrasil_decision_forests/learner/generic_worker/generic_worker.pb.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.pb.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace generic_worker {
class GenericWorker : public distribute::AbstractWorker {
 public:
  static constexpr char kWorkerKey[] = "GENERIC_WORKER";

  absl::Status Setup(distribute::Blob serialized_welcome) override;

  utils::StatusOr<distribute::Blob> RunRequest(
      distribute::Blob serialized_request) override;

  absl::Status Done() override {
    done_was_called_ = true;
    return absl::OkStatus();
  }

 private:
  absl::Status TrainModel(const proto::Request::TrainModel& request,
                          proto::Result::TrainModel* result);

  absl::Status EvaluateModel(const proto::Request::EvaluateModel& request,
                             proto::Result::EvaluateModel* result);

  proto::Welcome welcome_;

  // Set to true when Done is called on the "GenericWorker".
  std::atomic<bool> done_was_called_ =  {false};
};

}  // namespace generic_worker
}  // namespace model

namespace distribute {
using GenericWorker = model::generic_worker::GenericWorker;
REGISTER_Distribution_Worker(GenericWorker, GenericWorker::kWorkerKey);
}  // namespace distribute

}  // namespace yggdrasil_decision_forests

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_LEARNER_generic_worker_GENERIC_WORKER_H_
