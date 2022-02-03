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

#ifndef THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_CLI_WORKER_H_
#define THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_CLI_WORKER_H_

#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.pb.h"
#include "yggdrasil_decision_forests/utils/distribute_cli/common.h"
#include "yggdrasil_decision_forests/utils/distribute_cli/distribute_cli.pb.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace distribute_cli {

class Worker : public distribute::AbstractWorker {
 public:
  absl::Status Setup(distribute::Blob serialized_welcome) override;

  utils::StatusOr<distribute::Blob> RunRequest(
      distribute::Blob serialized_request) override;

  absl::Status Done() override {
    done_was_called_ = true;
    return absl::OkStatus();
  }

 private:
  absl::Status Command(const proto::Request::Command& request,
                       proto::Result::Command* result);

  absl::Status RunCommand(const absl::string_view command,
                          const absl::string_view log_path);

  proto::Welcome welcome_;

  // Set to true when Done is called on the "GenericWorker".
  std::atomic<bool> done_was_called_ = {false};
};

}  // namespace distribute_cli
}  // namespace utils

namespace distribute {
using DistributeCLIWorker = utils::distribute_cli::Worker;
REGISTER_Distribution_Worker(DistributeCLIWorker,
                             utils::distribute_cli::kWorkerKey);
}  // namespace distribute

}  // namespace yggdrasil_decision_forests

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_CLI_WORKER_H_
