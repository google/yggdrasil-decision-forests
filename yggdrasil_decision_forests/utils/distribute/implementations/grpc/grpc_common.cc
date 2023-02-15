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

#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_common.h"

#include "absl/strings/match.h"

namespace yggdrasil_decision_forests {
namespace distribute {

bool IsTransientError(const grpc::Status& status) {
  return (status.error_message() == "Socket closed" ||
          status.error_message() == "Transport closed" ||
          status.error_message() == "Connection reset by peer" ||
          status.error_message() == "Broken pipe" ||
          status.error_message() == "keepalive watchdog timeout" ||
          absl::StartsWith(status.error_message(),
                           "failed to connect to all addresses"));
}

void ConfigureClientContext(grpc::ClientContext* context) {
  // Use default context creation.
  //
  // Do not set wait_for_ready=true or long deadline as fast connection failure
  // is necessary for on-the-fly change of worker address.
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
