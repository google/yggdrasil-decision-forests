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

// A worker waiting for jobs sent by the GRPC manager.

#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_worker.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(int, port, -1, "Port");
ABSL_FLAG(bool, use_loas, false, "Use LOAS.");

namespace yggdrasil_decision_forests {
namespace distribute {
namespace grpc_worker {

void GRPCWorker() {
  const auto port = absl::GetFlag(FLAGS_port);
  YDF_LOG(INFO) << "Start GRPC worker on port " << port;
  QCHECK_OK(WorkerMain(port, absl::GetFlag(FLAGS_use_loas)));
  YDF_LOG(INFO) << "Stop GRPC worker";
}

}  // namespace grpc_worker
}  // namespace distribute
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(argv[0], &argc, &argv, true);
  yggdrasil_decision_forests::distribute::grpc_worker::GRPCWorker();
  return 0;
}
