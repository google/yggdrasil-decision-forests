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

#include "yggdrasil_decision_forests/utils/logging_default.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"

namespace yggdrasil_decision_forests {
namespace logging {
int logging_level = 2;
}
}  // namespace yggdrasil_decision_forests

ABSL_FLAG(bool, alsologtostderr, true,
         "If true, stream the logs to std::clog. If false, ignore the logs.");

void InitLogging(const char* usage, int* argc, char*** argv,
                 bool remove_flags) {
  absl::SetProgramUsageMessage(usage);
  absl::ParseCommandLine(*argc, *argv);
}
