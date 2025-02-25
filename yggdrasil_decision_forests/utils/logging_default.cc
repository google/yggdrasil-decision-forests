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

#include "yggdrasil_decision_forests/utils/logging_default.h"

#include "absl/base/log_severity.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/flags.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#ifndef YDF_NO_ALSOLOGTOSTDERR_FLAG
ABSL_FLAG(bool, alsologtostderr, false, "Log all messages to stderr");
#endif

namespace yggdrasil_decision_forests::logging {
void InitLoggingLib() {
  absl::InitializeLog();
  // By default, absl is configured to:
  //    minloglevel = info
  //    stderrthreshold = error
  // So, no LOG(INFO) is visible.
}
}  // namespace yggdrasil_decision_forests::logging

void InitLogging(const char* usage, int* argc, char*** argv,
                 bool remove_flags) {
  absl::InitializeLog();
  absl::SetProgramUsageMessage(usage);
  absl::ParseCommandLine(*argc, *argv);
#ifndef YDF_NO_ALSOLOGTOSTDERR_FLAG
  if (absl::GetFlag(FLAGS_alsologtostderr)) {
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
    absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
  }
#endif
}
