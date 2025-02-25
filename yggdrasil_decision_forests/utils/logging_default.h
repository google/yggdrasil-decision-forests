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

// Logging options.
// IWYU pragma: private, include "yggdrasil_decision_forests/utils/logging.h"
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_LOGGING_DEFAULT_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_LOGGING_DEFAULT_H_

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
// IWYU pragma: begin_exports
#include "absl/log/check.h"
#include "absl/log/log.h"
// IWYU pragma: end_exports

// Initialize the logging. Should be called when the program starts.
void InitLogging(const char* usage, int* argc, char*** argv, bool remove_flags);

namespace yggdrasil_decision_forests {
namespace logging {

// Initialize logging for a library.
void InitLoggingLib();

// Sets the amount of logging:
// 0: Only fatal i.e. before a crash of the program.
// 1: Only warning and fatal.
// 2: Info, warning and fatal i.e. all logging. Default.
inline void SetLoggingLevel(int level) {
  absl::LogSeverityAtLeast absl_level;
  switch (level) {
    case 0:
      absl_level = absl::LogSeverityAtLeast::kFatal;
      break;
    case 1:
      absl_level = absl::LogSeverityAtLeast::kWarning;
      break;
    default:
      absl_level = absl::LogSeverityAtLeast::kInfo;
      break;
  }
  absl::log_internal::RawSetStderrThreshold(absl_level);
}

// If true, logging messages include the timestamp, filename and line. True by
// default.
inline void SetDetails(bool value) {
  absl::log_internal::RawEnableLogPrefix(value);
}

}  // namespace logging
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_LOGGING_DEFAULT_H_
