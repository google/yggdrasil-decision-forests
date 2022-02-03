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

#ifndef THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_CLI_COMMON_H_
#define THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_CLI_COMMON_H_

#include <string>

#include "absl/strings/string_view.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace distribute_cli {

// Registered name of the worker.
extern const char kWorkerKey[];

// Unique identifier for the command i.e. hash.
// Used as filename for the log and execution status files.
std::string CommandToInternalCommandId(const absl::string_view command);

}  // namespace distribute_cli
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_CLI_COMMON_H_
