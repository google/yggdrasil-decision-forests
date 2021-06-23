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

// Utility to manage snapshot indices from a process that can be interrupted at
// any time.
//
// Usage example:
//   AddSnapshot("/tmp/snap",5);
//   AddSnapshot("/tmp/snap",11);
//   AddSnapshot("/tmp/snap",6);
//   GetGreatestSnapshot("/tmp/snap") // Returns 11.
//
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_SNAPSHOT_IO_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_SNAPSHOT_IO_H_

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Adds a snapshot record. The content of the directory is used to manage
// snapshot tracking.
absl::Status AddSnapshot(absl::string_view directory, int index);

// Retrieves the largest snapshot index. Returns an error if not snapshot
// records are available.
utils::StatusOr<int> GetGreatestSnapshot(absl::string_view directory);

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_SNAPSHOT_IO_H_
