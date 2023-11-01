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

// Utility to manage snapshot indices from a process that can be interrupted at
// any time.
//
// Usage example:
//   AddSnapshot("/tmp/snap",5);
//   AddSnapshot("/tmp/snap",11);
//   AddSnapshot("/tmp/snap",6);
//   GetGreatestSnapshot("/tmp/snap") // Returns 11.
//   GetSnapshots("/tmp/snap") // Returns {5,6,11}.
//
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_SNAPSHOT_IO_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_SNAPSHOT_IO_H_

#include <deque>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace yggdrasil_decision_forests::utils {

// Adds a snapshot record. The content of the directory is used to manage
// snapshot tracking.
absl::Status AddSnapshot(absl::string_view directory, int index);

// Retrieves the sorted list of snapshot indexes.
absl::StatusOr<std::deque<int>> GetSnapshots(absl::string_view directory);

// Retrieves the largest snapshot index. Returns an error if not snapshot
// records are available.
absl::StatusOr<int> GetGreatestSnapshot(absl::string_view directory);

// Removes snapshots until there is only "keep" snapshots left. Returns the list
// of removed snapshots.
std::vector<int> RemoveOldSnapshots(absl::string_view directory, int keep,
                                    std::deque<int>& snapshots);

}  // namespace yggdrasil_decision_forests::utils

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_SNAPSHOT_IO_H_
