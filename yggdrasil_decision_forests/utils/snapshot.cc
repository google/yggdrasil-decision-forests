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

#include "yggdrasil_decision_forests/utils/snapshot.h"

#include <algorithm>
#include <deque>
#include <regex>  // NOLINT
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {
constexpr char kBaseFileName[] = "snapshot_";

std::string SnapshotPath(const absl::string_view directory, const int index) {
  return file::JoinPath(directory, absl::StrCat(kBaseFileName, index));
}

}  // namespace

absl::Status AddSnapshot(const absl::string_view directory, const int index) {
  RETURN_IF_ERROR(file::RecursivelyCreateDir(directory, file::Defaults()));
  return file::SetContent(SnapshotPath(directory, index), "");
}

absl::StatusOr<std::deque<int>> GetSnapshots(
    const absl::string_view directory) {
  std::vector<std::string> results;
  RETURN_IF_ERROR(
      file::Match(file::JoinPath(directory, absl::StrCat(kBaseFileName, "*")),
                  &results, file::Defaults()));
  std::deque<int> snapshot_idxs;
  std::regex pattern(absl::StrCat(".*", kBaseFileName, "([0-9]+)"));
  for (const std::string& result : results) {
    std::smatch match;
    if (!std::regex_match(result, match, pattern)) {
      continue;
    }
    int index;
    if (absl::SimpleAtoi(match[1].str(), &index)) {
      snapshot_idxs.push_back(index);
    }
  }
  absl::c_sort(snapshot_idxs);
  return snapshot_idxs;
}

absl::StatusOr<int> GetGreatestSnapshot(const absl::string_view directory) {
  ASSIGN_OR_RETURN(const std::deque<int> snapshots, GetSnapshots(directory));
  if (snapshots.empty()) {
    return absl::NotFoundError("No snapshots");
  }
  return snapshots.back();
}

std::vector<int> RemoveOldSnapshots(const absl::string_view directory,
                                    const int keep,
                                    std::deque<int>& snapshots) {
  std::vector<int> removed_idxs;
  while (snapshots.size() > keep) {
    const int removed_idx = snapshots.front();
    snapshots.pop_front();
    removed_idxs.push_back(removed_idx);

    // Try to remove the snapshot file.
    file::RecursivelyDelete(SnapshotPath(directory, removed_idx),
                            file::Defaults())
        .IgnoreError();
  }
  return removed_idxs;
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
