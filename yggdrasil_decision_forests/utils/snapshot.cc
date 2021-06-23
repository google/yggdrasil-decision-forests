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

#include "yggdrasil_decision_forests/utils/snapshot.h"

#include <regex>  // NOLINT

#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {
constexpr char kBaseFileName[] = "snapshot_";
}

absl::Status AddSnapshot(absl::string_view directory, int index) {
  RETURN_IF_ERROR(file::RecursivelyCreateDir(directory, file::Defaults()));
  return file::SetContent(
      file::JoinPath(directory, absl::StrCat(kBaseFileName, index)), "");
}

utils::StatusOr<int> GetGreatestSnapshot(absl::string_view directory) {
  std::vector<std::string> results;
  RETURN_IF_ERROR(
      file::Match(file::JoinPath(directory, absl::StrCat(kBaseFileName, "*")),
                  &results, file::Defaults()));
  if (results.empty()) {
    return absl::NotFoundError("No snapshots");
  }
  int greatest_index = 0;
  bool found_match = false;
  std::regex pattern(absl::StrCat(".*", kBaseFileName, "([0-9]+)"));
  for (const auto& result : results) {
    std::smatch match;
    if (!std::regex_match(result, match, pattern)) {
      continue;
    }
    int index;
    if (!absl::SimpleAtoi(match[1].str(), &index)) {
      continue;
    }
    if (!found_match) {
      greatest_index = index;
      found_match = true;
    } else {
      greatest_index = std::max(greatest_index, index);
    }
  }
  if (!found_match) {
    return absl::NotFoundError("No snapshots");
  }
  return greatest_index;
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
