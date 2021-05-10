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

#include "yggdrasil_decision_forests/utils/sharded_io.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"

namespace yggdrasil_decision_forests {
namespace utils {

absl::Status ExpandInputShards(const absl::string_view sharded_path,
                               std::vector<std::string>* paths) {
  // Split by ",".
  const std::vector<std::string> level_1 = absl::StrSplit(sharded_path, ',');

  // Sharding.
  std::vector<std::string> level_2;
  for (const auto& level_1_item : level_1) {
    std::vector<std::string> level_1_item_sharded;
    if (file::GenerateShardedFilenames(level_1_item, &level_1_item_sharded)) {
      level_2.insert(level_2.end(), level_1_item_sharded.begin(),
                     level_1_item_sharded.end());
    } else {
      level_2.push_back(level_1_item);
    }
  }

  // Matching.
  std::vector<std::string> level_3;
  for (const auto& level_2_item : level_2) {
    std::vector<std::string> level_2_item_sharded;
    if (file::Match(level_2_item, &level_2_item_sharded, file::Defaults())
            .ok()) {
      level_3.insert(level_3.end(), level_2_item_sharded.begin(),
                     level_2_item_sharded.end());
    } else {
      level_3.push_back(level_2_item);
    }
  }

  *paths = level_3;

  std::sort(paths->begin(), paths->end());
  if (paths->empty()) {
    return absl::NotFoundError(
        absl::StrCat("No files matching: ", sharded_path));
  }
  return absl::OkStatus();
}

absl::Status ExpandOutputShards(const absl::string_view sharded_path,
                                std::vector<std::string>* paths) {
  if (!file::GenerateShardedFilenames(sharded_path, paths)) {
    paths->emplace_back(sharded_path);
  }
  std::sort(paths->begin(), paths->end());
  return absl::OkStatus();
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
