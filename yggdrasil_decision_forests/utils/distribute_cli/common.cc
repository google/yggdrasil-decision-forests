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

#include "yggdrasil_decision_forests/utils/distribute_cli/common.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/hash.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace distribute_cli {

const char kWorkerKey[] = "DISTRIBUTE_CLI";

std::string CommandToInternalCommandId(const absl::string_view command) {
  const auto int128_hash = utils::hash::HashStringViewToUint128(command);
  return absl::StrCat(absl::Uint128Low64(int128_hash), "-",
                      absl::Uint128High64(int128_hash));
}

void BaseOutput(const absl::string_view log_dir,
                const absl::string_view internal_command_id,
                std::string* output_dir, std::string* output_base_filename) {
  // Number of characters in each intermediate subdirectory.
  constexpr int sub_directory_name_length = 3;
  *output_dir = std::string(log_dir);
  int idx = 0;
  while (idx + sub_directory_name_length < internal_command_id.size()) {
    *output_dir = file::JoinPath(
        *output_dir,
        internal_command_id.substr(idx, sub_directory_name_length));
    idx += sub_directory_name_length;
  }
  *output_base_filename = std::string(internal_command_id.substr(idx));
  DCHECK_GT(output_base_filename->size(), 0);
}

}  // namespace distribute_cli
}  // namespace utils
}  // namespace yggdrasil_decision_forests
