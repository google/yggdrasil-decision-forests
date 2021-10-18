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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"

#include <limits>

#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace dataset_cache {

std::string ColumnPath(absl::string_view directory, int column_idx) {
  return file::JoinPath(directory, absl::StrCat(kFilenameColumn, column_idx));
}

std::string ShardMetadataPath(absl::string_view directory, int shard_idx,
                              int num_shards) {
  return file::JoinPath(
      directory, kFilenameRaw,
      ShardFilename(kFilenameShardMetadataNoUnderscore, shard_idx, num_shards));
}

std::string RawColumnFilePath(absl::string_view directory, int column_idx,
                              int shard_idx, int num_shards) {
  return file::JoinPath(
      RawColumnFileDirectory(directory, column_idx),
      ShardFilename(kFilenameShardNoUnderscore, shard_idx, num_shards));
}

std::string RawColumnFileDirectory(absl::string_view directory,
                                   int column_idx) {
  return file::JoinPath(directory, kFilenameRaw,
                        absl::StrCat(kFilenameColumn, column_idx));
}

uint64_t MaskDeltaBit(uint64_t num_examples) {
  return uint64_t{1} << (64 - utils::CountLeadingZeroes64(num_examples));
}

uint64_t MaskExampleIdx(uint64_t num_examples) {
  return MaskDeltaBit(num_examples) - 1;
}

uint64_t MaxValue(uint64_t num_examples) {
  return MaskExampleIdx(num_examples);
}

}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
