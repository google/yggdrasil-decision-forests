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

std::string PartialRawColumnFilePath(absl::string_view directory,
                                     int column_idx, int shard_idx) {
  return file::JoinPath(PartialRawColumnFileDirectory(directory, column_idx),
                        absl::StrCat(kFilenameShard, shard_idx));
}

std::string RawColumnFileDirectory(absl::string_view directory,
                                   int column_idx) {
  return file::JoinPath(directory, kFilenameRaw,
                        absl::StrCat(kFilenameColumn, column_idx));
}

std::string PartialRawColumnFileDirectory(absl::string_view directory,
                                          int column_idx) {
  return file::JoinPath(directory, kFilenamePartialRaw,
                        absl::StrCat(kFilenameColumn, column_idx));
}

std::string ShardMetadataPath(absl::string_view directory, int shard_idx) {
  return file::JoinPath(
      directory, kFilenameRaw,
      absl::StrCat(kFilenameShard, shard_idx, kFilenameMetaDataPostfix));
}

uint64_t MaskDeltaBit(uint64_t num_examples) {
  return uint64_t{1} << (64 - utils::CountLeadingZeroes64(num_examples));
}

uint64_t MaskExampleIdx(uint64_t num_examples) {
  return MaskDeltaBit(num_examples) - 1;
}

uint64_t MaxValueWithDeltaBit(uint64_t num_examples) {
  return MaskDeltaBit(num_examples) | num_examples;
}

float DiscretizedNumericalToNumerical(
    const std::vector<float>& boundaries,
    const DiscretizedIndexedNumericalType value) {
  DCHECK_GT(boundaries.size(), 0);
  DCHECK_LE(value, boundaries.size());
  if (value == 0) {
    return std::min(
        std::nextafter(boundaries[0], -std::numeric_limits<float>::infinity()),
        boundaries[0] - 1.f);
  }
  if (value == boundaries.size()) {
    return std::max(std::nextafter(boundaries[boundaries.size() - 1],
                                   std::numeric_limits<float>::infinity()),
                    boundaries[boundaries.size() - 1] + 1.f);
  }
  return decision_tree::MidThreshold(boundaries[value - 1], boundaries[value]);
}

DiscretizedIndexedNumericalType NumericalToDiscretizedNumerical(
    const std::vector<float>& boundaries, float value) {
  const auto it = std::upper_bound(boundaries.begin(), boundaries.end(), value);
  return std::distance(boundaries.begin(), it);
}

utils::StatusOr<std::vector<float>>
ExtractDiscretizedBoundariesWithoutDownsampling(
    const std::vector<std::pair<float, model::SignedExampleIdx>>&
        value_and_example_idxs,
    int64_t num_unique_values) {
  // Get the sorted list of unique values.
  std::vector<float> sorted_unique_values;
  sorted_unique_values.reserve(num_unique_values);
  if (!value_and_example_idxs.empty()) {
    sorted_unique_values.push_back(value_and_example_idxs.front().first);
  }
  for (size_t sorted_idx = 1; sorted_idx < value_and_example_idxs.size();
       sorted_idx++) {
    if (value_and_example_idxs[sorted_idx - 1].first <
        value_and_example_idxs[sorted_idx].first) {
      sorted_unique_values.push_back(value_and_example_idxs[sorted_idx].first);
    }
  }

  // Get the list of boundary values.
  std::vector<float> boundaries(sorted_unique_values.size() - 1);
  for (size_t boundary_idx = 0; boundary_idx < boundaries.size();
       boundary_idx++) {
    boundaries[boundary_idx] =
        decision_tree::MidThreshold(sorted_unique_values[boundary_idx],
                                    sorted_unique_values[boundary_idx + 1]);
  }
  return boundaries;
}

utils::StatusOr<std::vector<float>>
ExtractDiscretizedBoundariesWithDownsampling(
    const std::vector<std::pair<float, model::SignedExampleIdx>>&
        value_and_example_idxs,
    int64_t num_unique_values, int64_t num_discretized_values) {
  // Gather the unique values and observation count.
  std::vector<std::pair<float, int>> unique_values_and_counts;
  unique_values_and_counts.reserve(num_unique_values);

  int current_count = 0;
  float current_value = std::numeric_limits<float>::quiet_NaN();
  for (const auto& value_and_example_idx : value_and_example_idxs) {
    if (value_and_example_idx.first != current_value) {
      unique_values_and_counts.push_back(
          {value_and_example_idx.first, current_count});
      current_value = value_and_example_idx.first;
      current_count = 0;
    }
    current_count++;
  }
  if (current_count > 0) {
    unique_values_and_counts.push_back({current_value, current_count});
  }

  return dataset::GenDiscretizedBoundaries(unique_values_and_counts,
                                           num_discretized_values, 1, {});
}

}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
