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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"

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

int DeltaBitIdx(uint64_t num_examples) {
  return 64 - absl::countl_zero(num_examples);
}

uint64_t MaskDeltaBitFromDeltaBitIdx(int deltabit) {
  return uint64_t{1} << deltabit;
}

uint64_t MaskExampleIdxFromDeltaBitIdx(int deltabit) {
  return MaskDeltaBitFromDeltaBitIdx(deltabit) - 1;
}

uint64_t MaskDeltaBit(uint64_t num_examples) {
  return MaskDeltaBitFromDeltaBitIdx(DeltaBitIdx(num_examples));
}

uint64_t MaskExampleIdx(uint64_t num_examples) {
  return MaskExampleIdxFromDeltaBitIdx(DeltaBitIdx(num_examples));
}

uint64_t MaxValueWithDeltaBit(uint64_t num_examples) {
  return MaskDeltaBitFromDeltaBitIdx(DeltaBitIdx(num_examples)) | num_examples;
}

uint64_t MaxValueWithDeltaBitFromDeltaBitIdx(int deltabit) {
  return MaskDeltaBitFromDeltaBitIdx(deltabit) |
         MaskExampleIdxFromDeltaBitIdx(deltabit);
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

absl::StatusOr<std::vector<float>>
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

absl::StatusOr<std::vector<float>> ExtractDiscretizedBoundariesWithDownsampling(
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

bool HasAllRequiredFiles(absl::string_view cache_path, const int num_columns,
                         const int num_shards) {
  LOG(INFO) << "Checking required files in partial cache.";

  using model::distributed_decision_tree::dataset_cache::proto::
      PartialColumnShardMetadata;

  std::atomic<bool> is_valid{true};
  {
    utils::concurrency::ThreadPool thread_pool("HasAllRequiredFiles",
                                               /*num_threads=*/20);

    // Parse all the metadata.pb files.
    thread_pool.StartWorkers();
    for (int col_idx = 0; col_idx < num_columns; col_idx++) {
      for (int shard_idx = 0; shard_idx < num_shards; shard_idx++) {
        const auto shard_meta_data_path = absl::StrCat(
            PartialRawColumnFilePath(cache_path, col_idx, shard_idx),
            kFilenameMetaDataPostfix);
        thread_pool.Schedule([shard_meta_data_path, &is_valid]() {
          if (!is_valid) {
            return;
          }
          PartialColumnShardMetadata ignore;
          const auto status = file::GetBinaryProto(shard_meta_data_path,
                                                   &ignore, file::Defaults());
          if (!status.ok()) {
            LOG(INFO) << "Cannot parse " << shard_meta_data_path
                      << ". Issue: " << status.message();
            is_valid = false;
          }
        });
      }
    }
  }
  return is_valid;
}

}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
