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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_DATASET_CACHE_COMMON_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_DATASET_CACHE_COMMON_H_

#include <string>

#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/learner/types.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace dataset_cache {

// How to represent a label.
typedef int16_t ClassificationLabelType;
typedef float RegressionLabelType;

// How to index an example.
typedef SignedExampleIdx ExampleIdxType;
// How to represent a categorical value.
typedef int32_t CategoricalType;
typedef int32_t DiscretizedIndexedNumericalType;
typedef int8_t BooleanType;

// File and directory names.
constexpr char kFilenameColumn[] = "column_";
constexpr char kFilenameShard[] = "shard_";
constexpr char kFilenameShardNoUnderscore[] = "shard";

constexpr char kFilenameIndexed[] = "indexed";
constexpr char kFilenameRaw[] = "raw";

constexpr char kFilenameExampleIdx[] = "example_idx_with_delta_";
constexpr char kFilenameExampleIdxNoUnderscore[] = "example_idx_with_delta";
constexpr char kFilenameDeltaValue[] = "delta_value_";
constexpr char kFilenameDeltaValueNoUnderscore[] = "delta_value";
constexpr char kFilenameBoundaryValue[] = "boundary_value_";
constexpr char kFilenameBoundaryValueNoUnderscore[] = "boundary_value";
constexpr char kFilenameDiscretizedValues[] = "discretized_values_";
constexpr char kFilenameDiscretizedValuesNoUnderscore[] = "discretized_values";

// Meta-data file names.
constexpr char kFilenameShardMetaData[] = "shard_metadata.pb";
constexpr char kFilenamePresortedMetaData[] = "presorted_metadata.pb";
constexpr char kFilenameMetaData[] = "metadata.pb";
constexpr char kFilenameMetaDataPostfix[] = "_metadata.pb";
constexpr char kFilenameShardMetadataNoUnderscore[] = "shard_metadata";

constexpr char kFilenameTmp[] = "tmp";

// Path of the file containing the column data.
std::string ColumnPath(absl::string_view directory, int column_idx);

// Path of the file containing the meta data of a single shard.
std::string ShardMetadataPath(absl::string_view directory, int shard_idx,
                              int num_shards);

// Path to the file containing the raw / in-order column data.
std::string RawColumnFilePath(absl::string_view directory, int column_idx,
                              int shard_idx, int num_shards);

// Path to the directory containing the raw / in-order column data.
std::string RawColumnFileDirectory(absl::string_view directory, int column_idx);

// Background:
// An "example index with delta bit" is an example index (i.e. an integer
// specifying the index of an example in a list) with an extra bit of
// information called the "delta bit". For a given numerical feature value, the
// delta bit is 1 iif. this example index is the smallest example index with
// this numerical feature value. When examples are sorted according to a feature
// value, the delta bit is true iif. the value of the feature changes between
// the current and previous example.
//
// The two following functions extract the mask of the delta bit and the example
// index (without delta bit).
//
// For example, if the dataset contains 10 examples. The first 4 bits will
// encode the example index, and the 5-th bit will be the delta-bit. In this
// case, we will have:
//   MaskDeltaBit(1000) == 0b10000
//   MaskExampleIdx(1000) == 0b01111
//
uint64_t MaskDeltaBit(uint64_t num_examples);
uint64_t MaskExampleIdx(uint64_t num_examples);

// Integer value such that [-MaxValue-1, MaxValue] use the same encoding
// precision as "num_examples" examples and a delta bit.
uint64_t MaxValue(uint64_t num_examples);
}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_COMMON_H_
