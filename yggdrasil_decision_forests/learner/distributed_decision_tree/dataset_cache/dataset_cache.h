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

// A dataset cache is a directory containing a dataset indexed / rearranged for
// distributed training of a decision forest model. It can be created from files
// with the "CreateDatasetCacheFromShardedFiles" method.
//
// The structure of one such directory is as follows:
//
//   metadata.pb: Global meta-data about the cache. CacheMetadata proto stored
//     in binary.
//   raw/column_{i}/shard_{j}-of-{k}: Features values ordered by example index:
//     Miss values are replaced with the corresponding
//     "replacement_missing_value" field in the proto meta data. Numerical
//     values are stored as floats. Categorical values (integer or string) are
//     stored as variable precision integer (optimized according to the maximum
//     number of possible values).
//   indexed/column_{i}/{example_idx_with_delta,discretized_values,
//     boundary_value}_{j}-of-{k}: Pre-sorted index of numerical features.
//   tmp: Temporary files for the creation of the cache. Empty after a cache
//     finished being created.
//
// Use the constant in "dataset_cache_common.h" for the exact names.
//
// A "partial dataset cache" is a format close but different from the (final)
// "dataset cache". Unlike the "dataset cache" that contains global meta-data
// (e.g. the mean value of a feature), indexes (e.g. example indices ordered
// according to specific feature index), and dictionary (e.g. the string->int
// mapping of a categorical-string feature), the "partial dataset cache" is a
// pure "dump" of data without aggregated global information. A "partial dataset
// cache" can be created in parallel by different workers without need for
// synchronization. A partial dataset cache can then be converted into a (final)
// dataset cache with the "CreateDatasetCacheFromPartialDatasetCache" method.
//
// The format is as follows:
//
//   partial_metadata.pb: The *only* global meta-data about the cache. Contains
//     the name of the features and the number of shards (generally each worker
//     will write in own shards). PartialDatasetMetadata proto stored in binary.
//   raw/column_{i}/shard_{j}-of-{k}: Features values ordered by example index:
//     Miss values can be present. Numerical values are stored as floats.
//     Categorical integer values are stored as int32. Categorical string values
//     are stored as int32 with a per-feature+per-shard dictionary stored in the
//     per-feature+per-shard meta-data proto.
//   raw/column_{i}/shard_{j}-of-{k}_metadata.pb: Meta-data of this blob of
//     data. PartialColumnShardMetadata proto stored in binary.
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_DATASET_CACHE_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_DATASET_CACHE_H_

#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache.pb.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.pb.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace dataset_cache {

// Creates a dataset cache from a dataset.
//
// If distributed computation is used, the "CreateDatasetCacheFromShardedFiles"
// calling process does not ready any dataset file directly.
//
// If "effective_columns" is null, all the columns in the dataspec are used.
absl::Status CreateDatasetCacheFromShardedFiles(
    absl::string_view typed_path,
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<int>* columns, absl::string_view cache_directory,
    const proto::CreateDatasetCacheConfig& config,
    const distribute::proto::Config& distribute_config);

// Create a dataset cache from a partially created dataset cache.
//
// A partial dataset cache only contains raw per-column and per-shard
// data (i.e. no indexed data, no meta-data about multiple columns or multiple
// shards).
//
// If "delete_source_file=True", the content of the partial dataset cache is
// deleted as it is being consumed. Note that this does not present to resume
// "CreateDatasetCacheFromPartialDatasetCache" if either the manager or the
// workers are interrupted.
//
absl::Status CreateDatasetCacheFromPartialDatasetCache(
    const dataset::proto::DataSpecification& data_spec,
    absl::string_view partial_cache_directory,
    absl::string_view cache_directory,
    const proto::CreateDatasetCacheConfig& config,
    const distribute::proto::Config& distribute_config,
    bool delete_source_file);

utils::StatusOr<proto::CacheMetadata> LoadCacheMetadata(absl::string_view path);

// Human readable report about the metadata. If "features" is specified,
// computes the statistics on the "features" features only. Otherwise, computes
// the statistics on all the features.
std::string MetaDataReport(
    const proto::CacheMetadata& metadata,
    const absl::optional<std::vector<int>>& features = {});

namespace internal {

// Splits the dataset column by column and shards by shards.
//
// Set the following fields in the cache metadata: num_examples, num_shards.
absl::Status SeparateDatasetColumns(
    const std::vector<std::string>& dataset_shards,
    const std::string& dataset_type,
    const dataset::proto::DataSpecification& data_spec,
    absl::string_view cache_directory, const std::vector<int>& columns,
    const proto::CreateDatasetCacheConfig& config,
    distribute::AbstractManager* distribute_manager,
    proto::CacheMetadata* cache_metadata);

// Converts the raw value in the partial dataset cache to the value in the final
// dataset cache. This is a simple copy of the features values with some
// transformation:
//   - Replace missing numerical values with a specific value.
//   - Re-encode the integer categorical features to optimal integer precision.
absl::Status ConvertPartialToFinalRawData(
    const dataset::proto::DataSpecification& data_spec,
    const proto::PartialDatasetMetadata& partial_metadata,
    absl::string_view partial_cache_directory,
    absl::string_view final_cache_directory, const std::vector<int>& columns,
    const proto::CreateDatasetCacheConfig& config, bool delete_source_file,
    distribute::AbstractManager* distribute_manager,
    proto::CacheMetadata* cache_metadata);

// Sort the numerical columns.
absl::Status SortNumericalColumns(
    const dataset::proto::DataSpecification& data_spec,
    absl::string_view cache_directory, const std::vector<int>& columns,
    const proto::CreateDatasetCacheConfig& config,
    distribute::AbstractManager* distribute_manager,
    proto::CacheMetadata* cache_metadata);

// Initializes the meta-data content from the dataspec, column and configs.
// TODO(gbm): Make "InitializeMetadata" return a "CacheMetadata" directly.
absl::Status InitializeMetadata(
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<int>& columns,
    const proto::CreateDatasetCacheConfig& config,
    proto::CacheMetadata* metadata);

}  // namespace internal
}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_H_
