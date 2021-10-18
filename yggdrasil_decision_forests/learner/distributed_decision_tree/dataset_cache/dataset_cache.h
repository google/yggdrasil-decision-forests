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
// distributed training of a decision forest model.
//
// The structure of one such directory is as follow:
//
//   metadata.pb: Global meta-data about the cache.
//
//   raw/column_{i}/shard_{j}-of-{k}: Features values ordered by example index.
//
//   indexed/column_{i}/{example_idx_with_delta,discretized_values,
//     boundary_value}_{j}-of-{k}: Pre-sorted index of numerical features.
//
//   tmp: Temporary files for the creation of the cache. Empty after a cache
//   finished being created.
//
// Use the constant in "dataset_cache_common.h" for the exact names.
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

utils::StatusOr<proto::CacheMetadata> LoadCacheMetadata(
    const absl::string_view path);

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
