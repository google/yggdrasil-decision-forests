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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache.h"

#include <numeric>

#include "absl/container/flat_hash_map.h"
#include "yggdrasil_decision_forests/dataset/formats.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"
#include "yggdrasil_decision_forests/learner/types.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/math.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace dataset_cache {
namespace {

// Number of request that each worker is expected to execute in parallel.
// This value impact the communication overhead (lower is better), the
// sensitivity to slow worker (higher is better), and the Ram usage of the
// workers (lower is better).
//
// TODO(gbm): Parametrize or set automatically according to the amount of
// available RAM / CPU on each worker.
constexpr int kNumParallelQueriesPerWorker = 5;

// Number of shards in the cache dataset. Increasing this value make the
// creation of the dataset cache more robust to slow workers (good) but increase
// the number of files each worker has to open when creating and reading the
// dataset cache.
constexpr int kNumShardPerWorkers = 10;

// List the typed shards and prefix from a typed sharded dataset path.
// TODO(gbm): Distribute or multi-thread the listing of shards for large
// datasets.
//
// For example:
//     ListShards("csv:/a/b@2", &shards, &type)
//     // type <= "csv"
//     // shards <= { "/a/b-00000-of-00002", "/a/b-00001-of-00002"}
//
absl::Status ListShards(const absl::string_view typed_path,
                        std::vector<std::string>* shards, std::string* type) {
  std::string non_typed_path;
  ASSIGN_OR_RETURN(std::tie(*type, non_typed_path),
                   dataset::SplitTypeAndPath(typed_path));
  return utils::ExpandInputShards(non_typed_path, shards);
}

// Returns "column_idxs" if "column_idxs" is set. Else, returns all the column
// indices in the dataset.
utils::StatusOr<std::vector<int>> GetColumnsOrAll(
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<int>* column_idxs,
    const proto::CreateDatasetCacheConfig& config) {
  std::vector<int> effective_columns;

  if (column_idxs) {
    effective_columns = *column_idxs;
    if (config.has_label_column_idx()) {
      effective_columns.push_back(config.label_column_idx());
    }
    if (config.has_weight_column_idx()) {
      effective_columns.push_back(config.weight_column_idx());
    }
    std::sort(effective_columns.begin(), effective_columns.end());
    effective_columns.erase(
        std::unique(effective_columns.begin(), effective_columns.end()),
        effective_columns.end());
  } else {
    effective_columns.resize(data_spec.columns_size());
    std::iota(effective_columns.begin(), effective_columns.end(), 0);
  }
  return effective_columns;
}

}  // namespace

absl::Status CreateDatasetCacheFromPartialDatasetCache(
    const dataset::proto::DataSpecification& data_spec,
    absl::string_view partial_cache_directory,
    absl::string_view final_cache_directory,
    const proto::CreateDatasetCacheConfig& config,
    const distribute::proto::Config& distribute_config,
    const bool delete_source_file) {
  const auto begin = absl::Now();
  LOG(INFO) << "Create dataset cache " << final_cache_directory
            << " from partial dataset cache " << partial_cache_directory;

  // Check if the cache is already there.
  const auto done_path = file::JoinPath(final_cache_directory, kFilenameDone);
  ASSIGN_OR_RETURN(const bool already_exist, file::FileExists(done_path));
  if (already_exist) {
    LOG(INFO) << "The dataset cache already exist.";
    return absl::OkStatus();
  }

  // Create the directory structure.
  RETURN_IF_ERROR(
      file::RecursivelyCreateDir(final_cache_directory, file::Defaults()));
  RETURN_IF_ERROR(file::RecursivelyCreateDir(
      file::JoinPath(final_cache_directory, kFilenameIndexed),
      file::Defaults()));
  RETURN_IF_ERROR(file::RecursivelyCreateDir(
      file::JoinPath(final_cache_directory, kFilenameRaw), file::Defaults()));
  RETURN_IF_ERROR(file::RecursivelyCreateDir(
      file::JoinPath(final_cache_directory, kFilenameTmp), file::Defaults()));

  // Initialize the distribution manager.
  proto::WorkerWelcome welcome;
  ASSIGN_OR_RETURN(
      auto distribute_manager,
      distribute::CreateManager(
          distribute_config,
          /*worker_name=*/"CREATE_DATASET_CACHE_WORKER",
          /*welcome_blob=*/welcome.SerializeAsString(),
          // Each worker is expected to do up to QueryPerWorker tasks in
          // parallel.
          /*parallel_execution_per_worker=*/kNumParallelQueriesPerWorker));

  std::vector<int> columns(data_spec.columns_size());
  std::iota(columns.begin(), columns.end(), 0);

  proto::CacheMetadata metadata;
  RETURN_IF_ERROR(
      internal::InitializeMetadata(data_spec, columns, config, &metadata));

  // Load the partial meta-data.
  proto::PartialDatasetMetadata partial_meta_data;
  CHECK_OK(file::GetBinaryProto(
      file::JoinPath(partial_cache_directory, kFilenamePartialMetaData),
      &partial_meta_data, file::Defaults()));
  metadata.set_num_examples(data_spec.created_num_rows());
  metadata.set_num_shards_in_feature_cache(partial_meta_data.num_shards());

  // TODO(gbm): Index the categorical-string features.

  // Copy / transform the raw feature values.
  RETURN_IF_ERROR(internal::ConvertPartialToFinalRawData(
      data_spec, partial_meta_data, partial_cache_directory,
      final_cache_directory, columns, config, delete_source_file,
      distribute_manager.get(), &metadata));

  // Pre-sort the numerical columns.
  RETURN_IF_ERROR(internal::SortNumericalColumns(
      data_spec, final_cache_directory, columns, config,
      distribute_manager.get(), &metadata));

  // Export the cache header.
  const auto metadata_path =
      file::JoinPath(final_cache_directory, kFilenameMetaData);
  RETURN_IF_ERROR(
      file::SetBinaryProto(metadata_path, metadata, file::Defaults()));

  RETURN_IF_ERROR(distribute_manager->Done());
  RETURN_IF_ERROR(file::SetContent(done_path, "done"));

  LOG(INFO) << "Dataset cache meta-data:\n" << MetaDataReport(metadata);
  LOG(INFO) << "Dataset cache created in " << absl::Now() - begin;

  LOG(INFO) << "Raw meta-data:\n" << metadata.DebugString();

  return absl::OkStatus();
}

absl::Status CreateDatasetCacheFromShardedFiles(
    const absl::string_view typed_path,
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<int>* columns, const absl::string_view cache_directory,
    const proto::CreateDatasetCacheConfig& config,
    const distribute::proto::Config& distribute_config) {
  const auto begin = absl::Now();
  LOG(INFO) << "Create dataset cache in " << cache_directory << " for dataset "
            << typed_path;

  // Check if the cache is already there.
  const auto done_path = file::JoinPath(cache_directory, kFilenameDone);
  ASSIGN_OR_RETURN(const bool already_exist, file::FileExists(done_path));
  if (already_exist) {
    LOG(INFO) << "The dataset cache already exist.";
    return absl::OkStatus();
  }

  // Create the directory structure.
  RETURN_IF_ERROR(
      file::RecursivelyCreateDir(cache_directory, file::Defaults()));
  RETURN_IF_ERROR(file::RecursivelyCreateDir(
      file::JoinPath(cache_directory, kFilenameIndexed), file::Defaults()));
  RETURN_IF_ERROR(file::RecursivelyCreateDir(
      file::JoinPath(cache_directory, kFilenameRaw), file::Defaults()));
  RETURN_IF_ERROR(file::RecursivelyCreateDir(
      file::JoinPath(cache_directory, kFilenameTmp), file::Defaults()));

  // Initialize the distribution manager.
  proto::WorkerWelcome welcome;
  ASSIGN_OR_RETURN(
      auto distribute_manager,
      distribute::CreateManager(
          distribute_config,
          /*worker_name=*/"CREATE_DATASET_CACHE_WORKER",
          /*welcome_blob=*/welcome.SerializeAsString(),
          // Each worker is expected to do up to QueryPerWorker tasks in
          // parallel.
          /*parallel_execution_per_worker=*/kNumParallelQueriesPerWorker));

  // List the columns in the dataset.
  ASSIGN_OR_RETURN(const auto effective_columns,
                   GetColumnsOrAll(data_spec, columns, config));
  LOG(INFO) << "Found " << effective_columns.size() << " column(s)";

  proto::CacheMetadata metadata;
  RETURN_IF_ERROR(internal::InitializeMetadata(data_spec, effective_columns,
                                               config, &metadata));

  // List the shards in the input dataset.
  std::vector<std::string> dataset_shards;
  std::string dataset_type;
  RETURN_IF_ERROR(ListShards(typed_path, &dataset_shards, &dataset_type));
  LOG(INFO) << "Found " << dataset_shards.size() << " shard(s)";

  // Separate the columns of individual shards.
  RETURN_IF_ERROR(internal::SeparateDatasetColumns(
      dataset_shards, dataset_type, data_spec, cache_directory,
      effective_columns, config, distribute_manager.get(), &metadata));

  // Pre-sort the numerical columns.
  RETURN_IF_ERROR(internal::SortNumericalColumns(
      data_spec, cache_directory, effective_columns, config,
      distribute_manager.get(), &metadata));

  // Export the cache header.
  const auto metadata_path = file::JoinPath(cache_directory, kFilenameMetaData);
  RETURN_IF_ERROR(
      file::SetBinaryProto(metadata_path, metadata, file::Defaults()));

  RETURN_IF_ERROR(distribute_manager->Done());
  RETURN_IF_ERROR(file::SetContent(done_path, "done"));

  LOG(INFO) << "Dataset cache meta-data:\n" << MetaDataReport(metadata);
  LOG(INFO) << "Dataset cache created in " << absl::Now() - begin;
  return absl::OkStatus();
}

utils::StatusOr<proto::CacheMetadata> LoadCacheMetadata(
    const absl::string_view path) {
  proto::CacheMetadata meta_data;
  RETURN_IF_ERROR(file::GetBinaryProto(file::JoinPath(path, kFilenameMetaData),
                                       &meta_data, file::Defaults()));
  return meta_data;
}

std::string MetaDataReport(const proto::CacheMetadata& metadata,
                           const absl::optional<std::vector<int>>& features) {
  // List the feature used to compute the statistics.
  std::vector<int> effective_features;
  if (features.has_value()) {
    effective_features = features.value();
  } else {
    // Use all the available features.
    effective_features.resize(metadata.columns_size());
    std::iota(effective_features.begin(), effective_features.end(), 0);
  }

  std::string report;

  absl::flat_hash_map<int, int> count_by_types;
  size_t sum_num_unique_values = 0;
  size_t num_numerical = 0;
  size_t num_numerical_discretized = 0;
  size_t num_numerical_less_100_values = 0;
  size_t num_numerical_less_16k_values = 0;
  size_t sum_num_discretized_values = 0;

  // Converts the "CacheMetadata::Column::type" integer type to a string
  // representation.
  const auto column_type_to_string = [](const int type) -> std::string {
    switch (type) {
      case 2:
        return "NUMERICAL";
      case 3:
        return "CATEGORICAL";
      case 4:
        return "BOOLEAN";
      default:
        return absl::StrCat("Unknown type ", type);
    }
  };

  for (const auto feature : effective_features) {
    DCHECK_GE(feature, 0);
    DCHECK_LT(feature, metadata.columns_size());

    const auto& column = metadata.columns(feature);
    count_by_types[column.type_case()]++;
    if (column.has_numerical()) {
      sum_num_unique_values += column.numerical().num_unique_values();
      if (column.numerical().discretized()) {
        num_numerical_discretized++;
        sum_num_discretized_values +=
            column.numerical().num_discretized_values();
      }
      if (column.numerical().num_unique_values() <= 100) {
        num_numerical_less_100_values++;
      }
      if (column.numerical().num_unique_values() <= 16000) {
        num_numerical_less_16k_values++;
      }
      num_numerical++;
    }
  }

  absl::SubstituteAndAppend(&report, "Number of columns: $0\n",
                            metadata.columns_size());
  absl::SubstituteAndAppend(&report, "Number of examples: $0\n",
                            metadata.num_examples());
  absl::SubstituteAndAppend(&report, "Statistics on $0 / $1 features\n",
                            effective_features.size(), metadata.columns_size());

  absl::StrAppend(&report, "Columns by type\n");
  for (const auto& count_by_type : count_by_types) {
    absl::SubstituteAndAppend(&report, "\t column-type: $0 count: $1\n",
                              column_type_to_string(count_by_type.first),
                              count_by_type.second);
  }

  if (num_numerical > 0) {
    absl::StrAppend(&report, "Numerical columns:\n");
    absl::SubstituteAndAppend(&report, "\tMean number of unique values: $0\n",
                              sum_num_unique_values / num_numerical);
    absl::SubstituteAndAppend(
        &report, "\tRatio of discretized numerical columns: $0 ($1)\n",
        static_cast<float>(num_numerical_discretized) / num_numerical,
        num_numerical_discretized);
    absl::SubstituteAndAppend(
        &report, "\tRatio of numerical columns with <=100 values: $0 ($1)\n",
        static_cast<float>(num_numerical_less_100_values) / num_numerical,
        num_numerical_less_100_values);
    absl::SubstituteAndAppend(
        &report, "\tRatio of numerical columns with <=16k values: $0 ($1)\n",
        static_cast<float>(num_numerical_less_16k_values) / num_numerical,
        num_numerical_less_16k_values);
    absl::SubstituteAndAppend(
        &report, "\tMean number of unique values for discretized columns: $0\n",
        static_cast<float>(sum_num_discretized_values) /
            num_numerical_discretized);
  }
  return report;
}

namespace internal {
absl::Status SeparateDatasetColumns(
    const std::vector<std::string>& dataset_shards,
    const std::string& dataset_type,
    const dataset::proto::DataSpecification& data_spec,
    const absl::string_view cache_directory, const std::vector<int>& columns,
    const proto::CreateDatasetCacheConfig& config,
    distribute::AbstractManager* distribute_manager,
    proto::CacheMetadata* cache_metadata) {
  LOG(INFO) << "Start separating dataset by columns";

  cache_metadata->set_num_examples(0);

  // Common part of the requests.
  proto::WorkerRequest generic_request;
  auto& request = *generic_request.mutable_separate_dataset_columns();
  *request.mutable_columns() = {columns.begin(), columns.end()};
  *request.mutable_dataspec() = data_spec;
  if (config.remove_zero_weighted_examples() &&
      config.has_weight_column_idx()) {
    request.set_column_idx_remove_example_with_zero(config.weight_column_idx());
  }
  request.set_output_directory(std::string(cache_directory));

  // Each request will combine "shards_per_requests" input shards (from the
  // input dataset; all the column values are in the same file) into 1 output
  // shards (each column in a separate file).
  //
  // See definition of kNumShardPerWorkers for a high level explanation.
  const int shards_per_requests = std::max<int>(
      1, dataset_shards.size() /
             (distribute_manager->NumWorkers() * kNumShardPerWorkers));
  const int num_output_shards =
      (dataset_shards.size() + shards_per_requests - 1) / shards_per_requests;
  request.set_num_shards(num_output_shards);

  RETURN_IF_ERROR(distribute_manager->SetParallelExecutionPerWorker(1));

  LOG(INFO) << "Create " << num_output_shards
            << " shards in the dataset cache from the " << dataset_shards.size()
            << " shards of the original dataset i.e. ~" << shards_per_requests
            << " shards to prepare for each of the "
            << distribute_manager->NumWorkers() << " workers";

  cache_metadata->set_num_shards_in_feature_cache(num_output_shards);
  int pending_requests = 0;
  for (int output_shard_idx = 0; output_shard_idx < num_output_shards;
       output_shard_idx++) {
    // Check if the job was already executed.
    const auto metadata_path =
        ShardMetadataPath(cache_directory, output_shard_idx, num_output_shards);
    ASSIGN_OR_RETURN(const bool already_exist, file::FileExists(metadata_path));
    if (already_exist) {
      LOG(INFO) << "The result of job #" << output_shard_idx
                << " is already there.";

      proto::ShardMetadata metadata;
      RETURN_IF_ERROR(
          file::GetBinaryProto(metadata_path, &metadata, file::Defaults()));

      cache_metadata->set_num_examples(cache_metadata->num_examples() +
                                       metadata.num_examples());
      continue;
    }

    // Create the job.
    int begin_shard_idx = output_shard_idx * shards_per_requests;
    int end_shard_idx = std::min(static_cast<int>(dataset_shards.size()),
                                 (output_shard_idx + 1) * shards_per_requests);
    request.set_shard_idx(output_shard_idx);
    request.set_dataset_path(absl::StrCat(
        dataset_type, ":",
        absl::StrJoin(dataset_shards.begin() + begin_shard_idx,
                      dataset_shards.begin() + end_shard_idx, ",")));
    RETURN_IF_ERROR(distribute_manager->AsynchronousProtoRequest(
        generic_request,
        /*worker_idx=*/output_shard_idx % distribute_manager->NumWorkers()));
    pending_requests++;
  }

  // Receive and rename the results.
  for (int result_idx = 0; result_idx < pending_requests; result_idx++) {
    LOG_INFO_EVERY_N_SEC(10, _ << "\tSeparate the dataset by columns "
                               << (result_idx + 1) << "/" << pending_requests);
    ASSIGN_OR_RETURN(
        const auto generic_result,
        distribute_manager->NextAsynchronousProtoAnswer<proto::WorkerResult>());
    const auto& result = generic_result.separate_dataset_columns();

    // Save the meta-data information.
    const auto metadata_path = ShardMetadataPath(
        cache_directory, result.shard_idx(), num_output_shards);

    proto::ShardMetadata metadata;
    metadata.set_num_examples(result.num_examples());
    RETURN_IF_ERROR(
        file::SetBinaryProto(metadata_path, metadata, file::Defaults()));

    cache_metadata->set_num_examples(cache_metadata->num_examples() +
                                     metadata.num_examples());
  }

  RETURN_IF_ERROR(distribute_manager->SetParallelExecutionPerWorker(
      kNumParallelQueriesPerWorker));

  LOG(INFO) << "Column separation done. " << cache_metadata->num_examples()
            << " example(s) found";
  return absl::OkStatus();
}

absl::Status ConvertPartialToFinalRawData(
    const dataset::proto::DataSpecification& data_spec,
    const proto::PartialDatasetMetadata& partial_metadata,
    absl::string_view partial_cache_directory,
    absl::string_view final_cache_directory, const std::vector<int>& columns,
    const proto::CreateDatasetCacheConfig& config,
    const bool delete_source_file,
    distribute::AbstractManager* distribute_manager,
    proto::CacheMetadata* cache_metadata) {
  LOG(INFO) << "Convert partial to final raw data";

  // Common part of the requests.
  proto::WorkerRequest generic_request;
  auto& request = *generic_request.mutable_convert_partial_to_final_raw_data();
  request.set_partial_cache_directory(std::string(partial_cache_directory));
  request.set_final_cache_directory(std::string(final_cache_directory));
  request.set_num_shards(partial_metadata.num_shards());
  request.set_delete_source_file(delete_source_file);

  int pending_requests = 0;
  for (int shard_idx = 0; shard_idx < partial_metadata.num_shards();
       shard_idx++) {
    request.set_shard_idx(shard_idx);
    for (int column_idx : columns) {
      request.set_column_idx(column_idx);

      const auto& column_spec = data_spec.columns(column_idx);
      switch (column_spec.type()) {
        case dataset::proto::NUMERICAL:
          request.mutable_numerical()->set_nan_value_replacement(
              cache_metadata->columns(column_idx)
                  .numerical()
                  .replacement_missing_value());
          break;
        case dataset::proto::CATEGORICAL:
          if (column_spec.categorical().is_already_integerized()) {
            request.mutable_categorical_int()->set_nan_value_replacement(
                column_spec.categorical().most_frequent_value());
            request.mutable_categorical_int()->set_max_value(
                cache_metadata->columns(column_idx).categorical().num_values());
          } else {
            *request.mutable_categorical_string()->mutable_items() =
                column_spec.categorical().items();
            request.mutable_categorical_string()->set_nan_value_replacement(
                column_spec.categorical().most_frequent_value());
          }
          break;
        default:
          return absl::InternalError(absl::Substitute(
              "Conversion not implemented for column of type $0",
              column_spec.type()));
      }

      RETURN_IF_ERROR(
          distribute_manager->AsynchronousProtoRequest(generic_request));
      pending_requests++;
    }
  }

  // Receive and rename the results.
  for (int result_idx = 0; result_idx < pending_requests; result_idx++) {
    LOG_INFO_EVERY_N_SEC(10, _ << "\tconverted columns " << (result_idx + 1)
                               << "/" << pending_requests);

    ASSIGN_OR_RETURN(
        const auto generic_result,
        distribute_manager->NextAsynchronousProtoAnswer<proto::WorkerResult>());
  }

  return absl::OkStatus();
}

absl::Status SortNumericalColumns(
    const dataset::proto::DataSpecification& data_spec,
    absl::string_view cache_directory, const std::vector<int>& columns,
    const proto::CreateDatasetCacheConfig& config,
    distribute::AbstractManager* distribute_manager,
    proto::CacheMetadata* cache_metadata) {
  LOG(INFO) << "Start sorting numerical columns";

  // Common part of the requests.
  proto::WorkerRequest generic_request;
  auto& request = *generic_request.mutable_sort_numerical_column();
  request.set_output_base_directory(
      file::JoinPath(cache_directory, kFilenameTmp));
  request.set_num_examples(cache_metadata->num_examples());
  request.set_cache_directory(std::string(cache_directory));

  // We assume that a cache entry takes 4 bytes.
  request.set_num_example_per_output_shards(
      config.index_cache_file_size_bytes() / sizeof(SignedExampleIdx));
  cache_metadata->set_num_shards_in_index_cache(utils::CeilDiV<int64_t>(
      cache_metadata->num_examples(), request.num_example_per_output_shards()));
  request.set_num_shards_in_output_shards(
      cache_metadata->num_shards_in_index_cache());

  request.set_max_unique_values_for_discretized_numerical(
      config.max_unique_values_for_discretized_numerical());
  request.set_force_numerical_discretization(
      config.force_numerical_discretization());

  int pending_requests = 0;
  for (int column_idx : columns) {
    const auto& column_spec = data_spec.columns(column_idx);
    if (column_spec.type() != dataset::proto::ColumnType::NUMERICAL) {
      continue;
    }

    // Check if the job was already executed.
    const auto metadata_path = file::JoinPath(
        cache_directory, kFilenameIndexed,
        absl::StrCat(kFilenameColumn, column_idx), kFilenamePresortedMetaData);
    ASSIGN_OR_RETURN(const bool already_exist, file::FileExists(metadata_path));
    if (already_exist) {
      proto::SortedColumnMetadata metadata;
      RETURN_IF_ERROR(
          file::GetBinaryProto(metadata_path, &metadata, file::Defaults()));

      auto* column_metadata =
          cache_metadata->mutable_columns(column_idx)->mutable_numerical();
      column_metadata->MergeFrom(metadata.metadata());
      LOG(INFO) << "The result of job for column #" << column_idx
                << " is already there.";
      continue;
    }

    // Create the job.
    request.set_column_idx(column_idx);
    request.set_num_shards(cache_metadata->num_shards_in_feature_cache());
    request.set_replacement_missing_value(cache_metadata->columns(column_idx)
                                              .numerical()
                                              .replacement_missing_value());

    RETURN_IF_ERROR(
        distribute_manager->AsynchronousProtoRequest(generic_request));
    pending_requests++;
  }

  // Receive and rename the results.
  for (int result_idx = 0; result_idx < pending_requests; result_idx++) {
    LOG_INFO_EVERY_N_SEC(10, _ << "\tsorting numerical columns "
                               << (result_idx + 1) << "/" << pending_requests);

    ASSIGN_OR_RETURN(
        const auto generic_result,
        distribute_manager->NextAsynchronousProtoAnswer<proto::WorkerResult>());
    const auto& result = generic_result.sort_numerical_column();

    // Rename the output directory.
    const auto final_directory =
        file::JoinPath(cache_directory, kFilenameIndexed,
                       absl::StrCat(kFilenameColumn, result.column_idx()));
    ASSIGN_OR_RETURN(const bool already_exist,
                     file::FileExists(final_directory));
    if (already_exist) {
      LOG(WARNING) << "The directory result of job on column #"
                   << result.column_idx() << " already exist.";
    } else {
      RETURN_IF_ERROR(file::Rename(result.output_directory(), final_directory,
                                   file::Defaults()));
    }

    // Save the meta-data information.
    const auto metadata_path =
        file::JoinPath(final_directory, kFilenamePresortedMetaData);

    proto::SortedColumnMetadata metadata;
    *metadata.mutable_metadata() = result.metadata();
    auto* column_metadata = cache_metadata->mutable_columns(result.column_idx())
                                ->mutable_numerical();
    column_metadata->MergeFrom(result.metadata());

    RETURN_IF_ERROR(
        file::SetBinaryProto(metadata_path, metadata, file::Defaults()));
  }
  return absl::OkStatus();
}

absl::Status InitializeMetadata(
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<int>& columns,
    const proto::CreateDatasetCacheConfig& config,
    proto::CacheMetadata* metadata) {
  // Label values, if any.
  if (config.has_label_column_idx()) {
    metadata->set_label_column_idx(config.label_column_idx());
  }
  if (config.has_weight_column_idx()) {
    metadata->set_weight_column_idx(config.weight_column_idx());
  }

  // Column meta-data.
  for (int col_idx = 0; col_idx < data_spec.columns_size(); col_idx++) {
    metadata->add_columns();
  }
  for (const auto col_idx : columns) {
    const auto& src = data_spec.columns(col_idx);
    auto* dst = metadata->mutable_columns(col_idx);
    dst->set_available(true);
    switch (src.type()) {
      case dataset::proto::ColumnType::NUMERICAL:
        dst->mutable_numerical()->set_replacement_missing_value(
            src.numerical().mean());
        break;

      case dataset::proto::ColumnType::CATEGORICAL:
        dst->mutable_categorical()->set_num_values(
            src.categorical().number_of_unique_values());
        dst->mutable_categorical()->set_replacement_missing_value(
            src.categorical().most_frequent_value());
        break;

      case dataset::proto::ColumnType::BOOLEAN:
        dst->mutable_boolean()->set_replacement_missing_value(
            src.boolean().count_true() >= src.boolean().count_false());
        break;

      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Non supported type ", dataset::proto::ColumnType_Name(src.type()),
            " for column ", src.name()));
    }
  }

  if (config.has_remove_zero_weighted_examples()) {
    if (!config.has_weight_column_idx()) {
      return absl::InvalidArgumentError(
          "\"remove_zero_weighted_examples\" without a weight column");
    }
    if (data_spec.columns(config.weight_column_idx()).type() !=
        dataset::proto::NUMERICAL) {
      return absl::InvalidArgumentError(
          "\"remove_zero_weighted_examples\" only support numerical weighted "
          "columns.");
    }
  }

  return absl::OkStatus();
}

}  // namespace internal
}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
