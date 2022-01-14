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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_worker.h"

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"
#include "yggdrasil_decision_forests/learner/types.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace dataset_cache {

constexpr char CreateDatasetCacheWorker::kWorkerKey[];

namespace {
using Blob = distribute::Blob;

// Number of threads for IO operations.
//
// TODO(gbm): Parametrize.
constexpr int kNumThreads = 10;

absl::Status SeparateNumericalColumn(const int column_idx,
                                     const dataset::proto::Column& column_spec,
                                     const dataset::VerticalDataset& dataset,
                                     absl::string_view path) {
  proto::CreateDatasetCacheConfig config;
  FloatColumnWriter writer;
  RETURN_IF_ERROR(writer.Open(path));
  const auto& values =
      dataset
          .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(column_idx)
          ->values();

  const float missing_value_replacement = column_spec.numerical().mean();
  std::vector<float> buffer(kIOBufferSizeInBytes / sizeof(float));
  size_t begin = 0;
  while (begin < values.size()) {
    const auto num = std::min(values.size() - begin, buffer.size());
    std::transform(values.begin() + begin, values.begin() + begin + num,
                   buffer.begin(), [missing_value_replacement](float value) {
                     if (std::isnan(value)) {
                       return missing_value_replacement;
                     } else {
                       return value;
                     }
                   });
    RETURN_IF_ERROR(writer.WriteValues(absl::Span<float>(buffer.data(), num)));
    begin += num;
  }
  return writer.Close();
}

absl::Status SeparateCategoricalColumn(
    const int column_idx, const dataset::proto::Column& column_spec,
    const dataset::VerticalDataset& dataset, absl::string_view path) {
  proto::CreateDatasetCacheConfig config;
  IntegerColumnWriter writer;
  RETURN_IF_ERROR(writer.Open(
      path,
      /*max_value=*/column_spec.categorical().number_of_unique_values() - 1));
  const auto& values =
      dataset
          .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
              column_idx)
          ->values();

  const int32_t missing_value_replacement =
      column_spec.categorical().most_frequent_value();
  std::vector<int32_t> buffer(kIOBufferSizeInBytes / sizeof(int32_t));
  size_t begin = 0;
  while (begin < values.size()) {
    const auto num = std::min(values.size() - begin, buffer.size());
    std::transform(values.begin() + begin, values.begin() + begin + num,
                   buffer.begin(),
                   [missing_value_replacement](int32_t value) -> int32_t {
                     if (value < 0) {
                       return missing_value_replacement;
                     } else {
                       return value;
                     }
                   });
    RETURN_IF_ERROR(
        writer.WriteValues<int32_t>(absl::Span<int32_t>(buffer.data(), num)));
    begin += num;
  }
  return writer.Close();
}

absl::Status SeparateBooleanColumn(const int column_idx,
                                   const dataset::proto::Column& column_spec,
                                   const dataset::VerticalDataset& dataset,
                                   absl::string_view path) {
  proto::CreateDatasetCacheConfig config;
  IntegerColumnWriter writer;
  RETURN_IF_ERROR(writer.Open(path,
                              /*max_value=*/2));
  const auto& values =
      dataset
          .ColumnWithCast<dataset::VerticalDataset::BooleanColumn>(column_idx)
          ->values();

  const int8_t missing_value_replacement =
      column_spec.boolean().count_true() >= column_spec.boolean().count_false();
  std::vector<int8_t> buffer(kIOBufferSizeInBytes / sizeof(int8_t));
  size_t begin = 0;
  while (begin < values.size()) {
    const auto num = std::min(values.size() - begin, buffer.size());
    std::transform(
        values.begin() + begin, values.begin() + begin + num, buffer.begin(),
        [missing_value_replacement](char value) -> int8_t {
          DCHECK_GE(value, 0);
          DCHECK_LT(value, 3);
          if (value == dataset::VerticalDataset::BooleanColumn::kNaValue) {
            return missing_value_replacement;
          } else {
            return value;
          }
        });
    RETURN_IF_ERROR(
        writer.WriteValues<int8_t>(absl::Span<int8_t>(buffer.data(), num)));
    begin += num;
  }
  return writer.Close();
}

}  // namespace

absl::Status CreateDatasetCacheWorker::SeparateDatasetColumn(
    const dataset::VerticalDataset& dataset, const int column_idx,
    const int shard_idx, const int num_shards,
    const absl::string_view temp_directory,
    const absl::string_view output_directory) {
  const auto temp_column_path = ColumnPath(temp_directory, column_idx);

  const auto& column_spec = dataset.data_spec().columns(column_idx);
  switch (column_spec.type()) {
    case dataset::proto::ColumnType::NUMERICAL:
      RETURN_IF_ERROR(SeparateNumericalColumn(column_idx, column_spec, dataset,
                                              temp_column_path));
      break;
    case dataset::proto::ColumnType::CATEGORICAL:
      RETURN_IF_ERROR(SeparateCategoricalColumn(column_idx, column_spec,
                                                dataset, temp_column_path));
      break;
    case dataset::proto::ColumnType::BOOLEAN:
      RETURN_IF_ERROR(SeparateBooleanColumn(column_idx, column_spec, dataset,
                                            temp_column_path));
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Non supported column type ",
                       dataset::proto::ColumnType_Name(column_spec.type()),
                       " for columns \"", column_spec.name(), "\""));
  }

  // Move the file from the tmp to its final destination.
  RETURN_IF_ERROR(file::RecursivelyCreateDir(
      RawColumnFileDirectory(output_directory, column_idx), file::Defaults()));
  const auto final_column_path =
      RawColumnFilePath(output_directory, column_idx, shard_idx, num_shards);

  if (!file::Rename(temp_column_path, final_column_path, file::Defaults())
           .ok()) {
    LOG(WARNING) << "Already existing final file. Multiple workers seems to "
                    "work on the same shard.";
  }

  return absl::OkStatus();
}

absl::Status CreateDatasetCacheWorker::SeparateDatasetColumns(
    const proto::WorkerRequest::SeparateDatasetColumns& request,
    proto::WorkerResult::SeparateDatasetColumns* result) {
  LOG(INFO) << "Separate dataset columns on " << request.dataset_path();
  result->set_shard_idx(request.shard_idx());

  // TODO(gbm): Use a dataset reader directly with multi-threaded reading.
  LOG(INFO) << "Reading dataset";
  dataset::VerticalDataset dataset;
  dataset::LoadConfig load_dataset_config;
  load_dataset_config.num_threads = kNumThreads;
  load_dataset_config.load_columns = {request.columns().begin(),
                                      request.columns().end()};

  if (request.has_column_idx_remove_example_with_zero()) {
    const auto weight_attribute = request.column_idx_remove_example_with_zero();
    load_dataset_config.load_example =
        [weight_attribute](const dataset::proto::Example& example) {
          return example.attributes(weight_attribute).numerical() > 0.f;
        };
  }

  RETURN_IF_ERROR(dataset::LoadVerticalDataset(request.dataset_path(),
                                               request.dataspec(), &dataset, {},
                                               load_dataset_config));

  result->set_num_examples(dataset.nrow());

  const auto temp_directory = file::JoinPath(
      request.output_directory(), kFilenameTmp, utils::GenUniqueId());
  RETURN_IF_ERROR(file::RecursivelyCreateDir(temp_directory, file::Defaults()));

  utils::concurrency::Mutex mutex_worker_status;
  absl::Status worker_status;
  int exported_columns = 0;

  {
    utils::concurrency::ThreadPool thread_pool("ExportColumns", kNumThreads);
    thread_pool.StartWorkers();
    for (const auto column_idx : request.columns()) {
      thread_pool.Schedule([&, column_idx]() {
        {
          utils::concurrency::MutexLock l(&mutex_worker_status);
          if (!worker_status.ok()) {
            return;
          }
          LOG_INFO_EVERY_N_SEC(30, _ << "Exporting columns "
                                     << (exported_columns + 1) << "/"
                                     << request.columns_size());
        }

        const auto local_status = SeparateDatasetColumn(
            dataset, column_idx, request.shard_idx(), request.num_shards(),
            temp_directory, request.output_directory());
        {
          utils::concurrency::MutexLock l(&mutex_worker_status);
          worker_status.Update(local_status);
          exported_columns++;
        }
      });
    }
  }

  RETURN_IF_ERROR(file::RecursivelyDelete(temp_directory, file::Defaults()));

  return worker_status;
}

absl::Status CreateDatasetCacheWorker::SortNumericalColumn(
    const proto::WorkerRequest::SortNumericalColumn& request,
    proto::WorkerResult::SortNumericalColumn* result) {
  LOG(INFO) << "Sorting numerical column #" << request.column_idx();

  // Read the values.
  LOG(INFO) << "Allocate " << request.num_examples()
            << " examples for column  #" << request.column_idx();
  // TODO(gbm): Read the shards in parallel.
  std::vector<std::pair<float, model::SignedExampleIdx>> value_and_example_idxs(
      request.num_examples());
  LOG(INFO) << "Start reading column  #" << request.column_idx();
  const int input_buffer_size = kIOBufferSizeInBytes / sizeof(float);
  ShardedFloatColumnReader reader;
  RETURN_IF_ERROR(reader.Open(
      file::JoinPath(request.cache_directory(), kFilenameRaw,
                     absl::StrCat(kFilenameColumn, request.column_idx()),
                     kFilenameShardNoUnderscore),
      /*max_num_values=*/input_buffer_size,
      /*begin_shard_idx=*/0, /*end_shard_idx=*/request.num_shards()));
  size_t example_idx = 0;
  while (true) {
    RETURN_IF_ERROR(reader.Next());
    const auto values = reader.Values();
    if (values.empty()) {
      break;
    }
    for (const float value : values) {
      value_and_example_idxs[example_idx] = {value, example_idx};
      example_idx++;
    }
  }
  RETURN_IF_ERROR(reader.Close());

  // Sort the values.
  LOG(INFO) << "Sort the numerical values of column #" << request.column_idx();
  std::sort(value_and_example_idxs.begin(), value_and_example_idxs.end());

  // Export the sorted values.
  LOG(INFO) << "Export the pre-sorted numerical values of column #"
            << request.column_idx();

  result->set_output_directory(
      file::JoinPath(request.output_base_directory(), utils::GenUniqueId()));
  RETURN_IF_ERROR(
      file::RecursivelyCreateDir(result->output_directory(), file::Defaults()));
  result->set_column_idx(request.column_idx());
  result->mutable_metadata()->set_replacement_missing_value(
      request.replacement_missing_value());

  // Count the number of unique values.
  int64_t num_unique_values = 0;
  for (size_t sorted_idx = 1; sorted_idx < value_and_example_idxs.size();
       sorted_idx++) {
    if (value_and_example_idxs[sorted_idx - 1].first <
        value_and_example_idxs[sorted_idx].first) {
      num_unique_values++;
    }
  }
  result->mutable_metadata()->set_num_unique_values(num_unique_values);
  LOG(INFO) << "Found " << num_unique_values << "/" << request.num_examples()
            << " unique values on numerical column #" << request.column_idx()
            << ".";

  // Select how export the values (pre-sorted or discretized).
  result->mutable_metadata()->set_discretized(
      request.force_numerical_discretization() ||
      num_unique_values <=
          request.max_unique_values_for_discretized_numerical());

  if (result->metadata().discretized()) {
    LOG(INFO) << "Exported column  column #" << request.column_idx()
              << " as pre-discretized";
    RETURN_IF_ERROR(ExportSortedDiscretizedNumericalColumn(
        request, value_and_example_idxs, num_unique_values, result));
  } else {
    LOG(INFO) << "Exported column  column #" << request.column_idx()
              << " as pre-sorted";
    RETURN_IF_ERROR(
        ExportSortedNumericalColumn(request, value_and_example_idxs, result));
  }

  LOG(INFO) << "Done exporting column #" << request.column_idx() << " with "
            << num_unique_values << "/" << request.num_examples()
            << " unique values.";
  return absl::OkStatus();
}

absl::Status CreateDatasetCacheWorker::ExportSortedNumericalColumn(
    const proto::WorkerRequest::SortNumericalColumn& request,
    const std::vector<std::pair<float, model::SignedExampleIdx>>&
        value_and_example_idxs,
    proto::WorkerResult::SortNumericalColumn* result) {
  proto::CreateDatasetCacheConfig config;

  IntegerColumnWriter example_idx_writer;
  FloatColumnWriter values_writer;

  // Number of remaining examples the current shard can receive. When a shard
  // is full, a new one is created. If remaining_examples_in_shard=0, a new
  // shard will be created on the next example.
  int64_t remaining_examples_in_shard = 0;
  int next_output_shard_idx = 0;

  RETURN_IF_ERROR(values_writer.Open(
      file::JoinPath(result->output_directory(),
                     ShardFilename(kFilenameDeltaValueNoUnderscore, 0, 1))));

  RETURN_IF_ERROR(
      values_writer.WriteValues({value_and_example_idxs.front().first}));
  const auto max_value_example_idx =
      MaxValueWithDeltaBit(request.num_examples());

  // Output buffers;
  std::vector<float> value_buffer;
  std::vector<model::SignedExampleIdx> example_idx_buffer;
  value_buffer.reserve(kIOBufferSizeInBytes / sizeof(float));
  example_idx_buffer.reserve(kIOBufferSizeInBytes /
                             sizeof(model::SignedExampleIdx));

  // TODO(gbm): Distribute writing.
  const auto delta_bit_mask = MaskDeltaBit(request.num_examples());
  for (size_t sorted_idx = 0; sorted_idx < value_and_example_idxs.size();
       sorted_idx++) {
    const bool delta_bit =
        sorted_idx > 0 && value_and_example_idxs[sorted_idx - 1].first <
                              value_and_example_idxs[sorted_idx].first;
    model::SignedExampleIdx example_idx_with_delta_bit =
        value_and_example_idxs[sorted_idx].second;
    if (delta_bit) {
      example_idx_with_delta_bit |= delta_bit_mask;

      value_buffer.push_back(value_and_example_idxs[sorted_idx].first);
      if (value_buffer.size() >= kIOBufferSizeInBytes / sizeof(float)) {
        RETURN_IF_ERROR(values_writer.WriteValues(value_buffer));
        value_buffer.clear();
      }
    }

    if (remaining_examples_in_shard == 0) {
      if (sorted_idx > 0) {
        // Close the current shard.
        RETURN_IF_ERROR(example_idx_writer.Close());
      }
      // Open a new shard.
      RETURN_IF_ERROR(example_idx_writer.Open(
          file::JoinPath(result->output_directory(),
                         ShardFilename(kFilenameExampleIdxNoUnderscore,
                                       next_output_shard_idx++,
                                       request.num_shards_in_output_shards())),

          /*max_value=*/max_value_example_idx));
      remaining_examples_in_shard = request.num_example_per_output_shards();
    }

    example_idx_buffer.push_back(example_idx_with_delta_bit);
    if (example_idx_buffer.size() >=
        kIOBufferSizeInBytes / sizeof(model::SignedExampleIdx)) {
      RETURN_IF_ERROR(example_idx_writer.WriteValues<model::SignedExampleIdx>(
          example_idx_buffer));
      example_idx_buffer.clear();
    }

    remaining_examples_in_shard--;
  }

  // Flush buffers.
  RETURN_IF_ERROR(values_writer.WriteValues(value_buffer));
  RETURN_IF_ERROR(example_idx_writer.WriteValues<model::SignedExampleIdx>(
      example_idx_buffer));

  RETURN_IF_ERROR(example_idx_writer.Close());
  RETURN_IF_ERROR(values_writer.Close());

  if (next_output_shard_idx != request.num_shards_in_output_shards()) {
    return absl::InternalError(
        absl::Substitute("Unexpected number of generated shards in sorted "
                         "numerical feature #$0. $1 != $2",
                         request.column_idx(), next_output_shard_idx,
                         request.num_shards_in_output_shards()));
  }

  return absl::OkStatus();
}

absl::Status CreateDatasetCacheWorker::ExportSortedDiscretizedNumericalColumn(
    const proto::WorkerRequest::SortNumericalColumn& request,
    const std::vector<std::pair<float, model::SignedExampleIdx>>&
        value_and_example_idxs,
    int64_t num_unique_values,
    proto::WorkerResult::SortNumericalColumn* result) {
  proto::CreateDatasetCacheConfig config;

  std::vector<float> boundaries;
  if (num_unique_values <=
      request.max_unique_values_for_discretized_numerical()) {
    ASSIGN_OR_RETURN(boundaries,
                     ExtractDiscretizedBoundariesWithoutDownsampling(
                         value_and_example_idxs, num_unique_values));

  } else {
    ASSIGN_OR_RETURN(
        boundaries, ExtractDiscretizedBoundariesWithDownsampling(
                        value_and_example_idxs, num_unique_values,
                        request.max_unique_values_for_discretized_numerical()));
  }

  const auto num_discretized_values = boundaries.size() + 1;
  result->mutable_metadata()->set_num_discretized_values(
      num_discretized_values);

  result->mutable_metadata()->set_discretized_replacement_missing_value(
      NumericalToDiscretizedNumerical(boundaries,
                                      request.replacement_missing_value()));

  // Export the boundary values.
  FloatColumnWriter boundary_writer;
  RETURN_IF_ERROR(boundary_writer.Open(
      file::JoinPath(result->output_directory(),
                     ShardFilename(kFilenameBoundaryValueNoUnderscore, 0, 1))));
  RETURN_IF_ERROR(boundary_writer.WriteValues(boundaries));
  RETURN_IF_ERROR(boundary_writer.Close());

  // Indexed the values.
  ShardedFloatColumnReader values_reader;
  IntegerColumnWriter indexed_values_writer;
  bool indexed_values_writer_is_open = false;

  const int buffer_size = kIOBufferSizeInBytes / sizeof(float);
  std::vector<DiscretizedIndexedNumericalType> indexed_value_buffer;

  RETURN_IF_ERROR(values_reader.Open(
      file::JoinPath(request.cache_directory(), kFilenameRaw,
                     absl::StrCat(kFilenameColumn, request.column_idx()),
                     kFilenameShardNoUnderscore),
      /*max_num_values=*/buffer_size,
      /*begin_shard_idx=*/0, /*end_shard_idx=*/request.num_shards()));

  int64_t remaining_examples_in_shard = 0;
  int next_output_shard_idx = 0;

  while (true) {
    RETURN_IF_ERROR(values_reader.Next());
    const auto values = values_reader.Values();
    if (values.empty()) {
      break;
    }

    for (const auto value : values) {
      if (remaining_examples_in_shard == 0) {
        if (indexed_values_writer_is_open) {
          // Close the current shard.
          RETURN_IF_ERROR(indexed_values_writer.Close());
        }
        // Open a new shard.
        RETURN_IF_ERROR(indexed_values_writer.Open(
            file::JoinPath(
                result->output_directory(),
                ShardFilename(kFilenameDiscretizedValuesNoUnderscore,
                              next_output_shard_idx++,
                              request.num_shards_in_output_shards())),
            /*max_value=*/num_discretized_values));
        indexed_values_writer_is_open = true;
        remaining_examples_in_shard = request.num_example_per_output_shards();
      }

      indexed_value_buffer.push_back(
          NumericalToDiscretizedNumerical(boundaries, value));
      if (indexed_value_buffer.size() >=
          kIOBufferSizeInBytes / sizeof(model::SignedExampleIdx)) {
        RETURN_IF_ERROR(
            indexed_values_writer.WriteValues<DiscretizedIndexedNumericalType>(
                indexed_value_buffer));
        indexed_value_buffer.clear();
      }

      remaining_examples_in_shard--;
    }
  }

  RETURN_IF_ERROR(values_reader.Close());

  if (indexed_values_writer_is_open) {
    RETURN_IF_ERROR(
        indexed_values_writer.WriteValues<DiscretizedIndexedNumericalType>(
            indexed_value_buffer));
    indexed_value_buffer.clear();

    RETURN_IF_ERROR(indexed_values_writer.Close());
  }

  if (next_output_shard_idx != request.num_shards_in_output_shards()) {
    return absl::InternalError(
        absl::Substitute("Unexpected number of generated shards in discretized "
                         "numerical feature #$0. $1 != $2",
                         request.column_idx(), next_output_shard_idx,
                         request.num_shards_in_output_shards()));
  }

  result->mutable_metadata()->set_num_discretized_shards(next_output_shard_idx);
  return absl::OkStatus();
}

absl::Status ConvertPartialToFinalRawDataNumerical(
    const absl::string_view input_file, const absl::string_view output_file,
    const proto::WorkerRequest::ConvertPartialToFinalRawData::Numerical&
        transformation) {
  const int input_buffer_size = kIOBufferSizeInBytes / sizeof(float);

  FloatColumnReader input_stream;
  RETURN_IF_ERROR(input_stream.Open(input_file, input_buffer_size));

  FloatColumnWriter output_stream;
  RETURN_IF_ERROR(output_stream.Open(output_file));

  std::vector<float> transform_buffer;
  while (true) {
    RETURN_IF_ERROR(input_stream.Next());
    const auto values = input_stream.Values();
    if (values.empty()) {
      break;
    }

    transform_buffer.resize(values.size());
    std::transform(values.begin(), values.end(), transform_buffer.begin(),
                   [&transformation](const float value) -> float {
                     if (std::isnan(value)) {
                       return transformation.nan_value_replacement();
                     } else {
                       return value;
                     }
                   });

    RETURN_IF_ERROR(output_stream.WriteValues(transform_buffer));
  }

  RETURN_IF_ERROR(input_stream.Close());
  RETURN_IF_ERROR(output_stream.Close());
  return absl::OkStatus();
}

absl::Status ConvertPartialToFinalRawDataCategoricalInt(
    const absl::string_view input_file, const absl::string_view output_file,
    const proto::WorkerRequest::ConvertPartialToFinalRawData::CategoricalInt&
        transformation) {
  const int input_buffer_size = kIOBufferSizeInBytes / sizeof(float);

  IntegerColumnReader<int32_t> input_stream;
  RETURN_IF_ERROR(input_stream.Open(
      input_file, std::numeric_limits<int32_t>::max(), input_buffer_size));

  IntegerColumnWriter output_stream;
  RETURN_IF_ERROR(output_stream.Open(output_file, transformation.max_value()));

  std::vector<int32_t> transform_buffer;
  while (true) {
    RETURN_IF_ERROR(input_stream.Next());
    const auto values = input_stream.Values();
    if (values.empty()) {
      break;
    }

    transform_buffer.resize(values.size());
    std::transform(values.begin(), values.end(), transform_buffer.begin(),
                   [&transformation](const int32_t value) -> int32_t {
                     if (value < 0) {
                       return transformation.nan_value_replacement();
                     }
                     return value;
                   });

    RETURN_IF_ERROR(
        output_stream.WriteValues(absl::Span<const int32_t>(transform_buffer)));
  }

  RETURN_IF_ERROR(input_stream.Close());
  RETURN_IF_ERROR(output_stream.Close());
  return absl::OkStatus();
}

absl::Status ConvertPartialToFinalRawDataCategoricalString(
    const absl::string_view input_file, const absl::string_view output_file,
    const proto::WorkerRequest::ConvertPartialToFinalRawData::CategoricalString&
        transformation,
    const proto::PartialColumnShardMetadata& meta_data) {
  // Compute the re-mapping.
  // Value "i" is transformed into "mapping[i]". If i<0, replace the value with
  // "nan_value_replacement".
  std::vector<int32_t> mapping(meta_data.categorical().items_size());
  for (const auto& src_item : meta_data.categorical().items()) {
    const auto it_src = transformation.items().find(src_item.first);
    if (it_src == transformation.items().end()) {
      // This item is unknown in the final dictionary. It is transformed to
      // Out-of-vocabulary.
      mapping[src_item.second.index()] = dataset::kOutOfDictionaryItemIndex;
    } else {
      mapping[src_item.second.index()] = it_src->second.index();
    }
  }

  const int input_buffer_size = kIOBufferSizeInBytes / sizeof(float);

  IntegerColumnReader<int32_t> input_stream;
  RETURN_IF_ERROR(input_stream.Open(
      input_file, std::numeric_limits<int32_t>::max(), input_buffer_size));

  IntegerColumnWriter output_stream;
  RETURN_IF_ERROR(output_stream.Open(output_file, transformation.items_size()));

  std::vector<int32_t> transform_buffer;
  while (true) {
    RETURN_IF_ERROR(input_stream.Next());
    const auto values = input_stream.Values();
    if (values.empty()) {
      break;
    }

    transform_buffer.resize(values.size());
    std::transform(values.begin(), values.end(), transform_buffer.begin(),
                   [&transformation, &mapping](const int32_t value) -> int32_t {
                     if (value < 0) {
                       return transformation.nan_value_replacement();
                     }
                     return mapping[value];
                   });

    RETURN_IF_ERROR(
        output_stream.WriteValues(absl::Span<const int32_t>(transform_buffer)));
  }

  RETURN_IF_ERROR(input_stream.Close());
  RETURN_IF_ERROR(output_stream.Close());
  return absl::OkStatus();
}

absl::Status CreateDatasetCacheWorker::ConvertPartialToFinalRawData(
    const proto::WorkerRequest::ConvertPartialToFinalRawData& request,
    proto::WorkerResult::ConvertPartialToFinalRawData* result) {
  LOG(INFO) << "Convert partial to final for column #" << request.column_idx()
            << " and shard #" << request.shard_idx();

  // Get the various paths.
  const auto tmp_file = file::JoinPath(request.final_cache_directory(),
                                       kFilenameTmp, utils::GenUniqueId());
  const auto input_file =
      PartialRawColumnFilePath(request.partial_cache_directory(),
                               request.column_idx(), request.shard_idx());

  // Meta-data specific for the shard+column.
  const auto input_metadata_path =
      absl::StrCat(input_file, kFilenameMetaDataPostfix);
  proto::PartialColumnShardMetadata meta_data;
  RETURN_IF_ERROR(
      file::GetBinaryProto(input_metadata_path, &meta_data, file::Defaults()));

  const auto output_file =
      RawColumnFilePath(request.final_cache_directory(), request.column_idx(),
                        request.shard_idx(), request.num_shards());
  const auto output_directory = RawColumnFileDirectory(
      request.final_cache_directory(), request.column_idx());

  RETURN_IF_ERROR(
      file::RecursivelyCreateDir(output_directory, file::Defaults()));

  ASSIGN_OR_RETURN(const bool already_exist, file::FileExists(output_file));
  if (already_exist) {
    LOG(INFO) << "The result already exist.";
    return absl::OkStatus();
  }

  switch (request.transformation_case()) {
    case proto::WorkerRequest_ConvertPartialToFinalRawData::kNumerical:
      RETURN_IF_ERROR(ConvertPartialToFinalRawDataNumerical(
          input_file, tmp_file, request.numerical()));
      break;

    case proto::WorkerRequest_ConvertPartialToFinalRawData::kCategoricalInt:
      RETURN_IF_ERROR(ConvertPartialToFinalRawDataCategoricalInt(
          input_file, tmp_file, request.categorical_int()));
      break;

    case proto::WorkerRequest_ConvertPartialToFinalRawData::
        kCategoricalString: {
      RETURN_IF_ERROR(ConvertPartialToFinalRawDataCategoricalString(
          input_file, tmp_file, request.categorical_string(), meta_data));
      break;
    }

    case proto::WorkerRequest_ConvertPartialToFinalRawData::
        TRANSFORMATION_NOT_SET:
      return absl::InternalError("Transformation not set");
  }

  if (!file::Rename(tmp_file, output_file, file::Defaults()).ok()) {
    LOG(WARNING) << "Already existing final file. Multiple workers seems to "
                    "work on the same shard.";
  }

  return absl::OkStatus();
}

absl::Status CreateDatasetCacheWorker::Setup(Blob serialized_welcome) {
  ASSIGN_OR_RETURN(welcome_, utils::ParseBinaryProto<proto::WorkerWelcome>(
                                 serialized_welcome));
  return absl::OkStatus();
}

utils::StatusOr<Blob> CreateDatasetCacheWorker::RunRequest(
    Blob serialized_request) {
  ASSIGN_OR_RETURN(auto request, utils::ParseBinaryProto<proto::WorkerRequest>(
                                     serialized_request));
  proto::WorkerResult result;
  switch (request.type_case()) {
    case proto::WorkerRequest::kSeparateDatasetColumns:
      RETURN_IF_ERROR(
          SeparateDatasetColumns(request.separate_dataset_columns(),
                                 result.mutable_separate_dataset_columns()));
      break;
    case proto::WorkerRequest::kSortNumericalColumn:
      RETURN_IF_ERROR(
          SortNumericalColumn(request.sort_numerical_column(),
                              result.mutable_sort_numerical_column()));
      break;
    case proto::WorkerRequest::kConvertPartialToFinalRawData:
      RETURN_IF_ERROR(ConvertPartialToFinalRawData(
          request.convert_partial_to_final_raw_data(),
          result.mutable_convert_partial_to_final_raw_data()));
      break;
    case proto::WorkerRequest::TYPE_NOT_SET:
      return absl::InvalidArgumentError("Request without type");
  }
  return result.SerializeAsString();
}

}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
