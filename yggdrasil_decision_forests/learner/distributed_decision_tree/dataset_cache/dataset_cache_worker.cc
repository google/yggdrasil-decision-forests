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

  absl::Mutex mutex_worker_status;
  absl::Status worker_status;
  int exported_columns = 0;

  {
    utils::concurrency::ThreadPool thread_pool("ExportColumns", kNumThreads);
    thread_pool.StartWorkers();
    for (const auto column_idx : request.columns()) {
      thread_pool.Schedule([&, column_idx]() {
        {
          absl::MutexLock l(&mutex_worker_status);
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
          absl::MutexLock l(&mutex_worker_status);
          worker_status.Update(local_status);
          exported_columns++;
        }
      });
    }
  }

  RETURN_IF_ERROR(file::RecursivelyDelete(temp_directory, file::Defaults()));

  return worker_status;
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
    case proto::WorkerRequest::TYPE_NOT_SET:
      return absl::InvalidArgumentError("Request without type");
  }
  return result.SerializeAsString();
}

}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
