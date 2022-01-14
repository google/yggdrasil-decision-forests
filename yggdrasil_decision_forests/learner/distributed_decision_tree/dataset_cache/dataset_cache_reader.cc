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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_reader.h"

#include <numeric>

#include "yggdrasil_decision_forests/dataset/formats.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace dataset_cache {

std::string DatasetCacheReader::MetadataInformation() const {
  auto report = MetaDataReport(meta_data(), features_);
  absl::SubstituteAndAppend(&report, "Number of loaded columns: $0\n",
                            features_.size());
  return report;
}

utils::StatusOr<std::unique_ptr<DatasetCacheReader>> DatasetCacheReader::Create(
    absl::string_view path, const proto::DatasetCacheReaderOptions& options) {
  const auto begin = absl::Now();
  auto cache = absl::WrapUnique(new DatasetCacheReader(path, options));
  RETURN_IF_ERROR(file::GetBinaryProto(file::JoinPath(path, kFilenameMetaData),
                                       &cache->meta_data_, file::Defaults()));

  // List the features available to the reader.
  if (options.features().empty()) {
    // Make all the features available.
    cache->features_.resize(cache->meta_data_.columns_size());
    std::iota(cache->features_.begin(), cache->features_.end(), 0);
  } else {
    cache->features_ = {options.features().begin(), options.features().end()};
  }
  std::sort(cache->features_.begin(), cache->features_.end());
  LOG(INFO) << "Create dataset cache reader on " << cache->features_.size()
            << " / " << cache->meta_data_.columns_size() << " feature(s) and "
            << cache->meta_data_.num_examples() << " example(s)";

  if (cache->meta_data_.has_weight_column_idx()) {
    // Load the weight values.
    LOG(INFO) << "Loading weights in memory";
    cache->weights_.reserve(cache->meta_data_.num_examples());
    RETURN_IF_ERROR(ShardedFloatColumnReader::ReadAndAppend(
        file::JoinPath(path, kFilenameRaw,
                       absl::StrCat(kFilenameColumn,
                                    cache->meta_data_.weight_column_idx()),
                       kFilenameShardNoUnderscore),
        /*begin_shard_idx=*/0,
        /*end_shard_idx=*/cache->meta_data_.num_shards_in_feature_cache(),
        &cache->weights_));
  }

  if (cache->meta_data_.has_label_column_idx()) {
    LOG(INFO) << "Loading labels in memory";

    const auto label_column_idx = cache->meta_data_.label_column_idx();
    const auto& label_column_metadata =
        cache->meta_data_.columns(label_column_idx);
    switch (label_column_metadata.type_case()) {
      case proto::CacheMetadata_Column::kCategorical: {
        // Load the categorical label values.
        cache->classification_labels_.reserve(cache->meta_data_.num_examples());
        RETURN_IF_ERROR(
            ShardedIntegerColumnReader<ClassificationLabelType>::ReadAndAppend(
                file::JoinPath(path, kFilenameRaw,
                               absl::StrCat(kFilenameColumn, label_column_idx),
                               kFilenameShardNoUnderscore),
                /*max_value=*/
                label_column_metadata.categorical().num_values(),
                /*begin_shard_idx=*/0,
                /*end_shard_idx=*/
                cache->meta_data_.num_shards_in_feature_cache(),
                &cache->classification_labels_));
      } break;

      case proto::CacheMetadata_Column::kNumerical: {
        // Load the numerical label values.
        cache->regression_labels_.reserve(cache->meta_data_.num_examples());
        RETURN_IF_ERROR(ShardedFloatColumnReader::ReadAndAppend(
            file::JoinPath(path, kFilenameRaw,
                           absl::StrCat(kFilenameColumn, label_column_idx),
                           kFilenameShardNoUnderscore),
            /*begin_shard_idx=*/0,
            /*end_shard_idx=*/
            cache->meta_data_.num_shards_in_feature_cache(),
            &cache->regression_labels_));
      } break;

      case proto::CacheMetadata_Column::kBoolean:
        return absl::InvalidArgumentError("Boolean label not supported.");

      case proto::CacheMetadata_Column::TYPE_NOT_SET:
        return absl::InvalidArgumentError("Label type not set");
    }
  }

  if (options.load_cache_in_memory()) {
    RETURN_IF_ERROR(cache->InitializeAndLoadInMemoryCache());
  }

  LOG(INFO) << "Dataset cache meta-data:\n" << cache->MetadataInformation();
  LOG(INFO) << "Dataset cache reader created in " << absl::Now() - begin;
  return cache;
}

absl::Status DatasetCacheReader::NonBlockingLoadingAndUnloadingFeatures(
    const std::vector<int>& load_features,
    const std::vector<int>& unload_features, const int num_threads) {
  if (!options_.load_cache_in_memory()) {
    return absl::OkStatus();
  }

  if (IsNonBlockingLoadingInProgress()) {
    return absl::InternalError(
        "Non-blocking feature loading already in progress.");
  }
  utils::concurrency::MutexLock lock(&non_blocking_.status_mutex);
  non_blocking_.is_running = true;
  non_blocking_.status = {};  // Clear status
  non_blocking_.load_features = load_features;
  non_blocking_.unload_features = unload_features;

  non_blocking_.loading_thread =
      absl::make_unique<utils::concurrency::Thread>([this, num_threads]() {
        const auto begin = absl::Now();

        {
          utils::concurrency::ThreadPool pool(
              "LoadFeatures",
              std::min<int>(non_blocking_.load_features.size(), num_threads));
          pool.StartWorkers();
          for (const int column_idx : non_blocking_.load_features) {
            pool.Schedule([&, column_idx]() {
              {
                utils::concurrency::MutexLock l(&non_blocking_.status_mutex);
                if (!non_blocking_.status.ok()) {
                  return;
                }
              }
              size_t column_memory_usage;
              const auto status =
                  LoadInMemoryCacheColumn(column_idx, &column_memory_usage);
              utils::concurrency::MutexLock l(&non_blocking_.status_mutex);
              non_blocking_.status.Update(status);
            });
          }
        }  // Join on "pool".

        LOG(INFO) << "Non-blocking feature update done in "
                  << (absl::Now() - begin);
        DCHECK(non_blocking_.is_running);
        non_blocking_.is_running = false;
      });

  return absl::OkStatus();
}

bool DatasetCacheReader::IsNonBlockingLoadingInProgress() {
  return non_blocking_.loading_thread != nullptr;
}

const std::vector<int>&
DatasetCacheReader::NonBlockingLoadingInProgressLoadedFeatures() const {
  return non_blocking_.load_features;
}

const std::vector<int>&
DatasetCacheReader::NonBlockingLoadingInProgressUnloadedFeatures() const {
  return non_blocking_.unload_features;
}

utils::StatusOr<bool> DatasetCacheReader::CheckAndUpdateNonBlockingLoading() {
  utils::concurrency::MutexLock lock(&non_blocking_.status_mutex);
  if (non_blocking_.is_running) {
    // Still running.
    return true;
  }
  if (non_blocking_.loading_thread == nullptr) {
    // Not running.
    return false;
  }
  LOG(INFO) << "Non-blocking work done. Making the new features available";

  // Wait for the loading thread.
  CHECK(non_blocking_.loading_thread != nullptr);
  non_blocking_.loading_thread->Join();
  non_blocking_.loading_thread.reset();

  // Propagate the loading thread error (if any).
  if (!non_blocking_.status.ok()) {
    LOG(INFO) << "Error in non-blocking loading: "
              << non_blocking_.status.message();
    return absl::Status(non_blocking_.status);
  }

  for (const int column_idx : non_blocking_.unload_features) {
    RETURN_IF_ERROR(UnloadInMemoryCacheColumn(column_idx));
  }

  // Update the meta-data. After this, the changes are visible to the user.
  RETURN_IF_ERROR(ApplyLoadingAndUnloadingFeaturesToMetadata(
      non_blocking_.load_features, non_blocking_.unload_features));
  return false;
}

absl::Status DatasetCacheReader::LoadingAndUnloadingFeatures(
    const std::vector<int>& load_features,
    const std::vector<int>& unload_features) {
  if (IsNonBlockingLoadingInProgress()) {
    return absl::InternalError(
        "Non-blocking feature loading already in progress.");
  }

  LOG(INFO) << "Loading " << load_features.size() << " and unloading "
            << unload_features.size() << " feature(s)";

  if (options_.load_cache_in_memory()) {
    const auto begin = absl::Now();

    for (const int column_idx : unload_features) {
      RETURN_IF_ERROR(UnloadInMemoryCacheColumn(column_idx));
    }

    if (!load_features.empty()) {
      absl::Status worker_status;
      {
        utils::concurrency::Mutex mutex_worker_status;
        utils::concurrency::ThreadPool pool(
            "LoadFeatures", std::min<int>(load_features.size(), 20));
        pool.StartWorkers();

        for (const int column_idx : load_features) {
          pool.Schedule([&, column_idx]() {
            {
              utils::concurrency::MutexLock l(&mutex_worker_status);
              if (!worker_status.ok()) {
                return;
              }
            }
            size_t column_memory_usage;
            const auto status =
                LoadInMemoryCacheColumn(column_idx, &column_memory_usage);
            utils::concurrency::MutexLock l(&mutex_worker_status);
            worker_status.Update(status);
          });
        }
      }
      RETURN_IF_ERROR(worker_status);
    }

    LOG(INFO) << "Update loaded features in " << (absl::Now() - begin);
  }

  return ApplyLoadingAndUnloadingFeaturesToMetadata(load_features,
                                                    unload_features);
}

absl::Status DatasetCacheReader::ApplyLoadingAndUnloadingFeaturesToMetadata(
    const std::vector<int>& load_features,
    const std::vector<int>& unload_features) {
  std::vector<int> new_features;
  new_features.reserve(features_.size() + load_features.size() -
                       unload_features.size());
  std::set_difference(features_.begin(), features_.end(),
                      unload_features.begin(), unload_features.end(),
                      std::back_inserter(new_features));
  new_features.insert(new_features.end(), load_features.begin(),
                      load_features.end());
  std::sort(new_features.begin(), new_features.end());
  if (new_features.size() !=
      features_.size() + load_features.size() - unload_features.size()) {
    return absl::InternalError(absl::Substitute(
        "Unexpected number of features after load/unload features ($0) + "
        "load_features ($1) unload_features ($2) != new_features ($3)",
        features_.size(), load_features.size(), unload_features.size(),
        new_features.size()));
  }
  features_ = new_features;
  return absl::OkStatus();
}

bool DatasetCacheReader::has_feature(const int feature) const {
  const auto it = std::lower_bound(features_.begin(), features_.end(), feature);
  return (it != features_.end()) && (*it == feature);
}

absl::Status DatasetCacheReader::UnloadInMemoryCacheColumn(
    const int column_idx) {
  const auto& column = meta_data().columns(column_idx);
  switch (column.type_case()) {
    case proto::CacheMetadata_Column::kCategorical:
      DCHECK(in_memory_cache_.inorder_categorical_columns_[column_idx] !=
             nullptr);
      in_memory_cache_.inorder_categorical_columns_[column_idx].reset();
      break;

    case proto::CacheMetadata_Column::kNumerical:
      DCHECK(in_memory_cache_.inorder_numerical_columns_[column_idx] !=
             nullptr);
      if (column.numerical().discretized()) {
        DCHECK(in_memory_cache_
                   .inorder_discretized_numerical_columns_[column_idx] !=
               nullptr);
        DCHECK(!in_memory_cache_
                    .boundaries_of_discretized_numerical_columns_[column_idx]
                    .empty());
      } else {
        DCHECK(in_memory_cache_
                   .presorted_numerical_example_idx_columns_[column_idx] !=
               nullptr);
        DCHECK(in_memory_cache_
                   .presorted_numerical_unique_values_columns_[column_idx] !=
               nullptr);
      }

      in_memory_cache_.inorder_numerical_columns_[column_idx].reset();
      in_memory_cache_.presorted_numerical_example_idx_columns_[column_idx]
          .reset();
      in_memory_cache_.presorted_numerical_unique_values_columns_[column_idx]
          .reset();
      in_memory_cache_.inorder_discretized_numerical_columns_[column_idx]
          .reset();
      in_memory_cache_.boundaries_of_discretized_numerical_columns_[column_idx]
          .clear();
      break;

    case proto::CacheMetadata_Column::kBoolean:
      DCHECK(in_memory_cache_.inorder_boolean_columns_[column_idx] != nullptr);
      in_memory_cache_.inorder_boolean_columns_[column_idx].reset();
      break;

    case proto::CacheMetadata_Column::TYPE_NOT_SET:
      break;
  }
  return absl::OkStatus();
}

absl::Status DatasetCacheReader::LoadInMemoryCacheColumn(int column_idx,
                                                         size_t* memory_usage) {
  *memory_usage = 0;
  const auto& column = meta_data().columns(column_idx);
  switch (column.type_case()) {
    case proto::CacheMetadata_Column::kCategorical: {
      auto& dst = in_memory_cache_.inorder_categorical_columns_[column_idx];
      DCHECK(dst == nullptr);
      dst = absl::make_unique<
          InMemoryIntegerColumnReaderFactory<CategoricalType>>();
      const auto max_value =
          meta_data_.columns(column_idx).categorical().num_values();
      dst->Reserve(meta_data_.num_examples(), max_value);
      RETURN_IF_ERROR(dst->Load(
          file::JoinPath(path_, kFilenameRaw,
                         absl::StrCat(kFilenameColumn, column_idx),
                         kFilenameShardNoUnderscore),
          /*max_value=*/
          max_value,
          /*max_num_values=*/options_.reading_buffer(),
          /*begin_shard_idx=*/0,
          /*end_shard_idx=*/meta_data_.num_shards_in_feature_cache()));
      *memory_usage += dst->MemoryUsage();
    } break;

    case proto::CacheMetadata_Column::kNumerical: {
      auto& dst_in_order =
          in_memory_cache_.inorder_numerical_columns_[column_idx];
      auto& dst_presorted_example_idxs =
          in_memory_cache_.presorted_numerical_example_idx_columns_[column_idx];
      auto& dst_presorted_unique_values =
          in_memory_cache_
              .presorted_numerical_unique_values_columns_[column_idx];
      auto& dst_discretized_values =
          in_memory_cache_.inorder_discretized_numerical_columns_[column_idx];
      auto& dst_discretized_boundaries =
          in_memory_cache_
              .boundaries_of_discretized_numerical_columns_[column_idx];

      DCHECK(dst_in_order == nullptr);
      DCHECK(dst_presorted_example_idxs == nullptr);
      DCHECK(dst_presorted_unique_values == nullptr);
      DCHECK(dst_discretized_values == nullptr);
      DCHECK(dst_discretized_boundaries.empty());

      // Raw numerical value.
      // TODO: Do not load the raw value if they are discretized.
      dst_in_order = absl::make_unique<InMemoryFloatColumnReaderFactory>();
      dst_in_order->Reserve(meta_data_.num_examples());
      RETURN_IF_ERROR(dst_in_order->Load(
          file::JoinPath(path_, kFilenameRaw,
                         absl::StrCat(kFilenameColumn, column_idx),
                         kFilenameShardNoUnderscore),
          /*max_num_values=*/options_.reading_buffer(),
          /*begin_shard_idx=*/0,
          /*end_shard_idx=*/meta_data_.num_shards_in_feature_cache()));
      *memory_usage += dst_in_order->MemoryUsage();

      if (column.numerical().discretized()) {
        dst_discretized_values =
            absl::make_unique<InMemoryIntegerColumnReaderFactory<
                DiscretizedIndexedNumericalType>>();
        dst_discretized_values->Reserve(
            meta_data_.num_examples(),
            column.numerical().num_discretized_values());
        RETURN_IF_ERROR(dst_discretized_values->Load(
            file::JoinPath(path_, kFilenameIndexed,
                           absl::StrCat(kFilenameColumn, column_idx),
                           kFilenameDiscretizedValuesNoUnderscore),
            /*max_value=*/column.numerical().num_discretized_values(),
            /*max_num_values=*/options_.reading_buffer(),
            /*begin_shard_idx=*/0,
            /*end_shard_idx=*/column.numerical().num_discretized_shards()));
        *memory_usage += dst_discretized_values->MemoryUsage();

        dst_discretized_boundaries.reserve(
            column.numerical().num_discretized_values() - 1);
        RETURN_IF_ERROR(ShardedFloatColumnReader::ReadAndAppend(
            file::JoinPath(path_, kFilenameIndexed,
                           absl::StrCat(kFilenameColumn, column_idx),
                           kFilenameBoundaryValueNoUnderscore),
            /*begin_shard_idx=*/0,
            /*end_shard_idx=*/1, &dst_discretized_boundaries));
        *memory_usage += dst_discretized_boundaries.size() * sizeof(float);
      } else {
        dst_presorted_example_idxs = absl::make_unique<
            InMemoryIntegerColumnReaderFactory<ExampleIdxType>>();
        dst_presorted_example_idxs->Reserve(
            meta_data_.num_examples(),
            MaxValueWithDeltaBit(meta_data_.num_examples()));
        RETURN_IF_ERROR(dst_presorted_example_idxs->Load(
            file::JoinPath(path_, kFilenameIndexed,
                           absl::StrCat(kFilenameColumn, column_idx),
                           kFilenameExampleIdxNoUnderscore),
            /*max_value=*/MaxValueWithDeltaBit(meta_data_.num_examples()),
            /*max_num_values=*/options_.reading_buffer(),
            /*begin_shard_idx=*/0,
            /*end_shard_idx=*/meta_data_.num_shards_in_index_cache()));
        *memory_usage += dst_presorted_example_idxs->MemoryUsage();

        dst_presorted_unique_values =
            absl::make_unique<InMemoryFloatColumnReaderFactory>();
        dst_presorted_unique_values->Reserve(
            column.numerical().num_unique_values());
        RETURN_IF_ERROR(dst_presorted_unique_values->Load(
            file::JoinPath(path_, kFilenameIndexed,
                           absl::StrCat(kFilenameColumn, column_idx),
                           kFilenameDeltaValueNoUnderscore),
            /*max_num_values=*/options_.reading_buffer(),
            /*begin_shard_idx=*/0,
            /*end_shard_idx=*/1));
        *memory_usage += dst_presorted_unique_values->MemoryUsage();
      }
    } break;

    case proto::CacheMetadata_Column::kBoolean: {
      auto& dst = in_memory_cache_.inorder_boolean_columns_[column_idx];
      DCHECK(dst == nullptr);

      dst =
          absl::make_unique<InMemoryIntegerColumnReaderFactory<BooleanType>>();
      dst->Reserve(meta_data_.num_examples(), 2);
      RETURN_IF_ERROR(dst->Load(
          file::JoinPath(path_, kFilenameRaw,
                         absl::StrCat(kFilenameColumn, column_idx),
                         kFilenameShardNoUnderscore),
          /*max_value=*/2,
          /*max_num_values=*/options_.reading_buffer(),
          /*begin_shard_idx=*/0,
          /*end_shard_idx=*/meta_data_.num_shards_in_feature_cache()));
      *memory_usage += dst->MemoryUsage();
    } break;

    case proto::CacheMetadata_Column::TYPE_NOT_SET:
      break;
  }
  return absl::OkStatus();
}

absl::Status DatasetCacheReader::InitializeAndLoadInMemoryCache() {
  LOG(INFO) << "Loading features in memory";

  const auto num_columns = meta_data().columns_size();
  in_memory_cache_.inorder_categorical_columns_.resize(num_columns);
  in_memory_cache_.inorder_numerical_columns_.resize(num_columns);
  in_memory_cache_.presorted_numerical_example_idx_columns_.resize(num_columns);
  in_memory_cache_.presorted_numerical_unique_values_columns_.resize(
      num_columns);
  in_memory_cache_.inorder_boolean_columns_.resize(num_columns);
  in_memory_cache_.inorder_discretized_numerical_columns_.resize(num_columns);
  in_memory_cache_.boundaries_of_discretized_numerical_columns_.resize(
      num_columns);

  const auto begin = absl::Now();
  std::atomic<size_t> memory_usage{0};

  absl::Status worker_status;
  {
    utils::concurrency::Mutex mutex_worker_status;
    utils::concurrency::ThreadPool pool("LoadFeatures", 20);
    pool.StartWorkers();
    for (const int column_idx : features_) {
      pool.Schedule([&, column_idx]() {
        {
          utils::concurrency::MutexLock l(&mutex_worker_status);
          if (!worker_status.ok()) {
            return;
          }
        }
        size_t column_memory_usage;
        const auto status =
            LoadInMemoryCacheColumn(column_idx, &column_memory_usage);
        memory_usage += column_memory_usage;
        utils::concurrency::MutexLock l(&mutex_worker_status);
        worker_status.Update(status);
      });
    }
  }
  RETURN_IF_ERROR(worker_status);

  load_in_memory_duration_ = absl::Now() - begin;
  LOG(INFO) << "Features loaded in memory in " << load_in_memory_duration_
            << " for " << (memory_usage / (1024 * 1024)) << " MB";
  return absl::OkStatus();
}

uint64_t DatasetCacheReader::num_examples() const {
  return meta_data_.num_examples();
}

const std::vector<ClassificationLabelType>&
DatasetCacheReader::categorical_labels() const {
  return classification_labels_;
}

const std::vector<RegressionLabelType>& DatasetCacheReader::regression_labels()
    const {
  return regression_labels_;
}

const std::vector<float>& DatasetCacheReader::weights() const {
  return weights_;
}

utils::StatusOr<std::unique_ptr<AbstractIntegerColumnIterator<ExampleIdxType>>>
DatasetCacheReader::PresortedNumericalFeatureExampleIterator(
    int column_idx) const {
  if (!meta_data().columns(column_idx).has_numerical()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Column ", column_idx, " is not numerical"));
  }

  if (options_.load_cache_in_memory()) {
    if (in_memory_cache_.presorted_numerical_example_idx_columns_[column_idx] ==
        nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Column ", column_idx, " is not available"));
    }

    return in_memory_cache_
        .presorted_numerical_example_idx_columns_[column_idx]
        ->CreateIterator();
  }

  auto reader = absl::make_unique<ShardedIntegerColumnReader<ExampleIdxType>>();
  RETURN_IF_ERROR(reader->Open(
      file::JoinPath(path_, kFilenameIndexed,
                     absl::StrCat(kFilenameColumn, column_idx),
                     kFilenameExampleIdxNoUnderscore),
      /*max_value=*/MaxValueWithDeltaBit(meta_data_.num_examples()),
      /*max_num_values=*/options_.reading_buffer(),
      /*begin_shard_idx=*/0,
      /*end_shard_idx=*/meta_data_.num_shards_in_index_cache()));
  return reader;
}

utils::StatusOr<std::unique_ptr<AbstractFloatColumnIterator>>
DatasetCacheReader::PresortedNumericalFeatureValueIterator(
    int column_idx) const {
  if (!meta_data().columns(column_idx).has_numerical()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Column ", column_idx, " is not numerical"));
  }

  if (options_.load_cache_in_memory()) {
    if (in_memory_cache_
            .presorted_numerical_unique_values_columns_[column_idx] ==
        nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Column ", column_idx, " is not available"));
    }
    return in_memory_cache_
        .presorted_numerical_unique_values_columns_[column_idx]
        ->CreateIterator();
  }

  auto reader = absl::make_unique<ShardedFloatColumnReader>();
  RETURN_IF_ERROR(
      reader->Open(file::JoinPath(path_, kFilenameIndexed,
                                  absl::StrCat(kFilenameColumn, column_idx),
                                  kFilenameDeltaValueNoUnderscore),
                   /*max_num_values=*/options_.reading_buffer(),
                   /*begin_shard_idx=*/0,
                   /*end_shard_idx=*/1));
  return reader;
}

utils::StatusOr<std::unique_ptr<AbstractFloatColumnIterator>>
DatasetCacheReader::InOrderNumericalFeatureValueIterator(int column_idx) const {
  if (!meta_data().columns(column_idx).has_numerical()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Column ", column_idx, " is not numerical"));
  }

  if (options_.load_cache_in_memory()) {
    if (in_memory_cache_.inorder_numerical_columns_[column_idx] == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Column ", column_idx, " is not available"));
    }
    return in_memory_cache_.inorder_numerical_columns_[column_idx]
        ->CreateIterator();
  }

  auto reader = absl::make_unique<ShardedFloatColumnReader>();
  RETURN_IF_ERROR(
      reader->Open(file::JoinPath(path_, kFilenameRaw,
                                  absl::StrCat(kFilenameColumn, column_idx),
                                  kFilenameShardNoUnderscore),
                   /*max_num_values=*/options_.reading_buffer(),
                   /*begin_shard_idx=*/0,
                   /*end_shard_idx=*/meta_data_.num_shards_in_feature_cache()));
  return reader;
}

utils::StatusOr<std::unique_ptr<AbstractIntegerColumnIterator<CategoricalType>>>
DatasetCacheReader::InOrderCategoricalFeatureValueIterator(
    int column_idx) const {
  if (!meta_data().columns(column_idx).has_categorical()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Column ", column_idx, " is not categorical"));
  }

  if (options_.load_cache_in_memory()) {
    if (in_memory_cache_.inorder_categorical_columns_[column_idx] == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Column ", column_idx, " is not available"));
    }

    return in_memory_cache_.inorder_categorical_columns_[column_idx]
        ->CreateIterator();
  }

  auto reader =
      absl::make_unique<ShardedIntegerColumnReader<CategoricalType>>();
  RETURN_IF_ERROR(reader->Open(
      file::JoinPath(path_, kFilenameRaw,
                     absl::StrCat(kFilenameColumn, column_idx),
                     kFilenameShardNoUnderscore),
      /*max_value=*/meta_data_.columns(column_idx).categorical().num_values(),
      /*max_num_values=*/options_.reading_buffer(),
      /*begin_shard_idx=*/0,
      /*end_shard_idx=*/meta_data_.num_shards_in_feature_cache()));
  return reader;
}

utils::StatusOr<std::unique_ptr<AbstractIntegerColumnIterator<BooleanType>>>
DatasetCacheReader::InOrderBooleanFeatureValueIterator(int column_idx) const {
  if (!meta_data().columns(column_idx).has_boolean()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Column ", column_idx, " is not boolean"));
  }

  if (options_.load_cache_in_memory()) {
    if (in_memory_cache_.inorder_boolean_columns_[column_idx] == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Column ", column_idx, " is not available"));
    }

    return in_memory_cache_.inorder_boolean_columns_[column_idx]
        ->CreateIterator();
  }

  auto reader = absl::make_unique<ShardedIntegerColumnReader<BooleanType>>();
  RETURN_IF_ERROR(
      reader->Open(file::JoinPath(path_, kFilenameRaw,
                                  absl::StrCat(kFilenameColumn, column_idx),
                                  kFilenameShardNoUnderscore),
                   /*max_value=*/2,
                   /*max_num_values=*/options_.reading_buffer(),
                   /*begin_shard_idx=*/0,
                   /*end_shard_idx=*/meta_data_.num_shards_in_feature_cache()));
  return reader;
}

utils::StatusOr<std::unique_ptr<
    AbstractIntegerColumnIterator<DiscretizedIndexedNumericalType>>>
DatasetCacheReader::InOrderDiscretizedNumericalFeatureValueIterator(
    int column_idx) const {
  if (!meta_data().columns(column_idx).has_numerical()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Column ", column_idx, " is not numerical"));
  }

  if (!meta_data().columns(column_idx).numerical().discretized()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Column ", column_idx, " is not discretized"));
  }

  if (options_.load_cache_in_memory()) {
    if (in_memory_cache_.inorder_discretized_numerical_columns_[column_idx] ==
        nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Column ", column_idx, " is not available"));
    }

    return in_memory_cache_.inorder_discretized_numerical_columns_[column_idx]
        ->CreateIterator();
  }

  auto reader = absl::make_unique<
      ShardedIntegerColumnReader<DiscretizedIndexedNumericalType>>();
  RETURN_IF_ERROR(reader->Open(
      file::JoinPath(path_, kFilenameIndexed,
                     absl::StrCat(kFilenameColumn, column_idx),
                     kFilenameDiscretizedValuesNoUnderscore),
      /*max_value=*/
      meta_data_.columns(column_idx).numerical().num_discretized_values(),
      /*max_num_values=*/options_.reading_buffer(),
      /*begin_shard_idx=*/0,
      /*end_shard_idx=*/
      meta_data().columns(column_idx).numerical().num_discretized_shards()));
  return reader;
}

utils::StatusOr<std::unique_ptr<
    AbstractIntegerColumnIterator<DiscretizedIndexedNumericalType>>>
DatasetCacheReader::InOrderDiscretizedNumericalFeatureValueIterator(
    int column_idx, size_t begin_idx, size_t end_idx) const {
  if (!meta_data().columns(column_idx).has_numerical()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Column ", column_idx, " is not numerical"));
  }

  if (!meta_data().columns(column_idx).numerical().discretized()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Column ", column_idx, " is not discretized"));
  }

  if (options_.load_cache_in_memory()) {
    if (in_memory_cache_.inorder_discretized_numerical_columns_[column_idx] ==
        nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Column ", column_idx, " is not available"));
    }

    return in_memory_cache_.inorder_discretized_numerical_columns_[column_idx]
        ->CreateIterator(begin_idx, end_idx);
  }

  return absl::InvalidArgumentError(
      "InOrderDiscretizedNumericalFeatureValueIterator does not support "
      "begin/end index without in-memory caching");
}

const std::vector<float>&
DatasetCacheReader::DiscretizedNumericalFeatureBoundaries(
    int column_idx) const {
  return in_memory_cache_
      .boundaries_of_discretized_numerical_columns_[column_idx];
}

}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
namespace dataset {

using model::distributed_decision_tree::dataset_cache::kFilenameMetaDataPostfix;
using model::distributed_decision_tree::dataset_cache::kFilenamePartialMetaData;
using model::distributed_decision_tree::dataset_cache::PartialRawColumnFilePath;
using model::distributed_decision_tree::dataset_cache::proto::
    PartialColumnShardMetadata;
using model::distributed_decision_tree::dataset_cache::proto::
    PartialDatasetMetadata;

void PartialDatasetCacheDataSpecCreator::InferColumnsAndTypes(
    const std::vector<std::string>& paths,
    const proto::DataSpecificationGuide& guide,
    proto::DataSpecification* data_spec) {
  if (paths.size() != 1) {
    LOG(FATAL)
        << "The inference of dataspec on a partial dataset cache requires "
           " exactly one file path";
  }
  const auto& cache_path = paths.front();

  PartialDatasetMetadata partial_meta_data;
  CHECK_OK(
      file::GetBinaryProto(file::JoinPath(cache_path, kFilenamePartialMetaData),
                           &partial_meta_data, file::Defaults()));

  for (int col_idx = 0; col_idx < partial_meta_data.column_names_size();
       col_idx++) {
    const auto col_name = partial_meta_data.column_names(col_idx);

    // Load the column+shard meta-data.
    PartialColumnShardMetadata shard_meta_data;
    CHECK_OK(file::GetBinaryProto(
        absl::StrCat(
            PartialRawColumnFilePath(cache_path, col_idx, /*shard_idx=*/0),
            kFilenameMetaDataPostfix),
        &shard_meta_data, file::Defaults()));

    // Create the column name and type.
    proto::Column* column = data_spec->add_columns();
    column->set_name(col_name);
    switch (shard_meta_data.type_case()) {
      case PartialColumnShardMetadata::kNumerical:
        column->set_type(dataset::proto::ColumnType::NUMERICAL);
        break;
      case PartialColumnShardMetadata::kCategorical:
        column->set_type(dataset::proto::ColumnType::CATEGORICAL);
        column->mutable_categorical()->set_is_already_integerized(
            shard_meta_data.categorical().has_number_of_unique_values());
        break;
      case PartialColumnShardMetadata::TYPE_NOT_SET:
        break;
    }
  }
}

void PartialDatasetCacheDataSpecCreator::ComputeColumnStatisticsColumnAndShard(
    const int col_idx, const PartialColumnShardMetadata& shard_meta_data,
    proto::DataSpecification* data_spec,
    proto::DataSpecificationAccumulator* accumulator) {
  proto::Column* column = data_spec->mutable_columns(col_idx);
  auto* col_accumulator = accumulator->mutable_columns(col_idx);

  if (col_idx == 0) {
    // We only count the number of examples in the first columns.
    data_spec->set_created_num_rows(data_spec->created_num_rows() +
                                    shard_meta_data.num_examples());
  }
  column->set_count_nas(column->count_nas() +
                        shard_meta_data.num_missing_examples());

  switch (shard_meta_data.type_case()) {
    case PartialColumnShardMetadata::kNumerical:
      col_accumulator->set_kahan_sum(
          col_accumulator->kahan_sum() +
          shard_meta_data.numerical().mean() *
              (shard_meta_data.num_examples() -
               shard_meta_data.num_missing_examples()));

      if (!col_accumulator->has_min_value() ||
          shard_meta_data.numerical().min() < col_accumulator->min_value()) {
        col_accumulator->set_min_value(shard_meta_data.numerical().min());
      }

      if (!col_accumulator->has_max_value() ||
          shard_meta_data.numerical().max() > col_accumulator->max_value()) {
        col_accumulator->set_max_value(shard_meta_data.numerical().max());
      }
      break;

    case PartialColumnShardMetadata::kCategorical: {
      const auto& src_categorical = shard_meta_data.categorical();
      auto* dst_categorical = column->mutable_categorical();

      if (dst_categorical->is_already_integerized()) {
        // Maximum value of "number_of_unique_values" seen in all the shards.
        dst_categorical->set_number_of_unique_values(
            std::max(dst_categorical->number_of_unique_values(),
                     src_categorical.number_of_unique_values()));
      } else {
        for (const auto& src_item : src_categorical.items()) {
          auto it_dst = dst_categorical->mutable_items()->find(src_item.first);
          if (it_dst == dst_categorical->items().end()) {
            // A new item.
            (*dst_categorical->mutable_items())[src_item.first].set_count(
                src_item.second.count());
          } else {
            // Increase the count of the known item.
            it_dst->second.set_count(it_dst->second.count() +
                                     src_item.second.count());
          }
        }
      }
    } break;

    case PartialColumnShardMetadata::TYPE_NOT_SET:
      break;
  }
}

void PartialDatasetCacheDataSpecCreator::ComputeColumnStatistics(
    const std::vector<std::string>& paths,
    const proto::DataSpecificationGuide& guide,
    proto::DataSpecification* data_spec,
    proto::DataSpecificationAccumulator* accumulator) {
  DCHECK_EQ(paths.size(), 1);
  const auto& cache_path = paths.front();

  PartialDatasetMetadata partial_meta_data;
  CHECK_OK(
      file::GetBinaryProto(file::JoinPath(cache_path, kFilenamePartialMetaData),
                           &partial_meta_data, file::Defaults()));

  std::vector<int64_t> num_examples_per_columns(data_spec->columns_size(), 0);
  utils::concurrency::Mutex mutex_data;
  {
    utils::concurrency::ThreadPool thread_pool("InferDataspec",
                                               /*num_threads=*/20);
    thread_pool.StartWorkers();
    for (int col_idx = 0; col_idx < data_spec->columns_size(); col_idx++) {
      for (int shard_idx = 0; shard_idx < partial_meta_data.num_shards();
           shard_idx++) {
        const auto shard_meta_data_path = absl::StrCat(
            PartialRawColumnFilePath(cache_path, col_idx, shard_idx),
            kFilenameMetaDataPostfix);
        thread_pool.Schedule([shard_meta_data_path, &mutex_data, accumulator,
                              data_spec, col_idx, &num_examples_per_columns]() {
          PartialColumnShardMetadata shard_meta_data;
          CHECK_OK(file::GetBinaryProto(shard_meta_data_path, &shard_meta_data,
                                        file::Defaults()));

          utils::concurrency::MutexLock l(&mutex_data);

          num_examples_per_columns[col_idx] += shard_meta_data.num_examples();
          ComputeColumnStatisticsColumnAndShard(col_idx, shard_meta_data,
                                                data_spec, accumulator);
        });
      }
    }
  }

  if (!num_examples_per_columns.empty()) {
    for (int col_idx = 0; col_idx < num_examples_per_columns.size();
         col_idx++) {
      if (num_examples_per_columns[col_idx] !=
          num_examples_per_columns.front()) {
        LOG(FATAL) << "Invalid partial dataset cache: The different columns do "
                      "not have the same number of examples.";
      }
    }
  }
}

utils::StatusOr<int64_t> PartialDatasetCacheDataSpecCreator::CountExamples(
    absl::string_view path) {
  return absl::UnimplementedError("CountExamples not implemented");
}

}  // namespace dataset

}  // namespace yggdrasil_decision_forests
