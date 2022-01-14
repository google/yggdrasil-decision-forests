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

// Reader utility for a cache created with dataset_cache.h.
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_DATASET_CACHE_READER_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_DATASET_CACHE_READER_H_

#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache.pb.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.pb.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace dataset_cache {

// Utility class to access a "dataset cache".
class DatasetCacheReader {
 public:
  // Creates the accessor utility class.
  //
  // Args:
  //   path: Path to the dataset cache i.e. the directory passed to the
  //     "cache_directory" argument of the CreateDatasetCacheFromShardedFiles
  //     dataset cache creation method.
  //   options: Configure how the cache is read. The default options are
  //     satisfying in most cases.
  //
  static utils::StatusOr<std::unique_ptr<DatasetCacheReader>> Create(
      absl::string_view path, const proto::DatasetCacheReaderOptions& options);

  ~DatasetCacheReader() {
    // TODO(gbm): Interrupt the non-blocking feature loading (if any).
    // CHECK_OK(WaitFeatureLoadingIsDone());
    if (non_blocking_.loading_thread) {
      non_blocking_.loading_thread->Join();
      non_blocking_.loading_thread.reset();
    }
  }

  // Number of examples in the cache.
  uint64_t num_examples() const;

  // Classification labels. Empty if there is not classification labels.
  const std::vector<ClassificationLabelType>& categorical_labels() const;

  // Regression labels. Empty if there is not regression labels.
  const std::vector<RegressionLabelType>& regression_labels() const;

  // Trainings weights. Empty if the training examples are not weighted.
  const std::vector<float>& weights() const;

  // Iterator over the delta-bit example indices orded according to the
  // "column_idx"-th numerical column.
  //
  // See the documentation of "MaskDeltaBit" and "MaskExampleIdx" for an
  // explanation of the "delta-bit example indices" concept.
  utils::StatusOr<
      std::unique_ptr<AbstractIntegerColumnIterator<ExampleIdxType>>>
  PresortedNumericalFeatureExampleIterator(int column_idx) const;

  // Iterator over the sorted unique values of the "column_idx"-th numerical
  // column.
  utils::StatusOr<std::unique_ptr<AbstractFloatColumnIterator>>
  PresortedNumericalFeatureValueIterator(int column_idx) const;

  // Iterator over the "column_idx"-th numerical column ordedd by example index.
  utils::StatusOr<std::unique_ptr<AbstractFloatColumnIterator>>
  InOrderNumericalFeatureValueIterator(int column_idx) const;

  // Iterator over the "column_idx"-th categorical column ordedd by example
  // index.
  utils::StatusOr<
      std::unique_ptr<AbstractIntegerColumnIterator<CategoricalType>>>
  InOrderCategoricalFeatureValueIterator(int column_idx) const;

  // Iterator over the "column_idx"-th boolean column ordedd by example index.
  utils::StatusOr<std::unique_ptr<AbstractIntegerColumnIterator<BooleanType>>>
  InOrderBooleanFeatureValueIterator(int column_idx) const;

  // Iterator over the "column_idx"-th discretized numerical column ordedd by
  // example index.
  utils::StatusOr<std::unique_ptr<
      AbstractIntegerColumnIterator<DiscretizedIndexedNumericalType>>>
  InOrderDiscretizedNumericalFeatureValueIterator(int column_idx) const;

  // Iterator over a subset of the "column_idx"-th discretized numerical column
  // ordedd by example index.
  utils::StatusOr<std::unique_ptr<
      AbstractIntegerColumnIterator<DiscretizedIndexedNumericalType>>>
  InOrderDiscretizedNumericalFeatureValueIterator(int column_idx,
                                                  size_t begin_idx,
                                                  size_t end_idx) const;

  // Discretization boundaries of the "column_idx"-th discretized numerical
  // column.
  const std::vector<float>& DiscretizedNumericalFeatureBoundaries(
      int column_idx) const;

  const proto::CacheMetadata& meta_data() const { return meta_data_; }

  // Compact human readable information about the metadata.
  std::string MetadataInformation() const;

  // Features, sorted by index value, available in the reader.
  std::vector<int> features() const { return features_; }

  // Tests if a feature is available in the reader.
  bool has_feature(int feature) const;

  // Load and unload a set of features.
  absl::Status LoadingAndUnloadingFeatures(
      const std::vector<int>& load_features,
      const std::vector<int>& unload_features);

  // Start loading and unloading a set of features asynchronously. Returns
  // immediately.
  //
  // Progresses can be checked regularly with
  // "CheckAndUpdateNonBlockingLoading()".
  //
  // Calling another method that changes the dataset reader (e.g.
  // "LoadingAndUnloadingFeatures") while features are being loaded will raise
  // an error.
  //
  // DatasetCacheReader's destructor will try to interrupt and wait for the
  // feature loading to be done. I.e. it is safe to destruct a
  // DatasetCacheReader that is loading features.
  //
  // Usage example:
  //
  //   DatasetCacheReader reader(features={0,1});
  //   reader.NonBlockingLoadingAndUnloadingFeatures(load_features={2,3},
  //      unload_features={1})
  //
  //   while(CheckAndUpdateNonBlockingLoading().value()) {
  //      // Use the features 0 and 1 in the reader.
  //   }
  //   // Can use the features 0, 2 and 3 in the reader.
  //
  absl::Status NonBlockingLoadingAndUnloadingFeatures(
      const std::vector<int>& load_features,
      const std::vector<int>& unload_features, const int num_threads = 10);

  // Indicates if features are currently loaded with
  // "NonBlockingLoadingAndUnloadingFeatures". This value is updated by
  // "CheckAndUpdateNonBlockingLoading()".
  bool IsNonBlockingLoadingInProgress();

  // Checks for the completion of the non-blocking dataset loading.
  // If no loading is in progress, return false.
  // If there is a loading in progress, return true.
  // If the loading in progress failed, return the error.
  // If the loading in progress just completed, finalize it (i.e. the user can
  // access the feature values and "NonBlockingLoadingInProgress()" will now
  // return false), and return false.
  utils::StatusOr<bool> CheckAndUpdateNonBlockingLoading();

  // Features being loaded. Empty if there are not features being pre-loaded at
  // this time.
  const std::vector<int>& NonBlockingLoadingInProgressLoadedFeatures() const;
  const std::vector<int>& NonBlockingLoadingInProgressUnloadedFeatures() const;

  // Duration of the initial loading of features in memory i.e. duration of
  // "InitializeAndLoadInMemoryCache".
  absl::Duration load_in_memory_duration() const {
    return load_in_memory_duration_;
  }

 private:
  DatasetCacheReader(absl::string_view path,
                     const proto::DatasetCacheReaderOptions& options)
      : path_(path), options_(options) {}

  // Initialize the internal structure and load the feature columns in RAM.
  absl::Status InitializeAndLoadInMemoryCache();

  // Loads a single column in RAM.
  absl::Status LoadInMemoryCacheColumn(int column_idx, size_t* memory_usage);

  // Unloads a single column from RAM.
  absl::Status UnloadInMemoryCacheColumn(int column_idx);

  // Updates the meta-data to make the specified features available. Note: This
  // method is not in charge of actually loading/unloading the features (i.e.
  // LoadInMemoryCacheColumn and UnloadInMemoryCacheColumn). Instead, the
  // feature loading/unload should already have been done.
  absl::Status ApplyLoadingAndUnloadingFeaturesToMetadata(
      const std::vector<int>& load_features,
      const std::vector<int>& unload_features);

  std::string path_;
  proto::CacheMetadata meta_data_;
  proto::DatasetCacheReaderOptions options_;

  // Example weights. Empty is the examples are not weighted.
  std::vector<float> weights_;

  // Classification label values. Empty if the dataset does not have a
  // classification label.
  std::vector<ClassificationLabelType> classification_labels_;

  // Regression label values. Empty if the dataset does not have a
  // regression label.
  std::vector<RegressionLabelType> regression_labels_;

  // List of the features available for reading. Sorted in increasing order.
  std::vector<int> features_;

  absl::Duration load_in_memory_duration_;

  struct NonBlocking {
    // Loading thread.
    std::unique_ptr<utils::concurrency::Thread> loading_thread;

    // Status of the loading thread. True set by the manager. False set by the
    // thread.
    std::atomic_bool is_running{false};

    absl::Status status;
    utils::concurrency::Mutex status_mutex;

    // Features being loaded / unloaded.
    std::vector<int> load_features;
    std::vector<int> unload_features;
  };
  NonBlocking non_blocking_;

  struct {
    // Sorted numerical.
    std::vector<std::unique_ptr<InMemoryFloatColumnReaderFactory>>
        inorder_numerical_columns_;
    std::vector<
        std::unique_ptr<InMemoryIntegerColumnReaderFactory<ExampleIdxType>>>
        presorted_numerical_example_idx_columns_;
    std::vector<std::unique_ptr<InMemoryFloatColumnReaderFactory>>
        presorted_numerical_unique_values_columns_;

    // Discretized numerical
    std::vector<std::unique_ptr<
        InMemoryIntegerColumnReaderFactory<DiscretizedIndexedNumericalType>>>
        inorder_discretized_numerical_columns_;
    std::vector<std::vector<float>>
        boundaries_of_discretized_numerical_columns_;

    // Categorical.
    std::vector<
        std::unique_ptr<InMemoryIntegerColumnReaderFactory<CategoricalType>>>
        inorder_categorical_columns_;

    // Boolean.
    std::vector<
        std::unique_ptr<InMemoryIntegerColumnReaderFactory<BooleanType>>>
        inorder_boolean_columns_;
  } in_memory_cache_;
};

}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model

namespace dataset {

// Creates a dataspec for a partial dataset cache.
//
// See "dataset_cache.h" for an explanation of the dataset cache.
class PartialDatasetCacheDataSpecCreator : public AbstractDataSpecCreator {
 public:
  void InferColumnsAndTypes(const std::vector<std::string>& paths,
                            const proto::DataSpecificationGuide& guide,
                            proto::DataSpecification* data_spec) override;

  void ComputeColumnStatistics(
      const std::vector<std::string>& paths,
      const proto::DataSpecificationGuide& guide,
      proto::DataSpecification* data_spec,
      proto::DataSpecificationAccumulator* accumulator) override;

  utils::StatusOr<int64_t> CountExamples(absl::string_view path) override;

 private:
  // Compute the statistics of a single column on a single shard.
  static void ComputeColumnStatisticsColumnAndShard(
      int col_idx,
      const model::distributed_decision_tree::dataset_cache::proto::
          PartialColumnShardMetadata& shard_meta_data,
      proto::DataSpecification* data_spec,
      proto::DataSpecificationAccumulator* accumulator);
};

REGISTER_AbstractDataSpecCreator(PartialDatasetCacheDataSpecCreator,
                                 "FORMAT_PARTIAL_DATASET_CACHE");

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_DATASET_CACHE_READER_H_
