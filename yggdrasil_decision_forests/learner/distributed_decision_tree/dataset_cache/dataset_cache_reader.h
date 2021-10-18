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
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache.pb.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"
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
  //     satifying in most cases.
  //
  static utils::StatusOr<std::unique_ptr<DatasetCacheReader>> Create(
      absl::string_view path, const proto::DatasetCacheReaderOptions& options);

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

 private:
  DatasetCacheReader(absl::string_view path,
                     const proto::DatasetCacheReaderOptions& options)
      : path_(path), options_(options) {}

  absl::Status LoadInMemoryCache();

  absl::Status LoadLoadInMemoryCacheColumn(int column_idx,
                                           size_t* memory_usage);

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
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_DATASET_CACHE_READER_H_
