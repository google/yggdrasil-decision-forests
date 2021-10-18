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

// Remote worker for the computation of a dataset cache.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_DATASET_CACHE_WORKER_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_DATASET_CACHE_WORKER_H_

#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache.pb.h"
#include "yggdrasil_decision_forests/learner/types.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace dataset_cache {

class CreateDatasetCacheWorker : public distribute::AbstractWorker {
 public:
  static constexpr char kWorkerKey[] = "CREATE_DATASET_CACHE_WORKER";

  absl::Status Setup(distribute::Blob serialized_welcome) override;

  utils::StatusOr<distribute::Blob> RunRequest(
      distribute::Blob serialized_request) override;

  absl::Status Done() override { return absl::OkStatus(); }

 private:
  absl::Status SeparateDatasetColumns(
      const proto::WorkerRequest::SeparateDatasetColumns& request,
      proto::WorkerResult::SeparateDatasetColumns* result);

  absl::Status SeparateDatasetColumn(const dataset::VerticalDataset& dataset,
                                     int column_idx, int shard_idx,
                                     const int num_shards,
                                     absl::string_view temp_directory,
                                     absl::string_view output_directory);

  absl::Status ConvertPartialToFinalRawData(
      const proto::WorkerRequest::ConvertPartialToFinalRawData& request,
      proto::WorkerResult::ConvertPartialToFinalRawData* result);

  absl::Status SortNumericalColumn(
      const proto::WorkerRequest::SortNumericalColumn& request,
      proto::WorkerResult::SortNumericalColumn* result);

  // Export sorted numerical values. Used by "SortNumericalColumn".
  absl::Status ExportSortedNumericalColumn(
      const proto::WorkerRequest::SortNumericalColumn& request,
      const std::vector<std::pair<float, model::SignedExampleIdx>>&
          value_and_example_idxs,
      proto::WorkerResult::SortNumericalColumn* result);

  // Export discretized numerical values. Used by "SortNumericalColumn".
  absl::Status ExportSortedDiscretizedNumericalColumn(
      const proto::WorkerRequest::SortNumericalColumn& request,
      const std::vector<std::pair<float, model::SignedExampleIdx>>&
          value_and_example_idxs,
      int64_t num_unique_values,
      proto::WorkerResult::SortNumericalColumn* result);

  proto::WorkerWelcome welcome_;
};

}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model

namespace distribute {
using CreateDatasetCacheWorker =
    model::distributed_decision_tree::dataset_cache::CreateDatasetCacheWorker;
REGISTER_Distribution_Worker(CreateDatasetCacheWorker,
                             CreateDatasetCacheWorker::kWorkerKey);
}  // namespace distribute

}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_DATASET_CACHE_WORKER_H_
