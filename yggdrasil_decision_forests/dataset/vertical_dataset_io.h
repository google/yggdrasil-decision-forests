/*
 * Copyright 2022 Google LLC.
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

// Definition of VerticalDataset, a in-memory storage of data column by column.
//
// Note: VerticalDataset is used to in the implementation of the original Random
// Forest algorithm (see ../algorithm/random_forest.h).
//

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_VERTICAL_DATASET_IO_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_VERTICAL_DATASET_IO_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// Advanced options for dataset loading.
struct LoadConfig {
  // Number of reading threads. Only used for multi-sharded datasets.
  // num_threads=1 is more memory efficient than num_threads>1.
  int num_threads = 10;
  // If specified, only load this subset of columns.
  std::optional<std::vector<int>> load_columns;
  // If specified, only load the examples that evaluate to true.
  std::optional<std::function<bool(const proto::Example&)>> load_example;
  // If set and true, stop the data ingestion.
  std::atomic<bool>* stop = nullptr;
};

// Load the dataset content from a file (or a set of files).
//
// If "required_columns" is not provided, all the columns specified in
// "data_spec" should be present in the dataset (some of the values can still
// be missing). If "required_columns" is provided, only the columns in
// "required_columns" should be present. If a column non defined in
// "required_columns" is not present, all its value will be assumed to be
// "missing".
//
// Args:
//   typed_path: Path to the dataset with a prefix format. For example
//     "csv:/tmp/dataset.csv". Support sharding, globbing and comma separate
//     paths.
//   data_spec: Definition of the dataset columns. dataset: Where to store the
//     dataset. required_columns: Control if non existing features are
//     allowed. See comment above for the specific semantic of
//     "required_columns".
//   config: Optional additional configuration options.
//
absl::Status LoadVerticalDataset(
    absl::string_view typed_path, const proto::DataSpecification& data_spec,
    VerticalDataset* dataset,
    const std::optional<std::vector<int>>& required_columns = {},
    const LoadConfig& config = {});

// Save the dataset to a file (or a set of files). If
// num_records_by_shard==-1, all the examples will be written in the first
// shard.
absl::Status SaveVerticalDataset(const VerticalDataset& dataset,
                                 const absl::string_view typed_path,
                                 int64_t num_records_by_shard = -1);

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_VERTICAL_DATASET_IO_H_
