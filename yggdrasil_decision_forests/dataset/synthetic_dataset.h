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

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_SYNTHETIC_DATASET_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_SYNTHETIC_DATASET_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/synthetic_dataset.pb.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// Generates a synthetic dataset.
// See "synthetic_dataset.cc" header.
absl::Status GenerateSyntheticDataset(
    const proto::SyntheticDatasetOptions& options,
    absl::string_view typed_path);

// Generates three datasets with the same underlying patters. The terms "train",
// "valid" and "test" are indicative as there are no differences between those
// datasets. "typed_path_valid" can be empty iif. "ratio_valid==0".
//
// The ratio of examples in the train dataset is "1 - ratio_valid - ratio_test".
absl::Status GenerateSyntheticDatasetTrainValidTest(
    const proto::SyntheticDatasetOptions& options,
    absl::string_view typed_path_train, absl::string_view typed_path_valid,
    absl::string_view typed_path_test, float ratio_valid, float ratio_test);

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_SYNTHETIC_DATASET_H_
