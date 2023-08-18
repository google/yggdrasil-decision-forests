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

// Library of readers for supported dataset formats.
//
// All the "in-order" readers are guaranteed to read examples in the same order
// (even for sharded files).
//
// Supported readers are:
//   - CreateExampleReader: Sequential "in-order" local reading.
//
//
// See proto::DatasetFormat for a list of supported dataset format.
//
// Usage example:
//
//   proto::DataSpecification data_spec = ...
//   const string dataset_path = "sstable+tfe:/path_to_dataset.tfe-sstable@2";
//   auto reader = CreateExampleReader(dataset_path, data_spec);
//   proto::Example example;
//   while (reader->Next(&example)) {
//     ... do something with "example"...
//   }

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_READER_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_READER_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example_reader_interface.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// Creates a stream reader of proto::Example from any of the supported dataset
// input formats. Supported formats are:
//   - csv
//   - sstable of tf.Examples
//   - recordio of tf.Examples
//
// If "required_columns" is not provided, all the columns specified in
// "data_spec" should be present in the dataset (some of the values can still
// be missing). If "required_columns" is provided, only the columns in
// "required_columns" should be present. If a column non defined in
// "required_columns" is not present, all its value will be assumed to be
// "missing".
//
absl::StatusOr<std::unique_ptr<ExampleReaderInterface>> CreateExampleReader(
    absl::string_view typed_path, const proto::DataSpecification& data_spec,
    const absl::optional<std::vector<int>>& required_columns = {});

// Checks if the format of a typed dataset is supported i.e. a dataset reader is
// registered for this format. Returns true, if the format is supported. Returns
// false if the format is not supported. Returns an error if the typed path
// cannot be parsed. Note: This function does not read the target file.
absl::StatusOr<bool> IsFormatSupported(absl::string_view typed_path);

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_READER_H_
