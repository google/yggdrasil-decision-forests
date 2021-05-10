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

// Library for the writing of proto::Example on drive.
//
// This library works similarly to :example_reader.
//
// Currently supported formats:
//   - CSV
//
// Usage example:
//
//  proto::DataSpecification data_spec = ...
//  const string dataset_path = "csv:/path_to_dataset.csv";
//  ASSIGN_OR_RETURN(auto writer, CreateExampleWriter(dataset_path, data_spec));
//  proto::Example example;
//  for(const proto::Example& example : list_of_examples) {
//    writer.Write(example);
//  }

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_WRITER_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_WRITER_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/example/example.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/example_writer_interface.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// Creates a stream writer of proto::Example from any of the supported dataset
// output formats. If num_records_by_shard==-1, all the examples are written in
// the first shard.
utils::StatusOr<std::unique_ptr<ExampleWriterInterface>> CreateExampleWriter(
    absl::string_view typed_path, const proto::DataSpecification& data_spec,
    int64_t num_records_by_shard = -1);

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_WRITER_H_
