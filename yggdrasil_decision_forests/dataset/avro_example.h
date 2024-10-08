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

// Utility to read Avro files.

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_AVRO_EXAMPLE_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_AVRO_EXAMPLE_H_

#include <cstddef>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/avro.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"

namespace yggdrasil_decision_forests::dataset::avro {

// Creates a dataspec from the Avro file.
absl::StatusOr<dataset::proto::DataSpecification> CreateDataspec(
    absl::string_view path, dataset::proto::DataSpecificationGuide& guide);

namespace internal {

// Infers the dataspec from the Avro file i.e. find the columns, but do not
// set the statistics.
absl::StatusOr<dataset::proto::DataSpecification> InferDataspec(
    const std::vector<AvroField>& fields,
    dataset::proto::DataSpecificationGuide& guide,
    std::vector<proto::ColumnGuide>* unstacked_guides);
}  // namespace internal

}  // namespace yggdrasil_decision_forests::dataset::avro

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_AVRO_EXAMPLE_H_
