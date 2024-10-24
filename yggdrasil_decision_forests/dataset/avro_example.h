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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/avro.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example_reader_interface.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"

namespace yggdrasil_decision_forests::dataset::avro {

// Creates a dataspec from the Avro file.
absl::StatusOr<dataset::proto::DataSpecification> CreateDataspec(
    absl::string_view path,
    const dataset::proto::DataSpecificationGuide& guide);

absl::StatusOr<dataset::proto::DataSpecification> CreateDataspecImpl(
    std::unique_ptr<AvroReader> reader,
    const dataset::proto::DataSpecificationGuide& guide);

class AvroExampleReader final : public ExampleReaderInterface {
 public:
  explicit AvroExampleReader(const proto::DataSpecification& data_spec,
                             std::optional<std::vector<int>> required_columns)
      : sharded_reader_(data_spec, required_columns) {}

  absl::StatusOr<bool> Next(proto::Example* example) override {
    return sharded_reader_.Next(example);
  }

  absl::Status Open(absl::string_view sharded_path) override {
    return sharded_reader_.Open(sharded_path);
  }

 private:
  class Implementation final : public utils::ShardedReader<proto::Example> {
   public:
    explicit Implementation(
        const proto::DataSpecification& data_spec,
        const std::optional<std::vector<int>>& required_columns)
        : dataspec_(data_spec), required_columns_(required_columns) {}

   protected:
    // Opens the Avro file at "path", and check that the header is as expected.
    absl::Status OpenShard(absl::string_view path) override;

    // Scans a new row in the Avro file, and parses it as a proto:Example.
    absl::StatusOr<bool> NextInShard(proto::Example* example) override;

   private:
    // The data spec.
    const proto::DataSpecification dataspec_;

    // Currently, open file;
    std::unique_ptr<AvroReader> reader_;

    // Mapping between the Avro field index and the column index for the
    // univariate features. -1's are used for ignored fields.
    std::vector<int> univariate_field_idx_to_column_idx_;

    // Mapping between the Avro field index and the unstacked index for the
    // multivariate features. -1's are used for ignored fields.
    std::vector<int> multivariate_field_idx_to_unroll_idx_;

    const std::optional<std::vector<int>> required_columns_;
  };

  Implementation sharded_reader_;
};

REGISTER_ExampleReaderInterface(AvroExampleReader, "FORMAT_AVRO");

class AvroDataSpecCreator : public AbstractDataSpecCreator {
 public:
  absl::Status CreateDataspec(const std::vector<std::string>& paths,
                              const proto::DataSpecificationGuide& guide,
                              proto::DataSpecification* data_spec) override;

  absl::StatusOr<int64_t> CountExamples(absl::string_view path) override {
    return absl::UnimplementedError(
        "CountExamples not implemented for AVRO format");
  }
};

REGISTER_AbstractDataSpecCreator(AvroDataSpecCreator, "FORMAT_AVRO");

namespace internal {

// Infers the dataspec from the Avro file i.e. find the columns, but do not
// set the statistics.
absl::StatusOr<dataset::proto::DataSpecification> InferDataspec(
    const std::vector<AvroField>& fields,
    const dataset::proto::DataSpecificationGuide& guide,
    std::vector<proto::ColumnGuide>* unstacked_guides);
}  // namespace internal

}  // namespace yggdrasil_decision_forests::dataset::avro

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_AVRO_EXAMPLE_H_
