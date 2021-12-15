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

// Support for dataset stored as CSV.
//
#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_CSV_EXAMPLE_READER_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_CSV_EXAMPLE_READER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/example_reader_interface.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/csv.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// Example reader from a csv file.
// This class is thread-compatible (you can use it from multiple threads at the
// same time, but you are in charge of the synchronization).
//
// Since the signature of "Next" is similar for "ShardedReader" and
// "ExampleReaderInterface", this class is implemented as inheritance +
// interface.
class CsvExampleReader final : public ExampleReaderInterface {
 public:
  explicit CsvExampleReader(
      const proto::DataSpecification& data_spec,
      absl::optional<std::vector<int>> ensure_non_missing);

  utils::StatusOr<bool> Next(proto::Example* example) override {
    return sharded_csv_reader_.Next(example);
  }

  absl::Status Open(absl::string_view sharded_path) override {
    return sharded_csv_reader_.Open(sharded_path);
  }

 private:
  class Implementation final : public utils::ShardedReader<proto::Example> {
   public:
    explicit Implementation(
        const proto::DataSpecification& data_spec,
        absl::optional<std::vector<int>> ensure_non_missing);

   protected:
    // Opens the .csv file at "path", and check that the header is as expected.
    absl::Status OpenShard(absl::string_view path) override;

    // Scans a new row in the csv file, and parses it as a proto:Example.
    utils::StatusOr<bool> NextInShard(proto::Example* example) override;

   private:
    // The data spec.
    const proto::DataSpecification data_spec_;

    // Currently open file;
    std::unique_ptr<yggdrasil_decision_forests::utils::csv::Reader> csv_reader_;
    file::InputFileCloser file_closer_;

    // Matching between data_spec column indices and csv field indices.
    std::vector<int> col_idx_to_field_idx_;

    // Header of the csv file.
    std::vector<std::string> csv_header_;

    const absl::optional<std::vector<int>> ensure_non_missing_;
  };

  Implementation sharded_csv_reader_;
};

REGISTER_ExampleReaderInterface(CsvExampleReader, "FORMAT_CSV");

class CsvDataSpecCreator : public AbstractDataSpecCreator {
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
};

REGISTER_AbstractDataSpecCreator(CsvDataSpecCreator, "FORMAT_CSV");

// Determine the most likely type of the attribute according to the current
// most likely value type and an observed string value.
proto::ColumnType InferType(const proto::DataSpecificationGuide& guide,
                            absl::string_view value,
                            const proto::Tokenizer& tokenizer,
                            proto::ColumnType previous_type);

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_CSV_EXAMPLE_READER_H_
