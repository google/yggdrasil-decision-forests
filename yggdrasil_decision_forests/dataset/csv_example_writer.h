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

// Example writer from CSV dataset format.

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_CSV_EXAMPLE_WRITER_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_CSV_EXAMPLE_WRITER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/example_writer_interface.h"
#include "yggdrasil_decision_forests/utils/csv.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// Example writer for csv files.
class CsvExampleWriter final : public ExampleWriterInterface {
 public:
  explicit CsvExampleWriter(const proto::DataSpecification& data_spec);

  absl::Status Write(const proto::Example& example) override {
    return sharded_csv_writer_.Write(example);
  }

  absl::Status Open(absl::string_view sharded_path,
                    const int64_t num_records_by_shard) override {
    return sharded_csv_writer_.Open(sharded_path, num_records_by_shard);
  }

 private:
  class Implementation final : public utils::ShardedWriter<proto::Example> {
   public:
    explicit Implementation(const proto::DataSpecification& data_spec);

    absl::Status CloseWithStatus() { return file_closer_.Close(); };

   protected:
    absl::Status OpenShard(absl::string_view path) final;
    absl::Status WriteInShard(const proto::Example& example) final;

   private:
    // The data spec.
    const proto::DataSpecification data_spec_;

    // Currently open file;
    std::unique_ptr<yggdrasil_decision_forests::utils::csv::Writer> csv_writer;
    file::OutputFileCloser file_closer_;

    std::vector<std::string> buffer_;
  };

  Implementation sharded_csv_writer_;
};

REGISTER_ExampleWriterInterface(CsvExampleWriter, "FORMAT_CSV");

// Converts a proto::Example into an array of string that can be saved in a csv
// file. The output "csv_fields[i]" is the string representation of the "i-th"
// column of "example".
void ExampleToCsvRow(const proto::Example& example,
                     const proto::DataSpecification& data_spec,
                     std::vector<std::string>* csv_fields);

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_CSV_EXAMPLE_WRITER_H_
