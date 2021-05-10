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

#include "yggdrasil_decision_forests/dataset/csv_example_writer.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/utils/csv.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace dataset {

CsvExampleWriter::Implementation::Implementation(
    const proto::DataSpecification& data_spec)
    : data_spec_(data_spec) {}

absl::Status CsvExampleWriter::Implementation::OpenShard(
    const absl::string_view path) {
  ASSIGN_OR_RETURN(auto file_handle, file::OpenOutputFile(path));
  csv_writer =
      absl::make_unique<yggdrasil_decision_forests::utils::csv::Writer>(
          file_handle.get());
  RETURN_IF_ERROR(file_closer_.reset(std::move(file_handle)));

  // Header.
  buffer_.resize(data_spec_.columns_size());
  for (int col_idx = 0; col_idx < data_spec_.columns_size(); col_idx++) {
    buffer_[col_idx] = data_spec_.columns(col_idx).name();
  }
  return csv_writer->WriteRow({buffer_.begin(), buffer_.end()});
}

absl::Status CsvExampleWriter::Implementation::WriteInShard(
    const proto::Example& example) {
  ExampleToCsvRow(example, data_spec_, &buffer_);
  return csv_writer->WriteRow({buffer_.begin(), buffer_.end()});
}

CsvExampleWriter::CsvExampleWriter(const proto::DataSpecification& data_spec)
    : sharded_csv_writer_(data_spec) {}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
