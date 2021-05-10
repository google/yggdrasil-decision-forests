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

#include "yggdrasil_decision_forests/dataset/example_writer.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/example_writer_interface.h"
#include "yggdrasil_decision_forests/dataset/formats.h"
#include "yggdrasil_decision_forests/dataset/formats.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace dataset {

utils::StatusOr<std::unique_ptr<ExampleWriterInterface>> CreateExampleWriter(
    absl::string_view typed_path, const proto::DataSpecification& data_spec,
    const int64_t num_records_by_shard) {
  std::string sharded_path;
  proto::DatasetFormat format;
  std::tie(sharded_path, format) = GetDatasetPathAndType(typed_path);

  const std::string& format_name = proto::DatasetFormat_Name(format);
  ASSIGN_OR_RETURN(
      auto writer,
      ExampleWriterInterfaceRegisterer::Create(format_name, data_spec),
      _ << "When creating an example writer to create " << sharded_path
        << ". Make sure the format dependency is linked");
  RETURN_IF_ERROR(writer->Open(sharded_path, num_records_by_shard));
  return std::move(writer);
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
