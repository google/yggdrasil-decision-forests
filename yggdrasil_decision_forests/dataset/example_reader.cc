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

#include "yggdrasil_decision_forests/dataset/example_reader.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example_reader_interface.h"
#include "yggdrasil_decision_forests/dataset/formats.h"
#include "yggdrasil_decision_forests/dataset/formats.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace dataset {

absl::StatusOr<std::unique_ptr<ExampleReaderInterface>> CreateExampleReader(
    const absl::string_view typed_path,
    const proto::DataSpecification& data_spec,
    const absl::optional<std::vector<int>>& required_columns) {
  std::string sharded_path;
  proto::DatasetFormat format;
  ASSIGN_OR_RETURN(std::tie(sharded_path, format),
                   GetDatasetPathAndTypeOrStatus(typed_path));

  const std::string& format_name = proto::DatasetFormat_Name(format);
  ASSIGN_OR_RETURN(auto reader,
                   ExampleReaderInterfaceRegisterer::Create(
                       format_name, data_spec, required_columns),
                   _ << "When creating an example reader to read "
                     << sharded_path
                     << ". Make sure the format dependency is linked");
  RETURN_IF_ERROR(reader->Open(sharded_path));
  return std::move(reader);
}

absl::StatusOr<bool> IsFormatSupported(absl::string_view typed_path) {
  const auto path_format_or = GetDatasetPathAndTypeOrStatus(typed_path);
  if (!path_format_or.ok()) {
    LOG(WARNING) << "Cannot parse typed path: " << path_format_or.status();
    return false;
  }
  std::string sharded_path;
  proto::DatasetFormat format;
  std::tie(sharded_path, format) = path_format_or.value();
  const std::string& format_name = proto::DatasetFormat_Name(format);
  return ExampleReaderInterfaceRegisterer::IsName(format_name);
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
