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

#include "yggdrasil_decision_forests/dataset/formats.h"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/formats.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace dataset {

using proto::DatasetFormat;

absl::StatusOr<std::pair<std::string, std::string>> SplitTypeAndPath(
    const absl::string_view typed_path) {
  const int sep_pos = typed_path.find_first_of(':');
  if (sep_pos == -1) {
    return absl::InvalidArgumentError(
        absl::Substitute("Cannot parse \"$0\" as \"type:path\"", typed_path));
  }
  const auto path = typed_path.substr(sep_pos + 1);
  const auto format_str = typed_path.substr(0, sep_pos);
  return std::make_pair(std::string(format_str), std::string(path));
}

bool IsTypedPath(const absl::string_view maybe_typed_path) {
  return SplitTypeAndPath(maybe_typed_path).ok();
}

absl::StatusOr<std::string> GetTypedPath(const std::string& path) {
  if (dataset::IsTypedPath(path)) {
    return path;
  }
  if (absl::EndsWith(path, ".csv")) {
    return absl::StrCat("csv:", path);
  } else {
    return absl::InvalidArgumentError(
        absl::Substitute("Could not determine file type of $0. Please "
                         "provide a typed path, e.g. csv:/path/to/my/file \n"
                         "Supported formats: $1",
                         path, ListSupportedFormats()));
  }
}

std::pair<std::string, proto::DatasetFormat> GetDatasetPathAndType(
    const absl::string_view typed_path) {
  return GetDatasetPathAndTypeOrStatus(typed_path).value();
}

absl::StatusOr<std::pair<std::string, proto::DatasetFormat>>
GetDatasetPathAndTypeOrStatus(const absl::string_view typed_path) {
  std::string path, prefix;
  std::tie(prefix, path) = SplitTypeAndPath(typed_path).value();

  static const google::protobuf::EnumDescriptor* enum_descriptor =
      google::protobuf::GetEnumDescriptor<DatasetFormat>();
  for (int format_idx = 0; format_idx < enum_descriptor->value_count();
       format_idx++) {
    const auto format = static_cast<DatasetFormat>(
        enum_descriptor->value(format_idx)->number());
    if (format == proto::INVALID) {
      continue;
    }
    if (DatasetFormatToPrefix(format) == prefix) {
      return std::make_pair(std::string(path), format);
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown format \"", prefix, "\" in \"", typed_path, "\""));
}

std::string FormatToRecommendedExtension(proto::DatasetFormat format) {
  switch (format) {
    case proto::INVALID:
      LOG(FATAL) << "Invalid format";
      break;
    case proto::FORMAT_CSV:
      return "csv";
    case proto::FORMAT_TFE_TFRECORD:
      return "tfrecord";
    case proto::FORMAT_TFE_TFRECORDV2:
      return "tfrecordv2";
    case proto::FORMAT_PARTIAL_DATASET_CACHE:
      return "partial_dataset_cache";
  }
}

std::string DatasetFormatToPrefix(proto::DatasetFormat format) {
  switch (format) {
    case proto::INVALID:
      LOG(FATAL) << "Invalid format";
      break;
    case proto::FORMAT_CSV:
      return FORMAT_CSV;
    case proto::FORMAT_TFE_TFRECORD:
      return FORMAT_TFE_TFRECORD;
    case proto::FORMAT_TFE_TFRECORDV2:
      return FORMAT_TFE_TFRECORDV2;
    case proto::FORMAT_PARTIAL_DATASET_CACHE:
      return FORMAT_PARTIAL_DATASET_CACHE;
  }
}

std::string ListSupportedFormats() {
  std::vector<std::string> supported_prefixes;
  static const google::protobuf::EnumDescriptor* enum_descriptor =
      google::protobuf::GetEnumDescriptor<DatasetFormat>();
  for (int i = 0; i < enum_descriptor->value_count(); i++) {
    const google::protobuf::EnumValueDescriptor* format_idx = enum_descriptor->value(i);
    const auto format = static_cast<DatasetFormat>(format_idx->number());
    if (format != proto::INVALID) {
      supported_prefixes.push_back(DatasetFormatToPrefix(format));
    }
  }
  return absl::StrJoin(supported_prefixes, ", ");
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
