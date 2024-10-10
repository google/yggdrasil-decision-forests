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
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace dataset {

struct Format {
  absl::string_view extension;
  absl::string_view prefix;
  absl::string_view prefix_alias = "";
  proto::DatasetFormat proto_format;
};

const std::vector<Format>& GetFormats() {
  static const std::vector<Format>* formats = []() {
    auto* formats = new std::vector<Format>();

    // RFC 4180-compliant CSV file.
    formats->push_back({
        .extension = "csv",
        .prefix = FORMAT_CSV,
        .proto_format = proto::FORMAT_CSV,
    });

    // GZip compressed TF Record of binary serialized TensorFlow.Example proto.
    // Use TensorFlow API to read the file (required TensorFlow C++ to be
    // linked). Deprecated: Use FORMAT_TFE_TFRECORD_COMPRESSED_V2 instead.
    formats->push_back({
        .extension = "tfrecord",
        .prefix = FORMAT_TFE_TFRECORD,
        .proto_format = proto::FORMAT_TFE_TFRECORD,
    });

    // Uncompressed TF Record of binary serialized TensorFlow.Example proto.
    // Does not require TensorFlow C++ to be linked.
    formats->push_back({
        .extension = "tfrecord",
        .prefix = FORMAT_TFE_TFRECORDV2,
        .prefix_alias = "tfrecord-nocompression",
        .proto_format = proto::FORMAT_TFE_TFRECORDV2,
    });

    // GZip compressed TF Record of binary serialized TensorFlow.Example proto.
    // Does not require TensorFlow C++ to be linked.
    formats->push_back({
        .extension = "tfrecord",
        .prefix = "tfrecord",
        .prefix_alias = "tfrecordv2+gz+tfe",
        .proto_format = proto::FORMAT_TFE_TFRECORD_COMPRESSED_V2,
    });

    formats->push_back({
        .extension = "avro",
        .prefix = "avro",
        .proto_format = proto::FORMAT_AVRO,
    });

    // Partially computed (e.g. non indexed) dataset cache.
    formats->push_back({
        .extension = "partial_dataset_cache",
        .prefix = FORMAT_PARTIAL_DATASET_CACHE,
        .proto_format = proto::FORMAT_PARTIAL_DATASET_CACHE,
    });

    return formats;
  }();
  return *formats;
}

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

std::string FormatToRecommendedExtension(proto::DatasetFormat proto_format) {
  for (const auto& format : GetFormats()) {
    if (format.proto_format == proto_format) {
      return std::string(format.extension);
    }
  }
  return "";
}

absl::StatusOr<proto::DatasetFormat> PrefixToFormat(
    const absl::string_view prefix) {
  for (const auto& format : GetFormats()) {
    if (format.prefix == prefix || format.prefix_alias == prefix) {
      return format.proto_format;
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("The format prefix \"", prefix,
                   "\" is unknown. Make sure the format reader is linked to "
                   "the binary."));
}

absl::StatusOr<std::pair<std::string, proto::DatasetFormat>>
GetDatasetPathAndTypeOrStatus(const absl::string_view typed_path) {
  std::string path, prefix;
  ASSIGN_OR_RETURN(std::tie(prefix, path), SplitTypeAndPath(typed_path));
  ASSIGN_OR_RETURN(const auto format, PrefixToFormat(prefix));
  return std::make_pair(std::string(path), format);
}

std::string DatasetFormatToPrefix(proto::DatasetFormat proto_format) {
  for (const auto& format : GetFormats()) {
    if (format.proto_format == proto_format) {
      return std::string(format.prefix);
    }
  }
  return "unknown";
}

std::string ListSupportedFormats() {
  std::vector<std::string> supported_prefixes;
  for (const auto& format : GetFormats()) {
    supported_prefixes.push_back(std::string(format.prefix));
  }
  return absl::StrJoin(supported_prefixes, ", ");
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
