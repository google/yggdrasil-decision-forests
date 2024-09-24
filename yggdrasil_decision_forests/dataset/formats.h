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

// Dataset formats.
//
// Naming:
//   Typed path: A string of the form "[dataset type]:[dataset path]".

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_FORMATS_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_FORMATS_H_

#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/formats.pb.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// File prefixes indicative of the file format.
const char* const FORMAT_CSV = "csv";
const char* const FORMAT_TFE_TFRECORD = "tfrecord+tfe";
const char* const FORMAT_TFE_TFRECORDV2 = "tfrecordv2+tfe";
const char* const FORMAT_PARTIAL_DATASET_CACHE = "partial_dataset_cache";

// Splits the format and path from a typed path.
std::pair<std::string, proto::DatasetFormat> GetDatasetPathAndType(
    absl::string_view typed_path);

// Same as "GetDatasetPathAndType", but return a status in case of error.
absl::StatusOr<std::pair<std::string, proto::DatasetFormat>>
GetDatasetPathAndTypeOrStatus(absl::string_view typed_path);

// Tests if a string is a typed path.
bool IsTypedPath(absl::string_view maybe_typed_path);

// If the given path is already typed, return it. Otherwise, try to convert it
// into a typed path or fail if this does not work.
absl::StatusOr<std::string> GetTypedPath(const std::string& path);

// Splits a "[type]:[path]" into a pair of strings "type" and "path". Do not
// check that "type" is a valid type of "path" a valid path.
absl::StatusOr<std::pair<std::string, std::string>> SplitTypeAndPath(
    absl::string_view typed_path);

// Gets the recommended file extension for a dataset format.
std::string FormatToRecommendedExtension(proto::DatasetFormat format);

// Formats to type prefix.
std::string DatasetFormatToPrefix(proto::DatasetFormat format);

// Returns comma-separated supported formats.
std::string ListSupportedFormats();

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_FORMATS_H_
