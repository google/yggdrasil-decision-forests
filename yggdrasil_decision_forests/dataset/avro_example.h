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

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"

namespace yggdrasil_decision_forests::dataset::avro {

enum class AvroType {
  kUnknown = 0,
  kNull = 1,
  kBoolean = 2,
  kInt = 3,
  kLong = 4,
  kFloat = 5,
  kDouble = 6,
  kString = 7,
  kBytes = 8,
  kArray = 9,
};

struct AvroField {
  std::string name;
  AvroType type;
  AvroType sub_type = AvroType::kUnknown;  // Only used if type==kArray.
  bool optional = false;  // If the field is ["null", <something> ].

  bool operator==(const AvroField& other) const {
    return name == other.name && type == other.type &&
           optional == other.optional && sub_type == other.sub_type;
  }

  friend std::ostream& operator<<(std::ostream& os, const AvroField& field);
};

// Class to read Avro files.
// Avro 1.12.0 file format:
// https://avro.apache.org/docs/1.12.0/specification/
class AvroReader {
 public:
  static absl::StatusOr<std::unique_ptr<AvroReader>> Create(
      absl::string_view path);

  static absl::StatusOr<std::string> ExtractSchema(absl::string_view path);

  absl::Status Close();

  ~AvroReader();

  const std::vector<AvroField>& fields() const { return fields_; }

  const std::string& sync_marker() const { return sync_marker_; }

 private:
  AvroReader(std::unique_ptr<utils::InputByteStream>&& stream);

  absl::StatusOr<std::string> ReadHeader();

  std::unique_ptr<utils::InputByteStream> stream_;
  std::vector<AvroField> fields_;
  std::string sync_marker_;
};

namespace internal {

absl::StatusOr<std::string> ReadString(utils::InputByteStream* stream);
absl::StatusOr<size_t> ReadInteger(utils::InputByteStream* stream);
absl::StatusOr<double> ReadDouble(utils::InputByteStream* stream);
absl::StatusOr<float> ReadFloat(utils::InputByteStream* stream);
absl::StatusOr<bool> ReadBoolean(utils::InputByteStream* stream);

}  // namespace internal

}  // namespace yggdrasil_decision_forests::dataset::avro

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_AVRO_EXAMPLE_H_
