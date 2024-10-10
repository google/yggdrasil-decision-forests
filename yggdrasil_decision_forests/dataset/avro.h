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

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_AVRO_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_AVRO_H_

#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
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

std::string TypeToString(AvroType type);
absl::StatusOr<AvroType> ParseType(absl::string_view key);

enum class AvroCodec {
  kNull = 0,
  kDeflate = 1,
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
  // Creates a reader for the given Avro file.
  static absl::StatusOr<std::unique_ptr<AvroReader>> Create(
      absl::string_view path);

  // Extracts the schema of the given Avro file.
  static absl::StatusOr<std::string> ExtractSchema(absl::string_view path);

  // Reads the next record. Returns false if the end of the file is reached.
  absl::StatusOr<bool> ReadNextRecord();

  // Skips all the fields (i.e., read with ReadNextField*).
  absl::Status SkipAllFieldsInRecord();

  // Reads the next field. Returns nullopt if the field is optional and not set.
  absl::StatusOr<absl::optional<bool>> ReadNextFieldBoolean(
      const AvroField& field);
  absl::StatusOr<absl::optional<int64_t>> ReadNextFieldInteger(
      const AvroField& field);
  absl::StatusOr<absl::optional<float>> ReadNextFieldFloat(
      const AvroField& field);
  absl::StatusOr<absl::optional<double>> ReadNextFieldDouble(
      const AvroField& field);
  absl::StatusOr<bool> ReadNextFieldString(const AvroField& field,
                                           std::string* value);
  absl::StatusOr<bool> ReadNextFieldArrayFloat(const AvroField& field,
                                               std::vector<float>* values);
  absl::StatusOr<bool> ReadNextFieldArrayDouble(const AvroField& field,
                                                std::vector<double>* values);
  absl::StatusOr<bool> ReadNextFieldArrayDoubleIntoFloat(
      const AvroField& field, std::vector<float>* values);
  absl::StatusOr<bool> ReadNextFieldArrayString(
      const AvroField& field, std::vector<std::string>* values);

  // Closes the reader.
  absl::Status Close();

  ~AvroReader();

  // Returns the fields of the Avro file.
  const std::vector<AvroField>& fields() const { return fields_; }

  const std::string& sync_marker() const { return sync_marker_; }

 private:
  AvroReader(std::unique_ptr<utils::InputByteStream>&& stream);

  // Reads the header of the Avro file. Should be called only once.
  absl::StatusOr<std::string> ReadHeader();

  // Reads the next block of the Avro file.
  absl::StatusOr<bool> ReadNextBlock();

  std::unique_ptr<utils::InputByteStream> stream_;

  std::vector<AvroField> fields_;

  std::string sync_marker_;
  std::string new_sync_marker_;

  AvroCodec codec_ = AvroCodec::kNull;

  // Raw and uncompressed data of the current block.
  std::string current_block_;
  std::string current_block_decompressed_;
  std::string zlib_working_buffer_;
  absl::optional<utils::StringViewInputByteStream> current_block_reader_;

  size_t num_objects_in_current_block_ = 0;
  size_t next_object_in_current_block_ = 0;
};

namespace internal {

absl::Status ReadString(utils::InputByteStream* stream, std::string* value);
absl::StatusOr<int64_t> ReadInteger(utils::InputByteStream* stream);
absl::StatusOr<double> ReadDouble(utils::InputByteStream* stream);
absl::StatusOr<float> ReadFloat(utils::InputByteStream* stream);
absl::StatusOr<bool> ReadBoolean(utils::InputByteStream* stream);

}  // namespace internal

}  // namespace yggdrasil_decision_forests::dataset::avro

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_AVRO_H_
