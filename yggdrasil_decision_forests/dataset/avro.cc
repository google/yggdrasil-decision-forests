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

#include "yggdrasil_decision_forests/dataset/avro.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"
#include "yggdrasil_decision_forests/utils/bytestream.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/zlib.h"

#define MAYBE_SKIP_OPTIONAL(FIELD)                       \
  if (field.optional) {                                  \
    ASSIGN_OR_RETURN(const auto _has_value,              \
                     current_block_reader_->ReadByte()); \
    if (!_has_value) {                                   \
      return absl::nullopt;                              \
    }                                                    \
  }

namespace yggdrasil_decision_forests::dataset::avro {
namespace {
absl::Status JsonIsObject(const nlohmann::json& value) {
  if (!value.is_object()) {
    return absl::InvalidArgumentError("Schema is not a json object");
  }
  return absl::OkStatus();
}

absl::StatusOr<const nlohmann::json*> GetJsonField(const nlohmann::json& value,
                                                   const std::string& name) {
  auto it = value.find(name);
  if (it == value.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Field \"", name, "\" not found"));
  }
  return &*it;
}

absl::StatusOr<const std::string> GetJsonStringField(
    const nlohmann::json& value, const std::string& name) {
  auto it = value.find(name);
  if (it == value.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Field \"", name, "\" not found"));
  }
  if (!it->is_string()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Field \"", name, "\" is not a string"));
  }
  return it->get<std::string>();
}

}  // namespace

absl::StatusOr<AvroType> ParseType(absl::string_view key) {
  if (key == "null") {
    return AvroType::kNull;
  } else if (key == "boolean") {
    return AvroType::kBoolean;
  } else if (key == "long") {
    return AvroType::kLong;
  } else if (key == "int") {
    return AvroType::kInt;
  } else if (key == "float") {
    return AvroType::kFloat;
  } else if (key == "double") {
    return AvroType::kDouble;
  } else if (key == "string") {
    return AvroType::kString;
  } else if (key == "bytes") {
    return AvroType::kBytes;
  } else if (key == "array") {
    return AvroType::kArray;
  }
  return absl::InvalidArgumentError(absl::StrCat("Unsupported type=", key));
}

std::string TypeToString(AvroType type) {
  switch (type) {
    case AvroType::kNull:
      return "null";
    case AvroType::kBoolean:
      return "boolean";
    case AvroType::kLong:
      return "long";
    case AvroType::kInt:
      return "int";
    case AvroType::kFloat:
      return "float";
    case AvroType::kDouble:
      return "double";
    case AvroType::kString:
      return "string";
    case AvroType::kBytes:
      return "bytes";
    case AvroType::kArray:
      return "array";
    case AvroType::kUnknown:
      return "unknown";
  }
}

std::ostream& operator<<(std::ostream& os, const AvroField& field) {
  os << "AvroField(name=\"" << field.name
     << "\", type=" << TypeToString(field.type)
     << ", sub_type=" << TypeToString(field.sub_type)
     << ", optional=" << field.optional << ")";
  return os;
}

AvroReader::AvroReader(std::unique_ptr<utils::InputByteStream>&& stream)
    : stream_(std::move(stream)) {}

AvroReader::~AvroReader() { Close().IgnoreError(); }

absl::StatusOr<std::unique_ptr<AvroReader>> AvroReader::Create(
    absl::string_view path) {
  ASSIGN_OR_RETURN(std::unique_ptr<utils::InputByteStream> file_handle,
                   file::OpenInputFile(path));
  auto reader = absl::WrapUnique(new AvroReader(std::move(file_handle)));
  ASSIGN_OR_RETURN(const auto schema, reader->ReadHeader());
  return std::move(reader);
}

absl::Status AvroReader::Close() {
  if (stream_) {
    RETURN_IF_ERROR(stream_->Close());
    stream_.reset();
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> AvroReader::ExtractSchema(absl::string_view path) {
  ASSIGN_OR_RETURN(std::unique_ptr<utils::InputByteStream> file_handle,
                   file::OpenInputFile(path));
  AvroReader reader(std::move(file_handle));
  ASSIGN_OR_RETURN(const auto schema, reader.ReadHeader());
  RETURN_IF_ERROR(reader.Close());
  return schema;
}

absl::StatusOr<std::string> AvroReader::ReadHeader() {
  if (!sync_marker_.empty()) {
    return absl::InvalidArgumentError("The header was already read");
  }

  // Magic number.
  char buffer[4];
  ASSIGN_OR_RETURN(bool has_read, stream_->ReadExactly(buffer, 4));
  STATUS_CHECK(has_read);
  if (buffer[0] != 'O' || buffer[1] != 'b' || buffer[2] != 'j' ||
      buffer[3] != 1) {
    return absl::InvalidArgumentError("Not an Avro file");
  }

  // Read the meta-data.
  std::string schema;
  while (true) {
    ASSIGN_OR_RETURN(const size_t num_blocks,
                     internal::ReadInteger(stream_.get()));
    if (num_blocks == 0) {
      break;
    }
    for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
      std::string key;
      std::string value;
      RETURN_IF_ERROR(internal::ReadString(stream_.get(), &key));
      RETURN_IF_ERROR(internal::ReadString(stream_.get(), &value));
      if (key == "avro.codec") {
        if (value == "null") {
          codec_ = AvroCodec::kNull;
        } else if (value == "deflate") {
          codec_ = AvroCodec::kDeflate;
        } else {
          return absl::InvalidArgumentError(
              absl::StrCat("Unsupported codec: ", value));
        }
      } else if (key == "avro.schema") {
        schema = value;
      }
    }
  }

  sync_marker_.resize(16);
  ASSIGN_OR_RETURN(has_read, stream_->ReadExactly(&sync_marker_[0], 16));
  STATUS_CHECK(has_read);

  // Parse the meta-data.
  nlohmann::json json_schema = nlohmann::json::parse(schema);

  if (json_schema.is_discarded()) {
    return absl::InvalidArgumentError("Failed to parse schema");
  }
  RETURN_IF_ERROR(JsonIsObject(json_schema));
  ASSIGN_OR_RETURN(const auto json_type,
                   GetJsonStringField(json_schema, "type"));
  STATUS_CHECK_EQ(json_type, "record");
  ASSIGN_OR_RETURN(const auto* json_fields,
                   GetJsonField(json_schema, "fields"));
  STATUS_CHECK(json_fields->is_array());

  for (const auto& json_field : json_fields->items()) {
    ASSIGN_OR_RETURN(const auto json_name,
                     GetJsonStringField(json_field.value(), "name"));
    ASSIGN_OR_RETURN(const auto* json_sub_type,
                     GetJsonField(json_field.value(), "type"));
    AvroType type = AvroType::kUnknown;
    AvroType sub_type = AvroType::kUnknown;
    bool optional = false;

    if (json_sub_type->is_string()) {
      const std::string str_type = json_sub_type->get<std::string>();
      // Scalar
      ASSIGN_OR_RETURN(type, ParseType(str_type));
    } else if (json_sub_type->is_array()) {
      // Optional
      optional = true;

      // const auto& json_sub_type_array = json_sub_type->GetArray();
      if (json_sub_type->size() == 2 && (*json_sub_type)[0].is_string() &&
          (*json_sub_type)[0].get<std::string>() == std::string("null")) {
        if ((*json_sub_type)[1].is_string()) {
          // Scalar
          ASSIGN_OR_RETURN(type,
                           ParseType((*json_sub_type)[1].get<std::string>()));
        }
        if ((*json_sub_type)[1].is_object()) {
          // Array
          ASSIGN_OR_RETURN(const auto json_sub_sub_type,
                           GetJsonStringField((*json_sub_type)[1], "type"));
          ASSIGN_OR_RETURN(const auto json_items,
                           GetJsonStringField((*json_sub_type)[1], "items"));
          if (json_sub_sub_type == "array") {
            ASSIGN_OR_RETURN(sub_type, ParseType(json_items));
            type = AvroType::kArray;
          }
        }
      }
    } else if (json_sub_type->is_object()) {
      // Array
      ASSIGN_OR_RETURN(const auto json_sub_sub_type,
                       GetJsonStringField(*json_sub_type, "type"));
      ASSIGN_OR_RETURN(const auto json_items,
                       GetJsonStringField(*json_sub_type, "items"));

      if (json_sub_sub_type == "array") {
        ASSIGN_OR_RETURN(sub_type, ParseType(json_items));
        type = AvroType::kArray;
      }
    }

    if (type == AvroType::kUnknown) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported type=", json_sub_type->get<std::string>(),
          " for field \"", json_name,
          "\". YDF only supports the following types: null, "
          "boolean, long, int, float, double, string, bytes, array of <scalar "
          "type>, [null, <scalar type>], [null, array[<scalar type>]]."));
    }

    fields_.push_back(AvroField{
        .name = json_name,
        .type = type,
        .sub_type = sub_type,
        .optional = optional,
    });
  }

  return schema;
}

absl::StatusOr<bool> AvroReader::ReadNextBlock() {
  const auto num_objects_in_block_or = internal::ReadInteger(stream_.get());
  if (!num_objects_in_block_or.ok()) {
    return false;
  }
  num_objects_in_current_block_ = num_objects_in_block_or.value();
  next_object_in_current_block_ = 0;

  ASSIGN_OR_RETURN(const auto block_size, internal::ReadInteger(stream_.get()));

  current_block_.resize(block_size);
  ASSIGN_OR_RETURN(bool has_read,
                   stream_->ReadExactly(&current_block_[0], block_size));
  if (!has_read) {
    return absl::InvalidArgumentError("Unexpected end of stream");
  }

  switch (codec_) {
    case AvroCodec::kNull:
      current_block_reader_ = utils::StringViewInputByteStream(current_block_);
      break;
    case AvroCodec::kDeflate:
      zlib_working_buffer_.resize(1024 * 1024);
      RETURN_IF_ERROR(utils::Inflate(
          current_block_, &current_block_decompressed_, &zlib_working_buffer_));
      current_block_reader_ =
          utils::StringViewInputByteStream(current_block_decompressed_);
      break;
  }

  new_sync_marker_.resize(16);
  ASSIGN_OR_RETURN(has_read, stream_->ReadExactly(&new_sync_marker_[0], 16));
  STATUS_CHECK(has_read);
  if (new_sync_marker_ != sync_marker_) {
    return absl::InvalidArgumentError(
        "Non matching sync marker. The file looks corrupted.");
  }

  return true;
}

absl::StatusOr<bool> AvroReader::ReadNextRecord() {
  if (!current_block_reader_.has_value() ||
      current_block_reader_.value().left() == 0) {
    // Read a new block of data.
    DCHECK_EQ(next_object_in_current_block_, num_objects_in_current_block_);
    ASSIGN_OR_RETURN(const bool has_next_block, ReadNextBlock());
    if (!has_next_block) {
      return false;
    }
  }
  next_object_in_current_block_++;
  return true;
}

absl::StatusOr<absl::optional<bool>> AvroReader::ReadNextFieldBoolean(
    const AvroField& field) {
  MAYBE_SKIP_OPTIONAL(field);
  ASSIGN_OR_RETURN(const auto value, current_block_reader_->ReadByte());
  return value;
}

absl::StatusOr<absl::optional<int64_t>> AvroReader::ReadNextFieldInteger(
    const AvroField& field) {
  MAYBE_SKIP_OPTIONAL(field);
  return internal::ReadInteger(&current_block_reader_.value());
}

absl::StatusOr<absl::optional<float>> AvroReader::ReadNextFieldFloat(
    const AvroField& field) {
  MAYBE_SKIP_OPTIONAL(field);
  return internal::ReadFloat(&current_block_reader_.value());
}

absl::StatusOr<absl::optional<double>> AvroReader::ReadNextFieldDouble(
    const AvroField& field) {
  MAYBE_SKIP_OPTIONAL(field);
  return internal::ReadDouble(&current_block_reader_.value());
}

absl::StatusOr<bool> AvroReader::ReadNextFieldString(const AvroField& field,
                                                     std::string* value) {
  if (field.optional) {
    ASSIGN_OR_RETURN(const auto has_value, current_block_reader_->ReadByte());
    if (!has_value) {
      return false;
    }
  }
  RETURN_IF_ERROR(internal::ReadString(&current_block_reader_.value(), value));
  return true;
}

absl::StatusOr<bool> AvroReader::ReadNextFieldArrayFloat(
    const AvroField& field, std::vector<float>* values) {
  values->clear();
  if (field.optional) {
    ASSIGN_OR_RETURN(const auto has_value, current_block_reader_->ReadByte());
    if (!has_value) {
      return false;
    }
  }
  while (true) {
    ASSIGN_OR_RETURN(auto num_values,
                     internal::ReadInteger(&current_block_reader_.value()));
    values->reserve(values->size() + num_values);
    if (num_values == 0) {
      break;
    }
    if (num_values < 0) {
      ASSIGN_OR_RETURN(auto block_size,
                       internal::ReadInteger(&current_block_reader_.value()));
      (void)block_size;
      num_values = -num_values;
    }
    for (size_t value_idx = 0; value_idx < num_values; value_idx++) {
      ASSIGN_OR_RETURN(auto value,
                       internal::ReadFloat(&current_block_reader_.value()));
      values->push_back(value);
    }
  }

  return true;
}

absl::StatusOr<bool> AvroReader::ReadNextFieldArrayDouble(
    const AvroField& field, std::vector<double>* values) {
  values->clear();
  if (field.optional) {
    ASSIGN_OR_RETURN(const auto has_value, current_block_reader_->ReadByte());
    if (!has_value) {
      return false;
    }
  }
  while (true) {
    ASSIGN_OR_RETURN(auto num_values,
                     internal::ReadInteger(&current_block_reader_.value()));
    values->reserve(values->size() + num_values);
    if (num_values == 0) {
      break;
    }
    if (num_values < 0) {
      ASSIGN_OR_RETURN(auto block_size,
                       internal::ReadInteger(&current_block_reader_.value()));
      (void)block_size;
      num_values = -num_values;
    }
    for (size_t value_idx = 0; value_idx < num_values; value_idx++) {
      ASSIGN_OR_RETURN(auto value,
                       internal::ReadDouble(&current_block_reader_.value()));
      values->push_back(value);
    }
  }

  return true;
}

absl::StatusOr<bool> AvroReader::ReadNextFieldArrayDoubleIntoFloat(
    const AvroField& field, std::vector<float>* values) {
  values->clear();
  if (field.optional) {
    ASSIGN_OR_RETURN(const auto has_value, current_block_reader_->ReadByte());
    if (!has_value) {
      return false;
    }
  }
  while (true) {
    ASSIGN_OR_RETURN(auto num_values,
                     internal::ReadInteger(&current_block_reader_.value()));
    if (num_values == 0) {
      break;
    }
    values->reserve(values->size() + num_values);
    if (num_values < 0) {
      ASSIGN_OR_RETURN(auto block_size,
                       internal::ReadInteger(&current_block_reader_.value()));
      (void)block_size;
      num_values = -num_values;
    }
    for (size_t value_idx = 0; value_idx < num_values; value_idx++) {
      ASSIGN_OR_RETURN(auto value,
                       internal::ReadDouble(&current_block_reader_.value()));
      values->push_back(value);
    }
  }

  return true;
}

absl::StatusOr<bool> AvroReader::ReadNextFieldArrayString(
    const AvroField& field, std::vector<std::string>* values) {
  if (field.optional) {
    ASSIGN_OR_RETURN(const auto has_value, current_block_reader_->ReadByte());
    if (!has_value) {
      return false;
    }
  }

  while (true) {
    ASSIGN_OR_RETURN(auto num_values,
                     internal::ReadInteger(&current_block_reader_.value()));
    if (num_values == 0) {
      break;
    }
    values->reserve(values->size() + num_values);
    if (num_values < 0) {
      ASSIGN_OR_RETURN(auto block_size,
                       internal::ReadInteger(&current_block_reader_.value()));
      (void)block_size;
      num_values = -num_values;
    }
    for (size_t value_idx = 0; value_idx < num_values; value_idx++) {
      std::string sub_value;
      RETURN_IF_ERROR(
          internal::ReadString(&current_block_reader_.value(), &sub_value));
      values->push_back(std::move(sub_value));
    }
  }

  return true;
}

namespace internal {

absl::StatusOr<int64_t> ReadInteger(utils::InputByteStream* stream) {
  // Note: Integers are encoded with variable length + zigzag encoding.

  //  Variable length decoding
  size_t value = 0;
  size_t shift = 0;
  char buffer;
  while (true) {
    ASSIGN_OR_RETURN(bool has_read, stream->ReadExactly(&buffer, 1));
    if (!has_read) {
      return absl::InvalidArgumentError("Unexpected end of stream");
    }
    value |= static_cast<size_t>(buffer & 0x7F) << shift;
    if ((buffer & 0x80) == 0) {
      break;
    }
    shift += 7;
  }

  // Zigzag decoding
  return (value >> 1) ^ -(value & 1);
}

absl::Status ReadString(utils::InputByteStream* stream, std::string* value) {
  ASSIGN_OR_RETURN(const auto length, ReadInteger(stream));
  value->resize(length);
  if (length > 0) {
    ASSIGN_OR_RETURN(bool has_read, stream->ReadExactly(&(*value)[0], length));
    if (!has_read) {
      return absl::InvalidArgumentError("Unexpected end of stream");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<double> ReadDouble(utils::InputByteStream* stream) {
  double value;
  ASSIGN_OR_RETURN(bool has_read,
                   stream->ReadExactly(reinterpret_cast<char*>(&value), 8));
  if (!has_read) {
    return absl::InvalidArgumentError("Unexpected end of stream");
  }
  return value;
}

absl::StatusOr<float> ReadFloat(utils::InputByteStream* stream) {
  float value;
  ASSIGN_OR_RETURN(bool has_read,
                   stream->ReadExactly(reinterpret_cast<char*>(&value), 4));
  if (!has_read) {
    return absl::InvalidArgumentError("Unexpected end of stream");
  }
  return value;
}

absl::StatusOr<bool> ReadBoolean(utils::InputByteStream* stream) {
  char value;
  ASSIGN_OR_RETURN(bool has_read, stream->ReadExactly(&value, 1));
  if (!has_read) {
    return absl::InvalidArgumentError("Unexpected end of stream");
  }
  return value != 0;
}

}  // namespace internal

}  // namespace yggdrasil_decision_forests::dataset::avro
