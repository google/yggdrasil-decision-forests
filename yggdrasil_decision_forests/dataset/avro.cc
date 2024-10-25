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
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"
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
        absl::StrCat("Field \"", name, "\" is not a string: ", it->dump()));
  }
  return it->get<std::string>();
}

// Extracts the "something" in a ["null", <something>].
absl::StatusOr<const nlohmann::json*> ExtractTypeInArrayNullType(
    const nlohmann::json& src) {
  if (src.size() == 2 && src[0].is_string() &&
      src[0].get<std::string>() == std::string("null")) {
    return &src[1];
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Expected Avro schema with [\"null\", <something>]. Instead got: ",
      src.dump()));
}

// Extracts the "something" in a {"type": "array", "items": <something>}.
absl::StatusOr<const nlohmann::json*> ExtractItemsInTypeArrayObject(
    const nlohmann::json& src) {
  if (src["type"].is_string() && src["type"].get<std::string>() == "array") {
    return &src["items"];
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Expected Avro schema with {\"type\": \"array\", \"items\": "
                   "<something>}. Instead got: ",
                   src.dump()));
}

struct RecursiveAvroField {
  RecursiveAvroField(AvroType type = AvroType::kUnknown,
                     std::unique_ptr<RecursiveAvroField> sub_type = {},
                     bool optional = false)
      : type(type), sub_type(std::move(sub_type)), optional(optional) {}

  AvroType type;
  std::unique_ptr<RecursiveAvroField> sub_type;  // Only used if type==kArray.
  bool optional;
};

absl::StatusOr<std::unique_ptr<RecursiveAvroField>> ParseRecursiveAvroField(
    const nlohmann::json& json_type) {
  if (json_type.is_string()) {
    // type: "<scalar>"
    ASSIGN_OR_RETURN(const auto type, ParseType(json_type.get<std::string>()));
    return std::make_unique<RecursiveAvroField>(type);
  } else if (json_type.is_object()) {
    // type: {"type": "array", "items": <scalar>}
    // TODO: Add support for {"type": "array", "items": <something>}
    ASSIGN_OR_RETURN(const auto json_items,
                     ExtractItemsInTypeArrayObject(json_type));
    ASSIGN_OR_RETURN(auto sub_type, ParseRecursiveAvroField(*json_items));
    return std::make_unique<RecursiveAvroField>(AvroType::kArray,
                                                std::move(sub_type));
  } else if (json_type.is_array()) {
    // type: ["null", <something>]
    ASSIGN_OR_RETURN(const auto json_sub_type,
                     ExtractTypeInArrayNullType(json_type));
    ASSIGN_OR_RETURN(auto sub_type, ParseRecursiveAvroField(*json_sub_type));
    if (sub_type->optional) {
      return absl::InvalidArgumentError(
          "Avro schema contains two optional tags on the same field.");
    }
    sub_type->optional = true;
    return sub_type;
  }
  return absl::InvalidArgumentError("Unsupported Avro schema");
}

absl::StatusOr<AvroField> RecursiveAvroFieldToAvroField(
    const RecursiveAvroField& src) {
  AvroField dst{.type = src.type, .optional = src.optional};

  if (src.type == AvroType::kArray) {
    dst.sub_type = src.sub_type->type;
    dst.sub_optional = src.sub_type->optional;
    if (dst.sub_type == AvroType::kArray) {
      dst.sub_sub_type = src.sub_type->sub_type->type;
      dst.sub_sub_optional = src.sub_type->sub_type->optional;
      if (dst.sub_sub_type == AvroType::kArray) {
        return absl::InvalidArgumentError(
            "Unsupported Avro schema with more than 3 non-optional levels");
      }
    }
  }

  return dst;
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
     << ", sub_sub_type=" << TypeToString(field.sub_sub_type)
     << ", optional=" << field.optional
     << ", sub_optional=" << field.sub_optional
     << ", sub_sub_optional=" << field.sub_sub_optional << ")";
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
  ASSIGN_OR_RETURN(reader->schema_string_, reader->ReadHeader(),
                   _ << "While reading header of " << path);
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
    ASSIGN_OR_RETURN(const auto* json_type,
                     GetJsonField(json_field.value(), "type"));

    ASSIGN_OR_RETURN(const auto recursive_field,
                     ParseRecursiveAvroField(*json_type),
                     _ << "Unsupported Avro schema for field " << json_name
                       << ":" << json_type->dump());
    ASSIGN_OR_RETURN(auto field,
                     RecursiveAvroFieldToAvroField(*recursive_field),
                     _ << "Unsupported Avro schema for field " << json_name
                       << ":" << json_type->dump());
    field.name = json_name;
    fields_.push_back(field);
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

absl::StatusOr<std::optional<bool>> AvroReader::ReadNextFieldBoolean(
    const AvroField& field) {
  MAYBE_SKIP_OPTIONAL(field);
  ASSIGN_OR_RETURN(const auto value, current_block_reader_->ReadByte());
  return value;
}

absl::StatusOr<std::optional<int64_t>> AvroReader::ReadNextFieldInteger(
    const AvroField& field) {
  MAYBE_SKIP_OPTIONAL(field);
  return internal::ReadInteger(&current_block_reader_.value());
}

absl::StatusOr<std::optional<float>> AvroReader::ReadNextFieldFloat(
    const AvroField& field) {
  MAYBE_SKIP_OPTIONAL(field);
  return internal::ReadFloat(&current_block_reader_.value());
}

absl::StatusOr<std::optional<double>> AvroReader::ReadNextFieldDouble(
    const AvroField& field) {
  MAYBE_SKIP_OPTIONAL(field);
  return internal::ReadDouble(&current_block_reader_.value());
}

absl::StatusOr<bool> AvroReader::ReadNextFieldString(const AvroField& field,
                                                     std::string* value) {
  STATUS_CHECK(field.type == AvroType::kString ||
               field.type == AvroType::kBytes);
  if (field.optional) {
    ASSIGN_OR_RETURN(const auto has_value, current_block_reader_->ReadByte());
    if (!has_value) {
      return false;
    }
    STATUS_CHECK_EQ(has_value, 2);
  }
  RETURN_IF_ERROR(internal::ReadString(&current_block_reader_.value(), value));
  return true;
}

template <typename T, typename R>
absl::StatusOr<bool> AvroReader::ReadNextFieldArrayFloatingPointTemplate(
    const AvroField& field, std::vector<T>* values) {
  STATUS_CHECK(field.type == AvroType::kArray);
  STATUS_CHECK(field.sub_type == AvroType::kFloat ||
               field.sub_type == AvroType::kDouble);

  values->clear();
  if (field.optional) {
    ASSIGN_OR_RETURN(const auto has_value, current_block_reader_->ReadByte());
    if (!has_value) {
      return false;
    }
    STATUS_CHECK_EQ(has_value, 2);
  }
  while (true) {
    ASSIGN_OR_RETURN(auto num_values,
                     internal::ReadInteger(&current_block_reader_.value()));
    if (num_values == 0) {
      break;
    }
    if (num_values < 0) {
      ASSIGN_OR_RETURN(auto block_size,
                       internal::ReadInteger(&current_block_reader_.value()));
      (void)block_size;
      num_values = -num_values;
    }
    STATUS_CHECK_GE(num_values, 0);
    values->reserve(values->size() + num_values);
    for (size_t value_idx = 0; value_idx < num_values; value_idx++) {
      if (field.sub_optional) {
        ASSIGN_OR_RETURN(const auto has_sub_value,
                         current_block_reader_->ReadByte());
        if (!has_sub_value) {
          values->push_back(std::numeric_limits<T>::quiet_NaN());
          continue;
        }
        STATUS_CHECK_EQ(has_sub_value, 2);
      }
      ASSIGN_OR_RETURN(R value, internal::ReadFloatingPointTemplate<R>(
                                    &current_block_reader_.value()));
      values->push_back(value);
    }
  }
  return true;
}

absl::StatusOr<bool> AvroReader::ReadNextFieldArrayFloat(
    const AvroField& field, std::vector<float>* values) {
  STATUS_CHECK(field.sub_type == AvroType::kFloat);
  return ReadNextFieldArrayFloatingPointTemplate<float>(field, values);
}

absl::StatusOr<bool> AvroReader::ReadNextFieldArrayDouble(
    const AvroField& field, std::vector<double>* values) {
  STATUS_CHECK(field.sub_type == AvroType::kDouble);
  return ReadNextFieldArrayFloatingPointTemplate<double>(field, values);
}

absl::StatusOr<bool> AvroReader::ReadNextFieldArrayDoubleIntoFloat(
    const AvroField& field, std::vector<float>* values) {
  STATUS_CHECK(field.sub_type == AvroType::kDouble);
  return ReadNextFieldArrayFloatingPointTemplate<float, double>(field, values);
}

absl::StatusOr<bool> AvroReader::ReadNextFieldArrayString(
    const AvroField& field, std::vector<std::string>* values) {
  STATUS_CHECK(field.type == AvroType::kArray);
  STATUS_CHECK(field.sub_type == AvroType::kString ||
               field.sub_type == AvroType::kBytes);

  if (field.optional) {
    ASSIGN_OR_RETURN(const auto has_value, current_block_reader_->ReadByte());
    if (!has_value) {
      return false;
    }
    STATUS_CHECK_EQ(has_value, 2);
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
      if (field.sub_optional) {
        ASSIGN_OR_RETURN(const auto has_sub_value,
                         current_block_reader_->ReadByte());
        if (!has_sub_value) {
          values->push_back("");
          continue;
        }
      }
      std::string sub_value;
      RETURN_IF_ERROR(
          internal::ReadString(&current_block_reader_.value(), &sub_value));
      values->push_back(std::move(sub_value));
    }
  }

  return true;
}

absl::StatusOr<bool> AvroReader::ReadNextFieldArrayArrayFloat(
    const AvroField& field, std::vector<std::vector<float>>* values) {
  STATUS_CHECK(field.sub_sub_type == AvroType::kFloat);
  return ReadNextFieldArrayArrayFloatingPointTemplate<float>(field, values);
}

absl::StatusOr<bool> AvroReader::ReadNextFieldArrayArrayDoubleIntoFloat(
    const AvroField& field, std::vector<std::vector<float>>* values) {
  STATUS_CHECK(field.sub_sub_type == AvroType::kDouble);
  return ReadNextFieldArrayArrayFloatingPointTemplate<float, double>(field,
                                                                     values);
}

template <typename T, typename R>
absl::StatusOr<bool> AvroReader::ReadNextFieldArrayArrayFloatingPointTemplate(
    const AvroField& field, std::vector<std::vector<T>>* values) {
  STATUS_CHECK(field.type == AvroType::kArray);
  STATUS_CHECK(field.sub_type == AvroType::kArray);
  STATUS_CHECK(field.sub_sub_type == AvroType::kFloat ||
               field.sub_sub_type == AvroType::kDouble);

  values->clear();
  if (field.optional) {
    ASSIGN_OR_RETURN(const auto has_value, current_block_reader_->ReadByte());
    if (!has_value) {
      return false;
    }
    STATUS_CHECK_EQ(has_value, 2);
  }

  while (true) {
    ASSIGN_OR_RETURN(auto num_values,
                     internal::ReadInteger(&current_block_reader_.value()));
    if (num_values == 0) {
      break;
    }
    if (num_values < 0) {
      ASSIGN_OR_RETURN(auto block_size,
                       internal::ReadInteger(&current_block_reader_.value()));
      (void)block_size;
      num_values = -num_values;
    }
    STATUS_CHECK_GE(num_values, 0);
    values->reserve(values->size() + num_values);
    for (size_t value_idx = 0; value_idx < num_values; value_idx++) {
      values->emplace_back();
      // Note: Missing values are replaced with an empty list.
      RETURN_IF_ERROR((ReadNextFieldArrayFloatingPointTemplate<T, R>(
                           AvroField{.name = field.name,
                                     .type = field.sub_type,
                                     .optional = field.sub_optional,
                                     .sub_type = field.sub_sub_type,
                                     .sub_optional = field.sub_sub_optional},
                           &values->back())
                           .status()));
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

template <typename T>
absl::StatusOr<T> ReadFloatingPointTemplate(utils::InputByteStream* stream) {
  if (std::is_same_v<T, float>) {
    return ReadFloat(stream);
  } else if (std::is_same_v<T, double>) {
    return ReadDouble(stream);
  }
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
