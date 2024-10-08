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

#include "yggdrasil_decision_forests/dataset/avro_example.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::dataset {

AvroReader::AvroReader(std::unique_ptr<utils::InputByteStream>&& stream)
    : stream_(std::move(stream)) {}

AvroReader::~AvroReader() { Close().IgnoreError(); }

absl::StatusOr<std::unique_ptr<AvroReader>> AvroReader::Create(
    absl::string_view path) {
  return absl::UnimplementedError("Not implemented");
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
  // Magic number.
  char buffer[4];
  ASSIGN_OR_RETURN(bool has_read, stream_->ReadExactly(buffer, 4));
  STATUS_CHECK(has_read);
  if (buffer[0] != 'O' || buffer[1] != 'b' || buffer[2] != 'j' ||
      buffer[3] != 1) {
    return absl::InvalidArgumentError("Not an Avro file");
  }

  std::string schema;
  std::string codec = "null";  // Default codec.
  ASSIGN_OR_RETURN(const size_t num_blocks,
                   internal::ReadInteger(stream_.get()));
  for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
    ASSIGN_OR_RETURN(const auto key, internal::ReadString(stream_.get()));
    ASSIGN_OR_RETURN(const auto value, internal::ReadString(stream_.get()));
    if (key == "avro.codec") {
      codec = value;
    } else if (key == "avro.schema") {
      schema = value;
    }
  }
  return schema;
}

namespace internal {

absl::StatusOr<size_t> ReadInteger(utils::InputByteStream* stream) {
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

absl::StatusOr<std::string> ReadString(utils::InputByteStream* stream) {
  ASSIGN_OR_RETURN(const auto length, ReadInteger(stream));
  std::string value(length, 0);
  if (length > 0) {
    ASSIGN_OR_RETURN(bool has_read, stream->ReadExactly(value.data(), length));
    if (!has_read) {
      return absl::InvalidArgumentError("Unexpected end of stream");
    }
  }
  return value;
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

}  // namespace yggdrasil_decision_forests::dataset
