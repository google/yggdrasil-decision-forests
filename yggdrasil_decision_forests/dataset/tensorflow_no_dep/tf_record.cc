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

#include "yggdrasil_decision_forests/dataset/tensorflow_no_dep/tf_record.h"

#include <stdint.h>

#include <memory>

#include "absl/base/port.h"
#include "absl/crc/crc32c.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"

namespace yggdrasil_decision_forests::dataset::tensorflow_no_dep {
namespace {
constexpr char kInvalidDataMessage[] =
    "The file is not a non-compressed TFRecord or it is corrupted. If "
    "you have a compressed TFRecord, decompress it first.";

static const uint32_t kMaskDelta = 0xa282ead8ul;

// Mask applied by TF Record over the CRCs.
// See tensorflow/tsl/lib/hash/crc32c.h
// Note: TF Record does NOT compute CRC over a string containing a CRC.
inline uint32_t Mask(const uint32_t crc) {
  return ((crc >> 15) | (crc << 17)) + kMaskDelta;
}

inline uint32_t Unmask(const uint32_t masked_crc) {
  uint32_t rot = masked_crc - kMaskDelta;
  return ((rot >> 17) | (rot << 15));
}

}  // namespace

absl::StatusOr<absl::crc32c_t> TFRecordReader::ReadCRC() {
  uint32_t value;
  ASSIGN_OR_RETURN(const bool has_content,
                   stream_->ReadExactly((char*)&value, sizeof(uint32_t)));
  if (!has_content) {
    return absl::InvalidArgumentError("Empty stream");
  }
  return absl::crc32c_t(Unmask(absl::little_endian::ToHost32(value)));
}

TFRecordReader::~TFRecordReader() {
  if (stream_) {
    YDF_LOG(WARNING) << "Destruction of a non closed TFRecordReader";
    Close().IgnoreError();
  }
}

absl::StatusOr<std::unique_ptr<TFRecordReader>> TFRecordReader::Create(
    const absl::string_view path) {
  ASSIGN_OR_RETURN(auto stream, file::OpenInputFile(path));
  return absl::make_unique<TFRecordReader>(std::move(stream));
}

absl::StatusOr<bool> TFRecordReader::Next(google::protobuf::MessageLite* message) {
  uint64_t raw_length;
  ASSIGN_OR_RETURN(bool has_content,
                   stream_->ReadExactly((char*)&raw_length, sizeof(uint64_t)));
  if (!has_content) {
    return false;
  }
  const uint64_t length = absl::little_endian::ToHost64(raw_length);

  ASSIGN_OR_RETURN(const absl::crc32c_t raw_length_expected_crc, ReadCRC());
  const absl::crc32c_t raw_length_real_checksum = absl::ComputeCrc32c(
      absl::string_view((char*)&raw_length, sizeof(uint64_t)));
  if (raw_length_expected_crc != raw_length_real_checksum) {
    return absl::InvalidArgumentError(kInvalidDataMessage);
  }

  buffer_.resize(length);
  // TODO: Use buffer_.data() in c++>=17.
  if (length > 0) {
    ASSIGN_OR_RETURN(has_content, stream_->ReadExactly(&buffer_[0], length));
  }
  if (!has_content) {
    return absl::InvalidArgumentError(kInvalidDataMessage);
  }
  ASSIGN_OR_RETURN(const absl::crc32c_t data_expected_crc, ReadCRC());
  const absl::crc32c_t data_real_checksum = absl::ComputeCrc32c(buffer_);
  if (data_expected_crc != data_real_checksum) {
    return absl::InvalidArgumentError(kInvalidDataMessage);
  }

  if (message && !message->ParseFromString(buffer_)) {
    return absl::InvalidArgumentError(kInvalidDataMessage);
  }

  return true;
}

// Closes the stream.
absl::Status TFRecordReader::Close() {
  if (stream_) {
    RETURN_IF_ERROR(stream_->Close());
    stream_.reset();
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<TFRecordWriter>> TFRecordWriter::Create(
    absl::string_view path) {
  ASSIGN_OR_RETURN(auto stream, file::OpenOutputFile(path));
  return absl::make_unique<TFRecordWriter>(std::move(stream));
}

TFRecordWriter::~TFRecordWriter() {
  if (stream_) {
    YDF_LOG(WARNING) << "Destruction of a non closed TFRecordWriter";
    Close().IgnoreError();
  }
}

absl::Status TFRecordWriter::Write(const google::protobuf::MessageLite& message) {
  if (!message.SerializeToString(&buffer_)) {
    return absl::InternalError("Cannot serialize message");
  }
  return Write(buffer_);
}

absl::Status TFRecordWriter::Write(const absl::string_view data) {
  uint64_t length = data.size();
  RETURN_IF_ERROR(
      stream_->Write(absl::string_view((char*)&length, sizeof(uint64_t))));

  const uint64_t net_length = absl::little_endian::FromHost64(length);
  const uint32_t net_length_checksum =
      Mask(static_cast<uint32_t>(absl::ComputeCrc32c(
          absl::string_view((char*)&net_length, sizeof(uint64_t)))));
  RETURN_IF_ERROR(stream_->Write(
      absl::string_view((char*)&net_length_checksum, sizeof(uint32_t))));

  RETURN_IF_ERROR(stream_->Write(data));

  const uint32_t net_data_checksum =
      Mask(static_cast<uint32_t>(absl::ComputeCrc32c(data)));
  RETURN_IF_ERROR(stream_->Write(
      absl::string_view((char*)&net_data_checksum, sizeof(uint32_t))));

  return absl::OkStatus();
}

absl::Status TFRecordWriter::Close() {
  if (stream_) {
    RETURN_IF_ERROR(stream_->Close());
    stream_.reset();
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::dataset::tensorflow_no_dep
