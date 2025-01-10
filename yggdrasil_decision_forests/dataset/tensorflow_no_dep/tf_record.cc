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
#include <utility>

#include "absl/base/internal/endian.h"
#include "absl/crc/crc32c.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/zlib.h"

namespace yggdrasil_decision_forests::dataset::tensorflow_no_dep {
namespace {
constexpr char kInvalidDataMessage[] =
    "The data is not a valid non-compressed TF Record. The data is either "
    "corrupted or (more likely) a gzip compressed TFRecord. In this later "
    "case, fix the type prefix in the filepath. For example, replace "
    "'tfrecordv2+tfe:' with 'tfrecord:' (recommended) or  ('tfrecord+tfe').";

static const uint32_t kMaskDelta = 0xa282ead8ul;

// Mask applied by TF Record over the CRCs.
// See tensorflow/compiler/xla/tsl/lib/hash/crc32c.h
// Note: TF Record does NOT compute CRC over a string containing a CRC.
inline uint32_t Mask(const uint32_t crc) {
  return ((crc >> 15) | (crc << 17)) + kMaskDelta;
}

inline uint32_t Unmask(const uint32_t masked_crc) {
  uint32_t rot = masked_crc - kMaskDelta;
  return ((rot >> 17) | (rot << 15));
}

template <typename T>
constexpr absl::string_view GetView(const T& data) {
  return absl::string_view((char*)&data, sizeof(T));
}

}  // namespace

absl::StatusOr<absl::crc32c_t> TFRecordReader::ReadCRC() {
  uint32_t value;
  ASSIGN_OR_RETURN(const bool has_content,
                   stream().ReadExactly((char*)&value, sizeof(uint32_t)));
  if (!has_content) {
    return absl::InvalidArgumentError("Empty stream");
  }
  return absl::crc32c_t(Unmask(absl::little_endian::ToHost32(value)));
}

TFRecordReader::~TFRecordReader() {
  if (raw_stream_) {
    LOG(WARNING) << "Destruction of a non closed TFRecordReader";
    Close().IgnoreError();
  }
}

absl::StatusOr<std::unique_ptr<TFRecordReader>> TFRecordReader::Create(
    const absl::string_view path, bool compressed) {
  auto reader = std::make_unique<TFRecordReader>();

  ASSIGN_OR_RETURN(reader->raw_stream_, file::OpenInputFile(path));
  if (compressed) {
    ASSIGN_OR_RETURN(reader->zlib_stream_, utils::GZipInputByteStream::Create(
                                               reader->raw_stream_.get()));
  }
  return reader;
}

absl::StatusOr<bool> TFRecordReader::Next(google::protobuf::MessageLite* message) {
  uint64_t raw_length;
  ASSIGN_OR_RETURN(bool has_content,
                   stream().ReadExactly((char*)&raw_length, sizeof(uint64_t)));
  if (!has_content) {
    return false;
  }
  const uint64_t length = absl::little_endian::ToHost64(raw_length);

  ASSIGN_OR_RETURN(const absl::crc32c_t raw_length_expected_crc, ReadCRC());
  const absl::crc32c_t raw_length_real_checksum =
      absl::ComputeCrc32c(GetView(raw_length));
  if (raw_length_expected_crc != raw_length_real_checksum) {
    return absl::InvalidArgumentError(kInvalidDataMessage);
  }

  buffer_.resize(length);
  // TODO: Use buffer_.data() in c++>=17.
  if (length > 0) {
    ASSIGN_OR_RETURN(has_content, stream().ReadExactly(&buffer_[0], length));
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
  if (zlib_stream_) {
    RETURN_IF_ERROR(zlib_stream_->Close());
    zlib_stream_.reset();
  }
  if (raw_stream_) {
    RETURN_IF_ERROR(raw_stream_->Close());
    raw_stream_.reset();
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<TFRecordWriter>> TFRecordWriter::Create(
    absl::string_view path, bool compressed) {
  auto writer = std::make_unique<TFRecordWriter>();
  ASSIGN_OR_RETURN(writer->raw_stream_, file::OpenOutputFile(path));
  if (compressed) {
    ASSIGN_OR_RETURN(writer->zlib_stream_, utils::GZipOutputByteStream::Create(
                                               writer->raw_stream_.get()));
  }
  return writer;
}

TFRecordWriter::~TFRecordWriter() {
  if (raw_stream_) {
    LOG(WARNING) << "Destruction of a non closed TFRecordWriter";
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
  RETURN_IF_ERROR(stream().Write(GetView(length)));

  const uint64_t net_length = absl::little_endian::FromHost64(length);
  const uint32_t net_length_checksum =
      Mask(static_cast<uint32_t>(absl::ComputeCrc32c(GetView(net_length))));
  RETURN_IF_ERROR(stream().Write(GetView(net_length_checksum)));

  RETURN_IF_ERROR(stream().Write(data));

  const uint32_t net_data_checksum =
      Mask(static_cast<uint32_t>(absl::ComputeCrc32c(data)));
  RETURN_IF_ERROR(stream().Write(GetView(net_data_checksum)));

  return absl::OkStatus();
}

absl::Status TFRecordWriter::Close() {
  if (zlib_stream_) {
    RETURN_IF_ERROR(zlib_stream_->Close());
    zlib_stream_.reset();
  }
  if (raw_stream_) {
    RETURN_IF_ERROR(raw_stream_->Close());
    raw_stream_.reset();
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::dataset::tensorflow_no_dep
