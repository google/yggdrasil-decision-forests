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

#include "yggdrasil_decision_forests/utils/blob_sequence.h"

#include <cstdint>
#include <string>

#include "absl/base/internal/endian.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/zlib.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace blob_sequence {

// See "FileHeader::version" for the definition of the versions.
constexpr int kCurrentVersion = 1;

absl::StatusOr<Reader> Reader::Create(utils::InputByteStream* stream) {
  Reader reader;
  reader.raw_stream_ = stream;

  internal::FileHeader header;
  ASSIGN_OR_RETURN(const auto has_content,
                   reader.raw_stream_->ReadExactly(
                       (char*)&header, sizeof(internal::FileHeader)));
  if (!has_content) {
    return absl::InvalidArgumentError("Empty stream");
  }
  if (header.magic[0] != 'B' || header.magic[1] != 'S') {
    return absl::InvalidArgumentError("Invalid header");
  }
  reader.version_ = absl::little_endian::ToHost16(header.version);

  if (reader.version_ > kCurrentVersion) {
    return absl::InvalidArgumentError(absl::Substitute(
        "The blob sequence file's version ($0) is greater than "
        "the blob sequence library ($1). Update your code.",
        reader.version_, kCurrentVersion));
  }

  if (reader.version_ >= 1) {
    reader.compression_ = static_cast<Compression>(header.compression);
  } else {
    reader.compression_ = Compression::kNone;
  }

  switch (reader.compression_) {
    case Compression::kNone:
      break;
    case Compression::kGZIP:
      ASSIGN_OR_RETURN(reader.gzip_stream_,
                       utils::GZipInputByteStream::Create(reader.raw_stream_));
      break;
  }

  return reader;
}

absl::StatusOr<bool> Reader::Read(std::string* blob) {
  internal::RecordHeader header;
  ASSIGN_OR_RETURN(
      auto has_content,
      stream().ReadExactly((char*)&header, sizeof(internal::RecordHeader)));
  if (!has_content) {
    // End of BS.
    return false;
  }

  header.length = absl::little_endian::ToHost32(header.length);

  blob->resize(header.length);
  ASSIGN_OR_RETURN(has_content,
                   stream().ReadExactly(&(*blob)[0], header.length));
  if (!has_content) {
    return absl::InvalidArgumentError("Truncated blob");
  }

  return true;
}

absl::Status Reader::Close() {
  if (gzip_stream_) {
    RETURN_IF_ERROR(gzip_stream_->Close());
    gzip_stream_.reset();
  }
  return absl::OkStatus();
}

absl::StatusOr<Writer> Writer::Create(utils::OutputByteStream* stream,
                                      Compression compression) {
  Writer writer;
  writer.raw_stream_ = stream;

  internal::FileHeader header;
  header.magic[0] = 'B';
  header.magic[1] = 'S';
  header.version = absl::little_endian::FromHost16(kCurrentVersion);
  header.compression = static_cast<uint8_t>(compression);

  RETURN_IF_ERROR(writer.raw_stream_->Write(
      absl::string_view((char*)&header, sizeof(internal::FileHeader))));

  switch (compression) {
    case Compression::kNone:
      break;
    case Compression::kGZIP:
      ASSIGN_OR_RETURN(writer.gzip_stream_,
                       utils::GZipOutputByteStream::Create(writer.raw_stream_));
      break;
  }

  return writer;
}

absl::Status Writer::Write(const absl::string_view blob) {
  internal::RecordHeader header;
  header.length = absl::little_endian::FromHost32(blob.size());

  RETURN_IF_ERROR(stream().Write(
      absl::string_view((char*)&header, sizeof(internal::RecordHeader))));

  return stream().Write(blob);
}

absl::Status Writer::Close() {
  if (gzip_stream_) {
    RETURN_IF_ERROR(gzip_stream_->Close());
    gzip_stream_.reset();
  }
  return absl::OkStatus();
}

}  // namespace blob_sequence
}  // namespace utils
}  // namespace yggdrasil_decision_forests
