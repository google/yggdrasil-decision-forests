/*
 * Copyright 2021 Google LLC.
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

#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace blob_sequence {

utils::StatusOr<Reader> Reader::Create(utils::InputByteStream* stream) {
  Reader reader;
  reader.stream_ = stream;

  internal::FileHeader header;
  ASSIGN_OR_RETURN(const auto has_content,
                   reader.stream_->ReadExactly((char*)&header,
                                               sizeof(internal::FileHeader)));
  if (!has_content) {
    return absl::InvalidArgumentError("Empty stream");
  }
  if (header.magic[0] != 'B' || header.magic[1] != 'S') {
    return absl::InvalidArgumentError("Invalid header");
  }
  reader.version_ = absl::little_endian::ToHost16(header.version);

  return reader;
}

utils::StatusOr<bool> Reader::Read(std::string* blob) {
  internal::RecordHeader header;
  ASSIGN_OR_RETURN(
      auto has_content,
      stream_->ReadExactly((char*)&header, sizeof(internal::RecordHeader)));
  if (!has_content) {
    // End of BS.
    return false;
  }

  header.length = absl::little_endian::ToHost32(header.length);

  blob->resize(header.length);
  ASSIGN_OR_RETURN(has_content,
                   stream_->ReadExactly(&(*blob)[0], header.length));
  if (!has_content) {
    return absl::InvalidArgumentError("Truncated blob");
  }

  return true;
}

absl::Status Reader::Close() { return absl::OkStatus(); }

utils::StatusOr<Writer> Writer::Create(utils::OutputByteStream* stream) {
  Writer writer;
  writer.stream_ = stream;

  internal::FileHeader header;
  header.magic[0] = 'B';
  header.magic[1] = 'S';
  header.version = absl::little_endian::FromHost16(0);

  RETURN_IF_ERROR(writer.stream_->Write(
      absl::string_view((char*)&header, sizeof(internal::FileHeader))));

  return writer;
}

absl::Status Writer::Write(const absl::string_view blob) {
  internal::RecordHeader header;
  header.length = absl::little_endian::FromHost32(blob.size());

  RETURN_IF_ERROR(stream_->Write(
      absl::string_view((char*)&header, sizeof(internal::RecordHeader))));

  return stream_->Write(blob);
}

absl::Status Writer::Close() { return absl::OkStatus(); }

}  // namespace blob_sequence
}  // namespace utils
}  // namespace yggdrasil_decision_forests
