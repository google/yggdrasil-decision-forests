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

// A BlogSequence (BS) is a sequence of blobs (list of bytes) stored in a byte
// stream (e.g. a file).
//
// Writing usage example:
//
//   auto output_stream = file::OpenOutputFile(path).value();
//   auto writer = blob_sequence::Writer::Create(output_stream.get()).value();
//   CHECK_OK(writer.Write("HELLO"));
//   CHECK_OK(writer.Write("WORLD"));
//   CHECK_OK(writer.Close());
//   CHECK_OK(output_stream->Close());
//
// Reading usage example:
//
//   auto input_stream = file::OpenInputFile(path).value();
//   auto reader = blob_sequence::Reader::Create(input_stream.get()).value();
//   std::string blob;
//   CHECK(reader.Read(&blob).value());
//   CHECK(reader.Read(&blob).value());
//   CHECK_OK(reader.Close());
//   CHECK_OK(input_stream->Close());
//
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_BLOB_SEQUENCE_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_BLOB_SEQUENCE_H_

#include "yggdrasil_decision_forests/utils/bytestream.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace blob_sequence {

// Blog sequence reader.
class Reader {
 public:
  // Creates a reader attached to a stream. Does not take ownership of "stream".
  static utils::StatusOr<Reader> Create(utils::InputByteStream* stream);

  // Creates a non attached reader.
  Reader() {}

  // Reads the next blob. Return false iff no more blobs are available.
  utils::StatusOr<bool> Read(std::string* blob);

  // Closes the reader. Does not close the stream (passed in the constructor)
  // Should be called BEFORE the stream is closed (if the stream has the concept
  // of being closed).
  absl::Status Close();

 private:
  // Non-owned input stream.
  InputByteStream* stream_ = nullptr;
  uint16_t version_;
};

// Blog sequence writer.
class Writer {
 public:
  // Creates a writer attached to a stream.  Does not take ownership of
  // "stream".
  static utils::StatusOr<Writer> Create(utils::OutputByteStream* stream);

  // Creates a non attached writer.
  Writer() {}

  // Writes a blob.
  absl::Status Write(const absl::string_view blob);

  // Closes the writer. Does not close the stream passed in the constructor.
  // Should be called BEFORE the stream is closed (if the stream has the concept
  // of being closed).
  absl::Status Close();

 private:
  // Non-owned output stream.
  OutputByteStream* stream_ = nullptr;
};

namespace internal {

// File header.
// Integer are stored in little endian.
struct FileHeader {
  // Should be 'BS';
  uint8_t magic[2];

  // Version of the format.
  // Version:
  //   0: Initial version.
  uint16_t version;

  // Reserved until used (instead of creating a per-version header).
  // Should remain zero until used.
  uint32_t reserved = 0;
};

// Record header.
// Integer are stored in little endian.
struct RecordHeader {
  // Size of the record in bytes, excluding the header.
  uint32_t length;
};

};  // namespace internal

}  // namespace blob_sequence
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_BLOB_SEQUENCE_H_
