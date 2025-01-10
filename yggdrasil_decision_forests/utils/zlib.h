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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_ZLIB_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_ZLIB_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"

#define ZLIB_CONST
#include <zlib.h>

namespace yggdrasil_decision_forests::utils {

class GZipInputByteStream : public utils::InputByteStream {
 public:
  static absl::StatusOr<std::unique_ptr<GZipInputByteStream>> Create(
      absl::Nonnull<utils::InputByteStream*> stream,
      size_t buffer_size = 1024 * 1024);

  GZipInputByteStream(utils::InputByteStream* stream, size_t buffer_size);
  ~GZipInputByteStream() override;

  absl::StatusOr<int> ReadUpTo(char* buffer, int max_read) override;
  absl::StatusOr<bool> ReadExactly(char* buffer, int num_read) override;
  absl::Status Close() override;

 private:
  absl::Status CloseDeflateStream();

  // Size of the compressed and uncompressed buffers.
  size_t buffer_size_;
  // Non-owned underlying input stream.
  InputByteStream* stream_ = nullptr;
  // Buffer of compressed data.
  std::vector<Bytef> input_buffer_;
  // Buffer of uncompressed data.
  std::vector<Bytef> output_buffer_;
  // Position of available data in the output buffer to return in the next
  // "ReadUpTo" or "ReadExactly" call.
  size_t output_buffer_begin_ = 0;
  size_t output_buffer_end_ = 0;
  // zlib decompression state machine.
  z_stream deflate_stream_;
  // Was "deflate_stream_" allocated?
  bool deflate_stream_is_allocated_ = false;
};

class GZipOutputByteStream : public utils::OutputByteStream {
 public:
  // Creates a gzip compression stream.
  // Args:
  //   stream: Stream of non-compressed data to compress.
  //   compression_level: Compression level between 0 (not compressed) and 9.
  //   buffer_size: Size of the working buffer. The minimum size depends on the
  //     compressed data, but 1MB should work in most cases.
  //   raw_deflate: If true, uses the raw deflate algorithm (!= zlib or gzip).
  static absl::StatusOr<std::unique_ptr<GZipOutputByteStream>> Create(
      absl::Nonnull<utils::OutputByteStream*> stream,
      int compression_level = Z_DEFAULT_COMPRESSION,
      size_t buffer_size = 1024 * 1024, bool raw_deflate = false);

  GZipOutputByteStream(utils::OutputByteStream* stream, size_t buffer_size);
  ~GZipOutputByteStream() override;

  absl::Status Write(absl::string_view chunk) override;
  absl::Status Flush();
  absl::Status Close() override;

 private:
  absl::Status CloseInflateStream();
  absl::Status WriteImpl(absl::string_view chunk, bool flush);

  // Size of the compressed and uncompressed buffers.
  size_t buffer_size_;
  // Non-owned underlying stream of compressed data.
  OutputByteStream& stream_;
  // Buffer of compressed data.
  std::vector<Bytef> output_buffer_;
  // zlib decompression state machine.
  z_stream deflate_stream_;
  // Was "deflate_stream_" allocated?
  bool deflate_stream_is_allocated_ = false;
};

// Inflates (i.e. decompress) "input" and appends it to "output".
absl::Status Inflate(absl::string_view input, std::string* output,
                     std::string* working_buffer, bool raw_deflate = false);

}  // namespace yggdrasil_decision_forests::utils

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_ZLIB_H_
