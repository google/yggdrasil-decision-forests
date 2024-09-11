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

#ifndef THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_ZLIB_H_
#define THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_ZLIB_H_

#include <cstddef>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"
#include <zlib.h>

namespace yggdrasil_decision_forests::utils {

class GZipInputByteStream : public utils::InputByteStream {
 public:
  static absl::StatusOr<std::unique_ptr<GZipInputByteStream>> Create(
      std::unique_ptr<utils::InputByteStream>&& stream,
      size_t buffer_size = 3 /*1024 * 1024*/);

  GZipInputByteStream(std::unique_ptr<utils::InputByteStream>&& stream,
                      size_t buffer_size);
  ~GZipInputByteStream() override;

  absl::StatusOr<int> ReadUpTo(char* buffer, int max_read) override;
  absl::StatusOr<bool> ReadExactly(char* buffer, int num_read) override;
  absl::Status Close();

 private:
  absl::Status CloseDeflateStream();

  // Size of the compressed and uncompressed buffers.
  size_t buffer_size_;
  // Underlying stream of compressed data.
  std::unique_ptr<utils::InputByteStream> stream_;
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
  bool deflate_stream_is_allocated_ = true;
};

}  // namespace yggdrasil_decision_forests::utils

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_ZLIB_H_
