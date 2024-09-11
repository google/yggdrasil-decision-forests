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

#ifndef THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_GZIP_H_
#define THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_GZIP_H_

#include "yggdrasil_decision_forests/utils/zlib.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include <zconf.h>
#include <zlib.h>

namespace yggdrasil_decision_forests::utils {

absl::StatusOr<std::unique_ptr<GZipInputByteStream>>
GZipInputByteStream::Create(std::unique_ptr<utils::InputByteStream>&& stream,
                            size_t buffer_size) {
  return std::make_unique<GZipInputByteStream>(std::move(stream), buffer_size);
}

GZipInputByteStream::GZipInputByteStream(
    std::unique_ptr<utils::InputByteStream>&& stream, size_t buffer_size)
    : buffer_size_(buffer_size), stream_(std::move(stream)) {
  deflate_stream_.zalloc = Z_NULL;
  deflate_stream_.zfree = Z_NULL;
  deflate_stream_.opaque = Z_NULL;
  deflate_stream_.avail_in = 0;
  deflate_stream_.next_in = Z_NULL;
  if (inflateInit2(&deflate_stream_, 16 + MAX_WBITS) != Z_OK) {
    CHECK(false);
  }

  input_buffer_.resize(buffer_size_);
  output_buffer_.resize(buffer_size_);
  deflate_stream_.next_in = input_buffer_.data();
  deflate_stream_.next_out = output_buffer_.data();

  deflate_stream_.avail_in = 0;
  deflate_stream_.avail_out = 0;
}

absl::StatusOr<int> GZipInputByteStream::ReadUpTo(char* buffer, int max_read) {
  while (true) {    // Compressed data reading block.
    while (true) {  // Decompression block.
      // 1. Is there decompressed data available?
      if (output_buffer_begin_ < output_buffer_end_) {
        // Copy data from the output butter to the user buffer
        size_t n = std::min(static_cast<size_t>(max_read),
                            output_buffer_end_ - output_buffer_begin_);
        std::memcpy(buffer, output_buffer_.data() + output_buffer_begin_, n);
        output_buffer_begin_ += n;
        return n;
      }

      if (deflate_stream_.avail_in == 0) {
        // There are not more compressed data available loaded in the input
        // buffer.
        break;
      }

      // 2. Continue the decompression of the data in the input buffer.
      deflate_stream_.avail_out = buffer_size_;
      deflate_stream_.next_out = output_buffer_.data();
      int error_status = inflate(&deflate_stream_, Z_NO_FLUSH);
      switch (error_status) {
        case Z_NEED_DICT:
        case Z_DATA_ERROR:
        case Z_MEM_ERROR:
          inflateEnd(&deflate_stream_);
          return absl::InternalError("Internal error");
      }
      const int num_decompressed = buffer_size_ - deflate_stream_.avail_out;
      output_buffer_begin_ = 0;
      output_buffer_end_ = num_decompressed;

      if (num_decompressed == 0) {
        break;
      }
    }

    // 3. Read non-compressed data from the underlying stream to the input
    // buffer.
    ASSIGN_OR_RETURN(
        int compressed_available,
        stream_->ReadUpTo(reinterpret_cast<char*>(input_buffer_.data()),
                          buffer_size_));

    if (compressed_available == 0) {
      // No more data. End of stream.
      return 0;
    }
    deflate_stream_.next_in = input_buffer_.data();
    deflate_stream_.avail_in = compressed_available;
    output_buffer_begin_ = 0;
    output_buffer_end_ = 0;
  }
}

absl::StatusOr<bool> GZipInputByteStream::ReadExactly(char* buffer,
                                                      int num_read) {
  while (num_read > 0) {
    ASSIGN_OR_RETURN(int n, ReadUpTo(buffer, num_read));
    if (n == 0) {
      return false;
    }
    num_read -= n;
    buffer += n;
  }
  return true;
}

GZipInputByteStream::~GZipInputByteStream() {
  CloseDeflateStream().IgnoreError();
}

absl::Status GZipInputByteStream::Close() {
  RETURN_IF_ERROR(CloseDeflateStream());
  if (stream_) {
    return stream_->Close();
  }
  return absl::OkStatus();
}

absl::Status GZipInputByteStream::CloseDeflateStream() {
  if (deflate_stream_is_allocated_) {
    deflate_stream_is_allocated_ = false;
    if (inflateEnd(&deflate_stream_) != Z_OK) {
      return absl::InternalError("Cannot close deflate");
    }
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::utils

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_GZIP_H_
