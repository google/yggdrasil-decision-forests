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
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include <zconf.h>

#define ZLIB_CONST
#include <zlib.h>

namespace yggdrasil_decision_forests::utils {

absl::StatusOr<std::unique_ptr<GZipInputByteStream>>
GZipInputByteStream::Create(std::unique_ptr<utils::InputByteStream>&& stream,
                            size_t buffer_size) {
  auto gz_stream =
      std::make_unique<GZipInputByteStream>(std::move(stream), buffer_size);
  std::memset(&gz_stream->deflate_stream_, 0,
              sizeof(gz_stream->deflate_stream_));
  if (inflateInit2(&gz_stream->deflate_stream_, 16 + MAX_WBITS) != Z_OK) {
    return absl::InternalError("Cannot initialize gzip stream");
  }
  gz_stream->deflate_stream_is_allocated_ = true;
  return gz_stream;
}

GZipInputByteStream::GZipInputByteStream(
    std::unique_ptr<utils::InputByteStream>&& stream, size_t buffer_size)
    : buffer_size_(buffer_size), stream_(std::move(stream)) {
  input_buffer_.resize(buffer_size_);
  output_buffer_.resize(buffer_size_);
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
      const auto zlib_error = inflate(&deflate_stream_, Z_NO_FLUSH);
      if (zlib_error != Z_OK && zlib_error != Z_STREAM_END) {
        inflateEnd(&deflate_stream_);
        return absl::InternalError(absl::StrCat("Internal error", zlib_error));
      }

      const int produced_bytes = buffer_size_ - deflate_stream_.avail_out;
      output_buffer_begin_ = 0;
      output_buffer_end_ = produced_bytes;
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

absl::StatusOr<std::unique_ptr<GZipOutputByteStream>>
GZipOutputByteStream::Create(std::unique_ptr<utils::OutputByteStream>&& stream,
                             int compression_level, size_t buffer_size) {
  if (compression_level != Z_DEFAULT_COMPRESSION) {
    STATUS_CHECK_GT(compression_level, Z_NO_COMPRESSION);
    STATUS_CHECK_LT(compression_level, Z_BEST_COMPRESSION);
  }
  auto gz_stream =
      std::make_unique<GZipOutputByteStream>(std::move(stream), buffer_size);
  std::memset(&gz_stream->deflate_stream_, 0,
              sizeof(gz_stream->deflate_stream_));
  if (deflateInit2(&gz_stream->deflate_stream_, compression_level, Z_DEFLATED,
                   MAX_WBITS + 16,
                   /*memLevel=*/8,  // 8 is the recommended default
                   Z_DEFAULT_STRATEGY) != Z_OK) {
    return absl::InternalError("Cannot initialize gzip stream");
  }
  gz_stream->deflate_stream_is_allocated_ = true;
  return gz_stream;
}

GZipOutputByteStream::GZipOutputByteStream(
    std::unique_ptr<utils::OutputByteStream>&& stream, size_t buffer_size)
    : buffer_size_(buffer_size), stream_(std::move(stream)) {
  output_buffer_.resize(buffer_size_);
}

GZipOutputByteStream::~GZipOutputByteStream() {
  CloseInflateStream().IgnoreError();
}

absl::Status GZipOutputByteStream::Write(absl::string_view chunk) {
  return WriteImpl(chunk, false);
}

absl::Status GZipOutputByteStream::WriteImpl(absl::string_view chunk,
                                             bool flush) {
  if (chunk.empty() && !flush) {
    return absl::OkStatus();
  }
  deflate_stream_.next_in = reinterpret_cast<const Bytef*>(chunk.data());
  deflate_stream_.avail_in = chunk.size();

  while (true) {
    deflate_stream_.next_out = output_buffer_.data();
    deflate_stream_.avail_out = buffer_size_;

    const auto zlib_error =
        deflate(&deflate_stream_, flush ? Z_FINISH : Z_NO_FLUSH);

    if (flush) {
      if (zlib_error != Z_STREAM_END && !chunk.empty()) {
        deflateEnd(&deflate_stream_);
        return absl::InternalError(absl::StrCat("Internal error ", zlib_error,
                                                ". Output buffer too small"));
      }
    } else {
      if (zlib_error != Z_OK) {
        deflateEnd(&deflate_stream_);
        return absl::InternalError(absl::StrCat("Internal error ", zlib_error));
      }
    }

    const size_t compressed_bytes = buffer_size_ - deflate_stream_.avail_out;

    if (compressed_bytes > 0) {
      RETURN_IF_ERROR(stream_->Write(absl::string_view{
          reinterpret_cast<char*>(output_buffer_.data()), compressed_bytes}));
    }

    if (deflate_stream_.avail_out != 0) {
      break;
    }
  }

  return absl::OkStatus();
}

absl::Status GZipOutputByteStream::Close() {
  RETURN_IF_ERROR(CloseInflateStream());
  if (stream_) {
    return stream_->Close();
  }
  return absl::OkStatus();
}

absl::Status GZipOutputByteStream::CloseInflateStream() {
  if (deflate_stream_is_allocated_) {
    deflate_stream_is_allocated_ = false;
    RETURN_IF_ERROR(WriteImpl("", true));
    if (deflateEnd(&deflate_stream_) != Z_OK) {
      return absl::InternalError("Cannot close deflate");
    }
  }
  return absl::OkStatus();
}

absl::Status Inflate(absl::string_view input, std::string* output,
                     std::string* working_buffer) {
  if (working_buffer->size() < 1024) {
    return absl::InvalidArgumentError(
        "worker buffer should be at least 1024 bytes");
  }
  z_stream stream;
  std::memset(&stream, 0, sizeof(stream));
  // Note: A negative window size indicate to use the raw deflate algorithm (!=
  // zlib or gzip).
  if (inflateInit2(&stream, -15) != Z_OK) {
    return absl::InternalError("Cannot initialize gzip stream");
  }
  stream.next_in = reinterpret_cast<const Bytef*>(input.data());
  stream.avail_in = input.size();

  while (true) {
    stream.next_out = reinterpret_cast<Bytef*>(&(*working_buffer)[0]);
    stream.avail_out = working_buffer->size();
    const auto zlib_error = inflate(&stream, Z_NO_FLUSH);
    if (zlib_error != Z_OK && zlib_error != Z_STREAM_END) {
      inflateEnd(&stream);
      return absl::InternalError(absl::StrCat("Internal error", zlib_error));
    }
    if (stream.avail_out == 0) {
      break;
    }
    const size_t produced_bytes = working_buffer->size() - stream.avail_out;
    absl::StrAppend(output,
                    absl::string_view{working_buffer->data(), produced_bytes});
    if (zlib_error == Z_STREAM_END) {
      break;
    }
  }
  inflateEnd(&stream);

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::utils

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_GZIP_H_
