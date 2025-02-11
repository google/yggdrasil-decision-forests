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

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_TENSORFLOW_NO_DEP_TF_RECORD_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_TENSORFLOW_NO_DEP_TF_RECORD_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/crc/crc32c.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/zlib.h"

namespace yggdrasil_decision_forests::dataset::tensorflow_no_dep {

// Reads a TFRecord container.
// Currently, only supports non-compressed TFRecords.
class TFRecordReader {
 public:
  // Opens a TFRecord for reading.
  static absl::StatusOr<std::unique_ptr<TFRecordReader>> Create(
      absl::string_view path, bool compressed = false);

  ~TFRecordReader();

  // Gets the next message. Returns true and populate "message" if a new message
  // is available. Return false and do not modify "message" if the end-of-file
  // is reached.
  //
  // If message==null, skip the message.
  absl::StatusOr<bool> Next(google::protobuf::MessageLite* message);

  // Closes the stream.
  absl::Status Close();

  // Value of the last read record. Includes skipped messages.
  const std::string& buffer() const { return buffer_; }

  TFRecordReader() {}

 private:
  utils::InputByteStream& stream() {
    return zlib_stream_ ? *zlib_stream_ : *raw_stream_;
  }

  // Reads a CRC.
  absl::StatusOr<absl::crc32c_t> ReadCRC();

  // Underlying stream. If nullptr, the reader is closed.
  std::unique_ptr<utils::InputByteStream> raw_stream_;
  // Optional stream, if zlib compression is enabled.
  std::unique_ptr<utils::GZipInputByteStream> zlib_stream_;
  std::string buffer_;
};

// Reads a set of sharded TFRecords.
template <typename T>
class ShardedTFRecordReader : public utils::ShardedReader<T> {
 public:
  ShardedTFRecordReader(bool compressed = false) : compressed_(compressed) {};
  absl::Status OpenShard(absl::string_view path) override;
  absl::StatusOr<bool> NextInShard(T* example) override;

 private:
  std::unique_ptr<TFRecordReader> reader_;
  bool compressed_;
  DISALLOW_COPY_AND_ASSIGN(ShardedTFRecordReader);
};

template <typename T>
class ShardedCompressedTFRecordReader : public ShardedTFRecordReader<T> {
 public:
  ShardedCompressedTFRecordReader() : ShardedTFRecordReader<T>(true) {};
};

// Writes a TFRecord container.
// Currently, only supports non-compressed TFRecords.
class TFRecordWriter {
 public:
  // Opens a TFRecord for reading.
  static absl::StatusOr<std::unique_ptr<TFRecordWriter>> Create(
      absl::string_view path, bool compressed = false);

  ~TFRecordWriter();

  absl::Status Write(const google::protobuf::MessageLite& message);
  absl::Status Write(absl::string_view data);

  // Closes the stream.
  absl::Status Close();

  TFRecordWriter() {}

 private:
  utils::OutputByteStream& stream() {
    return zlib_stream_ ? *zlib_stream_ : *raw_stream_;
  }
  // Underlying stream. If nullptr, the writer is closed.
  std::unique_ptr<utils::OutputByteStream> raw_stream_;
  // Optional stream, if zlib compression is enabled.
  std::unique_ptr<utils::GZipOutputByteStream> zlib_stream_;
  std::string buffer_;
};

// Write a set of sharded TFRecords.
template <typename T>
class ShardedTFRecordWriter : public utils::ShardedWriter<T> {
 public:
  ShardedTFRecordWriter(bool compressed = false) : compressed_(compressed) {};
  absl::Status OpenShard(absl::string_view path) final;
  absl::Status WriteInShard(const T& value) final;
  absl::Status CloseWithStatus() final;

 private:
  std::unique_ptr<TFRecordWriter> writer_;
  bool compressed_;
  DISALLOW_COPY_AND_ASSIGN(ShardedTFRecordWriter);
};

template <typename T>
class ShardedCompressedTFRecordWriter : public ShardedTFRecordWriter<T> {
 public:
  ShardedCompressedTFRecordWriter() : ShardedTFRecordWriter<T>(true) {};
  DISALLOW_COPY_AND_ASSIGN(ShardedCompressedTFRecordWriter);
};

// Template implementations
// ========================

template <typename T>
absl::Status ShardedTFRecordReader<T>::OpenShard(const absl::string_view path) {
  if (reader_) {
    RETURN_IF_ERROR(reader_->Close());
    reader_.reset();
  }
  ASSIGN_OR_RETURN(reader_, TFRecordReader::Create(path, compressed_));
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<bool> ShardedTFRecordReader<T>::NextInShard(T* example) {
  ASSIGN_OR_RETURN(const bool has_value, reader_->Next(example));
  if (!has_value) {
    RETURN_IF_ERROR(reader_->Close());
    reader_.reset();
  }
  return has_value;
}

template <typename T>
absl::Status ShardedTFRecordWriter<T>::OpenShard(absl::string_view path) {
  if (writer_) {
    RETURN_IF_ERROR(writer_->Close());
    writer_.reset();
  }
  ASSIGN_OR_RETURN(writer_, TFRecordWriter::Create(path, compressed_));
  return absl::OkStatus();
}

template <typename T>
absl::Status ShardedTFRecordWriter<T>::WriteInShard(const T& value) {
  return writer_->Write(value);
}

template <typename T>
absl::Status ShardedTFRecordWriter<T>::CloseWithStatus() {
  if (writer_) {
    RETURN_IF_ERROR(writer_->Close());
    writer_.reset();
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::dataset::tensorflow_no_dep

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_TENSORFLOW_NO_DEP_TF_RECORD_H_
