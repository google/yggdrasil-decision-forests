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

#include <stdint.h>

#include <memory>
#include <string>
#include <utility>

#include "absl/crc/crc32c.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"

namespace yggdrasil_decision_forests::dataset::tensorflow_no_dep {

// Reads a TFRecord container.
// Currently, only supports non-compressed TFRecords.
class TFRecordReader {
 public:
  // Opens a TFRecord for reading.
  static absl::StatusOr<std::unique_ptr<TFRecordReader>> Create(
      absl::string_view path);

  ~TFRecordReader();

  // Gets the next message. Returns true and populate "message" if a new message
  // is available. Return false and do not modify "message" if the end-of-file
  // is reached.
  //
  // If message==null, skip the message.
  absl::StatusOr<bool> Next(google::protobuf::MessageLite* message);

  // Closes the stream.
  absl::Status Close();

  TFRecordReader(std::unique_ptr<file::FileInputByteStream>&& stream)
      : stream_(std::move(stream)) {}

  // Value of the last read record. Includes skipped messages.
  const std::string& buffer() const { return buffer_; }

 private:
  // Reads a CRC.
  absl::StatusOr<absl::crc32c_t> ReadCRC();

  std::unique_ptr<file::FileInputByteStream> stream_;
  std::string buffer_;
};

// Reads a set of sharded TFRecords.
template <typename T>
class ShardedTFRecordReader : public utils::ShardedReader<T> {
 public:
  ShardedTFRecordReader() = default;
  absl::Status OpenShard(absl::string_view path) override;
  absl::StatusOr<bool> NextInShard(T* example) override;

 private:
  std::unique_ptr<TFRecordReader> reader_;
  DISALLOW_COPY_AND_ASSIGN(ShardedTFRecordReader);
};

// Writes a TFRecord container.
// Currently, only supports non-compressed TFRecords.
class TFRecordWriter {
 public:
  // Opens a TFRecord for reading.
  static absl::StatusOr<std::unique_ptr<TFRecordWriter>> Create(
      absl::string_view path);

  ~TFRecordWriter();

  absl::Status Write(const google::protobuf::MessageLite& message);
  absl::Status Write(absl::string_view data);

  // Closes the stream.
  absl::Status Close();

  TFRecordWriter(std::unique_ptr<file::FileOutputByteStream>&& stream)
      : stream_(std::move(stream)) {}

 private:
  std::unique_ptr<file::FileOutputByteStream> stream_;
  std::string buffer_;
};

// Write a set of sharded TFRecords.
template <typename T>
class ShardedTFRecordWriter : public utils::ShardedWriter<T> {
 public:
  ShardedTFRecordWriter() = default;
  absl::Status OpenShard(absl::string_view path) final;
  absl::Status WriteInShard(const T& value) final;
  absl::Status CloseWithStatus() final;

 private:
  std::unique_ptr<TFRecordWriter> writer_;
  DISALLOW_COPY_AND_ASSIGN(ShardedTFRecordWriter);
};

// Template implementations
// ========================

template <typename T>
absl::Status ShardedTFRecordReader<T>::OpenShard(const absl::string_view path) {
  if (reader_) {
    RETURN_IF_ERROR(reader_->Close());
    reader_.reset();
  }
  ASSIGN_OR_RETURN(reader_, TFRecordReader::Create(path));
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
  ASSIGN_OR_RETURN(writer_, TFRecordWriter::Create(path));
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
