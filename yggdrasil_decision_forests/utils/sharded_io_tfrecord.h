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

#ifndef YGGDRASIL_DECISION_FORESTS_TOOL_SHARDED_IO_TFRECORD_H_
#define YGGDRASIL_DECISION_FORESTS_TOOL_SHARDED_IO_TFRECORD_H_

#include <string>

#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/file_system.h"

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/tensorflow.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Specialization of ShardedReader for TFRecords: Class for the sequential
// reading of sharded TFRecords.
template <typename T>
class TFRecordShardedReader : public ShardedReader<T> {
 public:
  TFRecordShardedReader() = default;
  absl::Status OpenShard(absl::string_view path) override;
  utils::StatusOr<bool> NextInShard(T* example) override;

 private:
  std::unique_ptr<tensorflow::io::SequentialRecordReader> reader_;
  std::unique_ptr<tensorflow::RandomAccessFile> file_;
  tensorflow::tstring buffer_;

  DISALLOW_COPY_AND_ASSIGN(TFRecordShardedReader);
};

// Specialization of ShardedWriter for TFRecords: Class for the sequential
// writing of sharded TFRecords.
template <typename T>
class TFRecordShardedWriter : public ShardedWriter<T> {
 public:
  TFRecordShardedWriter() = default;
  ~TFRecordShardedWriter() override;
  absl::Status OpenShard(absl::string_view path) final;
  absl::Status WriteInShard(const T& value) final;
  absl::Status CloseWithStatus() final;

 private:
  std::unique_ptr<tensorflow::io::RecordWriter> writer_;
  std::unique_ptr<tensorflow::WritableFile> file_;
  std::string buffer_;

  DISALLOW_COPY_AND_ASSIGN(TFRecordShardedWriter);
};

template <typename T>
absl::Status TFRecordShardedReader<T>::OpenShard(const absl::string_view path) {
  RETURN_IF_ERROR(ToUtilStatus(tensorflow::Env::Default()->NewRandomAccessFile(
      std::string(path), &file_)));
  reader_ = absl::make_unique<tensorflow::io::SequentialRecordReader>(
      file_.get(),
      tensorflow::io::RecordReaderOptions::CreateRecordReaderOptions("GZIP"));

  return absl::OkStatus();
}

template <typename T>
utils::StatusOr<bool> TFRecordShardedReader<T>::NextInShard(T* example) {
  const auto tf_status = reader_->ReadRecord(&buffer_);
  if (tf_status.ok()) {
    // Valid example.
    example->ParseFromArray(buffer_.data(), buffer_.size());
    return true;
  } else if (tf_status.code() == tensorflow::error::OUT_OF_RANGE) {
    // No more examples available.
    return false;
  } else {
    // Reading error.
    return ToUtilStatus(tf_status);
  }
}

template <typename T>
absl::Status TFRecordShardedWriter<T>::OpenShard(const absl::string_view path) {
  RETURN_IF_ERROR(CloseWithStatus());
  RETURN_IF_ERROR(ToUtilStatus(
      tensorflow::Env::Default()->NewWritableFile(std::string(path), &file_)));
  writer_ = absl::make_unique<tensorflow::io::RecordWriter>(
      file_.get(),
      tensorflow::io::RecordWriterOptions::CreateRecordWriterOptions("GZIP"));
  return absl::OkStatus();
}

template <typename T>
absl::Status TFRecordShardedWriter<T>::WriteInShard(const T& value) {
  buffer_.clear();
  value.AppendToString(&buffer_);
  return ToUtilStatus(writer_->WriteRecord(buffer_));
}

template <typename T>
absl::Status TFRecordShardedWriter<T>::CloseWithStatus() {
  if (!writer_) {
    return absl::OkStatus();
  }
  RETURN_IF_ERROR(ToUtilStatus(writer_->Close()));
  writer_ = nullptr;
  RETURN_IF_ERROR(ToUtilStatus(file_->Close()));
  file_ = nullptr;
  return absl::OkStatus();
}

template <typename T>
TFRecordShardedWriter<T>::~TFRecordShardedWriter() {
  CHECK_OK(CloseWithStatus());
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_TOOL_SHARDED_IO_TFRECORD_H_
