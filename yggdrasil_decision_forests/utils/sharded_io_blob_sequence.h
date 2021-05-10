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

#ifndef YGGDRASIL_DECISION_FORESTS_TOOL_SHARDED_IO_BLOG_SEQUENCE_H_
#define YGGDRASIL_DECISION_FORESTS_TOOL_SHARDED_IO_BLOG_SEQUENCE_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/utils/blob_sequence.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Specialization of ShardedReader for TFRecords: Class for the sequential
// reading of sharded TFRecords.
template <typename T>
class BlobSequenceShardedReader : public ShardedReader<T> {
 public:
  BlobSequenceShardedReader() = default;
  absl::Status OpenShard(absl::string_view path) override;
  utils::StatusOr<bool> NextInShard(T* example) override;

 private:
  blob_sequence::Reader reader_;
  file::InputFileCloser file_closer_;
  std::string buffer_;

  DISALLOW_COPY_AND_ASSIGN(BlobSequenceShardedReader);
};

// Specialization of ShardedWriter for TFRecords: Class for the sequential
// writing of sharded TFRecords.
template <typename T>
class BlobSequenceShardedWriter : public ShardedWriter<T> {
 public:
  BlobSequenceShardedWriter() = default;
  ~BlobSequenceShardedWriter() override;
  absl::Status OpenShard(absl::string_view path) final;
  absl::Status WriteInShard(const T& value) final;
  absl::Status CloseWithStatus() final;

 private:
  blob_sequence::Writer writer_;
  file::OutputFileCloser file_closer_;
  std::string buffer_;

  DISALLOW_COPY_AND_ASSIGN(BlobSequenceShardedWriter);
};

template <typename T>
absl::Status BlobSequenceShardedReader<T>::OpenShard(
    const absl::string_view path) {
  ASSIGN_OR_RETURN(auto stream, file::OpenInputFile(path));
  RETURN_IF_ERROR(file_closer_.reset(std::move(stream)));
  ASSIGN_OR_RETURN(reader_,
                   blob_sequence::Reader::Create(file_closer_.stream()));
  return absl::OkStatus();
}

template <typename T>
utils::StatusOr<bool> BlobSequenceShardedReader<T>::NextInShard(T* example) {
  ASSIGN_OR_RETURN(const auto has_content, reader_.Read(&buffer_));
  if (!has_content) {
    return false;
  }
  example->ParseFromArray(buffer_.data(), buffer_.size());
  return true;
}

template <typename T>
absl::Status BlobSequenceShardedWriter<T>::OpenShard(
    const absl::string_view path) {
  RETURN_IF_ERROR(CloseWithStatus());

  ASSIGN_OR_RETURN(auto stream, file::OpenOutputFile(path));
  RETURN_IF_ERROR(file_closer_.reset(std::move(stream)));
  ASSIGN_OR_RETURN(writer_,
                   blob_sequence::Writer::Create(file_closer_.stream()));
  return absl::OkStatus();
}

template <typename T>
absl::Status BlobSequenceShardedWriter<T>::WriteInShard(const T& value) {
  buffer_.clear();
  value.AppendToString(&buffer_);
  return writer_.Write(buffer_);
}

template <typename T>
absl::Status BlobSequenceShardedWriter<T>::CloseWithStatus() {
  if (file_closer_.stream() != nullptr) {
    return absl::OkStatus();
  }
  RETURN_IF_ERROR(writer_.Close());
  return file_closer_.Close();
}

template <typename T>
BlobSequenceShardedWriter<T>::~BlobSequenceShardedWriter() {
  CHECK_OK(CloseWithStatus());
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_TOOL_SHARDED_IO_BLOG_SEQUENCE_H_
