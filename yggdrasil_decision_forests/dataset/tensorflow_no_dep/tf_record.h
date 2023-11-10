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
  absl::StatusOr<bool> Next(google::protobuf::MessageLite* message);

  // Closes the stream.
  absl::Status Close();

  TFRecordReader(std::unique_ptr<file::FileInputByteStream>&& stream)
      : stream_(std::move(stream)) {}

 private:
  // Reads a CRC.
  absl::StatusOr<absl::crc32c_t> ReadCRC();

  std::unique_ptr<file::FileInputByteStream> stream_;
  std::string buffer_;
};

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

// Template implementations
// ========================

template <typename T>
absl::Status ShardedTFRecordReader<T>::OpenShard(const absl::string_view path) {
  ASSIGN_OR_RETURN(reader_, TFRecordReader::Create(path));
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<bool> ShardedTFRecordReader<T>::NextInShard(T* example) {
  return reader_->Next(example);
}

}  // namespace yggdrasil_decision_forests::dataset::tensorflow_no_dep

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_TENSORFLOW_NO_DEP_TF_RECORD_H_
