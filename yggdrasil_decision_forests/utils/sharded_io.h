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

// Utility to read and write sharded files.

#ifndef YGGDRASIL_DECISION_FORESTS_TOOL_SHARDED_IO_H_
#define YGGDRASIL_DECISION_FORESTS_TOOL_SHARDED_IO_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Generates the file paths corresponding to a simple path, a sharded path, a
// glob or a comma separated set of path.
//
// Comma separation, sharding and glob can be combined.
// The output paths are stored alphabetically.
//
// Example:
//
//   a/b/c
//   a/b/c@2
//   a/*/c@2
//   a/b/c1,a/b/c2
absl::Status ExpandInputShards(absl::string_view sharded_path,
                               std::vector<std::string>* paths);

// Generates the file paths corresponding to a sharded path, or a simple
// path.
absl::Status ExpandOutputShards(const absl::string_view sharded_path,
                                std::vector<std::string>* paths);

// Helper class for the sequential reading of sharded files.
template <typename T>
class ShardedReader {
 public:
  virtual ~ShardedReader() = default;

  // Open a typed sharded path.
  absl::Status Open(absl::string_view sharded_path);

  // Open a list of files to read.
  absl::Status Open(const std::vector<std::string>& paths);

  // Try to retrieve the next available value. If no more value are
  // available, returns false.
  utils::StatusOr<bool> Next(T* value);

 protected:
  // Start reading a given file (i.e. not a sharded path).
  virtual absl::Status OpenShard(absl::string_view path) = 0;

  // Try to retive the next available example in the last file open with
  // "OpenShard". Returns false if no more examples are available.
  virtual utils::StatusOr<bool> NextInShard(T* value) = 0;

 private:
  // Try to open the next file in "paths_". If all files have been read, returns
  // false.
  utils::StatusOr<bool> OpenNextShard();

  // List of files to read.
  std::vector<std::string> paths_;

  // Index of the file currently read. -1 (initial value) indicates that no file
  // is currently read.
  int cur_path_idx_ = -1;
};

// Helper class for the sequential writing of sharded files.
template <typename T>
class ShardedWriter {
 public:
  virtual ~ShardedWriter() = default;

  // Open a typed sharded path. If num_records_by_shard==-1, all the examples
  // are written in the first shard.
  absl::Status Open(absl::string_view sharded_path,
                    const int64_t num_records_by_shard);

  // Write a new record.
  absl::Status Write(const T& value);

  // Closes and return an error if something went wrong. The destructor
  // will also close the file, but one cannot catch the error in that
  // case, and it will lead to a CHECK() failure.
  virtual absl::Status CloseWithStatus() = 0;

 protected:
  // Start writing in a given file (i.e. not a sharded path).
  virtual absl::Status OpenShard(absl::string_view path) = 0;

  // Write a new record in the currently open shard i.e. the shard previously
  // open with "OpenShard".
  virtual absl::Status WriteInShard(const T& value) = 0;

 private:
  // Open the next shard i.e. call "OpenShard" with the path of the next shard.
  absl::Status OpenNextShard();

  // List of files to read.
  std::vector<std::string> paths_;

  // Index of the file currently read. -1 (initial value) indicates that no file
  // is currently read.
  int cur_path_idx_ = -1;

  // Number of records per shard.
  int64_t num_records_by_shard_;

  // Number of records written in the current shard so far.
  int64_t num_records_in_cur_shard_;
};

template <typename T>
absl::Status ShardedWriter<T>::Open(absl::string_view sharded_path,
                                    const int64_t num_records_by_shard) {
  RETURN_IF_ERROR(ExpandOutputShards(sharded_path, &paths_));
  num_records_by_shard_ = num_records_by_shard;
  num_records_in_cur_shard_ = 0;
  return OpenNextShard();
}

template <typename T>
absl::Status ShardedWriter<T>::Write(const T& value) {
  if (num_records_by_shard_ != -1 &&
      num_records_in_cur_shard_ >= num_records_by_shard_) {
    RETURN_IF_ERROR(OpenNextShard());
  }
  RETURN_IF_ERROR(WriteInShard(value));
  num_records_in_cur_shard_++;
  return absl::OkStatus();
}

template <typename T>
absl::Status ShardedWriter<T>::OpenNextShard() {
  num_records_in_cur_shard_ = 0;
  if (cur_path_idx_ + 1 >= paths_.size()) {
    LOG(INFO)
        << "Not enough shards allocated. Continue to write in the last shard.";
    return absl::OkStatus();
  }
  cur_path_idx_++;
  return OpenShard(paths_[cur_path_idx_]);
}

template <typename T>
absl::Status ShardedReader<T>::Open(const absl::string_view sharded_path) {
  RETURN_IF_ERROR(ExpandInputShards(sharded_path, &paths_));
  ASSIGN_OR_RETURN(bool has_value, OpenNextShard());
  if (!has_value) {
    return absl::NotFoundError(absl::StrCat(sharded_path, " is empty."));
  }
  return absl::OkStatus();
}

template <typename T>
absl::Status ShardedReader<T>::Open(const std::vector<std::string>& paths) {
  paths_ = paths;
  ASSIGN_OR_RETURN(bool has_value, OpenNextShard());
  if (!has_value) {
    return absl::NotFoundError("No file provided.");
  }
  return absl::OkStatus();
}

template <typename T>
utils::StatusOr<bool> ShardedReader<T>::Next(T* value) {
  bool has_next_shard;
  do {
    ASSIGN_OR_RETURN(bool has_next, NextInShard(value));
    if (has_next) {
      return true;
    }
    ASSIGN_OR_RETURN(has_next_shard, OpenNextShard());
  } while (has_next_shard);
  return false;
}

template <typename T>
utils::StatusOr<bool> ShardedReader<T>::OpenNextShard() {
  cur_path_idx_++;
  if (cur_path_idx_ >= paths_.size()) {
    return false;
  }
  RETURN_IF_ERROR(OpenShard(paths_[cur_path_idx_]));
  return true;
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_TOOL_SHARDED_IO_H_
