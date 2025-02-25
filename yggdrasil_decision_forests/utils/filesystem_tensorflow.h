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

// IWYU pragma: private, include "yggdrasil_decision_forests/utils/filesystem.h"
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_FILESYSTEM_TENSORFLOW_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_FILESYSTEM_TENSORFLOW_H_

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"
#include "yggdrasil_decision_forests/utils/filesystem_interface.h"
#include "yggdrasil_decision_forests/utils/filesystem_tensorflow_interface.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"

#define EXTERNAL_FILESYSTEM

namespace file {

inline yggdrasil_decision_forests ::utils::filesystem::FileSystemInterface&
Interface() {
  return yggdrasil_decision_forests ::utils::filesystem::tf_impl::Interface();
}

// Joins a list of directory or file names with a directory separator.
//
// Usage example:
//   JoinPathList({"a","b"})
//   // Returns "a/b" (on linux).
//
std::string JoinPathList(std::initializer_list<absl::string_view> paths);

// Joins a list of directory / file names with a directory separator.
//
// Usage example:
//   JoinPathList("a","b")
//   // Returns "a/b" (on linux).
//
template <typename... Args>
std::string JoinPath(Args... args) {
  return JoinPathList({args...});
}

// Generates the list of files that match a sharded path.
//
// Usage example:
//   std::vector<std::string> names;
//   GenerateShardedFilenames("a/b@3", &names);
//   // "names" is now "a/b-00000-to-00003", "a/b-00001-to-00003" and
//   // "a/b-00002-to-00003".
//
// The shard count should be specified i.e. "a/b@*" is not valid.
//
bool GenerateShardedFilenames(absl::string_view spec,
                              std::vector<std::string>* names);

// Generates a sharded file path.
//
// Usage example:
//   GenerateShardedFileSpec("/a/b",5) -> "/a/b@5"
inline std::string GenerateShardedFileSpec(absl::string_view path,
                                           int num_shards) {
  return absl::StrCat(path, "@", num_shards);
}

// Generates the list of files that match a pattern.
//
// Syntax:
//   - "*" matches any number of characters.
//   - "?" match any one character.
//   - "[" and "]" behaving like regular expressions.
//
// For example, "a/b*" might returns the files "a/b1", "a/b2" and "a/btoto".
//
// Does not work recursively. The pattern does not apply in directories.
//
absl::Status Match(absl::string_view pattern, std::vector<std::string>* results,
                   int options);

// Creates the directory "path" as well as all its missing parents.
absl::Status RecursivelyCreateDir(absl::string_view path, int options);

// Delete the directory "path".
absl::Status RecursivelyDelete(absl::string_view path, int options);

// Placeholder empty function used in the "options" argument of some functions.
constexpr int Defaults() { return 0; }

class FileInputByteStream
    : public yggdrasil_decision_forests::utils::InputByteStream {
 public:
  FileInputByteStream() : stream_(Interface().CreateInputByteStream()) {}
  absl::Status Open(absl::string_view path) { return stream_->Open(path); }
  absl::StatusOr<int> ReadUpTo(char* buffer, int max_read) {
    return stream_->ReadUpTo(buffer, max_read);
  }
  absl::StatusOr<bool> ReadExactly(char* buffer, int num_read) {
    return stream_->ReadExactly(buffer, num_read);
  }
  absl::Status Close() { return stream_->Close(); }

 private:
  std::unique_ptr<yggdrasil_decision_forests::utils::FileInputByteStream>
      stream_;
};

class FileOutputByteStream
    : public yggdrasil_decision_forests::utils::OutputByteStream {
 public:
  FileOutputByteStream() : stream_(Interface().CreateOutputByteStream()) {}
  absl::Status Open(absl::string_view path) { return stream_->Open(path); }
  absl::Status Write(absl::string_view chunk) { return stream_->Write(chunk); }
  absl::Status Close() { return stream_->Close(); }

 private:
  std::unique_ptr<yggdrasil_decision_forests::utils::FileOutputByteStream>
      stream_;
};

// Exports a proto to disk in binary format.
absl::Status SetBinaryProto(absl::string_view path,
                            const google::protobuf::MessageLite& message, int unused);

// Import a proto from disk in binary format.
absl::Status GetBinaryProto(absl::string_view path,
                            google::protobuf::MessageLite* message, int unused);

// Exports a proto to disk in text format.
absl::Status SetTextProto(absl::string_view path,
                          const google::protobuf::Message& message, int unused);

// Import a proto from disk in text format.
absl::Status GetTextProto(absl::string_view path, google::protobuf::Message* message,
                          int unused);

// Tests if a file exist.
absl::StatusOr<bool> FileExists(absl::string_view path);

// Gets the basename of the path.
//
// Usage example:
//   std::string basename = GetBasename("/path/to/my/file.txt");
//   EXPECT_EQ(basename, "file.txt");
std::string GetBasename(absl::string_view path);

// Renames a file or a directory.
absl::Status Rename(absl::string_view from, absl::string_view to, int options);

// Sets the file as immutable. An immutable file cannot be modified (only
// removed). Some distributed file systems can share immutable files more
// efficiently.
absl::Status SetImmutable(absl::string_view path);

}  // namespace file

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_FILESYSTEM_TENSORFLOW_H_
