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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_FILESYSTEM_INTERFACE_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_FILESYSTEM_INTERFACE_H_

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"

namespace yggdrasil_decision_forests::utils::filesystem {

class FileSystemInterface {
 public:
  virtual ~FileSystemInterface() = default;

  virtual std::string JoinPathList(
      std::initializer_list<absl::string_view> paths) = 0;

  virtual bool GenerateShardedFilenames(absl::string_view spec,
                                        std::vector<std::string>* names) = 0;

  virtual absl::Status Match(absl::string_view pattern,
                             std::vector<std::string>* results,
                             int options) = 0;

  virtual absl::Status RecursivelyCreateDir(absl::string_view path,
                                            int options) = 0;

  virtual absl::Status RecursivelyDelete(absl::string_view path,
                                         int options) = 0;

  virtual absl::StatusOr<bool> FileExists(absl::string_view path) = 0;

  virtual absl::Status Rename(absl::string_view from, absl::string_view to,
                              int options) = 0;

  virtual std::string GetBasename(absl::string_view path) = 0;

  virtual std::unique_ptr<FileInputByteStream> CreateInputByteStream() = 0;

  virtual std::unique_ptr<FileOutputByteStream> CreateOutputByteStream() = 0;
};
}  // namespace yggdrasil_decision_forests::utils::filesystem

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_FILESYSTEM_TENSORFLOW_INTERFACE_H_
