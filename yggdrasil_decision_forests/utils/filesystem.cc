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

#include "yggdrasil_decision_forests/utils/filesystem.h"

#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace file {

yggdrasil_decision_forests::utils::StatusOr<
    std::unique_ptr<FileInputByteStream>>
OpenInputFile(absl::string_view path) {
  auto reader = absl::make_unique<FileInputByteStream>();
  RETURN_IF_ERROR(reader->Open(path));
  return std::move(reader);
}

yggdrasil_decision_forests::utils::StatusOr<
    std::unique_ptr<FileOutputByteStream>>
OpenOutputFile(absl::string_view path) {
  auto writer = absl::make_unique<FileOutputByteStream>();
  RETURN_IF_ERROR(writer->Open(path));
  return std::move(writer);
}

yggdrasil_decision_forests::utils::StatusOr<std::string> GetContent(
    absl::string_view path) {
  ASSIGN_OR_RETURN(auto file_handle, OpenInputFile(path));
  InputFileCloser closer(std::move(file_handle));
  ASSIGN_OR_RETURN(auto content, closer.stream()->ReadAll());
  RETURN_IF_ERROR(closer.Close());
  return content;
}

absl::Status SetContent(absl::string_view path, absl::string_view content) {
  ASSIGN_OR_RETURN(auto file_handle, OpenOutputFile(path));
  OutputFileCloser closer(std::move(file_handle));
  RETURN_IF_ERROR(closer.stream()->Write(content));
  return closer.Close();
}

}  // namespace file
