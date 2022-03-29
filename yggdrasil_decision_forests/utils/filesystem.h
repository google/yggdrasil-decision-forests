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

#ifndef YGGDRASIL_DECISION_FORESTS_FILESYSTEM_H_
#define YGGDRASIL_DECISION_FORESTS_FILESYSTEM_H_

// clang-format off
#if defined YGG_FILESYSTEM_USES_TENSORFLOW
#include "yggdrasil_decision_forests/utils/filesystem_tensorflow.h"
#else
#include "yggdrasil_decision_forests/utils/filesystem_default.h"
#endif
// clang-format on

#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace file {

// Open a file for reading.
yggdrasil_decision_forests::utils::StatusOr<
    std::unique_ptr<FileInputByteStream>>
OpenInputFile(absl::string_view path);

// Open a file for writing.
yggdrasil_decision_forests::utils::StatusOr<
    std::unique_ptr<FileOutputByteStream>>
OpenOutputFile(absl::string_view path);

// Reads the content of a file.
yggdrasil_decision_forests::utils::StatusOr<std::string> GetContent(
    absl::string_view path);

// Sets the content of a file.
absl::Status SetContent(absl::string_view path, absl::string_view content);

// Takes ownership and closes a file at destruction.
template <typename FileStream>
class GenericFileCloser {
 public:
  GenericFileCloser() {}

  explicit GenericFileCloser(std::unique_ptr<FileStream> stream)
      : stream_(std::move(stream)) {}

  ~GenericFileCloser() { CHECK_OK(Close()); }

  // Returns a borrowed pointer to the stream. Ownership is not transferred.
  FileStream* stream() { return stream_.get(); }

  absl::Status reset(std::unique_ptr<FileStream> stream) {
    RETURN_IF_ERROR(Close());
    stream_ = std::move(stream);
    return absl::OkStatus();
  }

  absl::Status Close() {
    if (stream_) {
      // If "Close" fails, it will not be called again in the destructor.
      auto stream = std::move(stream_);
      RETURN_IF_ERROR(stream->Close());
      stream_.reset();
    }
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<FileStream> stream_;
};

using InputFileCloser = GenericFileCloser<FileInputByteStream>;
using OutputFileCloser = GenericFileCloser<FileOutputByteStream>;

}  // namespace file

#endif  // YGGDRASIL_DECISION_FORESTS_FILESYSTEM_H_
