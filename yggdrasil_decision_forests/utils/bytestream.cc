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

#include "yggdrasil_decision_forests/utils/bytestream.h"

#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace utils {

utils::StatusOr<std::string> InputByteStream::ReadAll() {
  absl::Cord result;
  char buffer[1024];
  while (true) {
    ASSIGN_OR_RETURN(const int read_bytes, ReadUpTo(buffer, sizeof(buffer)));
    if (read_bytes == 0) {
      break;
    }
    result.Append(absl::string_view(buffer, read_bytes));
  }
  return std::string(result);
}

utils::StatusOr<int> StringInputByteStream::ReadUpTo(char* buffer,
                                                     int max_read) {
  const int num_read =
      std::min(static_cast<int>(content_.size()) - current_, max_read);
  if (num_read > 0) {
    std::memcpy(buffer, content_.data() + current_, num_read);
  }
  current_ += num_read;
  return num_read;
}

utils::StatusOr<bool> StringInputByteStream::ReadExactly(char* buffer,
                                                         int num_read) {
  if (current_ == content_.size()) {
    return false;
  }
  if (current_ + num_read > content_.size()) {
    return absl::OutOfRangeError("Insufficient available bytes");
  }
  if (num_read > 0) {
    std::memcpy(buffer, content_.data() + current_, num_read);
  }
  current_ += num_read;
  return true;
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
