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

#include "yggdrasil_decision_forests/utils/bytestream.h"

#include <algorithm>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace utils {

absl::StatusOr<std::string> InputByteStream::ReadAll() {
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

absl::StatusOr<char> InputByteStream::ReadByte() {
  char value;
  ASSIGN_OR_RETURN(const auto has_value, ReadExactly(&value, 1));
  if (!has_value) {
    return absl::OutOfRangeError("Insufficient available bytes");
  }
  return value;
}

absl::StatusOr<int> StringInputByteStream::ReadUpTo(char* buffer,
                                                    int max_read) {
  return stream_.ReadUpTo(buffer, max_read);
}

absl::StatusOr<bool> StringInputByteStream::ReadExactly(char* buffer,
                                                        int num_read) {
  return stream_.ReadExactly(buffer, num_read);
}

absl::StatusOr<int> StringViewInputByteStream::ReadUpTo(char* buffer,
                                                        int max_read) {
  const int num_read = std::min(
      static_cast<int>(content_.size()) - static_cast<int>(current_), max_read);
  if (num_read > 0) {
    std::memcpy(buffer, content_.data() + current_, num_read);
  }
  current_ += num_read;
  return num_read;
}

absl::StatusOr<bool> StringViewInputByteStream::ReadExactly(char* buffer,
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

absl::Status StringOutputByteStream::Write(const absl::string_view chunk) {
  content_.Append(chunk);
  return absl::OkStatus();
}

absl::string_view StringOutputByteStream::ToString() {
  return content_.Flatten();
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
