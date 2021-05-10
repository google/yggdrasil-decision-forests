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

// Interface around an input or output stream of bytes.
//
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_BYTESTREAM_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_BYTESTREAM_H_

#include <string>

#include "absl/base/macros.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Input stream of bytes.
class InputByteStream {
 public:
  virtual ~InputByteStream() = default;

  // Reads up to "max_read" bytes of data. Returns the number of read bytes.
  // Returns 0 iif. the stream is over.
  virtual utils::StatusOr<int> ReadUpTo(char* buffer, int max_read) = 0;

  // Reads exactly to "num_read" bytes of data. Return true if the bytes are
  // read. Return false if the stream was already over. Fails if some but not
  // all bytes where read.
  virtual utils::StatusOr<bool> ReadExactly(char* buffer, int num_read) = 0;

  // Reads and returns the entire content of the stream.
  utils::StatusOr<std::string> ReadAll();
};

// Wraps a InputByteStream around a std::string.
class StringInputByteStream : public InputByteStream {
 public:
  StringInputByteStream(std::string content) : content_(std::move(content)) {}

  utils::StatusOr<int> ReadUpTo(char* buffer, int max_read) override;

  utils::StatusOr<bool> ReadExactly(char* buffer, int num_read) override;

 private:
  // String content.
  std::string content_;

  // Next character to read in "content_".
  int current_ = 0;
};

// Output stream of bytes.
class OutputByteStream {
 public:
  virtual ~OutputByteStream() = default;

  // Writes a chunk of bytes.
  virtual absl::Status Write(absl::string_view chunk) = 0;
};

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_BYTESTREAM_H_
