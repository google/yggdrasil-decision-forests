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

// Utilities for the manipulation of Protobufs.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_PROTOBUF_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_PROTOBUF_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "src/google/protobuf/message.h"
#include "src/google/protobuf/text_format.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Deserializes a proto from its text representation.
template <typename T>
absl::StatusOr<T> ParseTextProto(absl::string_view raw) {
  T message;
  if (!google::protobuf::TextFormat::ParseFromString(std::string(raw), &message)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot parse protobuf ", typeid(T).name(), " from text"));
  }
  return message;
}

// Deserializes a proto from its binary representation.
template <typename T>
absl::StatusOr<T> ParseBinaryProto(absl::string_view raw) {
  T message;
  if (!message.ParseFromString(std::string(raw))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot parse protobuf ", typeid(T).name(), " from binary text"));
  }
  return message;
}

// Reduces the size to new_size.
template <typename T>
inline void Truncate(google::protobuf::RepeatedPtrField<T>* array, int new_size) {
  const int size = array->size();
  DCHECK_GE(size, new_size);
  array->DeleteSubrange(new_size, size - new_size);
}

// Interface for the sequential reading of protos.
template <typename T>
class ProtoReaderInterface {
 public:
  // Try to retrieve the next available value. If no more value are
  // available, returns false.
  virtual absl::StatusOr<bool> Next(T* value) = 0;
};

// Interface for the sequential writing of protos.
template <typename T>
class ProtoWriterInterface {
 public:
  // Write a new record.
  virtual absl::Status Write(const T& value) = 0;
};

// This method should be used instead of "message.DebugString()" whenever
// the full message (unredacted) is to be serialized in a way that can be
// deserialized with ParseTextProto(). single_line_mode controls whether the
// output is in single-line or multi-line (default).
absl::StatusOr<std::string> SerializeTextProto(const google::protobuf::Message& message,
                                               bool single_line_mode = false);

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_PROTOBUF_H_
