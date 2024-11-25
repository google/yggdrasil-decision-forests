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

#include <cstddef>
#include <optional>
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
template <typename T>
absl::StatusOr<std::string> SerializeTextProto(const T& message,
                                               bool single_line_mode = false) {
#ifdef YGG_PROTOBUF_LITE
  return absl::UnimplementedError(
      "YDF has been compiled with YGG_PROTOBUF_LITE. Cannot serialize proto "
      "message.");
#else
  std::string serialized_message;
  google::protobuf::TextFormat::Printer printer;
  if (single_line_mode) {
    printer.SetSingleLineMode(true);
  }
  if (!printer.PrintToString(message, &serialized_message)) {
    return absl::InvalidArgumentError("Cannot serialize proto message.");
  }
  if (single_line_mode && !serialized_message.empty() &&
      serialized_message.back() == ' ') {
    serialized_message.pop_back();
  }
  return serialized_message;
#endif  // YGG_PROTOBUF_LITE
}

// Returns the approximate size of a proto in bytes. If the proto size cannot be
// computed (e.g., if compiled with ProtoLite, returns {}).
template <typename T>
std::optional<std::size_t> ProtoSizeInBytes(const T& message) {
#ifdef YGG_PROTOBUF_LITE
  return {};
#else
  return message.SpaceUsedLong();
#endif
}

// Tells if it is possible to compute the size of a proto i.e. will
// ProtoSizeInBytes return a value.
inline bool ProtoSizeInBytesIsAvailable() {
#ifdef YGG_PROTOBUF_LITE
  return false;
#else
  return true;
#endif
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_PROTOBUF_H_
