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

#include "yggdrasil_decision_forests/utils/filesystem_tensorflow.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace file {

std::string JoinPathList(std::initializer_list<absl::string_view> paths) {
  return Interface().JoinPathList(paths);
}

bool GenerateShardedFilenames(absl::string_view spec,
                              std::vector<std::string>* names) {
  return Interface().GenerateShardedFilenames(spec, names);
}

absl::Status Match(absl::string_view pattern, std::vector<std::string>* results,
                   int options) {
  return Interface().Match(pattern, results, options);
}

absl::Status RecursivelyCreateDir(absl::string_view path, int options) {
  return Interface().RecursivelyCreateDir(path, options);
}

absl::Status RecursivelyDelete(absl::string_view path, int options) {
  return Interface().RecursivelyDelete(path, options);
}

absl::StatusOr<bool> FileExists(absl::string_view path) {
  return Interface().FileExists(path);
}

absl::Status Rename(absl::string_view from, absl::string_view to, int options) {
  return Interface().Rename(from, to, options);
}

std::string GetBasename(absl::string_view path) {
  return Interface().GetBasename(path);
}

absl::Status SetImmutable(absl::string_view path) { return absl::OkStatus(); }

absl::Status SetBinaryProto(absl::string_view path,
                            const google::protobuf::MessageLite& message, int unused) {
  auto writer = std::make_unique<FileOutputByteStream>();
  RETURN_IF_ERROR(writer->Open(path));
  auto write_status = writer->Write(message.SerializeAsString());
  RETURN_IF_ERROR(writer->Close());
  return write_status;
}

absl::Status GetBinaryProto(absl::string_view path,
                            google::protobuf::MessageLite* message, int unused) {
  auto reader = std::make_unique<FileInputByteStream>();
  RETURN_IF_ERROR(reader->Open(path));
  auto content_or = reader->ReadAll();
  RETURN_IF_ERROR(reader->Close());
  RETURN_IF_ERROR(content_or.status());

  if (!message->ParseFromString(content_or.value())) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot parse binary proto from ", path));
  }
  return absl::OkStatus();
}

absl::Status SetTextProto(absl::string_view path,
                          const google::protobuf::Message& message, int unused) {
  std::string content;
  google::protobuf::TextFormat::PrintToString(message, &content);
  auto writer = std::make_unique<FileOutputByteStream>();
  RETURN_IF_ERROR(writer->Open(path));
  auto write_status = writer->Write(content);
  RETURN_IF_ERROR(writer->Close());
  return write_status;
}

absl::Status GetTextProto(absl::string_view path, google::protobuf::Message* message,
                          int unused) {
  auto reader = std::make_unique<FileInputByteStream>();
  RETURN_IF_ERROR(reader->Open(path));
  auto content_or = reader->ReadAll();
  RETURN_IF_ERROR(reader->Close());
  RETURN_IF_ERROR(content_or.status());
  if (!google::protobuf::TextFormat::ParseFromString(content_or.value(), message)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot parse text proto from ", path));
  }
  return absl::OkStatus();
}

}  // namespace file
