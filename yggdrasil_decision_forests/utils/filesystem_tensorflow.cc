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

#include "yggdrasil_decision_forests/utils/filesystem_tensorflow.h"

#include <regex>  // NOLINT

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/tensorflow.h"

namespace tensorflow {

RandomAccessFileWrapper::~RandomAccessFileWrapper() {
  delete item_;
  item_ = nullptr;
}

WritableFileWrapper::~WritableFileWrapper() {
  delete item_;
  item_ = nullptr;
}

}  // namespace tensorflow

namespace file {

using yggdrasil_decision_forests::utils::ToUtilStatus;

std::string JoinPathList(std::initializer_list<absl::string_view> paths) {
  return ::tensorflow::io::internal::JoinPathImpl(paths);
}

bool GenerateShardedFilenames(absl::string_view spec,
                              std::vector<std::string>* names) {
  std::regex num_shard_pattern(R"((.*)\@(\*|[0-9]+)(?:(\..+))?)");
  std::smatch match;
  std::string str_spec(spec);
  if (!std::regex_match(str_spec, match, num_shard_pattern)) {
    return false;
  }
  if (match.size() != 4) {
    return false;
  }
  const auto prefix = match[1].str();
  const auto count = match[2].str();
  const auto suffix = match[3].str();

  int int_count;
  if (count == "*") {
    LOG(WARNING) << "Non defined shard count not supported in " << spec;
    return false;
  } else if (absl::SimpleAtoi(count, &int_count)) {
  } else {
    return false;
  }

  for (int idx = 0; idx < int_count; idx++) {
    names->push_back(
        absl::StrFormat("%s-%05d-of-%05d%s", prefix, idx, int_count, suffix));
  }
  return true;
}

absl::Status Match(absl::string_view pattern, std::vector<std::string>* results,
                   const int options) {
  RETURN_IF_ERROR(yggdrasil_decision_forests::utils::ToUtilStatus(
      tensorflow::Env::Default()->GetMatchingPaths(std::string(pattern),
                                                   results)));
  std::sort(results->begin(), results->end());
  return absl::OkStatus();
}

absl::Status RecursivelyCreateDir(absl::string_view path, int options) {
  return yggdrasil_decision_forests::utils::ToUtilStatus(
      tensorflow::Env::Default()->RecursivelyCreateDir(std::string(path)));
}

absl::Status RecursivelyDelete(absl::string_view path, int options) {
  int64_t ignore_1, ignore_2;
  return yggdrasil_decision_forests::utils::ToUtilStatus(
      tensorflow::Env::Default()->DeleteRecursively(std::string(path),
                                                    &ignore_1, &ignore_2));
}

absl::Status FileInputByteStream::Open(absl::string_view path) {
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  RETURN_IF_ERROR(ToUtilStatus(tensorflow::Env::Default()->NewRandomAccessFile(
      std::string(path), &file)));
  file_ =
      absl::make_unique<tensorflow::RandomAccessFileWrapper>(file.release());
  offset_ = 0;
  return absl::OkStatus();
}

yggdrasil_decision_forests::utils::StatusOr<int> FileInputByteStream::ReadUpTo(
    char* buffer, int max_read) {
  absl::string_view result;
  if (max_read > scrath_.size()) {
    scrath_.resize(max_read);
  }
  const auto tf_status =
      file_->item()->Read(offset_, max_read, &result, &scrath_[0]);
  if (!tf_status.ok() && tf_status.code() != tensorflow::error::OUT_OF_RANGE) {
    return ToUtilStatus(tf_status);
  }
  offset_ += result.size();
  std::memcpy(buffer, result.data(), result.size());
  return result.size();
}

yggdrasil_decision_forests::utils::StatusOr<bool>
FileInputByteStream::ReadExactly(char* buffer, int num_read) {
  absl::string_view result;
  if (num_read > scrath_.size()) {
    scrath_.resize(num_read);
  }
  const auto tf_status =
      file_->item()->Read(offset_, num_read, &result, &scrath_[0]);
  if (!tf_status.ok()) {
    if (tf_status.code() == tensorflow::error::OUT_OF_RANGE && result.empty() &&
        num_read > 0) {
      return false;
    }
    return ToUtilStatus(tf_status);
  }
  offset_ += result.size();
  std::memcpy(buffer, result.data(), result.size());
  return true;
}

absl::Status FileInputByteStream::Close() {
  file_.reset();
  return absl::OkStatus();
}

absl::Status FileOutputByteStream::Open(absl::string_view path) {
  std::unique_ptr<tensorflow::WritableFile> file;
  RETURN_IF_ERROR(ToUtilStatus(
      tensorflow::Env::Default()->NewWritableFile(std::string(path), &file)));
  file_ = absl::make_unique<tensorflow::WritableFileWrapper>(file.release());
  return absl::OkStatus();
}

absl::Status FileOutputByteStream::Write(absl::string_view chunk) {
  return ToUtilStatus(file_->item()->Append(chunk));
}

absl::Status FileOutputByteStream::Close() {
  return ToUtilStatus(file_->item()->Close());
}

absl::Status SetBinaryProto(absl::string_view path,
                            const google::protobuf::MessageLite& message, int unused) {
  auto writer = absl::make_unique<FileOutputByteStream>();
  RETURN_IF_ERROR(writer->Open(path));
  auto write_status = writer->Write(message.SerializeAsString());
  RETURN_IF_ERROR(writer->Close());
  return write_status;
}

absl::Status GetBinaryProto(absl::string_view path,
                            google::protobuf::MessageLite* message, int unused) {
  auto reader = absl::make_unique<FileInputByteStream>();
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
  auto writer = absl::make_unique<FileOutputByteStream>();
  RETURN_IF_ERROR(writer->Open(path));
  auto write_status = writer->Write(content);
  RETURN_IF_ERROR(writer->Close());
  return write_status;
}

absl::Status GetTextProto(absl::string_view path, google::protobuf::Message* message,
                          int unused) {
  auto reader = absl::make_unique<FileInputByteStream>();
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

yggdrasil_decision_forests::utils::StatusOr<bool> FileExists(
    absl::string_view path) {
  const auto exist_status =
      tensorflow::Env::Default()->FileExists(std::string(path));
  if (exist_status.ok()) {
    return true;
  }
  if (exist_status.code() == tensorflow::error::NOT_FOUND) {
    return false;
  }
  return ToUtilStatus(exist_status);
}

absl::Status Rename(absl::string_view from, absl::string_view to, int options) {
  return ToUtilStatus(tensorflow::Env::Default()->RenameFile(std::string(from),
                                                             std::string(to)));
}

std::string GetBasename(absl::string_view path) {
  return std::string(tensorflow::io::Basename(path));
}

}  // namespace file
