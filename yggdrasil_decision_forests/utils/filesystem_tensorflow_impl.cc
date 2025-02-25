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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <regex>  // NOLINT
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "src/google/protobuf/message.h"
#include "src/google/protobuf/message_lite.h"
#include "src/google/protobuf/text_format.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/path.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"
#include "yggdrasil_decision_forests/utils/filesystem_interface.h"
#include "yggdrasil_decision_forests/utils/filesystem_tensorflow_interface.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

// Wrappers to shield tensorflow macros from yggdrasil macros.
namespace yggdrasil_decision_forests::utils::filesystem::tf_impl {

class RandomAccessFileWrapper {
 public:
  explicit RandomAccessFileWrapper(::tensorflow::RandomAccessFile* item)
      : item_(item) {}
  ~RandomAccessFileWrapper();

  ::tensorflow::RandomAccessFile* item() { return item_; }

 private:
  ::tensorflow::RandomAccessFile* item_;
};

class WritableFileWrapper {
 public:
  explicit WritableFileWrapper(::tensorflow::WritableFile* item)
      : item_(item) {}
  ~WritableFileWrapper();

  ::tensorflow::WritableFile* item() { return item_; }

 private:
  ::tensorflow::WritableFile* item_;
};

RandomAccessFileWrapper::~RandomAccessFileWrapper() {
  delete item_;
  item_ = nullptr;
}

WritableFileWrapper::~WritableFileWrapper() {
  delete item_;
  item_ = nullptr;
}

class FileInputByteStream
    : public yggdrasil_decision_forests::utils::FileInputByteStream {
 public:
  absl::Status Open(absl::string_view path) override;
  absl::StatusOr<int> ReadUpTo(char* buffer, int max_read) override;
  absl::StatusOr<bool> ReadExactly(char* buffer, int num_read) override;
  absl::Status Close() override;

 private:
  std::unique_ptr<RandomAccessFileWrapper> file_;
  uint64_t offset_ = 0;
  std::string scrath_;
};

class FileOutputByteStream
    : public yggdrasil_decision_forests::utils::FileOutputByteStream {
 public:
  absl::Status Open(absl::string_view path) override;
  absl::Status Write(absl::string_view chunk) override;
  absl::Status Close() override;

 private:
  std::unique_ptr<WritableFileWrapper> file_;
};

absl::Status FileInputByteStream::Open(absl::string_view path) {
  std::unique_ptr<::tensorflow::RandomAccessFile> file;
  RETURN_IF_ERROR(tensorflow::Env::Default()->NewRandomAccessFile(
      std::string(path), &file));
  file_ = std::make_unique<RandomAccessFileWrapper>(file.release());
  offset_ = 0;
  return absl::OkStatus();
}

absl::StatusOr<int> FileInputByteStream::ReadUpTo(char* buffer, int max_read) {
  absl::string_view result;
  if (max_read > scrath_.size()) {
    scrath_.resize(max_read);
  }
  const auto status =
      file_->item()->Read(offset_, max_read, &result, &scrath_[0]);
  if (!status.ok() && status.code() != absl::StatusCode::kOutOfRange) {
    return status;
  }
  offset_ += result.size();
  std::memcpy(buffer, result.data(), result.size());
  return result.size();
}

absl::StatusOr<bool> FileInputByteStream::ReadExactly(char* buffer,
                                                      int num_read) {
  absl::string_view result;
  if (num_read > scrath_.size()) {
    scrath_.resize(num_read);
  }
  const auto status =
      file_->item()->Read(offset_, num_read, &result, &scrath_[0]);
  if (!status.ok()) {
    if (status.code() == absl::StatusCode::kOutOfRange && result.empty() &&
        num_read > 0) {
      return false;
    }
    return status;
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
  std::unique_ptr<::tensorflow::WritableFile> file;
  RETURN_IF_ERROR(
      tensorflow::Env::Default()->NewWritableFile(std::string(path), &file));
  file_ = std::make_unique<WritableFileWrapper>(file.release());
  return absl::OkStatus();
}

absl::Status FileOutputByteStream::Write(absl::string_view chunk) {
  return file_->item()->Append(chunk);
}

absl::Status FileOutputByteStream::Close() { return file_->item()->Close(); }

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

class FileSystemImplementation : public FileSystemInterface {
 public:
  ~FileSystemImplementation() {}

  std::string JoinPathList(
      std::initializer_list<absl::string_view> paths) override {
    return ::tensorflow::io::internal::JoinPathImpl(paths);
  }

  bool GenerateShardedFilenames(absl::string_view spec,
                                std::vector<std::string>* names) override {
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

  absl::Status Match(absl::string_view pattern,
                     std::vector<std::string>* results,
                     const int options) override {
    RETURN_IF_ERROR(tensorflow::Env::Default()->GetMatchingPaths(
        std::string(pattern), results));
    std::sort(results->begin(), results->end());
    return absl::OkStatus();
  }

  absl::Status RecursivelyCreateDir(absl::string_view path,
                                    int options) override {
    return tensorflow::Env::Default()->RecursivelyCreateDir(std::string(path));
  }

  absl::Status RecursivelyDelete(absl::string_view path, int options) override {
    int64_t ignore_1, ignore_2;
    return tensorflow::Env::Default()->DeleteRecursively(std::string(path),
                                                         &ignore_1, &ignore_2);
  }

  absl::StatusOr<bool> FileExists(absl::string_view path) override {
    const auto exist_status =
        tensorflow::Env::Default()->FileExists(std::string(path));
    if (exist_status.ok()) {
      return true;
    }
    if (exist_status.code() == absl::StatusCode::kNotFound) {
      return false;
    }
    return exist_status;
  }

  absl::Status Rename(absl::string_view from, absl::string_view to,
                      int options) override {
    return tensorflow::Env::Default()->RenameFile(std::string(from),
                                                  std::string(to));
  }

  std::string GetBasename(absl::string_view path) override {
    return std::string(tensorflow::io::Basename(path));
  }

  virtual std::unique_ptr<
      yggdrasil_decision_forests::utils::FileInputByteStream>
  CreateInputByteStream() override {
    return std::make_unique<FileInputByteStream>();
  }

  virtual std::unique_ptr<
      yggdrasil_decision_forests::utils::FileOutputByteStream>
  CreateOutputByteStream() override {
    return std::make_unique<FileOutputByteStream>();
  }
};

int init() {
  SetInterface(std::make_unique<FileSystemImplementation>());
  return 0;
}

static const int a = init();

}  // namespace yggdrasil_decision_forests::utils::filesystem::tf_impl
