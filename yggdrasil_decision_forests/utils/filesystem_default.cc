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

#include "yggdrasil_decision_forests/utils/filesystem_default.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <ios>
#include <memory>
#include <regex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/utils/filesystem_interface.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

#if __cplusplus > 201402L
namespace fs = std::filesystem;
#else
namespace fs = std::experimental::filesystem;
#endif

// Converts a absl::string_view into an object compatible with std::filesystem.
#ifdef ABSL_USES_STD_STRING_VIEW
#define SV_ABSL_TO_STD(X) X
#else
#define SV_ABSL_TO_STD(X) std::string(X)
#endif

namespace yggdrasil_decision_forests::utils::filesystem {

// GCS implementation; if linked.
std::unique_ptr<FileSystemInterface> gcs_implementation;

void SetGCSImplementation(std::unique_ptr<FileSystemInterface>&& value) {
  gcs_implementation = std::move(value);
}

bool HasGCSImplementation() { return gcs_implementation != nullptr; }

FileSystemInterface& GCSImplementation() {
  if (!gcs_implementation) {
    LOG(FATAL)
        << "TensorFlow filesystem dependency not linked. Make sure to add "
           "yggdrasil_decision_forests/utils:filesystem_tensorflow_impl as a "
           "dependency to your project.";
  }
  return *gcs_implementation;
}

absl::optional<GCSPath> GCSPath::Parse(absl::string_view path) {
  constexpr char kPrefix[] = "gs://";
  constexpr int kPrefixLen = 5;
  if (!absl::StartsWith(path, kPrefix)) {
    return {};
  }
  const auto sep = path.find('/', kPrefixLen);
  if (sep == -1) {
    return {};
  }
  const auto x = GCSPath{
      .bucket = std::string(path.substr(kPrefixLen, sep - kPrefixLen)),
      .object = std::string(path.substr(sep + 1)),
  };
  return x;
}

absl::Status STLFileInputByteStream::Open(absl::string_view path) {
  file_stream_.open(std::string(path), std::ios::binary);
  if (!file_stream_.is_open()) {
    return absl::Status(absl::StatusCode::kUnknown,
                        absl::StrCat("Failed to read open ", path,
                                     " with error:", std::strerror(errno)));
  }
  return absl::OkStatus();
}

absl::StatusOr<int> STLFileInputByteStream::ReadUpTo(char* buffer,
                                                     int max_read) {
  file_stream_.read(buffer, max_read);
  if (file_stream_.bad()) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to read chunk");
  }
  return file_stream_.gcount();
}

absl::StatusOr<bool> STLFileInputByteStream::ReadExactly(char* buffer,
                                                         int num_read) {
  file_stream_.read(buffer, num_read);
  const auto read_count = file_stream_.gcount();
  if (file_stream_.bad() || (read_count > 0 && read_count < num_read)) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to read chunk");
  }
  return read_count > 0 || num_read == 0;
}

absl::Status STLFileInputByteStream::Close() {
  file_stream_.close();
  if (file_stream_.bad()) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to close file");
  }
  return absl::OkStatus();
}

absl::Status STLFileOutputByteStream::Open(absl::string_view path) {
  file_stream_.open(std::string(path), std::ios::binary);
  if (!file_stream_.is_open()) {
    return absl::Status(absl::StatusCode::kUnknown,
                        absl::StrCat("Failed to write open ", path,
                                     " with error:", std::strerror(errno)));
  }
  return absl::OkStatus();
}

absl::Status STLFileOutputByteStream::Write(absl::string_view chunk) {
  file_stream_.write(chunk.data(), chunk.size());
  if (file_stream_.bad()) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to write chunk");
  }
  return absl::OkStatus();
}

absl::Status STLFileOutputByteStream::Close() {
  file_stream_.close();
  if (file_stream_.bad()) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to close file");
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::utils::filesystem

namespace file {

std::string JoinPathList(std::initializer_list<absl::string_view> paths) {
  fs::path all_paths;
  for (const auto& path : paths) {
    all_paths /= SV_ABSL_TO_STD(path);
  }
  return all_paths.string();
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
  // Support for GCS
  if (::yggdrasil_decision_forests::utils::filesystem::HasGCSImplementation()) {
    const auto cloud_path =
        ::yggdrasil_decision_forests::utils::filesystem::GCSPath::Parse(
            pattern);
    if (cloud_path.has_value()) {
      return ::yggdrasil_decision_forests::utils::filesystem::
          GCSImplementation()
              .Match(pattern, results, options);
    }
  }

  try {
    const auto search_dir = fs::path(SV_ABSL_TO_STD(pattern)).parent_path();
    const auto filename = fs::path(SV_ABSL_TO_STD(pattern)).filename().string();
    std::string regexp_filename =
        absl::StrReplaceAll(filename, {{".", "\\."}, {"*", ".*"}, {"?", "."}});
    std::regex regexp_pattern(regexp_filename);
    std::error_code error;

    const fs::directory_iterator path_end;
    for (auto path = fs::directory_iterator(search_dir, error);
         !error && path != path_end; path.increment(error)) {
      if (!fs::is_regular_file(path->path())) {
        continue;
      }
      if (std::regex_match(path->path().filename().string(), regexp_pattern)) {
        results->push_back(path->path().string());
      }
    }
    if (error) {
      return absl::NotFoundError(error.message());
    }

    std::sort(results->begin(), results->end());
    return absl::OkStatus();
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(e.what());
  }
}

absl::Status RecursivelyCreateDir(absl::string_view path, int options) {
  // Support for GCS
  if (::yggdrasil_decision_forests::utils::filesystem::HasGCSImplementation()) {
    const auto cloud_path =
        ::yggdrasil_decision_forests::utils::filesystem::GCSPath::Parse(path);
    if (cloud_path.has_value()) {
      return ::yggdrasil_decision_forests::utils::filesystem::
          GCSImplementation()
              .RecursivelyCreateDir(path, options);
    }
  }

  try {
    fs::create_directories(SV_ABSL_TO_STD(path));
    return absl::OkStatus();
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(e.what());
  }
}

// Delete the directory "path".
absl::Status RecursivelyDelete(absl::string_view path, int options) {
  // Support for GCS
  if (::yggdrasil_decision_forests::utils::filesystem::HasGCSImplementation()) {
    const auto cloud_path =
        ::yggdrasil_decision_forests::utils::filesystem::GCSPath::Parse(path);
    if (cloud_path.has_value()) {
      return ::yggdrasil_decision_forests::utils::filesystem::
          GCSImplementation()
              .RecursivelyDelete(path, options);
    }
  }

  try {
    fs::remove(SV_ABSL_TO_STD(path));
    return absl::OkStatus();
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(e.what());
  }
}

absl::Status FileInputByteStream::Open(absl::string_view path) {
  stream_.reset();

  // Support for GCS
  if (::yggdrasil_decision_forests::utils::filesystem::HasGCSImplementation()) {
    const auto cloud_path =
        ::yggdrasil_decision_forests::utils::filesystem::GCSPath::Parse(path);
    if (cloud_path.has_value()) {
      stream_ =
          ::yggdrasil_decision_forests::utils::filesystem::GCSImplementation()
              .CreateInputByteStream();
    }
  }

  if (!stream_) {
    stream_ = std::make_unique<yggdrasil_decision_forests::utils::filesystem::
                                   STLFileInputByteStream>();
  }
  return stream_->Open(path);
}

absl::StatusOr<int> FileInputByteStream::ReadUpTo(char* buffer, int max_read) {
  return stream_->ReadUpTo(buffer, max_read);
}

absl::StatusOr<bool> FileInputByteStream::ReadExactly(char* buffer,
                                                      int num_read) {
  return stream_->ReadExactly(buffer, num_read);
}

absl::Status FileInputByteStream::Close() { return stream_->Close(); }

absl::Status FileOutputByteStream::Open(absl::string_view path) {
  stream_.reset();

  // Support for GCS
  if (::yggdrasil_decision_forests::utils::filesystem::HasGCSImplementation()) {
    const auto cloud_path =
        ::yggdrasil_decision_forests::utils::filesystem::GCSPath::Parse(path);
    if (cloud_path.has_value()) {
      stream_ =
          ::yggdrasil_decision_forests::utils::filesystem::GCSImplementation()
              .CreateOutputByteStream();
    }
  }

  if (!stream_) {
    stream_ = std::make_unique<yggdrasil_decision_forests::utils::filesystem::
                                   STLFileOutputByteStream>();
  }
  return stream_->Open(path);
}

absl::Status FileOutputByteStream::Write(absl::string_view chunk) {
  return stream_->Write(chunk);
}

absl::Status FileOutputByteStream::Close() { return stream_->Close(); }

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

absl::StatusOr<bool> FileExists(absl::string_view path) {
  // Support for GCS
  if (::yggdrasil_decision_forests::utils::filesystem::HasGCSImplementation()) {
    const auto cloud_path =
        ::yggdrasil_decision_forests::utils::filesystem::GCSPath::Parse(path);
    if (cloud_path.has_value()) {
      return ::yggdrasil_decision_forests::utils::filesystem::
          GCSImplementation()
              .FileExists(path);
    }
  }

  return fs::exists(SV_ABSL_TO_STD(path));
}

absl::Status Rename(absl::string_view from, absl::string_view to, int options) {
  // Support for GCS
  if (::yggdrasil_decision_forests::utils::filesystem::HasGCSImplementation()) {
    const auto from_cloud_path =
        ::yggdrasil_decision_forests::utils::filesystem::GCSPath::Parse(from);
    const auto to_cloud_path =
        ::yggdrasil_decision_forests::utils::filesystem::GCSPath::Parse(to);
    if (from_cloud_path.has_value() != to_cloud_path.has_value()) {
      return absl::InvalidArgumentError(
          "Cannot move object between google cloud storage and local.");
    }
    if (from_cloud_path.has_value()) {
      return ::yggdrasil_decision_forests::utils::filesystem::
          GCSImplementation()
              .Rename(from, to, options);
    }
  }

  try {
    fs::rename(SV_ABSL_TO_STD(from), SV_ABSL_TO_STD(to));
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(e.what());
  }
  return absl::OkStatus();
}

std::string GetBasename(absl::string_view path) {
  try {
    auto filename = fs::path(std::string(path)).filename().string();
    return filename;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error parsing basename of " << path << ": " << e.what();
    return "";
  }
}

absl::Status SetImmutable(absl::string_view path) { return absl::OkStatus(); }

}  // namespace file
