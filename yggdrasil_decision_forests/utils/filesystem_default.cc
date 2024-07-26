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

#if __cplusplus > 201402L
#include <filesystem>
#else
#include <experimental/filesystem>
#endif
#include <ios>
#include <regex>  // NOLINT
#include <string>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

#ifdef YGG_FILESYSTEM_USES_DEFAULT_WITH_GCS
#include "google/cloud/storage/client.h"
#endif

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

// Same as "RETURN_IF_ERROR", but for "expr" returning a Google Cloud Storage
// status.
#ifndef GCS_RETURN_IF_ERROR
#define GCS_RETURN_IF_ERROR(expr)                           \
  {                                                         \
    auto _status = (expr);                                  \
    if (ABSL_PREDICT_FALSE(!_status.ok())) {                \
      return absl::InvalidArgumentError(_status.message()); \
    }                                                       \
  }
#endif

namespace file {

namespace ygg = ::yggdrasil_decision_forests;

#ifdef YGG_FILESYSTEM_USES_DEFAULT_WITH_GCS
::google::cloud::storage::Client GetGCSClient() {
  return ::google::cloud::storage::Client();
}
#else
absl::Status ErrorGCSNotLinked() {
  return absl::InvalidArgumentError(
      "Cannot use Google Cloud Storage paths (i.e. gs://<bucket>/<object>) "
      "because YDF was compiled without GCS support.");
}
#endif

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
    YDF_LOG(WARNING) << "Non defined shard count not supported in " << spec;
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

absl::Status MatchLocal(absl::string_view pattern,
                        std::vector<std::string>* results, const int options) {
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
      return absl::InvalidArgumentError(error.message());
    }

    std::sort(results->begin(), results->end());
    return absl::OkStatus();
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(e.what());
  }
}

#ifdef YGG_FILESYSTEM_USES_DEFAULT_WITH_GCS
absl::Status MatchGCS(const GCSPath& path, std::vector<std::string>* results) {
  auto client = GetGCSClient();

  // The prefix is an efficient filter applies on the candidate server side
  // (unlike the regex which is applied client side).
  const auto end_of_prefix = path.object.find_first_of("?*[]");
  std::string prefix;
  if (end_of_prefix != -1) {
    prefix = path.object.substr(0, end_of_prefix);
  }

  auto candidates =
      client.ListObjects(path.bucket, google::cloud::storage::Prefix(prefix));

  std::string regexp_filename =
      absl::StrReplaceAll(path.object, {{".", "\\."}, {"*", ".*"}, {"?", "."}});
  std::regex regexp_pattern(regexp_filename);

  for (const auto& candidate : candidates) {
    GCS_RETURN_IF_ERROR(candidate.status());
    if (!std::regex_match(candidate->name(), regexp_pattern)) {
      continue;
    }
    results->push_back(
        absl::StrCat("gs://", candidate->bucket(), "/", candidate->name()));
  }
  std::sort(results->begin(), results->end());
  return absl::OkStatus();
}
#endif

absl::Status Match(absl::string_view pattern, std::vector<std::string>* results,
                   const int options) {
  const auto cloud_path = GCSPath::Parse(pattern);
  if (cloud_path.has_value()) {
#ifndef YGG_FILESYSTEM_USES_DEFAULT_WITH_GCS
    return ErrorGCSNotLinked();
#else
    return MatchGCS(cloud_path.value(), results);
#endif
  } else {
    return MatchLocal(pattern, results, options);
  }
}

absl::Status RecursivelyCreateDir(absl::string_view path, int options) {
  const auto cloud_path = GCSPath::Parse(path);
  if (cloud_path.has_value()) {
#ifndef YGG_FILESYSTEM_USES_DEFAULT_WITH_GCS
    return ErrorGCSNotLinked();
#else
// Nothing to be done.
#endif
  } else {
    try {
      fs::create_directories(SV_ABSL_TO_STD(path));
    } catch (const std::exception& e) {
      return absl::InvalidArgumentError(e.what());
    }
  }
  return absl::OkStatus();
}

// Delete the directory "path".
absl::Status RecursivelyDelete(absl::string_view path, int options) {
  const auto cloud_path = GCSPath::Parse(path);
  if (cloud_path.has_value()) {
#ifndef YGG_FILESYSTEM_USES_DEFAULT_WITH_GCS
    return ErrorGCSNotLinked();
#else
    GCS_RETURN_IF_ERROR(GetGCSClient().DeleteObject(cloud_path.value().bucket,
                                                    cloud_path.value().object));
#endif

  } else {
    try {
      fs::remove(SV_ABSL_TO_STD(path));
    } catch (const std::exception& e) {
      return absl::InvalidArgumentError(e.what());
    }
  }
  return absl::OkStatus();
}

absl::Status FileInputByteStream::Open(absl::string_view path) {
  const auto cloud_path = GCSPath::Parse(path);
  if (cloud_path.has_value()) {
#ifndef YGG_FILESYSTEM_USES_DEFAULT_WITH_GCS
    return ErrorGCSNotLinked();
#else
    gcs_stream_ = GetGCSClient().ReadObject(cloud_path.value().bucket,
                                            cloud_path.value().object);
    if (!gcs_stream_.status().ok()) {
      return absl::Status(absl::StatusCode::kUnknown,
                          absl::StrCat("Failed to open ", path, " with error ",
                                       gcs_stream_.status().message()));
    }
    stream_ = &gcs_stream_;
    backend_ = FilesystemBackend::kGCS;
#endif
  } else {
    file_stream_.open(std::string(path), std::ios::binary);
    stream_ = &file_stream_;
    backend_ = FilesystemBackend::kLocal;
  }
  if (stream_->bad()) {
    return absl::Status(absl::StatusCode::kUnknown,
                        absl::StrCat("Failed to open ", path));
  }
  return absl::OkStatus();
}

absl::StatusOr<int> FileInputByteStream::ReadUpTo(char* buffer, int max_read) {
  stream_->read(buffer, max_read);
  if (stream_->bad()) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to read chunk");
  }
  return stream_->gcount();
}

absl::StatusOr<bool> FileInputByteStream::ReadExactly(char* buffer,
                                                      int num_read) {
  stream_->read(buffer, num_read);
  const auto read_count = stream_->gcount();
  if (stream_->bad() || (read_count > 0 && read_count < num_read)) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to read chunk");
  }
  return read_count > 0 || num_read == 0;
}

absl::Status FileInputByteStream::Close() {
  switch (backend_) {
    case FilesystemBackend::kLocal:
      file_stream_.close();
      break;
#ifdef YGG_FILESYSTEM_USES_DEFAULT_WITH_GCS
    case FilesystemBackend::kGCS:
      gcs_stream_.Close();
      break;
#endif
  }
  if (stream_->bad()) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to close file");
  }
  return absl::OkStatus();
}

absl::Status FileOutputByteStream::Open(absl::string_view path) {
  const auto cloud_path = GCSPath::Parse(path);
  if (cloud_path.has_value()) {
#ifndef YGG_FILESYSTEM_USES_DEFAULT_WITH_GCS
    return ErrorGCSNotLinked();
#else
    gcs_stream_ = GetGCSClient().WriteObject(cloud_path.value().bucket,
                                             cloud_path.value().object);
    if (!gcs_stream_.last_status().ok()) {
      return absl::Status(absl::StatusCode::kUnknown,
                          absl::StrCat("Failed to open ", path, " with error ",
                                       gcs_stream_.last_status().message()));
    }
    stream_ = &gcs_stream_;
    backend_ = FilesystemBackend::kGCS;
#endif
  } else {
    file_stream_.open(std::string(path), std::ios::binary);
    stream_ = &file_stream_;
    backend_ = FilesystemBackend::kLocal;
  }
  if (stream_->bad()) {
    return absl::Status(absl::StatusCode::kUnknown,
                        absl::StrCat("Failed to open ", path));
  }
  return absl::OkStatus();
}

absl::Status FileOutputByteStream::Write(absl::string_view chunk) {
  stream_->write(chunk.data(), chunk.size());
  if (stream_->bad()) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to write chunk");
  }
  return absl::OkStatus();
}

absl::Status FileOutputByteStream::Close() {
  switch (backend_) {
    case FilesystemBackend::kLocal:
      file_stream_.close();
      break;
#ifdef YGG_FILESYSTEM_USES_DEFAULT_WITH_GCS
    case FilesystemBackend::kGCS:
      gcs_stream_.Close();
      break;
#endif
  }
  if (stream_->bad()) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to close file");
  }
  return absl::OkStatus();
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

absl::StatusOr<bool> FileExists(absl::string_view path) {
  const auto cloud_path = GCSPath::Parse(path);
  if (cloud_path.has_value()) {
#ifndef YGG_FILESYSTEM_USES_DEFAULT_WITH_GCS
    return ErrorGCSNotLinked();
#else
    return GetGCSClient()
        .GetObjectMetadata(cloud_path.value().bucket, cloud_path.value().object)
        .ok();
#endif
  } else {
    return fs::exists(SV_ABSL_TO_STD(path));
  }
}

absl::Status Rename(absl::string_view from, absl::string_view to, int options) {
  const auto from_cloud_path = GCSPath::Parse(from);
  const auto to_cloud_path = GCSPath::Parse(to);
  if (from_cloud_path.has_value() != to_cloud_path.has_value()) {
    return absl::InvalidArgumentError(
        "Cannot move object between google cloud storage and local.");
  }
  if (from_cloud_path.has_value()) {
#ifndef YGG_FILESYSTEM_USES_DEFAULT_WITH_GCS
    return ErrorGCSNotLinked();
#else
    auto client = GetGCSClient();
    GCS_RETURN_IF_ERROR(
        client
            .RewriteObjectBlocking(
                from_cloud_path.value().bucket, from_cloud_path.value().object,
                to_cloud_path.value().bucket, to_cloud_path.value().object)
            .status());
    GCS_RETURN_IF_ERROR(client.DeleteObject(from_cloud_path.value().bucket,
                                            from_cloud_path.value().object));
#endif
  } else {
    try {
      fs::rename(SV_ABSL_TO_STD(from), SV_ABSL_TO_STD(to));
    } catch (const std::exception& e) {
      return absl::InvalidArgumentError(e.what());
    }
  }
  return absl::OkStatus();
}

std::string GetBasename(absl::string_view path) {
  try {
    auto filename = fs::path(std::string(path)).filename().string();
#if __cplusplus == 201402L
    // The experimental C++14 filesystem reports a . if the filename is empty.
    if (filename == ".") {
      return "";
    }
#endif
    return filename;
  } catch (const std::exception& e) {
    YDF_LOG(ERROR) << "Error parsing basename of " << path << ": " << e.what();
    return "";
  }
}

absl::Status SetImmutable(absl::string_view path) { return absl::OkStatus(); }

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

}  // namespace file
