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
#include <cerrno>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <regex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "google/cloud/storage/client.h"
#include "yggdrasil_decision_forests/utils/bytestream.h"
#include "yggdrasil_decision_forests/utils/filesystem_default.h"
#include "yggdrasil_decision_forests/utils/filesystem_interface.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

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

namespace yggdrasil_decision_forests::utils::filesystem::gcs {
namespace {
::google::cloud::storage::Client GetGCSClient() {
  return ::google::cloud::storage::Client();
}

absl::StatusOr<GCSPath> GetGCSPath(const absl::string_view path) {
  const auto cloud_path = GCSPath::Parse(path);
  if (!cloud_path.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The path \"", path, "\" does not look like a Cloud Path."));
  }
  return std::move(*cloud_path);
}

}  // namespace

class FileInputByteStream
    : public yggdrasil_decision_forests::utils::FileInputByteStream {
 public:
  absl::Status Open(absl::string_view path) override;
  absl::StatusOr<int> ReadUpTo(char* buffer, int max_read) override;
  absl::StatusOr<bool> ReadExactly(char* buffer, int num_read) override;
  absl::Status Close() override;

 private:
  ::google::cloud::storage::ObjectReadStream gcs_stream_;
};

class FileOutputByteStream
    : public yggdrasil_decision_forests::utils::FileOutputByteStream {
 public:
  absl::Status Open(absl::string_view path) override;
  absl::Status Write(absl::string_view chunk) override;
  absl::Status Close() override;

 private:
  ::google::cloud::storage::ObjectWriteStream gcs_stream_;
};

absl::Status FileInputByteStream::Open(absl::string_view path) {
  ASSIGN_OR_RETURN(const auto cloud_path, GetGCSPath(path));
  gcs_stream_ = GetGCSClient().ReadObject(cloud_path.bucket, cloud_path.object);
  if (!gcs_stream_.status().ok()) {
    return absl::Status(
        absl::StatusCode::kUnknown,
        absl::StrCat("Failed to gcs read open ", path, " with error ",
                     gcs_stream_.status().message()));
  }
  if (gcs_stream_.bad()) {
    return absl::Status(absl::StatusCode::kUnknown,
                        absl::StrCat("Failed to read open ", path,
                                     " with error:", std::strerror(errno)));
  }
  return absl::OkStatus();
}

absl::StatusOr<int> FileInputByteStream::ReadUpTo(char* buffer, int max_read) {
  gcs_stream_.read(buffer, max_read);
  if (gcs_stream_.bad()) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to read chunk");
  }
  return gcs_stream_.gcount();
}

absl::StatusOr<bool> FileInputByteStream::ReadExactly(char* buffer,
                                                      int num_read) {
  gcs_stream_.read(buffer, num_read);
  const auto read_count = gcs_stream_.gcount();
  if (gcs_stream_.bad() || (read_count > 0 && read_count < num_read)) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to read chunk");
  }
  return read_count > 0 || num_read == 0;
}

absl::Status FileInputByteStream::Close() {
  gcs_stream_.Close();
  if (gcs_stream_.bad()) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to close file");
  }
  return absl::OkStatus();
}

absl::Status FileOutputByteStream::Open(absl::string_view path) {
  ASSIGN_OR_RETURN(const auto cloud_path, GetGCSPath(path));
  gcs_stream_ =
      GetGCSClient().WriteObject(cloud_path.bucket, cloud_path.object);
  if (!gcs_stream_.last_status().ok()) {
    return absl::Status(
        absl::StatusCode::kUnknown,
        absl::StrCat("Failed to gcs write open ", path, " with error ",
                     gcs_stream_.last_status().message()));
  }
  if (gcs_stream_.bad()) {
    return absl::Status(absl::StatusCode::kUnknown,
                        absl::StrCat("Failed to write open ", path,
                                     " with error:", std::strerror(errno)));
  }
  return absl::OkStatus();
}

absl::Status FileOutputByteStream::Write(absl::string_view chunk) {
  gcs_stream_.write(chunk.data(), chunk.size());
  if (gcs_stream_.bad()) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to write chunk");
  }
  return absl::OkStatus();
}

absl::Status FileOutputByteStream::Close() {
  gcs_stream_.Close();
  if (gcs_stream_.bad()) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to close file");
  }
  return absl::OkStatus();
}

class FileSystemImplementation : public FileSystemInterface {
 public:
  ~FileSystemImplementation() {}

  std::string JoinPathList(
      std::initializer_list<absl::string_view> paths) override {
    LOG(FATAL) << "Not implemented";
    return "";
  }

  bool GenerateShardedFilenames(absl::string_view spec,
                                std::vector<std::string>* names) override {
    LOG(FATAL) << "Not implemented";
    return true;
  }

  absl::Status Match(absl::string_view pattern,
                     std::vector<std::string>* results,
                     const int options) override {
    ASSIGN_OR_RETURN(const auto cloud_path, GetGCSPath(pattern));
    auto client = GetGCSClient();

    // The prefix is an efficient filter applies on the candidate server side
    // (unlike the regex which is applied client side).
    const auto end_of_prefix = cloud_path.object.find_first_of("?*[]");
    std::string prefix;
    if (end_of_prefix != -1) {
      prefix = cloud_path.object.substr(0, end_of_prefix);
    }

    auto candidates = client.ListObjects(
        cloud_path.bucket, google::cloud::storage::Prefix(prefix));

    std::string regexp_filename = absl::StrReplaceAll(
        cloud_path.object, {{".", "\\."}, {"*", ".*"}, {"?", "."}});
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

  absl::Status RecursivelyCreateDir(absl::string_view path,
                                    int options) override {
    // Nothing to be done.
    return absl::OkStatus();
  }

  absl::Status RecursivelyDelete(absl::string_view path, int options) override {
    ASSIGN_OR_RETURN(const auto cloud_path, GetGCSPath(path));
    // Note: client.DeleteObject does not work with directories.
    auto client = GetGCSClient();
    GCS_RETURN_IF_ERROR(::google::cloud::storage::DeleteByPrefix(
        client, cloud_path.bucket, cloud_path.object));
    return absl::OkStatus();
  }

  absl::StatusOr<bool> FileExists(absl::string_view path) override {
    ASSIGN_OR_RETURN(const auto cloud_path, GetGCSPath(path));
    return GetGCSClient()
        .GetObjectMetadata(cloud_path.bucket, cloud_path.object)
        .ok();
  }

  absl::Status Rename(absl::string_view from, absl::string_view to,
                      int options) override {
    ASSIGN_OR_RETURN(const auto from_cloud_path, GetGCSPath(from));
    ASSIGN_OR_RETURN(const auto to_cloud_path, GetGCSPath(to));
    auto client = GetGCSClient();
    GCS_RETURN_IF_ERROR(client
                            .RewriteObjectBlocking(
                                from_cloud_path.bucket, from_cloud_path.object,
                                to_cloud_path.bucket, to_cloud_path.object)
                            .status());
    GCS_RETURN_IF_ERROR(
        client.DeleteObject(from_cloud_path.bucket, from_cloud_path.object));
    return absl::OkStatus();
  }

  std::string GetBasename(absl::string_view path) override {
    LOG(FATAL) << "Not implemented";
    return "";
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
  SetGCSImplementation(std::make_unique<FileSystemImplementation>());
  return 0;
}

static const int a = init();

}  // namespace yggdrasil_decision_forests::utils::filesystem::gcs
