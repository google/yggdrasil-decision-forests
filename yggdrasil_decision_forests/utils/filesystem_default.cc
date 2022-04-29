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

#include "yggdrasil_decision_forests/utils/filesystem_default.h"

#include <filesystem>
#include <regex>  // NOLINT

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

// Converts a absl::string_view into an object compatible with std::filesystem.
#ifdef ABSL_USES_STD_STRING_VIEW
#define SV_ABSL_TO_STD(X) X
#else
#define SV_ABSL_TO_STD(X) std::string(X)
#endif

namespace file {

namespace ygg = ::yggdrasil_decision_forests;

std::string JoinPathList(std::initializer_list<absl::string_view> paths) {
  std::filesystem::path all_paths;
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
  try {
    const auto search_dir =
        std::filesystem::path(SV_ABSL_TO_STD(pattern)).parent_path();
    const auto filename =
        std::filesystem::path(SV_ABSL_TO_STD(pattern)).filename().string();
    std::string regexp_filename =
        absl::StrReplaceAll(filename, {{".", "\\."}, {"*", ".*"}, {"?", "."}});
    std::regex regexp_pattern(regexp_filename);
    for (auto& path : std::filesystem::directory_iterator(search_dir)) {
      if (std::regex_match(path.path().filename().string(), regexp_pattern)) {
        results->push_back(path.path().string());
      }
    }
    std::sort(results->begin(), results->end());
    return absl::OkStatus();
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(e.what());
  }
}

absl::Status RecursivelyCreateDir(absl::string_view path, int options) {
  try {
    std::filesystem::create_directories(SV_ABSL_TO_STD(path));
    return absl::OkStatus();
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(e.what());
  }
}

// Delete the directory "path".
absl::Status RecursivelyDelete(absl::string_view path, int options) {
  try {
    std::filesystem::remove(SV_ABSL_TO_STD(path));
    return absl::OkStatus();
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(e.what());
  }
}

absl::Status FileInputByteStream::Open(absl::string_view path) {
  file_ = std::fopen(std::string(path).c_str(), "rb");
  if (!file_) {
    return absl::Status(absl::StatusCode::kUnknown,
                        absl::StrCat("Failed to open ", path));
  }
  return absl::OkStatus();
}

yggdrasil_decision_forests::utils::StatusOr<int> FileInputByteStream::ReadUpTo(
    char* buffer, int max_read) {
  const int num_read = std::fread(buffer, sizeof(char), max_read, file_);
  if (num_read < 0) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to read chunk");
  }
  return num_read;
}

ygg::utils::StatusOr<bool> FileInputByteStream::ReadExactly(char* buffer,
                                                            int num_read) {
  const int read_count = std::fread(buffer, sizeof(char), num_read, file_);
  // read_count: -1=Error, 0=EOF, >0 & <num_read=Partial read, num_read=Success.
  if (read_count < 0 || (read_count > 0 && read_count < num_read)) {
    return absl::Status(absl::StatusCode::kUnknown, "Failed to read chunk");
  }
  return read_count > 0 || num_read == 0;
}

absl::Status FileInputByteStream::Close() {
  std::fclose(file_);
  return absl::OkStatus();
}

absl::Status FileOutputByteStream::Open(absl::string_view path) {
  file_ = std::fopen(std::string(path).c_str(), "wb");
  if (!file_) {
    return absl::Status(absl::StatusCode::kUnknown,
                        absl::StrCat("Failed to open ", path));
  }
  return absl::OkStatus();
}

absl::Status FileOutputByteStream::Write(absl::string_view chunk) {
  std::fwrite(chunk.data(), sizeof chunk[0], chunk.size(), file_);
  return absl::OkStatus();
}

absl::Status FileOutputByteStream::Close() {
  std::fclose(file_);
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

yggdrasil_decision_forests::utils::StatusOr<bool> FileExists(
    absl::string_view path) {
  return std::filesystem::exists(SV_ABSL_TO_STD(path));
}

absl::Status Rename(absl::string_view from, absl::string_view to, int options) {
  try {
    std::filesystem::rename(SV_ABSL_TO_STD(from), SV_ABSL_TO_STD(to));
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(e.what());
  }
  return absl::OkStatus();
}

std::string GetBasename(absl::string_view path) {
  try {
    return std::filesystem::path(path).filename();
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error parsing basename of " << path << ": " << e.what();
    return "";
  }
}

}  // namespace file
