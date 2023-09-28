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

#include "yggdrasil_decision_forests/utils/distribute_cli/distribute_cli_worker.h"

#ifndef _WIN32
#include <stdio.h>  // popen, pclose
#endif

#include <array>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace distribute_cli {

namespace {
using Blob = distribute::Blob;

}  // namespace

#ifndef _WIN32
absl::StatusOr<bool> Run(const std::string& command,
                         const std::string& log_path,
                         const bool display_commands_output) {
  // Output file stream for the logs.
  ASSIGN_OR_RETURN(auto log_handle, file::OpenOutputFile(log_path));
  file::OutputFileCloser log_closer(std::move(log_handle));

  std::array<char, 2048> buffer;
  FILE* pipe = popen(absl::StrCat(command, " 2>&1").c_str(), "r");
  if (!pipe) {
    YDF_LOG(WARNING) << "popen() failed";
    return absl::InvalidArgumentError("popen() failed");
  }
  absl::Status pending_status;
  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    const auto write_log_status = log_closer.stream()->Write(buffer.data());
    if (!write_log_status.ok()) {
      YDF_LOG(WARNING) << "Failure to write logs: "
                       << write_log_status.message();
      pending_status.Update(write_log_status);
      break;
    }
    if (display_commands_output) {
      puts(buffer.data());
    }
  }
  int pclose_status = pclose(pipe);
  // In some implementations, WEXITSTATUS requires a lvalue.
  auto error = WEXITSTATUS(pclose_status);
  RETURN_IF_ERROR(log_closer.Close());
  RETURN_IF_ERROR(pending_status);

  return error == 0;
}
#else

#include <Windows.h>
absl::StatusOr<bool> Run(const std::string& command,
                         const std::string& log_path,
                         const bool display_commands_output) {
  // Output file stream for the logs.
  ASSIGN_OR_RETURN(auto log_handle, file::OpenOutputFile(log_path));
  file::OutputFileCloser log_closer(std::move(log_handle));

  // Create pipes for capturing stdout and stderr
  HANDLE stdout_read, stdout_write;
  SECURITY_ATTRIBUTES sec;
  sec.nLength = sizeof(SECURITY_ATTRIBUTES);
  sec.bInheritHandle = true;
  sec.lpSecurityDescriptor = nullptr;

  if (!CreatePipe(&stdout_read, &stdout_write, &sec, 0)) {
    return false;
  }

  STARTUPINFOA si;
  PROCESS_INFORMATION pi;

  ZeroMemory(&si, sizeof(STARTUPINFOA));
  si.cb = sizeof(STARTUPINFOA);
  si.hStdError = stdout_write;   // Redirect stderr to the same pipe
  si.hStdOutput = stdout_write;  // Redirect stdout to the same pipe
  si.dwFlags |= STARTF_USESTDHANDLES;

  std::string mutable_command = command;
  if (!CreateProcessA(nullptr, &mutable_command[0], nullptr, nullptr, true, 0,
                      nullptr, nullptr, &si, &pi)) {
    CloseHandle(stdout_read);
    CloseHandle(stdout_write);
    return false;
  }

  CloseHandle(stdout_write);

  std::array<char, 2048> buffer;
  DWORD num_read_bytes;
  absl::Status pending_status;
  while (ReadFile(stdout_read, buffer.data(), buffer.size() - 1,
                  &num_read_bytes, NULL) != 0 &&
         num_read_bytes != 0) {
    buffer[num_read_bytes] = '\0';

    const auto write_log_status = log_closer.stream()->Write(buffer.data());
    if (!write_log_status.ok()) {
      YDF_LOG(WARNING) << "Failure to write logs: "
                       << write_log_status.message();
      pending_status.Update(write_log_status);
      break;
    }
    if (display_commands_output) {
      puts(buffer.data());
    }
  }
  RETURN_IF_ERROR(pending_status);

  CloseHandle(stdout_read);

  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD exit_code;
  GetExitCodeProcess(pi.hProcess, &exit_code);

  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  return exit_code == 0;
}
#endif

absl::Status Worker::RunCommand(const absl::string_view command,
                                const absl::string_view log_path) {
  file::RecursivelyDelete(log_path, file::Defaults()).IgnoreError();
  ASSIGN_OR_RETURN(const bool command_worked,
                   Run(std::string(command), std::string(log_path),
                       welcome_.display_commands_output()));
  if (command_worked) {
    return absl::OkStatus();
  }

  if (welcome_.display_output()) {
    YDF_LOG(INFO) << "The command failed";
  }

  std::string end_of_logs;
  // TODO: Only remember the end of the file.
  const auto end_of_logs_or = file::GetContent(log_path);
  if (end_of_logs_or.ok()) {
    end_of_logs = std::move(end_of_logs_or).value();
    constexpr int max_length = 5000;
    if (end_of_logs.size() > max_length) {
      end_of_logs = end_of_logs.substr(end_of_logs.size() - max_length);
    }
  } else {
    end_of_logs = "Logs not available.";
  }

  std::string error_message = absl::Substitute(
      "The following command failed:\n\n$0\n\nLog files: "
      "$1\n\nLast 5k "
      "characters of logs:\n\n$2",
      command, log_path, end_of_logs);
  return absl::InvalidArgumentError(error_message);
}

absl::Status Worker::Setup(Blob serialized_welcome) {
  ASSIGN_OR_RETURN(welcome_,
                   utils::ParseBinaryProto<proto::Welcome>(serialized_welcome));
  return absl::OkStatus();
}

absl::Status Worker::Command(const proto::Request::Command& request,
                             proto::Result::Command* result) {
  // TODO: Kill the command if "done_was_called_" becomes true.

  result->set_internal_command_id(request.internal_command_id());

  // Split of the path into a hierarchy of directory to avoid creating a single
  // directly with too many files/
  std::string output_dir;
  std::string output_base_filename;
  BaseOutput(welcome_.log_dir(), request.internal_command_id(), &output_dir,
             &output_base_filename);
  RETURN_IF_ERROR(file::RecursivelyCreateDir(output_dir, file::Defaults()));

  // Path to log the execution progress.
  const auto base_path = file::JoinPath(output_dir, output_base_filename);
  const auto done_path = absl::StrCat(base_path, ".done");
  const auto fail_path = absl::StrCat(base_path, ".fail");
  const auto progress_path = absl::StrCat(base_path, ".progress");
  const auto log_path = absl::StrCat(base_path, ".log");

  // Check if the command was already run.
  ASSIGN_OR_RETURN(const auto done_already_exist, file::FileExists(done_path));
  if (done_already_exist) {
    if (welcome_.display_output()) {
      YDF_LOG(INFO) << "The command " << request.internal_command_id()
                    << " was already run";
    }
    return absl::OkStatus();
  }

  // Note: The tf-filesystem does not support well empty files.
  file::RecursivelyDelete(progress_path, file::Defaults()).IgnoreError();
  RETURN_IF_ERROR(file::SetContent(progress_path, request.command()));

  // Effectively run the command.
  if (welcome_.display_output()) {
    YDF_LOG(INFO) << "Running command " << request.internal_command_id()
                  << ":\n"
                  << request.command() << "\nwith logs in: " << log_path;
  }
  const auto begin_time = absl::Now();

  const auto status = RunCommand(request.command(),
                                 /*log_path*/ log_path);

  if (!status.ok()) {
    if (welcome_.display_output()) {
      YDF_LOG(INFO) << "The command " << request.internal_command_id()
                    << " failed.\nThe full command was:\n\n"
                    << request.command() << "\n\nwith logs in: " << log_path;
    }
    file::RecursivelyDelete(fail_path, file::Defaults()).IgnoreError();
    RETURN_IF_ERROR(file::SetContent(fail_path, "fail"));
    return status;
  }

  if (welcome_.display_output()) {
    YDF_LOG(INFO) << "The command " << request.internal_command_id()
                  << " completed in " << (absl::Now() - begin_time);
  }
  file::RecursivelyDelete(done_path, file::Defaults()).IgnoreError();
  RETURN_IF_ERROR(file::SetContent(done_path, "done"));
  return absl::OkStatus();
}

absl::StatusOr<Blob> Worker::RunRequest(Blob serialized_request) {
  const auto begin_time = absl::Now();

  ASSIGN_OR_RETURN(auto request,
                   utils::ParseBinaryProto<proto::Request>(serialized_request));
  proto::Result result;
  if (request.has_request_id()) {
    result.set_request_id(request.request_id());
  }
  switch (request.type_case()) {
    case proto::Request::kCommand:
      RETURN_IF_ERROR(Command(request.command(), result.mutable_command()));
      break;
    case proto::Request::TYPE_NOT_SET:
      return absl::InvalidArgumentError("Request without type");
  }

  result.set_worker(WorkerIdx());
  result.set_duration(absl::ToDoubleSeconds(absl::Now() - begin_time));
  return result.SerializeAsString();
}

}  // namespace distribute_cli
}  // namespace utils
}  // namespace yggdrasil_decision_forests
