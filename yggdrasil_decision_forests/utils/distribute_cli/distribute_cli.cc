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

#include "yggdrasil_decision_forests/utils/distribute_cli/distribute_cli.h"

#include "yggdrasil_decision_forests/utils/distribute_cli/common.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace distribute_cli {

CommandBuilder::CommandBuilder(const absl::string_view binary)
    : binary_(binary) {}

CommandBuilder& CommandBuilder::Arg(const absl::string_view value) {
  args_.push_back(std::string(value));
  return *this;
}

CommandBuilder& CommandBuilder::Arg(const absl::string_view key,
                                    const absl::string_view value) {
  Arg(absl::StrCat(key, "=", value));
  return *this;
}

std::string CommandBuilder::Build() const {
  std::string command = binary_;
  for (const auto& arg : args_) {
    absl::StrAppend(&command, " \"", arg, "\"");
  }
  return command;
}

DistributeCLIManager::DistributeCLIManager(const proto::Config& config)
    : config_(config) {}

absl::Status DistributeCLIManager::Initialize() {
  if (distribute_manager_) {
    return absl::InvalidArgumentError("Already initialized");
  }

  // Create the log directory.
  log_dir_ =
      file::JoinPath(config_.distribute_config().working_directory(), "logs");
  if (!config_.skip_already_run_commands()) {
    log_dir_ = file::JoinPath(log_dir_, utils::GenUniqueId());
  }
  RETURN_IF_ERROR(file::RecursivelyCreateDir(log_dir_, file::Defaults()));
  if (config_.distribute_config().verbosity() >= 1) {
    LOG(INFO) << "Distribute CLI log directory: " << log_dir_;
  }

  // Start the distribute manager.
  proto::Welcome welcome;
  welcome.set_log_dir(log_dir_);
  welcome.set_display_output(config_.display_worker_output());
  ASSIGN_OR_RETURN(
      distribute_manager_,
      distribute::CreateManager(config_.distribute_config(),
                                /*worker_name=*/kWorkerKey,
                                /*welcome_blob=*/welcome.SerializeAsString(),
                                /*parallel_execution_per_worker=*/
                                config_.parallel_execution_per_worker()));
  return absl::OkStatus();
}

absl::Status DistributeCLIManager::Schedule(const absl::string_view command) {
  if (config_.distribute_config().verbosity() >= 2) {
    LOG(INFO) << "Schedule command: " << command;
  }
  proto::Request generic_request;
  auto& request = *generic_request.mutable_command();
  *request.mutable_command() = command;
  *request.mutable_internal_command_id() = CommandToInternalCommandId(command);

  const auto status =
      distribute_manager_->AsynchronousProtoRequest(std::move(generic_request));
  if (status.ok()) {
    pending_commands_++;
  }
  return status;
}

absl::Status DistributeCLIManager::WaitCompletion() {
  if (config_.distribute_config().verbosity() >= 1) {
    LOG(INFO) << "Running " << pending_commands_ << " commands";
  }
  const auto num_commands = pending_commands_;
  while (pending_commands_ > 0) {
    ASSIGN_OR_RETURN(
        const auto generic_result,
        distribute_manager_->NextAsynchronousProtoAnswer<proto::Result>());
    pending_commands_--;
    if (config_.distribute_config().verbosity() >= 1) {
      LOG_INFO_EVERY_N_SEC(30, _ << "\t" << (num_commands - pending_commands_)
                                 << " / " << num_commands
                                 << " commands completed");
    }
  }
  if (config_.distribute_config().verbosity() >= 1) {
    LOG(INFO) << "All commands completed";
  }
  return absl::OkStatus();
}

absl::Status DistributeCLIManager::Shutdown() {
  if (config_.distribute_config().verbosity() >= 2) {
    LOG(INFO) << "Shutting down Distribute CLI manager";
  }
  RETURN_IF_ERROR(distribute_manager_->Done());
  distribute_manager_.reset();
  return absl::OkStatus();
}

}  // namespace distribute_cli
}  // namespace utils
}  // namespace yggdrasil_decision_forests
