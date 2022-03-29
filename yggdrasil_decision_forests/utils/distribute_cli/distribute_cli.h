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

// Distribute CLI
//
// The Distribute CLI library distributes the execution of a set of command line
// calls using the "distribute" API.
//
// Usage example:
//   auto manager = DistributeCLIManager(CreateDistributeConfig(), {});
//   EXPECT_OK(manager.Initialize());
//   EXPECT_OK(manager.Schedule(CommandBuilder("ls").Arg("-l").Build()));
//   EXPECT_OK(manager.Schedule(CommandBuilder("ls").Arg("-h").Build()));
//   EXPECT_OK(manager.WaitCompletion());
//   EXPECT_OK(manager.Shutdown());
//
// In this example, "ls -l" and "ls -h" might be executed on the same or a
// a different distribute worker (depending on the configuration). If any of
// those comments fails, "WaitCompletion" will return an error.
//
#ifndef THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_CLI_H_
#define THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_CLI_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.pb.h"
#include "yggdrasil_decision_forests/utils/distribute_cli/distribute_cli.pb.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace distribute_cli {

// Utility for the creation of a commandline.
//
// Usage example:
//   std::string command = CommandBuilder("ls").Arg("-h").Build();
//
class CommandBuilder {
 public:
  CommandBuilder(const absl::string_view binary);
  CommandBuilder& Arg(const absl::string_view value);
  CommandBuilder& Arg(const absl::string_view key,
                      const absl::string_view value);
  std::string Build() const;

 private:
  std::string binary_;
  std::vector<std::string> args_;
};

// Manager.
// See header for a usage example.
class DistributeCLIManager {
 public:
  // Create the manager. Returns immediately.
  DistributeCLIManager(const proto::Config& config);

  // Initializes the manager and the workers. Blocks until all the workers are
  // available.
  absl::Status Initialize();

  // Schedules a new commands.
  // Args:
  //   uid: Unique identifier of the command to skip already executed commands
  //     is "skip_already_run_commands=true". If not specified, the uid is
  //     "command".
  absl::Status Schedule(const absl::string_view command,
                        const absl::optional<std::string>& uid = {});

  // Waits for all the previously scheduled commands to run. If a command fails,
  // returns immediately with the error. Following an error, the manager can be
  // stopped (using "Shutdown"), or "WaitCompletion" can be called again to wait
  // for the other commands to finish.
  absl::Status WaitCompletion();

  // Stops the manager and the workers. Wait for the workers to finish any
  // already running commands. Scheduled but not yet running commands are
  // ignored.
  absl::Status Shutdown();

  // Paths to the file that will contain the execution log of a command.
  std::string LogPathFromUid(const absl::string_view uid);

 private:
  // Paths to the file that will contain the execution log of a command.
  std::string LogPathFromInternalUid(
      const absl::string_view internal_command_id);

  // Schedules a new commands immediately.
  absl::Status ScheduleNow(const absl::string_view command,
                           const absl::optional<std::string>& uid);

  std::unique_ptr<distribute::AbstractManager> distribute_manager_;
  proto::Config config_;
  std::string log_dir_;

  // Commands sent to the workers when "WaitCompletion" is called.
  struct WaitingCommand {
    std::string command;
    absl::optional<std::string> uid;
  };
  std::vector<WaitingCommand> waiting_commands_;

  // Number of scheduled commands.
  int pending_commands_ = 0;

  absl::flat_hash_set<std::string> past_commands_;
};

}  // namespace distribute_cli
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_CLI_H_
