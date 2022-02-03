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

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/multi_thread/multi_thread.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace distribute_cli {
namespace {

// Create a single thread manager with 5 workers.
proto::Config CreateConfig() {
  proto::Config config;
  config.mutable_distribute_config()->set_implementation_key("MULTI_THREAD");
  config.mutable_distribute_config()
      ->MutableExtension(distribute::proto::multi_thread)
      ->set_num_workers(5);
  config.mutable_distribute_config()->set_verbosity(2);
  config.mutable_distribute_config()->set_working_directory(
      test::TmpDirectory());
  return config;
}

TEST(DistributeCLI, Ls) {
  auto manager = DistributeCLIManager(CreateConfig());
  EXPECT_OK(manager.Initialize());
  EXPECT_OK(manager.Schedule(CommandBuilder("ls").Arg("-l").Build()));
  EXPECT_OK(manager.Schedule(CommandBuilder("ls").Arg("-l").Arg("-h").Build()));
  EXPECT_OK(manager.WaitCompletion());
  EXPECT_OK(manager.Shutdown());
}

TEST(DistributeCLI, Error) {
  auto manager = DistributeCLIManager(CreateConfig());
  EXPECT_OK(manager.Initialize());
  EXPECT_OK(manager.Schedule(
      CommandBuilder("non_existing_command").Arg("-l").Build()));
  EXPECT_THAT(manager.WaitCompletion(),
              test::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_OK(manager.Shutdown());
}

}  // namespace
}  // namespace distribute_cli
}  // namespace utils
}  // namespace yggdrasil_decision_forests
