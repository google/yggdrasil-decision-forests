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

// Tests of the checkpoint helpers of the distributed GBT manager. Unlike
// "distributed_gradient_boosted_trees_test", runs with all the file systems
// (e.g. "--define=ydf_use_default_libs=1").

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/common.h"
#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/snapshot.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_gradient_boosted_trees {
namespace {

using test::StatusIs;

TEST(RemoveIncompleteCheckpoint, Base) {
  const std::string work_directory =
      file::JoinPath(test::TmpDirectory(), "remove_incomplete_checkpoint");
  const std::string checkpoint_root =
      file::JoinPath(work_directory, kFileNameCheckPoint);
  const std::string snapshot_dir =
      file::JoinPath(checkpoint_root, kFileNameSnapshot);

  // Before the first checkpoint: no snapshot directory, no checkpoint.
  ASSERT_OK(file::RecursivelyCreateDir(work_directory, file::Defaults()));
  ASSERT_OK_AND_ASSIGN(bool exists, file::FileExists(snapshot_dir));
  ASSERT_FALSE(exists);
  EXPECT_OK(internal::RemoveIncompleteCheckpoint(work_directory, 0));

  // An incomplete first checkpoint (no snapshot directory): removed.
  const std::string incomplete_dir = file::JoinPath(checkpoint_root, "0");
  ASSERT_OK(file::RecursivelyCreateDir(incomplete_dir, file::Defaults()));
  EXPECT_OK(internal::RemoveIncompleteCheckpoint(work_directory, 0));
  ASSERT_OK_AND_ASSIGN(exists, file::FileExists(incomplete_dir));
  EXPECT_FALSE(exists);

  // A complete checkpoint: not removed.
  const std::string complete_dir = file::JoinPath(checkpoint_root, "5");
  ASSERT_OK(file::RecursivelyCreateDir(complete_dir, file::Defaults()));
  ASSERT_OK(utils::AddSnapshot(snapshot_dir, 5));
  EXPECT_THAT(internal::RemoveIncompleteCheckpoint(work_directory, 5),
              StatusIs(absl::StatusCode::kInternal, "already exists"));
  ASSERT_OK_AND_ASSIGN(exists, file::FileExists(complete_dir));
  EXPECT_TRUE(exists);

  // An incomplete checkpoint with files. Not an error even if the directory
  // cannot be removed (the files are replaced when the checkpoint is created
  // again).
  const std::string incomplete_dir_2 = file::JoinPath(checkpoint_root, "10");
  ASSERT_OK(file::RecursivelyCreateDir(incomplete_dir_2, file::Defaults()));
  ASSERT_OK(file::SetContent(file::JoinPath(incomplete_dir_2, "leftover"),
                             "leftover"));
  EXPECT_OK(internal::RemoveIncompleteCheckpoint(work_directory, 10));
}

}  // namespace
}  // namespace distributed_gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
