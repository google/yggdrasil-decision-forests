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

#include "yggdrasil_decision_forests/utils/snapshot.h"

#include <deque>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"  // IWYU pragma: keep
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::utils {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::yggdrasil_decision_forests::test::StatusIs;

TEST(SnapShot, GetSnapshot) {
  const std::string snapshot_dir =
      file::JoinPath(test::TmpDirectory(), "get_snapshot");
  EXPECT_THAT(GetGreatestSnapshot(snapshot_dir).status(),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_OK(AddSnapshot(snapshot_dir, 5));
  EXPECT_OK(AddSnapshot(snapshot_dir, 11));
  ASSERT_OK_AND_ASSIGN(const int content, GetGreatestSnapshot(snapshot_dir));
  EXPECT_EQ(content, 11);
}

TEST(SnapShot, GetSnapshots) {
  const std::string snapshot_dir =
      file::JoinPath(test::TmpDirectory(), "get_snapshots");
  ASSERT_OK_AND_ASSIGN(const std::deque<int> content,
                       GetSnapshots(snapshot_dir));
  EXPECT_THAT(content, IsEmpty());
}

TEST(SnapShot, AddSnapshot) {
  const std::string snapshot_dir =
      file::JoinPath(test::TmpDirectory(), "add_snapshot");
  EXPECT_OK(AddSnapshot(snapshot_dir, 5));
  EXPECT_OK(AddSnapshot(snapshot_dir, 11));
  EXPECT_OK(AddSnapshot(snapshot_dir, 6));
  ASSERT_OK_AND_ASSIGN(const std::deque<int> content,
                       GetSnapshots(snapshot_dir));
  EXPECT_THAT(content, ElementsAre(5, 6, 11));
}

TEST(SnapShot, RemoveOldSnapshots) {
  const std::string snapshot_dir =
      file::JoinPath(test::TmpDirectory(), "remove_old_snapshots");
  EXPECT_OK(AddSnapshot(snapshot_dir, 5));
  EXPECT_OK(AddSnapshot(snapshot_dir, 11));
  EXPECT_OK(AddSnapshot(snapshot_dir, 6));

  std::deque<int> snapshots = {5, 6, 11};
  EXPECT_THAT(RemoveOldSnapshots(snapshot_dir, 2, snapshots), ElementsAre(5));

  EXPECT_THAT(snapshots, ElementsAre(6, 11));
  ASSERT_OK_AND_ASSIGN(const std::deque<int> content,
                       GetSnapshots(snapshot_dir));
  EXPECT_THAT(content, ElementsAre(6, 11));
}

}  // namespace
}  // namespace yggdrasil_decision_forests::utils