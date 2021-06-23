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

#include "yggdrasil_decision_forests/utils/snapshot.h"

#include "gmock/gmock.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

TEST(SnapShot, Base) {
  const auto snapshot_dir = file::JoinPath(test::TmpDirectory(), "snapshot");
  EXPECT_FALSE(GetGreatestSnapshot(snapshot_dir).ok());
  EXPECT_OK(AddSnapshot(snapshot_dir, 5));
  EXPECT_OK(AddSnapshot(snapshot_dir, 11));
  EXPECT_OK(AddSnapshot(snapshot_dir, 6));
  EXPECT_EQ(GetGreatestSnapshot(snapshot_dir).value(), 11);
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests