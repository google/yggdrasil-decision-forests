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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"

#include "gmock/gmock.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace dataset_cache {
namespace {

TEST(DeltaBit, Base) {
  EXPECT_EQ(MaskDeltaBit(0b1101), 0b10000);
  EXPECT_EQ(MaskExampleIdx(0b1101), 0b01111);
  EXPECT_EQ(MaxValue(0b1101), 0b01111);

  EXPECT_EQ(MaskDeltaBit(0b1000), 0b10000);
  EXPECT_EQ(MaskDeltaBit(0b1111), 0b10000);
}

}  // namespace
}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
