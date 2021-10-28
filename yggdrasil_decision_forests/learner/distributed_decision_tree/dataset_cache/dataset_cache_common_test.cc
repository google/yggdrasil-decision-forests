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

using testing::ElementsAre;

const double kEps = 0.00001;

TEST(DeltaBit, Base) {
  EXPECT_EQ(MaskDeltaBit(0b1101), 0b10000);
  EXPECT_EQ(MaskExampleIdx(0b1101), 0b01111);
  EXPECT_EQ(MaxValueWithDeltaBit(0b1101), 0b11101);

  EXPECT_EQ(MaskDeltaBit(0b1000), 0b10000);
  EXPECT_EQ(MaskDeltaBit(0b1111), 0b10000);
}

TEST(NumericalToDiscretizedNumerical, Base) {
  EXPECT_NEAR(DiscretizedNumericalToNumerical({1, 2, 3}, 0), 0, kEps);
  EXPECT_NEAR(DiscretizedNumericalToNumerical({1, 2, 3}, 1), 1.5, kEps);
  EXPECT_NEAR(DiscretizedNumericalToNumerical({1, 2, 3}, 2), 2.5, kEps);
  EXPECT_NEAR(DiscretizedNumericalToNumerical({1, 2, 3}, 3), 4, kEps);
}

TEST(DiscretizedNumericalToNumerical, Base) {
  EXPECT_EQ(NumericalToDiscretizedNumerical({1, 2, 3}, -1), 0);
  EXPECT_EQ(NumericalToDiscretizedNumerical({1, 2, 3}, 1.5), 1);
  EXPECT_EQ(NumericalToDiscretizedNumerical({1, 2, 3}, 2.5), 2);
  EXPECT_EQ(NumericalToDiscretizedNumerical({1, 2, 3}, 4), 3);
}

TEST(ExtractDiscretizedBoundariesWithoutDownsampling, Base) {
  const auto boundaries =
      ExtractDiscretizedBoundariesWithoutDownsampling(
          {{10.f, 0}, {11.f, 1}, {11.f, 2}, {12.f, 3}, {12.f, 4}, {13.f, 5}}, 4)
          .value();
  EXPECT_THAT(boundaries, ElementsAre(10.5, 11.5f, 12.5f));
}

TEST(ExtractDiscretizedBoundariesWithDownsampling, Base) {
  const auto boundaries =
      ExtractDiscretizedBoundariesWithDownsampling(
          {{10.f, 0}, {11.f, 1}, {11.f, 2}, {12.f, 3}, {12.f, 4}, {13.f, 5}}, 4,
          3)
          .value();
  EXPECT_THAT(boundaries, ElementsAre(11.5f, 12.5f));
}

}  // namespace
}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
