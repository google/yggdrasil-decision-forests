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

#include "yggdrasil_decision_forests/utils/adaptive_work.h"

#include "gtest/gtest.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

constexpr double kEpsilon = 0.0001;

TEST(AdaptativeWork, Base) {
  AdaptativeWork manager(20, 100., 25., 0.);

  EXPECT_NEAR(manager.OptimalApproximationFactor(), 1.0, kEpsilon);
  manager.ReportTaskDone(1.0, 10.);

  EXPECT_NEAR(manager.OptimalApproximationFactor(), 1.0, kEpsilon);
  manager.ReportTaskDone(1.0, 10.);

  EXPECT_NEAR(manager.OptimalApproximationFactor(), 1.0, kEpsilon);
  manager.ReportTaskDone(1.0, 10.);

  EXPECT_NEAR(manager.OptimalApproximationFactor(), 0.5, kEpsilon);
  manager.ReportTaskDone(0.5, 5.);

  EXPECT_NEAR(manager.OptimalApproximationFactor(), 0.5, kEpsilon);
  manager.ReportTaskDone(0.5, 5.);
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
