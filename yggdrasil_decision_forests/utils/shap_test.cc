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

#include "yggdrasil_decision_forests/utils/shap.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace yggdrasil_decision_forests::utils::shap {
namespace {

using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::FieldsAre;

constexpr double kMargin = 0.001;

// Checks "extend" against values computed with the SHAP python package,
// and check that "extend" and "unwind" are commutative.
TEST(Shape, extend_and_unwind) {
  internal::Path path;
  internal::extend(0.1, 0.9, 1, path);
  EXPECT_THAT(path,
              ElementsAre(FieldsAre(1, 0.1, 0.9, DoubleNear(1., kMargin))));

  internal::extend(0.2, 0.8, 2, path);
  EXPECT_THAT(path,
              ElementsAre(FieldsAre(1, 0.1, 0.9, DoubleNear(0.1, kMargin)),
                          FieldsAre(2, 0.2, 0.8, DoubleNear(0.4, kMargin))));

  internal::extend(0.3, 0.7, 3, path);
  EXPECT_THAT(path,
              ElementsAre(FieldsAre(1, 0.1, 0.9, DoubleNear(0.02, kMargin)),
                          FieldsAre(2, 0.2, 0.8, DoubleNear(0.0633, kMargin)),
                          FieldsAre(3, 0.3, 0.7, DoubleNear(0.1866, kMargin))));

  internal::unwind(2, path);
  EXPECT_THAT(path,
              ElementsAre(FieldsAre(1, 0.1, 0.9, DoubleNear(0.1, kMargin)),
                          FieldsAre(2, 0.2, 0.8, DoubleNear(0.4, kMargin))));

  internal::unwind(1, path);
  EXPECT_THAT(path,
              ElementsAre(FieldsAre(1, 0.1, 0.9, DoubleNear(1., kMargin))));
}

// Checks that "unwound_sum" is equivalent to the sum of weights after a
// "unwound" call.
TEST(Shape, unwound_sum) {
  internal::Path path;
  internal::extend(0.1, 0.9, 1, path);
  internal::extend(0.2, 0.8, 2, path);
  internal::extend(0.3, 0.7, 3, path);

  const double fast_sum = unwound_sum(2, path);

  internal::unwind(2, path);
  double slow_sum = 0;
  for (const auto& p : path) {
    slow_sum += p.weight;
  }

  EXPECT_NEAR(fast_sum, slow_sum, kMargin);
}

}  // namespace
}  // namespace yggdrasil_decision_forests::utils::shap
