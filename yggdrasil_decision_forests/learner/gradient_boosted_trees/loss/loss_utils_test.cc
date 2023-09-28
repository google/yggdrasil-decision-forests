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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_utils.h"

#include <vector>

#include "gmock/gmock.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

using ::testing::FloatNear;

constexpr double kTestPrecision = 0.000001;

TEST(CreateSetLeafValueFunctor, NonWeighted) {
  proto::GradientBoostedTreesTrainingConfig gbt_config;

  std::vector<float> gradient = {1, 2, 3};
  std::vector<float> hessian = {4, 5, 6};
  std::vector<UnsignedExampleIdx> selected_examples{0, 1};
  std::vector<float> weights = {};
  GradientData gradient_data{.gradient = gradient, .hessian = hessian};

  decision_tree::NodeWithChildren node;
  EXPECT_OK(SetLeafValueWithNewtonRaphsonStep<false>(
      gbt_config, selected_examples, weights, gradient_data, &node));

  EXPECT_THAT(node.node().regressor().top_value(),
              FloatNear(0.1 * (1. + 2.) / (4. + 5.), kTestPrecision));

  EXPECT_EQ(node.node().regressor().distribution().sum(), 1 + 2);
  EXPECT_EQ(node.node().regressor().distribution().sum_squares(),
            1 * 1 + 2 * 2);
  EXPECT_EQ(node.node().regressor().distribution().count(), 1 + 1);
}

TEST(CreateSetLeafValueFunctor, Weighted) {
  proto::GradientBoostedTreesTrainingConfig gbt_config;

  std::vector<float> gradient = {1, 2, 3};
  std::vector<float> hessian = {4, 5, 6};
  std::vector<UnsignedExampleIdx> selected_examples{0, 1};
  std::vector<float> weights = {1.f, 2.f, 3.f};
  GradientData gradient_data{.gradient = gradient, .hessian = hessian};

  decision_tree::NodeWithChildren node;
  EXPECT_OK(SetLeafValueWithNewtonRaphsonStep<true>(
      gbt_config, selected_examples, weights, gradient_data, &node));

  EXPECT_THAT(node.node().regressor().top_value(),
              FloatNear(0.1 * (1. * 1. + 2. * 2.) / (4. * 1. + 5. * 2.),
                        kTestPrecision));

  EXPECT_EQ(node.node().regressor().distribution().sum(), 1 * 1 + 2 * 2);
  EXPECT_EQ(node.node().regressor().distribution().sum_squares(),
            1 * 1 * 1 + 2 * 2 * 2);
  EXPECT_EQ(node.node().regressor().distribution().count(), 1 + 2);
}

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
