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
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/builder.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

using ::testing::DoubleNear;
using ::testing::FloatNear;
using ::testing::Pointwise;

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

dataset::VerticalDataset CreateNumericalDataset(int num_examples) {
  dataset::VerticalDataset dataset;
  *dataset.mutable_data_spec() =
      PARSE_TEST_PROTO(R"pb(columns { type: NUMERICAL name: "x" })pb");
  EXPECT_OK(dataset.CreateColumnsFromDataspec());
  auto* col = dataset
                  .MutableColumnWithCastWithStatus<
                      dataset::VerticalDataset::NumericalColumn>(0)
                  .value();
  for (int i = 0; i < num_examples; ++i) {
    col->Add(i);
  }
  dataset.set_nrow(num_examples);
  return dataset;
}

decision_tree::DecisionTree CreateThresholdTree(float threshold,
                                                float neg_value,
                                                float pos_value) {
  decision_tree::DecisionTree tree;
  auto [pos, neg] =
      decision_tree::TreeBuilder(&tree).ConditionIsGreater(0, threshold);
  pos.LeafRegression(pos_value);
  neg.LeafRegression(neg_value);
  return tree;
}

TEST(UpdatePredictions, ParallelMatchesSequential) {
  const auto dataset = CreateNumericalDataset(100);
  const auto tree0 = CreateThresholdTree(49.5f, -1.5f, 2.25f);
  const auto tree1 = CreateThresholdTree(29.5f, 0.5f, -0.25f);
  utils::concurrency::ThreadPool thread_pool(/*num_threads=*/4,
                                             {.name_prefix = "test"});

  const std::vector<std::vector<const decision_tree::DecisionTree*>> cases = {
      {&tree0}, {&tree0, &tree1}};
  for (const auto& trees : cases) {
    std::vector<float> seq_preds(dataset.nrow() * trees.size(), 0.5f);
    std::vector<float> par_preds = seq_preds;
    double seq_mean_abs = 0.0;
    double par_mean_abs = 0.0;
    ASSERT_OK(UpdatePredictions(trees, dataset, &seq_preds, &seq_mean_abs));
    ASSERT_OK(UpdatePredictions(trees, dataset, &par_preds, &par_mean_abs,
                                &thread_pool));
    EXPECT_THAT(par_preds, Pointwise(FloatNear(kTestPrecision), seq_preds));
    EXPECT_THAT(par_mean_abs, DoubleNear(seq_mean_abs, kTestPrecision));
    if (trees.size() == 1) {
      EXPECT_THAT(par_mean_abs,
                  DoubleNear((50 * 1.5 + 50 * 2.25) / 100.0, kTestPrecision));
    }
  }
}

TEST(UpdatePredictions, EmptyDataset) {
  const auto dataset = CreateNumericalDataset(0);
  const auto tree = CreateThresholdTree(0.0f, 1.0f, 2.0f);
  utils::concurrency::ThreadPool thread_pool(/*num_threads=*/4,
                                             {.name_prefix = "test"});
  std::vector<float> preds;
  const std::vector<std::vector<const decision_tree::DecisionTree*>> cases = {
      {&tree}, {&tree, &tree}};
  for (const auto& trees : cases) {
    double mean_abs = -1.0;
    ASSERT_OK(
        UpdatePredictions(trees, dataset, &preds, &mean_abs, &thread_pool));
    EXPECT_EQ(mean_abs, 0.0);
  }
}

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
