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

#include "gmock/gmock.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_binary_focal.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_binomial.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_cross_entropy_ndcg.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_ndcg.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

// TODO: Split the tests by implementation.

using testing::FloatNear;
using testing::NotNull;
using testing::SizeIs;

dataset::VerticalDataset CreateToyDataset() {
  dataset::VerticalDataset dataset;
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "a" }
    columns {
      type: CATEGORICAL
      name: "b"
      categorical { number_of_unique_values: 3 is_already_integerized: true }
    }
  )pb");
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "1"}, {"b", "1"}}));
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "2"}, {"b", "2"}}));
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "3"}, {"b", "1"}}));
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "4"}, {"b", "2"}}));
  return dataset;
}

TEST(GradientBoostedTrees, UpdateGradientsBinaryFocalLoss) {
  const auto dataset = CreateToyDataset();
  std::vector<float> weights(dataset.nrow(), 1.f);

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  proto::GradientBoostedTreesTrainingConfig config;
  config.set_use_hessian_gain(true);
  const auto loss_imp =
      BinaryFocalLoss(config, model::proto::Task::CLASSIFICATION,
                      dataset.data_spec().columns(1));
  CHECK_OK(internal::CreateGradientDataset(dataset,
                                           /* label_col_idx= */ 1,
                                           /*hessian_splits=*/false, loss_imp,
                                           &gradient_dataset, &gradients,
                                           &predictions));

  internal::SetInitialPredictions(
      loss_imp
          .InitialPredictions(dataset,
                              /* label_col_idx =*/1, weights)
          .value(),
      dataset.nrow(), &predictions);

  utils::RandomEngine random(1234);
  CHECK_OK(loss_imp.UpdateGradients(
      gradient_dataset, /* label_col_idx= */ 1, predictions,
      /*ranking_index=*/nullptr, &gradients, &random));

  const float test_prec = 0.000001f;

  // TODO: Implement and use "AllElementsNear" matcher.
  ASSERT_THAT(gradients.front().gradient, SizeIs(4));
  // Values validated with tensorflow focal loss implementation
  // (tfa.losses.sigmoid_focal_crossentropy).
  EXPECT_THAT(gradients.front().gradient[0], FloatNear(-0.149143f, test_prec));
  EXPECT_THAT(gradients.front().gradient[1], FloatNear(0.149143f, test_prec));
  EXPECT_THAT(gradients.front().gradient[2], FloatNear(-0.149143f, test_prec));
  EXPECT_THAT(gradients.front().gradient[3], FloatNear(0.149143f, test_prec));

  ASSERT_THAT(gradients.front().hessian, NotNull());
  const std::vector<float>& hessian = *gradients.front().hessian;
  EXPECT_THAT(hessian, SizeIs(4));
  EXPECT_THAT(hessian[0], FloatNear(0.199572f, test_prec));
  EXPECT_THAT(hessian[1], FloatNear(0.199572f, test_prec));
  EXPECT_THAT(hessian[2], FloatNear(0.199572f, test_prec));
  EXPECT_THAT(hessian[3], FloatNear(0.199572f, test_prec));
}

TEST(GradientBoostedTrees, UpdateGradientsBinaryFocalLossCustomPredictions) {
  const auto dataset = CreateToyDataset();
  std::vector<float> weights(dataset.nrow(), 1.f);

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  proto::GradientBoostedTreesTrainingConfig config;
  config.set_use_hessian_gain(true);
  const auto loss_imp =
      BinaryFocalLoss(config, model::proto::Task::CLASSIFICATION,
                      dataset.data_spec().columns(1));
  CHECK_OK(internal::CreateGradientDataset(dataset,
                                           /* label_col_idx= */ 1,
                                           /*hessian_splits=*/false, loss_imp,
                                           &gradient_dataset, &gradients,
                                           &predictions));

  const double test_prec = 0.000001f;

  predictions = {2.0, 2.0, -1.0, -1.0};

  utils::RandomEngine random(1234);
  CHECK_OK(loss_imp.UpdateGradients(
      gradient_dataset, /* label_col_idx= */ 1, predictions,
      /*ranking_index=*/nullptr, &gradients, &random));

  ASSERT_THAT(gradients.front().gradient, SizeIs(4));
  EXPECT_THAT(gradients.front().gradient[0], FloatNear(-0.538357f, test_prec));
  EXPECT_THAT(gradients.front().gradient[1], FloatNear(0.00243547f, test_prec));
  EXPECT_THAT(gradients.front().gradient[2], FloatNear(-0.0262906f, test_prec));
  EXPECT_THAT(gradients.front().gradient[3], FloatNear(0.384117f, test_prec));

  ASSERT_THAT(gradients.front().hessian, NotNull());
  const std::vector<float>& hessian = *gradients.front().hessian;
  EXPECT_THAT(hessian, SizeIs(4));
  EXPECT_THAT(hessian[0], FloatNear(0.0772814f, test_prec));
  EXPECT_THAT(hessian[1], FloatNear(0.00633879f, test_prec));
  EXPECT_THAT(hessian[2], FloatNear(0.0553163f, test_prec));
  EXPECT_THAT(hessian[3], FloatNear(0.226232f, test_prec));
}

TEST(GradientBoostedTrees, RankingIndex) {
  // Dataset containing two groups with relevance {1,3} and {2,4} respectively.
  const auto dataset = CreateToyDataset();
  std::vector<float> weights(dataset.nrow(), 1.f);

  RankingGroupsIndices index;
  index.Initialize(dataset, 0, 1);
  EXPECT_EQ(index.groups().size(), 2);

  // Perfect predictions.
  EXPECT_NEAR(index.NDCG({10, 11, 12, 13}, weights, 5), 1., 0.00001);

  // Perfect predictions again (the ranking across groups have no effect).
  EXPECT_NEAR(index.NDCG({10, 111, 12, 112}, weights, 5), 1., 0.00001);

  // Perfectly wrong predictions.
  // R> 0.7238181 = (sum((2^c(1,3)-1)/log2(seq(2)+1)) /
  // sum((2^c(3,1)-1)/log2(seq(2)+1)) +  sum((2^c(2,4)-1)/log2(seq(2)+1)) /
  // sum((2^c(4,2)-1)/log2(seq(2)+1)) )/2
  EXPECT_NEAR(index.NDCG({2, 2, 1, 1}, weights, 5), 0.723818, 0.00001);
}

TEST(GradientBoostedTrees, UpdateGradientsNDCG) {
  const auto dataset = CreateToyDataset();
  std::vector<float> weights(dataset.nrow(), 1.f);

  RankingGroupsIndices index;
  index.Initialize(dataset, 0, 1);

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  const auto loss_imp =
      NDCGLoss({}, model::proto::Task::RANKING, dataset.data_spec().columns(0));
  CHECK_OK(internal::CreateGradientDataset(dataset,
                                           /* label_col_idx= */ 0,
                                           /*hessian_splits=*/false, loss_imp,
                                           &gradient_dataset, &gradients,
                                           &predictions));

  internal::SetInitialPredictions(
      loss_imp
          .InitialPredictions(dataset,
                              /* label_col_idx =*/0, weights)
          .value(),
      dataset.nrow(), &predictions);

  utils::RandomEngine random(1234);
  CHECK_OK(loss_imp.UpdateGradients(gradient_dataset,
                                    /* label_col_idx= */ 0, predictions, &index,
                                    &gradients, &random));

  // Explanation:
  // - Element 0 is pushed down by element 2 (and in reverse).
  // - Element 1 is pushed down by element 3 (and in reverse).
  EXPECT_NEAR(gradients.front().gradient[0], -0.14509, 0.0001);
  EXPECT_NEAR(gradients.front().gradient[1], -0.13109, 0.0001);
  EXPECT_NEAR(gradients.front().gradient[2], 0.14509, 0.0001);
  EXPECT_NEAR(gradients.front().gradient[3], 0.13109, 0.0001);
}

TEST(GradientBoostedTrees, UpdateGradientsXeNDCGMart) {
  const auto dataset = CreateToyDataset();
  std::vector<float> weights(dataset.nrow(), 1.f);

  RankingGroupsIndices index;
  index.Initialize(dataset, 0, 1);

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  const auto loss_imp = CrossEntropyNDCGLoss({}, model::proto::Task::RANKING,
                                             dataset.data_spec().columns(0));
  CHECK_OK(internal::CreateGradientDataset(dataset,
                                           /* label_col_idx= */ 0,
                                           /*hessian_splits=*/false, loss_imp,
                                           &gradient_dataset, &gradients,
                                           &predictions));

  internal::SetInitialPredictions(
      loss_imp
          .InitialPredictions(dataset,
                              /* label_col_idx =*/0, weights)
          .value(),
      dataset.nrow(), &predictions);

  utils::RandomEngine random(1234);
  CHECK_OK(loss_imp.UpdateGradients(gradient_dataset,
                                    /* label_col_idx= */ 0, predictions, &index,
                                    &gradients, &random));

  // Explanation:
  // - Element 0 is pushed down by element 2 (and in reverse).
  // - Element 1 is pushed down by element 3 (and in reverse).
  EXPECT_NEAR(gradients.front().gradient[0], -0.33864, 0.0001);
  EXPECT_NEAR(gradients.front().gradient[1], -0.32854, 0.0001);
  EXPECT_NEAR(gradients.front().gradient[2], 0.33864, 0.0001);
  EXPECT_NEAR(gradients.front().gradient[3], 0.32854, 0.0001);
}

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
