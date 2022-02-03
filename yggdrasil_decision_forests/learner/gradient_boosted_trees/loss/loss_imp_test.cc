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

// TODO(gbm): Split the test by implementation.
// Note: Do not add anymore loss test in this file. Instead, create a loss
// specific test.

#include "gmock/gmock.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_binary_focal.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_binomial.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_cross_entropy_ndcg.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_mean_square_error.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_multinomial.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_ndcg.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

using testing::ElementsAre;
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
  dataset.AppendExample({{"a", "1"}, {"b", "1"}});
  dataset.AppendExample({{"a", "2"}, {"b", "2"}});
  dataset.AppendExample({{"a", "3"}, {"b", "1"}});
  dataset.AppendExample({{"a", "4"}, {"b", "2"}});
  return dataset;
}

TEST(GradientBoostedTrees, InitialPredictionsBinomialLogLikelihood) {
  const auto dataset = CreateToyDataset();
  std::vector<float> weights{1.f, 1.f, 1.f, 1.f};
  const auto loss_imp = BinomialLogLikelihoodLoss(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  const auto init_pred =
      loss_imp.InitialPredictions(dataset, /* label_col_idx= */ 1, weights)
          .value();
  EXPECT_EQ(init_pred.size(), 1);
  EXPECT_EQ(init_pred[0], 0.f);
}

TEST(GradientBoostedTrees, InitialPredictionsSquareError) {
  const auto dataset = CreateToyDataset();
  std::vector<float> weights{1.f, 1.f, 1.f, 1.f};
  const auto loss_imp = MeanSquaredErrorLoss({}, model::proto::Task::REGRESSION,
                                             dataset.data_spec().columns(0));
  const auto init_pred =
      loss_imp.InitialPredictions(dataset, /* label_col_idx= */ 0, weights)
          .value();
  EXPECT_EQ(init_pred.size(), 1);
  EXPECT_EQ(init_pred[0], (1.f + 2.f + 3.f + 4.f) / 4.f);  // Mean.
}

TEST(GradientBoostedTrees, InitialPredictionsMultinomialLogLikelihood) {
  const auto dataset = CreateToyDataset();
  std::vector<float> weights{1.f, 1.f, 1.f, 1.f};
  const auto loss_imp = MultinomialLogLikelihoodLoss(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  const auto init_pred =
      loss_imp.InitialPredictions(dataset, /* label_col_idx= */ 1, weights)
          .value();
  EXPECT_EQ(init_pred.size(), 2);
  EXPECT_EQ(init_pred, std::vector<float>({0.f, 0.f}));
}

TEST(GradientBoostedTrees, UpdateGradientsBinomialLogLikelihood) {
  const auto dataset = CreateToyDataset();
  std::vector<float> weights(dataset.nrow(), 1.f);

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  const auto loss_imp = BinomialLogLikelihoodLoss(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
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

  EXPECT_THAT(gradients.front().gradient, ElementsAre(-0.5f, 0.5, -0.5f, 0.5f));
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

  // TODO(gbm): Implement and use "AllElementsNear" matcher.
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

TEST(GradientBoostedTrees, UpdateGradientsSquaredError) {
  const auto dataset = CreateToyDataset();
  std::vector<float> weights(dataset.nrow(), 1.f);

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  const auto loss_imp = MeanSquaredErrorLoss({}, model::proto::Task::REGRESSION,
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
                                    /* label_col_idx= */ 0, predictions,
                                    /*ranking_index=*/nullptr, &gradients,
                                    &random));

  EXPECT_THAT(gradients.front().gradient,
              ElementsAre(1.f - 2.5f, 2.f - 2.5f, 3.f - 2.5f, 4.f - 2.5f));
}

TEST(GradientBoostedTrees, SetLabelDistributionBinomialLogLikelihood) {
  const auto dataset = CreateToyDataset();
  std::vector<float> weights(dataset.nrow(), 1.f);

  std::vector<GradientData> gradients;

  const auto loss_imp = BinomialLogLikelihoodLoss(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  dataset::VerticalDataset gradient_dataset;
  CHECK_OK(internal::CreateGradientDataset(dataset,
                                           /* label_col_idx= */ 1,
                                           /*hessian_splits=*/false, loss_imp,
                                           &gradient_dataset, &gradients,
                                           nullptr));
  EXPECT_EQ(gradients.size(), 1);

  std::vector<dataset::VerticalDataset::row_t> selected_examples{0, 1, 2, 3};
  std::vector<float> predictions(dataset.nrow(), 0.f);

  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(2);  // Gradient column.
  proto::GradientBoostedTreesTrainingConfig gbt_config;
  gbt_config.set_shrinkage(1.f);

  decision_tree::NodeWithChildren node;
  loss_imp.SetLeaf(gradient_dataset, selected_examples, weights, config,
                   config_link, predictions,
                   /* label_col_idx= */ 1, &node);

  // Node output: Half positive, half negative.
  // (2*(1-0.5)+2*(0-0.5))/( 4*0.5*(1-0.5) ) => 0
  EXPECT_EQ(node.node().regressor().top_value(), 0);
  // Distribution of the gradients:
  EXPECT_EQ(node.node().regressor().distribution().sum(), 0);
  EXPECT_EQ(node.node().regressor().distribution().sum_squares(), 0);
  // Same as the number of examples in the dataset.
  EXPECT_EQ(node.node().regressor().distribution().count(), 4.);
}

TEST(GradientBoostedTrees, SetLabelDistributionSquaredError) {
  const auto dataset = CreateToyDataset();
  std::vector<float> weights(dataset.nrow(), 1.f);

  proto::GradientBoostedTreesTrainingConfig gbt_config;
  gbt_config.set_shrinkage(1.f);
  std::vector<GradientData> gradients;
  dataset::VerticalDataset gradient_dataset;
  const auto loss_imp =
      MeanSquaredErrorLoss(gbt_config, model::proto::Task::REGRESSION,
                           dataset.data_spec().columns(0));
  CHECK_OK(internal::CreateGradientDataset(dataset,
                                           /* label_col_idx= */ 0,
                                           /*hessian_splits=*/false, loss_imp,
                                           &gradient_dataset, &gradients,
                                           nullptr));
  EXPECT_EQ(gradients.size(), 1);

  std::vector<dataset::VerticalDataset::row_t> selected_examples{0, 1, 2, 3};
  std::vector<float> predictions(dataset.nrow(), 0.f);

  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(2);  // Gradient column.

  decision_tree::NodeWithChildren node;
  loss_imp.SetLeaf(gradient_dataset, selected_examples, weights, config,
                   config_link, predictions,
                   /* label_col_idx= */ 0, &node);

  EXPECT_EQ(node.node().regressor().top_value(), 2.5f);  // Mean of the labels.
  // Distribution of the gradients:
  EXPECT_EQ(node.node().regressor().distribution().sum(), 0);
  EXPECT_EQ(node.node().regressor().distribution().sum_squares(), 0);
  // Same as the number of examples in the dataset.
  EXPECT_EQ(node.node().regressor().distribution().count(), 4.);
}

TEST(GradientBoostedTrees, ComputeLoss) {
  const auto dataset = CreateToyDataset();
  std::vector<float> weights(dataset.nrow(), 1.f);
  std::vector<float> predictions(dataset.nrow(), 0.f);
  const auto loss_imp = BinomialLogLikelihoodLoss(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  float loss_value;
  std::vector<float> secondary_metric;
  CHECK_OK(loss_imp.Loss(dataset,
                         /* label_col_idx= */ 1, predictions, weights, nullptr,
                         &loss_value, &secondary_metric));

  EXPECT_NEAR(loss_value, 2 * std::log(2), 0.0001);
  EXPECT_EQ(secondary_metric.size(), 1);
  EXPECT_NEAR(secondary_metric[0], 0.5f, 0.0001);
}

TEST(GradientBoostedTrees, SecondaryMetricName) {
  const auto dataset = CreateToyDataset();
  const auto loss_imp = BinomialLogLikelihoodLoss(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  EXPECT_THAT(loss_imp.SecondaryMetricNames(), ElementsAre("accuracy"));
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
