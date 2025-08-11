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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_ndcg.h"

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_cross_entropy_ndcg.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

using ::testing::Bool;
using ::testing::Combine;

// Margin of error for numerical tests. Note that this is by a factor of 10
// larger than for the other loss functions.
constexpr float kTestPrecision = 0.00001f;

using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::IsEmpty;
using ::testing::Pointwise;
using ::testing::SizeIs;

absl::StatusOr<dataset::VerticalDataset> CreateToyDataset(
    bool indicator_labels) {
  dataset::VerticalDataset dataset;
  // TODO Replace by a modern function when possible.
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "LABEL" }
    columns {
      type: CATEGORICAL
      name: "GROUP"
      categorical { number_of_unique_values: 3 is_already_integerized: true }
    }
  )pb");
  RETURN_IF_ERROR(dataset.CreateColumnsFromDataspec());
  if (indicator_labels) {
    RETURN_IF_ERROR(
        dataset.AppendExampleWithStatus({{"LABEL", "1"}, {"GROUP", "1"}}));
    RETURN_IF_ERROR(
        dataset.AppendExampleWithStatus({{"LABEL", "0"}, {"GROUP", "1"}}));
    RETURN_IF_ERROR(
        dataset.AppendExampleWithStatus({{"LABEL", "0"}, {"GROUP", "2"}}));
    RETURN_IF_ERROR(
        dataset.AppendExampleWithStatus({{"LABEL", "0"}, {"GROUP", "2"}}));
    RETURN_IF_ERROR(
        dataset.AppendExampleWithStatus({{"LABEL", "1"}, {"GROUP", "2"}}));
  } else {
    RETURN_IF_ERROR(
        dataset.AppendExampleWithStatus({{"LABEL", "3"}, {"GROUP", "1"}}));
    RETURN_IF_ERROR(
        dataset.AppendExampleWithStatus({{"LABEL", "1"}, {"GROUP", "1"}}));
    RETURN_IF_ERROR(
        dataset.AppendExampleWithStatus({{"LABEL", "0"}, {"GROUP", "2"}}));
    RETURN_IF_ERROR(
        dataset.AppendExampleWithStatus({{"LABEL", "2"}, {"GROUP", "2"}}));
    RETURN_IF_ERROR(
        dataset.AppendExampleWithStatus({{"LABEL", "4"}, {"GROUP", "2"}}));
  }
  return dataset;
}

class NDCGLossTest : public testing::TestWithParam<std::tuple<bool, bool>> {};

TEST(NDCGLossTest, RankingIndexInitialization) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset(false));

  RankingGroupsIndices index;
  EXPECT_OK(index.Initialize(dataset, 0, 1));
  ASSERT_THAT(index.groups(), SizeIs(2));
  ASSERT_THAT(index.groups()[0].items, SizeIs(2));
  ASSERT_THAT(index.groups()[1].items, SizeIs(3));
  EXPECT_EQ(index.groups()[0].items[0].example_idx, 0);
  EXPECT_EQ(index.groups()[0].items[0].relevance, 3);
  EXPECT_EQ(index.groups()[0].items[1].example_idx, 1);
  EXPECT_EQ(index.groups()[0].items[1].relevance, 1);
  EXPECT_EQ(index.groups()[1].items[0].example_idx, 4);
  EXPECT_EQ(index.groups()[1].items[0].relevance, 4);
  EXPECT_EQ(index.groups()[1].items[1].example_idx, 3);
  EXPECT_EQ(index.groups()[1].items[1].relevance, 2);
  EXPECT_EQ(index.groups()[1].items[2].example_idx, 2);
  EXPECT_EQ(index.groups()[1].items[2].relevance, 0);
}

TEST_P(NDCGLossTest, InitialPredictions) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset(false));
  const bool weighted = std::get<0>(GetParam());
  std::vector<float> weights;
  if (weighted) {
    weights = {1.f, 2.f, 3.f, 4.f, 5.f};
  }

  const NDCGLoss loss_imp(
      {{}, {}, model::proto::Task::RANKING, dataset.data_spec().columns(0)}, 5);
  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> init_pred,
      loss_imp.InitialPredictions(dataset, /* label_col_idx= */ 0, weights));
  // Initial predictions for NDCG loss are always 0.
  EXPECT_THAT(init_pred, ElementsAre(0.f));
}

TEST_P(NDCGLossTest, UpdateGradientsArbitraryLabels) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset(false));
  const bool threaded = std::get<1>(GetParam());

  NDCGLoss::Cache cache;
  auto& index = cache.ranking_index;
  EXPECT_OK(
      index.Initialize(dataset, /*label_col_idx=*/0, /*group_col_idx=*/1));
  EXPECT_THAT(index.groups(), SizeIs(2));

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  const NDCGLoss loss_imp(
      {{}, {}, model::proto::Task::RANKING, dataset.data_spec().columns(0)}, 5);
  ASSERT_OK(internal::CreateGradientDataset(dataset,
                                            /* label_col_idx=*/0, loss_imp,
                                            &gradient_dataset, &gradients,
                                            &predictions));
  predictions = {0.f, 1.f, 1.f, 0.5f, 0.f};

  utils::RandomEngine random(1234);
  if (threaded) {
    utils::concurrency::ThreadPool thread_pool(
        6, {.name_prefix = std::string("")});
    ASSERT_OK(loss_imp.UpdateGradients(gradient_dataset,
                                       /* label_col_idx= */ 0, predictions,
                                       &cache, &gradients, &random,
                                       &thread_pool));
  } else {
    ASSERT_OK(loss_imp.UpdateGradients(gradient_dataset,
                                       /* label_col_idx= */ 0, predictions,
                                       &cache, &gradients, &random));
  }
  ASSERT_THAT(gradients, Not(IsEmpty()));
  const std::vector<float>& gradient = gradients.front().gradient;
  EXPECT_THAT(gradient, ElementsAre(FloatNear(0.212146f, kTestPrecision),
                                    FloatNear(-0.212146f, kTestPrecision),
                                    FloatNear(-0.365370f, kTestPrecision),
                                    FloatNear(-0.017095f, kTestPrecision),
                                    FloatNear(0.382466f, kTestPrecision)));
}

TEST(NDCGLossTest, UpdateGradientsIndicatorLabels) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset(true));
  NDCGLoss::Cache cache;
  auto& index = cache.ranking_index;
  EXPECT_OK(
      index.Initialize(dataset, /*label_col_idx=*/0, /*group_col_idx=*/1));
  EXPECT_THAT(index.groups(), SizeIs(2));

  const NDCGLoss loss_imp_indicator(
      {{}, {}, model::proto::Task::RANKING, dataset.data_spec().columns(0)}, 5);
  proto::GradientBoostedTreesTrainingConfig gbt_config_no_indicator;
  gbt_config_no_indicator.mutable_internal()
      ->set_enable_ndcg_indicator_labels_optimization(false);
  const NDCGLoss loss_imp_no_indicator({{},
                                        gbt_config_no_indicator,
                                        model::proto::Task::RANKING,
                                        dataset.data_spec().columns(0)},
                                       5);
  dataset::VerticalDataset gradient_dataset_indicator;
  dataset::VerticalDataset gradient_dataset_no_indicator;
  std::vector<GradientData> gradients_indicator;
  std::vector<GradientData> gradients_no_indicator;

  {
    std::vector<float> predictions = {1.f, 0.f, 1.f, 0.5f, 0.f};
    ASSERT_OK(internal::CreateGradientDataset(
        dataset,
        /* label_col_idx=*/0, loss_imp_indicator, &gradient_dataset_indicator,
        &gradients_indicator, &predictions));
    utils::RandomEngine random(1234);
    CHECK_OK(loss_imp_indicator.UpdateGradients(gradient_dataset_indicator,
                                                /* label_col_idx=*/0,
                                                predictions, &cache,
                                                &gradients_indicator, &random));
  }
  {
    std::vector<float> predictions = {1.f, 0.f, 1.f, 0.5f, 0.f};
    ASSERT_OK(internal::CreateGradientDataset(
        dataset,
        /* label_col_idx=*/0, loss_imp_no_indicator,
        &gradient_dataset_no_indicator, &gradients_no_indicator, &predictions));
    utils::RandomEngine random(1234);
    CHECK_OK(loss_imp_no_indicator.UpdateGradients(
        gradient_dataset_no_indicator,
        /* label_col_idx=*/0, predictions, &cache, &gradients_no_indicator,
        &random));
  }

  ASSERT_THAT(gradients_indicator, Not(IsEmpty()));
  ASSERT_THAT(gradients_no_indicator, Not(IsEmpty()));
  ASSERT_THAT(gradients_no_indicator[0].gradient, SizeIs(5));
  EXPECT_THAT(gradients_indicator.front().gradient,
              Pointwise(FloatNear(kTestPrecision),
                        gradients_no_indicator.front().gradient));
}

TEST(NDCGLossTest, UpdateGradientsXeNDCGMart) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset(false));
  std::vector<float> weights;

  CrossEntropyNDCGLoss::Cache cache;
  auto& index = cache.ranking_index;
  EXPECT_OK(index.Initialize(dataset, 0, 1));
  EXPECT_THAT(index.groups(), SizeIs(2));

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  const CrossEntropyNDCGLoss loss_imp(
      {{}, {}, model::proto::Task::RANKING, dataset.data_spec().columns(0)});
  ASSERT_OK(internal::CreateGradientDataset(dataset,
                                            /* label_col_idx= */ 0, loss_imp,
                                            &gradient_dataset, &gradients,
                                            &predictions));
  predictions = {0.f, 1.f, 1.f, 0.5f, 0.f};

  utils::RandomEngine random(1234);
  ASSERT_OK(loss_imp.UpdateGradients(gradient_dataset,
                                     /* label_col_idx= */ 0, predictions,
                                     &cache, &gradients, &random));

  ASSERT_THAT(gradients, Not(IsEmpty()));
  const std::vector<float>& gradient = gradients.front().gradient;
  EXPECT_THAT(gradient, ElementsAre(FloatNear(0.569704592, kTestPrecision),
                                    FloatNear(-0.56970495, kTestPrecision),
                                    FloatNear(-0.377651036, kTestPrecision),
                                    FloatNear(-0.111444674, kTestPrecision),
                                    FloatNear(0.489095628, kTestPrecision)));
}

TEST_P(NDCGLossTest, ComputeRankingLossPerfectlyWrongPredictions) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset(false));
  const bool weighted = std::get<0>(GetParam());
  std::vector<float> weights;
  if (weighted) {
    weights = {1.f, 1.f, 3.f, 3.f, 3.f};
  }

  // Perfectly wrong predictions.
  std::vector<float> predictions = {0.f, 1.f, 1.f, 0.5f, 0.f};
  const NDCGLoss loss_imp(
      {{}, {}, model::proto::Task::RANKING, dataset.data_spec().columns(0)}, 5);
  NDCGLoss::Cache cache;
  auto& index = cache.ranking_index;
  EXPECT_OK(
      index.Initialize(dataset, /*label_col_idx=*/0, /*group_col_idx=*/1));
  ASSERT_OK_AND_ASSIGN(
      LossResults loss_results,
      loss_imp.Loss(dataset,
                    /* label_col_idx=*/0, predictions, weights, &cache));
  double expected_ndcg;
  if (weighted) {
    expected_ndcg = ((1. + 7 / log2(3)) / (7. + 1. / log2(3)) +
                     3. * (3. / log2(3) + 15. / 2.) / (15 + 3. / log2(3))) /
                    4.;
  } else {
    expected_ndcg = ((1. + 7 / log2(3)) / (7. + 1. / log2(3)) +
                     (3. / log2(3) + 15. / 2.) / (15 + 3. / log2(3))) /
                    2.;
  }
  EXPECT_NEAR(loss_results.loss, -expected_ndcg, kTestPrecision);
  EXPECT_THAT(loss_results.secondary_metrics,
              ElementsAre(FloatNear(expected_ndcg, kTestPrecision)));
}

TEST_P(NDCGLossTest, ComputeRankingLossPerfectlyWrongPredictionsTruncation1) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset(false));
  const bool weighted = std::get<0>(GetParam());
  std::vector<float> weights;
  if (weighted) {
    weights = {1.f, 1.f, 3.f, 3.f, 3.f};
  }

  // Perfectly wrong predictions.
  std::vector<float> predictions = {0.f, 1.f, 1.f, 0.5f, 0.f};
  proto::GradientBoostedTreesTrainingConfig gbt_config;
  gbt_config.mutable_lambda_mart_ndcg()->set_ndcg_truncation(1);
  const NDCGLoss loss_imp({{},
                           gbt_config,
                           model::proto::Task::RANKING,
                           dataset.data_spec().columns(0)},
                          1);
  NDCGLoss::Cache cache;
  auto& index = cache.ranking_index;
  EXPECT_OK(index.Initialize(dataset, 0, 1));
  ASSERT_OK_AND_ASSIGN(
      LossResults loss_results,
      loss_imp.Loss(dataset,
                    /* label_col_idx= */ 0, predictions, weights, &cache));
  double expected_ndcg;
  if (weighted) {
    expected_ndcg = (1. / 7.) / 4.;
  } else {
    expected_ndcg = (1. / 7.) / 2.;
  }
  EXPECT_NEAR(loss_results.loss, -expected_ndcg, kTestPrecision);
  EXPECT_THAT(loss_results.secondary_metrics,
              ElementsAre(FloatNear(expected_ndcg, kTestPrecision)));
}

TEST_P(NDCGLossTest, ComputeRankingLossPerfectPredictions) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset(false));
  const bool weighted = std::get<0>(GetParam());
  std::vector<float> weights;
  if (weighted) {
    weights = {1.f, 1.f, 3.f, 3.f, 3.f};
  }

  // Perfect predictions.
  std::vector<float> predictions = {1.f, 0.f, 0.f, 0.5f, 1.f};
  const NDCGLoss loss_imp(
      {{}, {}, model::proto::Task::RANKING, dataset.data_spec().columns(0)}, 5);
  NDCGLoss::Cache cache;
  auto& index = cache.ranking_index;
  EXPECT_OK(index.Initialize(dataset, 0, 1));
  ASSERT_OK_AND_ASSIGN(
      LossResults loss_results,
      loss_imp.Loss(dataset,
                    /* label_col_idx= */ 0, predictions, weights, &cache));
  EXPECT_NEAR(loss_results.loss, -1.f, kTestPrecision);
  EXPECT_THAT(loss_results.secondary_metrics,
              ElementsAre(FloatNear(1.f, kTestPrecision)));
}

TEST_P(NDCGLossTest, ComputeRankingLossPerfectGroupedPredictions) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset(false));
  const bool weighted = std::get<0>(GetParam());
  std::vector<float> weights;
  if (weighted) {
    weights = {1.f, 1.f, 3.f, 3.f, 3.f};
  }

  // Perfect predictions again (the ranking across groups has no effect).
  std::vector<float> predictions = {0.6f, 0.4f, 0.f, 0.5f, 1.f};
  const NDCGLoss loss_imp(
      {{}, {}, model::proto::Task::RANKING, dataset.data_spec().columns(0)}, 5);
  NDCGLoss::Cache cache;
  auto& index = cache.ranking_index;
  EXPECT_OK(index.Initialize(dataset, 0, 1));
  ASSERT_OK_AND_ASSIGN(
      LossResults loss_results,
      loss_imp.Loss(dataset,
                    /* label_col_idx= */ 0, predictions, weights, &cache));
  EXPECT_NEAR(loss_results.loss, -1.f, kTestPrecision);
  EXPECT_THAT(loss_results.secondary_metrics,
              ElementsAre(FloatNear(1.f, kTestPrecision)));
}

TEST(NDCGLossTest, SecondaryMetricName) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset(false));
  const auto loss_imp = NDCGLoss(
      {{}, {}, model::proto::Task::RANKING, dataset.data_spec().columns(0)}, 5);
  EXPECT_THAT(loss_imp.SecondaryMetricNames(), ElementsAre("NDCG@5"));
}

TEST(NDCGLossTest, SecondaryMetricNameTrucation10) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset(false));
  proto::GradientBoostedTreesTrainingConfig gbt_config;
  gbt_config.mutable_lambda_mart_ndcg()->set_ndcg_truncation(10);
  const auto loss_imp = NDCGLoss({{},
                                  gbt_config,
                                  model::proto::Task::RANKING,
                                  dataset.data_spec().columns(0)},
                                 10);
  EXPECT_THAT(loss_imp.SecondaryMetricNames(), ElementsAre("NDCG@10"));
}

TEST(NDCGLossTest, InvalidTruncation) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset(false));
  proto::GradientBoostedTreesTrainingConfig gbt_config;
  gbt_config.mutable_lambda_mart_ndcg()->set_ndcg_truncation(0);
  const auto loss_imp =
      NDCGLoss::RegistrationCreate({{},
                                    gbt_config,
                                    model::proto::Task::RANKING,
                                    dataset.data_spec().columns(0)});
  EXPECT_EQ(loss_imp.status().code(), absl::StatusCode::kInvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(NDCGLossTestSuite, NDCGLossTest,
                         Combine(Bool(), Bool()));

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
