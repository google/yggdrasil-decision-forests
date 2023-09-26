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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_mean_square_error.h"

#include "gmock/gmock.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

using ::testing::Bool;
using ::testing::Combine;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;

// Margin of error for numerical tests.
constexpr float kTestPrecision = 0.000001f;

absl::StatusOr<dataset::VerticalDataset> CreateToyDataset() {
  dataset::VerticalDataset dataset;
  // TODO Replace by a modern function when possible.
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "a" }
    columns {
      type: CATEGORICAL
      name: "b"
      categorical { number_of_unique_values: 3 is_already_integerized: true }
    }
  )pb");
  RETURN_IF_ERROR(dataset.CreateColumnsFromDataspec());
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "1"}, {"b", "1"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "2"}, {"b", "2"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "3"}, {"b", "1"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "4"}, {"b", "2"}}));
  return dataset;
}

class MeanSquareErrorLossTest
    : public testing::TestWithParam<std::tuple<bool, bool>> {};

TEST_P(MeanSquareErrorLossTest, InitialPredictions) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const bool weighted = std::get<0>(GetParam());
  std::vector<float> weights;
  if (weighted) {
    weights = {2.f, 4.f, 6.f, 8.f};
  }

  const MeanSquaredErrorLoss loss_imp({}, model::proto::Task::REGRESSION,
                                      dataset.data_spec().columns(0));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> init_pred,
      loss_imp.InitialPredictions(dataset, /* label_col_idx= */ 0, weights));
  if (weighted) {
    EXPECT_THAT(init_pred,
                ElementsAre((2.f + 8.f + 18.f + 32.f) / 20.f));  // Mean.
  } else {
    EXPECT_THAT(init_pred,
                ElementsAre((1.f + 2.f + 3.f + 4.f) / 4.f));  // Mean.
  }
}

TEST_P(MeanSquareErrorLossTest, UpdateGradients) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const bool weighted = std::get<0>(GetParam());
  std::vector<float> weights;
  if (weighted) {
    weights = {2.f, 4.f, 6.f, 8.f};
  }

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  const MeanSquaredErrorLoss loss_imp({}, model::proto::Task::REGRESSION,
                                      dataset.data_spec().columns(0));
  ASSERT_OK(internal::CreateGradientDataset(dataset,
                                            /* label_col_idx= */ 0,
                                            /*hessian_splits=*/false, loss_imp,
                                            &gradient_dataset, &gradients,
                                            &predictions));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> loss_initial_predictions,
      loss_imp.InitialPredictions(dataset,
                                  /* label_col_idx =*/0, weights));
  internal::SetInitialPredictions(loss_initial_predictions, dataset.nrow(),
                                  &predictions);

  utils::RandomEngine random(1234);
  ASSERT_OK(loss_imp.UpdateGradients(gradient_dataset,
                                     /* label_col_idx= */ 0, predictions,
                                     /*ranking_index=*/nullptr, &gradients,
                                     &random));

  ASSERT_THAT(gradients, Not(IsEmpty()));
  if (weighted) {
    EXPECT_THAT(gradients.front().gradient,
                ElementsAre(1.f - 3.f, 2.f - 3.f, 3.f - 3.f, 4.f - 3.f));
  } else {
    EXPECT_THAT(gradients.front().gradient,
                ElementsAre(1.f - 2.5f, 2.f - 2.5f, 3.f - 2.5f, 4.f - 2.5f));
  }
}

TEST_P(MeanSquareErrorLossTest, ComputeClassificationLoss) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const bool weighted = std::get<0>(GetParam());
  const bool threaded = std::get<1>(GetParam());
  std::vector<float> weights;
  if (weighted) {
    weights = {2.f, 4.f, 6.f, 8.f};
  }

  std::vector<float> predictions = {0.f, 0.f, 0.f, 0.f};
  const MeanSquaredErrorLoss loss_imp({}, model::proto::Task::REGRESSION,
                                      dataset.data_spec().columns(0));
  LossResults loss_results;
  if (threaded) {
    utils::concurrency::ThreadPool thread_pool("", 4);
    thread_pool.StartWorkers();
    ASSERT_OK_AND_ASSIGN(loss_results,
                         loss_imp.Loss(dataset,
                                       /* label_col_idx= */ 0, predictions,
                                       weights, nullptr, &thread_pool));
  } else {
    ASSERT_OK_AND_ASSIGN(
        loss_results,
        loss_imp.Loss(dataset,
                      /* label_col_idx= */ 0, predictions, weights, nullptr));
  }
  if (weighted) {
    EXPECT_NEAR(loss_results.loss, std::sqrt(200. / 20.), kTestPrecision);
    // For classification, the only secondary metric is also RMSE.
    EXPECT_THAT(loss_results.secondary_metrics,
                ElementsAre(FloatNear(std::sqrt(200. / 20.), kTestPrecision)));
  } else {
    EXPECT_NEAR(loss_results.loss, std::sqrt(30. / 4.), kTestPrecision);
    // For classification, the only secondary metric is also RMSE.
    EXPECT_THAT(loss_results.secondary_metrics,
                ElementsAre(FloatNear(std::sqrt(30. / 4.), kTestPrecision)));
  }
}

TEST_P(MeanSquareErrorLossTest, ComputeRankingLoss) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const bool weighted = std::get<0>(GetParam());
  const bool threaded = std::get<1>(GetParam());
  std::vector<float> weights;
  if (weighted) {
    weights = {2.f, 4.f, 6.f, 8.f};
  }

  std::vector<float> predictions = {0.f, 0.f, 0.f, 0.f};
  const MeanSquaredErrorLoss loss_imp({}, model::proto::Task::RANKING,
                                      dataset.data_spec().columns(0));
  RankingGroupsIndices index;
  EXPECT_OK(index.Initialize(dataset, 0, 1));
  LossResults loss_results;
  if (threaded) {
    utils::concurrency::ThreadPool thread_pool("", 4);
    thread_pool.StartWorkers();
    ASSERT_OK_AND_ASSIGN(loss_results,
                         loss_imp.Loss(dataset,
                                       /* label_col_idx= */ 0, predictions,
                                       weights, &index, &thread_pool));
  } else {
    ASSERT_OK_AND_ASSIGN(
        loss_results,
        loss_imp.Loss(dataset,
                      /* label_col_idx= */ 0, predictions, weights, &index));
  }
  if (weighted) {
    EXPECT_NEAR(loss_results.loss, std::sqrt(200. / 20.), kTestPrecision);
    //  For ranking, first secondary metric is RMSE, second secondary metric is
    //  NDCG@5.
    EXPECT_THAT(loss_results.secondary_metrics,
                ElementsAre(FloatNear(std::sqrt(200. / 20.), kTestPrecision),
                            FloatNear(0.86291, kTestPrecision)));
  } else {
    EXPECT_NEAR(loss_results.loss, std::sqrt(30. / 4.), kTestPrecision);
    //  For ranking, first secondary metric is RMSE, second secondary metric is
    //  NDCG@5.
    EXPECT_THAT(loss_results.secondary_metrics,
                ElementsAre(FloatNear(std::sqrt(30. / 4.), kTestPrecision),
                            FloatNear(0.861909, kTestPrecision)));
  }
}

TEST(MeanSquareErrorLossTest, SecondaryMetricNamesClassification) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const MeanSquaredErrorLoss loss_imp_classification(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  EXPECT_THAT(loss_imp_classification.SecondaryMetricNames(),
              ElementsAre("rmse"));
}

TEST(MeanSquareErrorLossTest, SecondaryMetricNamesRanking) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const MeanSquaredErrorLoss loss_imp_ranking({}, model::proto::Task::RANKING,
                                              dataset.data_spec().columns(1));
  EXPECT_THAT(loss_imp_ranking.SecondaryMetricNames(),
              ElementsAre("rmse", "NDCG@5"));
}

INSTANTIATE_TEST_SUITE_P(MeanSquareErrorLossTestWithWeightsAndThreads,
                         MeanSquareErrorLossTest, Combine(Bool(), Bool()));

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
