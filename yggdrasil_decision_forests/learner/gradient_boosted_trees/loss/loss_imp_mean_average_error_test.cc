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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_mean_average_error.h"

#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"  // IWYU pragma: keep
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"  // IWYU pragma: keep
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::model::gradient_boosted_trees {

namespace {

using ::testing::Combine;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Values;

// Margin of error for numerical tests.
constexpr float kTestPrecision = 0.000001f;

absl::StatusOr<dataset::VerticalDataset> CreateToyDataset(
    const bool even_num_examples = true) {
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
  if (even_num_examples) {
    RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "4"}, {"b", "2"}}));
  }
  return dataset;
}

std::vector<float> CreateToyWeights(const bool weighted) {
  if (weighted) {
    return {2.f, 4.f, 6.f, 8.f};
  } else {
    return {};
  }
}

enum class UseWeights : bool {
  kNo,
  kYes,
};

enum class UseMultithreading : bool {
  kNo,
  kYes,
};

class MeanAverageErrorLossWeightAndThreadingTest
    : public testing::TestWithParam<std::tuple<UseWeights, UseMultithreading>> {
 protected:
  void SetUp() override {
    const bool threaded = std::get<1>(GetParam()) == UseMultithreading::kYes;
    if (threaded) {
      thread_pool_ = std::make_unique<utils::concurrency::ThreadPool>("", 4);
      thread_pool_->StartWorkers();
    }
  }

  void TearDown() override { thread_pool_.reset(); }

  // The thread pool is only set if "UseMultithreading=kYes".
  std::unique_ptr<utils::concurrency::ThreadPool> thread_pool_;
};

class MeanAverageErrorLossWeightTest
    : public testing::TestWithParam<UseWeights> {};

TEST_P(MeanAverageErrorLossWeightTest, InitialPredictions) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const bool weighted = GetParam() == UseWeights::kYes;
  const std::vector<float> weights = CreateToyWeights(weighted);

  const MeanAverageErrorLoss loss_imp(/*gbt_config=*/{},
                                      model::proto::Task::REGRESSION,
                                      dataset.data_spec().columns(0));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> init_pred,
      loss_imp.InitialPredictions(dataset, /* label_col_idx= */ 0, weights));
  if (weighted) {
    EXPECT_THAT(init_pred, ElementsAre(4.f));
  } else {
    EXPECT_THAT(init_pred, ElementsAre(2.5f));
  }
}

TEST(MeanAverageErrorLossTestNonWeighted, InitialPredictionsOdd) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset(false));

  const MeanAverageErrorLoss loss_imp(/*gbt_config=*/{},
                                      model::proto::Task::REGRESSION,
                                      dataset.data_spec().columns(0));
  EXPECT_THAT(
      loss_imp.InitialPredictions(dataset, /* label_col_idx= */ 0, {}).value(),
      ElementsAre(2.f));
}

TEST_P(MeanAverageErrorLossWeightAndThreadingTest, UpdateGradients) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const bool weighted = std::get<0>(GetParam()) == UseWeights::kYes;

  const std::vector<float> weights = CreateToyWeights(weighted);

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  const MeanAverageErrorLoss loss_imp(/*gbt_config=*/{},
                                      model::proto::Task::REGRESSION,
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
                                     &random, thread_pool_.get()));

  ASSERT_THAT(gradients, Not(IsEmpty()));
  if (weighted) {
    EXPECT_THAT(gradients.front().gradient, ElementsAre(-1, -1, -1, 1));
  } else {
    EXPECT_THAT(gradients.front().gradient, ElementsAre(-1, -1, 1, 1));
  }
}

TEST_P(MeanAverageErrorLossWeightAndThreadingTest, ComputeLoss) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const bool weighted = std::get<0>(GetParam()) == UseWeights::kYes;
  const std::vector<float> weights = CreateToyWeights(weighted);

  const std::vector<float> predictions(4, 0.f);
  const MeanAverageErrorLoss loss_imp(/*gbt_config=*/{},
                                      model::proto::Task::REGRESSION,
                                      dataset.data_spec().columns(0));
  LossResults loss_results;
  ASSERT_OK_AND_ASSIGN(
      loss_results, loss_imp.Loss(dataset,
                                  /* label_col_idx= */ 0, predictions, weights,
                                  nullptr, thread_pool_.get()));

  if (weighted) {
    // MAE = \sum (abs(prediction_i - label_i) * weight_i) / \sum weight_i
    const float expected_mae =
        (1. * 2. + 2. * 4. + 3. * 6. + 4. * 8.) / (2 + 4 + 6 + 8);
    // RMSE = sqrt(\sum ((prediction_i - label_i)^2 * weight_i) / \sum weight_i)
    const float expected_rmse =
        std::sqrt((1. * 2. + 2. * 2. * 4. + 3. * 3. * 6. + 4. * 4. * 8.) /
                  (2 + 4 + 6 + 8));
    EXPECT_NEAR(loss_results.loss, expected_mae, kTestPrecision);
    EXPECT_THAT(loss_results.secondary_metrics,
                ElementsAre(FloatNear(expected_mae, kTestPrecision),
                            FloatNear(expected_rmse, kTestPrecision)));
  } else {
    // MAE = \sum abs(prediction_i - label_i) / num_examples
    const float expected_mae = (1. + 2. + 3. + 4.) / 4;
    // RMSE = sqrt(\sum (prediction_i - label_i)^2 / num_examples)
    const float expected_rmse =
        std::sqrt((1. + 2. * 2. + 3. * 3. + 4. * 4.) / 4);
    EXPECT_NEAR(loss_results.loss, expected_mae, kTestPrecision);
    EXPECT_THAT(loss_results.secondary_metrics,
                ElementsAre(FloatNear(expected_mae, kTestPrecision),
                            FloatNear(expected_rmse, kTestPrecision)));
  }
}

TEST(MeanAverageErrorLossTest, SecondaryMetricNames) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const MeanAverageErrorLoss loss_imp(/*gbt_config=*/{},
                                      model::proto::Task::REGRESSION,
                                      dataset.data_spec().columns(1));
  EXPECT_THAT(loss_imp.SecondaryMetricNames(), ElementsAre("mae", "rmse"));
}

INSTANTIATE_TEST_SUITE_P(MeanAverageErrorLossWeightAndThreadingTestWithValues,
                         MeanAverageErrorLossWeightAndThreadingTest,
                         Combine(Values(UseWeights::kNo, UseWeights::kYes),
                                 Values(UseMultithreading::kNo,
                                        UseMultithreading::kYes)));

INSTANTIATE_TEST_SUITE_P(MeanAverageErrorLossWeightTestWithvalues,
                         MeanAverageErrorLossWeightTest,
                         Values(UseWeights::kNo, UseWeights::kYes));

}  // namespace
}  // namespace yggdrasil_decision_forests::model::gradient_boosted_trees
