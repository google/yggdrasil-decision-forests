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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_poisson.h"

#include <cmath>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/random.h"
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

class PoissonLossTest : public testing::TestWithParam<std::tuple<bool, bool>> {
};

TEST(PoissonLossTest, LossStatusRegression) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const PoissonLoss loss_imp({}, model::proto::Task::REGRESSION,
                             dataset.data_spec().columns(0));
  EXPECT_OK(loss_imp.Status());
}

TEST(PoissonLossTest, LossStatusClassification) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const PoissonLoss loss_imp({}, model::proto::Task::CLASSIFICATION,
                             dataset.data_spec().columns(0));
  EXPECT_FALSE(loss_imp.Status().ok());
}

TEST(PoissonLossTest, LossStatusRanking) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const PoissonLoss loss_imp({}, model::proto::Task::RANKING,
                             dataset.data_spec().columns(0));
  EXPECT_FALSE(loss_imp.Status().ok());
}

TEST_P(PoissonLossTest, InitialPredictionsClassic) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const bool weighted = std::get<0>(GetParam());
  std::vector<float> weights;
  if (weighted) {
    weights = {2.f, 4.f, 6.f, 8.f};
  }

  const PoissonLoss loss_imp({}, model::proto::Task::REGRESSION,
                             dataset.data_spec().columns(0));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> init_pred,
      loss_imp.InitialPredictions(dataset, /* label_col_idx= */ 0, weights));
  if (weighted) {
    float log_mean = std::log((2. + 8. + 18. + 32.) / 20.);
    EXPECT_THAT(init_pred, ElementsAre(FloatNear(log_mean, kTestPrecision)));
  } else {
    float log_mean = std::log((1.f + 2.f + 3.f + 4.f) / 4.f);
    EXPECT_THAT(init_pred, ElementsAre(FloatNear(log_mean, kTestPrecision)));
  }
}

TEST(PoissonLossTest, InitialPredictionsLabelStatistics) {
  decision_tree::proto::LabelStatistics label_statistics;
  auto* labels = label_statistics.mutable_regression()->mutable_labels();
  labels->set_count(4);
  labels->set_sum(10);
  labels->set_sum_squares(30);

  dataset::proto::Column column_spec = PARSE_TEST_PROTO(R"pb(
    type: NUMERICAL
    name: "a"
  )pb");
  const PoissonLoss loss_imp({}, model::proto::Task::REGRESSION, column_spec);
  ASSERT_OK_AND_ASSIGN(const std::vector<float> init_pred,
                       loss_imp.InitialPredictions(label_statistics));

  float log_mean = std::log((1.f + 2.f + 3.f + 4.f) / 4.f);
  EXPECT_THAT(init_pred, ElementsAre(FloatNear(log_mean, kTestPrecision)));
}

TEST_P(PoissonLossTest, UpdateGradientsLabelColIdx) {
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
  const PoissonLoss loss_imp({}, model::proto::Task::REGRESSION,
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

  // Initial predictions are log(2.5) (unweighted) and log(3) (weighted).

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

TEST_P(PoissonLossTest, UpdateGradientsPredictions) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const bool weighted = std::get<0>(GetParam());
  const bool threaded = std::get<1>(GetParam());
  std::vector<float> weights;
  if (weighted) {
    weights = {2.f, 4.f, 6.f, 8.f};
  }

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  const PoissonLoss loss_imp({}, model::proto::Task::REGRESSION,
                             dataset.data_spec().columns(0));
  std::vector<float> labels = {1.f, 2.f, 3.f, 4.f};
  ASSERT_OK(internal::CreateGradientDataset(dataset,
                                            /* label_col_idx= */ 0,
                                            /*hessian_splits=*/false, loss_imp,
                                            &gradient_dataset, &gradients,
                                            &predictions));
  GradientDataRef compact_gradient(gradients.size());
  for (int i = 0; i < gradients.size(); i++) {
    compact_gradient[i] = {&(gradients)[i].gradient, &(gradients)[i].hessian};
  }

  // Initial predictions are log(2.5) (unweighted) and log(3) (weighted).
  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> loss_initial_predictions,
      loss_imp.InitialPredictions(dataset,
                                  /* label_col_idx =*/0, weights));
  internal::SetInitialPredictions(loss_initial_predictions, dataset.nrow(),
                                  &predictions);

  // Initial predictions are log(2.5) (unweighted) and log(3) (weighted).

  utils::RandomEngine random(1234);

  if (threaded) {
    utils::concurrency::ThreadPool thread_pool("", 4);
    thread_pool.StartWorkers();
    ASSERT_OK(loss_imp.UpdateGradients(
        labels, predictions,
        /*ranking_index=*/nullptr, &compact_gradient, &random, &thread_pool));
  } else {
    ASSERT_OK(loss_imp.UpdateGradients(labels, predictions,
                                       /*ranking_index=*/nullptr,
                                       &compact_gradient, &random,
                                       /*thread_pool=*/nullptr));
  }

  ASSERT_THAT(gradients, Not(IsEmpty()));
  if (weighted) {
    EXPECT_THAT(gradients.front().gradient,
                ElementsAre(1.f - 3.f, 2.f - 3.f, 3.f - 3.f, 4.f - 3.f));
  } else {
    EXPECT_THAT(gradients.front().gradient,
                ElementsAre(1.f - 2.5f, 2.f - 2.5f, 3.f - 2.5f, 4.f - 2.5f));
  }
}

TEST_P(PoissonLossTest, ComputeLoss) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const bool weighted = std::get<0>(GetParam());
  const bool threaded = std::get<1>(GetParam());
  std::vector<float> weights;
  if (weighted) {
    weights = {2.f, 4.f, 6.f, 8.f};
  }

  std::vector<float> predictions = {1.f, 1.f, 2.f, 2.f};
  const PoissonLoss loss_imp({}, model::proto::Task::REGRESSION,
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
    const float expected_loss =
        -(110.f - 6.f * std::exp(1.f) - 14.f * std::exp(2.f)) / 10.f;
    const float expected_rmse = (2. * (1. - std::exp(1)) * (1. - std::exp(1)) +
                                 4. * (2. - std::exp(1)) * (2. - std::exp(1)) +
                                 6. * (3. - std::exp(2)) * (3. - std::exp(2)) +
                                 8. * (4. - std::exp(2)) * (4. - std::exp(2))) /
                                20.;
    EXPECT_NEAR(loss_results.loss, expected_loss, kTestPrecision);
    // For classification, the only secondary metric is also RMSE.
    EXPECT_THAT(loss_results.secondary_metrics,
                ElementsAre(FloatNear(expected_rmse, kTestPrecision)));
  } else {
    const float expected_loss =
        -(17.f - 2.f * std::exp(1.f) - 2.f * std::exp(2.f)) / 2.f;
    const float expected_rmse = ((1. - std::exp(1)) * (1. - std::exp(1)) +
                                 (2. - std::exp(1)) * (2. - std::exp(1)) +
                                 (3. - std::exp(2)) * (3. - std::exp(2)) +
                                 (4. - std::exp(2)) * (4. - std::exp(2))) /
                                4.;
    EXPECT_NEAR(loss_results.loss, expected_loss, kTestPrecision);
    // The only secondary metric is RMSE.
    EXPECT_THAT(loss_results.secondary_metrics,
                ElementsAre(FloatNear(expected_rmse, kTestPrecision)));
  }
}


TEST(PoissonLossTest, SecondaryMetricNames) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const PoissonLoss loss_imp({}, model::proto::Task::REGRESSION,
                             dataset.data_spec().columns(1));
  EXPECT_THAT(loss_imp.SecondaryMetricNames(), ElementsAre("RMSE"));
}

INSTANTIATE_TEST_SUITE_P(PoissonLossTestWithWeights, PoissonLossTest,
                         Combine(Bool(), Bool()));

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
