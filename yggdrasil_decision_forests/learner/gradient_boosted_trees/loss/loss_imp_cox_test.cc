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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_cox.h"

#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Pair;
using ::yggdrasil_decision_forests::dataset::proto::DataSpecification;
using ::yggdrasil_decision_forests::model::proto::TrainingConfigLinking;
using Update = CoxProportionalHazardLoss::Update;

// Margin of error for numerical tests.
constexpr float kTestPrecision = 0.000001f;

class CoxProportionalHazardLossTest : public ::testing::Test {
 protected:
  void SetUp() override {
    random_ = utils::RandomEngine(1234);
    label_col_idx_ = 0;
    *full_dataset_.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
      columns { type: NUMERICAL name: "departure_age" }
      columns { type: BOOLEAN name: "event" }
      columns { type: NUMERICAL name: "arrival_age" }
    )pb");
    ASSERT_OK(full_dataset_.CreateColumnsFromDataspec());
    ASSERT_OK(full_dataset_.AppendExampleWithStatus(
        {{"departure_age", "7"}, {"event", "1"}, {"arrival_age", "3"}}));
    ASSERT_OK(full_dataset_.AppendExampleWithStatus(
        {{"departure_age", "9"}, {"event", "0"}, {"arrival_age", "0"}}));
    ASSERT_OK(full_dataset_.AppendExampleWithStatus(
        {{"departure_age", "4"}, {"event", "1"}, {"arrival_age", "2"}}));
    ASSERT_OK(full_dataset_.AppendExampleWithStatus(
        {{"departure_age", "5"}, {"event", "0"}, {"arrival_age", "0"}}));
    ASSERT_OK(full_dataset_.AppendExampleWithStatus(
        {{"departure_age", "9"}, {"event", "1"}, {"arrival_age", "5"}}));
  }

  dataset::VerticalDataset GetDatasetWithTruncation() {
    return std::move(full_dataset_);
  }

  dataset::VerticalDataset GetDatasetWithoutTruncation() {
    DataSpecification data_spec;
    *data_spec.add_columns() = full_dataset_.data_spec().columns(0);
    *data_spec.add_columns() = full_dataset_.data_spec().columns(1);
    return full_dataset_.ConvertToGivenDataspec(data_spec, {}).value();
  }

  utils::RandomEngine random_;
  int label_col_idx_;
  dataset::VerticalDataset full_dataset_;
};

TEST_F(CoxProportionalHazardLossTest, RightCensoringAndLeftTruncation) {
  const TrainingConfigLinking config = PARSE_TEST_PROTO(
      R"pb(label: 0 label_event_observed: 1 label_entry_age: 2)pb");

  auto dataset = GetDatasetWithTruncation();

  ASSERT_OK_AND_ASSIGN(const auto loss,
                       CoxProportionalHazardLoss::RegistrationCreate(
                           {config,
                            {},
                            model::proto::Task::SURVIVAL_ANALYSIS,
                            dataset.data_spec().columns(0)}));
  ASSERT_OK_AND_ASSIGN(auto cache, loss->CreateLossCache(dataset));

  std::vector<GradientData> gradients;
  dataset::VerticalDataset gradient_dataset;
  std::vector<float> predictions = {-0.8, 0.0, 0.2, -0.4, 0.5};
  ASSERT_OK(internal::CreateGradientDataset(dataset, label_col_idx_, *loss,
                                            &gradient_dataset, &gradients,
                                            &predictions));
  ASSERT_OK_AND_ASSIGN(
      auto loss_result,
      loss->Loss(dataset, label_col_idx_, predictions, {}, cache.get()));
  EXPECT_THAT(loss_result.loss, FloatNear(0.682227135, kTestPrecision));

  ASSERT_OK(loss->UpdateGradients(dataset, label_col_idx_, predictions,
                                  cache.get(), &gradients, &random_));

  ASSERT_THAT(gradients, Not(IsEmpty()));
  EXPECT_THAT(gradients.front().gradient,
              ElementsAre(FloatNear(0.144095331, kTestPrecision),
                          FloatNear(-0.199926272, kTestPrecision),
                          FloatNear(0.126885131, kTestPrecision),
                          FloatNear(-0.0401262902, kTestPrecision),
                          FloatNear(-0.0309279412, kTestPrecision)));
  EXPECT_THAT(gradients.front().hessian,
              ElementsAre(FloatNear(0.0480802134, kTestPrecision),
                          FloatNear(0.132664084, kTestPrecision),
                          FloatNear(0.0463859476, kTestPrecision),
                          FloatNear(0.0320756957, kTestPrecision),
                          FloatNear(0.0967936218, kTestPrecision)));
}

TEST_F(CoxProportionalHazardLossTest, RightCensoringNoLeftTruncation) {
  const TrainingConfigLinking config =
      PARSE_TEST_PROTO(R"pb(label: 0 label_event_observed: 1)pb");

  auto dataset = GetDatasetWithoutTruncation();

  ASSERT_OK_AND_ASSIGN(const auto loss,
                       CoxProportionalHazardLoss::RegistrationCreate(
                           {config,
                            {},
                            model::proto::Task::SURVIVAL_ANALYSIS,
                            dataset.data_spec().columns(0)}));
  ASSERT_OK_AND_ASSIGN(auto cache, loss->CreateLossCache(dataset));

  std::vector<GradientData> gradients;
  dataset::VerticalDataset gradient_dataset;
  std::vector<float> predictions = {-0.8, 0.0, 0.2, -0.4, 0.5};
  ASSERT_OK(internal::CreateGradientDataset(dataset, label_col_idx_, *loss,
                                            &gradient_dataset, &gradients,
                                            &predictions));

  ASSERT_OK_AND_ASSIGN(
      auto loss_result,
      loss->Loss(dataset, label_col_idx_, predictions, {}, cache.get()));
  EXPECT_THAT(loss_result.loss, FloatNear(0.76244804, kTestPrecision));

  ASSERT_OK(loss->UpdateGradients(dataset, label_col_idx_, predictions,
                                  cache.get(), &gradients, &random_));

  ASSERT_THAT(gradients, Not(IsEmpty()));
  EXPECT_THAT(gradients.front().gradient,
              ElementsAre(FloatNear(0.152982801, kTestPrecision),
                          FloatNear(-0.180146873, kTestPrecision),
                          FloatNear(0.151043758, kTestPrecision),
                          FloatNear(-0.0268677603, kTestPrecision),
                          FloatNear(-0.0970119685, kTestPrecision)));
  EXPECT_THAT(gradients.front().hessian,
              ElementsAre(FloatNear(0.0411883146, kTestPrecision),
                          FloatNear(0.122768782, kTestPrecision),
                          FloatNear(0.0369726755, kTestPrecision),
                          FloatNear(0.0232583769, kTestPrecision),
                          FloatNear(0.141042158, kTestPrecision)));
}

class CoxProportionalHazardLossStabilityTest : public ::testing::Test {
 protected:
  // Builds a right-censored dataset (no left truncation) with the given
  // departure ages and event indicators.
  dataset::VerticalDataset BuildDataset(
      const std::vector<float>& departure_ages,
      const std::vector<bool>& events) {
    dataset::VerticalDataset dataset;
    *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
      columns { type: NUMERICAL name: "departure_age" }
      columns { type: BOOLEAN name: "event" }
    )pb");
    CHECK_OK(dataset.CreateColumnsFromDataspec());
    for (int i = 0; i < departure_ages.size(); ++i) {
      CHECK_OK(dataset.AppendExampleWithStatus(
          {{"departure_age", absl::StrCat(departure_ages[i])},
           {"event", events[i] ? "1" : "0"}}));
    }
    return dataset;
  }

  absl::StatusOr<std::unique_ptr<AbstractLoss>> BuildLoss(
      const dataset::VerticalDataset& dataset) {
    const TrainingConfigLinking config =
        PARSE_TEST_PROTO(R"pb(label: 0 label_event_observed: 1)pb");
    return CoxProportionalHazardLoss::RegistrationCreate(
        {config,
         {},
         model::proto::Task::SURVIVAL_ANALYSIS,
         dataset.data_spec().columns(0)});
  }

  utils::RandomEngine random_;
};

// A single example with a very large log-hazard must not wipe out the
// contributions of the other examples from the risk set.
TEST_F(CoxProportionalHazardLossStabilityTest, LargeLogHazardRange) {
  // Example 0 is censored first and has a log-hazard so large that
  // `exp(40) + 3 == exp(40)` in double precision.
  auto dataset = BuildDataset(/*departure_ages=*/{1, 2, 3, 4},
                              /*events=*/{false, true, true, true});
  ASSERT_OK_AND_ASSIGN(const auto loss, BuildLoss(dataset));
  ASSERT_OK_AND_ASSIGN(auto cache, loss->CreateLossCache(dataset));

  const std::vector<float> predictions = {40.f, 0.f, 0.f, 0.f};
  ASSERT_OK_AND_ASSIGN(
      auto loss_result,
      loss->Loss(dataset, /*label_col_idx=*/0, predictions, {}, cache.get()));

  // Risk sets of the three events are {1,2,3}, {2,3} and {3}.
  const float expected = (std::log(3.f) + std::log(2.f) + std::log(1.f)) / 4.f;
  EXPECT_THAT(loss_result.loss, FloatNear(expected, kTestPrecision));
}

// The Cox partial likelihood is invariant to adding a constant to all the
// log-hazard predictions. Large predictions must not overflow `exp`.
TEST_F(CoxProportionalHazardLossStabilityTest, InvariantToGlobalShift) {
  auto dataset = BuildDataset(/*departure_ages=*/{1, 2, 3, 4},
                              /*events=*/{true, false, true, true});
  ASSERT_OK_AND_ASSIGN(const auto loss, BuildLoss(dataset));
  ASSERT_OK_AND_ASSIGN(auto cache, loss->CreateLossCache(dataset));

  const std::vector<float> predictions = {-0.8f, 0.f, 0.2f, 0.5f};
  std::vector<float> shifted_predictions;
  for (const float prediction : predictions) {
    shifted_predictions.push_back(prediction + 800.f);
  }

  ASSERT_OK_AND_ASSIGN(
      auto loss_result,
      loss->Loss(dataset, /*label_col_idx=*/0, predictions, {}, cache.get()));
  ASSERT_OK_AND_ASSIGN(auto shifted_loss_result,
                       loss->Loss(dataset, /*label_col_idx=*/0,
                                  shifted_predictions, {}, cache.get()));

  // The tolerance accounts for the float32 rounding of "prediction + 800"
  // itself (the ulp of 800.f is ~6e-5), not for the loss computation.
  EXPECT_THAT(shifted_loss_result.loss, FloatNear(loss_result.loss, 1e-4f));
}

// Gradients must be finite and hessians non-negative even for extreme
// predictions.
TEST_F(CoxProportionalHazardLossStabilityTest, FiniteGradientsAndHessians) {
  auto dataset = BuildDataset(/*departure_ages=*/{1, 2, 3, 4},
                              /*events=*/{false, true, true, true});
  ASSERT_OK_AND_ASSIGN(const auto loss, BuildLoss(dataset));
  ASSERT_OK_AND_ASSIGN(auto cache, loss->CreateLossCache(dataset));

  std::vector<GradientData> gradients;
  dataset::VerticalDataset gradient_dataset;
  std::vector<float> predictions = {40.f, 0.f, 0.f, 0.f};
  ASSERT_OK(internal::CreateGradientDataset(dataset, /*label_col_idx=*/0, *loss,
                                            &gradient_dataset, &gradients,
                                            &predictions));
  ASSERT_OK(loss->UpdateGradients(dataset, /*label_col_idx=*/0, predictions,
                                  cache.get(), &gradients, &random_));

  ASSERT_THAT(gradients, Not(IsEmpty()));
  for (const float gradient : gradients.front().gradient) {
    EXPECT_TRUE(std::isfinite(gradient)) << "gradient: " << gradient;
  }
  for (const float hessian : gradients.front().hessian) {
    EXPECT_TRUE(std::isfinite(hessian)) << "hessian: " << hessian;
    EXPECT_GE(hessian, 0.f);
  }
}

// An extreme spread between the log-hazards leaves the examples at risk with a
// hazard many orders of magnitude below the largest one. The reciprocals of the
// sum of the hazards at risk must stay finite.
TEST_F(CoxProportionalHazardLossStabilityTest, ExtremePredictionSpread) {
  auto dataset = BuildDataset(/*departure_ages=*/{1, 2, 3, 4},
                              /*events=*/{false, true, true, true});
  ASSERT_OK_AND_ASSIGN(const auto loss, BuildLoss(dataset));
  ASSERT_OK_AND_ASSIGN(auto cache, loss->CreateLossCache(dataset));

  std::vector<GradientData> gradients;
  dataset::VerticalDataset gradient_dataset;
  // Example 0 is censored first, so the risk set of every event is made only
  // of examples whose hazard is exp(-3000) times the largest one.
  std::vector<float> predictions = {3000.f, 0.f, 0.f, 0.f};
  ASSERT_OK(internal::CreateGradientDataset(dataset, /*label_col_idx=*/0, *loss,
                                            &gradient_dataset, &gradients,
                                            &predictions));

  ASSERT_OK_AND_ASSIGN(
      auto loss_result,
      loss->Loss(dataset, /*label_col_idx=*/0, predictions, {}, cache.get()));
  EXPECT_TRUE(std::isfinite(loss_result.loss)) << "loss: " << loss_result.loss;

  ASSERT_OK(loss->UpdateGradients(dataset, /*label_col_idx=*/0, predictions,
                                  cache.get(), &gradients, &random_));
  for (const float gradient : gradients.front().gradient) {
    EXPECT_TRUE(std::isfinite(gradient)) << "gradient: " << gradient;
  }
  for (const float hessian : gradients.front().hessian) {
    EXPECT_TRUE(std::isfinite(hessian)) << "hessian: " << hessian;
    EXPECT_GE(hessian, 0.f);
  }
}

// Non-finite predictions must be reported instead of silently producing a
// null loss.
TEST_F(CoxProportionalHazardLossStabilityTest, NonFinitePredictionsAreAnError) {
  auto dataset =
      BuildDataset(/*departure_ages=*/{1, 2}, /*events=*/{true, true});
  ASSERT_OK_AND_ASSIGN(const auto loss, BuildLoss(dataset));
  ASSERT_OK_AND_ASSIGN(auto cache, loss->CreateLossCache(dataset));

  const std::vector<float> predictions = {
      0.f, std::numeric_limits<float>::quiet_NaN()};
  EXPECT_THAT(
      loss->Loss(dataset, /*label_col_idx=*/0, predictions, {}, cache.get())
          .status(),
      test::StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
