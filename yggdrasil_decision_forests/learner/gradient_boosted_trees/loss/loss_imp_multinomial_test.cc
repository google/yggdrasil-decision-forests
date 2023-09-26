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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_multinomial.h"

#include "gmock/gmock.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::IsEmpty;
using ::testing::IsNan;
using ::testing::Not;
using ::testing::SizeIs;

// Margin of error for numerical tests.
constexpr float kTestPrecision = 0.000001f;

absl::StatusOr<dataset::VerticalDataset> CreateToyDataset() {
  dataset::VerticalDataset dataset;
  // TODO Replace PARSE_TEST_PROTO by a modern function when
  // possible.
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "a" }
    columns {
      type: CATEGORICAL
      name: "b"
      categorical { number_of_unique_values: 4 is_already_integerized: true }
    }
  )pb");
  RETURN_IF_ERROR(dataset.CreateColumnsFromDataspec());
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "1"}, {"b", "1"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "2"}, {"b", "2"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "3"}, {"b", "3"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "4"}, {"b", "1"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "5"}, {"b", "2"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "6"}, {"b", "3"}}));
  return dataset;
}

// Returns a simple dataset with gradients in the second column.
absl::StatusOr<dataset::VerticalDataset> CreateToyGradientDataset() {
  dataset::VerticalDataset dataset;
  // TODO Replace PARSE_TEST_PROTO by a modern function when
  // possible.
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "a" }
    columns {
      type: CATEGORICAL
      name: "b"
      categorical { number_of_unique_values: 4 is_already_integerized: true }
    }
    columns { type: NUMERICAL name: "__gradient__0" }
  )pb");
  RETURN_IF_ERROR(dataset.CreateColumnsFromDataspec());
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus(
      {{"a", "1"}, {"b", "1"}, {"__gradient__0", ".25"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus(
      {{"a", "2"}, {"b", "2"}, {"__gradient__0", "-.25"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus(
      {{"a", "3"}, {"b", "3"}, {"__gradient__0", "0"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus(
      {{"a", "4"}, {"b", "1"}, {"__gradient__0", ".5"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus(
      {{"a", "5"}, {"b", "2"}, {"__gradient__0", "0"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus(
      {{"a", "6"}, {"b", "3"}, {"__gradient__0", ".25"}}));
  return dataset;
}

struct TestParameters {
  bool weighted;
  bool threaded;
};

class MultinomialLogLikelihoodLossTest
    : public testing::TestWithParam<TestParameters> {};

INSTANTIATE_TEST_SUITE_P(MultinomialLogLikelihoodLossTestSuite,
                         MultinomialLogLikelihoodLossTest,
                         testing::ValuesIn<TestParameters>({{true, true},
                                                            {true, false},
                                                            {false, true},
                                                            {false, false}}));

TEST_P(MultinomialLogLikelihoodLossTest, InitialPredictions) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  const bool weighted = GetParam().weighted;
  std::vector<float> weights;
  if (weighted) {
    weights = {2.f, 4.f, 6.f, 8.f, 10.f, 12.f};
  }

  const MultinomialLogLikelihoodLoss loss_imp(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> init_pred,
      loss_imp.InitialPredictions(dataset, /* label_col_idx= */ 1, weights));
  // Initial predictions for multinomial loss are always 0.
  EXPECT_THAT(init_pred, ElementsAre(0.f, 0.f, 0.f));
}

TEST(MultinomialLogLikelihoodLossTest, UpdateGradients) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const MultinomialLogLikelihoodLoss loss_imp(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  ASSERT_OK(internal::CreateGradientDataset(dataset,
                                            /* label_col_idx= */ 1,
                                            /*hessian_splits=*/false, loss_imp,
                                            &gradient_dataset, &gradients,
                                            &predictions));

  utils::RandomEngine random(1234);
  ASSERT_OK(loss_imp.UpdateGradients(
      gradient_dataset, /* label_col_idx= */ 1, predictions,
      /*ranking_index=*/nullptr, &gradients, &random));
  ASSERT_THAT(gradients, Not(IsEmpty()));

  const std::vector<float>& gradient = gradients.front().gradient;
  EXPECT_THAT(gradient, ElementsAre(FloatNear(2.f / 3.f, kTestPrecision),
                                    FloatNear(-1.f / 3.f, kTestPrecision),
                                    FloatNear(-1.f / 3.f, kTestPrecision),
                                    FloatNear(2.f / 3.f, kTestPrecision),
                                    FloatNear(-1.f / 3.f, kTestPrecision),
                                    FloatNear(-1.f / 3.f, kTestPrecision)));
}


TEST(MultinomialLogLikelihoodLossTest, SecondaryMetricName) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  const MultinomialLogLikelihoodLoss loss_imp(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  EXPECT_THAT(loss_imp.SecondaryMetricNames(), ElementsAre("accuracy"));
}

TEST_P(MultinomialLogLikelihoodLossTest, ComputeLoss) {
  const bool threaded = GetParam().threaded;
  const bool weighted = GetParam().weighted;
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  std::vector<float> weights;
  if (weighted) {
    weights = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  }
  auto label_column = dataset.data_spec().columns(1);
  std::vector<float> predictions(
      dataset.nrow() *
          (label_column.categorical().number_of_unique_values() - 1),
      0.f);
  const MultinomialLogLikelihoodLoss loss_imp(
      {}, model::proto::Task::CLASSIFICATION, label_column);
  LossResults loss_results;
  if (threaded) {
    utils::concurrency::ThreadPool thread_pool("", 4);
    thread_pool.StartWorkers();
    ASSERT_OK_AND_ASSIGN(loss_results,
                         loss_imp.Loss(dataset,
                                       /* label_col_idx= */ 1, predictions,
                                       weights, nullptr, &thread_pool));
  } else {
    ASSERT_OK_AND_ASSIGN(
        loss_results,
        loss_imp.Loss(dataset,
                      /* label_col_idx= */ 1, predictions, weights, nullptr));
  }

  if (weighted) {
    EXPECT_NEAR(loss_results.loss, std::log(3), kTestPrecision);
    EXPECT_THAT(loss_results.secondary_metrics,
                ElementsAre(FloatNear(5. / 21., kTestPrecision)));
    ASSERT_TRUE(loss_results.confusion_table.has_value());
    EXPECT_EQ(loss_results.confusion_table->at(1, 1), 5.);
    EXPECT_EQ(loss_results.confusion_table->at(2, 1), 7.);
    EXPECT_EQ(loss_results.confusion_table->at(3, 1), 9.);
    EXPECT_EQ(loss_results.confusion_table->sum(), 21.);
  } else {
    EXPECT_NEAR(loss_results.loss, std::log(3), kTestPrecision);
    EXPECT_THAT(loss_results.secondary_metrics,
                ElementsAre(FloatNear(0.333333f, kTestPrecision)));
    ASSERT_TRUE(loss_results.confusion_table.has_value());
    EXPECT_EQ(loss_results.confusion_table->at(1, 1), 2);
    EXPECT_EQ(loss_results.confusion_table->at(2, 1), 2);
    EXPECT_EQ(loss_results.confusion_table->at(3, 1), 2);
    EXPECT_EQ(loss_results.confusion_table->sum(), 6);
  }
}

TEST_P(MultinomialLogLikelihoodLossTest, ComputeLossWithNullWeights) {
  const bool threaded = GetParam().threaded;
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  std::vector<float> weights(dataset.nrow(), 0.f);
  auto label_column = dataset.data_spec().columns(1);
  std::vector<float> predictions(
      dataset.nrow() *
          (label_column.categorical().number_of_unique_values() - 1),
      0.f);
  const MultinomialLogLikelihoodLoss loss_imp(
      {}, model::proto::Task::CLASSIFICATION, label_column);
  LossResults loss_results;
  if (threaded) {
    utils::concurrency::ThreadPool thread_pool("", 4);
    thread_pool.StartWorkers();
    ASSERT_OK_AND_ASSIGN(
        loss_results,
        loss_imp.Loss(dataset,
                      /* label_col_idx= */ 1, predictions, weights, nullptr));
  } else {
    ASSERT_OK_AND_ASSIGN(
        loss_results,
        loss_imp.Loss(dataset,
                      /* label_col_idx= */ 1, predictions, weights, nullptr));
  }

  EXPECT_THAT(loss_results.loss, IsNan());
  EXPECT_THAT(loss_results.secondary_metrics, ElementsAre(IsNan()));
  EXPECT_EQ(loss_results.confusion_table, absl::nullopt);
}

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
