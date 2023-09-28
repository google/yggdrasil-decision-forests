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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_binomial.h"

#include "gmock/gmock.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
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
      categorical { number_of_unique_values: 3 is_already_integerized: true }
    }
    columns { type: NUMERICAL name: "__gradient__0" }
  )pb");
  RETURN_IF_ERROR(dataset.CreateColumnsFromDataspec());
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus(
      {{"a", "1"}, {"b", "1"}, {"__gradient__0", "4"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus(
      {{"a", "2"}, {"b", "2"}, {"__gradient__0", "-4"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus(
      {{"a", "3"}, {"b", "1"}, {"__gradient__0", "0"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus(
      {{"a", "4"}, {"b", "2"}, {"__gradient__0", "8"}}));
  return dataset;
}

class BinomialLogLikelihoodLossTest : public testing::TestWithParam<bool> {};

TEST_P(BinomialLogLikelihoodLossTest, InitialPredictions) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  const bool weighted = GetParam();
  std::vector<float> weights;
  if (weighted) {
    weights = {2.f, 4.f, 6.f, 8.f};
  }
  const auto loss_imp = BinomialLogLikelihoodLoss(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  ASSERT_OK_AND_ASSIGN(
      auto init_pred,
      loss_imp.InitialPredictions(dataset, /* label_col_idx= */ 1, weights));
  if (weighted) {
    EXPECT_THAT(init_pred,
                ElementsAre(FloatNear(std::log(3. / 2.), kTestPrecision)));
  } else {
    EXPECT_THAT(init_pred, ElementsAre(0.f));
  }
}

TEST(BinomialLogLikelihoodLossTest, UpdateGradients) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions = {0.f, 0.f, 0.f, 0.f};
  const auto loss_imp = BinomialLogLikelihoodLoss(
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
  EXPECT_THAT(gradients.front().gradient,
              ElementsAre(-0.5f, 0.5f, -0.5f, 0.5f));
}

TEST_P(BinomialLogLikelihoodLossTest, ComputeLoss) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  const bool weighted = GetParam();
  std::vector<float> weights;
  if (weighted) {
    weights = {1.f, 2.f, 3.f, 4.f};
  }
  std::vector<float> predictions(dataset.nrow(), 0.f);
  const auto loss_imp = BinomialLogLikelihoodLoss(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  ASSERT_OK_AND_ASSIGN(
      LossResults loss_results,
      loss_imp.Loss(dataset,
                    /* label_col_idx= */ 1, predictions, weights, nullptr));
  if (weighted) {
    EXPECT_NEAR(loss_results.loss, 2 * std::log(2), kTestPrecision);
    EXPECT_THAT(loss_results.secondary_metrics,
                ElementsAre(FloatNear(0.4f, kTestPrecision)));
    ASSERT_TRUE(loss_results.confusion_table.has_value());
    EXPECT_EQ(loss_results.confusion_table->at(1, 1), 4);
    EXPECT_EQ(loss_results.confusion_table->at(2, 1), 6);
    EXPECT_EQ(loss_results.confusion_table->sum(), 10);
  } else {
    EXPECT_NEAR(loss_results.loss, 2 * std::log(2), kTestPrecision);
    EXPECT_THAT(loss_results.secondary_metrics,
                ElementsAre(FloatNear(0.5f, kTestPrecision)));
    ASSERT_TRUE(loss_results.confusion_table.has_value());
    EXPECT_EQ(loss_results.confusion_table->at(1, 1), 2);
    EXPECT_EQ(loss_results.confusion_table->at(2, 1), 2);
    EXPECT_EQ(loss_results.confusion_table->sum(), 4);
  }
}

TEST(BinomialLogLikelihoodLossTest, ComputeLossWithNullWeights) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  std::vector<float> weights(dataset.nrow(), 0.f);
  std::vector<float> predictions(dataset.nrow(), 0.f);
  const auto loss_imp = BinomialLogLikelihoodLoss(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  ASSERT_OK_AND_ASSIGN(
      LossResults loss_results,
      loss_imp.Loss(dataset,
                    /* label_col_idx= */ 1, predictions, weights, nullptr));

  EXPECT_THAT(loss_results.loss, IsNan());
  EXPECT_THAT(loss_results.secondary_metrics, ElementsAre(IsNan()));
  EXPECT_EQ(loss_results.confusion_table, absl::nullopt);
}

TEST(BinomialLogLikelihoodLossTest, SecondaryMetricName) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  const auto loss_imp = BinomialLogLikelihoodLoss(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  EXPECT_THAT(loss_imp.SecondaryMetricNames(), ElementsAre("accuracy"));
}

INSTANTIATE_TEST_SUITE_P(BinomialLogLikelihoodLossTestWithWeights,
                         BinomialLogLikelihoodLossTest, testing::Bool());

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
