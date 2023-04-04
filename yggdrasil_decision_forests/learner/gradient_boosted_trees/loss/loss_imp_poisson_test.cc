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

#include "gmock/gmock.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatNear;

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

class PoissonLossTest : public testing::TestWithParam<bool> {};

TEST_P(PoissonLossTest, LossStatusRegression) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const PoissonLoss loss_imp({}, model::proto::Task::REGRESSION,
                             dataset.data_spec().columns(0));
  EXPECT_OK(loss_imp.Status());
}

TEST_P(PoissonLossTest, LossStatusClassification) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const PoissonLoss loss_imp({}, model::proto::Task::CLASSIFICATION,
                             dataset.data_spec().columns(0));
  EXPECT_FALSE(loss_imp.Status().ok());
}

TEST_P(PoissonLossTest, LossStatusRanking) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const PoissonLoss loss_imp({}, model::proto::Task::RANKING,
                             dataset.data_spec().columns(0));
  EXPECT_FALSE(loss_imp.Status().ok());
}

TEST_P(PoissonLossTest, InitialPredictionsClassic) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const bool weighted = GetParam();
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

INSTANTIATE_TEST_SUITE_P(PoissonLossTestWithWeights, PoissonLossTest,
                         testing::Bool());

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
