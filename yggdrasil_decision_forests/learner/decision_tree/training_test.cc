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

#include "yggdrasil_decision_forests/learner/decision_tree/training.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {
namespace {

using test::StatusIs;

// Margin of error for numerical tests.
constexpr float kTestPrecision = 0.000001f;

// Returns a simple datasets with gradients in the second column.
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

TEST(DecisionTreeTrainingTest, SetRegressionLabelDistributionWeighted) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyGradientDataset());
  std::vector<UnsignedExampleIdx> selected_examples = {0, 1, 2, 3};
  std::vector<float> weights = {2.f, 4.f, 6.f, 8.f};
  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(2);  // Gradient column.

  proto::Node node;

  ASSERT_OK(SetRegressionLabelDistribution(dataset, selected_examples, weights,
                                           config_link, &node));
  EXPECT_NEAR(node.regressor().top_value(), 2.8f, kTestPrecision);
  // // Distribution of the gradients:
  EXPECT_EQ(node.regressor().distribution().sum(), 56);
  EXPECT_EQ(node.regressor().distribution().sum_squares(),
            2 * 16 + 4 * 16 + 8 * 64);
  EXPECT_EQ(node.regressor().distribution().count(), 20);
}

TEST(DecisionTreeTrainingTest, SetRegressionLabelDistributionUnweighted) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyGradientDataset());
  std::vector<UnsignedExampleIdx> selected_examples = {0, 1, 2, 3};
  std::vector<float> weights;
  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(2);  // Gradient column.

  proto::Node node;

  ASSERT_OK(SetRegressionLabelDistribution(dataset, selected_examples, weights,
                                           config_link, &node));
  EXPECT_NEAR(node.regressor().top_value(), 2.f, kTestPrecision);
  // // Distribution of the gradients:
  EXPECT_EQ(node.regressor().distribution().sum(), 8);
  EXPECT_EQ(node.regressor().distribution().sum_squares(), 16 + 16 + 64);
  EXPECT_EQ(node.regressor().distribution().count(), 4);
}

TEST(DecisionTreeTrainingTest,
     SetRegressionLabelDistributionWeightedWithIncorrectSizedWeights) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyGradientDataset());
  std::vector<UnsignedExampleIdx> selected_examples = {0, 1, 2, 3};
  std::vector<float> weights = {2.f, 4.f, 6.f};
  model::proto::TrainingConfigLinking config_link;
  proto::Node node;

  EXPECT_THAT(SetRegressionLabelDistribution(dataset, selected_examples,
                                             weights, config_link, &node),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
