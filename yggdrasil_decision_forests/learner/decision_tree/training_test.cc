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
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {
namespace {

using test::StatusIs;
using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::Pointwise;

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

dataset::VerticalDataset CreateMHDLDataset() {
  dataset::VerticalDataset dataset;
  const auto label_col =
      dataset.AddColumn("l", dataset::proto::ColumnType::CATEGORICAL).value();
  label_col->mutable_categorical()->set_is_already_integerized(true);
  label_col->mutable_categorical()->set_number_of_unique_values(3);
  EXPECT_OK(
      dataset.AddColumn("f1", dataset::proto::ColumnType::NUMERICAL).status());
  EXPECT_OK(
      dataset.AddColumn("f2", dataset::proto::ColumnType::NUMERICAL).status());
  EXPECT_OK(dataset.CreateColumnsFromDataspec());

  // Those values are similar to the section 3.2 of the "Linear Discriminant
  // Analysis: A Detailed Tutorial" by Tharwat et al.
  dataset.AppendExample({{"l", "1"}, {"f1", "1"}, {"f2", "2"}});
  dataset.AppendExample({{"l", "1"}, {"f1", "2"}, {"f2", "3"}});
  dataset.AppendExample({{"l", "1"}, {"f1", "3"}, {"f2", "3"}});
  dataset.AppendExample({{"l", "1"}, {"f1", "4"}, {"f2", "5"}});
  dataset.AppendExample({{"l", "1"}, {"f1", "5"}, {"f2", "5"}});

  dataset.AppendExample({{"l", "2"}, {"f1", "4"}, {"f2", "2"}});
  dataset.AppendExample({{"l", "2"}, {"f1", "5"}, {"f2", "0"}});
  dataset.AppendExample({{"l", "2"}, {"f1", "5"}, {"f2", "2"}});
  dataset.AppendExample({{"l", "2"}, {"f1", "3"}, {"f2", "2"}});
  dataset.AppendExample({{"l", "2"}, {"f1", "5"}, {"f2", "3"}});
  dataset.AppendExample({{"l", "2"}, {"f1", "6"}, {"f2", "3"}});
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

  ASSERT_OK(SetRegressionLabelDistribution</*weighted=*/true>(
      dataset, selected_examples, weights, config_link, &node));
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

  ASSERT_OK(SetRegressionLabelDistribution</*weighted=*/false>(
      dataset, selected_examples, weights, config_link, &node));
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

  EXPECT_THAT(SetRegressionLabelDistribution</*weighted=*/true>(
                  dataset, selected_examples, weights, config_link, &node),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SparaseOblique, Classification) {
  const model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(0);
  config_link.add_numerical_features(1);
  config_link.add_numerical_features(2);

  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_sparse_oblique_split();
  dt_config.set_min_examples(1);
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::IN_NODE);

  dataset::VerticalDataset dataset;
  ASSERT_OK_AND_ASSIGN(
      auto label_col,
      dataset.AddColumn("l", dataset::proto::ColumnType::CATEGORICAL));
  label_col->mutable_categorical()->set_is_already_integerized(true);
  label_col->mutable_categorical()->set_number_of_unique_values(3);
  EXPECT_OK(
      dataset.AddColumn("f1", dataset::proto::ColumnType::NUMERICAL).status());
  EXPECT_OK(
      dataset.AddColumn("f2", dataset::proto::ColumnType::NUMERICAL).status());
  EXPECT_OK(dataset.CreateColumnsFromDataspec());

  dataset.AppendExample({{"l", "1"}, {"f1", "0.1"}, {"f2", "0.1"}});
  dataset.AppendExample({{"l", "1"}, {"f1", "0.9"}, {"f2", "0.9"}});
  dataset.AppendExample({{"l", "2"}, {"f1", "0.1"}, {"f2", "0.15"}});
  dataset.AppendExample({{"l", "2"}, {"f1", "0.9"}, {"f2", "0.95"}});

  ASSERT_OK_AND_ASSIGN(auto* label_data,
                       dataset.MutableColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(0));

  const std::vector<UnsignedExampleIdx> selected_examples = {0, 1, 2, 3};
  const std::vector<float> weights = {1.f, 1.f, 1.f, 1.f};

  ClassificationLabelStats label_stats(label_data->values());
  label_stats.num_label_classes = 3;
  label_stats.label_distribution.SetNumClasses(3);
  for (const auto example_idx : selected_examples) {
    label_stats.label_distribution.Add(label_data->values()[example_idx],
                                       weights[example_idx]);
  }

  proto::Node parent;
  InternalTrainConfig internal_config;
  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  utils::RandomEngine random;
  const auto result = FindBestConditionOblique(
                          dataset, selected_examples, weights, config,
                          config_link, dt_config, parent, internal_config,
                          label_stats, 50, &best_condition, &random, &cache)
                          .value();
  EXPECT_TRUE(result);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 2);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 4);
  EXPECT_NEAR(best_condition.split_score(), 0.693, 0.001);
}

TEST(MHLDTOblique, Classification) {
  const model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(0);
  config_link.add_numerical_features(1);
  config_link.add_numerical_features(2);

  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_mhld_oblique_split();
  dt_config.set_min_examples(1);
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::IN_NODE);

  dataset::VerticalDataset dataset;
  ASSERT_OK_AND_ASSIGN(
      auto label_col,
      dataset.AddColumn("l", dataset::proto::ColumnType::CATEGORICAL));
  label_col->mutable_categorical()->set_is_already_integerized(true);
  label_col->mutable_categorical()->set_number_of_unique_values(3);
  EXPECT_OK(
      dataset.AddColumn("f1", dataset::proto::ColumnType::NUMERICAL).status());
  EXPECT_OK(
      dataset.AddColumn("f2", dataset::proto::ColumnType::NUMERICAL).status());
  EXPECT_OK(dataset.CreateColumnsFromDataspec());

  dataset.AppendExample({{"l", "1"}, {"f1", "0.1"}, {"f2", "1.1"}});
  dataset.AppendExample({{"l", "1"}, {"f1", "0.9"}, {"f2", "1.9"}});
  dataset.AppendExample({{"l", "2"}, {"f1", "0.1"}, {"f2", "1.15"}});
  dataset.AppendExample({{"l", "2"}, {"f1", "0.9"}, {"f2", "1.95"}});

  ASSERT_OK_AND_ASSIGN(auto* label_data,
                       dataset.MutableColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(0));

  std::vector<UnsignedExampleIdx> selected_examples(dataset.nrow());
  std::iota(selected_examples.begin(), selected_examples.end(), 0);
  const std::vector<float> weights(selected_examples.size(), 1.f);

  ClassificationLabelStats label_stats(label_data->values());
  label_stats.num_label_classes = 3;
  label_stats.label_distribution.SetNumClasses(3);
  for (const auto example_idx : selected_examples) {
    label_stats.label_distribution.Add(label_data->values()[example_idx],
                                       weights[example_idx]);
  }

  proto::Node parent;
  InternalTrainConfig internal_config;
  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  utils::RandomEngine random;
  const auto result = FindBestConditionOblique(
                          dataset, selected_examples, weights, config,
                          config_link, dt_config, parent, internal_config,
                          label_stats, 50, &best_condition, &random, &cache)
                          .value();

  YDF_LOG(INFO) << "best_condition:\n" << best_condition.DebugString();

  EXPECT_TRUE(result);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 2);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 4);
  EXPECT_NEAR(best_condition.split_score(), 0.693, 0.001);
}

TEST(MHLDTOblique, Classification_Again) {
  const model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(0);
  config_link.add_numerical_features(1);
  config_link.add_numerical_features(2);

  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_mhld_oblique_split();
  dt_config.set_min_examples(1);
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::IN_NODE);

  dataset::VerticalDataset dataset = CreateMHDLDataset();

  ASSERT_OK_AND_ASSIGN(auto* label_data,
                       dataset.MutableColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(0));

  std::vector<UnsignedExampleIdx> selected_examples(dataset.nrow());
  std::iota(selected_examples.begin(), selected_examples.end(), 0);
  const std::vector<float> weights(selected_examples.size(), 1.f);

  ClassificationLabelStats label_stats(label_data->values());
  label_stats.num_label_classes = 3;
  label_stats.label_distribution.SetNumClasses(3);
  for (const auto example_idx : selected_examples) {
    label_stats.label_distribution.Add(label_data->values()[example_idx],
                                       weights[example_idx]);
  }

  proto::Node parent;
  InternalTrainConfig internal_config;
  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  utils::RandomEngine random;
  const auto result = FindBestConditionOblique(
                          dataset, selected_examples, weights, config,
                          config_link, dt_config, parent, internal_config,
                          label_stats, 50, &best_condition, &random, &cache)
                          .value();

  YDF_LOG(INFO) << "best_condition:\n" << best_condition.DebugString();

  EXPECT_TRUE(result);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 11);
  EXPECT_NEAR(best_condition.split_score(), 0.689, 0.001);
}

TEST(Oblique, ProjectionEvaluator) {
  dataset::VerticalDataset dataset = CreateMHDLDataset();

  model::proto::TrainingConfigLinking config_link;
  config_link.add_numerical_features(1);
  config_link.add_numerical_features(2);
  internal::ProjectionEvaluator proj(dataset, config_link.numerical_features());

  EXPECT_EQ(proj.NaReplacementValue(1), 0);
  EXPECT_EQ(proj.NaReplacementValue(2), 0);

  std::vector<float> values;
  EXPECT_OK(proj.ExtractAttribute(1, {1, 2}, &values));
  EXPECT_THAT(values, ElementsAre(2, 3));

  EXPECT_OK(proj.Evaluate({{1, 1.f}, {2, 2.f}}, {1, 2}, &values));
  EXPECT_THAT(values, ElementsAre(8, 9));
}

TEST(MHLDTOblique, LDACache) {
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_mhld_oblique_split();
  dt_config.set_min_examples(1);

  dataset::VerticalDataset dataset = CreateMHDLDataset();
  model::proto::TrainingConfigLinking config_link;
  config_link.add_numerical_features(1);
  config_link.add_numerical_features(2);
  internal::ProjectionEvaluator proj(dataset, config_link.numerical_features());

  ASSERT_OK_AND_ASSIGN(auto* label_data,
                       dataset.MutableColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(0));

  const std::vector<float> weights(label_data->values().size(), 1.f);

  internal::LDACache cache;
  ASSERT_OK(cache.ComputeClassification(dt_config, proj, {1, 2}, 3,
                                        label_data->values(), weights));

  std::vector<int> mapping;
  ASSERT_OK(cache.BuildMapping({2}, &mapping));
  EXPECT_THAT(mapping, ElementsAre(1));

  // Those values are similar to the section 3.2 of the "Linear Discriminant
  // Analysis: A Detailed Tutorial" by Tharwat et al.
  std::vector<double> extracted;
  const double eps = 0.001;
  ASSERT_OK(cache.GetSB({2}, &extracted));
  EXPECT_THAT(extracted, Pointwise(DoubleNear(eps), {6.981}));

  ASSERT_OK(cache.GetSB({1, 2}, &extracted));
  EXPECT_THAT(extracted,
              Pointwise(DoubleNear(eps), {7.575, -7.272, -7.272, 6.981}));

  ASSERT_OK(cache.GetSW({1, 2}, &extracted));
  EXPECT_THAT(extracted, Pointwise(DoubleNear(eps), {15.334, 9., 9., 13.201}));
}

TEST(MHLDTOblique, SubtractTransposeMultiplyAdd) {
  const double eps = 0.001;

  std::vector<double> a = {0, 2};
  std::vector<double> b = {2, 1};
  std::vector<double> output(4, 0);
  internal::SubtractTransposeMultiplyAdd(1., absl::MakeSpan(a),
                                         absl::MakeSpan(b), output);
  EXPECT_THAT(output, Pointwise(DoubleNear(eps), {4, -2, -2, 1}));
}

}  // namespace
}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
