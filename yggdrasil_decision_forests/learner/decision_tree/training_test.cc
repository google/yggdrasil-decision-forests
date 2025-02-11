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

#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/gpu.h"
#include "yggdrasil_decision_forests/learner/decision_tree/label.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/learner/decision_tree/preprocessing.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {
namespace {

using test::EqualsProto;
using test::StatusIs;
using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::Pointwise;
using ::yggdrasil_decision_forests::dataset::proto::ColumnType;

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
      dataset.AddColumn("l", ColumnType::CATEGORICAL).value();
  label_col->mutable_categorical()->set_is_already_integerized(true);
  label_col->mutable_categorical()->set_number_of_unique_values(3);
  EXPECT_OK(dataset.AddColumn("f1", ColumnType::NUMERICAL).status());
  EXPECT_OK(dataset.AddColumn("f2", ColumnType::NUMERICAL).status());
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

struct TrainTreeParam {
  std::string name;
  proto::DecisionTreeTrainingConfig::Internal::SortingStrategy sorting_strategy;
};

using TrainTree = testing::TestWithParam<TrainTreeParam>;

TEST_P(TrainTree, Base) {
  const TrainTreeParam& params = GetParam();
  // Dataset

  // We want:
  // "f1">=2.5 [s:0.347222 n:6 np:4 miss:0] ; pred:0.833333
  //     ├─(pos)─ "f2">=1.5 [s:0.0625 n:4 np:2 miss:0] ; pred:1.25
  //     |        ├─(pos)─ pred:1.5
  //     |        └─(neg)─ pred:1
  //     └─(neg)─ pred:0
  dataset::VerticalDataset dataset;
  ASSERT_OK(dataset.AddColumn("l", ColumnType::NUMERICAL).status());
  ASSERT_OK(dataset.AddColumn("f1", ColumnType::NUMERICAL).status());
  ASSERT_OK(dataset.AddColumn("f2", ColumnType::NUMERICAL).status());
  ASSERT_OK(dataset.CreateColumnsFromDataspec());
  dataset.AppendExample({{"l", "0"}, {"f1", "1"}, {"f2", "1"}});
  dataset.AppendExample({{"l", "0"}, {"f1", "2"}, {"f2", "2"}});
  dataset.AppendExample({{"l", "1"}, {"f1", "3"}, {"f2", "1"}});
  dataset.AppendExample({{"l", "1"}, {"f1", "4"}, {"f2", "1"}});
  dataset.AppendExample({{"l", "1.5"}, {"f1", "3"}, {"f2", "2"}});
  dataset.AppendExample({{"l", "1.5"}, {"f1", "4"}, {"f2", "2"}});

  // Training configuration
  const std::vector<UnsignedExampleIdx> selected_examples = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights = {};
  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  proto::DecisionTreeTrainingConfig dt_config;
  const model::proto::DeploymentConfig deployment;
  utils::RandomEngine random;

  config.set_task(model::proto::Task::REGRESSION);
  config_link.set_label(0);
  config_link.add_features(1);
  config_link.add_features(2);
  dt_config.set_min_examples(1);
  dt_config.mutable_axis_aligned_split();
  dt_config.mutable_internal()->set_sorting_strategy(params.sorting_strategy);
  dt_config.mutable_growing_strategy_local();
  dt_config.mutable_categorical()->mutable_cart();
  dt_config.set_num_candidate_attributes(-1);

  // Train a tree

  // Note: preprocessing is required for sorting_strategy=PRESORTED. For
  // sorting_strategy=IN_NODE, preprocessing is ignored.
  ASSERT_OK_AND_ASSIGN(const auto preprocessing,
                       decision_tree::PreprocessTrainingDataset(
                           dataset, config, config_link, dt_config, 1));

  DecisionTree tree;
  ASSERT_OK(DecisionTreeTrain(dataset, selected_examples, config, config_link,
                              dt_config, deployment, weights, &random, &tree,
                              {
                                  .preprocessing = &preprocessing,
                                  .duplicated_selected_examples = false,
                              }));

  std::string description;
  tree.AppendModelStructure(dataset.data_spec(), 0, &description);
  LOG(INFO) << "tree:\n" << description;

  EXPECT_EQ(description,
            R"(    "f1">=2.5 [s:0.347222 n:6 np:4 miss:0] ; pred:0.833333
        ├─(pos)─ "f2">=1.5 [s:0.0625 n:4 np:2 miss:0] ; pred:1.25
        |        ├─(pos)─ pred:1.5
        |        └─(neg)─ pred:1
        └─(neg)─ pred:0
)");
}

TEST_P(TrainTree, DiscretizedNumerical) {
  const TrainTreeParam& params = GetParam();
  // Dataset
  dataset::VerticalDataset dataset;
  ASSERT_OK(dataset.AddColumn("l", ColumnType::NUMERICAL).status());
  ASSERT_OK_AND_ASSIGN(
      auto* col_f1, dataset.AddColumn("f1", ColumnType::DISCRETIZED_NUMERICAL));
  ASSERT_OK_AND_ASSIGN(
      auto* col_f2, dataset.AddColumn("f2", ColumnType::DISCRETIZED_NUMERICAL));

  col_f1->mutable_discretized_numerical()->add_boundaries(0.5);
  col_f1->mutable_discretized_numerical()->add_boundaries(1.5);
  col_f1->mutable_discretized_numerical()->add_boundaries(2.5);
  col_f1->mutable_discretized_numerical()->add_boundaries(3.5);

  col_f2->mutable_discretized_numerical()->add_boundaries(0.5);
  col_f2->mutable_discretized_numerical()->add_boundaries(1.5);

  ASSERT_OK(dataset.CreateColumnsFromDataspec());
  dataset.AppendExample({{"l", "0"}, {"f1", "1"}, {"f2", "1"}});
  dataset.AppendExample({{"l", "0"}, {"f1", "2"}, {"f2", "2"}});
  dataset.AppendExample({{"l", "1"}, {"f1", "3"}, {"f2", "1"}});
  dataset.AppendExample({{"l", "1"}, {"f1", "4"}, {"f2", "1"}});
  dataset.AppendExample({{"l", "1.5"}, {"f1", "3"}, {"f2", "2"}});
  dataset.AppendExample({{"l", "1.5"}, {"f1", "4"}, {"f2", "2"}});

  // Training configuration
  const std::vector<UnsignedExampleIdx> selected_examples = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights = {};
  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  proto::DecisionTreeTrainingConfig dt_config;
  const model::proto::DeploymentConfig deployment;
  utils::RandomEngine random;

  config.set_task(model::proto::Task::REGRESSION);
  config_link.set_label(0);
  config_link.add_features(1);
  config_link.add_features(2);
  dt_config.set_min_examples(1);
  dt_config.mutable_axis_aligned_split();
  dt_config.mutable_internal()->set_sorting_strategy(params.sorting_strategy);
  dt_config.mutable_growing_strategy_local();
  dt_config.mutable_categorical()->mutable_cart();
  dt_config.set_num_candidate_attributes(-1);

  // Train a tree

  // Note: preprocessing is required for sorting_strategy=PRESORTED. For
  // sorting_strategy=IN_NODE, preprocessing is ignored.
  ASSERT_OK_AND_ASSIGN(const auto preprocessing,
                       decision_tree::PreprocessTrainingDataset(
                           dataset, config, config_link, dt_config, 1));

  DecisionTree tree;
  ASSERT_OK(DecisionTreeTrain(dataset, selected_examples, config, config_link,
                              dt_config, deployment, weights, &random, &tree,
                              {
                                  .preprocessing = &preprocessing,
                                  .duplicated_selected_examples = false,
                              }));

  std::string description;
  tree.AppendModelStructure(dataset.data_spec(), 0, &description);
  LOG(INFO) << "tree:\n" << description;

  EXPECT_EQ(
      description,
      R"(    "f1".index >= 3 i.e. "f1" >= 2.5 [s:0.347222 n:6 np:4 miss:0] ; pred:0.833333
        ├─(pos)─ "f2".index >= 2 i.e. "f2" >= 1.5 [s:0.0625 n:4 np:2 miss:0] ; pred:1.25
        |        ├─(pos)─ pred:1.5
        |        └─(neg)─ pred:1
        └─(neg)─ pred:0
)");
}

INSTANTIATE_TEST_SUITE_P(
    TrainTrees, TrainTree,
    testing::ValuesIn<TrainTreeParam>({
        {"presorted", proto::DecisionTreeTrainingConfig::Internal::PRESORTED},
        {"in_node", proto::DecisionTreeTrainingConfig::Internal::IN_NODE},
        {"forced_presorted",
         proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED},
    }),
    [](const testing::TestParamInfo<TrainTree::ParamType>& info) {
      return info.param.name;
    });

TEST(DecisionTreeTrainingTest, SetRegressionLabelDistributionWeighted) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyGradientDataset());
  std::vector<UnsignedExampleIdx> selected_examples = {0, 1, 2, 3};
  std::vector<float> weights = {2.f, 4.f, 6.f, 8.f};
  model::proto::TrainingConfig config;
  config.set_task(model::proto::Task::REGRESSION);
  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(2);  // Gradient column.

  NodeWithChildren node_with_children;
  auto& node = *node_with_children.mutable_node();

  ASSERT_OK(SetLabelDistribution(dataset, selected_examples, weights, config,
                                 config_link, &node_with_children));
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
  config.set_task(model::proto::Task::REGRESSION);
  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(2);  // Gradient column.

  NodeWithChildren node_with_children;
  auto& node = *node_with_children.mutable_node();

  ASSERT_OK(SetLabelDistribution(dataset, selected_examples, weights, config,
                                 config_link, &node_with_children));
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
  model::proto::TrainingConfig config;
  config.set_task(model::proto::Task::REGRESSION);
  model::proto::TrainingConfigLinking config_link;

  NodeWithChildren node_with_children;
  EXPECT_THAT(SetLabelDistribution(dataset, selected_examples, weights, config,
                                   config_link, &node_with_children),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SparseOblique, Classification) {
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

TEST(SparseOblique, ClassificationMaxNumFeatures) {
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
  dt_config.mutable_sparse_oblique_split()->set_max_num_features(1);

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
  EXPECT_EQ(best_condition.condition().oblique_condition().attributes_size(),
            1);
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

  LOG(INFO) << "best_condition:\n" << best_condition.DebugString();

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

  LOG(INFO) << "best_condition:\n" << best_condition.DebugString();

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

TEST(VectorSequenceCondition, Classification) {
  const model::proto::TrainingConfig config;

  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(0);
  config_link.add_features(1);

  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.set_min_examples(1);

  dataset::VerticalDataset dataset;
  auto* label_spec =
      dataset::AddColumn("l", dataset::proto::ColumnType::CATEGORICAL,
                         dataset.mutable_data_spec());
  label_spec->mutable_categorical()->set_is_already_integerized(true);
  label_spec->mutable_categorical()->set_number_of_unique_values(3);
  auto* feature_spec = dataset::AddColumn(
      "f1", dataset::proto::ColumnType::NUMERICAL_VECTOR_SEQUENCE,
      dataset.mutable_data_spec());
  feature_spec->mutable_numerical_vector_sequence()->set_vector_length(2);

  EXPECT_OK(dataset.CreateColumnsFromDataspec());

  ASSERT_OK_AND_ASSIGN(auto* label_col,
                       dataset.MutableColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(0));
  ASSERT_OK_AND_ASSIGN(
      auto* feature_col,
      dataset.MutableColumnWithCastWithStatus<
          dataset::VerticalDataset::NumericalVectorSequenceColumn>(1));

  label_col->Add(1);
  label_col->Add(1);
  label_col->Add(2);
  label_col->Add(2);

  feature_col->Add({0.f, 0.f});
  feature_col->Add({0.f, 0.f, 1.f, 1.f});
  feature_col->Add({0.f, 0.f, 0.5f, 0.5f});
  feature_col->Add({1.f, 1.f, 0.6f, 0.6f});

  dataset.set_nrow(4);

  std::vector<UnsignedExampleIdx> selected_examples(dataset.nrow());
  std::iota(selected_examples.begin(), selected_examples.end(), 0);
  const std::vector<float> weights(selected_examples.size(), 1.f);

  ClassificationLabelStats label_stats(label_col->values());
  label_stats.num_label_classes = 3;
  label_stats.label_distribution.SetNumClasses(3);
  for (const auto example_idx : selected_examples) {
    label_stats.label_distribution.Add(label_col->values()[example_idx],
                                       weights[example_idx]);
  }

  ASSERT_OK_AND_ASSIGN(auto vector_sequence_computer,
                       decision_tree::gpu::VectorSequenceComputer::Create(
                           {nullptr, feature_col}, /*use_gpu=*/false));

  proto::Node parent;
  InternalTrainConfig internal_config;
  internal_config.vector_sequence_computer = vector_sequence_computer.get();
  proto::NodeCondition condition;
  SplitterPerThreadCache cache;
  utils::RandomEngine random;
  const auto found_condition = FindBestConditionClassification(
      dataset, selected_examples, weights, config, config_link, dt_config,
      parent, internal_config, label_stats, 1, {}, &condition, &random, &cache);

  LOG(INFO) << "condition:\n" << condition.DebugString();
  const proto::NodeCondition expected_condition = PARSE_TEST_PROTO(R"pb(
    na_value: true
    attribute: 1
    condition {
      numerical_vector_sequence {
        closer_than {
          anchor { grounded: 0.6 grounded: 0.6 }
          threshold2: 0.16999999
        }
      }
    }
    num_training_examples_without_weight: 4
    num_training_examples_with_weight: 4
    split_score: 0.6931472
    num_pos_training_examples_without_weight: 2
    num_pos_training_examples_with_weight: 2
  )pb");
  EXPECT_EQ(found_condition, SplitSearchResult::kBetterSplitFound);
  EXPECT_THAT(condition, EqualsProto(expected_condition));

  ASSERT_OK(vector_sequence_computer->Release());
}

TEST(SplitExamplesInPlace, Base) {
  dataset::VerticalDataset dataset;
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "a" }
  )pb");
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "1"}}));
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "3"}}));
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "2"}}));
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "4"}}));

  std::vector<UnsignedExampleIdx> examples = {0, 1, 2, 3};
  std::vector<UnsignedExampleIdx> buffer;
  auto examples_rb =
      SelectedExamplesRollingBuffer::Create(absl::MakeSpan(examples), &buffer);

  proto::NodeCondition condition = PARSE_TEST_PROTO(R"pb(
    attribute: 0
    condition { higher_condition { threshold: 2.5 } }
    num_pos_training_examples_without_weight: 2
  )pb");

  ASSERT_OK_AND_ASSIGN(auto example_split,
                       internal::SplitExamplesInPlace(
                           dataset, examples_rb, condition,
                           /*dataset_is_dense=*/false,
                           /*error_on_wrong_splitter_statistics=*/true,
                           /*examples_are_training_examples=*/true));

  EXPECT_THAT(example_split.positive_examples.active, ElementsAre(1, 3));
  EXPECT_THAT(example_split.negative_examples.active, ElementsAre(0, 2));

  EXPECT_THAT(example_split.positive_examples.inactive, ElementsAre(0, 1));
  EXPECT_THAT(example_split.negative_examples.inactive, ElementsAre(2, 3));
}

}  // namespace
}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
