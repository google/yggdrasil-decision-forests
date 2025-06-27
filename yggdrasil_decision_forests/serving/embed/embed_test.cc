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

// Test the code to embed models.

#include "yggdrasil_decision_forests/serving/embed/embed.h"

#include <memory>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/builder.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::serving::embed {
namespace {

std::string TestDataDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/test_data");
}

struct TestData {
  std::unique_ptr<model::AbstractModel> model;
};

TestData BuildToyTestData() {
  auto model = std::make_unique<
      model::gradient_boosted_trees::GradientBoostedTreesModel>();

  dataset::AddCategoricalColumn("l", {"a", "b"}, model->mutable_data_spec())
      ->set_dtype(dataset::proto::DTYPE_INT8);
  dataset::AddNumericalColumn("f1", model->mutable_data_spec())
      ->set_dtype(dataset::proto::DTYPE_INT16);
  dataset::AddCategoricalColumn("f2", {"x", "y", "z"},
                                model->mutable_data_spec());

  model->set_task(model::proto::Task::CLASSIFICATION);
  model->set_label_col_idx(0);
  model->mutable_input_features()->push_back(1);
  model->mutable_input_features()->push_back(2);
  model->set_initial_predictions({1.f});
  model->set_loss(model::gradient_boosted_trees::proto::Loss::SQUARED_ERROR,
                  {});
  model->set_num_trees_per_iter(1);

  {
    auto tree = std::make_unique<model::decision_tree::DecisionTree>();
    model::decision_tree::TreeBuilder root(tree.get());
    auto [nl1, l1] = root.ConditionIsGreater(1, 1);
    auto [l2, nl2] = nl1.ConditionIsGreater(1, 2);
    auto [l3, l4] = nl2.ConditionContains(2, {1, 3});
    l1.LeafRegression(2);
    l2.LeafRegression(4);
    l3.LeafRegression(5);
    l4.LeafRegression(6);
    tree->SetLeafIndices();
    model->AddTree(std::move(tree));
  }

  {
    auto tree = std::make_unique<model::decision_tree::DecisionTree>();
    model::decision_tree::TreeBuilder root(tree.get());
    auto [l2, l1] = root.ConditionIsGreater(1, 2);
    l1.LeafRegression(2);
    l2.LeafRegression(4);
    tree->SetLeafIndices();
    model->AddTree(std::move(tree));
  }

  return TestData{.model = std::move(model)};
};

TEST(Embed, AdultBinaryGBTClass) {
  ASSERT_OK_AND_ASSIGN(
      auto model, model::LoadModel(file::JoinPath(
                      TestDataDir(), "model", "adult_binary_class_gbdt_v2")));
  auto model_gbt =
      dynamic_cast<model::gradient_boosted_trees::GradientBoostedTreesModel*>(
          model.get());
  model_gbt->mutable_decision_trees()->resize(3);

  ASSERT_OK_AND_ASSIGN(const auto embed, EmbedModelCC(*model));
  EXPECT_EQ(embed.size(), 1);
  EXPECT_TRUE(embed.contains("my_model.h"));

  test::ExpectEqualGolden(
      embed.at("my_model.h"),
      "yggdrasil_decision_forests/test_data/"
      "golden/embed/adult_binary_class_gbdt_v2_class.h.golden");
}

TEST(Embed, AdultBinaryGBTScore) {
  ASSERT_OK_AND_ASSIGN(const auto model, model::LoadModel(file::JoinPath(
                                             TestDataDir(), "model",
                                             "adult_binary_class_gbdt_v2")));
  auto model_gbt =
      dynamic_cast<model::gradient_boosted_trees::GradientBoostedTreesModel*>(
          model.get());
  model_gbt->mutable_decision_trees()->resize(3);

  proto::Options options;
  options.set_classification_output(proto::ClassificationOutput::SCORE);
  ASSERT_OK_AND_ASSIGN(const auto embed, EmbedModelCC(*model, options));
  EXPECT_EQ(embed.size(), 1);
  EXPECT_TRUE(embed.contains("my_model.h"));

  test::ExpectEqualGolden(
      embed.at("my_model.h"),
      "yggdrasil_decision_forests/test_data/"
      "golden/embed/adult_binary_class_gbdt_v2_score.h.golden");
}

TEST(Embed, AdultBinaryGBTProbability) {
  ASSERT_OK_AND_ASSIGN(const auto model, model::LoadModel(file::JoinPath(
                                             TestDataDir(), "model",
                                             "adult_binary_class_gbdt_v2")));
  auto model_gbt =
      dynamic_cast<model::gradient_boosted_trees::GradientBoostedTreesModel*>(
          model.get());
  model_gbt->mutable_decision_trees()->resize(3);
  proto::Options options;
  options.set_classification_output(proto::ClassificationOutput::PROBABILITY);
  ASSERT_OK_AND_ASSIGN(const auto embed, EmbedModelCC(*model, options));
  EXPECT_EQ(embed.size(), 1);
  EXPECT_TRUE(embed.contains("my_model.h"));

  test::ExpectEqualGolden(
      embed.at("my_model.h"),
      "yggdrasil_decision_forests/test_data/"
      "golden/embed/adult_binary_class_gbdt_v2_probability.h.golden");
}

TEST(Embed, AdultBinaryGBTProbabilityRouting) {
  ASSERT_OK_AND_ASSIGN(const auto model, model::LoadModel(file::JoinPath(
                                             TestDataDir(), "model",
                                             "adult_binary_class_gbdt_v2")));
  auto model_gbt =
      dynamic_cast<model::gradient_boosted_trees::GradientBoostedTreesModel*>(
          model.get());
  model_gbt->mutable_decision_trees()->resize(3);
  proto::Options options;
  options.set_classification_output(proto::ClassificationOutput::PROBABILITY);
  options.set_algorithm(proto::Algorithm::ROUTING);
  ASSERT_OK_AND_ASSIGN(const auto embed, EmbedModelCC(*model, options));
  EXPECT_EQ(embed.size(), 1);
  EXPECT_TRUE(embed.contains("my_model.h"));

  test::ExpectEqualGolden(
      embed.at("my_model.h"),
      "yggdrasil_decision_forests/test_data/"
      "golden/embed/adult_binary_class_gbdt_v2_probability_routing.h.golden");
}

TEST(Process, ManualBinaryGBT) {
  const auto test_data = BuildToyTestData();

  const auto* df = dynamic_cast<const model::DecisionForestInterface*>(
      test_data.model.get());
  ASSERT_TRUE(df);
  ASSERT_OK_AND_ASSIGN(const auto stats,
                       internal::ComputeStatistics(*test_data.model, *df));

  EXPECT_EQ(stats.num_features, 2);
  EXPECT_EQ(stats.num_trees, 2);
  EXPECT_EQ(stats.num_leaves, 6);
  EXPECT_EQ(stats.max_num_leaves_per_tree, 4);
  EXPECT_EQ(stats.max_depth, 3);
  EXPECT_TRUE(stats.is_classification());
  EXPECT_TRUE(stats.is_binary_classification());
  EXPECT_EQ(stats.num_classification_classes, 2);
  EXPECT_EQ(stats.num_conditions, 4);
  EXPECT_EQ(stats.task, model::proto::Task::CLASSIFICATION);

  EXPECT_TRUE(stats.has_conditions
                  [model::decision_tree::proto::Condition::kHigherCondition]);
  EXPECT_TRUE(stats.has_conditions
                  [model::decision_tree::proto::Condition::kContainsCondition]);
  EXPECT_FALSE(
      stats.has_conditions
          [model::decision_tree::proto::Condition::kTrueValueCondition]);

  ASSERT_OK_AND_ASSIGN(
      const auto internal_options,
      internal::ComputeInternalOptions(*test_data.model, *df, stats, {}));
  EXPECT_EQ(internal_options.feature_value_bytes, 2);
  EXPECT_EQ(internal_options.numerical_feature_is_float, false);

  EXPECT_EQ(internal_options.categorical_dicts.size(), 2);

  EXPECT_EQ(internal_options.categorical_dicts.at(0).sanitized_name, "Label");
  EXPECT_EQ(internal_options.categorical_dicts.at(0).is_label, true);
  EXPECT_THAT(internal_options.categorical_dicts.at(0).sanitized_items,
              testing::ElementsAre("A", "B"));

  EXPECT_EQ(internal_options.categorical_dicts.at(2).sanitized_name, "F2");
  EXPECT_EQ(internal_options.categorical_dicts.at(2).is_label, false);
  EXPECT_THAT(internal_options.categorical_dicts.at(2).sanitized_items,
              testing::ElementsAre("OutOfVocabulary", "X", "Y", "Z"));

  EXPECT_EQ(internal_options.feature_index_bytes, 1);
  EXPECT_EQ(internal_options.tree_index_bytes, 1);
  EXPECT_EQ(internal_options.node_index_bytes, 1);
  EXPECT_EQ(internal_options.node_offset_bytes, 1);
}

TEST(Process, RealBinaryGBT) {
  ASSERT_OK_AND_ASSIGN(const auto model, model::LoadModel(file::JoinPath(
                                             TestDataDir(), "model",
                                             "adult_binary_class_gbdt_v2")));
  const auto* df =
      dynamic_cast<const model::DecisionForestInterface*>(model.get());
  ASSERT_TRUE(df);
  ASSERT_OK_AND_ASSIGN(const auto stats,
                       internal::ComputeStatistics(*model, *df));

  EXPECT_EQ(stats.num_features, 14);
  EXPECT_EQ(stats.num_trees, 163);
  EXPECT_EQ(stats.num_leaves, 4476);
  EXPECT_EQ(stats.max_num_leaves_per_tree, 32);
  EXPECT_EQ(stats.max_depth, 5);
  EXPECT_TRUE(stats.is_classification());
  EXPECT_TRUE(stats.is_binary_classification());
  EXPECT_EQ(stats.num_classification_classes, 2);

  ASSERT_OK_AND_ASSIGN(
      const auto internal_options,
      internal::ComputeInternalOptions(*model, *df, stats, {}));
  EXPECT_EQ(internal_options.feature_value_bytes, 4);
  EXPECT_EQ(internal_options.numerical_feature_is_float, false);

  EXPECT_EQ(internal_options.feature_index_bytes, 1);
  EXPECT_EQ(internal_options.tree_index_bytes, 1);
  EXPECT_EQ(internal_options.node_index_bytes, 2);
  EXPECT_EQ(internal_options.node_offset_bytes, 1);
}

TEST(Process, RealMultiClassGBT) {
  ASSERT_OK_AND_ASSIGN(const auto model, model::LoadModel(file::JoinPath(
                                             TestDataDir(), "model",
                                             "iris_multi_class_gbdt_v2")));
  const auto* df =
      dynamic_cast<const model::DecisionForestInterface*>(model.get());
  ASSERT_TRUE(df);
  ASSERT_OK_AND_ASSIGN(const auto stats,
                       internal::ComputeStatistics(*model, *df));

  EXPECT_EQ(stats.num_features, 4);
  EXPECT_EQ(stats.num_trees, 54);
  EXPECT_EQ(stats.num_leaves, 611);
  EXPECT_EQ(stats.max_num_leaves_per_tree, 17);
  EXPECT_EQ(stats.max_depth, 5);
  EXPECT_TRUE(stats.is_classification());
  EXPECT_FALSE(stats.is_binary_classification());
  EXPECT_EQ(stats.num_classification_classes, 3);

  ASSERT_OK_AND_ASSIGN(
      const auto internal_options,
      internal::ComputeInternalOptions(*model, *df, stats, {}));
  EXPECT_EQ(internal_options.feature_value_bytes, 4);
  EXPECT_EQ(internal_options.numerical_feature_is_float, true);
}

struct ComputeInternalOptionsOutputCase {
  proto::Options options;
  internal::ModelStatistics stats;
  std::string expected_output_type;
};

SIMPLE_PARAMETERIZED_TEST(
    ComputeInternalOptionsOutput, ComputeInternalOptionsOutputCase,
    {
        // Classification + class
        {{},
         {.task = model::proto::Task::CLASSIFICATION,
          .num_classification_classes = 2},
         "bool"},

        {{},
         {.task = model::proto::Task::CLASSIFICATION,
          .num_classification_classes = 4},
         "uint8_t"},

        {{},
         {.task = model::proto::Task::CLASSIFICATION,
          .num_classification_classes = 299},
         "uint16_t"},

        // Classification + score
        {PARSE_TEST_PROTO("classification_output: SCORE"),
         {.task = model::proto::Task::CLASSIFICATION,
          .num_classification_classes = 2},
         ""},

        // Classification + probability
        {PARSE_TEST_PROTO("classification_output:PROBABILITY"),
         {.task = model::proto::Task::CLASSIFICATION,
          .num_classification_classes = 2},
         "float"},

        {PARSE_TEST_PROTO("classification_output:PROBABILITY"),
         {.task = model::proto::Task::CLASSIFICATION,
          .num_classification_classes = 4},
         "std::array<float, 4>"},

        // Regression
        {{}, {.task = model::proto::Task::REGRESSION}, "float"},

        {PARSE_TEST_PROTO("integerize_output:true"),
         {.task = model::proto::Task::REGRESSION},
         "int16_t"},
    }) {
  const auto& test_case = GetParam();
  internal::InternalOptions internal_options;
  ASSERT_OK(internal::ComputeInternalOptionsOutput(
      test_case.stats, test_case.options, &internal_options));
  EXPECT_EQ(internal_options.output_type, test_case.expected_output_type);
}

struct GenFeatureDefCase {
  dataset::proto::Column col_spec;
  internal::InternalOptions internal_options;
  std::string expected_type;
  absl::optional<std::string> expected_default_value;
};

SIMPLE_PARAMETERIZED_TEST(
    GenFeatureDef, GenFeatureDefCase,
    {
        {PARSE_TEST_PROTO("type: NUMERICAL"),
         {.feature_value_bytes = 4, .numerical_feature_is_float = true},
         "float",
         {}},

        {PARSE_TEST_PROTO("type: NUMERICAL"),
         {.feature_value_bytes = 4, .numerical_feature_is_float = false},
         "int32_t",
         {}},

        {PARSE_TEST_PROTO("type: NUMERICAL"),
         {.feature_value_bytes = 2, .numerical_feature_is_float = false},
         "int16_t",
         {}},

        {PARSE_TEST_PROTO("type: NUMERICAL"),
         {.feature_value_bytes = 1, .numerical_feature_is_float = false},
         "int8_t",
         {}},

        {PARSE_TEST_PROTO("type: CATEGORICAL"),
         {.feature_value_bytes = 4},
         "uint32_t",
         {}},

        {PARSE_TEST_PROTO("type: CATEGORICAL"),
         {.feature_value_bytes = 2},
         "uint16_t",
         {}},

        {PARSE_TEST_PROTO("type: CATEGORICAL"),
         {.feature_value_bytes = 1},
         "uint8_t",
         {}},

        {PARSE_TEST_PROTO("type: BOOLEAN"),
         {.feature_value_bytes = 2},
         "uint16_t",
         {}},
    }) {
  const auto& test_case = GetParam();
  ASSERT_OK_AND_ASSIGN(
      const auto value,
      internal::GenFeatureDef(test_case.col_spec, test_case.internal_options));
  EXPECT_EQ(value.type, test_case.expected_type);
  EXPECT_EQ(value.default_value, test_case.expected_default_value);
}

}  // namespace
}  // namespace yggdrasil_decision_forests::serving::embed
