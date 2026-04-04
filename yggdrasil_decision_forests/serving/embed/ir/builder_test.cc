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

#include "yggdrasil_decision_forests/serving/embed/ir/builder.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/isolation_forest/isolation_forest.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/serving/embed/ir/model_ir.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

using ::testing::ElementsAre;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

namespace yggdrasil_decision_forests::serving::embed::internal {
namespace {

model::random_forest::RandomForestModel BuildToyModel() {
  model::random_forest::RandomForestModel model;
  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "num" dtype: DTYPE_FLOAT32 }
    columns {
      type: CATEGORICAL
      name: "cat_int"
      categorical { is_already_integerized: true number_of_unique_values: 3 }
    }
    columns {
      type: CATEGORICAL
      name: "cat_voc"
      categorical {
        most_frequent_value: 0
        number_of_unique_values: 3
        is_already_integerized: false
        items {
          key: "a"
          value { index: 0 count: 10 }
        }
        items {
          key: "b"
          value { index: 1 count: 9 }
        }
        items {
          key: "c"
          value { index: 2 count: 8 }
        }
      }
    }
  )pb");
  model.set_task(model::proto::REGRESSION);
  model.set_data_spec(dataspec);
  *model.mutable_input_features() = {0, 1, 2};
  model.set_label_col_idx(0);
  model.set_winner_take_all_inference(false);
  {
    auto tree = std::make_unique<model::decision_tree::DecisionTree>();
    tree->CreateRoot();
    tree->mutable_root()->mutable_node()->mutable_regressor()->set_top_value(
        1.0);
    model.AddTree(std::move(tree));
  }
  {
    auto tree = std::make_unique<model::decision_tree::DecisionTree>();
    tree->CreateRoot();
    tree->mutable_root()->mutable_node()->mutable_regressor()->set_top_value(
        2.0);
    model.AddTree(std::move(tree));
  }
  return model;
}

TEST(ModelIRBuilder, NumTrees) {
  model::random_forest::RandomForestModel model = BuildToyModel();
  proto::Options options;
  ASSERT_OK_AND_ASSIGN(const auto model_ir,
                       ModelIRBuilder::Build(model, options));
  EXPECT_EQ(model_ir.num_trees, 2);
}

TEST(ModelIRBuilder, IsolationForestNotSupported) {
  model::isolation_forest::IsolationForestModel model;
  proto::Options options;
  EXPECT_FALSE(ModelIRBuilder::Build(model, options).ok());
}

TEST(ModelIRBuilder, Features) {
  model::random_forest::RandomForestModel model = BuildToyModel();
  proto::Options options;
  ASSERT_OK_AND_ASSIGN(const auto model_ir,
                       ModelIRBuilder::Build(model, options));

  EXPECT_EQ(model_ir.task, ModelIR::Task::kRegression);
  EXPECT_EQ(model_ir.activation, ModelIR::Activation::kEquality);
  EXPECT_EQ(model_ir.num_output_classes, 1);  // Default for regression
  EXPECT_THAT(model_ir.accumulator_initialization, SizeIs(1));

  ASSERT_THAT(model_ir.features, SizeIs(3));

  const auto& f0 = model_ir.features[0];
  EXPECT_EQ(f0.type, FeatureInfo::Type::kNumerical);
  EXPECT_EQ(f0.original_name, "num");
  EXPECT_TRUE(f0.is_label);
  EXPECT_TRUE(f0.is_float);

  const auto& f1 = model_ir.features[1];
  EXPECT_EQ(f1.type, FeatureInfo::Type::kIntegerizedCategorical);
  EXPECT_EQ(f1.original_name, "cat_int");
  EXPECT_FALSE(f1.is_label);
  ASSERT_TRUE(f1.maximum_value.has_value());
  EXPECT_EQ(*f1.maximum_value, 2);

  const auto& f2 = model_ir.features[2];
  EXPECT_EQ(f2.type, FeatureInfo::Type::kCategorical);
  EXPECT_EQ(f2.original_name, "cat_voc");
  EXPECT_FALSE(f2.is_label);
  EXPECT_THAT(f2.vocabulary, UnorderedElementsAre("a", "b", "c"));
}

TEST(ModelIRBuilder, UnsupportedFeatureType) {
  model::random_forest::RandomForestModel model;
  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns { type: CATEGORICAL_SET name: "cat_set" }
  )pb");
  model.set_data_spec(dataspec);
  *model.mutable_input_features() = {0};
  model.set_label_col_idx(
      0);  // Set label to same column just to have a valid model structure
  // Add a dummy tree
  auto tree = std::make_unique<model::decision_tree::DecisionTree>();
  tree->CreateRoot();
  tree->mutable_root()->mutable_node()->mutable_regressor()->set_top_value(1.0);
  model.AddTree(std::move(tree));

  proto::Options options;
  EXPECT_FALSE(ModelIRBuilder::Build(model, options).ok());
}

TEST(ModelIRBuilder, ClassificationBinary) {
  model::random_forest::RandomForestModel model;
  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns {
      type: CATEGORICAL
      name: "label"
      categorical {
        is_already_integerized: true
        number_of_unique_values: 3  # 0=OOD, 1=False, 2=True
      }
    }
  )pb");
  model.set_task(model::proto::CLASSIFICATION);
  model.set_data_spec(dataspec);
  *model.mutable_input_features() = {};
  model.set_label_col_idx(0);
  model.set_winner_take_all_inference(false);

  // Add a dummy tree
  auto tree = std::make_unique<model::decision_tree::DecisionTree>();
  tree->CreateRoot();
  tree->mutable_root()->mutable_node()->mutable_regressor()->set_top_value(1.0);
  model.AddTree(std::move(tree));

  proto::Options options;
  ASSERT_OK_AND_ASSIGN(const auto model_ir,
                       ModelIRBuilder::Build(model, options));

  EXPECT_EQ(model_ir.task, ModelIR::Task::kBinaryClassification);
  EXPECT_EQ(model_ir.num_output_classes, 1);
  EXPECT_EQ(model_ir.activation, ModelIR::Activation::kEquality);
}

TEST(ModelIRBuilder, ClassificationMulticlass) {
  model::random_forest::RandomForestModel model;
  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns {
      type: CATEGORICAL
      name: "label"
      categorical {
        is_already_integerized: true
        number_of_unique_values: 4  # 0=OOD, 1=A, 2=B, 3=C
      }
    }
  )pb");
  model.set_task(model::proto::CLASSIFICATION);
  model.set_data_spec(dataspec);
  *model.mutable_input_features() = {};
  model.set_label_col_idx(0);
  model.set_winner_take_all_inference(true);

  // Add a dummy tree
  auto tree = std::make_unique<model::decision_tree::DecisionTree>();
  tree->CreateRoot();
  tree->mutable_root()->mutable_node()->mutable_classifier()->set_top_value(1);
  model.AddTree(std::move(tree));

  proto::Options options;
  ASSERT_OK_AND_ASSIGN(const auto model_ir,
                       ModelIRBuilder::Build(model, options));

  EXPECT_EQ(model_ir.task, ModelIR::Task::kMulticlassClassification);
  EXPECT_EQ(model_ir.num_output_classes, 3);
  EXPECT_THAT(model_ir.accumulator_initialization, ElementsAre(0, 0, 0));
  EXPECT_EQ(model_ir.activation, ModelIR::Activation::kEquality);
}

TEST(ModelIRBuilder, GradientBoostedTrees) {
  model::gradient_boosted_trees::GradientBoostedTreesModel model;
  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "label" }
  )pb");
  model.set_task(model::proto::REGRESSION);
  model.set_data_spec(dataspec);
  *model.mutable_input_features() = {};
  model.set_label_col_idx(0);
  model.set_initial_predictions({1.5f});

  // Add a dummy tree
  auto tree = std::make_unique<model::decision_tree::DecisionTree>();
  tree->CreateRoot();
  tree->mutable_root()->mutable_node()->mutable_regressor()->set_top_value(1.0);
  model.AddTree(std::move(tree));

  proto::Options options;
  ASSERT_OK_AND_ASSIGN(const auto model_ir,
                       ModelIRBuilder::Build(model, options));

  EXPECT_EQ(model_ir.task, ModelIR::Task::kRegression);
  EXPECT_EQ(model_ir.activation, ModelIR::Activation::kEquality);
  EXPECT_THAT(model_ir.accumulator_initialization,
              UnorderedElementsAre(std::variant<double, int64_t>(1.5)));
}

TEST(ModelIRBuilder, MulticlassDistribution) {
  model::random_forest::RandomForestModel model;
  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns {
      type: CATEGORICAL
      name: "label"
      categorical {
        is_already_integerized: true
        number_of_unique_values: 4  # 0=OOD, 1=A, 2=B, 3=C
      }
    }
  )pb");
  model.set_task(model::proto::CLASSIFICATION);
  model.set_data_spec(dataspec);
  *model.mutable_input_features() = {};
  model.set_label_col_idx(0);
  model.set_winner_take_all_inference(false);

  auto tree = std::make_unique<model::decision_tree::DecisionTree>();
  tree->CreateRoot();
  // Leaf with distribution: Class 1 (10), Class 2 (20), Class 3 (30). Total 60.
  auto* dist = tree->mutable_root()
                   ->mutable_node()
                   ->mutable_classifier()
                   ->mutable_distribution();
  dist->set_sum(60.0);
  dist->add_counts(0);   // OOD
  dist->add_counts(10);  // Class 1
  dist->add_counts(20);  // Class 2
  dist->add_counts(30);  // Class 3

  model.AddTree(std::move(tree));

  proto::Options options;
  ASSERT_OK_AND_ASSIGN(const auto model_ir,
                       ModelIRBuilder::Build(model, options));

  EXPECT_EQ(model_ir.task, ModelIR::Task::kMulticlassClassification);
  // Leaf bank should contain 3 probabilities.
  ASSERT_THAT(model_ir.leaf_value_bank, SizeIs(3));

  ASSERT_THAT(model_ir.nodes, SizeIs(1));
  const auto& node = model_ir.nodes[0];
  EXPECT_EQ(node.type, Node::Type::kLeaf);
  // Offset should be 0.
  EXPECT_EQ(std::get<int64_t>(node.threshold_or_offset), 0);
}

}  // namespace

}  // namespace yggdrasil_decision_forests::serving::embed::internal
