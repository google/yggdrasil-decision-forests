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

#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/decision_tree/builder.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {
namespace {

using row_t = dataset::VerticalDataset::row_t;
using ::testing::ElementsAre;
using ::yggdrasil_decision_forests::dataset::proto::ColumnType;
using ::yggdrasil_decision_forests::dataset::proto::DataSpecification;

std::string DatasetDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/"
                        "test_data/dataset");
}

TEST(DecisionTree, GetLeafAndGetPath) {
  DecisionTree tree;
  tree.CreateRoot();
  tree.mutable_root()->CreateChildren();
  tree.mutable_root()->mutable_node()->mutable_condition()->set_attribute(0);
  tree.mutable_root()
      ->mutable_node()
      ->mutable_condition()
      ->mutable_condition()
      ->mutable_higher_condition()
      ->set_threshold(1);

  DataSpecification dataspec;
  auto* col_spec = dataspec.add_columns();
  col_spec->set_name("a");
  col_spec->set_type(ColumnType::NUMERICAL);

  dataset::VerticalDataset dataset;
  dataset.set_data_spec(dataspec);
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  auto* col = dataset
                  .MutableColumnWithCastWithStatus<
                      dataset::VerticalDataset::NumericalColumn>(0)
                  .value();
  col->Add(0);
  col->Add(2);
  dataset.set_nrow(2);

  // We get the first positive child.
  CHECK_EQ(&tree.GetLeaf(dataset, 0), &tree.root().neg_child()->node());
  CHECK_EQ(&tree.GetLeaf(dataset, 1), &tree.root().pos_child()->node());

  {
    std::vector<const NodeWithChildren*> path;
    tree.GetPath(dataset, 0, &path);
    CHECK_EQ(path.size(), 2);
    CHECK_EQ(path[0], &tree.root());
    CHECK_EQ(path[1], tree.root().neg_child());
  }

  {
    std::vector<const NodeWithChildren*> path;
    tree.GetPath(dataset, 1, &path);
    CHECK_EQ(path.size(), 2);
    CHECK_EQ(path[0], &tree.root());
    CHECK_EQ(path[1], tree.root().pos_child());
  }

  // Switch the 0 and 1 rows.
  CHECK_EQ(&tree.GetLeafWithSwappedAttribute(dataset, 0, 0, 1),
           &tree.root().pos_child()->node());
  CHECK_EQ(&tree.GetLeafWithSwappedAttribute(dataset, 1, 0, 0),
           &tree.root().neg_child()->node());

  // Replace the row 0 and 1 by themself i.e. equivalent to "GetLeaf".
  CHECK_EQ(&tree.GetLeafWithSwappedAttribute(dataset, 0, 0, 0),
           &tree.root().neg_child()->node());
  CHECK_EQ(&tree.GetLeafWithSwappedAttribute(dataset, 1, 0, 1),
           &tree.root().pos_child()->node());

  // Switch the row 0 and 1 for an unused attribute.
  CHECK_EQ(&tree.GetLeafWithSwappedAttribute(dataset, 0, 2, 1),
           &tree.root().neg_child()->node());
  CHECK_EQ(&tree.GetLeafWithSwappedAttribute(dataset, 1, 2, 0),
           &tree.root().pos_child()->node());
}

class EvalConditions : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string toy_dataset_path =
        absl::StrCat("csv:", file::JoinPath(DatasetDir(), "toy.csv"));
    dataset::proto::DataSpecificationGuide guide;
    guide.mutable_default_column_guide()
        ->mutable_categorial()
        ->set_min_vocab_frequency(1);
    DataSpecification data_spec;
    dataset::CreateDataSpec(toy_dataset_path, false, guide, &data_spec);
    CHECK_OK(LoadVerticalDataset(toy_dataset_path, data_spec, &dataset_));
  }

  // Check the evaluation of a condition on VerticalDataset and proto::Example.
  // Returns the string representation of the condition.
  std::string CheckCondition(absl::string_view text_proto_condition,
                             const int dataset_row,
                             const bool expected_result) {
    const proto::NodeCondition condition =
        PARSE_TEST_PROTO(text_proto_condition);
    EXPECT_EQ(EvalCondition(condition, dataset_, dataset_row), expected_result);
    dataset::proto::Example example;
    dataset_.ExtractExample(dataset_row, &example);
    EXPECT_EQ(EvalCondition(condition, example), expected_result);

    std::string description;
    AppendConditionDescription(dataset_.data_spec(), condition, &description);
    return description;
  }

  dataset::VerticalDataset dataset_;
};

TEST_F(EvalConditions, EvalConditionNA) {
  // The tested value is NA.
  CheckCondition(
      R"(
      na_value: true
      attribute: 1
      condition { na_condition {} }
      )",
      0, true);

  // The tested value is not NA.
  CheckCondition(
      R"(
      na_value: true
      attribute: 1
      condition { na_condition {} }
      )",
      1, false);
}

TEST_F(EvalConditions, EvalConditionHigher) {
  // The tested value is 2.
  CheckCondition(
      R"(
      na_value: true
      attribute: 1
      condition { higher_condition { threshold: 1 } }
      )",
      1, true);

  CheckCondition(
      R"(
      na_value: true
      attribute: 1
      condition { higher_condition { threshold: 3 } }
      )",
      1, false);

  // The tested value is NA.
  CheckCondition(
      R"(
      na_value: true
      attribute: 1
      condition { higher_condition { threshold: 1 } }
      )",
      0, true);

  CheckCondition(
      R"(
      na_value: false
      attribute: 1
      condition { higher_condition { threshold: 1 } }
      )",
      0, false);
}

TEST_F(EvalConditions, EvalConditionTrueValue) {
  // The tested value is false.
  CheckCondition(
      R"(
      na_value: true
      attribute: 6
      condition { true_value_condition {} }
      )",
      0, false);

  // The tested value is true.
  CheckCondition(
      R"(
      na_value: true
      attribute: 6
      condition { true_value_condition {} }
      )",
      1, true);
}

TEST_F(EvalConditions, EvalConditionContainsVector) {
  // The tested value is A=1.
  CheckCondition(
      R"(
      na_value: true
      attribute: 2
      condition { contains_condition { elements: 1 elements: 2 } }
      )",
      0, true);

  CheckCondition(
      R"(
      na_value: true
      attribute: 2
      condition { contains_condition { elements: 2 elements: 3 } }
      )",
      0, false);

  // The tested value is B=2.
  CheckCondition(
      R"(
      na_value: true
      attribute: 2
      condition { contains_condition { elements: 2 elements: 3 } }
      )",
      1, true);
}

TEST_F(EvalConditions, EvalConditionContainsVectorSet) {
  // The tested value is {X=1}.
  CheckCondition(
      R"(
      na_value: true
      attribute: 5
      condition { contains_condition { elements: 1 elements: 2 } }
      )",
      1, true);

  // The tested value is the *empty* set.
  CheckCondition(
      R"(
      na_value: true
      attribute: 5
      condition { contains_condition { elements: 1 elements: 2 } }
      )",
      0, false);
}

TEST_F(EvalConditions, EvalConditionContainsBitmap) {
  // The tested value is 1=A.
  CheckCondition(
      R"(
      na_value: true
      attribute: 2
      condition { contains_bitmap_condition { elements_bitmap: "\x06" } }
      )",
      0, true);

  CheckCondition(
      R"(
      na_value: true
      attribute: 2
      condition { contains_bitmap_condition { elements_bitmap: "\x0c" } }
      )",
      0, false);

  // The tested value is B=2.
  CheckCondition(
      R"(
      na_value: true
      attribute: 2
      condition { contains_bitmap_condition { elements_bitmap: "\x0c" } }
      )",
      1, true);
}

TEST_F(EvalConditions, EvalConditionContainsBitmapSet) {
  // The tested value is {X=1}. The condition is {1, 2}.
  CheckCondition(
      R"(
      na_value: true
      attribute: 5
      condition { contains_bitmap_condition { elements_bitmap: "\x06" } }
      )",
      1, true);

  // The tested value is the *empty* set. The condition is {1, 2}.
  CheckCondition(
      R"(
      na_value: true
      attribute: 5
      condition { contains_bitmap_condition { elements_bitmap: "\x06" } }
      )",
      0, false);
}

TEST_F(EvalConditions, EvalConditionOblique) {
  CheckCondition(
      R"(
      attribute: 0
      condition {
        oblique_condition {
          attributes: 0
          attributes: 1
          weights: 1
          weights: 1
          threshold: 1
        }
      }
      )",
      1, true);

  CheckCondition(
      R"(
      attribute: 0
      condition {
        oblique_condition {
          attributes: 0
          attributes: 1
          weights: 1
          weights: 1
          threshold: 5
        }
      }
      )",
      1, false);

  const auto description = CheckCondition(
      R"(
      attribute: 0
      condition {
        oblique_condition {
          attributes: 0
          attributes: 1
          weights: 1
          weights: -0.5
          threshold: 3.5
        }
      }
      )",
      1, false);
  EXPECT_EQ(description,
            R"("Num_1"x1+"Num_2"x-0.5>=3.5 [s:0 n:0 np:0 miss:0])");

  CheckCondition(
      R"(
      attribute: 0
      na_value: true
      condition {
        oblique_condition {
          attributes: 0
          attributes: 1
          weights: 1
          weights: 1
          threshold: 1
        }
      }
      )",
      0, true);

  CheckCondition(
      R"(
      attribute: 0
      na_value: false
      condition {
        oblique_condition {
          attributes: 0
          attributes: 1
          weights: 1
          weights: 1
          threshold: 1
        }
      }
      )",
      0, false);
}

TEST(DecisionTree, ScaleRegressorOutput) {
  DecisionTree tree;
  tree.CreateRoot();
  tree.mutable_root()->CreateChildren();
  tree.mutable_root()
      ->mutable_pos_child()
      ->mutable_node()
      ->mutable_regressor()
      ->set_top_value(1.f);
  tree.mutable_root()
      ->mutable_neg_child()
      ->mutable_node()
      ->mutable_regressor()
      ->set_top_value(2.f);
  tree.ScaleRegressorOutput(2.f);

  EXPECT_NEAR(tree.root().pos_child()->node().regressor().top_value(), 2.0f,
              0.0001f);
  EXPECT_NEAR(tree.root().neg_child()->node().regressor().top_value(), 4.0f,
              0.0001f);
}

TEST(DecisionTree, CountFeatureUsage) {
  DecisionTree tree;
  tree.CreateRoot();
  auto* n0 = tree.mutable_root();
  n0->CreateChildren();

  auto* c0 = n0->mutable_node()->mutable_condition();
  c0->set_attribute(0);
  c0->mutable_condition()->mutable_higher_condition()->set_threshold(1);

  auto* n2 = n0->mutable_neg_child();
  n2->CreateChildren();

  auto* c2 = n2->mutable_node()->mutable_condition();
  c2->set_attribute(1);
  c2->mutable_condition()->mutable_oblique_condition()->add_attributes(1);
  c2->mutable_condition()->mutable_oblique_condition()->add_attributes(2);

  std::unordered_map<int32_t, int64_t> feature_usage;
  tree.CountFeatureUsage(&feature_usage);

  EXPECT_EQ(feature_usage.size(), 3);
  EXPECT_EQ(feature_usage[0], 1);
  EXPECT_EQ(feature_usage[1], 1);
  EXPECT_EQ(feature_usage[2], 1);
}

TEST(DecisionTree, DoSortedRangesIntersect) {
  std::vector<int> a;
  std::vector<int> b;
  const auto intersect = [&]() -> bool {
    return DoSortedRangesIntersect(a.begin(), a.end(), b.begin(), b.end());
  };

  EXPECT_FALSE(intersect());

  a = {1, 2, 3};
  b = {0, 4, 5};
  EXPECT_FALSE(intersect());

  a = {1, 2, 3};
  b = {1, 4, 5};
  EXPECT_TRUE(intersect());

  a = {10};
  b = {10};
  EXPECT_TRUE(intersect());

  a = {1, 2, 3};
  b = {-2, -1, 3};
  EXPECT_TRUE(intersect());

  a = {1, 2, 3};
  b = {-2, 2, 8};
  EXPECT_TRUE(intersect());

  a = {};
  b = {0, 4, 5};
  EXPECT_FALSE(intersect());

  a = {1, 2, 3};
  b = {};
  EXPECT_FALSE(intersect());
}

TEST(DecisionTree, StructureMeanMinDepth) {
  std::vector<std::unique_ptr<decision_tree::DecisionTree>> trees;
  trees.push_back(std::make_unique<DecisionTree>());
  {
    auto& tree = *trees.back();
    tree.CreateRoot();

    tree.mutable_root()->CreateChildren();
    tree.mutable_root()->mutable_node()->mutable_condition()->set_attribute(0);

    tree.mutable_root()->mutable_pos_child()->CreateChildren();
    tree.mutable_root()
        ->mutable_pos_child()
        ->mutable_node()
        ->mutable_condition()
        ->set_attribute(1);
  }

  trees.push_back(std::make_unique<DecisionTree>());
  {
    auto& tree = *trees.back();
    tree.CreateRoot();

    tree.mutable_root()->CreateChildren();
    tree.mutable_root()->mutable_node()->mutable_condition()->set_attribute(0);

    tree.mutable_root()->mutable_pos_child()->CreateChildren();
    tree.mutable_root()
        ->mutable_pos_child()
        ->mutable_node()
        ->mutable_condition()
        ->set_attribute(3);
  }

  const float kMargin = 0.00001f;
  auto vi = StructureMeanMinDepth(trees, /*num_features=*/4);
  EXPECT_EQ(vi.size(), 3);

  EXPECT_EQ(vi[0].attribute_idx(), 0);
  EXPECT_NEAR(vi[0].importance(), 1, kMargin);

  EXPECT_EQ(vi[1].attribute_idx(), 1);
  EXPECT_NEAR(vi[1].importance(),
              1. / (1. + (1. + 1. + 1. + 2. + 2. + 1.) / 6.), kMargin);

  EXPECT_EQ(vi[2].attribute_idx(), 3);
  EXPECT_NEAR(vi[2].importance(),
              1. / (1. + (1. + 1. + 1. + 2. + 2. + 1.) / 6.), kMargin);
}

TEST(DecisionTree, Distance) {
  // Builds a decision tree with a single condition and two leaf nodes.
  //
  // attribute >= threshold
  //     ├─(neg)─ leaf #0
  //     └─(pos)─ leaf #1
  const auto make_tree = [](const float threshold)
      -> absl::StatusOr<std::unique_ptr<DecisionTree>> {
    auto tree = std::make_unique<DecisionTree>();
    tree->CreateRoot();
    NodeWithChildren* root = tree->mutable_root();
    STATUS_CHECK(root);
    tree->mutable_root()->CreateChildren();
    tree->mutable_root()->mutable_node()->mutable_condition()->set_attribute(0);
    tree->mutable_root()
        ->mutable_node()
        ->mutable_condition()
        ->mutable_condition()
        ->mutable_higher_condition()
        ->set_threshold(threshold);
    tree->SetLeafIndices();
    return tree;
  };

  // Build a forest with two trees.
  std::vector<std::unique_ptr<DecisionTree>> trees;
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<DecisionTree> tree, make_tree(0.5));
    trees.push_back(std::move(tree));
  }
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<DecisionTree> tree, make_tree(1.5));
    trees.push_back(std::move(tree));
  }

  DataSpecification dataspec;
  dataset::AddColumn("a", ColumnType::NUMERICAL, &dataspec);

  dataset::VerticalDataset dataset1;
  dataset1.set_data_spec(dataspec);
  ASSERT_OK(dataset1.CreateColumnsFromDataspec());
  dataset1.AppendExample({{"a", "0"}});  // Leaves #0 (tree #0) and #0 (tree #1)
  dataset1.AppendExample({{"a", "2"}});  // Leaves #1 and #1

  dataset::VerticalDataset dataset2;
  dataset2.set_data_spec(dataspec);
  ASSERT_OK(dataset2.CreateColumnsFromDataspec());
  dataset2.AppendExample({{"a", "1"}});   // Leaves #1 and #0
  dataset2.AppendExample({{"a", "-1"}});  // Leaves #0 and #0

  std::vector<float> distances(2 * 2);
  EXPECT_OK(Distance(trees, dataset1, dataset2, absl::MakeSpan(distances)));

  EXPECT_THAT(distances, ElementsAre(0.5f, 0.f, 0.5f, 1.f));
}

TEST(DecisionTree, WeightedDistance) {
  // Builds a decision tree with a single condition and two leaf nodes.
  //
  // attribute >= threshold
  //     ├─(neg)─ leaf #0
  //     └─(pos)─ leaf #1
  const auto make_tree = [](const float threshold)
      -> absl::StatusOr<std::unique_ptr<DecisionTree>> {
    auto tree = std::make_unique<DecisionTree>();
    tree->CreateRoot();
    NodeWithChildren* root = tree->mutable_root();
    STATUS_CHECK(root);
    tree->mutable_root()->CreateChildren();
    tree->mutable_root()->mutable_node()->mutable_condition()->set_attribute(0);
    tree->mutable_root()
        ->mutable_node()
        ->mutable_condition()
        ->mutable_condition()
        ->mutable_higher_condition()
        ->set_threshold(threshold);
    tree->SetLeafIndices();
    return tree;
  };

  // Build a forest with two trees.
  std::vector<std::unique_ptr<DecisionTree>> trees;
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<DecisionTree> tree, make_tree(0.5));
    trees.push_back(std::move(tree));
  }
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<DecisionTree> tree, make_tree(1.5));
    trees.push_back(std::move(tree));
  }

  DataSpecification dataspec;
  dataset::AddColumn("a", ColumnType::NUMERICAL, &dataspec);

  dataset::VerticalDataset dataset1;
  dataset1.set_data_spec(dataspec);
  ASSERT_OK(dataset1.CreateColumnsFromDataspec());
  dataset1.AppendExample({{"a", "0"}});  // Leaves #0 (tree #0) and #0 (tree #1)
  dataset1.AppendExample({{"a", "2"}});  // Leaves #1 and #1

  dataset::VerticalDataset dataset2;
  dataset2.set_data_spec(dataspec);
  ASSERT_OK(dataset2.CreateColumnsFromDataspec());
  dataset2.AppendExample({{"a", "1"}});   // Leaves #1 and #0
  dataset2.AppendExample({{"a", "-1"}});  // Leaves #0 and #0

  std::vector<float> distances(2 * 2);
  std::vector<float> weights{0.2, 0.8};
  EXPECT_OK(
      Distance(trees, dataset1, dataset2, absl::MakeSpan(distances), weights));

  EXPECT_THAT(distances, ElementsAre(0.2f, 0.f, 0.8f, 1.f));
}

TEST(DecisionTree, CheckStructureNACondition) {
  // Builds a decision tree with a single higherThan condition and two leaf
  // nodes.
  //
  // attribute >= threshold
  //     ├─(neg)─ leaf #0
  //     └─(pos)─ leaf #1
  const auto make_tree_without_na =
      []() -> absl::StatusOr<std::unique_ptr<DecisionTree>> {
    auto tree = std::make_unique<DecisionTree>();
    tree->CreateRoot();
    NodeWithChildren* root = tree->mutable_root();
    STATUS_CHECK(root);
    tree->mutable_root()->CreateChildren();
    tree->mutable_root()->mutable_node()->mutable_condition()->set_attribute(0);
    tree->mutable_root()
        ->mutable_node()
        ->mutable_condition()
        ->mutable_condition()
        ->mutable_higher_condition()
        ->set_threshold(0.5);
    tree->SetLeafIndices();
    return tree;
  };
  // Builds a decision tree with a two conditions
  //
  // attribute>=0.5
  //     ├─(pos)─ attribute is NA
  //     |        ├─(pos)
  //     |        └─(neg)
  //     └─(neg)
  const auto make_tree_with_na =
      []() -> absl::StatusOr<std::unique_ptr<DecisionTree>> {
    auto tree = std::make_unique<DecisionTree>();
    tree->CreateRoot();
    NodeWithChildren* root = tree->mutable_root();
    STATUS_CHECK(root);
    tree->mutable_root()->CreateChildren();
    tree->mutable_root()->mutable_node()->mutable_condition()->set_attribute(0);
    tree->mutable_root()
        ->mutable_node()
        ->mutable_condition()
        ->mutable_condition()
        ->mutable_higher_condition()
        ->set_threshold(0.5);
    tree->mutable_root()->mutable_pos_child()->CreateChildren();
    tree->mutable_root()
        ->mutable_pos_child()
        ->mutable_node()
        ->mutable_condition()
        ->set_attribute(0);
    tree->mutable_root()
        ->mutable_pos_child()
        ->mutable_node()
        ->mutable_condition()
        ->mutable_condition()
        ->mutable_na_condition();

    tree->SetLeafIndices();
    return tree;
  };

  // Build a forest with one trees.
  std::vector<std::unique_ptr<DecisionTree>> trees;
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<DecisionTree> tree,
                         make_tree_without_na());
    trees.push_back(std::move(tree));
  }

  DataSpecification dataspec;
  dataset::AddColumn("a", ColumnType::NUMERICAL, &dataspec);

  EXPECT_TRUE(
      CheckStructure(CheckStructureOptions::NACondition(), dataspec, trees));

  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<DecisionTree> tree,
                         make_tree_with_na());
    trees.push_back(std::move(tree));
  }

  EXPECT_FALSE(
      CheckStructure(CheckStructureOptions::NACondition(), dataspec, trees));
}

TEST(DecisionTree, InputFeatures) {
  auto tree = std::make_unique<DecisionTree>();
  TreeBuilder builder(tree.get());

  dataset::proto::DataSpecification dataspec;
  dataset::AddColumn("l", dataset::proto::ColumnType::NUMERICAL, &dataspec);
  dataset::AddColumn("f1", dataset::proto::ColumnType::NUMERICAL, &dataspec);
  dataset::AddColumn("f2", dataset::proto::ColumnType::NUMERICAL, &dataspec);
  dataset::AddColumn("f3", dataset::proto::ColumnType::NUMERICAL, &dataspec);
  dataset::AddColumn("f4", dataset::proto::ColumnType::NUMERICAL, &dataspec);

  auto [pos, l1] = builder.ConditionIsGreater(1, 1);
  auto [l2, l3] = pos.ConditionOblique({2, 3}, {0.5f, 0.5f}, 1);
  l1.LeafRegression(1);
  l2.LeafRegression(2);
  l3.LeafRegression(3);

  std::vector<std::unique_ptr<decision_tree::DecisionTree>> trees;
  trees.push_back(std::move(tree));

  EXPECT_THAT(input_features(trees), ElementsAre(1, 2, 3));
}

TEST(DecisionTree, DebugCompare) {
  dataset::proto::DataSpecification dataspec;
  dataset::AddColumn("l", dataset::proto::ColumnType::NUMERICAL, &dataspec);
  dataset::AddColumn("f1", dataset::proto::ColumnType::NUMERICAL, &dataspec);

  DecisionTree tree1;
  TreeBuilder builder1(&tree1);
  DecisionTree tree2;
  TreeBuilder builder2(&tree2);
  DecisionTree tree3;
  TreeBuilder builder3(&tree3);

  {
    auto [pos, l1] = builder1.ConditionIsGreater(1, 1);
    auto [l2, l3] = pos.ConditionIsGreater(1, 2);
    l1.LeafRegression(1);
    l2.LeafRegression(2);
    l3.LeafRegression(3);
  }

  {
    // Same as builder 1
    auto [pos, l1] = builder2.ConditionIsGreater(1, 1);
    auto [l2, l3] = pos.ConditionIsGreater(1, 2);
    l1.LeafRegression(1);
    l2.LeafRegression(2);
    l3.LeafRegression(3);
  }

  {
    auto [pos, l1] = builder3.ConditionIsGreater(1, 1);
    auto [l2, l3] = pos.ConditionIsGreater(1, 3);  // Replace value 2 with 3.
    l1.LeafRegression(1);
    l2.LeafRegression(2);
    l3.LeafRegression(3);
  }

  // Tree 1 and tree 2 are equal.
  EXPECT_TRUE(tree1.DebugCompare(dataspec, 0, tree2).empty());
}

TEST(DecisionTree, Depth) {
  dataset::proto::DataSpecification dataspec;
  dataset::AddColumn("f", dataset::proto::ColumnType::NUMERICAL, &dataspec);

  DecisionTree tree;
  TreeBuilder builder(&tree);

  auto [pos, l1] = builder.ConditionIsGreater(1, 1);
  auto [l2, l3] = pos.ConditionIsGreater(1, 2);
  l1.LeafRegression(1);
  l2.LeafRegression(2);
  l3.LeafRegression(3);
  tree.SetLeafIndices();

  EXPECT_EQ(tree.root().depth(), 0);
  EXPECT_EQ(tree.root().pos_child()->depth(), 1);
  EXPECT_EQ(tree.root().neg_child()->depth(), 1);
  EXPECT_EQ(tree.root().pos_child()->pos_child()->depth(), 2);
  EXPECT_EQ(tree.root().pos_child()->neg_child()->depth(), 2);
}

}  // namespace
}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
