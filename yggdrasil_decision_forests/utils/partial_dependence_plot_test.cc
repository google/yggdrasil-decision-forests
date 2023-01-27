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

#include "yggdrasil_decision_forests/utils/partial_dependence_plot.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

using model::decision_tree::DecisionTree;
using model::proto::Task;
using model::random_forest::RandomForestModel;
using test::EqualsProto;
using ::testing::ElementsAre;

class PartialDependencePlotTest : public testing::Test {
 protected:
  void SetUp() override {
    LoadDataSpec();
    CreateDataset();
  }

  void CreateDataset() {
    *dataset_.mutable_data_spec() = data_spec_;
    CHECK_OK(dataset_.CreateColumnsFromDataspec());

    for (int i = 0; i < 1000; i++) {
      CHECK_OK(dataset_.AppendExampleWithStatus({
          {"Num_1", absl::StrCat(i)},
          {"Cat_1", absl::StrCat(i % 3)},
          {"Bool_1", absl::StrCat(i % 2)},
          {"Label", absl::StrCat(i % 2)},
      }));
    }
  }

  void LoadDataSpec() {
    data_spec_ = PARSE_TEST_PROTO(
        R"pb(
          columns {
            type: NUMERICAL
            name: "Num_1"
            numerical { min_value: 2 max_value: 4 mean: 2.77 }
          }
          columns {
            type: CATEGORICAL
            name: "Cat_1"
            categorical {
              number_of_unique_values: 3
              is_already_integerized: true  # Better looking for debug
            }
          }
          columns {
            type: BOOLEAN
            name: "Bool_1"
            boolean { count_true: 7 count_false: 13 }
          }
          columns {
            type: CATEGORICAL
            name: "Label"
            categorical {
              number_of_unique_values: 2
              is_already_integerized: true  # Better looking for debug
            }
          }
        )pb");
  }

  // Create a tree of the following type:
  //                              Root
  //                  [Attribute 0 > root_threshold]
  //                  /                           \
  //              PosChild                       NegChild
  //   [Attribute 1 \in left categories]      [Attribute 1 \in right categories]
  //             /        \                     /        \
  //        PosPosChild   NegPosChild     PosNegChild    Attribute 2 True
  //          return 0      return 1        return 1       /       \
  //                                            PosNegNegChild    NegNegNegChild
  //                                               return 1          return 0
  std::unique_ptr<DecisionTree> CreateSimpleTree(
      const float root_threshold, const std::vector<int32_t>& left_categories,
      const std::vector<int32_t>& right_categories, const Task& task) {
    auto tree = std::make_unique<DecisionTree>();
    tree->CreateRoot();
    tree->mutable_root()->CreateChildren();
    tree->mutable_root()->mutable_node()->mutable_condition()->set_attribute(0);
    tree->mutable_root()
        ->mutable_node()
        ->mutable_condition()
        ->mutable_condition()
        ->mutable_higher_condition()
        ->set_threshold(root_threshold);

    auto* pos_child = tree->mutable_root()->mutable_pos_child();
    auto* neg_child = tree->mutable_root()->mutable_neg_child();

    pos_child->CreateChildren();
    pos_child->mutable_node()->mutable_condition()->set_attribute(1);
    for (const int32_t left_category : left_categories) {
      pos_child->mutable_node()
          ->mutable_condition()
          ->mutable_condition()
          ->mutable_contains_condition()
          ->add_elements(left_category);
    }

    neg_child->CreateChildren();
    neg_child->mutable_node()->mutable_condition()->set_attribute(1);
    for (const int32_t right_category : right_categories) {
      neg_child->mutable_node()
          ->mutable_condition()
          ->mutable_condition()
          ->mutable_contains_condition()
          ->add_elements(right_category);
    }

    auto* pos_pos_child = pos_child->mutable_pos_child()->mutable_node();
    auto* pos_neg_child = pos_child->mutable_neg_child()->mutable_node();
    auto* neg_pos_child = neg_child->mutable_pos_child()->mutable_node();

    auto* neg_neg_child = neg_child->mutable_neg_child();

    neg_neg_child->CreateChildren();
    neg_neg_child->mutable_node()->mutable_condition()->set_attribute(2);
    neg_neg_child->mutable_node()
        ->mutable_condition()
        ->mutable_condition()
        ->mutable_true_value_condition();

    auto* neg_neg_neg_child =
        neg_neg_child->mutable_neg_child()->mutable_node();
    auto* pos_neg_neg_child =
        neg_neg_child->mutable_pos_child()->mutable_node();

    switch (task) {
      case model::proto::Task::CLASSIFICATION:
        pos_pos_child->mutable_classifier()->set_top_value(0);
        pos_neg_child->mutable_classifier()->set_top_value(1);
        neg_pos_child->mutable_classifier()->set_top_value(1);
        neg_neg_neg_child->mutable_classifier()->set_top_value(0);
        pos_neg_neg_child->mutable_classifier()->set_top_value(1);
        break;
      case model::proto::Task::REGRESSION:
        pos_pos_child->mutable_regressor()->set_top_value(0);
        pos_neg_child->mutable_regressor()->set_top_value(1);
        neg_pos_child->mutable_regressor()->set_top_value(1);
        neg_neg_neg_child->mutable_regressor()->set_top_value(0);
        pos_neg_neg_child->mutable_regressor()->set_top_value(1);
        break;
      default:
        CHECK(false);
    }
    return tree;
  }

  // Create a model which contains two trees of the following type:
  // Tree 1:
  //                              Root
  //                        [Attribute 0 > 2]
  //                       /                 \
  //       [Attribute 1 \in  {0}]      [Attribute 1 \in {1,2}]
  //       /                  \        /                     \
  //     return 0          return 1   return 1          [Attribute 2 True]
  //                                                        /      \
  //                                                    return 1   return 0
  //
  // Tree 2:
  //                              Root
  //                        [Attribute 0 > 3]
  //                       /                 \
  //       [Attribute 1 \in  {0,1}]      [Attribute 1 \in {2}]
  //       /                  \        /                     \
  //     return 0          return 1   return 1          [Attribute 2 True]
  //                                                        /      \
  //                                                    return 1   return 0
  //
  // Predictions for model at different values:
  // Prediction 1
  // Attribute 0 : 2.5, Attribute 1: 0, Attribute 2: True
  //   Tree 1: Root is positive -> Pos Child is true -> returns 0 ;
  //   Tree 2: Root is negative -> Neg Child is false -> returns 1 ;
  //   0.5, 0.5
  //
  // Prediction 2
  // Attribute 0 : 2.5, Attribute 1: 1, Attribute 2: True
  //   Tree 1: Root is positive -> Pos Child is false -> returns 1 ;
  //   Tree 2: Root is negative -> Neg Child is false -> returns 1 ;
  //   0, 1
  //
  // Prediction 3
  // Attribute 0 : 2.5, Attribute 1: 2, Attribute 2: True
  //   Tree 1: Root is positive -> Pos Child is false -> returns 1 ;
  //   Tree 2: Root is negative -> Neg Child is true -> returns 1 ;
  //   0, 1
  //
  // Prediction 4
  // Attribute 0 : 3.5, Attribute 1: 0, Attribute 2: True
  //   Tree 1: Root is positive -> Pos Child is true -> returns 0 ;
  //   Tree 2: Root is positive -> Pos Child is true -> returns 0 ;
  //   1, 0
  //
  // Prediction 5
  // Attribute 0 : 3.5, Attribute 1: 1, Attribute 2: True
  //   Tree 1: Root is positive -> Pos Child is false -> returns 1 ;
  //   Tree 2: Root is positive -> Pos Child is true -> returns 0 ;
  //   0.5, 0.5
  //
  // Prediction 6
  // Attribute 0 : 3.5, Attribute 1: 2, Attribute 2: True
  //   Tree 1: Root is positive -> Pos Child is false -> returns 1 ;
  //   Tree 2: Root is positive -> Pos Child is false -> returns 1 ;
  //   0, 1
  std::unique_ptr<RandomForestModel> CreateSimpleModel(const Task& task) {
    auto model = std::make_unique<RandomForestModel>();
    auto tree_1 = CreateSimpleTree(2, {0}, {1, 2}, task);
    auto tree_2 = CreateSimpleTree(3, {0, 1}, {2}, task);
    model->AddTree(std::move(tree_1));
    model->AddTree(std::move(tree_2));

    model->set_task(task);
    model->set_label_col_idx(3);
    model->set_data_spec(data_spec_);
    model->mutable_input_features()->push_back(0);
    model->mutable_input_features()->push_back(1);

    std::string description;
    model->AppendModelStructure(&description);
    return model;
  }

  dataset::proto::Example CreateExample() {
    return PARSE_TEST_PROTO(R"pb(
      attributes { numerical: 2.9 }
      attributes { categorical: 1 }
      attributes { boolean: true }
      attributes { categorical: 0 }
      example_idx: 1
    )pb");
  }

  dataset::proto::DataSpecification data_spec_;
  dataset::VerticalDataset dataset_;
  proto::PartialDependencePlotSet pdp_set_;
  dataset::proto::Example example_;
};

TEST_F(PartialDependencePlotTest, InitializePDPSetClassification) {
  pdp_set_ =
      InitializePartialDependencePlotSet(
          data_spec_, {{0}, {1}, {0, 1}}, model::proto::Task::CLASSIFICATION,
          /*label_col_idx=*/3, /*num_numerical_bins=*/2, dataset_)
          .value();
  const proto::PartialDependencePlotSet expected_pdp_set = PARSE_TEST_PROTO(
      R"pb(
        pdps {
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 0 sum: 0 }
            }
            center_input_feature_values { numerical: 249.75 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 0 sum: 0 }
            }
            center_input_feature_values { numerical: 749.25 }
          }
          attribute_info {
            num_bins_per_input_feature: 2
            attribute_idx: 0
            num_observations_per_bins: 0
            num_observations_per_bins: 0
            numerical_boundaries: 499.5
          }
        }
        pdps {
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 0 sum: 0 }
            }
            center_input_feature_values { categorical: 0 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 0 sum: 0 }
            }
            center_input_feature_values { categorical: 1 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 0 sum: 0 }
            }
            center_input_feature_values { categorical: 2 }
          }
          attribute_info {
            num_bins_per_input_feature: 3
            attribute_idx: 1
            num_observations_per_bins: 0
            num_observations_per_bins: 0
            num_observations_per_bins: 0
          }
        }
        pdps {
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 0 sum: 0 }
            }
            center_input_feature_values { numerical: 249.75 }
            center_input_feature_values { categorical: 0 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 0 sum: 0 }
            }
            center_input_feature_values { numerical: 749.25 }
            center_input_feature_values { categorical: 0 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 0 sum: 0 }
            }
            center_input_feature_values { numerical: 249.75 }
            center_input_feature_values { categorical: 1 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 0 sum: 0 }
            }
            center_input_feature_values { numerical: 749.25 }
            center_input_feature_values { categorical: 1 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 0 sum: 0 }
            }
            center_input_feature_values { numerical: 249.75 }
            center_input_feature_values { categorical: 2 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 0 sum: 0 }
            }
            center_input_feature_values { numerical: 749.25 }
            center_input_feature_values { categorical: 2 }
          }
          attribute_info {
            num_bins_per_input_feature: 2
            attribute_idx: 0
            num_observations_per_bins: 0
            num_observations_per_bins: 0
            numerical_boundaries: 499.5
          }
          attribute_info {
            num_bins_per_input_feature: 3
            attribute_idx: 1
            num_observations_per_bins: 0
            num_observations_per_bins: 0
            num_observations_per_bins: 0
          }
        }
      )pb");
  EXPECT_THAT(pdp_set_, EqualsProto(expected_pdp_set));
}

TEST_F(PartialDependencePlotTest, InitializePDPSetRegression) {
  pdp_set_ = InitializePartialDependencePlotSet(
                 data_spec_, {{0}, {1}, {0, 1}}, model::proto::Task::REGRESSION,
                 /*label_col_idx=*/3, /*num_numerical_bins=*/2, dataset_)
                 .value();
  const proto::PartialDependencePlotSet expected_pdp_set = PARSE_TEST_PROTO(
      R"pb(
        pdps {
          pdp_bins {
            prediction { sum_of_regression_predictions: 0 }
            center_input_feature_values { numerical: 249.75 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 0 }
            center_input_feature_values { numerical: 749.25 }
          }
          attribute_info {
            num_bins_per_input_feature: 2
            attribute_idx: 0
            num_observations_per_bins: 0
            num_observations_per_bins: 0
            numerical_boundaries: 499.5
          }
        }
        pdps {
          pdp_bins {
            prediction { sum_of_regression_predictions: 0 }
            center_input_feature_values { categorical: 0 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 0 }
            center_input_feature_values { categorical: 1 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 0 }
            center_input_feature_values { categorical: 2 }
          }
          attribute_info {
            num_bins_per_input_feature: 3
            attribute_idx: 1
            num_observations_per_bins: 0
            num_observations_per_bins: 0
            num_observations_per_bins: 0
          }
        }
        pdps {
          pdp_bins {
            prediction { sum_of_regression_predictions: 0 }
            center_input_feature_values { numerical: 249.75 }
            center_input_feature_values { categorical: 0 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 0 }
            center_input_feature_values { numerical: 749.25 }
            center_input_feature_values { categorical: 0 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 0 }
            center_input_feature_values { numerical: 249.75 }
            center_input_feature_values { categorical: 1 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 0 }
            center_input_feature_values { numerical: 749.25 }
            center_input_feature_values { categorical: 1 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 0 }
            center_input_feature_values { numerical: 249.75 }
            center_input_feature_values { categorical: 2 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 0 }
            center_input_feature_values { numerical: 749.25 }
            center_input_feature_values { categorical: 2 }
          }
          attribute_info {
            num_bins_per_input_feature: 2
            attribute_idx: 0
            num_observations_per_bins: 0
            num_observations_per_bins: 0
            numerical_boundaries: 499.5
          }
          attribute_info {
            num_bins_per_input_feature: 3
            attribute_idx: 1
            num_observations_per_bins: 0
            num_observations_per_bins: 0
            num_observations_per_bins: 0
          }
        }
      )pb");
  EXPECT_THAT(pdp_set_, EqualsProto(expected_pdp_set));
}

TEST_F(PartialDependencePlotTest, UpdatePDPSetClassification) {
  pdp_set_ =
      InitializePartialDependencePlotSet(
          data_spec_, {{0}, {1}, {0, 1}}, model::proto::Task::CLASSIFICATION,
          /*label_col_idx=*/3, /*num_numerical_bins=*/2, dataset_)
          .value();
  auto model = CreateSimpleModel(model::proto::Task::CLASSIFICATION);
  const auto example = CreateExample();
  EXPECT_OK(UpdatePartialDependencePlotSet(*model, example, &pdp_set_));

  // Example has attribute 1 = 2.9, attribute 0 = 1, attribute 2 = true
  // Bin (Attribute 1: 2.5) -> Prediction 2 -> 0, 1
  // Bin (Attribute 1: 3.5) -> Prediction 5 -> 0.5, 0.5
  //
  // Bin (Attribute 2: 0) -> Prediction 1 -> 0.5, 0.5
  // Bin (Attribute 2: 1) -> Prediction 2 -> 0, 1
  // Bin (Attribute 2: 2) -> Prediction 3 -> 0, 1
  //
  // Bin (Attribute 1 : 2.5, Attribute 2: 0) -> Prediction 1 -> 0.5, 0.5
  // Bin (Attribute 1 : 2.5, Attribute 2: 1) -> Prediction 2 -> 0, 1
  // Bin (Attribute 1 : 2.5, Attribute 2: 2) -> Prediction 3 -> 0, 1
  // Bin (Attribute 1 : 3.5, Attribute 2: 0) -> Prediction 4 -> 1, 0
  // Bin (Attribute 1 : 3.5, Attribute 2: 1) -> Prediction 5 -> 0.5, 0.5
  // Bin (Attribute 1 : 3.5, Attribute 2: 2) -> Prediction 6 -> 0, 1
  const proto::PartialDependencePlotSet expected_pdp_set = PARSE_TEST_PROTO(
      R"pb(
        pdps {
          num_observations: 1
          pdp_bins {
            prediction {
              classification_class_distribution {
                counts: 0.5
                counts: 0.5
                sum: 1
              }
            }
            center_input_feature_values { numerical: 249.75 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution {
                counts: 0.5
                counts: 0.5
                sum: 1
              }
            }
            center_input_feature_values { numerical: 749.25 }
          }
          attribute_info {
            num_bins_per_input_feature: 2
            attribute_idx: 0
            num_observations_per_bins: 1
            num_observations_per_bins: 0
            numerical_boundaries: 499.5
          }
        }
        pdps {
          num_observations: 1
          pdp_bins {
            prediction {
              classification_class_distribution {
                counts: 0.5
                counts: 0.5
                sum: 1
              }
            }
            center_input_feature_values { categorical: 0 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 1 sum: 1 }
            }
            center_input_feature_values { categorical: 1 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 1 sum: 1 }
            }
            center_input_feature_values { categorical: 2 }
          }
          attribute_info {
            num_bins_per_input_feature: 3
            attribute_idx: 1
            num_observations_per_bins: 0
            num_observations_per_bins: 1
            num_observations_per_bins: 0
          }
        }
        pdps {
          num_observations: 1
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 1 counts: 0 sum: 1 }
            }
            center_input_feature_values { numerical: 249.75 }
            center_input_feature_values { categorical: 0 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 1 counts: 0 sum: 1 }
            }
            center_input_feature_values { numerical: 749.25 }
            center_input_feature_values { categorical: 0 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution {
                counts: 0.5
                counts: 0.5
                sum: 1
              }
            }
            center_input_feature_values { numerical: 249.75 }
            center_input_feature_values { categorical: 1 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution {
                counts: 0.5
                counts: 0.5
                sum: 1
              }
            }
            center_input_feature_values { numerical: 749.25 }
            center_input_feature_values { categorical: 1 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 1 sum: 1 }
            }
            center_input_feature_values { numerical: 249.75 }
            center_input_feature_values { categorical: 2 }
          }
          pdp_bins {
            prediction {
              classification_class_distribution { counts: 0 counts: 1 sum: 1 }
            }
            center_input_feature_values { numerical: 749.25 }
            center_input_feature_values { categorical: 2 }
          }
          attribute_info {
            num_bins_per_input_feature: 2
            attribute_idx: 0
            num_observations_per_bins: 1
            num_observations_per_bins: 0
            numerical_boundaries: 499.5
          }
          attribute_info {
            num_bins_per_input_feature: 3
            attribute_idx: 1
            num_observations_per_bins: 0
            num_observations_per_bins: 1
            num_observations_per_bins: 0
          }
        }
      )pb");
  EXPECT_THAT(pdp_set_, EqualsProto(expected_pdp_set));
}

TEST_F(PartialDependencePlotTest, UpdatePDPSetRegression) {
  pdp_set_ = InitializePartialDependencePlotSet(
                 data_spec_, {{0}, {1}, {0, 1}}, model::proto::Task::REGRESSION,
                 /*label_col_idx=*/3, /*num_numerical_bins=*/2, dataset_)
                 .value();
  auto model = CreateSimpleModel(model::proto::Task::REGRESSION);
  const auto example = CreateExample();
  EXPECT_OK(UpdatePartialDependencePlotSet(*model, example, &pdp_set_));

  // Example has attribute 1 = 2.9 and attribute 0 = 1
  // Bin (Attribute 1: 2.5) -> Prediction 2 -> 0, 1 -> 1
  // Bin (Attribute 1: 3.5) -> Prediction 5 -> 0.5, 0.5 -> 0.5
  //
  // Bin (Attribute 2: 0) -> Prediction 1 -> 0.5, 0.5 -> 0.5
  // Bin (Attribute 2: 1) -> Prediction 2 -> 0, 1 -> 1
  // Bin (Attribute 2: 2) -> Prediction 3 -> 0, 1 -> 1
  //
  // Bin (Attribute 1 : 2.5, Attribute 2: 0) -> Prediction 1 -> 0.5, 0.5 -> 0.5
  // Bin (Attribute 1 : 2.5, Attribute 2: 1) -> Prediction 2 -> 0, 1 -> 1
  // Bin (Attribute 1 : 2.5, Attribute 2: 2) -> Prediction 3 -> 0, 1 -> 1
  // Bin (Attribute 1 : 3.5, Attribute 2: 0) -> Prediction 4 -> 1, 0 -> 0
  // Bin (Attribute 1 : 3.5, Attribute 2: 1) -> Prediction 5 -> 0.5, 0.5 -> 0.5
  // Bin (Attribute 1 : 3.5, Attribute 2: 2) -> Prediction 5 -> 0, 1 -> 1
  const proto::PartialDependencePlotSet expected_pdp_set = PARSE_TEST_PROTO(
      R"pb(
        pdps {
          num_observations: 1
          pdp_bins {
            prediction { sum_of_regression_predictions: 0.5 }
            center_input_feature_values { numerical: 249.75 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 0.5 }
            center_input_feature_values { numerical: 749.25 }
          }
          attribute_info {
            num_bins_per_input_feature: 2
            attribute_idx: 0
            num_observations_per_bins: 1
            num_observations_per_bins: 0
            numerical_boundaries: 499.5
          }
        }
        pdps {
          num_observations: 1
          pdp_bins {
            prediction { sum_of_regression_predictions: 0.5 }
            center_input_feature_values { categorical: 0 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 1 }
            center_input_feature_values { categorical: 1 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 1 }
            center_input_feature_values { categorical: 2 }
          }
          attribute_info {
            num_bins_per_input_feature: 3
            attribute_idx: 1
            num_observations_per_bins: 0
            num_observations_per_bins: 1
            num_observations_per_bins: 0
          }
        }
        pdps {
          num_observations: 1
          pdp_bins {
            prediction { sum_of_regression_predictions: 0 }
            center_input_feature_values { numerical: 249.75 }
            center_input_feature_values { categorical: 0 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 0 }
            center_input_feature_values { numerical: 749.25 }
            center_input_feature_values { categorical: 0 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 0.5 }
            center_input_feature_values { numerical: 249.75 }
            center_input_feature_values { categorical: 1 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 0.5 }
            center_input_feature_values { numerical: 749.25 }
            center_input_feature_values { categorical: 1 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 1 }
            center_input_feature_values { numerical: 249.75 }
            center_input_feature_values { categorical: 2 }
          }
          pdp_bins {
            prediction { sum_of_regression_predictions: 1 }
            center_input_feature_values { numerical: 749.25 }
            center_input_feature_values { categorical: 2 }
          }
          attribute_info {
            num_bins_per_input_feature: 2
            attribute_idx: 0
            num_observations_per_bins: 1
            num_observations_per_bins: 0
            numerical_boundaries: 499.5
          }
          attribute_info {
            num_bins_per_input_feature: 3
            attribute_idx: 1
            num_observations_per_bins: 0
            num_observations_per_bins: 1
            num_observations_per_bins: 0
          }
        }
      )pb");
  EXPECT_THAT(pdp_set_, EqualsProto(expected_pdp_set));
}

TEST_F(PartialDependencePlotTest, AppendAttributesCombinations) {
  auto model = CreateSimpleModel(model::proto::Task::CLASSIFICATION);
  std::vector<std::vector<int>> attributes_1d;
  EXPECT_OK(AppendAttributesCombinations(*model, 1, &attributes_1d));
  EXPECT_THAT(attributes_1d,
              ElementsAre(std::vector<int>{0}, std::vector<int>{1}));

  std::vector<std::vector<int>> attributes_2d;
  EXPECT_OK(AppendAttributesCombinations(*model, 2, &attributes_2d));
  EXPECT_THAT(attributes_2d, ElementsAre(std::vector<int>{0, 1}));
}

TEST_F(PartialDependencePlotTest, AppendAttributesCombinations2D) {
  auto model = CreateSimpleModel(model::proto::Task::CLASSIFICATION);
  std::vector<std::vector<int>> attributes_1d;
  EXPECT_OK(AppendAttributesCombinations2D(
      *model, dataset::proto::ColumnType::NUMERICAL,
      dataset::proto::ColumnType::CATEGORICAL, &attributes_1d));
  EXPECT_THAT(attributes_1d, ElementsAre(std::vector<int>{0, 1}));
}

TEST_F(PartialDependencePlotTest, ExampleToBinIndex) {
  pdp_set_ = InitializePartialDependencePlotSet(
                 data_spec_, {{0}, {1}, {2}, {0, 1}},
                 model::proto::Task::CLASSIFICATION,
                 /*label_col_idx=*/3, /*num_numerical_bins=*/2, dataset_)
                 .value();

  auto model = CreateSimpleModel(model::proto::Task::CLASSIFICATION);
  const auto example = CreateExample();

  EXPECT_EQ(
      internal::ExampleToBinIndex(example, model->data_spec(), pdp_set_.pdps(0))
          .value(),
      0);
  EXPECT_EQ(
      internal::ExampleToBinIndex(example, model->data_spec(), pdp_set_.pdps(1))
          .value(),
      1);
  EXPECT_EQ(
      internal::ExampleToBinIndex(example, model->data_spec(), pdp_set_.pdps(2))
          .value(),
      1);
  EXPECT_EQ(
      internal::ExampleToBinIndex(example, model->data_spec(), pdp_set_.pdps(3))
          .value(),
      2);
}

TEST(GetBinsForOneAttribute, Numerical) {
  dataset::proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"pb(
        columns {
          name: "f"
          type: NUMERICAL
          numerical { min_value: 0 max_value: 1000 mean: 500 }
        }
      )pb");

  dataset::VerticalDataset dataset;
  *dataset.mutable_data_spec() = data_spec;
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  for (int i = 0; i < 1000; i++) {
    CHECK_OK(dataset.AppendExampleWithStatus({
        {"f", absl::StrCat(i)},
    }));
  }

  const auto bins =
      internal::GetBinsForOneAttribute(data_spec, /*attribute_idx=*/0,
                                       /*num_numerical_bins=*/5, dataset)
          .value();
  EXPECT_FALSE(bins.is_log);

  const float error = 0.1f;
  EXPECT_EQ(bins.centers.size(), 5);
  EXPECT_NEAR(bins.centers[0].numerical(), 99.75, error);
  EXPECT_NEAR(bins.centers[1].numerical(), 299.5, error);
  EXPECT_NEAR(bins.centers[2].numerical(), 499.5, error);
  EXPECT_NEAR(bins.centers[3].numerical(), 699.5, error);
  EXPECT_NEAR(bins.centers[4].numerical(), 899.25, error);

  EXPECT_EQ(bins.numerical_boundaries.size(), 4);
  EXPECT_NEAR(bins.numerical_boundaries[0], 199.5, error);
  EXPECT_NEAR(bins.numerical_boundaries[1], 399.5, error);
  EXPECT_NEAR(bins.numerical_boundaries[2], 599.5, error);
  EXPECT_NEAR(bins.numerical_boundaries[3], 799.5, error);
}

TEST(GetBinsForOneAttribute, NumericalLog) {
  dataset::proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"pb(
        columns {
          name: "f"
          type: NUMERICAL
          numerical { min_value: 0 max_value: 1000 mean: 500 }
        }
      )pb");

  dataset::VerticalDataset dataset;
  *dataset.mutable_data_spec() = data_spec;
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  for (double i = 0; i < 6; i += 0.01) {
    CHECK_OK(dataset.AppendExampleWithStatus({
        {"f", absl::StrCat(std::pow(10, i))},
    }));
  }

  const auto bins =
      internal::GetBinsForOneAttribute(data_spec, /*attribute_idx=*/0,
                                       /*num_numerical_bins=*/5, dataset)
          .value();
  EXPECT_TRUE(bins.is_log);

  const float error = 0.1f;
  EXPECT_EQ(bins.centers.size(), 5);
  EXPECT_NEAR(bins.centers[0].numerical(), 8.334, error);
  EXPECT_NEAR(bins.centers[1].numerical(), 131.999, error);
  EXPECT_NEAR(bins.centers[2].numerical(), 2092.044, error);
  EXPECT_NEAR(bins.centers[3].numerical(), 33156.679, error);
  EXPECT_NEAR(bins.centers[4].numerical(), 531188.812, error);

  EXPECT_EQ(bins.numerical_boundaries.size(), 4);
  EXPECT_NEAR(bins.numerical_boundaries[0], 15.668, error);
  EXPECT_NEAR(bins.numerical_boundaries[1], 248.329, error);
  EXPECT_NEAR(bins.numerical_boundaries[2], 3935.760, error);
  EXPECT_NEAR(bins.numerical_boundaries[3], 62377.601, error);
}

TEST(SortedUniqueCounts, Basic) {
  EXPECT_EQ((std::vector<std::pair<float, int>>{}),
            internal::SortedUniqueCounts({}));
  EXPECT_EQ((std::vector<std::pair<float, int>>{{1, 1}}),
            internal::SortedUniqueCounts({1}));
  EXPECT_EQ((std::vector<std::pair<float, int>>{{1, 3}, {2, 2}, {5, 1}}),
            internal::SortedUniqueCounts({1, 2, 5, 1, 1, 2}));
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ((std::vector<std::pair<float, int>>{{1, 2}, {2, 1}, {5, 1}}),
            internal::SortedUniqueCounts({1, nan, 5, nan, 1, 2}));
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
