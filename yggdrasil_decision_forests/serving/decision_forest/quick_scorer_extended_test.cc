/*
 * Copyright 2021 Google LLC.
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/utils/test.h"

#include "yggdrasil_decision_forests/serving/decision_forest/quick_scorer_extended.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace decision_forest {
namespace {

using model::decision_tree::DecisionTree;
using model::decision_tree::NodeWithChildren;
using model::decision_tree::proto::Condition;
using model::gradient_boosted_trees::GradientBoostedTreesModel;
using model::gradient_boosted_trees::proto::Loss;
using testing::ElementsAre;

void BuildToyModelAndToyDataset(const model::proto::Task task,
                                const bool use_cateset_feature,
                                GradientBoostedTreesModel* model,
                                dataset::VerticalDataset* dataset,
                                const int duplicate_factor = 1) {
  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "a" }
    columns { type: NUMERICAL name: "b" }
    columns {
      type: CATEGORICAL
      name: "c"
      categorical { is_already_integerized: true number_of_unique_values: 4 }
    }
    columns {
      type: DISCRETIZED_NUMERICAL
      name: "e"
      numerical { mean: -1 }
      discretized_numerical {
        boundaries: 0.0
        boundaries: 0.1
        boundaries: 0.2
        boundaries: 0.3
      }
    }
  )pb");

  if (use_cateset_feature) {
    auto* d = dataspec.add_columns();
    d->set_name("d");
    d->set_type(dataset::proto::ColumnType::CATEGORICAL_SET);
    d->mutable_categorical()->set_is_already_integerized(false);
    d->mutable_categorical()->set_number_of_unique_values(4);
    auto& items = *d->mutable_categorical()->mutable_items();
    items["v0"].set_index(0);
    items["v1"].set_index(1);
    items["v2"].set_index(2);
    items["v3"].set_index(3);
  }

  dataset->set_data_spec(dataspec);
  CHECK_OK(dataset->CreateColumnsFromDataspec());

  if (use_cateset_feature) {
    dataset->AppendExample(
        {{"a", "0.5"}, {"b", "0.5"}, {"c", "1"}, {"e", "1.5"}, {"d", "2 3"}});
  } else {
    dataset->AppendExample(
        {{"a", "0.5"}, {"b", "0.5"}, {"c", "1"}, {"e", "1.5"}});
  }

  struct NodeHelper {
    Condition* condition;
    NodeWithChildren* pos;
    NodeWithChildren* neg;
    NodeWithChildren* node;
  };

  const auto split_node = [](NodeWithChildren* node,
                             const int attribute) -> NodeHelper {
    node->CreateChildren();
    node->mutable_node()->mutable_condition()->set_attribute(attribute);
    return {/*.condition =*/
            node->mutable_node()->mutable_condition()->mutable_condition(),
            /*.pos =*/node->mutable_pos_child(),
            /*.neg =*/node->mutable_neg_child(),
            /*.node =*/node};
  };

  model->set_task(task);
  model->set_label_col_idx(0);
  model->set_data_spec(dataspec);
  model->set_loss(Loss::SQUARED_ERROR);
  model->mutable_initial_predictions()->push_back(duplicate_factor);

  for (int duplication_idx = 0; duplication_idx < duplicate_factor;
       duplication_idx++) {
    {
      auto tree = absl::make_unique<DecisionTree>();
      tree->CreateRoot();
      auto n1 = split_node(tree->mutable_root(), 1);
      n1.condition->mutable_higher_condition()->set_threshold(2.0f);

      auto n2 = split_node(n1.pos, 1);
      n2.condition->mutable_higher_condition()->set_threshold(3.0f);
      n2.pos->mutable_node()->mutable_regressor()->set_top_value(1.f);
      n2.neg->mutable_node()->mutable_regressor()->set_top_value(2.f);

      auto n3 = split_node(n1.neg, 1);
      n3.condition->mutable_higher_condition()->set_threshold(1.0f);
      n3.pos->mutable_node()->mutable_regressor()->set_top_value(3.f);
      n3.neg->mutable_node()->mutable_regressor()->set_top_value(4.f);

      model->mutable_decision_trees()->push_back(std::move(tree));
    }

    {
      auto tree = absl::make_unique<DecisionTree>();
      tree->CreateRoot();
      auto n1 = split_node(tree->mutable_root(), 2);
      n1.condition->mutable_contains_condition()->add_elements(1);
      n1.condition->mutable_contains_condition()->add_elements(2);

      auto n2 = split_node(n1.pos, 1);
      n2.condition->mutable_higher_condition()->set_threshold(2.5f);
      n2.pos->mutable_node()->mutable_regressor()->set_top_value(10.f);
      n2.neg->mutable_node()->mutable_regressor()->set_top_value(20.f);

      auto n3 = split_node(n1.neg, 1);
      n3.condition->mutable_higher_condition()->set_threshold(1.5f);
      n3.pos->mutable_node()->mutable_regressor()->set_top_value(30.f);
      n3.neg->mutable_node()->mutable_regressor()->set_top_value(40.f);

      model->mutable_decision_trees()->push_back(std::move(tree));
    }

    {
      auto tree = absl::make_unique<DecisionTree>();
      tree->CreateRoot();
      auto n1 = split_node(tree->mutable_root(), 1);
      n1.condition->mutable_higher_condition()->set_threshold(10.0f);

      auto n2 = split_node(n1.pos, 2);
      n2.condition->mutable_contains_bitmap_condition()->set_elements_bitmap(
          "\x02");  // [1]
      n2.pos->mutable_node()->mutable_regressor()->set_top_value(100.f);
      n2.neg->mutable_node()->mutable_regressor()->set_top_value(200.f);

      auto n3 = split_node(n1.neg, 2);
      n3.condition->mutable_contains_bitmap_condition()->set_elements_bitmap(
          "\x0A");  // [1,3]
      n3.pos->mutable_node()->mutable_regressor()->set_top_value(300.f);
      n3.neg->mutable_node()->mutable_regressor()->set_top_value(400.f);

      model->mutable_decision_trees()->push_back(std::move(tree));
    }

    {
      auto tree = absl::make_unique<DecisionTree>();
      tree->CreateRoot();
      auto n1 = split_node(tree->mutable_root(), 3);
      n1.condition->mutable_discretized_higher_condition()->set_threshold(
          2);  // value>=0.1f
      n1.neg->mutable_node()->mutable_regressor()->set_top_value(10000.f);
      n1.pos->mutable_node()->mutable_regressor()->set_top_value(20000.f);
      model->mutable_decision_trees()->push_back(std::move(tree));
    }

    if (use_cateset_feature) {
      auto tree = absl::make_unique<DecisionTree>();
      tree->CreateRoot();
      auto n1 = split_node(tree->mutable_root(), 4);
      n1.condition->mutable_contains_condition()->add_elements(2);
      n1.condition->mutable_contains_condition()->add_elements(3);
      n1.node->mutable_node()->mutable_condition()->set_na_value(true);

      n1.neg->mutable_node()->mutable_regressor()->set_top_value(1000.f);

      auto n2 = split_node(n1.pos, 4);
      n2.condition->mutable_contains_bitmap_condition()->set_elements_bitmap(
          "\x08");  // [3]
      n2.neg->mutable_node()->mutable_regressor()->set_top_value(2000.f);
      n2.pos->mutable_node()->mutable_regressor()->set_top_value(3000.f);
      n2.node->mutable_node()->mutable_condition()->set_na_value(false);
      model->mutable_decision_trees()->push_back(std::move(tree));
    }
  }
}

TEST(QuickScorer, Compilation) {
  GradientBoostedTreesModel model;
  dataset::VerticalDataset dataset;
  BuildToyModelAndToyDataset(model::proto::Task::REGRESSION,
                             /*use_cateset_feature=*/false, &model, &dataset);
  GradientBoostedTreesRegressionQuickScorerExtended quick_scorer_model;
  CHECK_OK(GenericToSpecializedModel(model, &quick_scorer_model));

  const auto model_description = DescribeQuickScorer(quick_scorer_model);
  LOG(INFO) << "Model:\n" << model_description;

  EXPECT_EQ(quick_scorer_model.features().input_features().size(), 3);

  // Examples in FORMAT_FEATURE_MAJOR, see decision_forest.h.
  using V = NumericalOrCategoricalValue;
  std::vector<V> examples = {
      // Feature 1
      V::Numerical(0.5f),
      V::Numerical(1.0f),
      V::Numerical(1.5f),
      V::Numerical(2.5f),
      V::Numerical(3.5f),
      // Feature 2
      V::Categorical(0),
      V::Categorical(1),
      V::Categorical(2),
      V::Categorical(0),
      V::Categorical(1),
      // Feature 3
      V::Numerical(0.00f),
      V::Numerical(0.05f),
      V::Numerical(0.10f),
      V::Numerical(0.20f),
      V::Numerical(0.30f),
  };
  std::vector<float> predictions;
  PredictQuickScorer(quick_scorer_model, examples, 5, &predictions);

  EXPECT_THAT(predictions,
              ElementsAre(1 + 4 + 40 + 400 + 10000, 1 + 3 + 20 + 300 + 10000,
                          1 + 3 + 20 + 400 + 20000, 1 + 2 + 30 + 400 + 20000,
                          1 + 1 + 10 + 300 + 20000));
}

TEST(QuickScorer, ExampleSet) {
  GradientBoostedTreesModel model;
  dataset::VerticalDataset dataset;
  BuildToyModelAndToyDataset(model::proto::Task::REGRESSION,
                             /*use_cateset_feature=*/true, &model, &dataset);
  GradientBoostedTreesRegressionQuickScorerExtended quick_scorer_model;
  CHECK_OK(GenericToSpecializedModel(model, &quick_scorer_model));

  const auto model_description = DescribeQuickScorer(quick_scorer_model);
  LOG(INFO) << "Model:\n" << model_description;

  GradientBoostedTreesRegressionQuickScorerExtended::ExampleSet examples(
      5, quick_scorer_model);
  examples.FillMissing(quick_scorer_model);

  EXPECT_EQ(quick_scorer_model.features().input_features().size(), 4);

  const auto feature_1 =
      GradientBoostedTreesRegressionQuickScorerExtended::ExampleSet::
          GetNumericalFeatureId("b", quick_scorer_model)
              .value();
  const auto feature_2 =
      GradientBoostedTreesRegressionQuickScorerExtended::ExampleSet::
          GetCategoricalFeatureId("c", quick_scorer_model)
              .value();
  const auto feature_3 =
      GradientBoostedTreesRegressionQuickScorerExtended::ExampleSet::
          GetCategoricalSetFeatureId("d", quick_scorer_model)
              .value();
  const auto feature_4 =
      GradientBoostedTreesRegressionQuickScorerExtended::ExampleSet::
          GetNumericalFeatureId("e", quick_scorer_model)
              .value();

  examples.SetNumerical(0, feature_1, 0.5f, quick_scorer_model);
  examples.SetNumerical(1, feature_1, 1.0f, quick_scorer_model);
  examples.SetNumerical(2, feature_1, 1.5f, quick_scorer_model);
  examples.SetNumerical(3, feature_1, 2.5f, quick_scorer_model);
  examples.SetNumerical(4, feature_1, 3.5f, quick_scorer_model);

  examples.SetCategorical(0, feature_2, 0, quick_scorer_model);
  examples.SetCategorical(1, feature_2, 1, quick_scorer_model);
  examples.SetCategorical(2, feature_2, 2, quick_scorer_model);
  examples.SetCategorical(3, feature_2, 0, quick_scorer_model);
  examples.SetCategorical(4, feature_2, 1, quick_scorer_model);

  examples.SetCategoricalSet(0, feature_3, {"v1"}, quick_scorer_model);
  examples.SetCategoricalSet(1, feature_3, {"v2"}, quick_scorer_model);
  examples.SetCategoricalSet(2, feature_3, {"v3"}, quick_scorer_model);
  examples.SetCategoricalSet(3, feature_3, std::vector<std::string>{"v2", "v3"},
                             quick_scorer_model);
  examples.SetMissingCategoricalSet(4, feature_3, quick_scorer_model);

  examples.SetNumerical(0, feature_4, 0.00f, quick_scorer_model);
  examples.SetNumerical(1, feature_4, 0.05f, quick_scorer_model);
  examples.SetNumerical(2, feature_4, 0.10f, quick_scorer_model);
  examples.SetNumerical(3, feature_4, 0.20f, quick_scorer_model);
  examples.SetNumerical(4, feature_4, 0.30f, quick_scorer_model);

  std::vector<float> predictions;
  Predict(quick_scorer_model, examples, 5, &predictions);

  EXPECT_THAT(predictions, ElementsAre(1 + 4 + 40 + 400 + 1000 + 10000,
                                       1 + 3 + 20 + 300 + 2000 + 10000,
                                       1 + 3 + 20 + 400 + 3000 + 20000,
                                       1 + 2 + 30 + 400 + 3000 + 20000,
                                       1 + 1 + 10 + 300 + 2000 + 20000));
}

TEST(QuickScorer, ExceedStackBuffer) {
  const int duplicate_factor = 200;

  GradientBoostedTreesModel model;

  dataset::VerticalDataset dataset;
  BuildToyModelAndToyDataset(model::proto::Task::REGRESSION,
                             /*use_cateset_feature=*/true, &model, &dataset,
                             duplicate_factor);
  GradientBoostedTreesRegressionQuickScorerExtended quick_scorer_model;
  CHECK_OK(GenericToSpecializedModel(model, &quick_scorer_model));

  const auto model_description = DescribeQuickScorer(quick_scorer_model);
  LOG(INFO) << "Model:\n" << model_description;

  GradientBoostedTreesRegressionQuickScorerExtended::ExampleSet examples(
      5, quick_scorer_model);
  examples.FillMissing(quick_scorer_model);

  EXPECT_EQ(quick_scorer_model.features().input_features().size(), 4);

  const auto feature_1 =
      GradientBoostedTreesRegressionQuickScorerExtended::ExampleSet::
          GetNumericalFeatureId("b", quick_scorer_model)
              .value();
  const auto feature_2 =
      GradientBoostedTreesRegressionQuickScorerExtended::ExampleSet::
          GetCategoricalFeatureId("c", quick_scorer_model)
              .value();
  const auto feature_3 =
      GradientBoostedTreesRegressionQuickScorerExtended::ExampleSet::
          GetCategoricalSetFeatureId("d", quick_scorer_model)
              .value();

  const auto feature_4 =
      GradientBoostedTreesRegressionQuickScorerExtended::ExampleSet::
          GetNumericalFeatureId("e", quick_scorer_model)
              .value();

  examples.SetNumerical(0, feature_1, 0.5f, quick_scorer_model);
  examples.SetNumerical(1, feature_1, 1.0f, quick_scorer_model);
  examples.SetNumerical(2, feature_1, 1.5f, quick_scorer_model);
  examples.SetNumerical(3, feature_1, 2.5f, quick_scorer_model);
  examples.SetNumerical(4, feature_1, 3.5f, quick_scorer_model);

  examples.SetCategorical(0, feature_2, 0, quick_scorer_model);
  examples.SetCategorical(1, feature_2, 1, quick_scorer_model);
  examples.SetCategorical(2, feature_2, 2, quick_scorer_model);
  examples.SetCategorical(3, feature_2, 0, quick_scorer_model);
  examples.SetCategorical(4, feature_2, 1, quick_scorer_model);

  examples.SetCategoricalSet(0, feature_3, {"v1"}, quick_scorer_model);
  examples.SetCategoricalSet(1, feature_3, {"v2"}, quick_scorer_model);
  examples.SetCategoricalSet(2, feature_3, {"v3"}, quick_scorer_model);
  examples.SetCategoricalSet(3, feature_3, std::vector<std::string>{"v2", "v3"},
                             quick_scorer_model);
  examples.SetMissingCategoricalSet(4, feature_3, quick_scorer_model);

  examples.SetNumerical(0, feature_4, 0.00f, quick_scorer_model);
  examples.SetNumerical(1, feature_4, 0.05f, quick_scorer_model);
  examples.SetNumerical(2, feature_4, 0.10f, quick_scorer_model);
  examples.SetNumerical(3, feature_4, 0.20f, quick_scorer_model);
  examples.SetNumerical(4, feature_4, 0.30f, quick_scorer_model);

  std::vector<float> predictions;
  Predict(quick_scorer_model, examples, 5, &predictions);

  EXPECT_THAT(
      predictions,
      ElementsAre((1 + 4 + 40 + 400 + 1000 + 10000) * duplicate_factor,
                  (1 + 3 + 20 + 300 + 2000 + 10000) * duplicate_factor,
                  (1 + 3 + 20 + 400 + 3000 + 20000) * duplicate_factor,
                  (1 + 2 + 30 + 400 + 3000 + 20000) * duplicate_factor,
                  (1 + 1 + 10 + 300 + 2000 + 20000) * duplicate_factor));
}

}  // namespace
}  // namespace decision_forest
}  // namespace serving
}  // namespace yggdrasil_decision_forests
