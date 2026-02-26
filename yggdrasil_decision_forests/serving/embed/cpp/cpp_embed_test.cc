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


#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/builder.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/embed/embed.h"
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

struct GoldenGeneratedCppCase {
  std::string model_filename;
  std::string golden_filename;
  proto::Algorithm::Enum algorithm;
  std::optional<proto::ClassificationOutput::Enum> output;
  int crop_num_trees = 3;
  bool categorical_from_string = false;
};

// Compare the generated .h files against golden files.
SIMPLE_PARAMETERIZED_TEST(
    GoldenGeneratedCpp, GoldenGeneratedCppCase,
    {
        // GBT
        {
            "adult_binary_class_gbdt_v2",
            "adult_binary_class_gbdt_v2_class.h.golden",
            proto::Algorithm::IF_ELSE,
        },
        {
            "adult_binary_class_gbdt_v2",
            "adult_binary_class_gbdt_v2_score.h.golden",
            proto::Algorithm::IF_ELSE,
            proto::ClassificationOutput::SCORE,
        },
        {
            "adult_binary_class_gbdt_v2",
            "adult_binary_class_gbdt_v2_probability.h.golden",
            proto::Algorithm::IF_ELSE,
            proto::ClassificationOutput::PROBABILITY,
        },
        {
            "adult_binary_class_gbdt_v2",
            "adult_binary_class_gbdt_v2_class_routing.h.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::CLASS,
        },
        {
            "adult_binary_class_gbdt_v2",
            "adult_binary_class_gbdt_v2_score_routing.h.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::SCORE,
        },
        {
            "adult_binary_class_gbdt_v2",
            "adult_binary_class_gbdt_v2_probability_routing.h.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::PROBABILITY,
        },
        {
            "adult_binary_class_gbdt_v2",
            "adult_binary_class_gbdt_v2_probability_routing_with_string_vocab."
            "h.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::PROBABILITY,
            3,
            true,
        },
        {
            "iris_multi_class_gbdt_v2",
            "iris_multi_class_gbdt_v2_probability_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
        },
        {
            "iris_multi_class_gbdt_v2",
            "iris_multi_class_gbdt_v2_probability_routing.h.golden",
            proto::Algorithm::ROUTING,
        },
        {
            "abalone_regression_gbdt_v2",
            "abalone_regression_gbdt_v2_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
        },
        {
            "abalone_regression_gbdt_v2",
            "abalone_regression_gbdt_v2_routing.h.golden",
            proto::Algorithm::ROUTING,
        },
        // RF
        {
            "adult_binary_class_rf_nwta_small",
            "adult_binary_class_rf_nwta_small_class_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
        },
        {
            "adult_binary_class_rf_nwta_small",
            "adult_binary_class_rf_nwta_small_proba_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
            proto::ClassificationOutput::PROBABILITY,
        },
        {
            "adult_binary_class_rf_nwta_small",
            "adult_binary_class_rf_nwta_small_proba_routing.h.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::PROBABILITY,
        },
        {
            "adult_binary_class_rf_wta_small",
            "adult_binary_class_rf_wta_small_class_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
        },
        {
            "adult_binary_class_rf_wta_small",
            "adult_binary_class_rf_wta_small_proba_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
            proto::ClassificationOutput::PROBABILITY,
        },
        {
            "adult_binary_class_rf_wta_small",
            "adult_binary_class_rf_wta_small_score_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
            proto::ClassificationOutput::SCORE,
        },
        {
            "adult_binary_class_rf_wta_small",
            "adult_binary_class_rf_wta_small_class_routing.h.golden",
            proto::Algorithm::ROUTING,
        },
        {
            "adult_binary_class_rf_wta_small",
            "adult_binary_class_rf_wta_small_proba_routing.h.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::PROBABILITY,
        },
        {
            "adult_binary_class_rf_wta_small",
            "adult_binary_class_rf_wta_small_score_routing.h.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::SCORE,
        },
        {
            "abalone_regression_rf_small",
            "abalone_regression_rf_small_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
        },
        {
            "abalone_regression_rf_small",
            "abalone_regression_rf_small_routing.h.golden",
            proto::Algorithm::ROUTING,
        },
        {
            "iris_multi_class_rf_nwta_small",
            "iris_multi_class_rf_nwta_small_class_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
        },
        {
            "iris_multi_class_rf_wta_small",
            "iris_multi_class_rf_wta_small_class_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
            proto::ClassificationOutput::CLASS,
        },
        {
            "iris_multi_class_rf_wta_small",
            "iris_multi_class_rf_wta_small_score_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
            proto::ClassificationOutput::SCORE,
        },
        {
            "iris_multi_class_rf_wta_small",
            "iris_multi_class_rf_wta_small_proba_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
            proto::ClassificationOutput::PROBABILITY,
        },
        {
            "iris_multi_class_rf_wta_small",
            "iris_multi_class_rf_wta_small_class_routing.h.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::CLASS,
        },
        {
            "iris_multi_class_rf_wta_small",
            "iris_multi_class_rf_wta_small_score_routing.h.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::SCORE,
        },
        {
            "iris_multi_class_rf_wta_small",
            "iris_multi_class_rf_wta_small_proba_routing.h.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::PROBABILITY,
        },
        {
            "iris_multi_class_rf_nwta_small",
            "iris_multi_class_rf_nwta_small_score_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
            proto::ClassificationOutput::SCORE,
        },
        {
            "iris_multi_class_rf_nwta_small",
            "iris_multi_class_rf_nwta_small_proba_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
            proto::ClassificationOutput::PROBABILITY,
        },
        {
            "iris_multi_class_rf_nwta_small",
            "iris_multi_class_rf_nwta_small_proba_routing.h.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::PROBABILITY,
        },
        {
            "adult_binary_class_gbdt_oblique",
            "adult_binary_class_gbdt_oblique_proba_routing.h.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::PROBABILITY,
        },
        {
            "adult_binary_class_gbdt_integerized",
            "adult_binary_class_gbdt_integerized_proba_routing.h.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::PROBABILITY,
        },
        {
            "adult_binary_class_gbdt_integerized",
            "adult_binary_class_gbdt_integerized_proba_if_else.h.golden",
            proto::Algorithm::IF_ELSE,
            proto::ClassificationOutput::PROBABILITY,
        },
    }) {
  const auto& test_case = GetParam();

  ASSERT_OK_AND_ASSIGN(
      auto model, model::LoadModel(file::JoinPath(TestDataDir(), "model",
                                                  test_case.model_filename)));
  auto df = dynamic_cast<model::DecisionForestInterface*>(model.get());
  if (df) {
    df->mutable_decision_trees()->resize(test_case.crop_num_trees);
  }

  proto::Options options;
  options.mutable_cpp();
  options.set_algorithm(test_case.algorithm);
  options.set_categorical_from_string(test_case.categorical_from_string);
  if (test_case.output.has_value()) {
    options.set_classification_output(*test_case.output);
  }
  ASSERT_OK_AND_ASSIGN(const auto embed, EmbedModel(*model, options));
  EXPECT_EQ(embed.size(), 1);
  EXPECT_TRUE(embed.contains("ydf_model.h"));

  test::ExpectEqualGolden(
      embed.at("ydf_model.h"),
      file::JoinPath("yggdrasil_decision_forests/test_data/"
                     "golden/embed/cpp",
                     test_case.golden_filename));
}

}  // namespace
}  // namespace yggdrasil_decision_forests::serving::embed
