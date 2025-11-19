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

#include "yggdrasil_decision_forests/utils/shap.h"

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/cart/cart.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::utils::shap {
namespace {

using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::FieldsAre;

constexpr double kMargin = 0.0001;
// Large margin when model training is involved.
constexpr double kMargin2 = 0.02;

double sigmoid(const float value) { return 1.f / (1.f + std::exp(-value)); }

// A model and a dataset.
struct TestData {
  dataset::VerticalDataset dataset;
  std::unique_ptr<model::AbstractModel> model;
};

// Creates a simple 2d L-shaped dataset and train a model on it.
TestData BuildTestData2DL(const model::proto::TrainingConfig& train_config) {
  TestData test_data;

  // Build dataset.
  auto& dataspec = *test_data.dataset.mutable_data_spec();
  dataspec = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "l" }
    columns { type: NUMERICAL name: "f1" }
    columns { type: NUMERICAL name: "f2" }
  )pb");
  if (train_config.has_weight_definition()) {
    auto* w_col = dataspec.add_columns();
    w_col->set_type(dataset::proto::NUMERICAL);
    w_col->set_name(train_config.weight_definition().attribute());
  }
  CHECK_OK(test_data.dataset.CreateColumnsFromDataspec());
  for (int i = 0; i < 100; i++) {
    if (train_config.has_weight_definition()) {
      const std::string w_col_name =
          train_config.weight_definition().attribute();
      CHECK_OK(test_data.dataset.AppendExampleWithStatus(
          {{"l", "1"}, {"f1", "0"}, {"f2", "0"}, {w_col_name, "1.0"}}));
      CHECK_OK(test_data.dataset.AppendExampleWithStatus(
          {{"l", "0"}, {"f1", "0"}, {"f2", "1"}, {w_col_name, "1.0"}}));
      CHECK_OK(test_data.dataset.AppendExampleWithStatus(
          {{"l", "1"}, {"f1", "1"}, {"f2", "0"}, {w_col_name, "1.0"}}));
      CHECK_OK(test_data.dataset.AppendExampleWithStatus(
          {{"l", "1"}, {"f1", "1"}, {"f2", "1"}, {w_col_name, "1.0"}}));
    } else {
      CHECK_OK(test_data.dataset.AppendExampleWithStatus(
          {{"l", "1"}, {"f1", "0"}, {"f2", "0"}}));
      CHECK_OK(test_data.dataset.AppendExampleWithStatus(
          {{"l", "0"}, {"f1", "0"}, {"f2", "1"}}));
      CHECK_OK(test_data.dataset.AppendExampleWithStatus(
          {{"l", "1"}, {"f1", "1"}, {"f2", "0"}}));
      CHECK_OK(test_data.dataset.AppendExampleWithStatus(
          {{"l", "1"}, {"f1", "1"}, {"f2", "1"}}));
    }
  }

  auto learner = model::GetLearner(train_config).value();
  test_data.model = learner->TrainWithStatus(test_data.dataset).value();

  // Save dataset to CSV to be tested on other library.
  // CHECK_OK(
  //     dataset::SaveVerticalDataset(test_data.dataset, "csv:/tmp/data.csv"));

  // Display the model structure
  // LOG(INFO) << "Trained model:\n"
  //          << test_data.model->DescriptionAndStatistics(true);
  return test_data;
}

// Loads a csv dataset and train a model on it.
absl::StatusOr<TestData> BuildTestDataOnDisk(
    const absl::string_view dataset_filename,
    model::proto::TrainingConfig train_config) {
  TestData test_data;

  // Build dataset
  const std::string dataset_path = absl::StrCat(
      "csv:", file::JoinPath(test::DataRootDirectory(),
                             "yggdrasil_decision_forests/test_data/"
                             "dataset",
                             dataset_filename));
  ASSIGN_OR_RETURN(const auto dataspec, dataset::CreateDataSpec(dataset_path));
  RETURN_IF_ERROR(
      dataset::LoadVerticalDataset(dataset_path, dataspec, &test_data.dataset));

  // If the training config has a weight definition which doesn't exist in the
  // dataset, create unit weights.
  if (train_config.has_weight_definition() &&
      test_data.dataset.ColumnNameToColumnIdx(
          train_config.weight_definition().attribute()) == -1) {
    const auto weights_col_name = train_config.weight_definition().attribute();
    dataset::proto::Column weights_column_spec;
    weights_column_spec.set_type(dataset::proto::NUMERICAL);
    weights_column_spec.set_name(weights_col_name);

    ASSIGN_OR_RETURN(auto* weights_column,
                     test_data.dataset.AddColumn(weights_column_spec));
    train_config.mutable_weight_definition()->set_attribute(weights_col_name);
    train_config.mutable_weight_definition()->mutable_numerical();

    auto unit_weight_attr = dataset::proto::Example::Attribute();
    unit_weight_attr.set_numerical(1.f);
    for (auto example_idx = 0; example_idx < weights_column->nrows();
         example_idx++) {
      weights_column->Set(example_idx, unit_weight_attr);
    }
  }

  ASSIGN_OR_RETURN(auto learner, model::GetLearner(train_config));
  ASSIGN_OR_RETURN(test_data.model,
                   learner->TrainWithStatus(test_data.dataset));
  return test_data;
}

// Checks "extend" against values computed with the SHAP python package,
// and check that "extend" and "unwind" are commutative.
TEST(ShapleyValues, extend_and_unwind) {
  internal::Path path;
  internal::extend(0.1, 0.9, 1, path);
  EXPECT_THAT(path,
              ElementsAre(FieldsAre(1, 0.1, 0.9, DoubleNear(1., kMargin))));

  internal::extend(0.2, 0.8, 2, path);
  EXPECT_THAT(path,
              ElementsAre(FieldsAre(1, 0.1, 0.9, DoubleNear(0.1, kMargin)),
                          FieldsAre(2, 0.2, 0.8, DoubleNear(0.4, kMargin))));

  internal::extend(0.3, 0.7, 3, path);
  EXPECT_THAT(path,
              ElementsAre(FieldsAre(1, 0.1, 0.9, DoubleNear(0.02, kMargin)),
                          FieldsAre(2, 0.2, 0.8, DoubleNear(0.0633, kMargin)),
                          FieldsAre(3, 0.3, 0.7, DoubleNear(0.1866, kMargin))));

  internal::unwind(2, path);
  EXPECT_THAT(path,
              ElementsAre(FieldsAre(1, 0.1, 0.9, DoubleNear(0.1, kMargin)),
                          FieldsAre(2, 0.2, 0.8, DoubleNear(0.4, kMargin))));

  internal::unwind(1, path);
  EXPECT_THAT(path,
              ElementsAre(FieldsAre(1, 0.1, 0.9, DoubleNear(1., kMargin))));
}

// Checks that "unwound_sum" is equivalent to the sum of weights after a
// "unwound" call.
TEST(ShapleyValues, unwound_sum) {
  internal::Path path;
  internal::extend(0.1, 0.9, 1, path);
  internal::extend(0.2, 0.8, 2, path);
  internal::extend(0.3, 0.7, 3, path);

  const double fast_sum = unwound_sum(2, path);

  internal::unwind(2, path);
  double slow_sum = 0;
  for (const auto& p : path) {
    slow_sum += p.weight;
  }

  EXPECT_NEAR(fast_sum, slow_sum, kMargin);
}

SIMPLE_PARAMETERIZED_TEST(ShapleyValuesRegressiveCART2DL, bool, {false, true}) {
  bool use_unit_weights = GetParam();
  model::proto::TrainingConfig train_config = PARSE_TEST_PROTO(R"pb(
    task: REGRESSION
    label: "l"
    learner: "CART"
    [yggdrasil_decision_forests.model.cart.proto.cart_config] {
      validation_ratio: 0
      decision_tree { max_depth: 3 }
    }
  )pb");

  if (use_unit_weights) {
    train_config.mutable_weight_definition()->set_attribute("weights");
    train_config.mutable_weight_definition()->mutable_numerical();
  }

  auto test_data = BuildTestData2DL(train_config);

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  ASSERT_OK_AND_ASSIGN(const auto expected_shape, GetShape(*test_data.model));

  // Example 0
  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));
  EXPECT_NEAR(shap_values.SumValues(0) + shap_values.bias()[0],
              prediction.regression().value(), kMargin);
  EXPECT_NEAR(prediction.regression().value(), 1., kMargin);

  if (use_unit_weights) {
    EXPECT_THAT(
        shap_values.values(),
        ElementsAre(DoubleNear(0., kMargin), DoubleNear(-0.125, kMargin),
                    DoubleNear(0.375, kMargin), DoubleNear(0., kMargin)));
  } else {
    EXPECT_THAT(shap_values.values(), ElementsAre(DoubleNear(0., kMargin),
                                                  DoubleNear(-0.125, kMargin),
                                                  DoubleNear(0.375, kMargin)));
  }
  EXPECT_EQ(shap_values.num_outputs(), expected_shape.num_outputs);
  EXPECT_EQ(shap_values.num_columns(), expected_shape.num_attributes);

  // Example 1
  test_data.dataset.ExtractExample(1, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));
  EXPECT_NEAR(shap_values.SumValues(0) + shap_values.bias()[0],
              prediction.regression().value(), kMargin);
  EXPECT_NEAR(prediction.regression().value(), 0., kMargin);
  auto values = shap_values.values();
  if (use_unit_weights) {
    EXPECT_EQ(values.back(), 0);
    values.pop_back();
  }
  EXPECT_THAT(values,
              ElementsAre(DoubleNear(0., kMargin), DoubleNear(-0.375, kMargin),
                          DoubleNear(-0.375, kMargin)));
}

SIMPLE_PARAMETERIZED_TEST(ShapleyValuesRegressiveGBT2DL, bool, {false, true}) {
  bool use_unit_weights = GetParam();
  model::proto::TrainingConfig train_config = PARSE_TEST_PROTO(R"pb(
    task: REGRESSION
    label: "l"
    learner: "GRADIENT_BOOSTED_TREES"
    [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
         .gradient_boosted_trees_config] {
      num_trees: 5
      shrinkage: 0.1
      validation_set_ratio: 0
      decision_tree { max_depth: 3 }
    }
  )pb");

  if (use_unit_weights) {
    train_config.mutable_weight_definition()->set_attribute("weights");
    train_config.mutable_weight_definition()->mutable_numerical();
  }

  auto test_data = BuildTestData2DL(train_config);

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));
  EXPECT_NEAR(shap_values.SumValues(0) + shap_values.bias()[0],
              prediction.regression().value(), kMargin);
  EXPECT_NEAR(prediction.regression().value(), 0.852378, kMargin);
  if (use_unit_weights) {
    EXPECT_THAT(
        shap_values.values(),
        ElementsAre(DoubleNear(0., kMargin), DoubleNear(-0.0511888, kMargin),
                    DoubleNear(0.153566, kMargin), DoubleNear(0., kMargin)));
  } else {
    EXPECT_THAT(
        shap_values.values(),
        ElementsAre(DoubleNear(0., kMargin), DoubleNear(-0.0511888, kMargin),
                    DoubleNear(0.153566, kMargin)));
  }

  test_data.dataset.ExtractExample(1, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));
  EXPECT_NEAR(shap_values.SumValues(0) + shap_values.bias()[0],
              prediction.regression().value(), kMargin);
  EXPECT_NEAR(prediction.regression().value(), 0.442867, kMargin);
  auto values = shap_values.values();
  if (use_unit_weights) {
    EXPECT_EQ(values.back(), 0);
    values.pop_back();
  }
  EXPECT_THAT(values, ElementsAre(DoubleNear(0., kMargin),
                                  DoubleNear(-0.153566, kMargin),
                                  DoubleNear(-0.153566, kMargin)));
}

SIMPLE_PARAMETERIZED_TEST(
    ShapleyValuesGBTRegressionAbaloneOnlyNumericalUnweighted, bool,
    {false, true}) {
  bool use_unit_weights = GetParam();
  model::proto::TrainingConfig train_config = PARSE_TEST_PROTO(R"pb(
    task: REGRESSION
    label: "Rings"
    learner: "GRADIENT_BOOSTED_TREES"
    features: "(LongestShell|Diameter|Height|WholeWeight|ShuckedWeight|VisceraWeight|ShellWeight)"
    [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
         .gradient_boosted_trees_config] {
      num_trees: 20
      shrinkage: 0.1
      validation_set_ratio: 0
      decision_tree { max_depth: 6 }
    }
  )pb");

  if (use_unit_weights) {
    train_config.mutable_weight_definition()->set_attribute("weights");
    train_config.mutable_weight_definition()->mutable_numerical();
  }

  ASSERT_OK_AND_ASSIGN(auto test_data,
                       BuildTestDataOnDisk("abalone.csv", train_config));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));

  // SHAP values are consistent with predictions.
  EXPECT_NEAR(shap_values.SumValues(0) + shap_values.bias()[0],
              prediction.regression().value(), kMargin);

  // Those values are checked against a SKLearn model interpreted with SHAP.
  // The model was trained with:
  /*
  GradientBoostingRegressor(
        n_estimators=20,
        max_depth=5,
        random_state=0,
        min_samples_leaf=5,
    )
  */
  EXPECT_NEAR(shap_values.SumValues(0), -1.15095, kMargin);
  auto values = shap_values.values();
  if (use_unit_weights) {
    EXPECT_EQ(values.back(), 0);
    values.pop_back();
  }
  EXPECT_THAT(values,
              ElementsAre(DoubleNear(0., kMargin),          // Type (not used)
                          DoubleNear(-0.0282628, kMargin),  // LongestShell
                          DoubleNear(0.0422179, kMargin),   // Diameter
                          DoubleNear(-0.208858, kMargin),   // Height
                          DoubleNear(-0.0816655, kMargin),  // WholeWeight
                          DoubleNear(0.608902, kMargin),    // ShuckedWeight
                          DoubleNear(0.016213, kMargin),    // VisceraWeight
                          DoubleNear(-1.49949, kMargin),    // ShellWeight
                          DoubleNear(0, kMargin)            // Rings (label)
                          ));

  if (use_unit_weights) {
    EXPECT_EQ(shap_values.ToString(test_data.model->data_spec()), R"(Values:
	Type: 0.000000
	LongestShell: -0.028263
	Diameter: 0.042218
	Height: -0.208858
	WholeWeight: -0.081666
	ShuckedWeight: 0.608902
	VisceraWeight: 0.016213
	ShellWeight: -1.499493
	Rings: 0.000000
	weights: 0.000000
Bias:
	9.93368
)");
  } else {
    EXPECT_EQ(shap_values.ToString(test_data.model->data_spec()), R"(Values:
	Type: 0.000000
	LongestShell: -0.028263
	Diameter: 0.042218
	Height: -0.208858
	WholeWeight: -0.081666
	ShuckedWeight: 0.608902
	VisceraWeight: 0.016213
	ShellWeight: -1.499493
	Rings: 0.000000
Bias:
	9.93368
)");
  }
}

TEST(ShapleyValues, ShapleyValuesGBTRegressionAbaloneOnlyNumericalWeighted) {
  ASSERT_OK_AND_ASSIGN(
      auto test_data,
      BuildTestDataOnDisk(
          "abalone.csv", PARSE_TEST_PROTO(R"pb(
            task: REGRESSION
            label: "Rings"
            learner: "GRADIENT_BOOSTED_TREES"
            features: "(LongestShell|Height|WholeWeight|ShuckedWeight|VisceraWeight|ShellWeight)"
            weight_definition: {
              attribute: "Diameter"
              numerical: {}
            }
            [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
                 .gradient_boosted_trees_config] {
              num_trees: 20
              shrinkage: 0.1
              validation_set_ratio: 0
              decision_tree { max_depth: 6 }
            }
          )pb")));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  // 0.455 , 0.095 , 0.514 , 0.2245, 0.101 , 0.15
  example.add_attributes()->set_numerical(0.455);
  example.add_attributes()->set_numerical(0.095);
  example.add_attributes()->set_numerical(0.514);
  example.add_attributes()->set_numerical(0.2245);
  example.add_attributes()->set_numerical(0.101);
  example.add_attributes()->set_numerical(0.15);

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));
  LOG(INFO) << "Prediction: " << prediction;

  // Those values are checked against a SKLearn model interpreted with SHAP.
  // The model was trained with:
  /*
  X_train = ds[["LongestShell",
                "Height",
                "WholeWeight",
                "ShuckedWeight",
                "VisceraWeight",
                "ShellWeight"]].to_numpy()
  sample_weights = ds["Rings"].to_numpy()
  y_train = ds["Diameter"].to_numpy()
  model = GradientBoostingRegressor(
      n_estimators=20,
      max_depth=5,
      random_state=0,
      min_samples_leaf=5,
  )
  model.fit(X_train, y_train, sample_weight=sample_weights)

  */
  EXPECT_NEAR(shap_values.SumValues(0), -1.352997385, kMargin2);
  EXPECT_THAT(shap_values.values(),
              ElementsAre(DoubleNear(0., kMargin),            // Type
                          DoubleNear(0.0415586, kMargin2),    // LongestShell
                          DoubleNear(0., kMargin),            // Diameter
                          DoubleNear(-0.26157079, kMargin2),  // Height
                          DoubleNear(-0.10964558, kMargin2),  // WholeWeight
                          DoubleNear(0.88903176, kMargin2),   // ShuckedWeight
                          DoubleNear(0.02981519, kMargin2),   // VisceraWeight
                          DoubleNear(-1.94218657, kMargin2),  // ShellWeight
                          DoubleNear(0., kMargin)             // Rings
                          ));
}

SIMPLE_PARAMETERIZED_TEST(ShapleyValuesGBTRegressionAbalone, bool,
                          {false, true}) {
  bool use_unit_weights = GetParam();
  model::proto::TrainingConfig train_config = PARSE_TEST_PROTO(R"pb(
    task: REGRESSION
    label: "Rings"
    learner: "GRADIENT_BOOSTED_TREES"
    [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
         .gradient_boosted_trees_config] {
      num_trees: 20
      shrinkage: 0.1
      validation_set_ratio: 0
      decision_tree { max_depth: 6 }
    }
  )pb");

  if (use_unit_weights) {
    train_config.mutable_weight_definition()->set_attribute("weights");
    train_config.mutable_weight_definition()->mutable_numerical();
  }

  ASSERT_OK_AND_ASSIGN(auto test_data,
                       BuildTestDataOnDisk("abalone.csv", train_config));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));

  // SHAP values are consistent with predictions.
  EXPECT_NEAR(shap_values.SumValues(0) + shap_values.bias()[0],
              prediction.regression().value(), kMargin);

  EXPECT_NEAR(shap_values.SumValues(0), -0.583168, kMargin);

  auto values = shap_values.values();
  if (use_unit_weights) {
    EXPECT_EQ(values.back(), 0);
    values.pop_back();
  }
  EXPECT_THAT(values,
              ElementsAre(DoubleNear(0.436316, kMargin),    // Type
                          DoubleNear(-0.0415546, kMargin),  // LongestShell
                          DoubleNear(0.141024, kMargin),    // Diameter
                          DoubleNear(-0.137744, kMargin),   // Height
                          // High correlation between the weights.
                          DoubleNear(-0.0364439, kMargin2),  // WholeWeight
                          DoubleNear(0.56424, kMargin2),     // ShuckedWeight
                          DoubleNear(0.013551, kMargin2),    // VisceraWeight
                          DoubleNear(-1.52256, kMargin2),    // ShellWeight
                          DoubleNear(0, kMargin)             // Rings (label)
                          ));
}

SIMPLE_PARAMETERIZED_TEST(ShapleyValuesRFRegressionAbalone, bool,
                          {false, true}) {
  bool use_unit_weights = GetParam();
  model::proto::TrainingConfig train_config = PARSE_TEST_PROTO(R"pb(
    task: REGRESSION
    label: "Rings"
    learner: "RANDOM_FOREST"
    [yggdrasil_decision_forests.model.random_forest.proto
         .random_forest_config] {
      num_trees: 10

    }
  )pb");

  if (use_unit_weights) {
    train_config.mutable_weight_definition()->set_attribute("weights");
    train_config.mutable_weight_definition()->mutable_numerical();
  }

  ASSERT_OK_AND_ASSIGN(auto test_data,
                       BuildTestDataOnDisk("abalone.csv", train_config));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));

  // SHAP values are consistent with predictions.
  EXPECT_NEAR(shap_values.SumValues(0) + shap_values.bias()[0],
              prediction.regression().value(), kMargin);
}

SIMPLE_PARAMETERIZED_TEST(ShapleyValuesGBTBinaryClassificationAdult, bool,
                          {false, true}) {
  bool use_unit_weights = GetParam();
  model::proto::TrainingConfig train_config = PARSE_TEST_PROTO(R"pb(
    task: CLASSIFICATION
    label: "income"
    # Since "education" and "education_num" are equivalent, a change of
    # implementation in the random generator can lead to a random
    # distribution of shap values between the two features. Therefore,
    # we remove one.
    learner: "GRADIENT_BOOSTED_TREES"
    features: "^(?!education_num$).+"
    [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
         .gradient_boosted_trees_config] {
      num_trees: 20
      shrinkage: 0.1
      validation_set_ratio: 0
      decision_tree { max_depth: 6 }
    }
  )pb");

  if (use_unit_weights) {
    train_config.mutable_weight_definition()->set_attribute("weights");
    train_config.mutable_weight_definition()->mutable_numerical();
  }

  ASSERT_OK_AND_ASSIGN(auto test_data,
                       BuildTestDataOnDisk("adult.csv", train_config));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));

  // SHAP values are consistent with predictions.
  EXPECT_NEAR(sigmoid(shap_values.SumValues(0) + shap_values.bias()[0]),
              prediction.classification().distribution().counts(2) /
                  prediction.classification().distribution().sum(),
              kMargin);

  auto values = shap_values.values();
  if (use_unit_weights) {
    EXPECT_EQ(values.back(), 0);
    values.pop_back();
  }
  EXPECT_THAT(values,
              ElementsAre(DoubleNear(0.223452, kMargin2),     // age
                          DoubleNear(0.00210725, kMargin2),   // workclass
                          DoubleNear(3.85583e-05, kMargin2),  // fnlwgt
                          DoubleNear(0.457365, kMargin2),     // education
                          DoubleNear(0., kMargin2),           // education_num
                          DoubleNear(-0.4427, kMargin2),      // marital_status
                          DoubleNear(-0.100012, kMargin2),    // occupation
                          DoubleNear(-0.616921, kMargin2),    // relationship
                          DoubleNear(-6.3003e-05, kMargin2),  // race
                          DoubleNear(0.0181753, kMargin2),    // sex
                          DoubleNear(-0.106942, kMargin2),    // capital_gain
                          DoubleNear(-0.0214827, kMargin2),   // capital_loss
                          DoubleNear(-0.141541, kMargin2),    // hours_per_week
                          DoubleNear(0.00130645, kMargin2),   // native_country
                          DoubleNear(0, kMargin2)             // income
                          ));
}

SIMPLE_PARAMETERIZED_TEST(ShapleyValuesRFBinaryClassificationWinnerTakeAllAdult,
                          bool, {false, true}) {
  bool use_unit_weights = GetParam();
  model::proto::TrainingConfig train_config = PARSE_TEST_PROTO(R"pb(
    task: CLASSIFICATION
    label: "income"
    learner: "RANDOM_FOREST"
    [yggdrasil_decision_forests.model.random_forest.proto
         .random_forest_config] { num_trees: 20 }
  )pb");

  if (use_unit_weights) {
    train_config.mutable_weight_definition()->set_attribute("weights");
    train_config.mutable_weight_definition()->mutable_numerical();
  }

  ASSERT_OK_AND_ASSIGN(auto test_data,
                       BuildTestDataOnDisk("adult.csv", train_config));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));

  for (int out = 0; out < shap_values.num_outputs(); out++) {
    const double proba = shap_values.SumValues(out) + shap_values.bias()[out];
    EXPECT_NEAR(proba,
                prediction.classification().distribution().counts(1 + out) /
                    prediction.classification().distribution().sum(),
                kMargin);
  }
}

SIMPLE_PARAMETERIZED_TEST(
    ShapleyValuesRFBinaryClassificationNonWinnerTakeAllAdult, bool,
    {false, true}) {
  bool use_unit_weights = GetParam();
  model::proto::TrainingConfig train_config = PARSE_TEST_PROTO(R"pb(
    task: CLASSIFICATION
    label: "income"
    learner: "RANDOM_FOREST"
    [yggdrasil_decision_forests.model.random_forest.proto
         .random_forest_config] {
      winner_take_all_inference: false
      num_trees: 20
    }
  )pb");

  if (use_unit_weights) {
    train_config.mutable_weight_definition()->set_attribute("weights");
    train_config.mutable_weight_definition()->mutable_numerical();
  }

  ASSERT_OK_AND_ASSIGN(auto test_data,
                       BuildTestDataOnDisk("adult.csv", train_config));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));

  for (int out = 0; out < shap_values.num_outputs(); out++) {
    const double proba = shap_values.SumValues(out) + shap_values.bias()[out];
    EXPECT_NEAR(proba,
                prediction.classification().distribution().counts(1 + out) /
                    prediction.classification().distribution().sum(),
                kMargin);
  }
}

SIMPLE_PARAMETERIZED_TEST(
    ShapleyValuesRFMultiClassClassificationWinnerTakeAllIris, bool,
    {false, true}) {
  bool use_unit_weights = GetParam();
  model::proto::TrainingConfig train_config = PARSE_TEST_PROTO(R"pb(
    task: CLASSIFICATION
    label: "class"
    learner: "RANDOM_FOREST"
  )pb");

  if (use_unit_weights) {
    train_config.mutable_weight_definition()->set_attribute("weights");
    train_config.mutable_weight_definition()->mutable_numerical();
  }

  ASSERT_OK_AND_ASSIGN(auto test_data,
                       BuildTestDataOnDisk("iris.csv", train_config));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));

  for (int out = 0; out < shap_values.num_outputs(); out++) {
    const double proba = shap_values.SumValues(out) + shap_values.bias()[out];
    EXPECT_NEAR(proba,
                prediction.classification().distribution().counts(1 + out) /
                    prediction.classification().distribution().sum(),
                kMargin);
  }
}

SIMPLE_PARAMETERIZED_TEST(
    ShapleyValuesRFMultiClassClassificationNonWinnerTakeAllIris, bool,
    {false, true}) {
  bool use_unit_weights = GetParam();
  model::proto::TrainingConfig train_config = PARSE_TEST_PROTO(R"pb(
    task: CLASSIFICATION
    label: "class"
    learner: "RANDOM_FOREST"
    [yggdrasil_decision_forests.model.random_forest.proto
         .random_forest_config] { winner_take_all_inference: false }
  )pb");

  if (use_unit_weights) {
    train_config.mutable_weight_definition()->set_attribute("weights");
    train_config.mutable_weight_definition()->mutable_numerical();
  }

  ASSERT_OK_AND_ASSIGN(auto test_data,
                       BuildTestDataOnDisk("iris.csv", train_config));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));

  for (int out = 0; out < shap_values.num_outputs(); out++) {
    const double proba = shap_values.SumValues(out) + shap_values.bias()[out];
    EXPECT_NEAR(proba,
                prediction.classification().distribution().counts(1 + out) /
                    prediction.classification().distribution().sum(),
                kMargin);
  }
}

TEST(ShapleyValues, ManualTree) {
  dataset::VerticalDataset dataset;

  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns {
      type: CATEGORICAL
      name: "l"
      categorical {
        number_of_unique_values: 3
        items {
          key: "<OOD>"
          value { index: 0 count: 0 }
        }
        items {
          key: "1"
          value { index: 2 count: 2 }
        }
        items {
          key: "0"
          value { index: 1 count: 2 }
        }
      }
      count_nas: 0
      dtype: DTYPE_INT64
    }
    columns {
      type: NUMERICAL
      name: "w"
      numerical {
        mean: 1.5
        min_value: 1
        max_value: 3
        standard_deviation: 0.8660254037844386
      }
      count_nas: 0
      dtype: DTYPE_FLOAT64
    }
    columns {
      type: NUMERICAL
      name: "f0"
      numerical { mean: 1.5 min_value: 1 max_value: 2 standard_deviation: 0.5 }
      count_nas: 0
      dtype: DTYPE_FLOAT64
    }
    columns {
      type: NUMERICAL
      name: "f1"
      numerical {
        mean: 1.125
        min_value: 0.5
        max_value: 2
        standard_deviation: 0.54486236794258425
      }
      count_nas: 0
      dtype: DTYPE_FLOAT64
    }
    created_num_rows: 4
  )pb");
  EXPECT_OK(dataset.CreateColumnsFromDataspec());
  CHECK_OK(dataset.AppendExampleWithStatus(
      {{"f0", "1.0"}, {"f1", "1.0"}, {"w", "3.0"}, {"l", "0"}}));
  CHECK_OK(dataset.AppendExampleWithStatus(
      {{"f0", "1.0"}, {"f1", "0.5"}, {"w", "1.0"}, {"l", "0"}}));
  CHECK_OK(dataset.AppendExampleWithStatus(
      {{"f0", "2.0"}, {"f1", "1.0"}, {"w", "1.0"}, {"l", "1"}}));
  CHECK_OK(dataset.AppendExampleWithStatus(
      {{"f0", "2.0"}, {"f1", "2.0"}, {"w", "1.0"}, {"l", "1"}}));
  model::proto::TrainingConfig train_config;
  train_config.set_learner(model::cart::CartLearner::kRegisteredName);
  train_config.set_task(model::proto::Task::CLASSIFICATION);
  train_config.set_label("l");
  train_config.add_features("f0");
  train_config.add_features("f1");
  *train_config.mutable_weight_definition()->mutable_attribute() = "w";
  train_config.mutable_weight_definition()->mutable_numerical();
  auto* cart_config =
      train_config.MutableExtension(model::cart::proto::cart_config);
  cart_config->mutable_decision_tree()->set_min_examples(1);
  cart_config->set_validation_ratio(0.0);

  ASSERT_OK_AND_ASSIGN(const auto learner, model::GetLearner(train_config));
  ASSERT_OK_AND_ASSIGN(const auto model, learner->TrainWithStatus(dataset));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  example.add_attributes()->set_categorical(0);
  example.add_attributes()->set_numerical(1.0);
  example.add_attributes()->set_numerical(0.5);
  example.add_attributes()->set_numerical(1000);
  CHECK_OK(tree_shap(*model, example, &shap_values));
}

TEST(ShapleyValues, ManualTreeWithStump) {
  dataset::VerticalDataset dataset;

  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns {
      type: NUMERICAL
      name: "l"
      numerical { mean: 0.5 min_value: 0 max_value: 1 standard_deviation: 0.5 }
      count_nas: 0
      dtype: DTYPE_FLOAT64
    }
    columns {
      type: NUMERICAL
      name: "w"
      numerical {
        mean: 1.5
        min_value: 1
        max_value: 3
        standard_deviation: 0.8660254037844386
      }
      count_nas: 0
      dtype: DTYPE_FLOAT64
    }
    columns {
      type: NUMERICAL
      name: "f0"
      numerical { mean: 1.5 min_value: 1 max_value: 2 standard_deviation: 0.5 }
      count_nas: 0
      dtype: DTYPE_FLOAT64
    }
    columns {
      type: NUMERICAL
      name: "f1"
      numerical {
        mean: 1.125
        min_value: 0.5
        max_value: 2
        standard_deviation: 0.54486236794258425
      }
      count_nas: 0
      dtype: DTYPE_FLOAT64
    }
    created_num_rows: 4
  )pb");
  EXPECT_OK(dataset.CreateColumnsFromDataspec());
  CHECK_OK(dataset.AppendExampleWithStatus(
      {{"f0", "1.0"}, {"f1", "1.0"}, {"w", "3.0"}, {"l", "0.0"}}));
  CHECK_OK(dataset.AppendExampleWithStatus(
      {{"f0", "1.0"}, {"f1", "0.5"}, {"w", "1.0"}, {"l", "0.0"}}));
  CHECK_OK(dataset.AppendExampleWithStatus(
      {{"f0", "2.0"}, {"f1", "1.0"}, {"w", "1.0"}, {"l", "1.0"}}));
  CHECK_OK(dataset.AppendExampleWithStatus(
      {{"f0", "2.0"}, {"f1", "2.0"}, {"w", "1.0"}, {"l", "1.0"}}));
  model::proto::TrainingConfig train_config;
  train_config.set_learner(model::gradient_boosted_trees::
                               GradientBoostedTreesLearner::kRegisteredName);
  train_config.set_task(model::proto::Task::REGRESSION);
  train_config.set_label("l");
  train_config.add_features("f0");
  train_config.add_features("f1");
  auto* gbt_config = train_config.MutableExtension(
      model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->mutable_decision_tree()->set_min_examples(1);
  gbt_config->set_shrinkage(1.);
  gbt_config->set_validation_set_ratio(0.0);
  gbt_config->set_num_trees(1);

  ASSERT_OK_AND_ASSIGN(const auto learner, model::GetLearner(train_config));
  ASSERT_OK_AND_ASSIGN(const auto model, learner->TrainWithStatus(dataset));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  example.add_attributes()->set_numerical(0);
  example.add_attributes()->set_numerical(1.0);
  example.add_attributes()->set_numerical(0.5);
  example.add_attributes()->set_numerical(1000);
  CHECK_OK(tree_shap(*model, example, &shap_values));

  auto* gbt_model =
      dynamic_cast<model::gradient_boosted_trees::GradientBoostedTreesModel*>(
          model.get());
  EXPECT_NE(gbt_model, nullptr);
  auto stump = std::make_unique<model::decision_tree::DecisionTree>();
  stump->CreateRoot();
  stump->mutable_root()->mutable_node()->mutable_regressor()->set_top_value(
      4.2);
  gbt_model->AddTree(std::move(stump));
  ExampleShapValues new_shap_values;
  CHECK_OK(tree_shap(*model, example, &new_shap_values));
  EXPECT_EQ(shap_values.SumValues(0), new_shap_values.SumValues(0));
  EXPECT_EQ(shap_values.num_outputs(), new_shap_values.num_outputs());
  EXPECT_EQ(shap_values.bias(), new_shap_values.bias());
  EXPECT_EQ(shap_values.values(), new_shap_values.values());
}

TEST(ShapleyValues, ShapOnGBTStumpOnly) {
  ASSERT_OK_AND_ASSIGN(
      auto test_data,
      BuildTestDataOnDisk(
          "abalone.csv", PARSE_TEST_PROTO(R"pb(
            task: REGRESSION
            label: "Rings"
            learner: "GRADIENT_BOOSTED_TREES"
            features: "(LongestShell|Height|WholeWeight|ShuckedWeight|VisceraWeight|ShellWeight)"
            weight_definition: {
              attribute: "Diameter"
              numerical: {}
            }
            [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
                 .gradient_boosted_trees_config] {
              num_trees: 1
              shrinkage: 0.1
              validation_set_ratio: 0
              decision_tree { max_depth: 1 }
            }
          )pb")));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  // 0.455 , 0.095 , 0.514 , 0.2245, 0.101 , 0.15
  example.add_attributes()->set_numerical(0.455);
  example.add_attributes()->set_numerical(0.095);
  example.add_attributes()->set_numerical(0.514);
  example.add_attributes()->set_numerical(0.2245);
  example.add_attributes()->set_numerical(0.101);
  example.add_attributes()->set_numerical(0.15);

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));

  // If each tree is just a root, no features are used for splits,
  // so all feature importances are zero.
  EXPECT_THAT(shap_values.values(),
              ElementsAre(DoubleNear(0., kMargin),  // Type
                          DoubleNear(0., kMargin),  // LongestShell
                          DoubleNear(0., kMargin),  // Diameter
                          DoubleNear(0., kMargin),  // Height
                          DoubleNear(0., kMargin),  // WholeWeight
                          DoubleNear(0., kMargin),  // ShuckedWeight
                          DoubleNear(0., kMargin),  // VisceraWeight
                          DoubleNear(0., kMargin),  // ShellWeight
                          DoubleNear(0., kMargin)   // Rings
                          ));
}

}  // namespace
}  // namespace yggdrasil_decision_forests::utils::shap
