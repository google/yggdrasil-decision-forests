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
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::utils::shap {
namespace {

using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::FieldsAre;

constexpr double kMargin = 0.001;
// Large margin when model training is involved.
constexpr double kMargin2 = 0.02;

double sigmoid(const double value) { return 1. / (1. + std::exp(-value)); }

// A model and a dataset.
struct TestData {
  dataset::VerticalDataset dataset;
  std::unique_ptr<model::AbstractModel> model;
};

// Creates a simple 2d L-shaped dataset and train a model on it.
TestData BuildTestData2DL(const model::proto::TrainingConfig& train_config) {
  TestData test_data;

  // Build dataset
  *test_data.dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "l" }
    columns { type: NUMERICAL name: "f1" }
    columns { type: NUMERICAL name: "f2" }
  )pb");
  CHECK_OK(test_data.dataset.CreateColumnsFromDataspec());
  for (int i = 0; i < 100; i++) {
    CHECK_OK(test_data.dataset.AppendExampleWithStatus(
        {{"l", "1"}, {"f1", "0"}, {"f2", "0"}}));
    CHECK_OK(test_data.dataset.AppendExampleWithStatus(
        {{"l", "0"}, {"f1", "0"}, {"f2", "1"}}));
    CHECK_OK(test_data.dataset.AppendExampleWithStatus(
        {{"l", "1"}, {"f1", "1"}, {"f2", "0"}}));
    CHECK_OK(test_data.dataset.AppendExampleWithStatus(
        {{"l", "1"}, {"f1", "1"}, {"f2", "1"}}));
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
TestData BuildTestDataOnDisk(const absl::string_view dataset_filename,
                             const model::proto::TrainingConfig& train_config) {
  TestData test_data;

  // Build dataset
  const std::string dataset_path = absl::StrCat(
      "csv:", file::JoinPath(test::DataRootDirectory(),
                             "yggdrasil_decision_forests/test_data/"
                             "dataset",
                             dataset_filename));
  const auto dataspec = dataset::CreateDataSpec(dataset_path).value();
  CHECK_OK(
      dataset::LoadVerticalDataset(dataset_path, dataspec, &test_data.dataset));
  auto learner = model::GetLearner(train_config).value();
  test_data.model = learner->TrainWithStatus(test_data.dataset).value();
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

// Train a model and prediction SHAP values.
TEST(ShapleyValues, ShapOnRegressiveCART2DL) {
  auto test_data = BuildTestData2DL(PARSE_TEST_PROTO(R"pb(
    task: REGRESSION
    label: "l"
    learner: "CART"
    [yggdrasil_decision_forests.model.cart.proto.cart_config] {
      validation_ratio: 0
      decision_tree { max_depth: 3 }
    }
  )pb"));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  ASSERT_OK_AND_ASSIGN(const auto expected_shape, GetShape(*test_data.model));

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));
  EXPECT_NEAR(shap_values.SumValues(0) + shap_values.bias()[0],
              prediction.regression().value(), kMargin);
  EXPECT_NEAR(prediction.regression().value(), 1., kMargin);
  EXPECT_THAT(shap_values.values(),
              ElementsAre(DoubleNear(0., kMargin), DoubleNear(-0.125, kMargin),
                          DoubleNear(0.375, kMargin)));
  EXPECT_EQ(shap_values.num_outputs(), expected_shape.num_outputs);
  EXPECT_EQ(shap_values.num_columns(), expected_shape.num_attributes);

  test_data.dataset.ExtractExample(1, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));
  EXPECT_NEAR(shap_values.SumValues(0) + shap_values.bias()[0],
              prediction.regression().value(), kMargin);
  EXPECT_NEAR(prediction.regression().value(), 0., kMargin);
  EXPECT_THAT(shap_values.values(),
              ElementsAre(DoubleNear(0., kMargin), DoubleNear(-0.375, kMargin),
                          DoubleNear(-0.375, kMargin)));
}

TEST(ShapleyValues, ShapOnRegressiveGBT2DL) {
  auto test_data = BuildTestData2DL(PARSE_TEST_PROTO(R"pb(
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
  )pb"));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));
  EXPECT_NEAR(shap_values.SumValues(0) + shap_values.bias()[0],
              prediction.regression().value(), kMargin);
  EXPECT_NEAR(prediction.regression().value(), 0.852378, kMargin);
  EXPECT_THAT(shap_values.values(), ElementsAre(DoubleNear(0., kMargin),
                                                DoubleNear(-0.0511888, kMargin),
                                                DoubleNear(0.153566, kMargin)));

  test_data.dataset.ExtractExample(1, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));
  EXPECT_NEAR(shap_values.SumValues(0) + shap_values.bias()[0],
              prediction.regression().value(), kMargin);
  EXPECT_NEAR(prediction.regression().value(), 0.442867, kMargin);
  EXPECT_THAT(
      shap_values.values(),
      ElementsAre(DoubleNear(0., kMargin), DoubleNear(-0.153566, kMargin),
                  DoubleNear(-0.153566, kMargin)));
}

TEST(ShapleyValues, ShapOnGBTRegressionAbaloneOnlyNumerical) {
  auto test_data = BuildTestDataOnDisk(
      "abalone.csv", PARSE_TEST_PROTO(R"pb(
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
      )pb"));

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
  EXPECT_THAT(shap_values.values(),
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

TEST(ShapleyValues, ShapOnGBTRegressionAbalone) {
  auto test_data = BuildTestDataOnDisk(
      "abalone.csv", PARSE_TEST_PROTO(R"pb(
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
      )pb"));

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
  EXPECT_THAT(shap_values.values(),
              ElementsAre(DoubleNear(0.436316, kMargin),    // Type
                          DoubleNear(-0.0415546, kMargin),  // LongestShell
                          DoubleNear(0.141024, kMargin),    // Diameter
                          DoubleNear(-0.137744, kMargin),   // Height
                          DoubleNear(-0.0364439, kMargin),  // WholeWeight
                          DoubleNear(0.56424, kMargin),     // ShuckedWeight
                          DoubleNear(0.013551, kMargin),    // VisceraWeight
                          DoubleNear(-1.52256, kMargin),    // ShellWeight
                          DoubleNear(0, kMargin)            // Rings (label)
                          ));
}

TEST(ShapleyValues, ShapOnRFRegressionAbalone) {
  auto test_data =
      BuildTestDataOnDisk("abalone.csv", PARSE_TEST_PROTO(R"pb(
                            task: REGRESSION
                            label: "Rings"
                            learner: "RANDOM_FOREST"
                            [yggdrasil_decision_forests.model.random_forest
                                 .proto.random_forest_config] {
                              num_trees: 10

                            }
                          )pb"));

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

TEST(ShapleyValues, ShapOnGBTBinaryClassificationAdult) {
  auto test_data = BuildTestDataOnDisk(
      "adult.csv", PARSE_TEST_PROTO(R"pb(
        task: CLASSIFICATION
        label: "income"
        # Since "education" and "education_num" are requivalent, a change of
        # implementation in the random generator can lead to a random
        # distribution of shap values between the two features. Therefore, we
        # remove one.
        learner: "GRADIENT_BOOSTED_TREES"
        features: "^(?!education_num$).+"
        [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
             .gradient_boosted_trees_config] {
          num_trees: 20
          shrinkage: 0.1
          validation_set_ratio: 0
          decision_tree { max_depth: 6 }
        }
      )pb"));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));

  // SHAP values are consistent with predictions.
  // TODO: Understand why the two values are not that close.
  EXPECT_NEAR(sigmoid(shap_values.SumValues(0) + shap_values.bias()[0]),
              prediction.classification().distribution().counts(2) /
                  prediction.classification().distribution().sum(),
              kMargin);

  EXPECT_THAT(shap_values.values(),
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

TEST(ShapleyValues, ShapOnRFBinaryClassificationWinnerTakeAllAdult) {
  auto test_data =
      BuildTestDataOnDisk("adult.csv", PARSE_TEST_PROTO(R"pb(
                            task: CLASSIFICATION
                            label: "income"
                            learner: "RANDOM_FOREST"
                            [yggdrasil_decision_forests.model.random_forest
                                 .proto.random_forest_config] { num_trees: 20 }
                          )pb"));

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

TEST(ShapleyValues, ShapOnRFBinaryClassificationNonWinnerTakeAllAdult) {
  auto test_data =
      BuildTestDataOnDisk("adult.csv", PARSE_TEST_PROTO(R"pb(
                            task: CLASSIFICATION
                            label: "income"
                            learner: "RANDOM_FOREST"
                            [yggdrasil_decision_forests.model.random_forest
                                 .proto.random_forest_config] {
                              winner_take_all_inference: false
                              num_trees: 20
                            }
                          )pb"));

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

  TEST(ShapleyValues, DISABLED_ShapOnGBTMultiClassClassificationIris) {
  auto test_data = BuildTestDataOnDisk(
      "iris.csv", PARSE_TEST_PROTO(R"pb(
        task: CLASSIFICATION
        label: "class"
        learner: "GRADIENT_BOOSTED_TREES"
        [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
             .gradient_boosted_trees_config] {
          num_trees: 20
          shrinkage: 0.1
          validation_set_ratio: 0
          decision_tree { max_depth: 6 }
        }
      )pb"));

  dataset::proto::Example example;
  ExampleShapValues shap_values;
  model::proto::Prediction prediction;

  ASSERT_OK_AND_ASSIGN(const auto expected_shape, GetShape(*test_data.model));

  test_data.dataset.ExtractExample(0, &example);
  test_data.model->Predict(example, &prediction);
  CHECK_OK(tree_shap(*test_data.model, example, &shap_values));
  EXPECT_EQ(shap_values.num_outputs(), expected_shape.num_outputs);
  EXPECT_EQ(shap_values.num_columns(), expected_shape.num_attributes);

  std::vector<double> exp_sum_shaps(shap_values.num_outputs(), 0);
  double sum_exp_sum_shaps = 0;
  for (int out = 0; out < shap_values.num_outputs(); out++) {
    const auto exp =
        std::exp(shap_values.SumValues(out) + shap_values.bias()[out]);
    exp_sum_shaps[out] = exp;
    sum_exp_sum_shaps += exp;
  }

  for (int out = 0; out < shap_values.num_outputs(); out++) {
    const double proba = exp_sum_shaps[out] / sum_exp_sum_shaps;
    EXPECT_NEAR(proba,
                prediction.classification().distribution().counts(1 + out) /
                    prediction.classification().distribution().sum(),
                kMargin);
  }

  EXPECT_THAT(
      shap_values.values(),
      ElementsAre(DoubleNear(0.00457877, kMargin2),   // Sepal.Length (0)
                  DoubleNear(-0.0835911, kMargin2),   // Sepal.Length (1)
                  DoubleNear(-0.00761932, kMargin2),  // Sepal.Length (2)
                  DoubleNear(-0.0394765, kMargin2),   // Sepal.Width (0)
                  DoubleNear(0.129913, kMargin2),     // Sepal.Width (1)
                  DoubleNear(0.000402365, kMargin2),  // Sepal.Width (2)
                  DoubleNear(-0.828567, kMargin2),    // Petal.Length (0)
                  DoubleNear(-0.59102, kMargin2),     // Petal.Length (1)
                  DoubleNear(2.06856, kMargin2),      // Petal.Length (2)
                  DoubleNear(-0.803198, kMargin2),    // Petal.Width (0)
                  DoubleNear(-1.14552, kMargin2),     // Petal.Width (1)
                  DoubleNear(1.23963, kMargin2),      // Petal.Width (2)
                  DoubleNear(0, kMargin2),            // class (0)
                  DoubleNear(0, kMargin2),            // class (1)
                  DoubleNear(0, kMargin2)             // class (2)));
                  ));
}

TEST(ShapleyValues, ShapOnRFMultiClassClassificationWinnerTakeAllIris) {
  auto test_data = BuildTestDataOnDisk("iris.csv", PARSE_TEST_PROTO(R"pb(
                                         task: CLASSIFICATION
                                         label: "class"
                                         learner: "RANDOM_FOREST"
                                       )pb"));

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

TEST(ShapleyValues, ShapOnRFMultiClassClassificationNonWinnerTakeAllIris) {
  auto test_data = BuildTestDataOnDisk(
      "iris.csv", PARSE_TEST_PROTO(R"pb(
        task: CLASSIFICATION
        label: "class"
        learner: "RANDOM_FOREST"
        [yggdrasil_decision_forests.model.random_forest.proto
             .random_forest_config] { winner_take_all_inference: false }
      )pb"));

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

}  // namespace
}  // namespace yggdrasil_decision_forests::utils::shap
