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

// Test the predictions value of models converted to C.
//
// Golden model predictions are generated with the following pattern in python:
/*

import pandas as pd
import ydf

ydf_root = ...
model_path = f"{ydf_root}/test_data/model/adult_binary_class_rf"
ds_path = f"{ydf_root}/test_data/dataset/adult_test.csv"
model = ydf.load_model(model_path)
ds = pd.read_csv(ds_path)
print(ds.iloc[0])
print(model.predict(ds[:1]))
*/

#include <cstdint>

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_abalone_regression_gbdt_v2_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_abalone_regression_rf_small_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_adult_binary_class_gbdt_oblique_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_adult_binary_class_gbdt_v2_class_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_adult_binary_class_gbdt_v2_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_adult_binary_class_gbdt_v2_score_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_adult_binary_class_rf_nwta_small_class_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_adult_binary_class_rf_nwta_small_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_adult_binary_class_rf_nwta_small_score_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_adult_binary_class_rf_wta_small_class_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_adult_binary_class_rf_wta_small_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_adult_binary_class_rf_wta_small_score_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_iris_multi_class_gbdt_v2_class_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_iris_multi_class_gbdt_v2_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_iris_multi_class_gbdt_v2_score_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_iris_multi_class_rf_nwta_small_class_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_iris_multi_class_rf_nwta_small_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_iris_multi_class_rf_nwta_small_score_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_iris_multi_class_rf_wta_small_class_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_iris_multi_class_rf_wta_small_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/c/test_model_iris_multi_class_rf_wta_small_score_routing.h"

namespace yggdrasil_decision_forests::serving::embed {
namespace {

template <typename T>
T CreateAbaloneExample() {
  return T{
      .type = TestModelAbaloneRegressionGbdtV2Routing_FeatureTypeEnum_M,
      .longestshell = 0.455f,
      .diameter = 0.365f,
      .height = 0.095f,
      .wholeweight = 0.514f,
      .shuckedweight = 0.2245f,
      .visceraweight = 0.101f,
      .shellweight = 0.15f,
  };
}

template <typename T>
T CreateAdultExample() {
  return T{
      .age = 39,
      .workclass =
          TestModelAdultBinaryClassGbdtV2ProbaRouting_FeatureWorkclassEnum_StateGov,
      .fnlwgt = 77516,
      .education =
          TestModelAdultBinaryClassGbdtV2ProbaRouting_FeatureEducationEnum_Bachelors,
      .education_num = 13,
      .marital_status =
          TestModelAdultBinaryClassGbdtV2ProbaRouting_FeatureMaritalStatusEnum_NeverMarried,
      .occupation =
          TestModelAdultBinaryClassGbdtV2ProbaRouting_FeatureOccupationEnum_AdmClerical,
      .relationship =
          TestModelAdultBinaryClassGbdtV2ProbaRouting_FeatureRelationshipEnum_NotInFamily,
      .race = TestModelAdultBinaryClassGbdtV2ProbaRouting_FeatureRaceEnum_White,
      .sex = TestModelAdultBinaryClassGbdtV2ProbaRouting_FeatureSexEnum_Male,
      .capital_gain = 2174,
      .capital_loss = 0,
      .hours_per_week = 40,
      .native_country =
          TestModelAdultBinaryClassGbdtV2ProbaRouting_FeatureNativeCountryEnum_UnitedStates,
  };
}

template <typename T>
T CreateIrisExample() {
  return T{
      .sepal_length = 5.1f,
      .sepal_width = 3.5f,
      .petal_length = 1.4f,
      .petal_width = 0.2f,
  };
}

constexpr double eps = 0.00001;

// Regression

TEST(Embed, test_model_abalone_regression_gbdt_v2_routing) {
  auto instance =
      CreateAbaloneExample<TestModelAbaloneRegressionGbdtV2Routing_Instance>();
  float pred = TestModelAbaloneRegressionGbdtV2Routing_Predict(&instance);

  EXPECT_NEAR(pred, 9.815921, eps);
}

TEST(Embed, test_model_abalone_regression_rf_small_routing) {
  auto instance =
      CreateAbaloneExample<TestModelAbaloneRegressionRfSmallRouting_Instance>();
  float pred = TestModelAbaloneRegressionRfSmallRouting_Predict(&instance);
  EXPECT_NEAR(pred, 11.092856, eps);
}

TEST(Embed, test_model_adult_binary_class_gbdt_v2_class_routing) {
  auto instance = CreateAdultExample<
      TestModelAdultBinaryClassGbdtV2ClassRouting_Instance>();
  float pred = TestModelAdultBinaryClassGbdtV2ClassRouting_Predict(&instance);
  EXPECT_EQ(pred, TestModelAdultBinaryClassGbdtV2ClassRouting_LabelEnum_Lt50K);
}

TEST(Embed, test_model_adult_binary_class_gbdt_v2_proba_routing) {
  auto instance = CreateAdultExample<
      TestModelAdultBinaryClassGbdtV2ProbaRouting_Instance>();
  float pred = TestModelAdultBinaryClassGbdtV2ProbaRouting_Predict(&instance);
  EXPECT_NEAR(pred, 0.01860435, eps);
}

TEST(Embed, test_model_adult_binary_class_gbdt_v2_score_routing) {
  auto instance = CreateAdultExample<
      TestModelAdultBinaryClassGbdtV2ScoreRouting_Instance>();
  float pred = TestModelAdultBinaryClassGbdtV2ScoreRouting_Predict(&instance);
  EXPECT_NEAR(pred, -3.96557950, eps);
}

TEST(Embed, test_model_adult_binary_class_gbdt_oblique_proba_routing) {
  auto instance = CreateAdultExample<
      TestModelAdultBinaryClassGbdtObliqueProbaRouting_Instance>();
  float pred =
      TestModelAdultBinaryClassGbdtObliqueProbaRouting_Predict(&instance);
  EXPECT_NEAR(pred, 0.03093987, eps);
}

TEST(Embed, test_model_adult_binary_class_rf_wta_small_class_routing) {
  auto instance = CreateAdultExample<
      TestModelAdultBinaryClassRfWtaSmallClassRouting_Instance>();
  float pred =
      TestModelAdultBinaryClassRfWtaSmallClassRouting_Predict(&instance);
  EXPECT_EQ(pred,
            TestModelAdultBinaryClassRfWtaSmallClassRouting_LabelEnum_Lt50K);
}

TEST(Embed, test_model_adult_binary_class_rf_wta_small_proba_routing) {
  auto instance = CreateAdultExample<
      TestModelAdultBinaryClassRfWtaSmallProbaRouting_Instance>();
  float pred =
      TestModelAdultBinaryClassRfWtaSmallProbaRouting_Predict(&instance);
  EXPECT_NEAR(pred, 0., eps);
}

TEST(Embed, test_model_adult_binary_class_rf_wta_small_score_routing) {
  auto instance = CreateAdultExample<
      TestModelAdultBinaryClassRfWtaSmallScoreRouting_Instance>();
  float pred =
      TestModelAdultBinaryClassRfWtaSmallScoreRouting_Predict(&instance);
  EXPECT_NEAR(pred, 0., eps);
}

TEST(Embed, test_model_adult_binary_class_rf_nwta_small_class_routing) {
  auto instance = CreateAdultExample<
      TestModelAdultBinaryClassRfNwtaSmallClassRouting_Instance>();
  float pred =
      TestModelAdultBinaryClassRfNwtaSmallClassRouting_Predict(&instance);
  EXPECT_EQ(pred,
            TestModelAdultBinaryClassRfNwtaSmallClassRouting_LabelEnum_Lt50K);
}

TEST(Embed, test_model_adult_binary_class_rf_nwta_small_proba_routing) {
  auto instance = CreateAdultExample<
      TestModelAdultBinaryClassRfNwtaSmallProbaRouting_Instance>();
  float pred =
      TestModelAdultBinaryClassRfNwtaSmallProbaRouting_Predict(&instance);
  EXPECT_NEAR(pred, 0.01538462, eps);
}

TEST(Embed, test_model_adult_binary_class_rf_nwta_small_score_routing) {
  auto instance = CreateAdultExample<
      TestModelAdultBinaryClassRfNwtaSmallScoreRouting_Instance>();
  float pred =
      TestModelAdultBinaryClassRfNwtaSmallScoreRouting_Predict(&instance);
  EXPECT_NEAR(pred, 0.01538462, eps);
}

TEST(Embed, test_model_iris_multi_class_gbdt_v2_class_routing) {
  auto instance =
      CreateIrisExample<TestModelIrisMultiClassGbdtV2ClassRouting_Instance>();
  float pred = TestModelIrisMultiClassGbdtV2ClassRouting_Predict(&instance);
  EXPECT_EQ(pred, TestModelIrisMultiClassGbdtV2ClassRouting_LabelEnum_Setosa);
}

TEST(Embed, test_model_iris_multi_class_gbdt_v2_proba_routing) {
  auto instance =
      CreateIrisExample<TestModelIrisMultiClassGbdtV2ProbaRouting_Instance>();
  float pred[3];
  TestModelIrisMultiClassGbdtV2ProbaRouting_Predict(&instance, pred);
  EXPECT_NEAR(pred[0], 0.9789308, eps);
  EXPECT_NEAR(pred[1], 0.01048146, eps);
  EXPECT_NEAR(pred[2], 0.01058776, eps);
}

TEST(Embed, test_model_iris_multi_class_gbdt_v2_score_routing) {
  auto instance =
      CreateIrisExample<TestModelIrisMultiClassGbdtV2ScoreRouting_Instance>();
  float pred[3];
  TestModelIrisMultiClassGbdtV2ScoreRouting_Predict(&instance, pred);
  EXPECT_NEAR(pred[0], 2.49707317, eps);
  EXPECT_NEAR(pred[1], -2.0397801, eps);
  EXPECT_NEAR(pred[2], -2.0296919, eps);
}

TEST(Embed, test_model_iris_multi_class_rf_nwta_small_class_routing) {
  auto instance = CreateIrisExample<
      TestModelIrisMultiClassRfNwtaSmallClassRouting_Instance>();
  float pred =
      TestModelIrisMultiClassRfNwtaSmallClassRouting_Predict(&instance);
  EXPECT_EQ(pred,
            TestModelIrisMultiClassRfNwtaSmallClassRouting_LabelEnum_Setosa);
}

TEST(Embed, test_model_iris_multi_class_rf_nwta_small_score_routing) {
  auto instance = CreateIrisExample<
      TestModelIrisMultiClassRfNwtaSmallScoreRouting_Instance>();
  float pred[3];
  TestModelIrisMultiClassRfNwtaSmallScoreRouting_Predict(&instance, pred);
  EXPECT_NEAR(pred[0], 1., eps);
  EXPECT_NEAR(pred[1], 0., eps);
  EXPECT_NEAR(pred[2], 0., eps);
}

TEST(Embed, test_model_iris_multi_class_rf_nwta_small_proba_routing) {
  auto instance = CreateIrisExample<
      TestModelIrisMultiClassRfNwtaSmallProbaRouting_Instance>();
  float pred[3];
  TestModelIrisMultiClassRfNwtaSmallProbaRouting_Predict(&instance, pred);
  EXPECT_NEAR(pred[0], 1., eps);
  EXPECT_NEAR(pred[1], 0., eps);
  EXPECT_NEAR(pred[2], 0., eps);
}

TEST(Embed, test_model_iris_multi_class_rf_wta_small_class_routing) {
  auto instance = CreateIrisExample<
      TestModelIrisMultiClassRfWtaSmallClassRouting_Instance>();
  float pred = TestModelIrisMultiClassRfWtaSmallClassRouting_Predict(&instance);
  EXPECT_EQ(pred,
            TestModelIrisMultiClassRfWtaSmallClassRouting_LabelEnum_Setosa);
}

TEST(Embed, test_model_iris_multi_class_rf_wta_small_score_routing) {
  auto instance = CreateIrisExample<
      TestModelIrisMultiClassRfWtaSmallScoreRouting_Instance>();
  uint8_t pred[3];
  TestModelIrisMultiClassRfWtaSmallScoreRouting_Predict(&instance, pred);
  EXPECT_NEAR(pred[0], 10, eps);
  EXPECT_NEAR(pred[1], 0, eps);
  EXPECT_NEAR(pred[2], 0, eps);
}

TEST(Embed, test_model_iris_multi_class_rf_wta_small_proba_routing) {
  auto instance = CreateIrisExample<
      TestModelIrisMultiClassRfWtaSmallProbaRouting_Instance>();
  float pred[3];
  TestModelIrisMultiClassRfWtaSmallProbaRouting_Predict(&instance, pred);
  EXPECT_NEAR(pred[0], 1., eps);
  EXPECT_NEAR(pred[1], 0., eps);
  EXPECT_NEAR(pred[2], 0., eps);
}

}  // namespace
}  // namespace yggdrasil_decision_forests::serving::embed
