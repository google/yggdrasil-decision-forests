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

// Test the predictions value of embedded models.
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

#include <array>
#include <cstdint>

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_abalone_regression_gbdt_v2.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_abalone_regression_gbdt_v2_routing.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_abalone_regression_rf_small.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_abalone_regression_rf_small_routing.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_gbdt_filegroup_filegroup.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_gbdt_v2_class.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_gbdt_v2_proba.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_gbdt_v2_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_gbdt_v2_score.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_rf_nwta_small_class.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_rf_nwta_small_proba.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_rf_nwta_small_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_rf_nwta_small_score.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_rf_wta_small_class.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_rf_wta_small_proba.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_rf_wta_small_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_rf_wta_small_score.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_iris_multi_class_gbdt_v2_class.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_iris_multi_class_gbdt_v2_proba.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_iris_multi_class_gbdt_v2_proba_routing.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_iris_multi_class_gbdt_v2_score.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_iris_multi_class_rf_nwta_small_class.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_iris_multi_class_rf_nwta_small_proba.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_iris_multi_class_rf_nwta_small_score.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_iris_multi_class_rf_wta_small_class.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_iris_multi_class_rf_wta_small_proba.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_iris_multi_class_rf_wta_small_score.h"

namespace yggdrasil_decision_forests::serving::embed {
namespace {

#define ADULT_EXAMPLE                                        \
  {                                                          \
      .age = 39,                                             \
      .workclass = FeatureWorkclass::kStateGov,              \
      .fnlwgt = 77516,                                       \
      .education = FeatureEducation::kBachelors,             \
      .education_num = 13,                                   \
      .marital_status = FeatureMaritalStatus::kNeverMarried, \
      .occupation = FeatureOccupation::kAdmClerical,         \
      .relationship = FeatureRelationship::kNotInFamily,     \
      .race = FeatureRace::kWhite,                           \
      .sex = FeatureSex::kMale,                              \
      .capital_gain = 2174,                                  \
      .capital_loss = 0,                                     \
      .hours_per_week = 40,                                  \
      .native_country = FeatureNativeCountry::kUnitedStates, \
  }

#define IRIS_EXAMPLE        \
  {                         \
      .sepal_length = 5.1f, \
      .sepal_width = 3.5f,  \
      .petal_length = 1.4f, \
      .petal_width = 0.2f,  \
  }

#define ABALONE_EXAMPLE         \
  {                             \
      .type = FeatureType::kM,  \
      .longestshell = 0.455f,   \
      .diameter = 0.365f,       \
      .height = 0.095f,         \
      .wholeweight = 0.514f,    \
      .shuckedweight = 0.2245f, \
      .visceraweight = 0.101f,  \
      .shellweight = 0.15f,     \
  }

constexpr double eps = 0.00001;

TEST(Embed, test_model_adult_binary_class_gbdt_filegroup_filegroup) {
  using namespace test_model_adult_binary_class_gbdt_filegroup_filegroup;
  const float pred = Predict(Instance{});
  (void)pred;
}

// GBT binary class

TEST(Embed, test_model_adult_binary_class_gbdt_v2_class) {
  using namespace test_model_adult_binary_class_gbdt_v2_class;
  const int pred = Predict(ADULT_EXAMPLE);
  EXPECT_EQ(pred, Label::kLt50K);
}

TEST(Embed, test_model_adult_binary_class_gbdt_v2_proba) {
  using namespace test_model_adult_binary_class_gbdt_v2_proba;
  const float pred = Predict(ADULT_EXAMPLE);
  EXPECT_NEAR(pred, 0.01860435, eps);
}

TEST(Embed, test_model_adult_binary_class_gbdt_v2_proba_routing) {
  using namespace test_model_adult_binary_class_gbdt_v2_proba_routing;
  const float pred = Predict(ADULT_EXAMPLE);
  EXPECT_NEAR(pred, 0.01860435, eps);
}

TEST(Embed, test_model_adult_binary_class_gbdt_v2_score) {
  using namespace test_model_adult_binary_class_gbdt_v2_score;
  const float pred = Predict(ADULT_EXAMPLE);
  EXPECT_NEAR(pred, -3.96557950, eps);
}

// RF binary class

TEST(Embed, test_model_adult_binary_class_rf_nwta_small_class) {
  using namespace test_model_adult_binary_class_rf_nwta_small_class;
  const int pred = Predict(ADULT_EXAMPLE);
  EXPECT_EQ(pred, Label::kLt50K);
}

TEST(Embed, test_model_adult_binary_class_rf_nwta_small_proba) {
  using namespace test_model_adult_binary_class_rf_nwta_small_proba;
  const float pred = Predict(ADULT_EXAMPLE);
  EXPECT_NEAR(pred, 0.01538462, eps);
}

TEST(Embed, test_model_adult_binary_class_rf_nwta_small_proba_routing) {
  using namespace test_model_adult_binary_class_rf_nwta_small_proba_routing;
  const float pred = Predict(ADULT_EXAMPLE);
  EXPECT_NEAR(pred, 0.01538462, eps);
}

TEST(Embed, test_model_adult_binary_class_rf_nwta_small_score) {
  using namespace test_model_adult_binary_class_rf_nwta_small_score;
  const float pred = Predict(ADULT_EXAMPLE);
  EXPECT_NEAR(pred, 0.01538462, eps);
}

TEST(Embed, test_model_adult_binary_class_rf_wta_small_class) {
  using namespace test_model_adult_binary_class_rf_wta_small_class;
  const int pred = Predict(ADULT_EXAMPLE);
  EXPECT_EQ(pred, Label::kLt50K);
}

TEST(Embed, test_model_adult_binary_class_rf_wta_small_proba) {
  using namespace test_model_adult_binary_class_rf_wta_small_proba;
  const float pred = Predict(ADULT_EXAMPLE);
  EXPECT_NEAR(pred, 0., eps);
}

TEST(Embed, test_model_adult_binary_class_rf_wta_small_proba_routing) {
  using namespace test_model_adult_binary_class_rf_wta_small_proba_routing;
  const float pred = Predict(ADULT_EXAMPLE);
  EXPECT_NEAR(pred, 0., eps);
}

TEST(Embed, test_model_adult_binary_class_rf_wta_small_score) {
  using namespace test_model_adult_binary_class_rf_wta_small_score;
  const float pred = Predict(ADULT_EXAMPLE);
  EXPECT_NEAR(pred, 0., eps);
}

// Regression

TEST(Embed, test_model_abalone_regression_gbdt_v2) {
  using namespace test_model_abalone_regression_gbdt_v2;
  const float pred = Predict(ABALONE_EXAMPLE);
  EXPECT_NEAR(pred, 9.815921, eps);
}

TEST(Embed, test_model_abalone_regression_gbdt_v2_routing) {
  using namespace test_model_abalone_regression_gbdt_v2_routing;
  const float pred = Predict(ABALONE_EXAMPLE);
  EXPECT_NEAR(pred, 9.815921, eps);
}

TEST(Embed, test_model_abalone_regression_rf_small) {
  using namespace test_model_abalone_regression_rf_small;
  const float pred = Predict(ABALONE_EXAMPLE);
  EXPECT_NEAR(pred, 11.092856, eps);
}

TEST(Embed, test_model_abalone_regression_rf_small_routing) {
  using namespace test_model_abalone_regression_rf_small_routing;
  const float pred = Predict(ABALONE_EXAMPLE);
  EXPECT_NEAR(pred, 11.092856, eps);
}

// GBT multi-class

TEST(Embed, test_model_iris_multi_class_gbdt_v2_class) {
  using namespace test_model_iris_multi_class_gbdt_v2_class;
  const int pred = Predict(IRIS_EXAMPLE);
  EXPECT_EQ(pred, Label::kSetosa);
}

TEST(Embed, test_model_iris_multi_class_gbdt_v2_score) {
  using namespace test_model_iris_multi_class_gbdt_v2_score;
  const std::array<float, 3> pred = Predict(IRIS_EXAMPLE);
  EXPECT_NEAR(pred[0], 2.49707317, eps);
  EXPECT_NEAR(pred[1], -2.0397801, eps);
  EXPECT_NEAR(pred[2], -2.0296919, eps);
}

TEST(Embed, test_model_iris_multi_class_gbdt_v2_proba) {
  using namespace test_model_iris_multi_class_gbdt_v2_proba;
  const std::array<float, 3> pred = Predict(IRIS_EXAMPLE);
  EXPECT_NEAR(pred[0], 0.9789308, eps);
  EXPECT_NEAR(pred[1], 0.01048146, eps);
  EXPECT_NEAR(pred[2], 0.01058776, eps);
}

TEST(Embed, test_model_iris_multi_class_gbdt_v2_proba_routing) {
  using namespace test_model_iris_multi_class_gbdt_v2_proba_routing;
  const std::array<float, 3> pred = Predict(IRIS_EXAMPLE);
  EXPECT_NEAR(pred[0], 0.9789308, eps);
  EXPECT_NEAR(pred[1], 0.01048146, eps);
  EXPECT_NEAR(pred[2], 0.01058776, eps);
}

// RF multi-class

TEST(Embed, test_model_iris_multi_class_rf_nwta_small_class) {
  using namespace test_model_iris_multi_class_rf_nwta_small_class;
  const int pred = Predict(IRIS_EXAMPLE);
  EXPECT_EQ(pred, Label::kSetosa);
}

TEST(Embed, test_model_iris_multi_class_rf_nwta_small_score) {
  using namespace test_model_iris_multi_class_rf_nwta_small_score;
  const std::array<float, 3> pred = Predict(IRIS_EXAMPLE);
  EXPECT_NEAR(pred[Label::kSetosa], 1., eps);
  EXPECT_NEAR(pred[Label::kVersicolor], 0., eps);
  EXPECT_NEAR(pred[Label::kVirginica], 0., eps);
}

TEST(Embed, test_model_iris_multi_class_rf_nwta_small_proba) {
  using namespace test_model_iris_multi_class_rf_nwta_small_proba;
  const std::array<float, 3> pred = Predict(IRIS_EXAMPLE);
  EXPECT_NEAR(pred[Label::kSetosa], 1., eps);
  EXPECT_NEAR(pred[Label::kVersicolor], 0., eps);
  EXPECT_NEAR(pred[Label::kVirginica], 0., eps);
}

TEST(Embed, test_model_iris_multi_class_rf_wta_small_class) {
  using namespace test_model_iris_multi_class_rf_wta_small_class;
  const int pred = Predict(IRIS_EXAMPLE);
  EXPECT_EQ(pred, Label::kSetosa);
}

TEST(Embed, test_model_iris_multi_class_rf_wta_small_score) {
  using namespace test_model_iris_multi_class_rf_wta_small_score;
  const std::array<uint8_t, 3> pred = Predict(IRIS_EXAMPLE);
  EXPECT_EQ(pred[Label::kSetosa], 10);
  EXPECT_EQ(pred[Label::kVersicolor], 0);
  EXPECT_EQ(pred[Label::kVirginica], 0);
}

TEST(Embed, test_model_iris_multi_class_rf_wta_small_proba) {
  using namespace test_model_iris_multi_class_rf_wta_small_proba;
  const std::array<float, 3> pred = Predict(IRIS_EXAMPLE);
  EXPECT_NEAR(pred[Label::kSetosa], 1., eps);
  EXPECT_NEAR(pred[Label::kVersicolor], 0., eps);
  EXPECT_NEAR(pred[Label::kVirginica], 0., eps);
}

}  // namespace
}  // namespace yggdrasil_decision_forests::serving::embed
