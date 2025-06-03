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

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_gbdt_filegroup_filegroup.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_gbdt_v2_class.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_gbdt_v2_proba.h"
#include "yggdrasil_decision_forests/serving/embed/test_model_adult_binary_class_gbdt_v2_score.h"

namespace yggdrasil_decision_forests::serving::embed {
namespace {

constexpr double eps = 0.00001;

TEST(Embed, test_model_adult_binary_class_gbdt_filegroup_filegroup) {
  using namespace test_model_adult_binary_class_gbdt_filegroup_filegroup;

  const float pred = Predict(Instance{});
  (void)pred;
}

TEST(Embed, test_model_adult_binary_class_gbdt_v2_class) {
  using namespace test_model_adult_binary_class_gbdt_v2_class;
  const float pred = Predict({
      .age = 39,
      .workclass = FeatureWorkclass::kStateGov,
      .fnlwgt = 77516,
      .education = FeatureEducation::kBachelors,
      .education_num = 13,
      .marital_status = FeatureMaritalStatus::kNeverMarried,
      .occupation = FeatureOccupation::kAdmClerical,
      .relationship = FeatureRelationship::kNotInFamily,
      .race = FeatureRace::kWhite,
      .sex = FeatureSex::kMale,
      .capital_gain = 2174,
      .capital_loss = 0,
      .hours_per_week = 40,
      .native_country = FeatureNativeCountry::kUnitedStates,
  });
  EXPECT_EQ(pred, Label::kLt50K);
}

TEST(Embed, test_model_adult_binary_class_gbdt_v2_proba) {
  using namespace test_model_adult_binary_class_gbdt_v2_proba;
  const float pred = Predict({
      .age = 39,
      .workclass = FeatureWorkclass::kStateGov,
      .fnlwgt = 77516,
      .education = FeatureEducation::kBachelors,
      .education_num = 13,
      .marital_status = FeatureMaritalStatus::kNeverMarried,
      .occupation = FeatureOccupation::kAdmClerical,
      .relationship = FeatureRelationship::kNotInFamily,
      .race = FeatureRace::kWhite,
      .sex = FeatureSex::kMale,
      .capital_gain = 2174,
      .capital_loss = 0,
      .hours_per_week = 40,
      .native_country = FeatureNativeCountry::kUnitedStates,
  });
  EXPECT_NEAR(pred, 0.01860435, eps);
}

TEST(Embed, test_model_adult_binary_class_gbdt_v2_score) {
  using namespace test_model_adult_binary_class_gbdt_v2_score;
  const float pred = Predict({
      .age = 39,
      .workclass = FeatureWorkclass::kStateGov,
      .fnlwgt = 77516,
      .education = FeatureEducation::kBachelors,
      .education_num = 13,
      .marital_status = FeatureMaritalStatus::kNeverMarried,
      .occupation = FeatureOccupation::kAdmClerical,
      .relationship = FeatureRelationship::kNotInFamily,
      .race = FeatureRace::kWhite,
      .sex = FeatureSex::kMale,
      .capital_gain = 2174,
      .capital_loss = 0,
      .hours_per_week = 40,
      .native_country = FeatureNativeCountry::kUnitedStates,
  });
  EXPECT_NEAR(pred, -3.96557950, eps);
}

}  // namespace
}  // namespace yggdrasil_decision_forests::serving::embed
