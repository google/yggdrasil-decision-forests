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

#include "yggdrasil_decision_forests/serving/embed/cpp/cpp_target_lowering.h"

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/serving/embed/ir/model_ir.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::serving::embed::internal {
namespace {

TEST(CppTargetLoweringTest, Smoke) {
  const ModelIR model_ir = {.num_trees = 5, .leaf_value_dims = 1};
  proto::Options options;
  ASSERT_OK_AND_ASSIGN(const auto cpp_ir,
                       CppTargetLowering::Lower(model_ir, options));
  EXPECT_EQ(cpp_ir.num_trees, 5);
}

TEST(CppTargetLoweringTest, Enums) {
  ModelIR model_ir;
  model_ir.leaf_value_dims = 1;
  // Categorical feature
  FeatureInfo feature;
  feature.type = FeatureInfo::Type::kCategorical;
  feature.original_name = "color";
  feature.vocabulary = {"<OOV>", "Red", "Blue", "Green"};
  model_ir.features.push_back(feature);

  // Label feature
  FeatureInfo label;
  label.type = FeatureInfo::Type::kCategorical;
  label.original_name = "output";
  label.vocabulary = {"<OOV>", "No", "Yes"};
  label.is_label = true;
  model_ir.features.push_back(label);

  proto::Options options;
  ASSERT_OK_AND_ASSIGN(const auto cpp_ir,
                       CppTargetLowering::Lower(model_ir, options));

  ASSERT_EQ(cpp_ir.enums.size(), 2);
  EXPECT_EQ(cpp_ir.enums[0].name, "Label");
  ASSERT_EQ(cpp_ir.enums[0].items.size(), 2);
  EXPECT_EQ(cpp_ir.enums[0].items[0].name, "kNo");
  EXPECT_EQ(cpp_ir.enums[0].items[0].value, 0);
  EXPECT_EQ(cpp_ir.enums[0].items[0].original, "No");
  EXPECT_EQ(cpp_ir.enums[0].items[1].name, "kYes");
  EXPECT_EQ(cpp_ir.enums[0].items[1].value, 1);
  EXPECT_EQ(cpp_ir.enums[0].items[1].original, "Yes");
  EXPECT_FALSE(cpp_ir.enums[0].generate_from_string_method);

  EXPECT_EQ(cpp_ir.enums[1].name, "FeatureColor");
  ASSERT_EQ(cpp_ir.enums[1].items.size(), 4);
  EXPECT_EQ(cpp_ir.enums[1].items[0].name, "kOutOfVocabulary");
  EXPECT_EQ(cpp_ir.enums[1].items[0].value, 0);
  EXPECT_EQ(cpp_ir.enums[1].items[0].original, "");
  EXPECT_EQ(cpp_ir.enums[1].items[1].name, "kRed");
  EXPECT_EQ(cpp_ir.enums[1].items[1].value, 1);
  EXPECT_EQ(cpp_ir.enums[1].items[1].original, "Red");
  EXPECT_FALSE(cpp_ir.enums[1].generate_from_string_method);

  ASSERT_EQ(cpp_ir.features.size(), 1);
  EXPECT_EQ(cpp_ir.features[0].var_name, "color");
  EXPECT_EQ(cpp_ir.features[0].cpp_type, "FeatureColor");
}

TEST(CppTargetLoweringTest, EnumsFromString) {
  ModelIR model_ir;
  model_ir.leaf_value_dims = 1;
  // Categorical feature
  FeatureInfo feature;
  feature.type = FeatureInfo::Type::kCategorical;
  feature.original_name = "color";
  feature.vocabulary = {"<OOV>", "Red", "Blue", "Green"};
  model_ir.features.push_back(feature);

  // Label feature
  FeatureInfo label;
  label.type = FeatureInfo::Type::kCategorical;
  label.original_name = "output";
  label.vocabulary = {"<OOV>", "No", "Yes"};
  label.is_label = true;
  model_ir.features.push_back(label);

  proto::Options options;
  options.set_categorical_from_string(true);
  ASSERT_OK_AND_ASSIGN(const auto cpp_ir,
                       CppTargetLowering::Lower(model_ir, options));

  ASSERT_EQ(cpp_ir.enums.size(), 2);
  EXPECT_EQ(cpp_ir.enums[0].name, "Label");
  EXPECT_FALSE(cpp_ir.enums[0].generate_from_string_method);

  EXPECT_EQ(cpp_ir.enums[1].name, "FeatureColor");
  EXPECT_TRUE(cpp_ir.enums[1].generate_from_string_method);

  ASSERT_EQ(cpp_ir.features.size(), 1);
  EXPECT_EQ(cpp_ir.features[0].var_name, "color");
  EXPECT_EQ(cpp_ir.features[0].cpp_type, "FeatureColor");
  EXPECT_EQ(cpp_ir.features[0].cpp_type, "FeatureColor");
}

TEST(CppTargetLoweringTest, Regression) {
  ModelIR model_ir;
  model_ir.leaf_value_dims = 1;
  model_ir.task = ModelIR::Task::kRegression;
  model_ir.num_output_classes = 1;
  model_ir.num_trees = 10;
  model_ir.accumulator_initialization = {0.0};

  proto::Options options;
  ASSERT_OK_AND_ASSIGN(const auto cpp_ir,
                       CppTargetLowering::Lower(model_ir, options));
}

TEST(CppTargetLoweringTest, Multiclass) {
  ModelIR model_ir;
  model_ir.leaf_value_dims = 1;
  model_ir.task = ModelIR::Task::kMulticlassClassification;
  model_ir.num_output_classes = 3;
  model_ir.num_trees = 10;
  model_ir.accumulator_initialization = {0.0, 0.0, 0.0};

  proto::Options options;
  options.set_classification_output(proto::ClassificationOutput::PROBABILITY);
  ASSERT_OK_AND_ASSIGN(const auto cpp_ir,
                       CppTargetLowering::Lower(model_ir, options));
}

}  // namespace
}  // namespace yggdrasil_decision_forests::serving::embed::internal
