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

#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

std::string TestDataDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/test_data");
}

TEST(GradientBoostedTrees, SaveAndLoadModelWithoutPrefix) {
  std::unique_ptr<model::AbstractModel> original_model;
  // TODO: Simplify this test by having it use a toy model.
  EXPECT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_gbdt"),
      &original_model));
  std::string model_path =
      file::JoinPath(test::TmpDirectory(), "saved_model_without_prefix");
  EXPECT_OK(SaveModel(model_path, original_model.get(), {}));

  std::unique_ptr<model::AbstractModel> loaded_model;
  EXPECT_OK(LoadModel(model_path, &loaded_model, {}));
  EXPECT_EQ(original_model->DescriptionAndStatistics(/*full_definition=*/true),
            loaded_model->DescriptionAndStatistics(/*full_definition=*/true));
}

TEST(GradientBoostedTrees, SaveAndLoadModelWithPrefix) {
  std::string saved_model_path =
      file::JoinPath(test::TmpDirectory(), "saved_models_with_prefixes");
  std::unique_ptr<model::AbstractModel> original_model_1;
  EXPECT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_gbdt"),
      &original_model_1));
  EXPECT_OK(SaveModel(saved_model_path, original_model_1.get(),
                      {/*file_prefix=*/"prefix_1_"}));

  std::unique_ptr<model::AbstractModel> original_model_2;
  EXPECT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_gbdt_32cat"),
      &original_model_2));
  EXPECT_OK(SaveModel(saved_model_path, original_model_2.get(),
                      {/*file_prefix=*/"prefix_2_"}));

  std::unique_ptr<model::AbstractModel> loaded_model_1;
  EXPECT_OK(LoadModel(saved_model_path, &loaded_model_1,
                      {/*file_prefix=*/"prefix_1_"}));
  EXPECT_EQ(
      original_model_1->DescriptionAndStatistics(/*full_definition=*/true),
      loaded_model_1->DescriptionAndStatistics(/*full_definition=*/true));

  std::unique_ptr<model::AbstractModel> loaded_model_2;
  EXPECT_OK(LoadModel(saved_model_path, &loaded_model_2,
                      {/*file_prefix=*/"prefix_2_"}));
  EXPECT_EQ(
      original_model_2->DescriptionAndStatistics(/*full_definition=*/true),
      loaded_model_2->DescriptionAndStatistics(/*full_definition=*/true));
}

TEST(GradientBoostedTrees, SaveAndLoadModelWithAutodetectedPrefix) {
  std::unique_ptr<model::AbstractModel> original_model;
  // TODO: Simplify this test by having it use the toy model
  // defined above.
  EXPECT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_gbdt"),
      &original_model));
  std::string model_path =
      file::JoinPath(test::TmpDirectory(), "saved_model_with_auto_prefix");
  EXPECT_OK(SaveModel(model_path, original_model.get(),
                      {/*.file_prefix =*/"prefix_1_"}));

  std::unique_ptr<model::AbstractModel> loaded_model;
  EXPECT_OK(LoadModel(model_path, &loaded_model, {}));
  EXPECT_EQ(original_model->DescriptionAndStatistics(/*full_definition=*/true),
            loaded_model->DescriptionAndStatistics(/*full_definition=*/true));
}

TEST(GradientBoostedTrees,
     FailingPrefixDetectionForMultipleModelsPerDirectory) {
  std::unique_ptr<model::AbstractModel> original_model;
  EXPECT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_gbdt"),
      &original_model));
  std::string model_path =
      file::JoinPath(test::TmpDirectory(), "saved_model_with_auto_prefix");
  ASSERT_OK(SaveModel(model_path, original_model.get(),
                      {/*.file_prefix =*/"prefix_1_"}));
  ASSERT_OK(SaveModel(model_path, original_model.get(),
                      {/*.file_prefix =*/"prefix_2_"}));

  std::unique_ptr<model::AbstractModel> loaded_model;
  EXPECT_THAT(LoadModel(model_path, &loaded_model, {}),
              test::StatusIs(absl::StatusCode::kFailedPrecondition));

  std::unique_ptr<model::AbstractModel> loaded_model_1;
  EXPECT_OK(LoadModel(model_path, &loaded_model_1,
                      /*io_options*/ {/*file_prefix=*/"prefix_1_"}));
  std::unique_ptr<model::AbstractModel> loaded_model_2;
  EXPECT_OK(LoadModel(model_path, &loaded_model_2,
                      /*io_options*/ {/*file_prefix=*/"prefix_2_"}));
}

TEST(GradientBoostedTrees, WeightedMeanAbsLeafValue) {
  decision_tree::DecisionTree tree;
  tree.CreateRoot();
  tree.mutable_root()->CreateChildren();
  tree.mutable_root()->mutable_node()->mutable_condition()->set_attribute(0);
  tree.mutable_root()
      ->mutable_node()
      ->mutable_condition()
      ->mutable_condition()
      ->mutable_higher_condition()
      ->set_threshold(1);
  auto* pos = tree.mutable_root()
                  ->mutable_pos_child()
                  ->mutable_node()
                  ->mutable_regressor();
  pos->set_top_value(1);
  pos->set_sum_weights(1);
  auto* neg = tree.mutable_root()
                  ->mutable_neg_child()
                  ->mutable_node()
                  ->mutable_regressor();
  neg->set_top_value(-0.5);
  neg->set_sum_weights(2);

  EXPECT_NEAR(internal::WeightedMeanAbsLeafValue(tree),
              (1. * 1. + 0.5 * 2.) / 3., 0.000001);
}

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
