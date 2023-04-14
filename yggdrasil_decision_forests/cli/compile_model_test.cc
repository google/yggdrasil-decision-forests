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

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/cli/generated_model.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/decision_forest/decision_forest.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace cli {
namespace {

std::string TestDataDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/test_data");
}

using test::EqualsProto;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::SizeIs;

// Margin of error for numerical tests.
constexpr float kTestPrecision = 0.00001f;

// TODO: Add a test with a model with oblique weights.
TEST(CompileModelTest, DataSpec) {
  ASSERT_OK_AND_ASSIGN(const auto compiled_model,
                       compiled_model::test_model::GetModel());

  std::unique_ptr<model::AbstractModel> uncompiled_model;
  ASSERT_OK(model::LoadModel(file::JoinPath(TestDataDir(), "model",
                                            "synthetic_ranking_gbdt_numerical"),
                             &uncompiled_model));
  EXPECT_THAT(compiled_model->internal_features.data_spec(),
              EqualsProto(uncompiled_model->data_spec()));
}

TEST(CompileModelTest, Metadata) {
  ASSERT_OK_AND_ASSIGN(const auto compiled_model,
                       compiled_model::test_model::GetModel());

  std::unique_ptr<model::AbstractModel> uncompiled_model;
  ASSERT_OK(model::LoadModel(file::JoinPath(TestDataDir(), "model",
                                            "synthetic_ranking_gbdt_numerical"),
                             &uncompiled_model));
  model::proto::Metadata uncompiled_model_metadata_proto;
  uncompiled_model->metadata().Export(&uncompiled_model_metadata_proto);
  // The owner is sanitized during model compilation.
  uncompiled_model_metadata_proto.clear_owner();
  EXPECT_THAT(compiled_model->metadata,
              EqualsProto(uncompiled_model_metadata_proto));
}

TEST(CompileModelTest, GBTModelParameters) {
  ASSERT_OK_AND_ASSIGN(const auto compiled_model,
                       compiled_model::test_model::GetModel());

  std::unique_ptr<model::AbstractModel> uncompiled_model;
  ASSERT_OK(model::LoadModel(file::JoinPath(TestDataDir(), "model",
                                            "synthetic_ranking_gbdt_numerical"),
                             &uncompiled_model));

  auto* gbt_model =
      dynamic_cast<model::gradient_boosted_trees::GradientBoostedTreesModel*>(
          uncompiled_model.get());
  ASSERT_NE(gbt_model, nullptr);
  EXPECT_THAT(compiled_model->root_offsets, SizeIs(gbt_model->NumTrees()));
  EXPECT_THAT(gbt_model->initial_predictions(),
              ElementsAre(compiled_model->initial_predictions));
}

TEST(CompiledModelTest, ModelPredictions) {
  ASSERT_OK_AND_ASSIGN(const auto compiled_model,
                       compiled_model::test_model::GetModel());

  std::unique_ptr<model::AbstractModel> uncompiled_model;
  ASSERT_OK(model::LoadModel(file::JoinPath(TestDataDir(), "model",
                                            "synthetic_ranking_gbdt_numerical"),
                             &uncompiled_model));

  const auto& test_ds_path = absl::StrCat(
      "csv:",
      file::JoinPath(TestDataDir(), "dataset", "synthetic_ranking_test.csv"));
  dataset::VerticalDataset dataset;
  ASSERT_OK(dataset::LoadVerticalDataset(
      test_ds_path, compiled_model->internal_features.data_spec(), &dataset));

  std::vector<float> slow_engine_predictions;
  slow_engine_predictions.resize(dataset.nrow());
  for (dataset::VerticalDataset::row_t example_idx = 0;
       example_idx < dataset.nrow(); example_idx++) {
    model::proto::Prediction prediction;
    uncompiled_model->Predict(dataset, example_idx, &prediction);
    slow_engine_predictions[example_idx] = prediction.ranking().relevance();
  }

  std::vector<float> flat_examples;
  auto feature_names =
      FeatureNames(compiled_model->internal_features.fixed_length_features());
  auto replacement_values =
      compiled_model->internal_features.fixed_length_na_replacement_values();
  ASSERT_OK(serving::decision_forest::LoadFlatBatchFromDataset(
      dataset, 0, dataset.nrow(), feature_names, replacement_values,
      &flat_examples, serving::ExampleFormat::FORMAT_EXAMPLE_MAJOR));

  std::vector<float> compiled_model_predictions;
  compiled_model_predictions.resize(dataset.nrow());
  yggdrasil_decision_forests::serving::decision_forest::PredictOptimizedV1(
      *compiled_model, flat_examples, dataset.nrow(),
      &compiled_model_predictions);
  EXPECT_THAT(compiled_model_predictions,
              Pointwise(FloatNear(kTestPrecision), slow_engine_predictions));
}

}  // namespace
}  // namespace cli
}  // namespace yggdrasil_decision_forests
