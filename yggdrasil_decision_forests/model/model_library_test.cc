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

#include "yggdrasil_decision_forests/model/model_library.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace {

std::string TestDataDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/test_data");
}

TEST(ModelLibrary, CreateAllModels) {
  for (const auto& model_name : AllRegisteredModels()) {
    std::unique_ptr<AbstractModel> mdl;
    EXPECT_OK(CreateEmptyModel(model_name, &mdl));
  }
}

TEST(ModelLibrary, DetectsSavedModelTrue) {
  std::unique_ptr<model::AbstractModel> model;
  std::string model_directory =
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_gbdt");
  ASSERT_OK_AND_ASSIGN(const bool is_tfdf_model,
                       IsTensorFlowSavedModel(model_directory));
  EXPECT_FALSE(is_tfdf_model);
}

TEST(ModelLibrary, DetectsTFDFModel) {
  std::unique_ptr<model::AbstractModel> model;
  std::string model_directory = file::JoinPath(
      TestDataDir(), "model", "adult_binary_class_gbdt_savedmodel");
  ASSERT_OK_AND_ASSIGN(const bool is_tfdf_model,
                       IsTensorFlowSavedModel(model_directory));
  EXPECT_TRUE(is_tfdf_model);
}

TEST(ModelLibrary, LoadTFDFModel) {
  std::unique_ptr<model::AbstractModel> model;
  std::string model_directory = file::JoinPath(
      TestDataDir(), "model", "adult_binary_class_gbdt_savedmodel");
  EXPECT_OK(LoadModel(model_directory, &model));
}

}  // namespace
}  // namespace model
}  // namespace yggdrasil_decision_forests
