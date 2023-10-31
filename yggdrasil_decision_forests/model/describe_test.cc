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

#include "yggdrasil_decision_forests/model/describe.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::model {
namespace {

std::string TestDataDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/test_data");
}

TEST(Describe, GBT) {
  std::unique_ptr<model::AbstractModel> model;
  ASSERT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_gbdt"),
      &model));
  ASSERT_OK_AND_ASSIGN(const auto html, DescribeModelHtml(*model, "123"));
  test::ExpectEqualGolden(html,
                          "yggdrasil_decision_forests/test_data/"
                          "golden/describe_gbt.html.expected");
}

TEST(Describe, RF) {
  std::unique_ptr<model::AbstractModel> model;
  ASSERT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_rf"), &model));
  ASSERT_OK_AND_ASSIGN(const auto html, DescribeModelHtml(*model, "123"));
  test::ExpectEqualGolden(html,
                          "yggdrasil_decision_forests/test_data/"
                          "golden/describe_rf.html.expected");
}

TEST(Describe, TunedGBT) {
  std::unique_ptr<model::AbstractModel> model;
  ASSERT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_gbdt_tuned"),
      &model));
  ASSERT_OK_AND_ASSIGN(const auto html, DescribeModelHtml(*model, "123"));
  test::ExpectEqualGolden(html,
                          "yggdrasil_decision_forests/test_data/"
                          "golden/describe_gbt_tuned.html.expected");
}

}  // namespace
}  // namespace yggdrasil_decision_forests::model
