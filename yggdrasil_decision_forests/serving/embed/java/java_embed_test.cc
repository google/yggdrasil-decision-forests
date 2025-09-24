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

// Test the code to embed models.

#include <memory>
#include <optional>
#include <string>

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/embed/embed.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::serving::embed {
namespace {

std::string TestDataDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/test_data");
}

struct TestData {
  std::unique_ptr<model::AbstractModel> model;
};

struct GoldenGeneratedJavaCase {
  std::string model_filename;
  std::string golden_filename;
  proto::Algorithm::Enum algorithm;
  std::optional<proto::ClassificationOutput::Enum> output;
  int crop_num_trees = 3;
  bool categorical_from_string = false;
};

// Compare the generated .h files against golden files.
SIMPLE_PARAMETERIZED_TEST(
    GoldenGeneratedJava, GoldenGeneratedJavaCase,
    {
        // GBT
        {
            "adult_binary_class_gbdt_v2",
            "adult_binary_class_gbdt_v2_probability_routing.java.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::PROBABILITY,
        },
        {
            "iris_multi_class_gbdt_v2",
            "iris_multi_class_gbdt_v2_probability_routing.java.golden",
            proto::Algorithm::ROUTING,
        },
        {
            "abalone_regression_gbdt_v2",
            "abalone_regression_gbdt_v2_routing.java.golden",
            proto::Algorithm::ROUTING,
        },
        {
            "adult_binary_class_gbdt_oblique",
            "adult_binary_class_gbdt_oblique_proba_routing.java.golden",
            proto::Algorithm::ROUTING,
            proto::ClassificationOutput::PROBABILITY,
        },
    }) {
  const auto& test_case = GetParam();

  ASSERT_OK_AND_ASSIGN(
      auto model, model::LoadModel(file::JoinPath(TestDataDir(), "model",
                                                  test_case.model_filename)));
  auto df = dynamic_cast<model::DecisionForestInterface*>(model.get());
  if (df) {
    df->mutable_decision_trees()->resize(test_case.crop_num_trees);
  }

  proto::Options options;
  options.set_name("YdfModel");
  options.mutable_java();
  options.set_algorithm(test_case.algorithm);
  options.set_categorical_from_string(test_case.categorical_from_string);
  if (test_case.output.has_value()) {
    options.set_classification_output(*test_case.output);
  }
  ASSERT_OK_AND_ASSIGN(const auto embed, EmbedModel(*model, options));
  EXPECT_EQ(embed.size(), 1);
  EXPECT_TRUE(embed.contains("YdfModel.java"));

  test::ExpectEqualGolden(
      embed.at("YdfModel.java"),
      file::JoinPath("yggdrasil_decision_forests/test_data/"
                     "golden/embed",
                     test_case.golden_filename));
}

}  // namespace
}  // namespace yggdrasil_decision_forests::serving::embed
