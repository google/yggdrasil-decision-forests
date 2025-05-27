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

#include "yggdrasil_decision_forests/serving/embed/embed.h"

#include <string>

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::serving::embed {
namespace {

std::string TestDataDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/test_data");
}

TEST(Embed, SimpleModel) {
  ASSERT_OK_AND_ASSIGN(const auto model,
                       model::LoadModel(file::JoinPath(
                           TestDataDir(), "model", "adult_binary_class_gbdt")));
  ASSERT_OK_AND_ASSIGN(const auto embed, EmbedModelCC(*model));
  EXPECT_EQ(embed.size(), 1);
  EXPECT_TRUE(embed.contains("my_model.h"));
  test::ExpectEqualGolden(embed.at("my_model.h"),
                          "yggdrasil_decision_forests/test_data/"
                          "golden/embed/model1.h.golden");
}

}  // namespace
}  // namespace yggdrasil_decision_forests::serving::embed
