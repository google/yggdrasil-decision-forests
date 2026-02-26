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

#include "gtest/gtest.h"

namespace yggdrasil_decision_forests::serving::embed {
namespace {

template <typename T>
T CreateAbaloneExample() {
  return T{
      .type = 1,
      .longestshell = 0.455f,
      .diameter = 0.365f,
      .height = 0.095f,
      .wholeweight = 0.514f,
      .shuckedweight = 0.2245f,
      .visceraweight = 0.101f,
      .shellweight = 0.15f,
  };
}

constexpr double eps = 0.00001;

// Regression

TEST(Embed, test_model_abalone_regression_gbdt_v2_routing) {
  // TODO : b/481596783 - Implement the actual test.
  EXPECT_NEAR(9.815922, 9.815921, eps);
}

}  // namespace
}  // namespace yggdrasil_decision_forests::serving::embed
