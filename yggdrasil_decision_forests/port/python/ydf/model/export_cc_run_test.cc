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

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ydf/model/export_cc_generated_lib.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace {

std::string TestDataDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/test_data");
}

TEST(RunModel, Base) {
  const auto model = exported_model_123::Load(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_gbdt"));
  ASSERT_OK(model.status());

  const auto predictions = model->Predict();

  YDF_LOG(INFO) << "Predictions:";
  for (const float p : predictions) {
    YDF_LOG(INFO) << p;
  }
}

}  // namespace
}  // namespace yggdrasil_decision_forests
