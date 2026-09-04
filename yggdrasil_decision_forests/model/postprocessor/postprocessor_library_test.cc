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

#include "yggdrasil_decision_forests/model/postprocessor/postprocessor_library.h"

#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/model/postprocessor/postprocessor.pb.h"
#include "yggdrasil_decision_forests/model/postprocessor/smoothed_pav_calibrator/smoothed_pav_calibrator.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests::model::postprocessor {
namespace {

using ::yggdrasil_decision_forests::test::StatusIs;

TEST(PostprocessorLibraryTest, CreatePostprocessorUnimplemented) {
  proto::Postprocessor config;
  config.set_enabled(true);

  EXPECT_THAT(CreatePostprocessor(config).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unknown postprocessor type."));
}

TEST(PostprocessorLibraryTest, CreateSmoothedPavCalibrator) {
  proto::Postprocessor config;
  config.set_enabled(true);
  auto* calibrator_config = config.mutable_smoothed_pav_calibrator();
  calibrator_config->add_x(0.0f);
  calibrator_config->add_x(1.0f);
  calibrator_config->add_y(0.0f);
  calibrator_config->add_y(1.0f);
  calibrator_config->add_slope(1.0f);
  calibrator_config->add_slope(1.0f);
  calibrator_config->set_n_grid(100);

  auto calibrator_or = CreatePostprocessor(config);
  ASSERT_OK(calibrator_or.status());
  auto calibrator = std::move(calibrator_or.value());
  ASSERT_NE(calibrator, nullptr);
  EXPECT_TRUE(calibrator->enabled());

  proto::Postprocessor exported_config;
  calibrator->ExportProto(&exported_config);
  EXPECT_TRUE(exported_config.has_smoothed_pav_calibrator());
}

}  // namespace
}  // namespace yggdrasil_decision_forests::model::postprocessor
