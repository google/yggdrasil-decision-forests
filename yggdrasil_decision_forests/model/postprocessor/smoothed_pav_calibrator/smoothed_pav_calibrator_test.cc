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

#include "yggdrasil_decision_forests/model/postprocessor/smoothed_pav_calibrator/smoothed_pav_calibrator.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/postprocessor/postprocessor.pb.h"
#include "yggdrasil_decision_forests/model/postprocessor/smoothed_pav_calibrator/smoothed_pav_calibrator.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"

namespace yggdrasil_decision_forests::model::postprocessor {
namespace {

using ::testing::FloatNear;
using ::testing::HasSubstr;

TEST(SmoothedPavCalibratorTest, BasicCalibration) {
  // Setup a calibrator that applies a linear calibration curve.
  // Knot 0: (x=0.0, y=0.2, slope=0.8)
  // Knot 1: (x=1.0, y=1.0, slope=0.8)
  // For x = 0.5, h(x) = 0.2 + 0.8 * 0.5 = 0.6.
  smoothed_pav_calibrator::proto::SmoothedPavCalibrator config;
  config.add_x(0.0f);
  config.add_x(1.0f);
  config.add_y(0.2f);
  config.add_y(1.0f);
  config.add_slope(0.8f);
  config.add_slope(0.8f);

  // Create the calibrator.
  auto calibrator = Create(config);
  ASSERT_NE(calibrator, nullptr);

  // Prepare a prediction object.
  // Positive class index = 2. Negative class index = 1.
  // We want to test calibration on a score of 0.5.
  // If positive count is 5 and sum is 10, positive probability is 5 / 10 = 0.5.
  yggdrasil_decision_forests::model::proto::Prediction prediction;
  auto* distribution =
      prediction.mutable_classification()->mutable_distribution();
  distribution->set_sum(10.0f);
  distribution->add_counts(0.0f);  // Out of vocabulary
  distribution->add_counts(5.0f);  // Negative class (index 1)
  distribution->add_counts(5.0f);  // Positive class (index 2)

  // Calibrate
  dataset::VerticalDataset dataset;
  calibrator->Process(dataset, 0, &prediction);

  // Expected calibrated score = 0.6.
  // Expected positive class counts = 0.6 * 10 = 6.0.
  // Expected negative class counts = (1 - 0.6) * 10 = 4.0.
  EXPECT_THAT(prediction.classification().distribution().counts(2),
              FloatNear(6.0f, 1e-5f));
  EXPECT_THAT(prediction.classification().distribution().counts(1),
              FloatNear(4.0f, 1e-5f));
}

TEST(SmoothedPavCalibratorTest, ProcessExample) {
  // Setup a calibrator with identity curve.
  smoothed_pav_calibrator::proto::SmoothedPavCalibrator config;
  config.add_x(0.0f);
  config.add_x(1.0f);
  config.add_y(0.0f);
  config.add_y(1.0f);
  config.add_slope(1.0f);
  config.add_slope(1.0f);

  auto calibrator = Create(config);
  ASSERT_NE(calibrator, nullptr);

  yggdrasil_decision_forests::model::proto::Prediction prediction;
  auto* distribution =
      prediction.mutable_classification()->mutable_distribution();
  distribution->set_sum(10.0f);
  distribution->add_counts(0.0f);
  distribution->add_counts(3.0f);  // Index 1
  distribution->add_counts(7.0f);  // Index 2

  dataset::proto::Example example;
  calibrator->Process(example, &prediction);

  // Expected unchanged counts
  EXPECT_THAT(prediction.classification().distribution().counts(2),
              FloatNear(7.0f, 1e-5f));
  EXPECT_THAT(prediction.classification().distribution().counts(1),
              FloatNear(3.0f, 1e-5f));
}

TEST(SmoothedPavCalibratorTest, ExportProto) {
  smoothed_pav_calibrator::proto::SmoothedPavCalibrator config;
  config.add_x(0.0f);
  config.add_x(0.5f);
  config.add_x(1.0f);
  config.add_y(0.0f);
  config.add_y(0.6f);
  config.add_y(1.0f);
  config.add_slope(1.2f);
  config.add_slope(1.0f);
  config.add_slope(0.8f);
  config.set_n_grid(500);

  auto calibrator = Create(config);
  ASSERT_NE(calibrator, nullptr);

  proto::Postprocessor exported_proto;
  calibrator->ExportProto(&exported_proto);

  EXPECT_TRUE(exported_proto.enabled());
  ASSERT_TRUE(exported_proto.has_smoothed_pav_calibrator());
  const auto& inner = exported_proto.smoothed_pav_calibrator();
  ASSERT_EQ(inner.x_size(), 3);
  EXPECT_THAT(inner.x(0), FloatNear(0.0f, 1e-5f));
  EXPECT_THAT(inner.x(1), FloatNear(0.5f, 1e-5f));
  EXPECT_THAT(inner.x(2), FloatNear(1.0f, 1e-5f));

  ASSERT_EQ(inner.y_size(), 3);
  EXPECT_THAT(inner.y(0), FloatNear(0.0f, 1e-5f));
  EXPECT_THAT(inner.y(1), FloatNear(0.6f, 1e-5f));
  EXPECT_THAT(inner.y(2), FloatNear(1.0f, 1e-5f));

  ASSERT_EQ(inner.slope_size(), 3);
  EXPECT_THAT(inner.slope(0), FloatNear(1.2f, 1e-5f));
  EXPECT_THAT(inner.slope(1), FloatNear(1.0f, 1e-5f));
  EXPECT_THAT(inner.slope(2), FloatNear(0.8f, 1e-5f));

  EXPECT_EQ(inner.n_grid(), 500);
}

TEST(SmoothedPavCalibratorTest, AppendDescription) {
  smoothed_pav_calibrator::proto::SmoothedPavCalibrator config;
  config.add_x(0.0f);
  config.add_x(1.0f);
  config.add_y(0.0f);
  config.add_y(1.0f);
  config.add_slope(1.0f);
  config.add_slope(1.0f);
  config.set_n_grid(100);

  auto calibrator = Create(config);
  ASSERT_NE(calibrator, nullptr);

  std::string description;
  calibrator->AppendDescription(&description);

  EXPECT_THAT(description, HasSubstr("Smoothed PAV calibrator\n"));
  EXPECT_THAT(description, HasSubstr("Number of raw bins: 2\n"));
  EXPECT_THAT(description, HasSubstr("Number of lookup table bins: 100\n"));
}

TEST(SmoothedPavCalibratorTest, BoundaryValues) {
  // Knots mapping [0.0, 1.0] -> [0.1, 0.9].
  smoothed_pav_calibrator::proto::SmoothedPavCalibrator config;
  config.add_x(0.0f);
  config.add_x(1.0f);
  config.add_y(0.1f);
  config.add_y(0.9f);
  config.add_slope(0.8f);
  config.add_slope(0.8f);

  auto calibrator = Create(config);
  ASSERT_NE(calibrator, nullptr);

  // Test p = 0.0 (positive count = 0.0, negative count = 10.0).
  {
    yggdrasil_decision_forests::model::proto::Prediction prediction;
    auto* distribution =
        prediction.mutable_classification()->mutable_distribution();
    distribution->set_sum(10.0f);
    distribution->add_counts(0.0f);
    distribution->add_counts(10.0f);  // Negative class
    distribution->add_counts(0.0f);   // Positive class

    dataset::proto::Example example;
    calibrator->Process(example, &prediction);

    // Expected calibrated score = 0.1 -> positive count = 1.0, negative = 9.0.
    EXPECT_THAT(prediction.classification().distribution().counts(2),
                FloatNear(1.0f, 1e-5f));
    EXPECT_THAT(prediction.classification().distribution().counts(1),
                FloatNear(9.0f, 1e-5f));
  }

  // Test p = 1.0 (positive count = 10.0, negative count = 0.0).
  {
    yggdrasil_decision_forests::model::proto::Prediction prediction;
    auto* distribution =
        prediction.mutable_classification()->mutable_distribution();
    distribution->set_sum(10.0f);
    distribution->add_counts(0.0f);
    distribution->add_counts(0.0f);    // Negative class
    distribution->add_counts(10.0f);   // Positive class

    dataset::proto::Example example;
    calibrator->Process(example, &prediction);

    // Expected calibrated score = 0.9 -> positive count = 9.0, negative = 1.0.
    EXPECT_THAT(prediction.classification().distribution().counts(2),
                FloatNear(9.0f, 1e-5f));
    EXPECT_THAT(prediction.classification().distribution().counts(1),
                FloatNear(1.0f, 1e-5f));
  }
}

}  // namespace
}  // namespace yggdrasil_decision_forests::model::postprocessor
