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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/postprocessor/abstract_postprocessor.pb.h"
#include "yggdrasil_decision_forests/model/postprocessor/smoothed_pav_calibrator/smoothed_pav_calibrator.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"

namespace yggdrasil_decision_forests::model::postprocessor {
namespace {

using ::testing::FloatNear;

TEST(SmoothedPavCalibratorTest, BasicCalibration) {
  // Setup a calibrator that applies a shift of +0.1 to predictions.
  // The delta lookup table has size 101, and all delta values are 0.1.
  smoothed_pav_calibrator::proto::SmoothedPavCalibrator config;
  for (int i = 0; i < 101; ++i) {
    config.add_table_delta(0.1f);
  }

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

  // Expected calibrated score = 0.5 + 0.1 = 0.6.
  // Expected positive class counts = 0.6 * 10 = 6.0.
  // Expected negative class counts = (1 - 0.6) * 10 = 4.0.
  EXPECT_THAT(prediction.classification().distribution().counts(2),
              FloatNear(6.0f, 1e-5f));
  EXPECT_THAT(prediction.classification().distribution().counts(1),
              FloatNear(4.0f, 1e-5f));
}

TEST(SmoothedPavCalibratorTest, ProcessExample) {
  // Setup a calibrator with identity delta (0.0).
  smoothed_pav_calibrator::proto::SmoothedPavCalibrator config;
  for (int i = 0; i < 101; ++i) {
    config.add_table_delta(0.0f);
  }

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
  config.add_table_delta(0.1f);
  config.add_table_delta(-0.1f);

  auto calibrator = Create(config);
  ASSERT_NE(calibrator, nullptr);

  proto::AbstractPostprocessor exported_proto;
  calibrator->ExportProto(&exported_proto);

  EXPECT_TRUE(exported_proto.enabled());
  ASSERT_TRUE(exported_proto.has_smoothed_pav_calibrator());
  const auto& inner = exported_proto.smoothed_pav_calibrator();
  ASSERT_EQ(inner.table_delta_size(), 2);
  EXPECT_THAT(inner.table_delta(0), FloatNear(0.1f, 1e-5f));
  EXPECT_THAT(inner.table_delta(1), FloatNear(-0.1f, 1e-5f));
}

}  // namespace
}  // namespace yggdrasil_decision_forests::model::postprocessor
