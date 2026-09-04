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

#include "yggdrasil_decision_forests/utils/reliability_diagram.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "yggdrasil_decision_forests/utils/smoothed_pav_calibration_fit.h"

namespace yggdrasil_decision_forests::utils::reliability_diagram {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::IsEmpty;
using AccumulatorType = BinAccumulator::AccumulatorType;

constexpr float kFloatEps = 1e-6f;

TEST(ReliabilityDiagramTest, InitialStateAndEmptyMetrics) {
  ReliabilityDiagram diag(10);
  EXPECT_EQ(diag.n_bins(), 10);
  EXPECT_THAT(diag.bin_mean_predicted(), IsEmpty());
  EXPECT_THAT(diag.bin_mean_observed(), IsEmpty());
  EXPECT_THAT(diag.bin_counts(), IsEmpty());
  EXPECT_FLOAT_EQ(diag.ece(), 0.0f);
  EXPECT_FLOAT_EQ(diag.mce(), 0.0f);
}

TEST(ReliabilityDiagramTest, BatchUpdateAndDiagnostics) {
  ReliabilityDiagram diag(10);

  // Populating bins 0 ([0.0, 0.1)), 4 ([0.4, 0.5)), and 9 ([0.9, 1.0]).
  const std::vector<AccumulatorType> p = {0.02f, 0.08f, 0.45f, 0.92f, 0.98f};
  const std::vector<AccumulatorType> y = {0.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  EXPECT_THAT(diag.update(p, y), IsOk());

  // Non-empty bins: bin 0, bin 4, bin 9 (3 bins in total).
  EXPECT_THAT(diag.bin_counts(), ElementsAre(2.0f, 1.0f, 2.0f));

  // bin 0 mean pred: (0.02 + 0.08) / 2 = 0.05
  // bin 4 mean pred: 0.45
  // bin 9 mean pred: (0.92 + 0.98) / 2 = 0.95
  const auto mean_preds = diag.bin_mean_predicted();
  ASSERT_EQ(mean_preds.size(), 3);
  EXPECT_THAT(mean_preds[0], FloatNear(0.05f, kFloatEps));
  EXPECT_THAT(mean_preds[1], FloatNear(0.45f, kFloatEps));
  EXPECT_THAT(mean_preds[2], FloatNear(0.95f, kFloatEps));

  // bin 0 mean obs: (0.0 + 1.0) / 2 = 0.5
  // bin 4 mean obs: 1.0
  // bin 9 mean obs: (1.0 + 1.0) / 2 = 1.0
  const auto mean_obs = diag.bin_mean_observed();
  ASSERT_EQ(mean_obs.size(), 3);
  EXPECT_THAT(mean_obs[0], FloatNear(0.5f, kFloatEps));
  EXPECT_THAT(mean_obs[1], FloatNear(1.0f, kFloatEps));
  EXPECT_THAT(mean_obs[2], FloatNear(1.0f, kFloatEps));

  // ECE and MCE:
  // bin 0 gap = |0.05 - 0.50| = 0.45, count = 2 -> weighted = 0.90
  // bin 4 gap = |0.45 - 1.00| = 0.55, count = 1 -> weighted = 0.55
  // bin 9 gap = |0.95 - 1.00| = 0.05, count = 2 -> weighted = 0.10
  // total count = 5, total weighted gap = 1.55
  // ECE = 1.55 / 5 = 0.31
  // MCE = max(0.45, 0.55, 0.05) = 0.55
  EXPECT_THAT(diag.ece(), FloatNear(0.31f, kFloatEps));
  EXPECT_THAT(diag.mce(), FloatNear(0.55f, kFloatEps));
}

TEST(ReliabilityDiagramTest, SinglePointStreamingUpdate) {
  ReliabilityDiagram diag_stream(5);
  ReliabilityDiagram diag_batch(5);

  const std::vector<AccumulatorType> p = {0.1f, 0.35f, 0.72f, 0.95f};
  const std::vector<AccumulatorType> y = {0.0f, 1.0f, 0.0f, 1.0f};

  for (size_t i = 0; i < p.size(); ++i) {
    EXPECT_THAT(diag_stream.update(p[i], y[i]), IsOk());
  }
  EXPECT_THAT(diag_batch.update(p, y), IsOk());

  EXPECT_EQ(diag_stream.bin_counts(), diag_batch.bin_counts());
  EXPECT_EQ(diag_stream.bin_mean_predicted(), diag_batch.bin_mean_predicted());
  EXPECT_EQ(diag_stream.bin_mean_observed(), diag_batch.bin_mean_observed());
  EXPECT_FLOAT_EQ(diag_stream.ece(), diag_batch.ece());
  EXPECT_FLOAT_EQ(diag_stream.mce(), diag_batch.mce());
}

TEST(ReliabilityDiagramTest, IncrementalMultipleBatchUpdates) {
  ReliabilityDiagram diag(10);

  // First chunk into bin 1 ([0.1, 0.2))
  const std::vector<AccumulatorType> p1 = {0.15f};
  const std::vector<AccumulatorType> y1 = {1.0f};
  EXPECT_THAT(diag.update(p1, y1), IsOk());

  // Second chunk into bin 1 and bin 8 ([0.8, 0.9))
  const std::vector<AccumulatorType> p2 = {0.18f, 0.85f};
  const std::vector<AccumulatorType> y2 = {0.0f, 1.0f};
  EXPECT_THAT(diag.update(p2, y2), IsOk());

  // Bin 1 now has 2 samples, Bin 8 has 1 sample.
  EXPECT_THAT(diag.bin_counts(), ElementsAre(2.0f, 1.0f));

  const auto mean_preds = diag.bin_mean_predicted();
  ASSERT_EQ(mean_preds.size(), 2);
  EXPECT_THAT(mean_preds[0], FloatNear((0.15f + 0.18f) / 2.0f, kFloatEps));
  EXPECT_THAT(mean_preds[1], FloatNear(0.85f, kFloatEps));

  const auto mean_obs = diag.bin_mean_observed();
  ASSERT_EQ(mean_obs.size(), 2);
  EXPECT_THAT(mean_obs[0], FloatNear(0.5f, kFloatEps));
  EXPECT_THAT(mean_obs[1], FloatNear(1.0f, kFloatEps));
}

TEST(ReliabilityDiagramTest, ErrorHandling) {
  ReliabilityDiagram diag(10);
  const std::vector<AccumulatorType> p = {0.1f, 0.2f};
  const std::vector<AccumulatorType> y = {1.0f};

  EXPECT_THAT(diag.update(p, y), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ReliabilityDiagramTest, BoundaryAndClampingBehavior) {
  // 5 bins: [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
  ReliabilityDiagram diag(5);

  const std::vector<AccumulatorType> p = {-0.5f, 0.0f, 1.0f, 1.5f};
  const std::vector<AccumulatorType> y = {0.0f, 1.0f, 0.0f, 1.0f};

  EXPECT_THAT(diag.update(p, y), IsOk());

  // -0.5 is clamped to 0.0 -> bin 0
  // 0.0 -> bin 0
  // 1.0 -> bin 4 (last bin)
  // 1.5 is clamped to 1.0 -> bin 4 (last bin)
  EXPECT_THAT(diag.bin_counts(), ElementsAre(2.0f, 2.0f));
}

TEST(ReliabilityDiagramTest, ResetFunctionality) {
  ReliabilityDiagram diag(5);

  const std::vector<AccumulatorType> p = {0.1f, 0.9f};
  const std::vector<AccumulatorType> y = {0.0f, 1.0f};
  EXPECT_THAT(diag.update(p, y), IsOk());
  EXPECT_EQ(diag.bin_counts().size(), 2);

  diag.reset();

  EXPECT_EQ(diag.n_bins(), 5);
  EXPECT_THAT(diag.bin_mean_predicted(), IsEmpty());
  EXPECT_THAT(diag.bin_mean_observed(), IsEmpty());
  EXPECT_THAT(diag.bin_counts(), IsEmpty());
  EXPECT_FLOAT_EQ(diag.ece(), 0.0f);
  EXPECT_FLOAT_EQ(diag.mce(), 0.0f);

  // Can accumulate new points after reset
  EXPECT_THAT(diag.update(0.5f, 1.0f), IsOk());
  EXPECT_THAT(diag.bin_counts(), ElementsAre(1.0f));
}

TEST(ReliabilityDiagramTest, CalibrationMetricsPerfectCalibration) {
  // 2 bins: [0, 0.5), [0.5, 1.0]
  ReliabilityDiagram diag(2);

  const std::vector<AccumulatorType> p = {0.25f, 0.25f, 0.25f, 0.25f,
                                          0.75f, 0.75f, 0.75f, 0.75f};
  const std::vector<AccumulatorType> y = {1.0f, 0.0f, 0.0f, 0.0f,
                                          1.0f, 1.0f, 1.0f, 0.0f};

  EXPECT_THAT(diag.update(p, y), IsOk());
  EXPECT_THAT(diag.ece(), FloatNear(0.0f, kFloatEps));
  EXPECT_THAT(diag.mce(), FloatNear(0.0f, kFloatEps));
}

TEST(ReliabilityDiagramTest, CalibrationMetricsKnownValues) {
  // 4 bins: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
  ReliabilityDiagram diag(4);

  // Bin 0 [0, 0.25): 30 samples at p=0.1, y has sum 9 (prob_true = 9/30 = 0.3)
  //   |0.1 - 0.3| = 0.2
  // Bin 3 [0.75, 1.0]: 10 samples at p=0.8, y has sum 2
  //   (prob_true = 2/10 = 0.2) |0.8 - 0.2| = 0.6
  std::vector<AccumulatorType> p;
  std::vector<AccumulatorType> y;

  for (int i = 0; i < 30; ++i) {
    p.push_back(0.1f);
    y.push_back(i < 9 ? 1.0f : 0.0f);
  }
  for (int i = 0; i < 10; ++i) {
    p.push_back(0.8f);
    y.push_back(i < 2 ? 1.0f : 0.0f);
  }

  EXPECT_THAT(diag.update(p, y), IsOk());

  // Total count = 40
  // Bin 0 weighted gap = 30 * 0.2 = 6.0
  // Bin 3 weighted gap = 10 * 0.6 = 6.0
  // ECE = (6.0 + 6.0) / 40 = 0.30
  // MCE = max(0.2, 0.6) = 0.60
  EXPECT_THAT(diag.ece(), FloatNear(0.30f, kFloatEps));
  EXPECT_THAT(diag.mce(), FloatNear(0.60f, kFloatEps));
}

}  // namespace
}  // namespace yggdrasil_decision_forests::utils::reliability_diagram
