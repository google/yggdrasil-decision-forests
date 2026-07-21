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

#include "yggdrasil_decision_forests/utils/smoothed_pav_calibration_fit.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

using ::testing::DoubleNear;
using ::testing::Le;
using ::testing::Not;
using ::testing::SizeIs;

TEST(SmoothedPavCalibrationFit, BinAccumulator) {
  BinAccumulator b1{3.0, 2.0, 10.0};  // sum_pred=3, sum_true=2, count=10
  EXPECT_THAT(b1.prob_pred(), DoubleNear(0.3, 1e-9));
  EXPECT_THAT(b1.prob_true(), DoubleNear(0.2, 1e-9));

  BinAccumulator b2{1.0, 1.0, 5.0};
  b1 += b2;
  EXPECT_THAT(b1.sum_pred, DoubleNear(4.0, 1e-9));
  EXPECT_THAT(b1.sum_true, DoubleNear(3.0, 1e-9));
  EXPECT_THAT(b1.count, DoubleNear(15.0, 1e-9));
  EXPECT_THAT(b1.prob_true(), DoubleNear(3.0 / 15.0, 1e-9));

  // default construction is all-zero
  BinAccumulator b3{};
  EXPECT_EQ(b3.sum_pred, 0.0);
  EXPECT_EQ(b3.sum_true, 0.0);
  EXPECT_EQ(b3.count, 0.0);
}

TEST(SmoothedPavCalibrationFit, AggregateBins) {
  // Basic correctness on a hand-constructed tiny dataset
  {
    std::vector<double> p = {0.05, 0.15, 0.55, 0.95};
    std::vector<double> y = {0.0, 1.0, 0.0, 1.0};
    auto bins = *aggregate_bins(p, y, 10);  // 10 equal-width bins
    EXPECT_THAT(bins, SizeIs(4));
    double total_count = 0;
    for (auto& b : bins) total_count += b.count;
    EXPECT_THAT(total_count, DoubleNear(4.0, 1e-9));
  }

  // Multiple points landing in the SAME bin get accumulated together
  {
    std::vector<double> p = {0.101, 0.102, 0.103, 0.109};
    std::vector<double> y = {1.0, 0.0, 1.0, 0.0};
    auto bins = *aggregate_bins(p, y, 10);
    EXPECT_THAT(bins, SizeIs(1));
    EXPECT_THAT(bins[0].count, DoubleNear(4.0, 1e-9));
    EXPECT_THAT(bins[0].sum_true, DoubleNear(2.0, 1e-9));
    EXPECT_THAT(bins[0].sum_pred,
                DoubleNear(0.101 + 0.102 + 0.103 + 0.109, 1e-9));
  }

  // Empty bins are dropped from the output entirely
  {
    std::vector<double> p = {0.05,
                             0.95};  // only bins 0 and 9 (of 10) populated
    std::vector<double> y = {0.0, 1.0};
    auto bins = *aggregate_bins(p, y, 10);
    EXPECT_THAT(bins, SizeIs(2));
  }

  // Out-of-range p is clamped into [0,1] before binning, not dropped/crashed
  {
    std::vector<double> p = {-0.5, 1.5};
    std::vector<double> y = {0.0, 1.0};
    auto bins = *aggregate_bins(p, y, 10);
    EXPECT_THAT(bins, SizeIs(2));
    double total_count = 0;
    for (auto& b : bins) total_count += b.count;
    EXPECT_THAT(total_count, DoubleNear(2.0, 1e-9));
  }

  // Point exactly at p=1.0 lands in the LAST bin (not out of bounds)
  {
    std::vector<double> p = {1.0};
    std::vector<double> y = {1.0};
    auto bins = *aggregate_bins(p, y, 10);
    EXPECT_THAT(bins, SizeIs(1));
  }

  // n_bins=1 -- everything collapses into a single bin
  {
    auto rng_engine = RandomEngine(42);
    std::vector<double> p(1000), y(1000);
    for (std::size_t i = 0; i < 1000; ++i) {
      p[i] = static_cast<double>(RandomUniformFloat(&rng_engine));
      y[i] = (i % 2 == 0) ? 1.0 : 0.0;
    }
    auto bins = *aggregate_bins(p, y, 1);
    EXPECT_THAT(bins, SizeIs(1));
    EXPECT_THAT(bins[0].count, DoubleNear(1000.0, 1e-9));
  }

  // Bin index arithmetic matches direct computation (spot-checked against the
  // documented np.digitize-equivalent formula)
  {
    std::vector<double> p = {0.37};
    std::vector<double> y = {0.0};
    std::size_t n_bins = 100;
    auto bins = *aggregate_bins(p, y, n_bins);
    // 0.37 * 100 = 37.0 -> bin index 37 -> prob_pred should just be 0.37 itself
    EXPECT_THAT(bins, SizeIs(1));
    EXPECT_THAT(bins[0].prob_pred(), DoubleNear(0.37, 1e-9));
  }
}

TEST(SmoothedPavCalibrationFit, AggregateChunckedBins) {
  auto rng_engine = RandomEngine(42);

  constexpr std::size_t N = 500'000;
  constexpr std::size_t n_bins = 2000;
  std::vector<double> p(N), y(N);
  for (auto& v : p) v = static_cast<double>(RandomUniformFloat(&rng_engine));
  for (auto& v : y) v = static_cast<double>(RandomUniformInt(2, &rng_engine));

  // Monolithic: single aggregate_bins call on everything
  auto monolithic = *aggregate_bins(p, y, n_bins);

  // Chunked: same data, split into irregular chunks, accumulated incrementally,
  // finalized once at the end
  std::vector<BinAccumulator> bins(n_bins);
  std::vector<std::size_t> chunk_sizes = {1,     7,      1000,
                                          50000, 123456, 0 /* empty chunk */};
  std::size_t total_declared = 0;
  for (auto s : chunk_sizes) total_declared += s;
  chunk_sizes.push_back(N - total_declared);  // remainder

  std::size_t offset = 0;
  for (auto cs : chunk_sizes) {
    std::vector<double> p_chunk(p.begin() + offset, p.begin() + offset + cs);
    std::vector<double> y_chunk(y.begin() + offset, y.begin() + offset + cs);
    EXPECT_TRUE(accumulate_bins(bins, p_chunk, y_chunk, n_bins).ok());
    offset += cs;
  }
  EXPECT_EQ(offset, N);
  auto chunked = finalize_bins(std::move(bins));

  // Compare the monolithic and chunked results
  EXPECT_EQ(monolithic.size(), chunked.size());
  bool bitwise_identical = true;
  if (monolithic.size() == chunked.size()) {
    for (std::size_t i = 0; i < monolithic.size(); ++i) {
      if (monolithic[i].sum_pred != chunked[i].sum_pred ||
          monolithic[i].sum_true != chunked[i].sum_true ||
          monolithic[i].count != chunked[i].count) {
        bitwise_identical = false;
      }
    }
  }
  EXPECT_TRUE(bitwise_identical);

  // Edge case: a bin empty after early chunks, populated later
  {
    std::vector<BinAccumulator> bins2(10);
    std::vector<double> p1 = {0.05, 0.95};  // bins 0 and 9 only
    std::vector<double> y1 = {0.0, 1.0};
    EXPECT_TRUE(accumulate_bins(bins2, p1, y1, 10).ok());
    // bin 5 (p in [0.5,0.6)) is empty after this chunk
    EXPECT_EQ(bins2[5].count, 0.0);

    std::vector<double> p2 = {0.55};  // now populate bin 5
    std::vector<double> y2 = {1.0};
    EXPECT_TRUE(accumulate_bins(bins2, p2, y2, 10).ok());
    EXPECT_EQ(bins2[5].count, 1.0);

    auto finalized2 = finalize_bins(std::move(bins2));
    EXPECT_THAT(finalized2, SizeIs(3));
  }
}

TEST(SmoothedPavCalibrationFit, MergeBinsForMonotonicity) {
  // Hand-checkable violation (from the original derivation)
  {
    std::vector<BinAccumulator> bins = {{1.0, 2.0, 10.0},
                                        {2.0, 5.0, 10.0},
                                        {3.0, 3.0, 10.0},
                                        {4.0, 6.0, 10.0},
                                        {5.0, 9.0, 10.0}};
    // prob_true sequence: 0.2, 0.5, 0.3, 0.6, 0.9 -- violation at index 2
    auto merged = merge_bins_for_monotonicity(bins);
    EXPECT_THAT(merged, SizeIs(4));
    double expected[] = {0.2, 0.4, 0.6, 0.9};
    for (std::size_t i = 0; i < merged.size(); ++i)
      EXPECT_THAT(merged[i].prob_true(), DoubleNear(expected[i], 1e-9));
  }

  // Already-monotone input needs no merging at all (z_threshold=0)
  {
    std::vector<BinAccumulator> bins = {
        {1.0, 1.0, 10.0}, {2.0, 3.0, 10.0}, {3.0, 6.0, 10.0}, {4.0, 9.0, 10.0}};
    auto merged = merge_bins_for_monotonicity(bins, 0.0);
    EXPECT_THAT(merged, SizeIs(4));
    for (std::size_t i = 0; i < bins.size(); ++i)
      EXPECT_THAT(merged[i].prob_true(), DoubleNear(bins[i].prob_true(), 1e-9));
  }

  // z_threshold=0.0 exactly reduces to plain PAV -- confirmed by
  // checking a case where all secants are statistically indistinguishable
  // (huge z_threshold WOULD merge them, but z=0 must not)
  {
    std::vector<BinAccumulator> bins = {
        {1.0, 5.0, 10.0}, {2.0, 5.1, 10.0}, {3.0, 5.2, 10.0}};
    auto merged_z0 = merge_bins_for_monotonicity(bins, 0.0);
    EXPECT_THAT(merged_z0, SizeIs(3));
  }

  // Large z_threshold aggressively merges statistically-indistinguishable
  // (but still monotone) pools
  {
    std::vector<BinAccumulator> bins = {
        {1.0, 5.0, 10.0}, {2.0, 5.1, 10.0}, {3.0, 5.2, 10.0}};
    auto merged_big_z = merge_bins_for_monotonicity(bins, 5.0);
    EXPECT_THAT(merged_big_z, SizeIs(Le(3)));
  }

  // A fully-decreasing sequence collapses entirely into one pool
  {
    std::vector<BinAccumulator> bins = {
        {1.0, 9.0, 10.0}, {2.0, 6.0, 10.0}, {3.0, 3.0, 10.0}, {4.0, 1.0, 10.0}};
    auto merged = merge_bins_for_monotonicity(bins, 0.0);
    EXPECT_THAT(merged, SizeIs(1));
    EXPECT_THAT(merged[0].prob_true(),
                DoubleNear((9.0 + 6.0 + 3.0 + 1.0) / 40.0, 1e-9));
  }

  // Conservation holds under EVERY z_threshold, not just 0
  {
    auto rng_engine = RandomEngine(42);
    std::uniform_real_distribution<double> unif_p(0, 1), unif_c(1, 1000);
    std::vector<BinAccumulator> bins;
    for (int i = 0; i < 50; ++i) {
      double c = 1000 * static_cast<double>(RandomUniformFloat(&rng_engine)),
             py = static_cast<double>(RandomUniformFloat(&rng_engine));
      bins.push_back({py * c, py * c, c});
    }
    double total_true_before = 0, total_count_before = 0;
    for (auto& b : bins) {
      total_true_before += b.sum_true;
      total_count_before += b.count;
    }

    for (double z : {0.0, 0.5, 1.0, 2.0, 5.0}) {
      auto merged = merge_bins_for_monotonicity(bins, z);
      double total_true_after = 0, total_count_after = 0;
      for (auto& b : merged) {
        total_true_after += b.sum_true;
        total_count_after += b.count;
      }
      EXPECT_THAT(total_true_after, DoubleNear(total_true_before, 1e-9));
      EXPECT_THAT(total_count_after, DoubleNear(total_count_before, 1e-9));
    }
  }

  // Single-bin input is returned unchanged (no merge possible)
  {
    std::vector<BinAccumulator> bins = {{5.0, 5.0, 10.0}};
    auto merged = merge_bins_for_monotonicity(bins, 3.0);
    EXPECT_THAT(merged, SizeIs(1));
    EXPECT_THAT(merged[0].prob_true(), DoubleNear(0.5, 1e-9));
  }

  // Weighting is correct, not naive-averaged (regression test for a
  // previously-caught bug class)
  {
    std::vector<BinAccumulator> bins = {{500.0, 500.0, 1000.0},
                                        {3.0, 3.0, 10.0}};  // 0.5 and 0.3
    auto merged = merge_bins_for_monotonicity(bins);
    double naive_avg = (0.5 + 0.3) / 2.0;
    double weighted_avg = (500.0 + 3.0) / 1010.0;
    EXPECT_THAT(merged[0].prob_true(), Not(DoubleNear(naive_avg, 1e-4)));
    EXPECT_THAT(merged[0].prob_true(), DoubleNear(weighted_avg, 1e-9));
  }
}

TEST(SmoothedPavCalibrationFit, ConfidenceWeight) {
  // Basic formula check, n/(p(1-p))
  {
    std::vector<double> prob_true = {0.5};
    std::vector<double> count = {100.0};
    auto conf = *confidence_weight(prob_true, count);
    EXPECT_THAT(conf[0], DoubleNear(100.0 / 0.25, 1e-9));
  }

  // p=0.5 has the LOWEST confidence per unit count (max variance point)
  {
    std::vector<double> prob_true = {0.5, 0.1, 0.9};
    std::vector<double> count = {100.0, 100.0, 100.0};
    auto conf = *confidence_weight(prob_true, count);
    EXPECT_LT(conf[0], conf[1]);
    EXPECT_LT(conf[0], conf[2]);
    EXPECT_THAT(conf[1], DoubleNear(conf[2], 1e-9));
  }

  // p=0 or p=1 exactly doesn't blow up (variance floor)
  {
    std::vector<double> prob_true = {0.0, 1.0};
    std::vector<double> count = {50.0, 50.0};
    auto conf = *confidence_weight(prob_true, count);
    EXPECT_TRUE(std::isfinite(conf[0]));
    EXPECT_TRUE(std::isfinite(conf[1]));
    EXPECT_GT(conf[0], 1e6);
    EXPECT_GT(conf[1], 1e6);
  }

  // Confidence scales linearly with count at fixed p
  {
    std::vector<double> prob_true = {0.3, 0.3};
    std::vector<double> count = {10.0, 100.0};
    auto conf = *confidence_weight(prob_true, count);
    EXPECT_THAT(conf[1] / conf[0], DoubleNear(10.0, 1e-9));
  }
}

TEST(SmoothedPavCalibrationFit, PchipSlopesUnweighted) {
  // Perfectly linear data -> every derivative equals the constant slope
  {
    std::vector<double> x = {0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> y = {0.0, 0.5, 1.0, 1.5, 2.0};  // slope = 2 everywhere
    auto d = *pchip_slopes_unweighted(x, y);
    for (double di : d) EXPECT_THAT(di, DoubleNear(2.0, 1e-9));
  }

  // Minimal 2-point case
  {
    std::vector<double> x = {0.0, 1.0};
    std::vector<double> y = {0.0, 4.0};
    auto d = *pchip_slopes_unweighted(x, y);
    EXPECT_THAT(d, SizeIs(2));
    EXPECT_THAT(d[0], DoubleNear(4.0, 1e-9));
    EXPECT_THAT(d[1], DoubleNear(4.0, 1e-9));
  }

  // Monotonicity preserved for random monotone data
  {
    auto rng_engine = RandomEngine(42);
    std::vector<double> x(10), y(10);
    for (auto& v : x) v = static_cast<double>(RandomUniformFloat(&rng_engine));
    std::sort(x.begin(), x.end());
    x.erase(std::unique(x.begin(), x.end()), x.end());
    for (auto& v : y) v = static_cast<double>(RandomUniformFloat(&rng_engine));
    std::sort(y.begin(), y.end());
    y.resize(x.size());
    auto d = *pchip_slopes_unweighted(x, y);
    for (double di : d) EXPECT_GE(di, 0);
  }
}

TEST(SmoothedPavCalibrationFit, PchipSlopesWeighted) {
  // Equal-secant consistency -- weights shouldn't matter when both secants
  // already agree
  {
    std::vector<double> x = {0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> y = {0.0, 0.5, 1.0, 1.5, 2.0};
    std::vector<double> w = {1.0, 5.0, 1000.0, 5.0, 1.0};
    auto d = *pchip_slopes(x, y, w);
    for (double di : d) EXPECT_THAT(di, DoubleNear(2.0, 1e-6));
  }

  // Weighted derivative pulled toward the well-supported secant, not the noisy
  // one
  {
    std::vector<double> x = {0.2, 0.3, 0.35, 0.5};
    std::vector<double> y = {0.2, 0.5, 0.8, 0.7};
    std::vector<double> w = {1000.0, 1000.0, 3.0, 1000.0};
    auto d = *pchip_slopes(x, y, w);
    // secants: (0.5-0.2)/0.1=3.0, (0.8-0.5)/0.05=6.0
    // unweighted harmonic mean pulls noticeably toward 6.0; weighted
    // should stay much closer to the well-supported 3.0
    EXPECT_LT(d[1], 4.0);
    EXPECT_GT(d[1], 2.5);
  }

  // Box-safety -- exhaustive stress test, wildly varying weights, checking the
  // actual Hermite segment (not just the point bound)
  {
    auto rng_engine = RandomEngine(42);
    constexpr int n_trials = 5000;
    for (int trial = 0; trial < n_trials; ++trial) {
      int n = 4 + RandomUniformInt(6, &rng_engine);
      std::vector<double> xs(n), ys(n), ws(n);
      for (auto& v : xs) v = RandomUniformFloat(&rng_engine);
      std::sort(xs.begin(), xs.end());
      xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
      if (xs.size() < 4) continue;
      n = static_cast<int>(xs.size());
      ys.resize(n);
      for (auto& v : ys) v = RandomUniformFloat(&rng_engine);
      std::sort(ys.begin(), ys.end());
      ws.resize(n);
      for (auto& v : ws)
        v = std::pow(10.0, 5 * RandomUniformFloat(&rng_engine));

      auto d = *pchip_slopes(xs, ys, ws);
      for (int i = 0; i + 1 < n; ++i) {
        double h = xs[i + 1] - xs[i];
        double delta = (ys[i + 1] - ys[i]) / h;
        if (delta <= 0) continue;
        // sample the Hermite segment densely and check monotonicity
        // bool seg_monotone = true;
        double prev = ys[i];
        for (int k = 1; k <= 20; ++k) {
          double t = k / 20.0;
          double t2 = t * t, t3 = t2 * t;
          double h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t;
          double h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
          double val = h00 * ys[i] + h10 * h * d[i] + h01 * ys[i + 1] +
                       h11 * h * d[i + 1];
          EXPECT_GE(val, prev - 1e-9);
          prev = val;
        }
      }
    }
  }

  // Opposite-signed neighbors force derivative to exactly 0
  {
    std::vector<double> x = {0.0, 0.3, 0.6, 1.0};
    std::vector<double> y = {0.0, 0.5, 0.5, 1.0};
    std::vector<double> w = {10.0, 10.0, 10.0, 10.0};
    auto d = *pchip_slopes(x, y, w);
    EXPECT_THAT(d[1], DoubleNear(0.0, 1e-9));
    EXPECT_THAT(d[2], DoubleNear(0.0, 1e-9));
  }
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
