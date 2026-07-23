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

#include "yggdrasil_decision_forests/utils/smoothed_pav_calibration_inference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/smoothed_pav_calibration_fit.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

using ::testing::DoubleNear;
using ::testing::SizeIs;

auto sigmoid = [](double z) {
  return 1.0 / (1.0 + std::exp(-std::clamp(z, -50.0, 50.0)));
};

double quantile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(q * (v.size() - 1))];
}

TEST(SmoothedPavCalibrationInference, PchipEvalSingle) {
  std::vector<double> x = {0.0, 0.3, 0.6, 1.0};
  std::vector<double> y = {0.0, 0.4, 0.7, 1.0};
  std::vector<double> w = {10.0, 10.0, 10.0, 10.0};
  auto d_or = pchip_slopes(x, y, w);
  EXPECT_THAT(d_or, absl_testing::IsOk());

  // Exact interpolation at every knot
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_THAT(internal::pchip_eval_single(x, y, *d_or, x[i]),
                DoubleNear(y[i], 1e-12));

  // Clamping below the domain
  EXPECT_THAT(internal::pchip_eval_single(x, y, *d_or, -5.0),
              DoubleNear(y.front(), 1e-9));
  EXPECT_THAT(internal::pchip_eval_single(x, y, *d_or, -1e300),
              DoubleNear(y.front(), 1e-9));

  // A3: clamping above the domain
  EXPECT_THAT(internal::pchip_eval_single(x, y, *d_or, 5.0),
              DoubleNear(y.back(), 1e-9));
  EXPECT_THAT(internal::pchip_eval_single(x, y, *d_or, 1e300),
              DoubleNear(y.back(), 1e-9));

  // Midpoint of a linear segment interpolates linearly
  {
    std::vector<double> xl = {0.0, 1.0};
    std::vector<double> yl = {0.0, 2.0};
    std::vector<double> wl = {1.0, 1.0};
    auto dl_or = pchip_slopes(xl, yl, wl);
    EXPECT_THAT(dl_or, absl_testing::IsOk());
    EXPECT_THAT(internal::pchip_eval_single(xl, yl, *dl_or, 0.5),
                DoubleNear(1.0, 1e-9));
  }

  // Monotone across a fine sweep, matching the underlying knot monotonicity
  {
    double prev = -1.0;
    for (int i = 0; i <= 1000; ++i) {
      double q = static_cast<double>(i) / 1000.0;
      double v = internal::pchip_eval_single(x, y, *d_or, q);
      EXPECT_GE(v, prev - 1e-12);
      prev = v;
    }
  }

  // Query exactly at a segment boundary (not just interior) is handled cleanly
  EXPECT_THAT(internal::pchip_eval_single(x, y, *d_or, 0.3),
              DoubleNear(0.4, 1e-9));
}

TEST(SmoothedPavCalibrationInference, LookupTableBasic) {
  // Identity table: delta(p) = 0 everywhere -> apply() should be a no-op
  std::vector<double> identity_delta(101, 0.0);
  CalibrationLookupTable identity_table(identity_delta);

  // Identity table returns its input unchanged
  for (double p : {0.0, 0.1, 0.5, 0.9, 1.0})
    EXPECT_THAT(identity_table.apply(p), DoubleNear(p, 1e-9));

  // grid_size() reports correctly
  EXPECT_EQ(identity_table.grid_size(), 101);

  // B3: raw_grid() exposes the underlying delta array unchanged
  auto grid = identity_table.raw_grid();
  EXPECT_THAT(grid, SizeIs(101));
  for (double v : grid) EXPECT_EQ(v, 0.0);

  // B4: a genuinely non-trivial table (constant positive shift, clamped)
  std::vector<double> shift_delta(101, 0.1);
  CalibrationLookupTable shift_table(shift_delta);
  EXPECT_THAT(shift_table.apply(0.5), DoubleNear(0.6, 1e-9));
  EXPECT_THAT(shift_table.apply(0.95), DoubleNear(1.0, 1e-9));
  EXPECT_LE(shift_table.apply(0.95), 1.0);

  // B5: out-of-range INPUT (not just near-boundary) clamps correctly
  EXPECT_THAT(shift_table.apply(-10.0),
              DoubleNear(shift_table.apply(0.0), 1e-9));
  EXPECT_THAT(shift_table.apply(10.0),
              DoubleNear(shift_table.apply(1.0), 1e-9));
}

TEST(SmoothedPavCalibrationInference, ApplyConsistency) {
  auto rng_engine = RandomEngine(42);
  std::vector<double> delta(500);
  for (std::size_t i = 0; i < delta.size(); ++i) {
    double p = static_cast<double>(i) / (delta.size() - 1);
    delta[i] = 0.05 * std::sin(6.0 * p);  // some arbitrary smooth wiggle
  }
  CalibrationLookupTable table(delta);

  std::vector<double> queries(2000);
  // Includes some out-of-range values
  for (auto& q : queries)
    q = 1.4 * static_cast<double>(RandomUniformFloat(&rng_engine)) - 0.2;

  std::vector<double> batch_out(queries.size());
  table.apply_batch(queries, batch_out);

  for (std::size_t i = 0; i < queries.size(); ++i)
    EXPECT_EQ(table.apply(queries[i]), batch_out[i]);
}

TEST(SmoothedPavCalibrationInference, BuildLookupTable) {
  // Build a small hand-designed monotone curve to construct a table from
  std::vector<double> x = {0.0, 0.2, 0.5, 0.8, 1.0};
  std::vector<double> y = {0.0, 0.3, 0.5, 0.7, 1.0};
  std::vector<double> w = {10.0, 10.0, 10.0, 10.0, 10.0};
  auto d_or = pchip_slopes(x, y, w);
  EXPECT_THAT(d_or, absl_testing::IsOk());
  FittedCalibrationCurve curve{x, y, *d_or};

  // Default grid size
  auto table = CalibrationLookupTable::Create(curve);
  EXPECT_EQ(table.grid_size(), 20000);

  // Custom grid size honored
  auto table_small = CalibrationLookupTable::Create(curve, 100);
  EXPECT_EQ(table_small.grid_size(), 100);

  // Table matches direct pchip_eval_single at the same query points (within
  // grid-interpolation tolerance)
  auto rng_engine = RandomEngine(42);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  for (int i = 0; i < 200; ++i) {
    double q = static_cast<double>(RandomUniformFloat(&rng_engine));
    double exact = internal::pchip_eval_single(x, y, *d_or, q);
    double via_table = table.apply(q);
    EXPECT_LE(std::abs(exact - via_table), 1e-3);
  }

  // Endpoints exact
  EXPECT_THAT(table.apply(0.0), DoubleNear(0.0, 1e-6));
  EXPECT_THAT(table.apply(1.0), DoubleNear(1.0, 1e-6));

  // Monotone across the full grid
  auto grid = table.raw_grid();
  for (std::size_t i = 1; i < grid.size(); ++i) {
    double p_prev = static_cast<double>(i - 1) / (grid.size() - 1);
    double p_curr = static_cast<double>(i) / (grid.size() - 1);
    double h_prev = p_prev + grid[i - 1];
    double h_curr = p_curr + grid[i];
    EXPECT_GE(h_curr, h_prev - 1e-9);
  }
}

TEST(SmoothedPavCalibrationInference, RoundTrip) {
  auto rng_engine = RandomEngine(42);
  std::normal_distribution<double> normal(0.0, 2.0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  constexpr int N = 200000;
  std::vector<double> p(N), y(N);
  constexpr double TRUE_T = 3.0;  // raw predictions need cooling by this factor
  for (int i = 0; i < N; ++i) {
    double raw_logit =
        10 * static_cast<double>(RandomUniformFloat(&rng_engine)) - 5.0;
    double true_p = sigmoid(raw_logit / TRUE_T);
    p[i] = sigmoid(raw_logit);
    y[i] = (static_cast<double>(RandomUniformFloat(&rng_engine)) < true_p)
               ? 1.0
               : 0.0;
  }

  auto curve_or = fit_calibration(p, y, 1000, 1.0);
  EXPECT_THAT(curve_or, absl_testing::IsOk());
  auto table = CalibrationLookupTable::Create(*curve_or, 20000);

  // E1: calibrated predictions should be markedly less extreme than raw
  // ones (since the true relationship needed real cooling)
  double raw_extreme_frac = 0.0, cal_extreme_frac = 0.0;
  for (int i = 0; i < N; ++i) {
    double cal = table.apply(p[i]);
    if (p[i] > 0.9 || p[i] < 0.1) raw_extreme_frac += 1.0;
    if (cal > 0.9 || cal < 0.1) cal_extreme_frac += 1.0;
  }
  raw_extreme_frac /= N;
  cal_extreme_frac /= N;
  EXPECT_LT(cal_extreme_frac, raw_extreme_frac);

  // E2: rank order is preserved (no inversions) on a sample -- the whole
  // point of this method vs. e.g. plain isotonic regression
  std::vector<std::size_t> idx(2000);
  for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(),
            [&](std::size_t a, std::size_t b) { return p[a] < p[b]; });
  for (std::size_t i = 1; i < idx.size(); ++i)
    EXPECT_GE(table.apply(p[idx[i]]), table.apply(p[idx[i - 1]]) - 1e-12);
}

TEST(SmoothedPavCalibrationInference, DistributionMatchingTables) {
  auto rng_engine = RandomEngine(42);
  auto make_beta = [](double a, double b) {
    return [a, b](RandomEngine* r) mutable {
      double x = RandomGammaFloat(a, 1.0, r), y = RandomGammaFloat(b, 1.0, r);
      return x / (x + y);
    };
  };
  auto dist_target = make_beta(2.0, 8.0);  // currently-deployed model
  auto dist_source = make_beta(8.0, 2.0);  // new model

  constexpr std::size_t N = 300000;
  constexpr std::size_t n_bins = 2000;
  std::vector<double> target_scores(N), source_scores(N);
  for (auto& s : target_scores) s = dist_target(&rng_engine);
  for (auto& s : source_scores) s = dist_source(&rng_engine);

  std::vector<BinAccumulator> source_bins(n_bins), target_bins(n_bins);

  // simulate chunked ingestion, exactly mirroring the calibration API's shape
  constexpr std::size_t chunk_size = 41000;
  for (std::size_t start = 0; start < N; start += chunk_size) {
    std::size_t end = std::min(start + chunk_size, N);
    std::vector<double> s_chunk(source_scores.begin() + start,
                                source_scores.begin() + end);
    std::vector<double> t_chunk(target_scores.begin() + start,
                                target_scores.begin() + end);
    auto status = accumulate_bins(source_bins, s_chunk, n_bins);
    EXPECT_TRUE(status.ok());
    status = accumulate_bins(target_bins, t_chunk, n_bins);
    EXPECT_TRUE(status.ok());
  }

  auto curves_or = fit_distribution_matching(source_bins, target_bins);
  EXPECT_THAT(curves_or, absl_testing::IsOk());
  auto tables = DistributionMatchingTables::Create(*curves_or, 20000);

  std::vector<double> eval_points(source_scores.begin(),
                                  source_scores.begin() + 5000);
  std::vector<double> matched_points(eval_points.size());
  tables.apply_batch(eval_points, matched_points);

  EXPECT_EQ(matched_points.size(), eval_points.size());

  // Distribution matching actually worked (same end-to-end check as before)
  double max_diff = 0.0;
  for (double q : {0.1, 0.25, 0.5, 0.75, 0.9}) {
    double qt = quantile(target_scores, q);
    // map all source scores through the clean API for a fair quantile
    // comparison
    std::vector<double> all_mapped(N);
    tables.apply_batch(source_scores, all_mapped);
    double qm = quantile(all_mapped, q);
    double diff = std::abs(qt - qm);
    max_diff = std::max(max_diff, diff);
    EXPECT_LE(diff, 0.02);
  }

  // Single apply() matches apply_batch() elementwise
  bool single_matches_batch = true;
  for (std::size_t i = 0; i < eval_points.size(); ++i)
    if (tables.apply(eval_points[i]) != matched_points[i])
      single_matches_batch = false;
  EXPECT_TRUE(single_matches_batch);

  auto F_source_manual_or = fit_score_distribution(source_bins);
  EXPECT_THAT(F_source_manual_or, absl_testing::IsOk());
  auto G_target_manual_or = fit_score_quantile_function(target_bins);
  EXPECT_THAT(G_target_manual_or, absl_testing::IsOk());
  auto table_F_manual =
      CalibrationLookupTable::Create(*F_source_manual_or, 20000);
  auto table_G_manual =
      CalibrationLookupTable::Create(*G_target_manual_or, 20000);

  for (double p : eval_points) {
    double manual = table_G_manual.apply(table_F_manual.apply(p));
    double clean = tables.apply(p);
    EXPECT_EQ(manual, clean);
  }

  double source_total = 0, target_total = 0;
  for (auto& b : source_bins) source_total += b.count;
  for (auto& b : target_bins) target_total += b.count;
  EXPECT_EQ(source_total, static_cast<double>(N));
  EXPECT_EQ(target_total, static_cast<double>(N));

  // Accessors on DistributionMatchingTables
  EXPECT_EQ(tables.source_distribution_table().grid_size(), 20000);
  EXPECT_EQ(tables.target_quantile_table().grid_size(), 20000);
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
