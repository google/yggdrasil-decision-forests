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
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace yggdrasil_decision_forests::utils {

absl::Status accumulate_bins(std::vector<BinAccumulator>& bins,
                             const std::vector<double>& p,
                             const std::vector<double>& y, std::size_t n_bins) {
  if (bins.size() != n_bins) {
    return absl::InvalidArgumentError(
        absl::StrCat("bins.size() = ", bins.size(), " != n_bins = ", n_bins));
  }
  if (y.size() != p.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("y.size() = ", y.size(), " != p.size() = ", p.size()));
  }

  const double scale = static_cast<double>(n_bins);

  for (std::size_t i = 0; i < p.size(); ++i) {
    double pi = std::clamp(p[i], 0.0, 1.0);
    // matches np.digitize(p, linspace(0,1,n_bins+1)) - 1, clipped
    auto idx = static_cast<std::size_t>(std::min(pi * scale, scale - 1.0));
    bins[idx].sum_pred += p[i];
    bins[idx].sum_true += y[i];
    bins[idx].count += 1.0;
  }
  return absl::OkStatus();
}

std::vector<BinAccumulator> finalize_bins(std::vector<BinAccumulator> bins) {
  bins.erase(
      std::remove_if(bins.begin(), bins.end(),
                     [](const BinAccumulator& b) { return b.count == 0.0; }),
      bins.end());
  return bins;
}

absl::StatusOr<std::vector<BinAccumulator>> aggregate_bins(
    const std::vector<double>& p, const std::vector<double>& y,
    std::size_t n_bins) {
  if (p.size() != y.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("p.size() = ", p.size(), " != y.size() = ", y.size()));
  }

  std::vector<BinAccumulator> bins(n_bins);
  const double scale = static_cast<double>(n_bins);

  for (std::size_t i = 0; i < p.size(); ++i) {
    double pi = std::clamp(p[i], 0.0, 1.0);
    // matches np.digitize(p, linspace(0,1,n_bins+1)) - 1, clipped
    auto idx = static_cast<std::size_t>(std::min(pi * scale, scale - 1.0));
    bins[idx].sum_pred += p[i];
    bins[idx].sum_true += y[i];
    bins[idx].count += 1.0;
  }

  return finalize_bins(std::move(bins));
}

absl::Status accumulate_bins(std::vector<BinAccumulator>& bins,
                             const std::vector<double>& p, std::size_t n_bins) {
  if (bins.size() != n_bins) {
    return absl::InvalidArgumentError(
        absl::StrCat("bins.size() = ", bins.size(), " != n_bins = ", n_bins));
  }

  const double scale = static_cast<double>(n_bins);
  for (std::size_t i = 0; i < p.size(); ++i) {
    double pi = std::clamp(p[i], 0.0, 1.0);
    auto idx = static_cast<std::size_t>(std::min(pi * scale, scale - 1.0));
    bins[idx].sum_pred += p[i];
    bins[idx].count += 1.0;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<BinAccumulator>> aggregate_bins(
    const std::vector<double>& p, std::size_t n_bins) {
  std::vector<BinAccumulator> bins(n_bins);
  auto status = accumulate_bins(bins, p, n_bins);
  if (!status.ok()) {
    return status;
  }
  return finalize_bins(std::move(bins));
}

std::vector<BinAccumulator> merge_bins_for_monotonicity(
    const std::vector<BinAccumulator>& bins, double z_threshold) {
  std::vector<BinAccumulator> stack;
  stack.reserve(bins.size());

  for (const auto& bin : bins) {
    stack.push_back(bin);
    while (stack.size() >= 2) {
      const auto& b2 = stack[stack.size() - 1];
      const auto& b1 = stack[stack.size() - 2];
      const double p1 = b1.prob_true();
      const double p2 = b2.prob_true();
      const bool violates = p1 > p2;

      bool merge_needed;
      if (!violates) {
        const double n1 = b1.count, n2 = b2.count;
        const double p_pool = (b1.sum_true + b2.sum_true) / (n1 + n2);
        const double se = std::sqrt(std::max(p_pool * (1.0 - p_pool), 1e-12) *
                                    (1.0 / n1 + 1.0 / n2));
        const double z = se > 0.0 ? (p2 - p1) / se : 0.0;
        merge_needed = std::abs(z) < z_threshold;
      } else {
        merge_needed = true;
      }

      if (!merge_needed) break;
      BinAccumulator last = stack.back();
      stack.pop_back();
      stack.back() += last;
    }
  }
  return stack;
}

MonotoneCurvePoints anchor_endpoints(const std::vector<double>& x,
                                     const std::vector<double>& y) {
  std::vector<double> xa, ya;
  xa.reserve(x.size() + 2);
  ya.reserve(y.size() + 2);
  xa.push_back(0.0);
  ya.push_back(0.0);
  xa.insert(xa.end(), x.begin(), x.end());
  ya.insert(ya.end(), y.begin(), y.end());
  xa.push_back(1.0);
  ya.push_back(1.0);
  return {std::move(xa), std::move(ya)};
}

absl::StatusOr<std::vector<double>> confidence_weight(
    const std::vector<double>& prob_true, const std::vector<double>& count,
    double eps) {
  if (prob_true.size() != count.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("prob_true.size() = ", prob_true.size(),
                     " != count.size() = ", count.size()));
  }

  std::vector<double> conf(prob_true.size());
  for (std::size_t i = 0; i < prob_true.size(); ++i) {
    const double variance = std::max(prob_true[i] * (1.0 - prob_true[i]), eps);
    conf[i] = count[i] / variance;
  }
  return conf;
}

absl::StatusOr<std::vector<double>> pchip_slopes_unweighted(
    const std::vector<double>& x, const std::vector<double>& y) {
  const std::size_t n = x.size();
  if (n < 2) {
    return absl::InvalidArgumentError(absl::StrCat("n = ", n, " < 2"));
  }
  if (x.size() != y.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("x.size() = ", x.size(), " != y.size() = ", y.size()));
  }

  std::vector<double> h(n - 1), delta(n - 1);
  for (std::size_t i = 0; i < n - 1; ++i) {
    h[i] = x[i + 1] - x[i];
    if (h[i] <= 0.0) {
      return absl::OutOfRangeError(
          absl::StrCat("h[", i, "] = ", h[i], " <= 0.0"));
    }
    delta[i] = (y[i + 1] - y[i]) / h[i];
  }

  std::vector<double> d(n, 0.0);
  for (std::size_t i = 1; i + 1 < n; ++i) {
    const double d0 = delta[i - 1];
    const double d1 = delta[i];
    if (d0 * d1 > 0.0) {
      const double w1 = 2.0 * h[i] + h[i - 1];
      const double w2 = h[i] + 2.0 * h[i - 1];
      d[i] = (w1 + w2) / (w1 / d0 + w2 / d1);
    }
  }

  auto edge_slope = [](double h0, double h1, double d0, double d1) -> double {
    double d_edge = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if (std::signbit(d_edge) != std::signbit(d0)) {
      d_edge = 0.0;
    } else if ((std::signbit(d0) != std::signbit(d1)) &&
               (std::abs(d_edge) > 3.0 * std::abs(d0))) {
      // Fritsch–Carlson's sufficient (not necessary) condition guarantees a
      // monotone cubic Hermite segment whenever both endpoint
      // derivative-to-secant ratios, d_i/delta and d_{i+1}/delta, lie in [0,
      // 3]. The constant 3 is the exact boundary of that box region: with one
      // ratio held at 0, it's the largest value the other can take before the
      // segment's derivative touches zero and monotonicity is no longer
      // guaranteed.
      d_edge = 3.0 * d0;
    }
    return d_edge;
  };

  if (n > 2) {
    d[0] = edge_slope(h[0], h[1], delta[0], delta[1]);
    d[n - 1] = edge_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
  } else {
    d[0] = delta[0];
    d[1] = delta[0];
  }
  return d;
}

absl::StatusOr<std::vector<double>> pchip_slopes(
    const std::vector<double>& x, const std::vector<double>& y,
    const std::vector<double>& weight) {
  const std::size_t n = x.size();
  if (n < 2) {
    return absl::InvalidArgumentError(absl::StrCat("n = ", n, " < 2"));
  }
  if (x.size() != y.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("x.size() = ", x.size(), " != y.size() = ", y.size()));
  }
  if (x.size() != weight.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "x.size() = ", x.size(), " != weight.size() = ", weight.size()));
  }

  std::vector<double> h(n - 1), delta(n - 1);
  for (std::size_t i = 0; i < n - 1; ++i) {
    h[i] = x[i + 1] - x[i];
    if (h[i] <= 0.0) {
      return absl::OutOfRangeError(
          absl::StrCat("h[", i, "] = ", h[i], " <= 0.0"));
    }
    delta[i] = (y[i + 1] - y[i]) / h[i];
  }

  std::vector<double> d(n, 0.0);
  for (std::size_t i = 1; i + 1 < n; ++i) {
    const double d0 = delta[i - 1];
    const double d1 = delta[i];
    if (d0 * d1 > 0.0) {
      const double conf0 =
          2.0 /
          (1.0 / weight[i - 1] + 1.0 / weight[i]);  // secant (i-1,i) confidence
      const double conf1 =
          2.0 /
          (1.0 / weight[i] + 1.0 / weight[i + 1]);  // secant (i,i+1) confidence
      const double w1 = (2.0 * h[i] + h[i - 1]) * conf0;
      const double w2 = (h[i] + 2.0 * h[i - 1]) * conf1;
      const double harmonic = (w1 + w2) / (w1 / d0 + w2 / d1);
      // Fritsch–Carlson's sufficient (not necessary) condition guarantees a
      // monotone cubic Hermite segment whenever both endpoint
      // derivative-to-secant ratios, d_i/delta and d_{i+1}/delta, lie in
      // [0, 3]. The constant 3 is the exact boundary of that box region: with
      // one ratio held at 0, it's the largest value the other can take before
      // the segment's derivative touches zero and monotonicity is no longer
      // guaranteed.
      const double safe_cap = 3.0 * std::min(d0, d1);
      d[i] = std::min(harmonic, safe_cap);
    }
    // else: opposite signs, or a flat neighbor -> d[i] stays 0,
    // forcing a horizontal segment through any tied/flat region.
  }

  auto edge_slope = [](double h0, double h1, double d0, double d1) -> double {
    double d_edge = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if (std::signbit(d_edge) != std::signbit(d0)) {
      d_edge = 0.0;
    } else if ((std::signbit(d0) != std::signbit(d1)) &&
               (std::abs(d_edge) > 3.0 * std::abs(d0))) {
      // Fritsch–Carlson's sufficient (not necessary) condition guarantees a
      // monotone cubic Hermite segment whenever both endpoint
      // derivative-to-secant ratios, d_i/delta and d_{i+1}/delta, lie in [0,
      // 3]. The constant 3 is the exact boundary of that box region: with one
      // ratio held at 0, it's the largest value the other can take before the
      // segment's derivative touches zero and monotonicity is no longer
      // guaranteed.
      d_edge = 3.0 * d0;
    }
    return d_edge;
  };

  if (n > 2) {
    d[0] = edge_slope(h[0], h[1], delta[0], delta[1]);
    d[n - 1] = edge_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
  } else {
    d[0] = delta[0];
    d[1] = delta[0];
  }

  return d;
}

absl::StatusOr<FittedCalibrationCurve> fit_monotone_curve(
    std::vector<double> x, std::vector<double> y, std::vector<double> w) {
  auto conf_or = confidence_weight(y, w);
  if (!conf_or.ok()) {
    return conf_or.status();
  }
  auto slopes_or = pchip_slopes(x, y, *conf_or);
  if (!slopes_or.ok()) {
    return slopes_or.status();
  }

  return FittedCalibrationCurve{std::move(x), std::move(y),
                                std::move(*slopes_or)};
}

absl::StatusOr<FittedCalibrationCurve> fit_calibration(
    const std::vector<BinAccumulator>& bins, double z_threshold) {
  auto finalized = finalize_bins(
      std::vector<BinAccumulator>(bins));  // copy; bins left untouched
  auto pools = merge_bins_for_monotonicity(finalized, z_threshold);

  std::vector<double> x, y, w;
  x.reserve(pools.size());
  y.reserve(pools.size());
  w.reserve(pools.size());
  for (const auto& pool : pools) {
    x.push_back(pool.prob_pred());
    y.push_back(pool.prob_true());
    w.push_back(pool.count);
  }

  return fit_monotone_curve(std::move(x), std::move(y), std::move(w));
}

absl::StatusOr<FittedCalibrationCurve> fit_calibration(
    const std::vector<double>& p, const std::vector<double>& y_raw,
    std::size_t n_bins, double z_threshold) {
  auto bins_or = aggregate_bins(p, y_raw, n_bins);
  if (!bins_or.ok()) {
    return bins_or.status();
  }
  return fit_calibration(*bins_or, z_threshold);
}

void cumulative_from_bins(const std::vector<BinAccumulator>& bins,
                          std::vector<double>& x, std::vector<double>& y,
                          std::vector<double>& w) {
  double total = 0.0;
  for (const auto& b : bins) {
    total += b.count;
  }

  x.clear();
  y.clear();
  w.clear();
  x.reserve(bins.size());
  y.reserve(bins.size());
  w.reserve(bins.size());

  double cumulative_before = 0.0;
  for (const auto& b : bins) {
    x.push_back(b.prob_pred());
    y.push_back(total > 0.0 ? (cumulative_before + 0.5 * b.count) / total
                            : 0.0);
    w.push_back(b.count);
    cumulative_before += b.count;
  }

  if (x.empty()) {
    x.push_back(0.0);
    y.push_back(0.0);
    w.push_back(0.0);
    x.push_back(1.0);
    y.push_back(1.0);
    w.push_back(0.0);
  }

  // Ensure endpoints are exactly (0,0) and (1,1).
  double x_new, y_new;
  if (x[0] != 0.0 || y[0] != 0.0) {
    if (x[0] == 0.0) {
      if (x.size() > 1)
        x_new = x[1] / 2.0;
      else
        x_new = std::numeric_limits<double>::epsilon();
    } else {
      x_new = x[0];
    }
    if (y[0] == 0.0) {
      if (y.size() > 1)
        y_new = y[1] / 2.0;
      else
        y_new = std::numeric_limits<double>::epsilon();
    } else {
      y_new = y[0];
    }
    x[0] = x_new;
    y[0] = y_new;
    x.insert(x.begin(), 0.0);
    y.insert(y.begin(), 0.0);
    w.insert(w.begin(), w[0]);
  }
  if (x.back() != 1.0 || y.back() != 1.0) {
    if (x.back() == 1.0) {
      if (x.size() > 1)
        x_new = (1.0 + x[x.size() - 2]) / 2.0;
      else
        x_new = 1.0 - std::numeric_limits<double>::epsilon();
    } else {
      x_new = x.back();
    }
    if (y.back() == 1.0) {
      if (y.size() > 1)
        y_new = (1.0 + y[y.size() - 2]) / 2.0;
      else
        y_new = 1.0 - std::numeric_limits<double>::epsilon();
    } else {
      y_new = y.back();
    }
    x.back() = x_new;
    y.back() = y_new;
    x.push_back(1.0);
    y.push_back(1.0);
    w.push_back(w.back());
  }
}

absl::StatusOr<FittedCalibrationCurve> fit_score_distribution(
    const std::vector<BinAccumulator>& bins) {
  auto finalized = finalize_bins(std::vector<BinAccumulator>(bins));
  std::vector<double> x, y, w;
  cumulative_from_bins(finalized, x, y, w);
  return fit_monotone_curve(std::move(x), std::move(y), std::move(w));
}

absl::StatusOr<FittedCalibrationCurve> fit_score_distribution(
    const std::vector<double>& scores, std::size_t n_bins) {
  auto bins_or = aggregate_bins(scores, n_bins);
  if (!bins_or.ok()) {
    return bins_or.status();
  }
  return fit_score_distribution(*bins_or);
}

absl::StatusOr<FittedCalibrationCurve> fit_score_quantile_function(
    const std::vector<BinAccumulator>& bins) {
  auto finalized = finalize_bins(std::vector<BinAccumulator>(bins));
  std::vector<double> x, y, w;
  cumulative_from_bins(finalized, x, y, w);
  // quantile function: input=cumulative fraction, output=score
  std::swap(x, y);
  return fit_monotone_curve(std::move(x), std::move(y), std::move(w));
}

absl::StatusOr<FittedCalibrationCurve> fit_score_quantile_function(
    const std::vector<double>& scores, std::size_t n_bins) {
  auto bins_or = aggregate_bins(scores, n_bins);
  if (!bins_or.ok()) {
    return bins_or.status();
  }
  return fit_score_quantile_function(*bins_or);
}

absl::StatusOr<DistributionMatchingCurves> fit_distribution_matching(
    const std::vector<BinAccumulator>& source_bins,
    const std::vector<BinAccumulator>& target_bins) {
  auto source_distribution_or = fit_score_distribution(source_bins);
  if (!source_distribution_or.ok()) {
    return source_distribution_or.status();
  }
  auto target_quantile_or = fit_score_quantile_function(target_bins);
  if (!target_quantile_or.ok()) {
    return target_quantile_or.status();
  }
  return DistributionMatchingCurves{*source_distribution_or,
                                    *target_quantile_or};
}

absl::StatusOr<DistributionMatchingCurves> fit_distribution_matching(
    const std::vector<double>& source_scores,
    const std::vector<double>& target_scores, std::size_t n_bins) {
  auto source_bins_or = aggregate_bins(source_scores, n_bins);
  if (!source_bins_or.ok()) {
    return source_bins_or.status();
  }
  auto target_bins_or = aggregate_bins(target_scores, n_bins);
  if (!target_bins_or.ok()) {
    return target_bins_or.status();
  }
  return fit_distribution_matching(*source_bins_or, *target_bins_or);
}

}  // namespace yggdrasil_decision_forests::utils
