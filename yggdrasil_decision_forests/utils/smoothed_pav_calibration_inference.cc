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
#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "yggdrasil_decision_forests/utils/smoothed_pav_calibration_fit.h"

namespace yggdrasil_decision_forests::utils {

double internal::pchip_eval_single(const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   const std::vector<double>& d, double xq) {
  xq = std::clamp(xq, x.front(), x.back());

  // std::upper_bound: first element > xq. The segment is
  // [idx-1, idx]. Clamp so the last point maps into the final segment.
  auto it = std::upper_bound(x.begin(), x.end(), xq);
  std::size_t idx = static_cast<std::size_t>(std::max(
      std::ptrdiff_t{1},
      std::min(it - x.begin(), static_cast<std::ptrdiff_t>(x.size()) - 1)));

  // This is a straightforward implementation of the equation here:
  // https://en.wikipedia.org/wiki/Monotone_cubic_interpolation#Cubic_interpolation
  const double x0 = x[idx - 1], x1 = x[idx];
  const double y0 = y[idx - 1], y1 = y[idx];
  const double d0 = d[idx - 1], d1 = d[idx];
  const double h = x1 - x0;
  const double t = (xq - x0) / h;
  const double t2 = t * t, t3 = t2 * t;

  // Cubic Hermite spline basis functions for t in [0,1]. See here:
  // https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Representations
  const double h00 = 2 * t3 - 3 * t2 + 1;
  const double h10 = t3 - 2 * t2 + t;
  const double h01 = -2 * t3 + 3 * t2;
  const double h11 = t3 - t2;

  return h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1;
}

double CalibrationLookupTable::apply(double p) const {
  const double pc = std::clamp(p, 0.0, 1.0);
  const double pos = pc * inv_step_;  // fractional grid index
  double integral_pos;
  const auto frac_pos = std::modf(pos, &integral_pos);
  const auto i0 = static_cast<std::size_t>(integral_pos);
  const std::size_t i1 = std::min(i0 + 1, delta_.size() - 1);

  // Linear Interpolation (NOTE: using std::lerp actually increases latency).
  const double delta = delta_[i0] + frac_pos * (delta_[i1] - delta_[i0]);
  // TODO: Change the table lookup to return h(p) directly.
  return std::clamp(pc + delta, 0.0, 1.0);
}

void CalibrationLookupTable::apply_batch(const std::vector<double>& p,
                                         std::vector<double>& out) const {
  DCHECK_EQ(p.size(), out.size());
  for (std::size_t i = 0; i < p.size(); ++i) {
    out[i] = apply(p[i]);
  }
}

CalibrationLookupTable CalibrationLookupTable::Create(
    const FittedCalibrationCurve& curve, std::size_t n_grid) {
  std::vector<double> delta(n_grid);
  const double step = 1.0 / static_cast<double>(n_grid - 1);
  for (std::size_t i = 0; i < n_grid; ++i) {
    const double p = static_cast<double>(i) * step;
    const double h = internal::pchip_eval_single(curve.x, curve.y, curve.d, p);
    delta[i] = h - p;
  }
  return CalibrationLookupTable(std::move(delta));
}

CalibrationLookupTable CalibrationLookupTable::Create(
    const FittedCalibrationCurve& curve,
    const FittedCalibrationCurve& reference, std::size_t n_grid) {
  std::vector<double> delta(n_grid);
  const double step = 1.0 / static_cast<double>(n_grid - 1);
  for (std::size_t i = 0; i < n_grid; ++i) {
    const double p = static_cast<double>(i) * step;
    const double h = internal::pchip_eval_single(curve.x, curve.y, curve.d, p);
    const double h_ref =
        internal::pchip_eval_single(reference.x, reference.y, reference.d, p);
    delta[i] = h - h_ref;
  }
  return CalibrationLookupTable(std::move(delta));
}

DistributionMatchingTables DistributionMatchingTables::Create(
    const DistributionMatchingCurves& curves, std::size_t n_grid) {
  return DistributionMatchingTables(
      CalibrationLookupTable::Create(curves.source_distribution, n_grid),
      CalibrationLookupTable::Create(curves.target_quantile, n_grid));
}

}  // namespace yggdrasil_decision_forests::utils
