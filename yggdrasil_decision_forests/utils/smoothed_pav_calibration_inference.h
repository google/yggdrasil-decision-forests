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

// Inference-side counterpart to smoothed_pav_calibration_fit.h. Takes the
// curves produced by fitting (FittedCalibrationCurve /
// DistributionMatchingCurves) and turns them into fast, allocation-free lookup
// structures. Two independent pieces, sharing the same offline/hot-path design:
//
//   1. CalibrationLookupTable + Create(curve, n_grid)
//      Evaluates a fitted curve once, offline, on a uniform grid of
//      n_grid points via pchip_eval_single (binary search + cubic
//      Hermite basis as knot spacing is generally irregular) and stores
//      delta(p) = h(p) - p. A second Create(curve,
//      reference, n_grid) overload instead stores delta(p) =
//      h(p) - h_ref(p), the difference between two curves; note that
//      CalibrationLookupTable::apply(p) always computes p + delta(p)
//      regardless of which overload built the table, so with the
//      reference overload apply(p) is p + (h(p) - h_ref(p)), not h(p)
//      directly.
//
//   2. DistributionMatchingTables + Create(curves, n_grid)
//      Groups two CalibrationLookupTables (one per curve in a
//      DistributionMatchingCurves) and wraps the nested composition
//      target_quantile_table.apply(source_distribution_table.apply(p))
//      behind the same apply()/apply_batch() interface as
//      CalibrationLookupTable itself. For remapping one model's score
//      distribution onto another's see fit_distribution_matching in
//      smoothed_pav_calibration_fit.h.
//
// In both cases, apply()/apply_batch() are the HOT PATH: because the
// stored grid is uniform, the containing segment is found by direct
// index arithmetic (p * (n_grid-1)) rather than a search. This has O(1), no
// std::upper_bound, no heap allocation, no branching beyond the
// boundary clamp. Tables are immutable after construction and therefore
// safe to share across threads without synchronization. pchip_eval_single
// is the one genuine binary search in this file, and it is only ever
// used offline, during table construction.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_SMOOTHED_PAV_CALIBRATION_INFERENCE_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_SMOOTHED_PAV_CALIBRATION_INFERENCE_H_

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "yggdrasil_decision_forests/utils/smoothed_pav_calibration_fit.h"

namespace yggdrasil_decision_forests::utils {

namespace internal {

constexpr std::size_t kDefaultNGrid = 20'000;

// Exact PCHIP evaluation at one query point via binary search (knot positions
// are generally irregularly spaced). Offline use only -- called during
// CalibrationLookupTable::Create()'s grid construction, never on the hot
// inference path.
// x, y, d: A fitted curve's knot positions, values, and PCHIP derivatives.
// xq:      Query point; clamped into [x.front(), x.back()] if outside.
// Returns: The interpolated value at xq.
double pchip_eval_single(const std::vector<double>& x,
                         const std::vector<double>& y,
                         const std::vector<double>& d, double xq);

}  // namespace internal

// Immutable, thread-safe (once constructed) fast lookup table storing
// delta(p) = h(p) - p (or a variant thereof) on a uniform grid, enabling
// O(1) hot-path evaluation via direct index arithmetic instead of search.
class CalibrationLookupTable {
 public:
  // Builds a lookup table from a fitted curve, evaluating delta(p) = h(p) - p
  // once, offline, on a uniform grid.
  // curve:  The fitted curve to build a table for.
  // n_grid: Number of uniform grid points; more = finer interpolation, at
  //         proportionally more memory and one-time construction cost.
  // Returns: The lookup table; apply(p) returns h(p) directly.
  static CalibrationLookupTable Create(
      const FittedCalibrationCurve& curve,
      std::size_t n_grid = internal::kDefaultNGrid);

  // Builds a lookup table measuring one curve against a REFERENCE curve
  // instead of the identity: delta(p) = h(p) - h_ref(p).
  // IMPORTANT: apply(p) still computes p + delta(p) unchanged, so it returns
  // p + (h(p) - h_ref(p)), NOT h(p) directly, unless h_ref(p) == p. Use
  // the single-curve overload if you want apply() to return h(p) directly.
  // curve:     The curve to measure.
  // reference: The curve to measure against.
  // n_grid:    Number of uniform grid points.
  // Returns:   The lookup table.
  static CalibrationLookupTable Create(
      const FittedCalibrationCurve& curve,
      const FittedCalibrationCurve& reference,
      std::size_t n_grid = internal::kDefaultNGrid);

  // Constructs the table directly from a precomputed delta grid.
  // Prefer Create() over calling this directly.
  // delta_grid: Uniform-grid delta values; must have at least 2 points.
  explicit CalibrationLookupTable(std::vector<double> delta_grid)
      : delta_(std::move(delta_grid)),
        inv_step_(static_cast<double>(delta_.size() - 1)) {
    DCHECK_GE(delta_.size(), 2);
  }

  // Hot path: O(1), no allocation, no search. Returns p + delta(p), clamped to
  // [0,1]. Out-of-range p is clamped before lookup.
  // p: Query point.
  double apply(double p) const;

  // Batch application; equivalent to calling apply() for each element, laid out
  // as a plain loop for straightforward auto-vectorization.
  // p:   Input query points.
  // out: Output buffer, same length as p; must not alias p.
  void apply_batch(const std::vector<double>& p,
                   std::vector<double>& out) const;

  // Number of points in the stored uniform grid.
  std::size_t grid_size() const { return delta_.size(); }

  // Direct access to the underlying stored delta grid.
  const std::vector<double>& raw_grid() const { return delta_; }

 private:
  std::vector<double> delta_;
  double inv_step_;  // = n_grid - 1; multiplying by p directly gives the
                     // fractional index
};

// Inference-side counterpart to DistributionMatchingCurves. apply()/
// apply_batch() wrap the nested composition target_quantile_table.apply(
// source_distribution_table.apply(p)), giving the same interface shape
// as CalibrationLookupTable itself.
class DistributionMatchingTables {
 public:
  // Builds both lookup tables needed for distribution matching in one call.
  // curves: Both fitted curves, as returned by fit_distribution_matching.
  // n_grid: Number of uniform grid points, used for both tables.
  // Returns: The grouped, ready-to-use tables.
  static DistributionMatchingTables Create(
      const DistributionMatchingCurves& curves,
      std::size_t n_grid = internal::kDefaultNGrid);

  // Prefer Create(curves, n_grid) over calling this directly.
  // source_distribution_table: Lookup table for the source model's score CDF.
  // target_quantile_table:     Lookup table for the target model's quantile
  //                            function.
  DistributionMatchingTables(CalibrationLookupTable source_distribution_table,
                             CalibrationLookupTable target_quantile_table)
      : source_distribution_table_(std::move(source_distribution_table)),
        target_quantile_table_(std::move(target_quantile_table)) {}

  // Maps one source-model score onto the target model's distribution.
  // p: Raw score from the source model.
  // Returns: The distribution-matched score.
  double apply(double p) const {
    return target_quantile_table_.apply(source_distribution_table_.apply(p));
  }

  // Batch version of apply().
  // p:   Input source-model scores.
  // out: Output buffer, same length as p; must not alias p.
  void apply_batch(const std::vector<double>& p,
                   std::vector<double>& out) const {
    DCHECK_EQ(p.size(), out.size());
    for (std::size_t i = 0; i < p.size(); ++i) {
      out[i] = apply(p[i]);
    }
  }

  // Direct access to the underlying source-distribution lookup table.
  const CalibrationLookupTable& source_distribution_table() const {
    return source_distribution_table_;
  }

  // Direct access to the underlying target-quantile lookup table.
  const CalibrationLookupTable& target_quantile_table() const {
    return target_quantile_table_;
  }

 private:
  CalibrationLookupTable source_distribution_table_;
  CalibrationLookupTable target_quantile_table_;
};

}  // namespace yggdrasil_decision_forests::utils

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_SMOOTHED_PAV_CALIBRATION_INFERENCE_H_
