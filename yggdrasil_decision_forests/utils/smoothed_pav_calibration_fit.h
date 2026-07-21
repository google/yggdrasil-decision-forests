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

// These are utilities for constructing a smoothed, PAV calibration and
// distribution-matching curves. This is intended to be performed "offline"
// where computational efficiency and latency are not of primary concern. The
// resulting calibration curve can then be used for fast O(1) inference via the
// utilities contained in smoothed_pav_calibration_inference.h (see that file
// for relevant details).
//
// Example calibration usage with chunked data (preferred API):
//   std::vector<yggdrasil_decision_forests::utils::BinAccumulator> raw_bins(
//       n_bins);
//   // Obviously this would likely not be such a trivial loop in practice.
//   absl::Status status = absl::OkStatus();
//   for (auto [p_chunk, y_chunk] : chunks) {
//     status = yggdrasil_decision_forests::utils::accumulate_bins(
//         raw_bins, p_chunk, y_chunk, n_bins);
//     if (!status.ok()) {
//       // Handle error.
//     }
//   }
//
//   auto curve_or = yggdrasil_decision_forests::utils::fit_calibration(
//       raw_bins, z_threshold);
//   if (!curve_or.ok()) {
//     // Handle error.
//   }
//
//   // ... from here use smoothed_pav_calibration_inference.h
//   // Precompute a lookup table at a specified grid resolution.
//   auto lookup_table = yggdrasil_decision_forests::utils::build_lookup_table(
//       *curve_or, n_grid);
//
//   std::vector<double> eval_points{0.0, /*...,*/ 1.0};
//   // Allocate the output vector.
//   std::vector<double> calibrated_points(eval_points.size());
//   lookup_table.apply_batch(eval_points, calibrated_points);
//
// Example distribution matching usage with chunked data (preferred API):
//   std::vector<yggdrasil_decision_forests::utils::BinAccumulator>
//       source_bins(n_bins), target_bins(n_bins);
//   // Obviously these would likely not be such trivial loops in practice.
//   absl::Status status = absl::OkStatus();
//   for (auto& chunk : source_chunks) {
//     status = yggdrasil_decision_forests::utils::accumulate_bins(
//         source_bins, chunk, n_bins);
//     if (!status.ok()) {
//       // Handle error.
//     }
//   }
//   for (auto& chunk : target_chunks) {
//     status = yggdrasil_decision_forests::utils::accumulate_bins(
//         target_bins, chunk, n_bins);
//     if (!status.ok()) {
//       // Handle error.
//     }
//   }
//
//   auto curves_or =
//       yggdrasil_decision_forests::utils::fit_distribution_matching(
//           source_bins, target_bins);
//   if (!curves_or.ok()) {
//     // Handle error.
//   }
//
//   // ... from here use smoothed_pav_calibration_inference.h
//   // Precompute lookup tables at a specified grid resolution.
//   auto tables = yggdrasil_decision_forests::utils::build_lookup_table(
//       *curves_or, n_grid);
//
//   std::vector<double> eval_points{0.0, /*...,*/ 1.0};
//   // Allocate the output vector.
//   std::vector<double> matched_points(eval_points.size());
//   tables.apply_batch(eval_points, matched_points);

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_SMOOTHED_PAV_CALIBRATION_FIT_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_SMOOTHED_PAV_CALIBRATION_FIT_H_

#include <cassert>
#include <cstddef>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace yggdrasil_decision_forests::utils {

// Per-bin sufficient statistics for both calibration-curve fitting and
// score-distribution fitting. Array-of-structs layout: PAV's merge step
// combines one whole (sum_pred, sum_true, count) tuple at a time.
// TODO: Add support for differently weighted samples.
struct BinAccumulator {
  // Sum of raw predicted probabilities falling in this bin.
  double sum_pred = 0.0;
  // Sum of observed 0/1 labels falling in this bin (unused for
  // score-only/distribution fitting).
  double sum_true = 0.0;
  // Number of samples falling in this bin.
  double count = 0.0;

  // Mean predicted probability in this bin. sum_pred / count.
  constexpr double prob_pred() const { return sum_pred / count; }

  // Mean observed label (empirical positive rate) in this bin.
  // sum_true / count.
  constexpr double prob_true() const { return sum_true / count; }

  // Merges another bin's statistics into this one (element-wise sum).
  // o: The bin to merge into this one.
  constexpr BinAccumulator& operator+=(const BinAccumulator& o) {
    sum_pred += o.sum_pred;
    sum_true += o.sum_true;
    count += o.count;
    return *this;
  }
};

// Bins one chunk of labeled (p, y) data into an existing, pre-sized
// accumulator array. Safe to call repeatedly across chunks when the full
// dataset can't be held in memory at once -- does not drop empty bins
// (a bin empty after this chunk may be populated by a later one); call
// finalize_bins() once, after the last chunk.
// bins:    Pre-sized accumulator array (size must equal n_bins), mutated in
//          place.
// p:       Raw predicted probabilities for this chunk.
// y:       Observed 0/1 labels for this chunk, same length as p.
// n_bins:  Total number of equal-width bins partitioning [0,1]; must match
//          bins.size().
absl::Status accumulate_bins(std::vector<BinAccumulator>& bins,
                             const std::vector<double>& p,
                             const std::vector<double>& y, std::size_t n_bins);

// Score-only (unlabeled) variant of accumulate_bins, for fitting a score
// distribution (see fit_score_distribution /
// fit_score_quantile_function) rather than a calibration curve -- there
// is no label. sum_true is never touched.
// bins:    Pre-sized accumulator array (size must equal n_bins), mutated in
//          place.
// p:       Raw scores for this chunk.
// n_bins:  Total number of equal-width bins partitioning [0,1]; must match
//          bins.size().
absl::Status accumulate_bins(std::vector<BinAccumulator>& bins,
                             const std::vector<double>& p, std::size_t n_bins);

// Single-pass convenience wrapper: bins all of (p, y) in one call and
// finalizes the result. Use accumulate_bins()/finalize_bins() directly
// instead if the full dataset can't be passed in one call.
// p:       Raw predicted probabilities.
// y:       Observed 0/1 labels, same length as p.
// n_bins:  Number of equal-width bins partitioning [0,1].
// Returns: Non-empty bins only, ordered by predicted probability.
absl::StatusOr<std::vector<BinAccumulator>> aggregate_bins(
    const std::vector<double>& p, const std::vector<double>& y,
    std::size_t n_bins);

// Single-pass convenience wrapper for the score-only case.
// p:       Raw scores.
// n_bins:  Number of equal-width bins partitioning [0,1].
// Returns: Non-empty bins only, ordered by score.
absl::StatusOr<std::vector<BinAccumulator>> aggregate_bins(
    const std::vector<double>& p, std::size_t n_bins);

// Drops empty (count == 0) bins. Idempotent: calling this on an
// already-finalized vector removes nothing, so it's safe to call
// unconditionally regardless of whether the input has been finalized yet.
// bins: Bin accumulators, e.g. straight from accumulate_bins() or already
// finalized.
// Returns: A new vector containing only the non-empty bins, in original order.
std::vector<BinAccumulator> finalize_bins(std::vector<BinAccumulator> bins);

// Weighted Pool-Adjacent-Violators: merges adjacent bins so that
// prob_true() is non-decreasing. With z_threshold > 0, also merges adjacent
// pools that are statistically indistinguishable (two-proportion z-test),
// removing noise-driven wiggle that plain PAV leaves behind. Do not use for
// score-distribution fitting (the z-test assumes independent samples, which
// adjacent cumulative counts are not).
// bins:         Bins ordered by predicted probability, count != 0.
// z_threshold:  Significance threshold for additional merging. 0.0 (default)
//               is exactly plain PAV (never merges non-violating pairs).
// Returns:      Merged pools, non-decreasing in prob_true().
std::vector<BinAccumulator> merge_bins_for_monotonicity(
    const std::vector<BinAccumulator>& bins, double z_threshold = 0.0);

// Plain (x, y) point-sequence pair, used as the return type for
// anchor_endpoints().
struct MonotoneCurvePoints {
  std::vector<double> x;
  std::vector<double> y;
};

// Prepends (0,0) and appends (1,1) to a point sequence, making the
// eventual fitted curve a genuine CDF-like map: h(0)=0, h(1)=1 exactly.
// x: Interior x-values (e.g. pool/bin representative predicted probabilities).
// y: Interior y-values, same length as x.
// Returns: x and y each with the two anchors added.
MonotoneCurvePoints anchor_endpoints(const std::vector<double>& x,
                                     const std::vector<double>& y);

// Inverse-variance confidence per point: n / max(p(1-p), eps), the exact
// precision of a binomial proportion estimate. Also exactly correct (not
// just a reasonable analogy) for empirical-CDF points, since a CDF value
// is itself a sample proportion with the same variance form.
// prob_true: Per-point proportion estimate (prob_true() for a calibration
//            curve, or cumulative fraction for a CDF).
// count:     Per-point sample count, same length as prob_true.
// eps:       Variance floor, keeps confidence finite at p=0 or p=1 (e.g. the
//            anchored endpoints) without needing any special-casing.
// Returns:   Confidence weight per point, same length as prob_true.
absl::StatusOr<std::vector<double>> confidence_weight(
    const std::vector<double>& prob_true, const std::vector<double>& count,
    double eps = 1e-6);

// Original (unweighted) Fritsch-Carlson derivative formula. Kept only as
// a reference for cross-checking against scipy/an unweighted
// implementation (this is not used by fit_calibration or any score-
// distribution fitting; see pchip_slopes for the version actually used).
// x: Strictly increasing knot positions.
// y: Non-decreasing values at each knot, same length as x.
// Returns: PCHIP derivative at each knot.
absl::StatusOr<std::vector<double>> pchip_slopes_unweighted(
    const std::vector<double>& x, const std::vector<double>& y);

// Box-safe, confidence-weighted Fritsch-Carlson derivatives. Each
// interior derivative is min(confidence-weighted harmonic mean of the two
// adjacent secants, 3*min(those secants)) (the safety bound is part of the
// formula itself, provably guaranteeing a monotone Hermite interpolant with
// no separate correction pass).
// x:      Strictly increasing knot positions.
// y:      Non-decreasing values at each knot, same length as x.
// weight: Confidence weight at each knot (see confidence_weight), same length
//         as x.
// Returns: Derivative at each knot, safe for monotone cubic Hermite
//          interpolation.
absl::StatusOr<std::vector<double>> pchip_slopes(
    const std::vector<double>& x, const std::vector<double>& y,
    const std::vector<double>& weight);

// A fitted monotone curve, ready for evaluation via pchip_eval_single or for
// building a fast lookup table via build_lookup_table.
struct FittedCalibrationCurve {
  // Strictly increasing knot positions; x[0]=0, x.back()=1.
  std::vector<double> x;
  // Non-decreasing values; y[0]=0, y.back()=1.
  std::vector<double> y;
  // PCHIP derivative at each knot.
  std::vector<double> d;
};

// Shared tail of the fitting pipeline (anchor -> fix endpoint ties ->
// confidence-weight -> PCHIP slopes), used by both calibration-curve
// fitting (after PAV) and score-distribution fitting (which skips PAV).
// x, y, w: Interior points and per-point weights, not yet anchored.
// Returns: The fully fitted, anchored curve.
absl::StatusOr<FittedCalibrationCurve> fit_monotone_curve(
    std::vector<double> x, std::vector<double> y, std::vector<double> w);

// Fits a calibration curve from already-accumulated bins. Takes bins by
// const reference and copies internally, so the same accumulated bins
// can be reused across multiple calls (e.g. comparing several
// z_threshold candidates without re-aggregating). Works whether or not
// bins has already been finalized (finalize_bins is idempotent).
// bins:        Accumulated (possibly unfinalized) bins.
// z_threshold: Passed through to merge_bins_for_monotonicity.
// Returns:     The fitted calibration curve.
absl::StatusOr<FittedCalibrationCurve> fit_calibration(
    const std::vector<BinAccumulator>& bins, double z_threshold = 0.0);

// Convenience wrapper: aggregates raw (p, y) in one call, then fits.
// p, y:         Raw predicted probabilities and labels.
// n_bins:       Number of equal-width bins.
// z_threshold:  Passed through to merge_bins_for_monotonicity.
// Returns:      The fitted calibration curve.
absl::StatusOr<FittedCalibrationCurve> fit_calibration(
    const std::vector<double>& p, const std::vector<double>& y_raw,
    std::size_t n_bins, double z_threshold = 0.0);

// Builds (score, cumulative fraction) pairs from finalized, score-sorted
// bins, using the midpoint plotting-position convention: y_i =
// (cumulative count strictly before bin i + half of bin i's own count) /
// total count -- centers y_i consistently with x_i (also a within-bin
// average), avoiding systematic bias between the two.
// bins: Finalized bins, ordered by score.
// x, y, w: Output parameters, cleared and filled with score, cumulative
//          fraction, and count respectively, one entry per bin.
void cumulative_from_bins(const std::vector<BinAccumulator>& bins,
                          std::vector<double>& x, std::vector<double>& y,
                          std::vector<double>& w);

// Fits F(x), the empirical score CDF, from already-accumulated score-only
// bins. Does not use PAV (a cumulative sum is monotone by construction).
// bins: Accumulated (possibly unfinalized) score-only bins.
// Returns: The fitted CDF, F: score -> cumulative fraction in [0,1].
absl::StatusOr<FittedCalibrationCurve> fit_score_distribution(
    const std::vector<BinAccumulator>& bins);

// Convenience wrapper: aggregates raw scores in one call, then fits.
// scores: Raw scores (unlabeled).
// n_bins: Number of equal-width bins.
// Returns: The fitted CDF.
absl::StatusOr<FittedCalibrationCurve> fit_score_distribution(
    const std::vector<double>& scores, std::size_t n_bins);

// Fits F^{-1}(u), the score quantile function, built directly (x and y
// swapped at construction time) rather than by numerically inverting F.
// bins: Accumulated (possibly unfinalized) score-only bins.
// Returns: The fitted quantile function, F^{-1}: cumulative fraction -> score.
absl::StatusOr<FittedCalibrationCurve> fit_score_quantile_function(
    const std::vector<BinAccumulator>& bins);

// Convenience wrapper: aggregates raw scores in one call, then fits.
// scores: Raw scores (unlabeled).
// n_bins: Number of equal-width bins.
// Returns: The fitted quantile function.
absl::StatusOr<FittedCalibrationCurve> fit_score_quantile_function(
    const std::vector<double>& scores, std::size_t n_bins);

// The two curves needed to remap one model's score distribution onto
// another's (quantile/histogram matching), grouped together.
struct DistributionMatchingCurves {
  // F_source: source model's own score CDF.
  FittedCalibrationCurve source_distribution;
  // F_target^{-1}: target model's quantile function.
  FittedCalibrationCurve target_quantile;
};

// Fits both curves needed for distribution matching in one call, so the
// caller never needs to invoke fit_score_distribution /
// fit_score_quantile_function individually.
// source_bins: Accumulated score-only bins for the model being remapped
//              (e.g. the new model with a different prediction distribution).
// target_bins: Accumulated score-only bins for the model whose distribution
//              is being reproduced (e.g. the currently-deployed model).
// Returns: Both fitted curves, grouped.
absl::StatusOr<DistributionMatchingCurves> fit_distribution_matching(
    const std::vector<BinAccumulator>& source_bins,
    const std::vector<BinAccumulator>& target_bins);

// Convenience wrapper: aggregates both raw score sets in one call, then fits.
// source_scores, target_scores: Raw scores for each model.
// n_bins: Number of equal-width bins, used for both.
// Returns: Both fitted curves, grouped.
absl::StatusOr<DistributionMatchingCurves> fit_distribution_matching(
    const std::vector<double>& source_scores,
    const std::vector<double>& target_scores, std::size_t n_bins);

}  // namespace yggdrasil_decision_forests::utils

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_SMOOTHED_PAV_CALIBRATION_FIT_H_
