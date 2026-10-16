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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/histogram_bnb.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_accumulator.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_scanner.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/label_accessor.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/splitter.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/training.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace {

constexpr int kNumSums = 4;
constexpr int kNumRanges = 4;

// Relative tolerance used to compare the upper bound of a bin to the score of
// the incumbent. Makes sure numerical errors never prune a bin containing a
// split as good as the incumbent.
constexpr double kBoundRelativeTolerance = 1e-10;

// Node constraints for the hessian accumulators (no constraints).
const decision_tree::NodeConstraints& NoConstraints() {
  static const decision_tree::NodeConstraints* constraints =
      new decision_tree::NodeConstraints();
  return *constraints;
}

utils::NormalDistributionDouble ToNormalDistribution(const BinStats& stats) {
  utils::proto::NormalDistributionDouble proto;
  proto.set_sum(stats.sum_gradients);
  proto.set_sum_squares(stats.sum_squared_gradients);
  proto.set_count(stats.sum_weights);
  utils::NormalDistributionDouble dist;
  dist.Load(proto);
  return dist;
}

// Weighted variance of the gradient times the sum of weights. Same as
// "utils::NormalDistributionDouble::VarTimesSumWeights".
double VarTimesSumWeights(const BinStats& stats) {
  if (stats.sum_weights <= 0) {
    return 0.;
  }
  return stats.sum_squared_gradients -
         (stats.sum_gradients * stats.sum_gradients) / stats.sum_weights;
}

// Hessian score of a set of examples. Same as
// "decision_tree::LabelHessianNumericalScoreAccumulator::ComputeScore".
double HessianScore(const BinStats& stats, const double l1, const double l2) {
  return decision_tree::LabelHessianNumericalScoreAccumulator::ComputeScore(
      stats.sum_gradients, stats.sum_hessians, l1, l2, NoConstraints());
}

// Sets the label statistics and example counts of a split given the
// statistics of its positive child.
void SetSplitLabelStatistics(const BinStats& parent, const BinStats& pos,
                             Split* split) {
  BinStats neg = parent;
  neg.SubSums(pos);
  BinStatsToLabelStatistics(neg, &split->label_statistics[0]);
  BinStatsToLabelStatistics(pos, &split->label_statistics[1]);
}

// Sets the label statistics of a categorical split.
absl::Status SetCategoricalSplitLabelStatistics(
    const BinStats& parent, const std::vector<BinStats>& per_value,
    Split* split) {
  ASSIGN_OR_RETURN(const auto positive_values,
                   decision_tree::ExactElementsFromContainsCondition(
                       per_value.size(), split->condition.condition()));
  BinStats pos;
  for (const auto value : positive_values) {
    if (value < 0 || value >= per_value.size()) {
      return absl::InternalError("Unexpected categorical value");
    }
    pos.Add(per_value[value]);
  }
  SetSplitLabelStatistics(parent, pos, split);
  return absl::OkStatus();
}

template <typename LabelBucket>
void SetNumericalWithHessianLabelBucket(const BinStats& stats,
                                        LabelBucket* bucket) {
  bucket->value = ToNormalDistribution(stats);
  bucket->sum_hessian = stats.sum_hessians;
  bucket->count = stats.count;
}

template <typename LabelBucket>
void SetHessianLabelBucket(const BinStats& stats, const double l1,
                           const double l2, LabelBucket* bucket) {
  bucket->content.sum_gradient = stats.sum_gradients;
  bucket->content.sum_hessian = stats.sum_hessians;
  bucket->content.sum_weight = stats.sum_weights;
  bucket->count = stats.count;
  // Same as "LabelHessianNumericalBucket::Filler::Finalize".
  if (bucket->content.sum_hessian > 0) {
    const double clamped_hessian =
        std::max(static_cast<double>(bucket->content.sum_hessian),
                 decision_tree::LabelHessianNumericalScoreAccumulator::
                     kMinHessianForNewtonStep);
    bucket->priority =
        decision_tree::l1_threshold(bucket->content.sum_gradient, l1) /
        (clamped_hessian + l2);
  } else {
    bucket->priority = 0.;
  }
}

}  // namespace

// BinStats
// ========

void BinStats::Add(const BinStats& other) {
  count += other.count;
  sum_weights += other.sum_weights;
  sum_gradients += other.sum_gradients;
  sum_squared_gradients += other.sum_squared_gradients;
  sum_hessians += other.sum_hessians;
  min_value = std::min(min_value, other.min_value);
  max_value = std::max(max_value, other.max_value);
  min_ratio = std::min(min_ratio, other.min_ratio);
  max_ratio = std::max(max_ratio, other.max_ratio);
}

void BinStats::SubSums(const BinStats& other) {
  count -= other.count;
  sum_weights -= other.sum_weights;
  sum_gradients -= other.sum_gradients;
  sum_squared_gradients -= other.sum_squared_gradients;
  sum_hessians -= other.sum_hessians;
}

void BinStatsToLabelStatistics(const BinStats& src,
                               decision_tree::proto::LabelStatistics* dst) {
  dst->Clear();
  dst->set_num_examples(src.count);
  auto* reg = dst->mutable_regression_with_hessian();
  reg->mutable_labels()->set_sum(src.sum_gradients);
  reg->mutable_labels()->set_sum_squares(src.sum_squared_gradients);
  reg->mutable_labels()->set_count(src.sum_weights);
  reg->set_sum_hessian(src.sum_hessians);
}

absl::StatusOr<BinStats> LabelStatisticsToBinStats(
    const decision_tree::proto::LabelStatistics& src) {
  if (!src.has_regression_with_hessian()) {
    return absl::InvalidArgumentError(
        "Label statistics without \"regression_with_hessian\"");
  }
  BinStats dst;
  dst.count = src.num_examples();
  dst.sum_gradients = src.regression_with_hessian().labels().sum();
  dst.sum_squared_gradients =
      src.regression_with_hessian().labels().sum_squares();
  dst.sum_weights = src.regression_with_hessian().labels().count();
  dst.sum_hessians = src.regression_with_hessian().sum_hessian();
  return dst;
}

void AppendBinToProto(const BinStats& bin, proto::BinStatistics* dst) {
  dst->add_counts(bin.count);
  dst->add_sums(bin.sum_weights);
  dst->add_sums(bin.sum_gradients);
  dst->add_sums(bin.sum_squared_gradients);
  dst->add_sums(bin.sum_hessians);
  dst->add_ranges(bin.min_value);
  dst->add_ranges(bin.max_value);
  dst->add_ranges(bin.min_ratio);
  dst->add_ranges(bin.max_ratio);
}

int NumBinsInProto(const proto::BinStatistics& src) {
  return src.counts_size();
}

BinStats BinFromProto(const proto::BinStatistics& src, const int bin_idx) {
  BinStats bin;
  bin.count = src.counts(bin_idx);
  bin.sum_weights = src.sums(kNumSums * bin_idx + 0);
  bin.sum_gradients = src.sums(kNumSums * bin_idx + 1);
  bin.sum_squared_gradients = src.sums(kNumSums * bin_idx + 2);
  bin.sum_hessians = src.sums(kNumSums * bin_idx + 3);
  bin.min_value = src.ranges(kNumRanges * bin_idx + 0);
  bin.max_value = src.ranges(kNumRanges * bin_idx + 1);
  bin.min_ratio = src.ranges(kNumRanges * bin_idx + 2);
  bin.max_ratio = src.ranges(kNumRanges * bin_idx + 3);
  return bin;
}

absl::StatusOr<std::vector<BinStats>> BinsFromProto(
    const proto::BinStatistics& src) {
  const int num_bins = NumBinsInProto(src);
  if (src.sums_size() != kNumSums * num_bins ||
      src.ranges_size() != kNumRanges * num_bins) {
    return absl::InvalidArgumentError("Malformed BinStatistics");
  }
  std::vector<BinStats> bins;
  bins.reserve(num_bins);
  for (int bin_idx = 0; bin_idx < num_bins; bin_idx++) {
    bins.push_back(BinFromProto(src, bin_idx));
  }
  return bins;
}

// SplitCriterion
// ==============

ScoreBound SplitCriterion::UpperBound(const BinStats& parent,
                                      const BinStats& prefix,
                                      const BinStats& bin) const {
  const double offset = ConvexOffset(parent);
  if (!std::isfinite(bin.min_ratio) || !std::isfinite(bin.max_ratio)) {
    return {std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
  }
  const double h_parent = H(parent);
  const double g_parent = G(parent);
  const double h0 = H(prefix);
  const double g0 = G(prefix);
  const double hk = H(bin);
  const double gk = G(bin);

  double best = -std::numeric_limits<double>::infinity();
  double scale = std::abs(offset);
  const auto eval = [&](const double h, const double g) {
    const double neg_term = ConvexTerm(parent, h, g);
    const double pos_term = ConvexTerm(parent, h_parent - h, g_parent - g);
    scale = std::max({scale, std::abs(neg_term), std::abs(pos_term)});
    best = std::max(best, neg_term + pos_term);
  };

  // The two vertices on the bin boundaries.
  eval(h0, g0);
  eval(h0 + hk, g0 + gk);

  // The two other vertices of the parallelogram.
  const double r_min = bin.min_ratio;
  const double r_max = bin.max_ratio;
  if (r_max > r_min && hk > 0) {
    // Path: slope r_max then r_min.
    const double t_hi = std::clamp((gk - r_min * hk) / (r_max - r_min), 0., hk);
    eval(h0 + t_hi, g0 + r_max * t_hi);
    // Path: slope r_min then r_max.
    const double t_lo = std::clamp((r_max * hk - gk) / (r_max - r_min), 0., hk);
    eval(h0 + t_lo, g0 + r_min * t_lo);
  }
  return {best - offset, scale};
}

// VarianceReductionCriterion
// ==========================

double VarianceReductionCriterion::Score(const BinStats& parent,
                                         const BinStats& neg,
                                         const BinStats& pos) const {
  // Same computation (and order of operations) as "decision_tree::Score" with
  // "LabelNumericalWithHessianBucket::Initializer::NormalizeScore".
  const double score_neg = VarTimesSumWeights(neg);
  const double score_pos = VarTimesSumWeights(pos);
  return (VarTimesSumWeights(parent) - (score_pos + score_neg)) /
         parent.sum_weights;
}

double VarianceReductionCriterion::ConvexTerm(const BinStats& parent,
                                              const double h,
                                              const double g) const {
  if (h <= 0 || parent.sum_weights <= 0) {
    return 0.;
  }
  return g * g / h / parent.sum_weights;
}

double VarianceReductionCriterion::ConvexOffset(const BinStats& parent) const {
  return ConvexTerm(parent, parent.sum_weights, parent.sum_gradients);
}

absl::StatusOr<bool> VarianceReductionCriterion::FindBestCategoricalSplit(
    const int feature, const int na_replacement, const BinStats& parent,
    const std::vector<BinStats>& per_value, const ExampleIndex min_examples,
    Split* split) const {
  using LabelBucket = decision_tree::LabelNumericalWithHessianBucket<true>;
  using ExampleBucketSet =
      decision_tree::ExampleBucketSet<decision_tree::ExampleBucket<
          decision_tree::FeatureCategoricalBucket, LabelBucket>>;
  const int num_values = per_value.size();
  const std::vector<int32_t> empty_attributes;
  decision_tree::FeatureCategoricalBucket::Filler feature_filler(
      num_values, na_replacement, empty_attributes);

  ExampleBucketSet example_bucket_set;
  example_bucket_set.items.resize(num_values);
  std::vector<std::pair<float, int32_t>> feature_order(num_values);
  for (int value = 0; value < num_values; value++) {
    auto& item = example_bucket_set.items[value];
    feature_filler.InitializeAndZero(value, &item.feature);
    SetNumericalWithHessianLabelBucket(per_value[value], &item.label);
    feature_order[value] = {item.label.value.Mean(), value};
  }
  // Same ordering as the existing distributed splitter.
  std::sort(feature_order.begin(), feature_order.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  decision_tree::proto::LabelStatistics parent_label_stats;
  BinStatsToLabelStatistics(parent, &parent_label_stats);
  const LabelBucket::Initializer initializer(parent_label_stats);

  decision_tree::PerThreadCacheV2 cache;
  split->condition.Clear();
  const auto result = decision_tree::ScanSplitsCustomOrder<
      ExampleBucketSet,
      decision_tree::LabelNumericalWithHessianScoreAccumulator>(
      feature_order, feature_filler, initializer, example_bucket_set,
      parent.count, min_examples, feature, &split->condition, &cache);
  if (result != decision_tree::SplitSearchResult::kBetterSplitFound) {
    return false;
  }
  RETURN_IF_ERROR(SetCategoricalSplitLabelStatistics(parent, per_value, split));
  return true;
}

absl::StatusOr<bool> VarianceReductionCriterion::FindBestBooleanSplit(
    const int feature, const bool na_replacement, const BinStats& parent,
    const std::vector<BinStats>& per_value, const ExampleIndex min_examples,
    Split* split) const {
  if (per_value.size() != 2) {
    return absl::InternalError("Boolean features should have 2 values");
  }
  using LabelBucket = decision_tree::LabelNumericalWithHessianBucket<true>;
  using ExampleBucketSet =
      decision_tree::ExampleBucketSet<decision_tree::ExampleBucket<
          decision_tree::FeatureBooleanBucket, LabelBucket>>;
  const std::vector<int8_t> empty_attributes;
  decision_tree::FeatureBooleanBucket::Filler feature_filler(na_replacement,
                                                             empty_attributes);
  ExampleBucketSet example_bucket_set;
  example_bucket_set.items.resize(2);
  for (int value = 0; value < 2; value++) {
    auto& item = example_bucket_set.items[value];
    feature_filler.InitializeAndZero(value, &item.feature);
    SetNumericalWithHessianLabelBucket(per_value[value], &item.label);
  }

  decision_tree::proto::LabelStatistics parent_label_stats;
  BinStatsToLabelStatistics(parent, &parent_label_stats);
  const LabelBucket::Initializer initializer(parent_label_stats);

  decision_tree::PerThreadCacheV2 cache;
  split->condition.Clear();
  const auto result = decision_tree::ScanSplits<
      ExampleBucketSet,
      decision_tree::LabelNumericalWithHessianScoreAccumulator>(
      feature_filler, initializer, example_bucket_set, parent.count,
      min_examples, feature, &split->condition, &cache);
  if (result != decision_tree::SplitSearchResult::kBetterSplitFound) {
    return false;
  }
  SetSplitLabelStatistics(parent, per_value[1], split);
  return true;
}

// HessianGainCriterion
// ====================

bool HessianGainCriterion::IsValidSplit(const BinStats& neg,
                                        const BinStats& pos) const {
  const double m = options_.min_sum_hessian_in_leaf;
  if (m > 0 && (neg.sum_hessians < m || pos.sum_hessians < m)) {
    return false;
  }
  return true;
}

double HessianGainCriterion::ParentScore(const BinStats& parent) const {
  return HessianScore(parent, options_.l1, options_.l2_numerical);
}

double HessianGainCriterion::Score(const BinStats& parent, const BinStats& neg,
                                   const BinStats& pos) const {
  // Same computation (and order of operations) as "decision_tree::Score" with
  // "LabelHessianNumericalBucket::Initializer::NormalizeScore".
  const double score_neg =
      HessianScore(neg, options_.l1, options_.l2_numerical);
  const double score_pos =
      HessianScore(pos, options_.l1, options_.l2_numerical);
  return (score_pos + score_neg) -
         (options_.subtract_parent ? ParentScore(parent) : 0.);
}

double HessianGainCriterion::MinimumScore(const BinStats& parent) const {
  return options_.subtract_parent ? 0. : ParentScore(parent);
}

double HessianGainCriterion::ConvexTerm(const BinStats& parent, const double h,
                                        const double g) const {
  // F without the clamping of the hessian: l1(g)^2 / (h + l2). This function
  // is jointly convex for h + l2 > 0, and greater or equal to the clamped
  // version.
  const double numerator = decision_tree::l1_threshold(g, options_.l1);
  const double denominator = std::max(h, 0.) + options_.l2_numerical;
  if (denominator <= 0) {
    return numerator == 0 ? 0. : std::numeric_limits<double>::infinity();
  }
  return numerator * numerator / denominator;
}

double HessianGainCriterion::ConvexOffset(const BinStats& parent) const {
  return options_.subtract_parent ? ParentScore(parent) : 0.;
}

absl::StatusOr<bool> HessianGainCriterion::FindBestCategoricalSplit(
    const int feature, const int na_replacement, const BinStats& parent,
    const std::vector<BinStats>& per_value, const ExampleIndex min_examples,
    Split* split) const {
  using ExampleBucketSet =
      decision_tree::FeatureCategoricalLabelHessianNumerical</*weighted=*/true>;
  using LabelBucket = decision_tree::LabelHessianNumericalBucket<true>;
  const int num_values = per_value.size();
  const std::vector<int32_t> empty_attributes;
  decision_tree::FeatureCategoricalBucket::Filler feature_filler(
      num_values, na_replacement, empty_attributes);

  ExampleBucketSet example_bucket_set;
  example_bucket_set.items.resize(num_values);
  for (int value = 0; value < num_values; value++) {
    auto& item = example_bucket_set.items[value];
    feature_filler.InitializeAndZero(value, &item.feature);
    SetHessianLabelBucket(per_value[value], options_.l1,
                          options_.l2_categorical, &item.label);
  }
  // Same ordering as "FillExampleBucketSet" with "require_label_sorting".
  std::sort(example_bucket_set.items.begin(), example_bucket_set.items.end(),
            typename ExampleBucketSet::ExampleBucketType::SortLabel());

  const LabelBucket::Initializer initializer(
      parent.sum_gradients, parent.sum_hessians, parent.sum_weights,
      options_.l1, options_.l2_categorical, options_.min_sum_hessian_in_leaf,
      options_.subtract_parent, /*monotonic_direction=*/0, NoConstraints());

  decision_tree::PerThreadCacheV2 cache;
  split->condition.Clear();
  const auto result = decision_tree::ScanSplits<
      ExampleBucketSet, decision_tree::LabelHessianNumericalScoreAccumulator>(
      feature_filler, initializer, example_bucket_set, parent.count,
      min_examples, feature, &split->condition, &cache);
  if (result != decision_tree::SplitSearchResult::kBetterSplitFound) {
    return false;
  }
  RETURN_IF_ERROR(SetCategoricalSplitLabelStatistics(parent, per_value, split));
  return true;
}

absl::StatusOr<bool> HessianGainCriterion::FindBestBooleanSplit(
    const int feature, const bool na_replacement, const BinStats& parent,
    const std::vector<BinStats>& per_value, const ExampleIndex min_examples,
    Split* split) const {
  if (per_value.size() != 2) {
    return absl::InternalError("Boolean features should have 2 values");
  }
  using ExampleBucketSet =
      decision_tree::FeatureBooleanLabelHessianNumerical</*weighted=*/true>;
  using LabelBucket = decision_tree::LabelHessianNumericalBucket<true>;
  const std::vector<int8_t> empty_attributes;
  decision_tree::FeatureBooleanBucket::Filler feature_filler(na_replacement,
                                                             empty_attributes);
  ExampleBucketSet example_bucket_set;
  example_bucket_set.items.resize(2);
  for (int value = 0; value < 2; value++) {
    auto& item = example_bucket_set.items[value];
    feature_filler.InitializeAndZero(value, &item.feature);
    SetHessianLabelBucket(per_value[value], options_.l1, options_.l2_numerical,
                          &item.label);
  }

  const LabelBucket::Initializer initializer(
      parent.sum_gradients, parent.sum_hessians, parent.sum_weights,
      options_.l1, options_.l2_numerical, options_.min_sum_hessian_in_leaf,
      options_.subtract_parent, /*monotonic_direction=*/0, NoConstraints());

  decision_tree::PerThreadCacheV2 cache;
  split->condition.Clear();
  const auto result = decision_tree::ScanSplits<
      ExampleBucketSet, decision_tree::LabelHessianNumericalScoreAccumulator>(
      feature_filler, initializer, example_bucket_set, parent.count,
      min_examples, feature, &split->condition, &cache);
  if (result != decision_tree::SplitSearchResult::kBetterSplitFound) {
    return false;
  }
  SetSplitLabelStatistics(parent, per_value[1], split);
  return true;
}

// Worker side
// ===========

int NumericalBinIndex(absl::Span<const float> boundaries, const float value) {
  return std::upper_bound(boundaries.begin(), boundaries.end(), value) -
         boundaries.begin();
}

namespace {

// Generic histogram filling.
template <typename GetBin>
absl::Status FillHistograms(const size_t num_examples,
                            const ExampleToNodeMap& example_to_node,
                            const std::vector<int>& node_to_slot,
                            const LabelData& labels, GetBin get_bin,
                            std::vector<std::vector<BinStats>>* histograms) {
  if (example_to_node.size() != num_examples) {
    return absl::InternalError(
        absl::Substitute("Unexpected number of examples $0 != $1",
                         example_to_node.size(), num_examples));
  }
  for (size_t example_idx = 0; example_idx < num_examples; example_idx++) {
    const NodeIndex node_idx = example_to_node[example_idx];
    if (node_idx == kClosedNode) {
      continue;
    }
    DCHECK_LT(node_idx, node_to_slot.size());
    const int slot = node_to_slot[node_idx];
    if (slot < 0) {
      continue;
    }
    float value;
    int bin_idx;
    get_bin(example_idx, &value, &bin_idx);
    auto& histogram = (*histograms)[slot];
    DCHECK_GE(bin_idx, 0);
    DCHECK_LT(bin_idx, histogram.size());
    labels.AddExample(example_idx, value, &histogram[bin_idx]);
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status FillNumericalHistograms(
    absl::Span<const float> values, absl::Span<const float> boundaries,
    const ExampleToNodeMap& example_to_node,
    const std::vector<int>& node_to_slot, const LabelData& labels,
    std::vector<std::vector<BinStats>>* histograms) {
  for (const auto& histogram : *histograms) {
    if (histogram.size() != boundaries.size() + 1) {
      return absl::InternalError("Wrong number of bins");
    }
  }
  return FillHistograms(
      values.size(), example_to_node, node_to_slot, labels,
      [&](const size_t example_idx, float* value, int* bin_idx) {
        *value = values[example_idx];
        *bin_idx = NumericalBinIndex(boundaries, *value);
      },
      histograms);
}

absl::Status FillCategoricalHistograms(
    absl::Span<const int32_t> values, const int num_values,
    const int na_replacement, const ExampleToNodeMap& example_to_node,
    const std::vector<int>& node_to_slot, const LabelData& labels,
    std::vector<std::vector<BinStats>>* histograms) {
  for (const auto& histogram : *histograms) {
    if (histogram.size() != num_values) {
      return absl::InternalError("Wrong number of bins");
    }
  }
  for (const auto value : values) {
    if (value >= num_values) {
      return absl::InvalidArgumentError("Categorical value out of range");
    }
  }
  return FillHistograms(
      values.size(), example_to_node, node_to_slot, labels,
      [&](const size_t example_idx, float* value, int* bin_idx) {
        int32_t v = values[example_idx];
        if (v < 0) {
          v = na_replacement;
        }
        *value = v;
        *bin_idx = v;
      },
      histograms);
}

absl::Status FillBooleanHistograms(
    absl::Span<const int8_t> values, const bool na_replacement,
    const ExampleToNodeMap& example_to_node,
    const std::vector<int>& node_to_slot, const LabelData& labels,
    std::vector<std::vector<BinStats>>* histograms) {
  for (const auto& histogram : *histograms) {
    if (histogram.size() != 2) {
      return absl::InternalError("Wrong number of bins");
    }
  }
  return FillHistograms(
      values.size(), example_to_node, node_to_slot, labels,
      [&](const size_t example_idx, float* value, int* bin_idx) {
        int8_t v = values[example_idx];
        if (v == dataset::VerticalDataset::BooleanColumn::kNaValue) {
          v = na_replacement;
        }
        *value = v;
        *bin_idx = v != 0;
      },
      histograms);
}

std::vector<std::vector<UnsignedExampleIdx>> ListExamplesPerNode(
    const ExampleToNodeMap& example_to_node, const int num_nodes) {
  std::vector<std::vector<UnsignedExampleIdx>> examples(num_nodes);
  for (size_t example_idx = 0; example_idx < example_to_node.size();
       example_idx++) {
    const NodeIndex node_idx = example_to_node[example_idx];
    if (node_idx != kClosedNode && node_idx < num_nodes) {
      examples[node_idx].push_back(example_idx);
    }
  }
  return examples;
}

absl::Status RefineBins(absl::Span<const float> values,
                        absl::Span<const UnsignedExampleIdx> candidates,
                        const std::vector<BinRefinement>& refinements,
                        const LabelData& labels,
                        std::vector<std::vector<BinStats>>* results,
                        std::vector<UnsignedExampleIdx>* survivors) {
  const int num_refinements = refinements.size();
  results->assign(num_refinements, {});
  if (survivors) {
    survivors->clear();
  }

  // Sort the refinements by range.
  std::vector<int> order(num_refinements);
  for (int i = 0; i < num_refinements; i++) {
    order[i] = i;
    if (!refinements[i].unique_values) {
      (*results)[i].assign(refinements[i].boundaries.size() + 1, {});
    }
  }
  std::sort(order.begin(), order.end(), [&](const int a, const int b) {
    return refinements[a].min_value < refinements[b].min_value;
  });
  std::vector<float> sorted_min_values(num_refinements);
  for (int i = 0; i < num_refinements; i++) {
    sorted_min_values[i] = refinements[order[i]].min_value;
    if (i > 0 && refinements[order[i]].min_value <=
                     refinements[order[i - 1]].max_value) {
      return absl::InvalidArgumentError("Overlapping refinement ranges");
    }
  }

  // Examples (value, example idx) in each "unique_values" refinement.
  std::vector<std::vector<std::pair<float, UnsignedExampleIdx>>> unique_items(
      num_refinements);

  for (const auto example_idx : candidates) {
    if (example_idx >= values.size()) {
      return absl::InternalError("Example index out of range");
    }
    const float value = values[example_idx];
    const int pos = std::upper_bound(sorted_min_values.begin(),
                                     sorted_min_values.end(), value) -
                    sorted_min_values.begin() - 1;
    if (pos < 0) {
      continue;
    }
    const int refinement_idx = order[pos];
    const auto& refinement = refinements[refinement_idx];
    if (value > refinement.max_value) {
      continue;
    }
    if (survivors) {
      survivors->push_back(example_idx);
    }
    if (refinement.unique_values) {
      unique_items[refinement_idx].push_back({value, example_idx});
    } else {
      const int bin_idx = NumericalBinIndex(refinement.boundaries, value);
      labels.AddExample(example_idx, value,
                        &(*results)[refinement_idx][bin_idx]);
    }
  }

  // Aggregate the unique values.
  for (int refinement_idx = 0; refinement_idx < num_refinements;
       refinement_idx++) {
    if (!refinements[refinement_idx].unique_values) {
      continue;
    }
    auto& items = unique_items[refinement_idx];
    std::stable_sort(
        items.begin(), items.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    auto& result = (*results)[refinement_idx];
    for (size_t i = 0; i < items.size(); i++) {
      if (i == 0 || items[i].first != items[i - 1].first) {
        result.emplace_back();
      }
      labels.AddExample(items[i].second, items[i].first, &result.back());
    }
  }
  return absl::OkStatus();
}

std::vector<BinStats> MergeUniqueValueBins(
    const std::vector<std::vector<BinStats>>& per_worker) {
  std::vector<BinStats> all;
  for (const auto& bins : per_worker) {
    for (const auto& bin : bins) {
      if (!bin.empty()) {
        all.push_back(bin);
      }
    }
  }
  std::stable_sort(all.begin(), all.end(),
                   [](const BinStats& a, const BinStats& b) {
                     return a.min_value < b.min_value;
                   });
  std::vector<BinStats> merged;
  for (const auto& bin : all) {
    if (!merged.empty() && merged.back().min_value == bin.min_value) {
      merged.back().Add(bin);
    } else {
      merged.push_back(bin);
    }
  }
  return merged;
}

std::vector<float> EqualWidthSubBinBoundaries(const float min_value,
                                              const float max_value,
                                              const int num_sub_bins) {
  std::vector<float> boundaries;
  if (!(min_value < max_value)) {
    return boundaries;
  }
  // Isolate the minimum value.
  boundaries.push_back(
      std::nextafter(min_value, std::numeric_limits<float>::infinity()));
  for (int i = 1; i < num_sub_bins; i++) {
    const float boundary = static_cast<float>(
        static_cast<double>(min_value) +
        (static_cast<double>(max_value) - static_cast<double>(min_value)) * i /
            num_sub_bins);
    if (boundary > min_value && boundary <= max_value) {
      boundaries.push_back(boundary);
    }
  }
  // Isolate the maximum value.
  boundaries.push_back(max_value);
  std::sort(boundaries.begin(), boundaries.end());
  boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                   boundaries.end());
  return boundaries;
}

// Manager side
// ============

NodeSplitSearch::NodeSplitSearch(const SplitCriterion* criterion,
                                 const BinStats& parent,
                                 const ExampleIndex min_examples)
    : criterion_(criterion), parent_(parent), min_examples_(min_examples) {}

void NodeSplitSearch::AddNumericalFeature(const int feature,
                                          const float na_replacement,
                                          const std::vector<BinStats>& bins) {
  NumericalFeature item;
  item.feature = feature;
  item.na_replacement = na_replacement;
  for (const auto& bin : bins) {
    if (!bin.empty()) {
      item.bins.push_back(bin);
    }
  }
  item.pruned.assign(item.bins.size(), false);
  numerical_.push_back(std::move(item));
}

absl::Status NodeSplitSearch::AddCategoricalFeature(
    const int feature, const int na_replacement,
    const std::vector<BinStats>& per_value) {
  Split split;
  ASSIGN_OR_RETURN(const bool found, criterion_->FindBestCategoricalSplit(
                                         feature, na_replacement, parent_,
                                         per_value, min_examples_, &split));
  if (found) {
    Candidate candidate;
    candidate.score = split.condition.split_score();
    candidate.feature = feature;
    candidate.is_numerical = false;
    candidate.split = std::move(split);
    categorical_candidates_.push_back(std::move(candidate));
  }
  return absl::OkStatus();
}

absl::Status NodeSplitSearch::AddBooleanFeature(
    const int feature, const bool na_replacement,
    const std::vector<BinStats>& per_value) {
  Split split;
  ASSIGN_OR_RETURN(const bool found, criterion_->FindBestBooleanSplit(
                                         feature, na_replacement, parent_,
                                         per_value, min_examples_, &split));
  if (found) {
    Candidate candidate;
    candidate.score = split.condition.split_score();
    candidate.feature = feature;
    candidate.is_numerical = false;
    candidate.split = std::move(split);
    categorical_candidates_.push_back(std::move(candidate));
  }
  return absl::OkStatus();
}

void NodeSplitSearch::Consider(const Candidate& candidate) {
  if (!best_.has_value() || candidate.score > best_->score ||
      (candidate.score == best_->score && candidate.feature < best_->feature)) {
    best_ = candidate;
  }
}

std::vector<NodeSplitSearch::LiveBin> NodeSplitSearch::UpdateAndListLiveBins() {
  // Update the incumbent.
  best_.reset();
  for (const auto& candidate : categorical_candidates_) {
    Consider(candidate);
  }
  const double min_score = criterion_->MinimumScore(parent_);
  const ExampleIndex num_examples = parent_.count;
  for (const auto& item : numerical_) {
    BinStats prefix;
    for (size_t bin_idx = 0; bin_idx + 1 < item.bins.size(); bin_idx++) {
      prefix.Add(item.bins[bin_idx]);
      if (prefix.count < min_examples_ ||
          num_examples - prefix.count < min_examples_) {
        continue;
      }
      BinStats pos = parent_;
      pos.SubSums(prefix);
      if (!criterion_->IsValidSplit(prefix, pos)) {
        continue;
      }
      const double score = criterion_->Score(parent_, prefix, pos);
      if (!(score > min_score)) {
        continue;
      }
      // Only consider the candidate if it is better than the incumbent (avoid
      // the copy of the statistics).
      if (best_.has_value() &&
          (score < best_->score ||
           (score == best_->score && item.feature >= best_->feature))) {
        continue;
      }
      Candidate candidate;
      candidate.score = score;
      candidate.feature = item.feature;
      candidate.is_numerical = true;
      candidate.neg = prefix;
      candidate.threshold = decision_tree::MidThreshold(
          item.bins[bin_idx].max_value, item.bins[bin_idx + 1].min_value);
      candidate.na_replacement = item.na_replacement;
      Consider(candidate);
    }
  }

  // List the live bins.
  const double threshold = best_.has_value() ? best_->score : min_score;
  std::vector<LiveBin> live_bins;
  for (int slot = 0; slot < numerical_.size(); slot++) {
    auto& item = numerical_[slot];
    BinStats prefix;
    for (size_t bin_idx = 0; bin_idx < item.bins.size(); bin_idx++) {
      const auto& bin = item.bins[bin_idx];
      if (!item.pruned[bin_idx]) {
        bool live = true;
        if (!(bin.min_value < bin.max_value)) {
          // A single unique value: No split inside the bin.
          live = false;
        } else {
          // Is there a split position inside the bin satisfying the minimum
          // number of examples?
          const int64_t lowest =
              std::max<int64_t>(prefix.count + 1, min_examples_);
          const int64_t highest = std::min<int64_t>(
              prefix.count + bin.count - 1,
              static_cast<int64_t>(num_examples) - min_examples_);
          if (lowest > highest) {
            live = false;
          }
        }
        double upper_bound = std::numeric_limits<double>::infinity();
        if (live) {
          const auto bound = criterion_->UpperBound(parent_, prefix, bin);
          upper_bound = bound.value;
          const double tolerance = kBoundRelativeTolerance *
                                   std::max(bound.scale, std::abs(threshold));
          if (!(bound.value > threshold - tolerance)) {
            live = false;
          }
        }
        if (live) {
          live_bins.push_back({slot, item.feature, static_cast<int>(bin_idx),
                               bin, upper_bound});
        } else {
          item.pruned[bin_idx] = true;
        }
      }
      prefix.Add(bin);
    }
  }
  return live_bins;
}

absl::Status NodeSplitSearch::ReplaceBin(const int feature_slot,
                                         const int bin_idx,
                                         std::vector<BinStats> sub_bins) {
  if (feature_slot < 0 || feature_slot >= numerical_.size()) {
    return absl::InternalError("Unexpected feature slot");
  }
  auto& item = numerical_[feature_slot];
  if (bin_idx < 0 || bin_idx >= item.bins.size()) {
    return absl::InternalError("Unexpected bin index");
  }
  // Check that the sub-bins cover the bin.
  int64_t count = 0;
  for (const auto& sub_bin : sub_bins) {
    count += sub_bin.count;
  }
  if (count != item.bins[bin_idx].count) {
    return absl::InternalError(absl::Substitute(
        "The refined bins of feature $0 contain $1 examples instead of $2.",
        item.feature, count, item.bins[bin_idx].count));
  }
  item.pending.push_back({bin_idx, std::move(sub_bins)});
  return absl::OkStatus();
}

absl::Status NodeSplitSearch::ApplyReplacements() {
  for (auto& item : numerical_) {
    if (item.pending.empty()) {
      continue;
    }
    std::sort(item.pending.begin(), item.pending.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<BinStats> new_bins;
    std::vector<bool> new_pruned;
    size_t pending_idx = 0;
    for (size_t bin_idx = 0; bin_idx < item.bins.size(); bin_idx++) {
      if (pending_idx < item.pending.size() &&
          item.pending[pending_idx].first == bin_idx) {
        for (auto& sub_bin : item.pending[pending_idx].second) {
          if (!sub_bin.empty()) {
            new_bins.push_back(sub_bin);
            new_pruned.push_back(false);
          }
        }
        pending_idx++;
        if (pending_idx < item.pending.size() &&
            item.pending[pending_idx].first == bin_idx) {
          return absl::InternalError("Bin replaced twice");
        }
      } else {
        new_bins.push_back(item.bins[bin_idx]);
        new_pruned.push_back(item.pruned[bin_idx]);
      }
    }
    // Check the ordering of the bins.
    for (size_t bin_idx = 1; bin_idx < new_bins.size(); bin_idx++) {
      if (!(new_bins[bin_idx - 1].max_value < new_bins[bin_idx].min_value)) {
        return absl::InternalError(absl::StrCat(
            "Overlapping bins after refinement for feature ", item.feature));
      }
    }
    item.bins = std::move(new_bins);
    item.pruned = std::move(new_pruned);
    item.pending.clear();
  }
  return absl::OkStatus();
}

double NodeSplitSearch::best_score() const {
  return best_.has_value() ? best_->score : criterion_->MinimumScore(parent_);
}

absl::StatusOr<Split> NodeSplitSearch::BestSplit() const {
  if (!best_.has_value()) {
    return absl::InternalError("No split found");
  }
  if (!best_->is_numerical) {
    return best_->split;
  }
  Split split;
  auto& condition = split.condition;
  condition.mutable_condition()->mutable_higher_condition()->set_threshold(
      best_->threshold);
  condition.set_na_value(best_->na_replacement >= best_->threshold);
  condition.set_attribute(best_->feature);
  BinStats pos = parent_;
  pos.SubSums(best_->neg);
  condition.set_num_training_examples_without_weight(parent_.count);
  condition.set_num_training_examples_with_weight(parent_.sum_weights);
  condition.set_num_pos_training_examples_without_weight(pos.count);
  condition.set_num_pos_training_examples_with_weight(pos.sum_weights);
  condition.set_split_score(best_->score);
  BinStatsToLabelStatistics(best_->neg, &split.label_statistics[0]);
  BinStatsToLabelStatistics(pos, &split.label_statistics[1]);
  return split;
}

int64_t NodeSplitSearch::NumBins() const {
  int64_t num_bins = 0;
  for (const auto& item : numerical_) {
    num_bins += item.bins.size();
  }
  return num_bins;
}

}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
