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

// Exact greedy split finding with a branch-and-bound over feature histograms.
//
// This library contains the generic (i.e. independent of the distribution
// framework and learning algorithm) components of the
// HISTOGRAM_BRANCH_AND_BOUND split finding:
//
//   - "BinStats": The statistics of a set of examples (e.g. a histogram bin):
//     count, sums of weights, gradients, squared gradients and hessians, and
//     the range of feature values and gradient/hessian ratios.
//   - "SplitCriterion": How to score a split from the statistics of its two
//     children, and how to upper bound the score of any split inside of a bin.
//   - Worker-side functions to compute histograms (and their refinements) over
//     the examples owned by a worker.
//   - "NodeSplitSearch": Manager-side state of the search of the best split of
//     a single open node.
//
// Setting
// =======
//
// Each example i has a weight w_i, a gradient grad_i and a hessian hess_i. The
// split criterion maps each example to a point (h_i, g_i) with h_i >= 0 and
// defines r_i = g_i / h_i. For a set of examples S, H(S) = sum h_i and G(S) =
// sum g_i. All the supported criteria have the form:
//
//   score(L, R) = f(H_L, G_L) + f(H_R, G_R) - constant
//
// where f is (or is upper bounded by) a jointly convex function. For any split
// inside a bin k (i.e. the left child contains all the bins < k and a subset of
// bin k), the point (H_L, G_L) is contained in the parallelogram with vertices
// A = (H_<k, G_<k), B = A + (H_k, G_k), and the two points where the lines of
// slope r_min and r_max starting at A and B intersect. Since the score is
// convex, its maximum over the parallelogram is reached at one of the four
// vertices. This gives an upper bound of the score of any split inside the bin
// that only depends on (H_<k, G_<k, H_k, G_k, r_min, r_max).
//
// Algorithm (for one node)
// ========================
//
//   1. Compute the histograms (with approximate quantile boundaries).
//   2. The incumbent is the best split among all the bin boundaries (and
//      categorical / boolean features).
//   3. A bin is "live" if it can contain a split strictly better than the
//      incumbent (according to the upper bound).
//   4. Live bins are refined (split into sub-bins, or into unique values if
//      small), and the process is repeated from 2 until no bin is live.
//
// The result is the exact greedy split (same as an exhaustive search over all
// the unique feature values).
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_HISTOGRAM_BNB_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_HISTOGRAM_BNB_H_

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/splitter.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/training.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {

// How the per-example ratio r_i = g_i / h_i is computed.
enum class RatioType {
  // r_i = grad_i (i.e. h_i = w_i, g_i = w_i * grad_i).
  kGradient,
  // r_i = grad_i / hess_i (i.e. h_i = w_i * hess_i, g_i = w_i * grad_i).
  kGradientOverHessian,
};

// Statistics of a set of examples.
struct BinStats {
  // Number of examples.
  int64_t count = 0;
  // Sum of w_i.
  double sum_weights = 0;
  // Sum of w_i * grad_i.
  double sum_gradients = 0;
  // Sum of w_i * grad_i^2.
  double sum_squared_gradients = 0;
  // Sum of w_i * hess_i.
  double sum_hessians = 0;
  // Range of the feature values.
  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();
  // Range of the ratios r_i. Set to [-inf, +inf] if the ratio is not defined
  // for one of the examples (e.g. non-positive hessian).
  float min_ratio = std::numeric_limits<float>::infinity();
  float max_ratio = -std::numeric_limits<float>::infinity();

  bool empty() const { return count == 0; }

  // Adds an example. The products are computed in float precision to match
  // the label statistics accumulation of the non-distributed learners.
  inline void AddExample(const float value, const float gradient,
                         const float hessian, const float weight,
                         const RatioType ratio_type) {
    const float weighted_gradient = gradient * weight;
    const float weighted_hessian = hessian * weight;
    count++;
    sum_weights += weight;
    sum_gradients += weighted_gradient;
    sum_squared_gradients += weighted_gradient * gradient;
    sum_hessians += weighted_hessian;
    if (value < min_value) min_value = value;
    if (value > max_value) max_value = value;
    // The ratio is computed from the exact (g_i, h_i) values accumulated in
    // the sums, and rounded outward to float precision. This guarantees the
    // validity of the bounds.
    const float h =
        ratio_type == RatioType::kGradient ? weight : weighted_hessian;
    if (!(h > 0.f)) {
      if (weighted_gradient != 0.f || h != 0.f) {
        min_ratio = -std::numeric_limits<float>::infinity();
        max_ratio = std::numeric_limits<float>::infinity();
      }
      // Examples with g_i=h_i=0 have no impact on the statistics.
      return;
    }
    const double ratio = static_cast<double>(weighted_gradient) / h;
    float lower = static_cast<float>(ratio);
    float upper = lower;
    if (lower > ratio) {
      lower = std::nextafter(lower, -std::numeric_limits<float>::infinity());
    }
    if (upper < ratio) {
      upper = std::nextafter(upper, std::numeric_limits<float>::infinity());
    }
    if (lower < min_ratio) min_ratio = lower;
    if (upper > max_ratio) max_ratio = upper;
  }

  // Adds the statistics (sums, counts and ranges) of another set of examples.
  void Add(const BinStats& other);

  // Subtracts the sums and counts of a subset of examples. The ranges are not
  // modified (and are not valid anymore).
  void SubSums(const BinStats& other);
};

// Converts a BinStats to/from the label statistics of a gradient (with hessian)
// regression.
void BinStatsToLabelStatistics(const BinStats& src,
                               decision_tree::proto::LabelStatistics* dst);
absl::StatusOr<BinStats> LabelStatisticsToBinStats(
    const decision_tree::proto::LabelStatistics& src);

// Proto (de-)serialization of a list of bins.
void AppendBinToProto(const BinStats& bin, proto::BinStatistics* dst);
int NumBinsInProto(const proto::BinStatistics& src);
BinStats BinFromProto(const proto::BinStatistics& src, int bin_idx);
absl::StatusOr<std::vector<BinStats>> BinsFromProto(
    const proto::BinStatistics& src);

// Upper bound of a split score.
struct ScoreBound {
  double value;
  // Magnitude of the terms used to compute the bound. Used to estimate the
  // numerical error.
  double scale;
};

// Scores a split from the statistics of its two children.
class SplitCriterion {
 public:
  virtual ~SplitCriterion() = default;

  virtual RatioType ratio_type() const = 0;

  // Coordinates of a set of examples in the (H, G) plane.
  virtual double H(const BinStats& stats) const = 0;
  double G(const BinStats& stats) const { return stats.sum_gradients; }

  // Tests if a split is valid (e.g. minimum sum of hessian in each child).
  virtual bool IsValidSplit(const BinStats& neg, const BinStats& pos) const {
    return true;
  }

  // Score of the split with negative (x < threshold) and positive children.
  // "parent" is the union of "neg" and "pos".
  virtual double Score(const BinStats& parent, const BinStats& neg,
                       const BinStats& pos) const = 0;

  // A split is only accepted if its score is strictly greater than this value.
  virtual double MinimumScore(const BinStats& parent) const = 0;

  // Upper bound of the score of any split where the negative child contains
  // "prefix" and a subset of "bin". The bound ignores the example count and
  // "IsValidSplit" constraints (which only reduce the set of possible splits).
  ScoreBound UpperBound(const BinStats& parent, const BinStats& prefix,
                        const BinStats& bin) const;

  // Finds the best split of a categorical feature (CART algorithm) given the
  // statistics of each categorical value. Returns true and sets "split" if a
  // valid split with score > MinimumScore is found.
  virtual absl::StatusOr<bool> FindBestCategoricalSplit(
      int feature, int na_replacement, const BinStats& parent,
      const std::vector<BinStats>& per_value, ExampleIndex min_examples,
      Split* split) const = 0;

  // Finds the best split of a boolean feature given the statistics of the
  // "false" and "true" values.
  virtual absl::StatusOr<bool> FindBestBooleanSplit(
      int feature, bool na_replacement, const BinStats& parent,
      const std::vector<BinStats>& per_value, ExampleIndex min_examples,
      Split* split) const = 0;

 protected:
  // Jointly convex function of (h, g) such that, for any split,
  //   Score(parent, L, R) <= ConvexTerm(parent, H(L), G(L)) +
  //     ConvexTerm(parent, H(R), G(R)) - ConvexOffset(parent).
  virtual double ConvexTerm(const BinStats& parent, double h,
                            double g) const = 0;
  virtual double ConvexOffset(const BinStats& parent) const = 0;
};

// Weighted variance reduction of the gradient. Same score as the
// non-distributed GBT learner with "use_hessian_gain=false":
//
//   score = (VarTimesW(parent) - VarTimesW(neg) - VarTimesW(pos)) / W(parent)
//
// with h_i = w_i, g_i = w_i * grad_i.
class VarianceReductionCriterion : public SplitCriterion {
 public:
  RatioType ratio_type() const override { return RatioType::kGradient; }
  double H(const BinStats& stats) const override { return stats.sum_weights; }
  double Score(const BinStats& parent, const BinStats& neg,
               const BinStats& pos) const override;
  double MinimumScore(const BinStats& parent) const override { return 0; }
  absl::StatusOr<bool> FindBestCategoricalSplit(
      int feature, int na_replacement, const BinStats& parent,
      const std::vector<BinStats>& per_value, ExampleIndex min_examples,
      Split* split) const override;
  absl::StatusOr<bool> FindBestBooleanSplit(
      int feature, bool na_replacement, const BinStats& parent,
      const std::vector<BinStats>& per_value, ExampleIndex min_examples,
      Split* split) const override;

 protected:
  double ConvexTerm(const BinStats& parent, double h, double g) const override;
  double ConvexOffset(const BinStats& parent) const override;
};

// Newton gain (a.k.a. hessian gain). Same score as the non-distributed GBT
// learner with "use_hessian_gain=true":
//
//   score = F(neg) + F(pos) - [F(parent) if subtract_parent]
//   F(S) = l1_threshold(G(S), l1)^2 / (max(H(S), 0.001) + l2)
//
// with h_i = w_i * hess_i, g_i = w_i * grad_i.
//
// The upper bound uses F without the hessian clamping, which is convex and
// greater or equal to F.
class HessianGainCriterion : public SplitCriterion {
 public:
  struct Options {
    double l1 = 0;
    double l2_numerical = 0;
    double l2_categorical = 0;
    double min_sum_hessian_in_leaf = 0;
    bool subtract_parent = false;
  };

  explicit HessianGainCriterion(const Options& options) : options_(options) {}

  RatioType ratio_type() const override {
    return RatioType::kGradientOverHessian;
  }
  double H(const BinStats& stats) const override { return stats.sum_hessians; }
  bool IsValidSplit(const BinStats& neg, const BinStats& pos) const override;
  double Score(const BinStats& parent, const BinStats& neg,
               const BinStats& pos) const override;
  double MinimumScore(const BinStats& parent) const override;
  absl::StatusOr<bool> FindBestCategoricalSplit(
      int feature, int na_replacement, const BinStats& parent,
      const std::vector<BinStats>& per_value, ExampleIndex min_examples,
      Split* split) const override;
  absl::StatusOr<bool> FindBestBooleanSplit(
      int feature, bool na_replacement, const BinStats& parent,
      const std::vector<BinStats>& per_value, ExampleIndex min_examples,
      Split* split) const override;

 protected:
  double ConvexTerm(const BinStats& parent, double h, double g) const override;
  double ConvexOffset(const BinStats& parent) const override;

 private:
  double ParentScore(const BinStats& parent) const;

  Options options_;
};

// Worker side
// ===========

// Label data of the examples owned by a worker for one weak model.
struct LabelData {
  absl::Span<const float> gradients;
  absl::Span<const float> hessians;
  // Empty if the examples are not weighted.
  absl::Span<const float> weights;
  RatioType ratio_type = RatioType::kGradient;

  inline void AddExample(const UnsignedExampleIdx example_idx,
                         const float value, BinStats* bin) const {
    bin->AddExample(value, gradients[example_idx], hessians[example_idx],
                    weights.empty() ? 1.f : weights[example_idx], ratio_type);
  }
};

// Index of the histogram bin of a numerical value. "boundaries" is sorted.
// Value "x" belongs to bin "i" where "i" is the number of boundaries <= x.
int NumericalBinIndex(absl::Span<const float> boundaries, float value);

// Computes the histograms of a feature for all the open nodes in a single pass
// over the examples.
//
// Args:
//   values: Feature values (indexed by example).
//   example_to_node: Open node of each example.
//   node_to_slot: For each open node, index in "histograms" or -1 if the node
//     does not need this histogram.
//   labels: Label data.
//   histograms: Output histograms. Each histogram should be allocated with the
//     right number of bins (initialized to empty) beforehand.
absl::Status FillNumericalHistograms(
    absl::Span<const float> values, absl::Span<const float> boundaries,
    const ExampleToNodeMap& example_to_node,
    const std::vector<int>& node_to_slot, const LabelData& labels,
    std::vector<std::vector<BinStats>>* histograms);

absl::Status FillCategoricalHistograms(
    absl::Span<const int32_t> values, int num_values, int na_replacement,
    const ExampleToNodeMap& example_to_node,
    const std::vector<int>& node_to_slot, const LabelData& labels,
    std::vector<std::vector<BinStats>>* histograms);

absl::Status FillBooleanHistograms(
    absl::Span<const int8_t> values, bool na_replacement,
    const ExampleToNodeMap& example_to_node,
    const std::vector<int>& node_to_slot, const LabelData& labels,
    std::vector<std::vector<BinStats>>* histograms);

// List the examples in each open node.
std::vector<std::vector<UnsignedExampleIdx>> ListExamplesPerNode(
    const ExampleToNodeMap& example_to_node, int num_nodes);

// Refinement of one bin of a numerical feature.
struct BinRefinement {
  // Examples with value in [min_value, max_value] are part of the bin.
  float min_value;
  float max_value;
  // If true, aggregate the statistics per unique value. Otherwise, split the
  // bin according to "boundaries".
  bool unique_values = false;
  std::vector<float> boundaries;
};

// Computes the refined statistics of a set of (disjoint) bins of the same
// feature and node.
//
// Args:
//   values: Feature values (indexed by example).
//   candidates: Examples that can be in the bins (e.g. the examples in the
//     node). The other examples are not tested.
//   refinements: The bins to refine. The [min_value, max_value] ranges should
//     not overlap.
//   labels: Label data.
//   results: One output for each refinement. For "unique_values", one
//     (non-empty) bin per unique value sorted by value. Otherwise, one bin per
//     sub-bin (boundaries.size()+1 bins).
//   survivors: If non-null, the subset of "candidates" contained in at least
//     one of the refined bins. Used to skip examples in the next refinement
//     rounds.
absl::Status RefineBins(absl::Span<const float> values,
                        absl::Span<const UnsignedExampleIdx> candidates,
                        const std::vector<BinRefinement>& refinements,
                        const LabelData& labels,
                        std::vector<std::vector<BinStats>>* results,
                        std::vector<UnsignedExampleIdx>* survivors);

// Merges the "unique values" refinements of several workers (each sorted by
// value) into a single list of bins (one per unique value, sorted by value).
std::vector<BinStats> MergeUniqueValueBins(
    const std::vector<std::vector<BinStats>>& per_worker);

// Boundaries of "num_sub_bins" equal-width sub-bins of the range [min_value,
// max_value] (with min_value < max_value). The minimum and maximum values are
// isolated in their own sub-bin. This guarantees that each refinement makes
// progress.
std::vector<float> EqualWidthSubBinBoundaries(float min_value, float max_value,
                                              int num_sub_bins);

// Manager side
// ============

// Search for the best split of one open node.
class NodeSplitSearch {
 public:
  // A bin that can contain a better split than the incumbent.
  struct LiveBin {
    // Index of the numerical feature in this search (i.e. "feature_slot").
    int feature_slot;
    // Attribute index.
    int feature;
    // Index of the bin in the feature histogram.
    int bin_idx;
    // Statistics of the bin.
    BinStats stats;
    // Upper bound of the score of the splits inside the bin.
    double upper_bound;
  };

  NodeSplitSearch(const SplitCriterion* criterion, const BinStats& parent,
                  ExampleIndex min_examples);

  // Sets the initial histogram of a numerical feature. "bins" are sorted by
  // feature value. Empty bins are removed.
  void AddNumericalFeature(int feature, float na_replacement,
                           const std::vector<BinStats>& bins);

  // Evaluates the splits of a categorical / boolean feature.
  absl::Status AddCategoricalFeature(int feature, int na_replacement,
                                     const std::vector<BinStats>& per_value);
  absl::Status AddBooleanFeature(int feature, bool na_replacement,
                                 const std::vector<BinStats>& per_value);

  // Updates the incumbent with all the numerical bin boundaries, and computes
  // the live bins. Bins that are not live are pruned (permanently).
  std::vector<LiveBin> UpdateAndListLiveBins();

  // Replaces a bin of a numerical feature by a list of sub-bins (sorted by
  // value) covering the same examples. Empty sub-bins are removed. All the
  // replacements of a round should be submitted before calling
  // "UpdateAndListLiveBins".
  absl::Status ReplaceBin(int feature_slot, int bin_idx,
                          std::vector<BinStats> sub_bins);

  // Applies all the replacements submitted with "ReplaceBin".
  absl::Status ApplyReplacements();

  // True if a valid split was found.
  bool has_split() const { return best_.has_value(); }

  // Score of the best split found so far.
  double best_score() const;

  // Best split found so far. Fails if "has_split()" is false.
  absl::StatusOr<Split> BestSplit() const;

  const BinStats& parent() const { return parent_; }

  // Number of bins (non-empty) currently in the histograms. For monitoring.
  int64_t NumBins() const;

 private:
  struct NumericalFeature {
    int feature;
    float na_replacement;
    std::vector<BinStats> bins;
    // If true, the bin cannot contain a split better than the incumbent.
    std::vector<bool> pruned;
    // Pending replacements (bin_idx -> sub-bins).
    std::vector<std::pair<int, std::vector<BinStats>>> pending;
  };

  struct Candidate {
    double score;
    int feature;
    // For numerical features: Statistics of the negative child and threshold.
    bool is_numerical;
    BinStats neg;
    float threshold;
    float na_replacement;
    // For categorical and boolean features: The split.
    Split split;
  };

  // Tests a candidate (ties broken by smallest feature index).
  void Consider(const Candidate& candidate);

  const SplitCriterion* criterion_;
  BinStats parent_;
  ExampleIndex min_examples_;
  std::vector<NumericalFeature> numerical_;
  // Best split of each categorical and boolean feature.
  std::vector<Candidate> categorical_candidates_;
  std::optional<Candidate> best_;
};

}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_HISTOGRAM_BNB_H_
