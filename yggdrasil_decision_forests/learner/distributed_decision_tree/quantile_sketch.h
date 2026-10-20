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

// Mergeable weighted quantile summary with certified rank bounds.
//
// A summary is a sorted list of entries (v, rmin, rmax, w) such that, for the
// set of summarized (weighted) observations:
//   rmin(v) <= W(x < v), rmax(v) >= W(x <= v), w(v) <= W(x == v).
//
// Summaries can be created from raw values, merged (e.g. across workers), and
// pruned to a maximum number of entries. Each operation keeps the rank bounds
// valid, so the rank error of a query is certified.
//
// The summary is used to compute the initial bin boundaries of the histogram
// branch-and-bound split finding (see "histogram_bnb.h"). Since the boundaries
// only impact the cost of the split finding (and not the found splits), an
// approximate summary is sufficient.
//
// Usage example:
//
//   // On each worker.
//   ASSIGN_OR_RETURN(auto summary, QuantileSummary::FromValues(values, {}));
//   summary.Prune(1024);
//   proto::QuantileSummary message;
//   summary.ToProto(&message);
//
//   // On the manager.
//   QuantileSummary merged;
//   for (const auto& message : messages) {
//     merged = QuantileSummary::Merge(merged, QuantileSummary(message));
//     merged.Prune(1024);
//   }
//   const auto boundaries = ExtractBinBoundaries(merged, /*num_bins=*/64);
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_QUANTILE_SKETCH_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_QUANTILE_SKETCH_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/training.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {

class QuantileSummary {
 public:
  struct Entry {
    float value;
    // Lower bound of the weight of the observations < value.
    double min_rank;
    // Upper bound of the weight of the observations <= value.
    double max_rank;
    // Lower bound of the weight of the observations == value.
    double weight;
  };

  // Empty summary.
  QuantileSummary() = default;

  // Imports a summary from a proto.
  explicit QuantileSummary(const proto::QuantileSummary& src);

  // Exact summary of a list of values. "weights" is either empty (all the
  // weights are 1) or has the same size as "values". NaN values and values
  // with non-positive weights are ignored.
  static absl::StatusOr<QuantileSummary> FromValues(
      absl::Span<const float> values, absl::Span<const float> weights);

  // Merges two summaries. The result summarizes the union of the two sets of
  // observations.
  static QuantileSummary Merge(const QuantileSummary& a,
                               const QuantileSummary& b);

  // Reduces the number of entries to at most "max_entries" (>=2) while keeping
  // the first and last entries. The selected entries minimize the worst-case
  // rank error of the quantiles q=i/(max_entries-1).
  void Prune(int max_entries);

  // Entry minimizing the worst-case rank error for the quantile "q" in [0, 1].
  // The summary should not be empty.
  const Entry& Query(double q) const;

  // Worst-case rank error (expressed in weight) of the entry "entry_idx" for
  // the target rank "target_rank".
  double WorstCaseRankError(int entry_idx, double target_rank) const;

  void ToProto(proto::QuantileSummary* dst) const;

  const std::vector<Entry>& entries() const { return entries_; }
  double total_weight() const { return total_weight_; }
  bool empty() const { return entries_.empty(); }

 private:
  // Bounds contributed by this summary at an arbitrary value "value".
  void Contribution(float value, double* min_rank, double* max_rank,
                    double* weight) const;

  std::vector<Entry> entries_;
  double total_weight_ = 0;
};

// Options to build a summary from a (possibly large) list of values.
struct StreamingSummaryOptions {
  // Maximum number of entries in the summary.
  int max_entries = 1024;
  // Maximum number of values used (uniformly sampled). -1 for no limit.
  int64_t max_examples = -1;
  // If set, stops accumulating values after this deadline (at least one
  // chunk of values is always processed).
  std::optional<absl::Time> deadline;
  // Number of values processed at the same time.
  int chunk_size = 4096;
};

// Builds a summary from a list of values (with possibly sampling and time
// limit). The values are processed by chunks with a binary-counter merging
// scheme so the memory usage and error are controlled. "weights" is either
// empty or has the same size as "values".
absl::StatusOr<QuantileSummary> BuildStreamingSummary(
    absl::Span<const float> values, absl::Span<const float> weights,
    const StreamingSummaryOptions& options, utils::RandomEngine* random);

// Computes the boundaries of "num_bins" bins approximating quantiles. The
// boundaries are sorted and unique. A value "x" belongs to bin "i" where "i" is
// the number of boundaries <= x. Values with a large weight (spanning several
// quantiles) are isolated in their own bin.
std::vector<float> ExtractBinBoundaries(const QuantileSummary& summary,
                                        int num_bins);

}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_QUANTILE_SKETCH_H_
