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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/quantile_sketch.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/training.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {

QuantileSummary::QuantileSummary(const proto::QuantileSummary& src) {
  const int n = src.values_size();
  entries_.resize(n);
  for (int i = 0; i < n; i++) {
    entries_[i] = {src.values(i), src.min_ranks(i), src.max_ranks(i),
                   src.weights(i)};
  }
  total_weight_ = src.total_weight();
}

void QuantileSummary::ToProto(proto::QuantileSummary* dst) const {
  dst->Clear();
  dst->mutable_values()->Reserve(entries_.size());
  dst->mutable_min_ranks()->Reserve(entries_.size());
  dst->mutable_max_ranks()->Reserve(entries_.size());
  dst->mutable_weights()->Reserve(entries_.size());
  for (const auto& entry : entries_) {
    dst->add_values(entry.value);
    dst->add_min_ranks(entry.min_rank);
    dst->add_max_ranks(entry.max_rank);
    dst->add_weights(entry.weight);
  }
  dst->set_total_weight(total_weight_);
}

absl::StatusOr<QuantileSummary> QuantileSummary::FromValues(
    absl::Span<const float> values, absl::Span<const float> weights) {
  if (!weights.empty() && weights.size() != values.size()) {
    return absl::InvalidArgumentError(
        "\"weights\" should be empty or of the same size as \"values\"");
  }
  std::vector<std::pair<float, float>> items;
  items.reserve(values.size());
  for (size_t i = 0; i < values.size(); i++) {
    const float value = values[i];
    const float weight = weights.empty() ? 1.f : weights[i];
    if (std::isnan(value) || !(weight > 0.f)) {
      continue;
    }
    items.push_back({value, weight});
  }
  std::sort(items.begin(), items.end());

  QuantileSummary summary;
  double cumulative = 0;
  size_t i = 0;
  while (i < items.size()) {
    const float value = items[i].first;
    double weight = 0;
    while (i < items.size() && items[i].first == value) {
      weight += items[i].second;
      i++;
    }
    summary.entries_.push_back(
        {value, cumulative, cumulative + weight, weight});
    cumulative += weight;
  }
  summary.total_weight_ = cumulative;
  return summary;
}

void QuantileSummary::Contribution(const float value, double* min_rank,
                                   double* max_rank, double* weight) const {
  const auto it = std::lower_bound(
      entries_.begin(), entries_.end(), value,
      [](const Entry& entry, const float v) { return entry.value < v; });
  if (it != entries_.end() && it->value == value) {
    *min_rank = it->min_rank;
    *max_rank = it->max_rank;
    *weight = it->weight;
    return;
  }
  // "value" is not an entry of the summary.
  *weight = 0;
  if (it == entries_.begin()) {
    *min_rank = 0;
  } else {
    const auto& prev = *(it - 1);
    *min_rank = prev.min_rank + prev.weight;
  }
  if (it == entries_.end()) {
    *max_rank = total_weight_;
  } else {
    *max_rank = it->max_rank - it->weight;
  }
}

QuantileSummary QuantileSummary::Merge(const QuantileSummary& a,
                                       const QuantileSummary& b) {
  if (a.empty()) {
    return b;
  }
  if (b.empty()) {
    return a;
  }
  // Union of the values.
  std::vector<float> values;
  values.reserve(a.entries_.size() + b.entries_.size());
  for (const auto& entry : a.entries_) {
    values.push_back(entry.value);
  }
  for (const auto& entry : b.entries_) {
    values.push_back(entry.value);
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());

  QuantileSummary merged;
  merged.entries_.reserve(values.size());
  for (const float value : values) {
    double a_min, a_max, a_w, b_min, b_max, b_w;
    a.Contribution(value, &a_min, &a_max, &a_w);
    b.Contribution(value, &b_min, &b_max, &b_w);
    merged.entries_.push_back({value, a_min + b_min, a_max + b_max, a_w + b_w});
  }
  merged.total_weight_ = a.total_weight_ + b.total_weight_;
  return merged;
}

double QuantileSummary::WorstCaseRankError(const int entry_idx,
                                           const double target_rank) const {
  const auto& entry = entries_[entry_idx];
  return std::max({0., (entry.max_rank - entry.weight) - target_rank,
                   target_rank - (entry.min_rank + entry.weight)});
}

void QuantileSummary::Prune(int max_entries) {
  max_entries = std::max(max_entries, 2);
  const int n = entries_.size();
  if (n <= max_entries) {
    return;
  }
  std::vector<bool> keep(n, false);
  keep.front() = true;
  keep.back() = true;
  // The rank bounds are non-decreasing with the entry index. Therefore, the
  // worst-case rank error is (weakly) unimodal in the entry index, and the
  // best entry index is non-decreasing with the target rank.
  int cursor = 0;
  for (int i = 1; i < max_entries - 1; i++) {
    const double target = total_weight_ * i / (max_entries - 1);
    while (cursor + 1 < n && WorstCaseRankError(cursor + 1, target) <
                                 WorstCaseRankError(cursor, target)) {
      cursor++;
    }
    keep[cursor] = true;
  }
  std::vector<Entry> pruned;
  pruned.reserve(max_entries);
  for (int i = 0; i < n; i++) {
    if (keep[i]) {
      pruned.push_back(entries_[i]);
    }
  }
  entries_ = std::move(pruned);
}

const QuantileSummary::Entry& QuantileSummary::Query(const double q) const {
  DCHECK(!entries_.empty());
  const double target = q * total_weight_;
  int best_idx = 0;
  double best_error = WorstCaseRankError(0, target);
  for (int i = 1; i < entries_.size(); i++) {
    const double error = WorstCaseRankError(i, target);
    if (error < best_error) {
      best_error = error;
      best_idx = i;
    }
  }
  return entries_[best_idx];
}

absl::StatusOr<QuantileSummary> BuildStreamingSummary(
    absl::Span<const float> values, absl::Span<const float> weights,
    const StreamingSummaryOptions& options, utils::RandomEngine* random) {
  if (!weights.empty() && weights.size() != values.size()) {
    return absl::InvalidArgumentError(
        "\"weights\" should be empty or of the same size as \"values\"");
  }
  const size_t num_values = values.size();
  const int chunk_size = std::max(1, options.chunk_size);

  // Select the values to process (selection sampling, Knuth's Algorithm S).
  // The relative order of the values is preserved.
  size_t num_selected = num_values;
  if (options.max_examples >= 0 &&
      static_cast<size_t>(options.max_examples) < num_values) {
    num_selected = options.max_examples;
  }
  std::uniform_real_distribution<double> unif01;
  size_t remaining_to_select = num_selected;
  size_t next_idx = 0;
  const auto next_selected_index = [&]() -> std::optional<size_t> {
    while (next_idx < num_values && remaining_to_select > 0) {
      const size_t idx = next_idx++;
      if (num_selected == num_values ||
          unif01(*random) * (num_values - idx) < remaining_to_select) {
        remaining_to_select--;
        return idx;
      }
    }
    return std::nullopt;
  };

  // Binary counter: levels[i] summarizes 2^i chunks.
  std::vector<std::optional<QuantileSummary>> levels;
  std::vector<float> chunk_values;
  std::vector<float> chunk_weights;
  chunk_values.reserve(chunk_size);
  if (!weights.empty()) {
    chunk_weights.reserve(chunk_size);
  }

  bool done = false;
  bool processed_one_chunk = false;
  while (!done) {
    chunk_values.clear();
    chunk_weights.clear();
    while (chunk_values.size() < chunk_size) {
      const auto idx = next_selected_index();
      if (!idx.has_value()) {
        done = true;
        break;
      }
      chunk_values.push_back(values[*idx]);
      if (!weights.empty()) {
        chunk_weights.push_back(weights[*idx]);
      }
    }
    if (chunk_values.empty()) {
      break;
    }
    ASSIGN_OR_RETURN(auto current,
                     QuantileSummary::FromValues(chunk_values, chunk_weights));
    current.Prune(options.max_entries);
    size_t level = 0;
    while (level < levels.size() && levels[level].has_value()) {
      current = QuantileSummary::Merge(current, *levels[level]);
      current.Prune(options.max_entries);
      levels[level].reset();
      level++;
    }
    if (level == levels.size()) {
      levels.emplace_back();
    }
    levels[level] = std::move(current);
    processed_one_chunk = true;

    if (processed_one_chunk && options.deadline.has_value() &&
        absl::Now() >= *options.deadline) {
      break;
    }
  }

  QuantileSummary result;
  for (auto& level : levels) {
    if (level.has_value()) {
      result = QuantileSummary::Merge(result, *level);
      result.Prune(options.max_entries);
    }
  }
  return result;
}

std::vector<float> ExtractBinBoundaries(const QuantileSummary& summary,
                                        const int num_bins) {
  std::vector<float> boundaries;
  if (summary.empty() || num_bins <= 1) {
    return boundaries;
  }
  std::vector<float> quantiles;
  quantiles.reserve(num_bins - 1);
  for (int i = 1; i < num_bins; i++) {
    quantiles.push_back(summary.Query(static_cast<double>(i) / num_bins).value);
  }
  std::sort(quantiles.begin(), quantiles.end());
  boundaries = quantiles;
  // Isolate the values spanning several quantiles in their own bin.
  for (size_t i = 1; i < quantiles.size(); i++) {
    if (quantiles[i] == quantiles[i - 1]) {
      boundaries.push_back(
          std::nextafter(quantiles[i], std::numeric_limits<float>::infinity()));
    }
  }
  std::sort(boundaries.begin(), boundaries.end());
  boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                   boundaries.end());
  return boundaries;
}

}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
