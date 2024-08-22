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

// Simple computation and display of histograms on integer values.
//
// Usage example:
//   std::vector<int> values = ...;
//   const auto histogram = Histogram::MakeUniform(values);
//   LOG(INFO) << "\n" << histogram.ToString();
//
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_HISTOGRAM_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_HISTOGRAM_H_

#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/utils/accurate_sum.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace histogram {

template <typename T>
class Histogram {
 public:
  // Empty histogram.
  Histogram() {}

  // Histogram from data.
  static Histogram MakeUniform(const std::vector<T>& values,
                               size_t max_bins = 10,
                               const std::vector<float>& weights = {});

  std::string ToString() const;

  const std::vector<T>& bounds() const { return bounds_; }
  const std::vector<double>& counts() const { return counts_; }
  const T minimum() const { return minimum_; }
  const T maximum() const { return maximum_; }

 private:
  std::vector<double> counts_;
  std::vector<T> bounds_;
  T minimum_ = 0;
  T maximum_ = 0;
  double sum_counts_ = 0;
  double sum_count_ignored_ = 0;
  double mean_ = 0;
  double sd_ = 0;
};

// Number of characters to display a value.
template <typename T>
int NumCharacters(T value) {
  return absl::StrCat(value).size();
}

template <typename T>
Histogram<T> Histogram<T>::MakeUniform(const std::vector<T>& values,
                                       size_t max_bins,
                                       const std::vector<float>& weights) {
  Histogram<T> hist;

  DCHECK(weights.empty() || weights.size() == values.size());

  // Empty histogram.
  if (values.empty()) {
    return hist;
  }

  // Determine the number of bounds.
  const auto min_max = std::minmax_element(values.begin(), values.end());
  hist.minimum_ = *min_max.first;
  hist.maximum_ = *min_max.second;

  size_t num_bins = max_bins;
  if constexpr (std::numeric_limits<T>::is_integer) {
    num_bins = std::min(num_bins,
                        static_cast<size_t>(1 + hist.maximum_ - hist.minimum_));
  }
  if (num_bins <= 0) {
    num_bins = 1;
  }
  hist.counts_.resize(num_bins);
  hist.bounds_.resize(num_bins + 1);

  // Set bounds uniformly.
  const auto range = hist.maximum_ - hist.minimum_;
  for (int bin_idx = 0; bin_idx < num_bins; bin_idx++) {
    const T bound = hist.minimum_ + (range + 1) * bin_idx / num_bins;
    hist.bounds_[bin_idx] = bound;
  }
  hist.bounds_.back() = hist.maximum_;

  // Fill bins.

  // Accumulators for the mean and standard deviation.
  AccurateSum sum;
  AccurateSum sum_square;
  AccurateSum sum_weights;

  for (size_t value_idx = 0; value_idx < values.size(); value_idx++) {
    const auto value = values[value_idx];
    const float weight = weights.empty() ? 1.0 : weights[value_idx];

    const double double_value = static_cast<double>(value) * weight;
    sum.Add(double_value);
    sum_square.Add(double_value * double_value);
    sum_weights.Add(weight);

    // Find bin with binary search.
    auto it_bound =
        std::upper_bound(hist.bounds_.begin(), hist.bounds_.end(), value);
    // Values outside of the bounds are ignored.
    // The upper bound of the last/top bin is inclusive.
    if (it_bound == hist.bounds_.begin()) {
      hist.sum_count_ignored_ += weight;
      continue;
    }
    if (it_bound == hist.bounds_.end()) {
      if (value == hist.bounds_.back()) {
        it_bound--;
      } else {
        hist.sum_count_ignored_ += weight;
        continue;
      }
    }
    const auto bin_idx = std::distance(hist.bounds_.begin(), it_bound) - 1;

    // Update bin.
    hist.counts_[bin_idx] += weight;
    hist.sum_counts_ += weight;
  }

  // Set statistics.
  hist.mean_ = sum.Sum() / sum_weights.Sum();
  hist.sd_ =
      std::sqrt(sum_square.Sum() / sum_weights.Sum() - hist.mean_ * hist.mean_);

  return hist;
}

template <typename T>
std::string Histogram<T>::ToString() const {
  std::string report;
  absl::SubstituteAndAppend(
      &report,
      "Count: $0 Average: $1 StdDev: $2\nMin: $3 Max: $4 Ignored: $5"
      "\n----------------------------------------------\n",
      sum_counts_, mean_, sd_, minimum_, maximum_, sum_count_ignored_);

  if (counts_.empty()) {
    return report;
  }
  const auto max_count = *std::max_element(counts_.begin(), counts_.end());

  // Number of characters for the different display elements.
  const int print_bar_size = 10;  // Maximum length of the bar.
  const int print_count_size = NumCharacters(max_count);
  int print_bound_size = 1;
  for (const auto bound : bounds_) {
    print_bound_size = std::max(print_bound_size, NumCharacters(bound));
  }

  // Print bins.
  size_t cumulative_count = 0;
  for (int bin_idx = 0; bin_idx < counts_.size(); bin_idx++) {
    const auto count = counts_[bin_idx];
    const char closing_bracket = (bin_idx < counts_.size() - 1) ? ')' : ']';
    int bar_size = 0;
    if (max_count > 0) {
      bar_size =
          std::round(print_bar_size * static_cast<double>(count) / max_count);
    }

    cumulative_count += count;
    double ratio = 0;
    double cumulative_ratio = 0;
    if (sum_counts_ > 0) {
      ratio = 100. * count / sum_counts_;
      cumulative_ratio = 100. * cumulative_count / sum_counts_;
    }

    absl::StrAppendFormat(&report, "[ %*g, %*g%c %*g %6.2f%% %6.2f%%",
                          print_bound_size, bounds_[bin_idx], print_bound_size,
                          bounds_[bin_idx + 1], closing_bracket,
                          print_count_size, count, ratio, cumulative_ratio);
    if (bar_size > 0) {
      absl::StrAppend(&report, " ", std::string(bar_size, '#'));
    }
    absl::StrAppend(&report, "\n");
  }
  return report;
}
// Divide a finite segment "[min_value, max_value]" into a set of "num_bins"
// contiguous non-overlapping equally-sized intervals. Each interval maps to an
// object "Content". Given a number "p \in [min_value, max_value]", the matching
// bin can be retrieved with the "[]" operator.
template <typename Key, typename Content>
class BucketizedContainer {
 public:
  BucketizedContainer(const Key& min_value, const Key& max_value,
                      const int num_bins)
      : content_(num_bins), min_value_(min_value), max_value_(max_value) {
    DCHECK_GT(num_bins, 0);
  }

  Content& operator[](const Key& key) { return content_[Index(key)]; }

  std::vector<Content>& ContentArray() { return content_; }

  int NumBins() const { return content_.size(); }

  Key BinCenter(const int bin_idx) const {
    return min_value_ + bin_idx * (max_value_ - min_value_) / content_.size();
  }

 private:
  int Index(const Key& key) {
    DCHECK_GE(key, min_value_);
    DCHECK_LE(key, max_value_);
    if (key == max_value_) {
      return content_.size() - 1;
    }
    return content_.size() * (key - min_value_) / (max_value_ - min_value_);
  }

  std::vector<Content> content_;
  Key min_value_;
  Key max_value_;
};

}  // namespace histogram
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_UTILS_HISTOGRAM_H_
