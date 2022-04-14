/*
 * Copyright 2021 Google LLC.
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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTION_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTION_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/html.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Describe a 1d normal distribution.
class NormalDistributionDouble {
 public:
  // Remove all observations.
  void Clear() {
    sum_ = 0.;
    sum_squares_ = 0.;
    count_ = 0.;
  }

  // Export to proto.
  void Save(proto::NormalDistributionDouble* proto) const;

  // Import form proto.
  void Load(const proto::NormalDistributionDouble& proto);

  // Add one observation.
  template <typename Value>
  inline void Add(const Value value, const Value weight) {
    const auto value_weight = value * weight;
    sum_ += value_weight;
    sum_squares_ += value_weight * value;
    count_ += weight;
  }

  template <typename Value>
  inline void Add(const Value value) {
    sum_ += value;
    sum_squares_ += value * value;
    count_ += Value{1};
  }

  template <typename Value>
  inline void Sub(const Value value, const Value weight) {
    const auto value_weight = value * weight;
    sum_ -= value_weight;
    sum_squares_ -= value_weight * value;
    count_ -= weight;
  }

  template <typename Value>
  inline void Sub(const Value value) {
    sum_ -= value;
    sum_squares_ -= value * value;
    count_ -= Value{1};
  }

  // Add "src" to "this" without normalization i.e. the result will be weighted
  // according to the number of observations in "src" and the initial number of
  // observations in "this".
  void Add(const NormalDistributionDouble& src) {
    sum_ += src.sum_;
    sum_squares_ += src.sum_squares_;
    count_ += src.count_;
  }

  // Subtract "src" from "this" without normalization.
  void Sub(const NormalDistributionDouble& src) {
    sum_ -= src.sum_;
    sum_squares_ -= src.sum_squares_;
    count_ -= src.count_;
  }

  // Mean of the observations.
  double Mean() const {
    if (count_ == 0) return 0.;
    return sum_ / count_;
  }

  // Standard deviation.
  double Std() const { return std::sqrt(Var()); }

  // Variance.
  double Var() const {
    if (count_ == 0) return 0.;
    return sum_squares_ / count_ - (sum_ * sum_) / (count_ * count_);
  }

  // Variance multiplied by the weighted number of observations.
  double VarTimesSumWeights() const {
    return sum_squares_ - (sum_ * sum_) / count_;
  }

  // Number of observations.
  double NumObservations() const { return count_; }

 private:
  double sum_ = 0;
  double sum_squares_ = 0;
  double count_ = 0;
};

// Mean of a normal distribution proto. This function is analog to
// NormalDistributionDouble::Mean on a proto.
inline double Mean(const proto::NormalDistributionDouble& dist) {
  if (dist.count() == 0) {
    return 0.;
  }
  return dist.sum() / dist.count();
}

// Confusion matrix between a binary attribute and a normal distribution.
class BinaryToNormalDistributionDouble {
 public:
  // Add an entry.
  void Add(bool bool_dim, double num_dim, const double weight = 1.) {
    split_[bool_dim].Add(num_dim, weight);
  }

  // Variance i.e. weighted sum of the variance of the normal distributions.
  double FinalVariance() const {
    const double frac_pos = pos().NumObservations() /
                            (pos().NumObservations() + neg().NumObservations());
    return frac_pos * pos().Var() + (1 - frac_pos) * neg().Var();
  }

  double FinalVarianceTimeWeights() const {
    return pos().VarTimesSumWeights() + neg().VarTimesSumWeights();
  }

  // Remove all observations.
  void Clear() {
    mutable_pos()->Clear();
    mutable_neg()->Clear();
  }

  // Total number of observations.
  double NumObservations() const {
    return pos().NumObservations() + neg().NumObservations();
  }

  // Get the normal distribution of the "negative" class.
  NormalDistributionDouble* mutable_neg() { return &split_[0]; }

  // Get the normal distribution  of the "positive" class.
  NormalDistributionDouble* mutable_pos() { return &split_[1]; }

  // Get the normal distribution  of the "negative" class.
  const NormalDistributionDouble& neg() const { return split_[0]; }

  // Get the normal distribution  of the "positive" class.
  const NormalDistributionDouble& pos() const { return split_[1]; }

 private:
  NormalDistributionDouble split_[2];
};

// Represents the (discrete) probability distribution of a random variable with
// natural (i.e. integer greater or equal to zero) support: counts[i]/sum is the
// probability of observation of i.
template <typename T>
class IntegerDistribution {
 public:
  IntegerDistribution() {}

  explicit IntegerDistribution(const int num_classes) : counts_(num_classes) {}

  // Remove all observations.
  void Clear();

  // Export to proto.
  template <typename P>
  void Save(P* proto) const;

  // Import form proto.
  template <typename P>
  void Load(const P& proto);

  // Most frequent class. Smallest index in case of ties. Return 0 if empty.
  int TopClass() const;

  // Number of classes i.e. max integer value + 1.
  int NumClasses() const { return counts_.size(); }

  // Normalize the histogram i.e. make sum_ = 1.
  void Normalize();

  // Normalize while ensuring that all counts are positive.
  void NormalizeAndClampPositive();

  // Define the number of classes.
  void SetNumClasses(const int c);

  void SetZero() {
    sum_ = 0;
    std::fill(counts_.begin(), counts_.end(), 0);
  }

  // Add one observation.
  void Add(int v);
  void Add(int v, T weight);

  // Subtract one observation.
  void Sub(int v);
  void Sub(int v, T weight);

  // Add the content of another IntegerDistribution.
  void Add(const IntegerDistribution<T>& v);

  // Subtracts the content of another IntegerDistribution.
  void Sub(const IntegerDistribution<T>& v);

  // Set the content of another IntegerDistribution. Like Add and Sub, the class
  // should be pre initialized.
  void Set(const IntegerDistribution<T>& v);

  // Add the content of an IntegerCounter with a total weight of 1.
  void AddNormalized(const IntegerDistribution<T>& v);

  // Subtract the content of an IntegerCounter with a total weight of 1.
  void SubNormalized(const IntegerDistribution<T>& v);

  // Add the content of an IntegerCounter with a total weight of 1.
  // Similar to "AddNormalized", when the input is a IntegerDistribution* proto.
  template <typename P>
  void AddNormalizedProto(const P& v);

  // Subtract the content of an IntegerCounter with a total weight of 1.
  // Similar to "SubNormalized", when the input is a IntegerDistribution* proto.
  template <typename P>
  void SubNormalizedProto(const P& v);

  // Number of observations.
  T NumObservations() const { return sum_; }

  // Query a value.
  T count(const int i) const { return counts_[i]; }

  // Entropy of the observations.
  double Entropy() const;

  // Proportion of observations equal to "value". If the distribution is empty,
  // returns minus infinity.
  float SafeProportionOrMinusInfinity(int value) const {
    if (sum_ > 0) {
      return counts_[value] / sum_;
    }
    return -std::numeric_limits<float>::infinity();
  }

  // private:
  T sum_ = 0;
  // Note: "counts_" that has a default inlined (much faster) size of 3 for
  // binary classification (one for each Missing/False/True values).
  absl::InlinedVector<T, 3> counts_;
};

using IntegerDistributionDouble = IntegerDistribution<double>;
using IntegerDistributionInt64 = IntegerDistribution<int64_t>;
using IntegerDistributionFloat = IntegerDistribution<float>;

// Confusion matrix between a binary attribute and an integer attribute.
template <typename T>
class BinaryToIntegerConfusionMatrix {
 public:
  // Add an entry.
  void Add(bool bool_dim, int int_dim, const T weight = 1.);

  // Entropy metrics.
  double InformationGain() const;
  double InitEntropy() const;
  double FinalEntropy() const;

  // Set the number of classes i.e. the possible values are [0, NumClasses() [.
  void SetNumClassesIntDim(int int_dim);

  // Total number of observations.
  T NumObservations() const;

  // Remove all observations.
  void Clear();

  // Get the integer counter of the "negative" class.
  IntegerDistribution<T>* mutable_neg() { return &split_[0]; }

  // Get the integer counter of the "positive" class.
  IntegerDistribution<T>* mutable_pos() { return &split_[1]; }

  // Get the integer counter of the "negative" class.
  const IntegerDistribution<T>& neg() const { return split_[0]; }

  // Get the integer counter of the "positive" class.
  const IntegerDistribution<T>& pos() const { return split_[1]; }

 private:
  IntegerDistribution<T> split_[2];
};

using BinaryToIntegerConfusionMatrixDouble =
    BinaryToIntegerConfusionMatrix<double>;
using BinaryToIntegerConfusionMatrixInt64 =
    BinaryToIntegerConfusionMatrix<int64_t>;

// Confusion matrix between two integer attributes.
template <typename T>
class IntegersConfusionMatrix {
 public:
  // Add an entry.
  void Add(int32_t row, int32_t col, const T weight);

  // Set the size i.e. r \in [0, nr - 1] and c \in [0, nc - 1].
  void SetSize(int32_t nrow, int32_t ncol);

  // Total number of observations.
  double NumObservations() const { return sum_; }

  int32_t nrow() const { return nrow_; }

  int32_t ncol() const { return ncol_; }

  void SetZero() {
    sum_ = 0;
    std::fill(counts_.begin(), counts_.end(), 0);
  }

  // Index of "counts_" of a cell specified by a (row,col) coordinate.
  inline static int Index(const int row, const int col, const int num_rows) {
    return row + col * num_rows;
  }

  inline T& at(const int32_t row, const int32_t col) {
    DCHECK_GE(row, 0);
    DCHECK_LT(row, nrow_);
    DCHECK_GE(col, 0);
    DCHECK_LT(col, ncol_);
    return counts_[Index(row, col, nrow_)];
  }

  inline T at(const int32_t row, const int32_t col) const {
    DCHECK_GE(row, 0);
    DCHECK_LT(row, nrow_);
    DCHECK_GE(col, 0);
    DCHECK_LT(col, ncol_);
    return counts_[Index(row, col, nrow_)];
  }

  // Import form proto version. "proto::IntegersConfusionMatrixDouble" is
  // equivalent to "IntegersConfusionMatrixDouble".
  template <typename P>
  void Load(const P& proto);

  // Create a text table representing the confusion matrix.
  void AppendTextReport(const dataset::proto::Column& column_spec,
                        std::string* result) const;

  // Create a text table representing the confusion matrix.
  void AppendTextReport(const std::vector<std::string>& column_labels,
                        const std::vector<std::string>& row_labels,
                        std::string* result) const;

  // Create a html table representing the confusion matrix.
  void AppendHtmlReport(const dataset::proto::Column& column_spec,
                        std::string* result) const;

  // Create a html table representing the confusion matrix.
  void AppendHtmlReport(const std::vector<std::string>& column_labels,
                        const std::vector<std::string>& row_labels,
                        absl::string_view corner_label,
                        std::string* result) const;

 private:
  T sum_ = 0;
  int32_t nrow_ = 0;
  int32_t ncol_ = 0;
  std::vector<T> counts_;
};

// Gets the index of the class with highest probability.
// Equivalent to IntegerDistribution::TopClass.
int TopClass(const proto::IntegerDistributionFloat& dist);
int TopClass(const proto::IntegerDistributionDouble& dist);

using IntegersConfusionMatrixDouble = IntegersConfusionMatrix<double>;
using IntegersConfusionMatrixInt64 = IntegersConfusionMatrix<int64_t>;

// Initialize a confusion matrix.
void InitializeConfusionMatrixProto(
    int32_t nr, int32_t nc, proto::IntegersConfusionMatrixDouble* confusion);

// Increment the value in one cell of a confusion matrix.
void AddToConfusionMatrixProto(int32_t r, int32_t c, double value,
                               proto::IntegersConfusionMatrixDouble* confusion);

// Add two initialized confusion matrices.
void AddToConfusionMatrixProto(const proto::IntegersConfusionMatrixDouble& src,
                               proto::IntegersConfusionMatrixDouble* dst);

// Sum of the elements along the diagonal i.e. the "matching" predictions.
double ConfusionMatrixProtoTrace(
    const proto::IntegersConfusionMatrixDouble& confusion);

// Sum all the elements of a given row.
double ConfusionMatrixProtoSumColumns(
    const proto::IntegersConfusionMatrixDouble& confusion, int32_t row);

// Initialize an IntegerDistributionProto.
template <typename T>
void InitializeIntegerDistributionProto(const int num_classes,
                                        T* integer_distribution_proto);

// Add the distribution "src" to "dst" with a given weight. Similar to
// IntegerDistribution::AddNormalized.
template <typename T>
void AddNormalizedToIntegerDistributionProto(const T& src, const float weight,
                                             T* dst);

// Add a single observation to "dst". Similar to IntegerDistribution::Add.
template <typename T>
void AddToIntegerDistributionProto(int value, float weight, T* dst);

// Get the probability density of a given value.
template <typename T>
double GetDensityIntegerDistributionProto(const T& dist, int value);

// Compute p log p without worrying about p=0.
inline double ProtectedPLogP(double p) {
  if (p <= 0) return 0.;
  return -p * log(p);
}

// Compute p log p (with p = a/b) without worrying about p=0. Returns 0 if
// a = b = 0  (which is the limit for a/b -> 0+).
inline double ProtectedPLogPInt(int64_t a, int64_t b) {
  DCHECK_GE(a, 0);
  DCHECK_LE(a, b);
  if (a == 0 || a == b) return 0.;
  DCHECK_GT(b, 0);
  return ProtectedPLogP(static_cast<double>(a) / b);
}

inline double ProtectedPLogPDouble(double a, double b) {
  if (a <= 0 || a >= b) return 0.;
  DCHECK_GT(b, 0);
  return ProtectedPLogP(a / b);
}

template <typename T>
void IntegersConfusionMatrix<T>::Add(const int32_t row, const int32_t col,
                                     const T weight) {
  at(row, col) += weight;
  sum_ += weight;
}

template <typename T>
void IntegersConfusionMatrix<T>::SetSize(const int32_t nrow,
                                         const int32_t ncol) {
  counts_.resize(nrow * ncol);
  nrow_ = nrow;
  ncol_ = ncol;
}

template <typename T>
void IntegersConfusionMatrix<T>::AppendTextReport(
    const std::vector<std::string>& column_labels,
    const std::vector<std::string>& row_labels, std::string* result) const {
  CHECK_EQ(column_labels.size(), ncol());
  CHECK_EQ(row_labels.size(), nrow());

  // Minimum margin (expressed in spaces) between displayed elements.
  const int margin = 2;

  // Maximum length of the values' string representation i.e. maximum of
  // "row_labels".
  int max_row_label_length = 0;
  for (const auto& label : row_labels) {
    if (label.size() > max_row_label_length) {
      max_row_label_length = label.size();
    }
  }

  // Maximum string length of the elements in each column.
  std::vector<int> max_length_per_col(ncol_);
  for (int col = 0; col < ncol_; col++) {
    // Counts.
    T max_value = 1;
    for (int row = 0; row < nrow_; row++) {
      const auto value = at(row, col);
      if (value > max_value) {
        max_value = value;
      }
    }
    // Column header.
    max_length_per_col[col] =
        std::max(static_cast<int>(column_labels[col].size()),
                 static_cast<int>(std::floor(std::log10(max_value))) + 1);
  }

  // Print "value" to the end of "result" using a left margin (similar to
  // "std::setw").
  const auto print_string = [&](int length, absl::string_view value) {
    const int preceding_spaces =
        std::max(length - static_cast<int>(value.size()), 0);
    absl::StrAppend(result, std::string(preceding_spaces, ' '), value);
  };
  const auto print_value = [&](int length, T value) {
    print_string(length, absl::StrCat(value));
  };

  // Print header.
  print_string(max_row_label_length, "");
  for (int col = 0; col < ncol_; col++) {
    print_string(max_length_per_col[col] + margin, column_labels[col]);
  }
  absl::StrAppend(result, "\n");

  // Print body.
  for (int row = 0; row < nrow_; row++) {
    print_string(max_row_label_length, row_labels[row]);
    for (int col = 0; col < ncol_; col++) {
      print_value(max_length_per_col[col] + margin, at(row, col));
    }
    absl::StrAppend(result, "\n");
  }
  absl::StrAppend(result, "Total: ", sum_, "\n");
}

template <typename T>
void IntegersConfusionMatrix<T>::AppendTextReport(
    const dataset::proto::Column& column_spec, std::string* result) const {
  CHECK_EQ(column_spec.categorical().number_of_unique_values(), ncol());
  CHECK_EQ(column_spec.categorical().number_of_unique_values(), nrow());

  // Extract the string representation of the column values.
  std::vector<std::string> labels(ncol_);
  for (int value = 0; value < ncol_; value++) {
    labels[value] = dataset::CategoricalIdxToRepresentation(column_spec, value);
  }
  absl::StrAppend(result, "truth\\prediction\n");
  AppendTextReport(labels, labels, result);
}

template <typename T>
void IntegersConfusionMatrix<T>::AppendHtmlReport(
    const dataset::proto::Column& column_spec, std::string* result) const {
  CHECK_EQ(column_spec.categorical().number_of_unique_values(), ncol());
  CHECK_EQ(column_spec.categorical().number_of_unique_values(), nrow());

  // Extract the string representation of the column values.
  std::vector<std::string> labels(ncol_);
  for (int value = 0; value < ncol_; value++) {
    labels[value] = dataset::CategoricalIdxToRepresentation(column_spec, value);
  }
  AppendHtmlReport(labels, labels, "Truth\\Prediction", result);
}

template <typename T>
void IntegersConfusionMatrix<T>::AppendHtmlReport(
    const std::vector<std::string>& column_labels,
    const std::vector<std::string>& row_labels, absl::string_view corner_label,
    std::string* result) const {
  namespace h = yggdrasil_decision_forests::utils::html;
  namespace a = yggdrasil_decision_forests::utils::html;
  // Table
  h::Html rows;
  // Header
  {
    h::Html row;
    row.Append(h::Th(a::Align("left"), corner_label));
    for (const auto& col_label : column_labels) {
      row.Append(h::Th(a::Align("right"), col_label));
    }
    rows.Append(h::Tr(row));
  }
  // Body
  for (int row = 0; row < nrow_; row++) {
    h::Html html_row;
    html_row.Append(h::Th(a::Align("right"), row_labels[row]));
    for (int col = 0; col < ncol_; col++) {
      const auto value = at(row, col);

      // We build a color "hsl(100°, 0.5, l)". If the lightness "l" = 1, the
      // color is white. If the lightness "l" = 0.5, the color is a strong
      // green. The lightness scales with the confusion matrix cell count: "all
      // values in a cell => l=0.5" => strong green" and "no values in a cell =>
      // l=1.0 => white".
      class h::Style style;
      style.BackgroundColorHSL(100. / 360., 0.5, 1. - 0.5 * value / sum_);
      html_row.Append(
          h::Td(a::Align("right"), a::Style(style), absl::StrCat(value)));
    }
    rows.Append(h::Tr(html_row));
  }
  auto content = h::Table(a::Class("confusion_matrix"), rows);
  content.Append(h::P("Total: ", absl::StrCat(sum_)));
  absl::StrAppend(result, std::string(content.content()));
}

template <typename T>
template <typename P>
void IntegersConfusionMatrix<T>::Load(const P& proto) {
  sum_ = proto.sum();
  nrow_ = proto.nrow();
  ncol_ = proto.ncol();
  counts_.assign(proto.counts().begin(), proto.counts().end());
}

template <typename T>
int IntegerDistribution<T>::TopClass() const {
  int top_index = 0;
  T top_value = 0.;
  for (int i = 0; i < counts_.size(); i++) {
    if (counts_[i] > top_value) {
      top_value = counts_[i];
      top_index = i;
    }
  }
  return top_index;
}

template <typename T>
void IntegerDistribution<T>::Clear() {
  sum_ = 0.;
  for (auto& v : counts_) {
    v = 0;
  }
}

template <typename T>
template <typename P>
void IntegerDistribution<T>::Save(P* proto) const {
  proto->set_sum(sum_);
  proto->mutable_counts()->Resize(counts_.size(), 0);
  for (int i = 0; i < counts_.size(); i++)
    proto->mutable_counts()->Set(i, counts_[i]);
}

template <typename T>
template <typename P>
void IntegerDistribution<T>::Load(const P& proto) {
  counts_.clear();
  sum_ = proto.sum();
  for (int i = 0; i < proto.counts_size(); i++)
    counts_.push_back(proto.counts(i));
}

template <typename T>
void IntegerDistribution<T>::Normalize() {
  if (sum_ == 0) return;
  DCHECK_GE(sum_, 0);
  for (auto& w : counts_) {
    DCHECK_GE(w, 0);
    w /= sum_;
  }
  sum_ = 1;
}

template <typename T>
void IntegerDistribution<T>::NormalizeAndClampPositive() {
  sum_ = 0;
  for (auto& w : counts_) {
    if (w < 0) {
      w = 0.f;
    }
    sum_ += w;
  }
  if (sum_ == 0) return;
  DCHECK_GE(sum_, 0);
  for (auto& w : counts_) {
    w /= sum_;
  }
  sum_ = 1;
}

template <typename T>
void IntegerDistribution<T>::SetNumClasses(const int c) {
  counts_.resize(c);
}

template <typename T>
void IntegerDistribution<T>::Add(const int v) {
  DCHECK_GE(v, 0);
  DCHECK_LT(v, counts_.size());
  sum_ += T{1};
  counts_[v] += T{1};
}

template <typename T>
void IntegerDistribution<T>::Add(const int v, const T weight) {
  DCHECK_GE(v, 0);
  DCHECK_LT(v, counts_.size());
  sum_ += weight;
  counts_[v] += weight;
}

template <typename T>
void IntegerDistribution<T>::Sub(const int v) {
  DCHECK_GE(v, 0);
  DCHECK_LT(v, counts_.size());
  sum_ -= T{1};
  counts_[v] -= T{1};
}

template <typename T>
void IntegerDistribution<T>::Sub(const int v, const T weight) {
  DCHECK_GE(v, 0);
  DCHECK_LT(v, counts_.size());
  sum_ -= weight;
  counts_[v] -= weight;
}

template <typename T>
void IntegerDistribution<T>::Add(const IntegerDistribution<T>& v) {
  DCHECK_EQ(counts_.size(), v.counts_.size());
  sum_ += v.sum_;
  for (int i = 0; i < counts_.size(); i++) {
    counts_[i] += v.counts_[i];
  }
}

template <typename T>
void IntegerDistribution<T>::Sub(const IntegerDistribution<T>& v) {
  DCHECK_EQ(counts_.size(), v.counts_.size());
  sum_ -= v.sum_;
  for (int i = 0; i < counts_.size(); i++) {
    counts_[i] -= v.counts_[i];
  }
}

template <typename T>
void IntegerDistribution<T>::Set(const IntegerDistribution<T>& v) {
  DCHECK_EQ(counts_.size(), v.counts_.size());
  sum_ = v.sum_;
  for (int i = 0; i < counts_.size(); i++) {
    counts_[i] = v.counts_[i];
  }
}

template <typename T>
void IntegerDistribution<T>::AddNormalized(const IntegerDistribution<T>& v) {
  DCHECK_EQ(NumClasses(), v.NumClasses());
  if (v.NumObservations() == 0) return;
  sum_++;
  for (int i = 0; i < counts_.size(); i++) {
    counts_[i] += v.count(i) / v.NumObservations();
  }
}

template <typename T>
template <typename P>
void IntegerDistribution<T>::AddNormalizedProto(const P& v) {
  DCHECK_EQ(NumClasses(), v.counts_size());
  if (v.sum() == 0) return;
  sum_++;
  for (int i = 0; i < counts_.size(); i++) {
    counts_[i] += v.counts(i) / v.sum();
  }
}

template <typename T>
template <typename P>
void IntegerDistribution<T>::SubNormalizedProto(const P& v) {
  DCHECK_EQ(NumClasses(), v.counts_size());
  if (v.sum() == 0) return;
  sum_--;
  for (int i = 0; i < counts_.size(); i++) {
    counts_[i] -= v.counts(i) / v.sum();
  }
}

template <typename T>
void AddNormalizedToIntegerDistributionProto(const T& src, const float weight,
                                             T* dst) {
  DCHECK_EQ(src.counts_size(), dst->counts_size());
  DCHECK_GT(src.sum(), 0);
  for (int i = 0; i < src.counts_size(); ++i) {
    const auto count_weight = src.counts(i) / src.sum() * weight;
    dst->set_counts(i, dst->counts(i) + count_weight);
  }
  dst->set_sum(dst->sum() + weight);
}

template <typename T>
void AddToIntegerDistributionProto(const int value, const float weight,
                                   T* dst) {
  dst->set_counts(value, dst->counts(value) + weight);
  dst->set_sum(dst->sum() + weight);
}

template <typename T>
double GetDensityIntegerDistributionProto(const T& dist, const int value) {
  if (dist.sum() == 0) {
    return 0;
  }
  return dist.counts(value) / dist.sum();
}

template <typename T>
void InitializeIntegerDistributionProto(const int num_classes,
                                        T* integer_distribution_proto) {
  DCHECK_EQ(integer_distribution_proto->counts_size(), 0);
  for (int i = 0; i < num_classes; ++i) {
    integer_distribution_proto->add_counts(0.0);
  }
  integer_distribution_proto->set_sum(0.0);
}

template <typename T>
void IntegerDistribution<T>::SubNormalized(const IntegerDistribution<T>& v) {
  DCHECK_EQ(NumClasses(), v.NumClasses());
  if (v.NumObservations() == 0) return;
  sum_--;
  for (int i = 0; i < counts_.size(); i++) {
    counts_[i] -= v.count(i) / v.NumObservations();
  }
}

template <typename T>
double IntegerDistribution<T>::Entropy() const {
  if (sum_ == 0) {
    return 0;
  }
  double entropy = 0;
  for (auto count : counts_) {
    entropy += ProtectedPLogPDouble(count, sum_);
  }
  return entropy;
}

// Entropy of a distribution [p, 1-p]. This function is safe and always returns
// a non-NAN value.
inline double BinaryDistributionEntropyD(const double p) {
  // Equivalent to if (p <= 0. || p >= 1. || std::isnan(p))
  if (!(p > 0.) || p >= 1.) {
    return 0.;
  }
  return -p * std::log(p) - (1. - p) * std::log(1. - p);
}

inline float BinaryDistributionEntropyF(const float p) {
  // Equivalent to if (p <= 0. || p >= 1. || std::isnan(p))
  if (!(p > 0.f) || p >= 1.f) {
    return 0.f;
  }
  return -p * std::log(p) - (1.f - p) * std::log(1.f - p);
}

template <typename T>
void BinaryToIntegerConfusionMatrix<T>::SetNumClassesIntDim(int int_dim) {
  DCHECK_GE(int_dim, 0);
  split_[0].SetNumClasses(int_dim);
  split_[1].SetNumClasses(int_dim);
}

template <typename T>
void BinaryToIntegerConfusionMatrix<T>::Add(const bool bool_dim,
                                            const int int_dim, const T weight) {
  split_[bool_dim].Add(int_dim, weight);
}

template <typename T>
T BinaryToIntegerConfusionMatrix<T>::NumObservations() const {
  return split_[0].NumObservations() + split_[1].NumObservations();
}

template <typename T>
void BinaryToIntegerConfusionMatrix<T>::Clear() {
  split_[0].Clear();
  split_[1].Clear();
}

template <typename T>
double BinaryToIntegerConfusionMatrix<T>::InformationGain() const {
  return InitEntropy() - FinalEntropy();
}

template <typename T>
double BinaryToIntegerConfusionMatrix<T>::InitEntropy() const {
  const T sum = NumObservations();
  if (sum == 0) return 0;
  CHECK_EQ(neg().NumClasses(), pos().NumClasses());
  const int num_classes = neg().NumClasses();
  double entropy = 0;
  for (int i = 0; i < num_classes; i++) {
    const T class_count = neg().count(i) + pos().count(i);
    entropy += ProtectedPLogPDouble(class_count, sum);
  }
  return entropy;
}

template <typename T>
double BinaryToIntegerConfusionMatrix<T>::FinalEntropy() const {
  const T sum = NumObservations();
  if (sum == 0) return 0;
  const double p_neg = static_cast<double>(neg().NumObservations()) / sum;
  return neg().Entropy() * p_neg + pos().Entropy() * (1 - p_neg);
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTION_H_
