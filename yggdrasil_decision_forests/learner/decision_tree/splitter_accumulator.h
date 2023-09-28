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

// Template classes used by the splitter to accumulate statistics: Buckets and
// accumulators.
//
// See "decision_tree_splitter_scanner.h" for an explanation of the use of these
// objects.
//
// Feature buckets
// ===============
// Available: FeatureNumericalBucket, FeatureCategoricalBucket,
// FeatureIsMissingBucket.
//
// Label buckets & accumulator
// ===========================
// Available: LabelNumerical{Bucket,ScoreAccumulator}.
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPLITTER_ACCUMULATOR_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPLITTER_ACCUMULATOR_H_

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_accumulator_uplift.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_structure.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {
namespace internal {

// Bucket data containers.
//
// Bucket definitions are templated to facilitate code reuse. Since buckets are
// constructed many times, it's worth saving memory aggressively and only
// construct fields that will actually be used. C++ does not support 0-byte
// objects, hence the unused fields cannot be set to void. Empty structs would
// occupy 1 byte for the unused field. Combining two fields into a struct is
// therefore the most space-efficient alternative.
struct BooleanValueAndWeight {
  bool value;
  float weight;
};

struct BooleanValueOnly {
  bool value;
};

struct IntegerValueAndWeight {
  int value;
  float weight;
};

struct IntegerValueOnly {
  int value;
};

struct FloatValueAndWeight {
  float value;
  float weight;
};

struct FloatValueOnly {
  float value;
};

struct FloatGradientHessianAndWeight {
  float gradient;
  float hessian;
  float weight;
};

struct FloatGradientHessianOnly {
  float gradient;
  float hessian;
};

struct FloatSumGradientHessianAndWeight {
  float sum_gradient;
  float sum_hessian;
  float sum_weight;
};

struct FloatSumGradientHessianOnly {
  float sum_gradient;
  float sum_hessian;
};

struct SumTruesAndWeights {
  double sum_trues;
  double sum_weights;
};

struct SumTruesOnly {
  double sum_trues;
};

}  // namespace internal

// ===============
// Feature Buckets
// ===============
//
// FeatureBucket Accumulates statistics about the features. They should
// implement the following methods:
//
// Ordering of the buckets. If "kRequireSorting=true", the bucket will be
// scanned in increasing order.
// bool operator<(const FeatureBucket& other) const;
//
// If true, the buckets will be sorted before being scanned.
// static constexpr bool kRequireSorting;
//
// Given the first and last filled buckets for a particular attribute, test if
// this attribute is valid. If invalid, the bucket is not scanned. Note:
// Different algorithms can invalid attributes differently.
// static bool IsValidAttribute(const FeatureBucket& first,
//                             const FeatureBucket& last);
//
// Are two consecutive buckets defining a valid split i.e. can
// "SetConditionFinal" be called on "left".
// static bool IsValidSplit(const FeatureBucket& left, const FeatureBucket&
// right)
//
// FeatureFiller controls the accumulation in a feature bucket. They should
// implement the following methods:
//
// Number of buckets to allocate.
// size_t NumBuckets() const;
//
// // Initialize the content of a bucket.
// void InitializeAndZero(const int bucket_idx, FeatureBucket* acc) const;
//
// // In which bucket a given example is falling.
// size_t GetBucketIndex(const size_t local_example_idx, const
// UnsignedExampleIdx example_idx) const;
//
// // Consume a training example.
// void ConsumeExample(const UnsignedExampleIdx example_idx, FeatureBucket* acc)
// const;
//
// Set the split function if this bucket is selected as
// the best greater bucket in the negative side of the split. All the
// following buckets will end in the positive side of the split.
// template<typename ExampleBucketSet>
// virtual void SetConditionFinal(const ExampleBucketSet& example_bucket_set,
//                                const size_t best_bucket_idx,
//                                proto::NodeCondition* condition) const = 0;
//
// Numerical feature.
struct FeatureNumericalBucket {
  // Numerical feature value.
  float value;

  // The buckets will be sorted according to the numerical values.
  static constexpr bool kRequireSorting = true;

  bool operator<(const FeatureNumericalBucket& other) const {
    return value < other.value;
  }

  static bool IsValidAttribute(const FeatureNumericalBucket& first,
                               const FeatureNumericalBucket& last) {
    return first.value != last.value;
  }

  static bool IsValidSplit(const FeatureNumericalBucket& left,
                           const FeatureNumericalBucket& right) {
    return IsValidAttribute(left, right);
  }

  class Filler {
   public:
    Filler(const UnsignedExampleIdx num_selected_examples,
           const float na_replacement, const std::vector<float>& attributes)
        : num_selected_examples_(num_selected_examples),
          na_replacement_(na_replacement),
          attributes_(attributes) {}

    size_t NumBuckets() const { return num_selected_examples_; }

    void InitializeAndZero(const int bucket_idx,
                           FeatureNumericalBucket* acc) const {}

    size_t GetBucketIndex(const size_t local_example_idx,
                          const UnsignedExampleIdx example_idx) const {
      return local_example_idx;
    }

    void ConsumeExample(const UnsignedExampleIdx example_idx,
                        FeatureNumericalBucket* acc) const {
      acc->value = GetValue(example_idx);
    }

    float GetValue(const UnsignedExampleIdx example_idx) const {
      const float attribute = attributes_[example_idx];
      return std::isnan(attribute) ? na_replacement_ : attribute;
    }

    void SetConditionFinalFromThresholds(
        const float threshold_1, const float threshold_2,
        proto::NodeCondition* condition) const {
      const float threshold = MidThreshold(threshold_1, threshold_2);
      condition->mutable_condition()->mutable_higher_condition()->set_threshold(
          threshold);
      condition->set_na_value(na_replacement_ >= threshold);
    }

    template <typename ExampleBucketSet>
    void SetConditionFinal(const ExampleBucketSet& example_bucket_set,
                           const size_t best_bucket_idx,
                           proto::NodeCondition* condition) const {
      const float threshold_1 =
          example_bucket_set.items[best_bucket_idx].feature.value;
      const float threshold_2 =
          example_bucket_set.items[best_bucket_idx + 1].feature.value;
      SetConditionFinalFromThresholds(threshold_1, threshold_2, condition);
    }

    float MissingValueReplacement() const { return na_replacement_; }

   private:
    const UnsignedExampleIdx num_selected_examples_;
    const float na_replacement_;
    const std::vector<float>& attributes_;
  };

  friend std::ostream& operator<<(std::ostream& os,
                                  const FeatureNumericalBucket& data);
};

inline std::ostream& operator<<(std::ostream& os,
                                const FeatureNumericalBucket& data) {
  os << "value:" << data.value;
  return os;
}

// Discretized Numerical feature.
struct FeatureDiscretizedNumericalBucket {
  static constexpr bool kRequireSorting = false;

  bool operator<(const FeatureDiscretizedNumericalBucket& other) const {
    NOTREACHED();
    return true;
  }

  static bool IsValidAttribute(const FeatureDiscretizedNumericalBucket& first,
                               const FeatureDiscretizedNumericalBucket& last) {
    return true;
  }

  static bool IsValidSplit(const FeatureDiscretizedNumericalBucket& left,
                           const FeatureDiscretizedNumericalBucket& right) {
    return true;
  }

  class Filler {
   public:
    Filler(const int num_bins,
           const dataset::DiscretizedNumericalIndex na_replacement,
           const std::vector<dataset::DiscretizedNumericalIndex>& attributes)
        : num_bins_(num_bins),
          na_replacement_(na_replacement),
          attributes_(attributes) {}

    size_t NumBuckets() const { return num_bins_; }

    void InitializeAndZero(const int bucket_idx,
                           FeatureDiscretizedNumericalBucket* acc) const {}

    size_t GetBucketIndex(const size_t local_example_idx,
                          const UnsignedExampleIdx example_idx) const {
      const auto attribute = attributes_[example_idx];
      return (attribute != dataset::kDiscretizedNumericalMissingValue)
                 ? attribute
                 : na_replacement_;
    }

    void ConsumeExample(const UnsignedExampleIdx example_idx,
                        FeatureDiscretizedNumericalBucket* acc) const {}

    template <typename ExampleBucketSet>
    void SetConditionFinal(const ExampleBucketSet& example_bucket_set,
                           const size_t best_bucket_idx,
                           proto::NodeCondition* condition) const {
      condition->mutable_condition()
          ->mutable_discretized_higher_condition()
          ->set_threshold(best_bucket_idx + 1);
      condition->set_na_value(na_replacement_ > best_bucket_idx);
    }

    template <typename ExampleBucketSet>
    void SetConditionInterpolatedFinal(
        const ExampleBucketSet& example_bucket_set,
        const size_t best_bucket_1_idx, const size_t best_bucket_2_idx,
        proto::NodeCondition* condition) const {
      // The "discretized_higher_condition" does not allow to express
      // condition with threshold other than the threshold of the discretized
      // value. Therefore, we do an interpolation in the bucket index domain and
      // round up to the smallest one.
      const int best_bucket_idx = (best_bucket_1_idx + best_bucket_2_idx) / 2;
      condition->mutable_condition()
          ->mutable_discretized_higher_condition()
          ->set_threshold(best_bucket_idx + 1);
      condition->set_na_value(na_replacement_ > best_bucket_idx);
    }

   private:
    int num_bins_;
    dataset::DiscretizedNumericalIndex na_replacement_;
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes_;
  };

  friend std::ostream& operator<<(
      std::ostream& os, const FeatureDiscretizedNumericalBucket& data);
};

inline std::ostream& operator<<(std::ostream& os,
                                const FeatureDiscretizedNumericalBucket& data) {
  // The feature bucket contains no information.
  return os;
}

// Categorical feature.
struct FeatureCategoricalBucket {
  int32_t value;
  static constexpr bool kRequireSorting = false;

  bool operator<(const FeatureCategoricalBucket& other) const {
    NOTREACHED();
    return true;
  }

  static bool IsValidAttribute(const FeatureCategoricalBucket& first,
                               const FeatureCategoricalBucket& last) {
    return true;
  }

  static bool IsValidSplit(const FeatureCategoricalBucket& left,
                           const FeatureCategoricalBucket& right) {
    return true;
  }

  class Filler {
   public:
    Filler(const int num_categorical_values, const int na_replacement,
           const std::vector<int32_t>& attributes)
        : num_categorical_values_(num_categorical_values),
          na_replacement_(na_replacement),
          attributes_(attributes) {}

    size_t NumBuckets() const { return num_categorical_values_; }

    void InitializeAndZero(const int bucket_idx,
                           FeatureCategoricalBucket* acc) const {
      acc->value = bucket_idx;
    }

    size_t GetBucketIndex(const size_t local_example_idx,
                          const UnsignedExampleIdx example_idx) const {
      const auto attribute = attributes_[example_idx];
      return (attribute ==
              dataset::VerticalDataset::CategoricalColumn::kNaValue)
                 ? na_replacement_
                 : attribute;
    }

    void ConsumeExample(const UnsignedExampleIdx example_idx,
                        FeatureCategoricalBucket* acc) const {}

    template <typename ExampleBucketSet>
    void SetConditionFinal(const ExampleBucketSet& example_bucket_set,
                           const size_t best_bucket_idx,
                           proto::NodeCondition* condition) const {
      std::vector<int32_t> positive_attribute_value;
      positive_attribute_value.reserve(num_categorical_values_ -
                                       best_bucket_idx - 1);
      bool na_replacement_in_pos = false;
      for (int bucket_idx = best_bucket_idx + 1;
           bucket_idx < num_categorical_values_; bucket_idx++) {
        const auto attribute_value =
            example_bucket_set.items[bucket_idx].feature.value;
        if (attribute_value == na_replacement_) {
          na_replacement_in_pos = true;
        }
        positive_attribute_value.push_back(attribute_value);
      }

      SetPositiveAttributeSetOfCategoricalContainsCondition(
          positive_attribute_value, num_categorical_values_, condition);

      condition->set_na_value(na_replacement_in_pos);
    }

    // Set the condition as "value \in mask" with
    // mask = [ example_bucket_set.items[bucket_idx].feature.value for
    //   bucket_idx in range(best_order_idx+1,example_bucket_set.items.size())]
    template <typename ExampleBucketSet>
    void SetConditionFinalWithOrder(
        const std::vector<std::pair<float, int32_t>>& bucket_order,
        const ExampleBucketSet& example_bucket_set, const size_t best_order_idx,
        proto::NodeCondition* condition) const {
      std::vector<int32_t> positive_attribute_value;
      positive_attribute_value.reserve(bucket_order.size() - best_order_idx -
                                       1);
      bool na_replacement_in_pos = false;
      for (int order_idx = best_order_idx + 1; order_idx < bucket_order.size();
           order_idx++) {
        const int32_t bucket_idx = bucket_order[order_idx].second;
        const int32_t attribute_value =
            example_bucket_set.items[bucket_idx].feature.value;
        if (attribute_value == na_replacement_) {
          na_replacement_in_pos = true;
        }
        positive_attribute_value.push_back(attribute_value);
      }

      SetPositiveAttributeSetOfCategoricalContainsCondition(
          positive_attribute_value, num_categorical_values_, condition);

      condition->set_na_value(na_replacement_in_pos);
    }

    // Set the condition as "value \in mask".
    void SetConditionFinalWithBuckets(const std::vector<int32_t>& mask,
                                      proto::NodeCondition* condition) const {
      bool na_replacement_in_pos =
          std::find(mask.begin(), mask.end(), na_replacement_) != mask.end();
      SetPositiveAttributeSetOfCategoricalContainsCondition(
          mask, num_categorical_values_, condition);
      condition->set_na_value(na_replacement_in_pos);
    }

   private:
    int num_categorical_values_;
    int na_replacement_;
    const std::vector<int32_t>& attributes_;
  };

  friend std::ostream& operator<<(std::ostream& os,
                                  const FeatureCategoricalBucket& data);
};

inline std::ostream& operator<<(std::ostream& os,
                                const FeatureCategoricalBucket& data) {
  os << "value:" << data.value;
  return os;
}

// Boolean feature.
struct FeatureBooleanBucket {
  static constexpr bool kRequireSorting = false;

  bool operator<(const FeatureBooleanBucket& other) const {
    NOTREACHED();
    return true;
  }

  static bool IsValidAttribute(const FeatureBooleanBucket& first,
                               const FeatureBooleanBucket& last) {
    return true;
  }

  static bool IsValidSplit(const FeatureBooleanBucket& left,
                           const FeatureBooleanBucket& right) {
    return true;
  }

  class Filler {
   public:
    explicit Filler(const bool na_replacement,
                    const std::vector<char>& attributes)
        : na_replacement_(na_replacement), attributes_(attributes) {}

    size_t NumBuckets() const { return 2; }

    void InitializeAndZero(const int bucket_idx,
                           FeatureBooleanBucket* acc) const {}

    size_t GetBucketIndex(const size_t local_example_idx,
                          const UnsignedExampleIdx example_idx) const {
      const auto attribute = attributes_[example_idx];
      return (attribute == dataset::VerticalDataset::BooleanColumn::kNaValue)
                 ? na_replacement_
                 : attribute;
    }

    void ConsumeExample(const UnsignedExampleIdx example_idx,
                        FeatureBooleanBucket* acc) const {}

    template <typename ExampleBucketSet>
    void SetConditionFinal(const ExampleBucketSet& example_bucket_set,
                           const size_t best_bucket_idx,
                           proto::NodeCondition* condition) const {
      condition->mutable_condition()->mutable_true_value_condition();
      condition->set_na_value(na_replacement_);
    }

   private:
    const bool na_replacement_;
    const std::vector<char>& attributes_;
  };

  friend std::ostream& operator<<(std::ostream& os,
                                  const FeatureBooleanBucket& data);
};

inline std::ostream& operator<<(std::ostream& os,
                                const FeatureBooleanBucket& data) {
  os << "Boolean";
  return os;
}

// Binary feature about the presence / absence of value.
struct FeatureIsMissingBucket {
  static constexpr bool kRequireSorting = false;

  bool operator<(const FeatureIsMissingBucket& other) const {
    NOTREACHED();
    return true;
  }

  static bool IsValidAttribute(const FeatureIsMissingBucket& first,
                               const FeatureIsMissingBucket& last) {
    return true;
  }

  static bool IsValidSplit(const FeatureIsMissingBucket& left,
                           const FeatureIsMissingBucket& right) {
    return true;
  }

  class Filler {
   public:
    explicit Filler(
        const dataset::VerticalDataset::AbstractColumn* const attributes)
        : attributes_(attributes) {}

    size_t NumBuckets() const { return 2; }

    void InitializeAndZero(const int bucket_idx,
                           FeatureIsMissingBucket* acc) const {}

    size_t GetBucketIndex(const size_t local_example_idx,
                          const UnsignedExampleIdx example_idx) const {
      return attributes_->IsNa(example_idx);
    }

    void ConsumeExample(const UnsignedExampleIdx example_idx,
                        FeatureIsMissingBucket* acc) const {}

    template <typename ExampleBucketSet>
    void SetConditionFinal(const ExampleBucketSet& example_bucket_set,
                           const size_t best_bucket_idx,
                           proto::NodeCondition* condition) const {
      condition->mutable_condition()->mutable_na_condition();
    }

   private:
    const dataset::VerticalDataset::AbstractColumn* const attributes_;
  };

  friend std::ostream& operator<<(std::ostream& os,
                                  const FeatureIsMissingBucket& data);
};

inline std::ostream& operator<<(std::ostream& os,
                                const FeatureIsMissingBucket& data) {
  os << "NA";
  return os;
}

// ============
// Accumulators
// ============
//
// ScoreAccumulators accumulate the label statistics for a set of buckets. They
// should implement the following methods.
//
// Score obtained for all the scanned bucket. The score of the split will be 1)
// the example weighted sum of the scores of the accumulators (if
// "kNormalizeByWeight=true"), or 2) the sum of scores of the accumulators (if
// "kNormalizeByWeight=false").
// double Score() const;
//
// Weighted number of scanned examples.
// double WeightedNumExamples() const;
//
struct LabelNumericalScoreAccumulator {
  static constexpr bool kNormalizeByWeight = false;

  double Score() const { return label.VarTimesSumWeights(); }

  double WeightedNumExamples() const { return label.NumObservations(); }

  void ImportLabelStats(const proto::LabelStatistics& src) {
    label.Load(src.regression().labels());
  }

  void ExportLabelStats(proto::LabelStatistics* dst) const {
    label.Save(dst->mutable_regression()->mutable_labels());
  }

  void Sub(LabelNumericalScoreAccumulator* dst) const { dst->label.Sub(label); }

  void Add(LabelNumericalScoreAccumulator* dst) const { dst->label.Add(label); }

  utils::NormalDistributionDouble label;
};

struct LabelNumericalWithHessianScoreAccumulator {
  static constexpr bool kNormalizeByWeight = false;

  double Score() const { return label.VarTimesSumWeights(); }

  double WeightedNumExamples() const { return label.NumObservations(); }

  void ImportLabelStats(const proto::LabelStatistics& src) {
    label.Load(src.regression_with_hessian().labels());
    sum_hessian = src.regression_with_hessian().sum_hessian();
  }

  void ExportLabelStats(proto::LabelStatistics* dst) const {
    label.Save(dst->mutable_regression_with_hessian()->mutable_labels());
    dst->mutable_regression_with_hessian()->set_sum_hessian(sum_hessian);
  }

  void Sub(LabelNumericalWithHessianScoreAccumulator* dst) const {
    dst->label.Sub(label);
    dst->sum_hessian -= sum_hessian;
  }

  void Add(LabelNumericalWithHessianScoreAccumulator* dst) const {
    dst->label.Add(label);
    dst->sum_hessian += sum_hessian;
  }

  utils::NormalDistributionDouble label;
  double sum_hessian;
};

struct LabelCategoricalScoreAccumulator {
  static constexpr bool kNormalizeByWeight = true;

  double Score() const { return label.Entropy(); }

  double WeightedNumExamples() const { return label.NumObservations(); }

  void SetZero() { label.Clear(); }

  void ImportLabelStats(const proto::LabelStatistics& src) {
    label.Load(src.classification().labels());
  }

  void ExportLabelStats(proto::LabelStatistics* dst) const {
    label.Save(dst->mutable_classification()->mutable_labels());
  }

  void Sub(LabelCategoricalScoreAccumulator* dst) const {
    dst->label.Sub(label);
  }

  void Add(LabelCategoricalScoreAccumulator* dst) const {
    dst->label.Add(label);
  }

  utils::IntegerDistributionDouble label;
};

struct LabelBinaryCategoricalScoreAccumulator {
  static constexpr bool kNormalizeByWeight = true;

  double Score() const {
    return utils::BinaryDistributionEntropyF(sum_trues / sum_weights);
  }

  double WeightedNumExamples() const { return sum_weights; }

  void Clear() {
    sum_trues = 0.;
    sum_weights = 0.;
  }

  void Set(const double trues, const double weights) {
    this->sum_trues = trues;
    this->sum_weights = weights;
  }

  void AddOne(const bool value, const float weights) {
    static float table[] = {0.f, 1.f};
    sum_trues += table[value] * weights;
    sum_weights += weights;
  }

  void AddOne(const bool value) {
    static float table[] = {0.f, 1.f};
    sum_trues += table[value];
    sum_weights += 1.;
  }

  void SubOne(const bool value, const float weights) {
    static float table[] = {0.f, 1.f};
    sum_trues -= table[value] * weights;
    sum_weights -= weights;
  }

  void SubOne(const bool value) {
    static float table[] = {0.f, 1.f};
    sum_trues -= table[value];
    sum_weights -= 1.;
  }

  void AddMany(const double trues, const double weights) {
    sum_trues += trues;
    sum_weights += weights;
  }

  void SubMany(const double trues, const double weights) {
    sum_trues -= trues;
    sum_weights -= weights;
  }

  double sum_trues;
  double sum_weights;
};

struct LabelHessianNumericalScoreAccumulator {
  static constexpr bool kNormalizeByWeight = false;

  // Minimum hessian value when computing hessian scores and leaf values.
  static constexpr double kMinHessianForNewtonStep = 0.001;

  double Score() const {
    const double numerator = l1_threshold(sum_gradient, hessian_l1);
    const double denominator =
        std::max(sum_hessian, kMinHessianForNewtonStep) + hessian_l2;

    if (constraints.min_max_output.has_value()) {
      const double leaf = numerator / denominator;
      const auto constraint_min = constraints.min_max_output.value().min;
      const auto constraint_max = constraints.min_max_output.value().max;
      if (leaf < constraint_min) {
        return std::abs(constraint_min * numerator) / denominator;
      } else if (leaf > constraint_max) {
        return std::abs(constraint_max * numerator) / denominator;
      }
    }

    // grad^2 / hessian
    return numerator * numerator / denominator;
  }

  // Leaf value without any constraint applied.
  double LeafNoConstraints() const {
    const double numerator = l1_threshold(sum_gradient, hessian_l1);
    const double denominator =
        std::max(sum_hessian, kMinHessianForNewtonStep) + hessian_l2;

    // grad / hessian
    return numerator / denominator;
  }

  double WeightedNumExamples() const { return sum_weights; }

  void SetRegularization(double l1, double l2) {
    hessian_l1 = l1;
    hessian_l2 = l2;
  }

  void Clear() {
    sum_gradient = 0.;
    sum_hessian = 0.;
    sum_weights = 0.;
  }

  template <typename T>
  void Set(const T gradient, const T hessian, const T weights) {
    this->sum_gradient = gradient;
    this->sum_hessian = hessian;
    this->sum_weights = weights;
  }

  template <typename T>
  void Add(const T gradient, const T hessian, const T weights) {
    sum_gradient += gradient;
    sum_hessian += hessian;
    sum_weights += weights;
  }

  template <typename T>
  void Sub(const T gradient, const T hessian, const T weights) {
    sum_gradient -= gradient;
    sum_hessian -= hessian;
    sum_weights -= weights;
  }

  double sum_gradient;
  double sum_hessian;
  double sum_weights;

  // Regularization parameters.
  double hessian_l1;
  double hessian_l2;

  // Optional constraint on the leaf values.
  // If set, constraints.min_max.has_value() is true.
  NodeConstraints constraints;
};

// ===============
// Label Buckets
// ===============
//
// LabelBucket accumulate statistics about labels in a bucket.
//
// LabelBuckets may be templated to improve performance for special cases such
// as unweighted datasets.
//
// LabelBuckets should implement the following methods:
//
// Add a bucket to an accumulator.
// void AddToScoreAcc(ScoreAccumulator* acc) const;
//
// Remove a bucket from an accumulator.
// void SubToScoreAcc(ScoreAccumulator* acc) const;
//
// Ordering between bucket. Used to sort the buckets if
// "require_label_sorting=true".
// bool operator<(const LabelBucket& other) const;
//
// LabelFillers control the accumulation of statistics in a label bucket. They
// need to implement the following methods:
//
// Initialize and zero a bucket before scanning. "acc" might be a previously
// used bucket.
// void InitializeAndZero(LabelBucket* acc) const;
//
// Add the statistics about one examples to the bucket.
// void ConsumeExample(const UnsignedExampleIdx example_idx, LabelBucket* acc)
// const;
//
// Initialize and empty an accumulator.
// void InitEmpty(ScoreAccumulator* acc) const;
//
// Initialize an accumulator and set it to contain all the training examples.
// void InitFull(ScoreAccumulator* acc) const;
//
// Normalize the score of the bucket. The final split score is:
// NormalizeScore(weighted sum of the positive and negative accumulators).
// double NormalizeScore(const double score) const;
//

template <bool weighted>
struct LabelNumericalOneValueBucket {
  typedef typename std::conditional_t<weighted, internal::FloatValueAndWeight,
                                      internal::FloatValueOnly>
      ValueAndMaybeWeight;
  ValueAndMaybeWeight content;
  static constexpr int count = 1;  // NOLINT

  void AddToScoreAcc(LabelNumericalScoreAccumulator* acc) const {
    if constexpr (weighted) {
      acc->label.Add(content.value, content.weight);
    } else {
      acc->label.Add(content.value);
    }
  }

  void SubToScoreAcc(LabelNumericalScoreAccumulator* acc) const {
    if constexpr (weighted) {
      acc->label.Sub(content.value, content.weight);
    } else {
      acc->label.Sub(content.value);
    }
  }

  class Initializer {
   public:
    Initializer(const utils::NormalDistributionDouble& label_distribution)
        : label_distribution_(label_distribution),
          initial_variance_time_weight_(
              label_distribution.VarTimesSumWeights()),
          sum_weights_(label_distribution.NumObservations()) {}

    void InitEmpty(LabelNumericalScoreAccumulator* acc) const {
      acc->label.Clear();
    }

    void InitFull(LabelNumericalScoreAccumulator* acc) const {
      acc->label = label_distribution_;
    }

    double NormalizeScore(const double score) const {
      return (initial_variance_time_weight_ - score) / sum_weights_;
    }

    bool IsValidSplit(const LabelNumericalScoreAccumulator& neg,
                      const LabelNumericalScoreAccumulator& pos) const {
      return true;
    }

    double MinimumScore() const { return 0; }

   private:
    const utils::NormalDistributionDouble& label_distribution_;
    const double initial_variance_time_weight_;
    const double sum_weights_;
  };

  class Filler {
   public:
    Filler(const std::vector<float>& label, const std::vector<float>& weights)
        : label_(label), weights_(weights) {
      if constexpr (weighted) {
        DCHECK_EQ(weights.size(), label.size());
      } else {
        DCHECK(weights.empty());
      }
    }

    void InitializeAndZero(LabelNumericalOneValueBucket* acc) const {}

    void Finalize(LabelNumericalOneValueBucket* acc) const {}

    void ConsumeExample(const UnsignedExampleIdx example_idx,
                        LabelNumericalOneValueBucket* acc) const {
      acc->content.value = label_[example_idx];
      if constexpr (weighted) {
        acc->content.weight = weights_[example_idx];
      }
    }

    template <typename ExampleIdx>
    void AddDirectToScoreAcc(const ExampleIdx example_idx,
                             LabelNumericalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->label.Add(label_[example_idx], weights_[example_idx]);
      } else {
        acc->label.Add(label_[example_idx]);
      }
    }

    template <typename ExampleIdx>
    void SubDirectToScoreAcc(const ExampleIdx example_idx,
                             LabelNumericalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->label.Sub(label_[example_idx], weights_[example_idx]);
      } else {
        acc->label.Sub(label_[example_idx]);
      }
    }

    template <typename ExampleIdx>
    void AddDirectToScoreAccWithDuplicates(
        const ExampleIdx example_idx, const int num_duplicates,
        LabelNumericalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->label.Add(label_[example_idx],
                       weights_[example_idx] * num_duplicates);
      } else {
        acc->label.Add(label_[example_idx], static_cast<float>(num_duplicates));
      }
    }

    template <typename ExampleIdx>
    void SubDirectToScoreAccWithDuplicates(
        const ExampleIdx example_idx, const int num_duplicates,
        LabelNumericalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->label.Sub(label_[example_idx],
                       weights_[example_idx] * num_duplicates);
      } else {
        acc->label.Sub(label_[example_idx], static_cast<float>(num_duplicates));
      }
    }

    template <typename ExampleIdx>
    void Prefetch(const ExampleIdx example_idx) const {
      PREFETCH(&label_[example_idx]);
      if constexpr (weighted) PREFETCH(&weights_[example_idx]);
    }

   private:
    const std::vector<float>& label_;
    const std::vector<float>& weights_;
  };

  friend std::ostream& operator<<(
      std::ostream& os, const LabelNumericalOneValueBucket<weighted>& data);
};

inline std::ostream& operator<<(
    std::ostream& os,
    const LabelNumericalOneValueBucket</*weighted=*/true>& data) {
  os << "value:" << data.content.value << " weight:" << data.content.weight
     << " count:" << data.count;
  return os;
}
inline std::ostream& operator<<(
    std::ostream& os,
    const LabelNumericalOneValueBucket</*weighted=*/false>& data) {
  os << "value:" << data.content.value << " count:" << data.count;
  return os;
}

template <bool weighted>
struct LabelHessianNumericalOneValueBucket {
  typedef typename std::conditional_t<weighted,
                                      internal::FloatGradientHessianAndWeight,
                                      internal::FloatGradientHessianOnly>
      GradientHessianAndMaybeWeight;
  GradientHessianAndMaybeWeight content;
  static constexpr int count = 1;  // NOLINT

  void AddToScoreAcc(LabelHessianNumericalScoreAccumulator* acc) const {
    if constexpr (weighted) {
      acc->Add(content.gradient, content.hessian, content.weight);
    } else {
      acc->Add(content.gradient, content.hessian, 1.f);
    }
  }

  void SubToScoreAcc(LabelHessianNumericalScoreAccumulator* acc) const {
    if constexpr (weighted) {
      acc->Sub(content.gradient, content.hessian, content.weight);
    } else {
      acc->Sub(content.gradient, content.hessian, 1.f);
    }
  }

  class Initializer {
   public:
    Initializer(const double sum_gradient, const double sum_hessian,
                const double sum_weights, const double hessian_l1,
                const double hessian_l2,
                const bool hessian_split_score_subtract_parent,
                const int8_t monotonic_direction,
                const NodeConstraints& constraints)
        : sum_gradient_(sum_gradient),
          sum_hessian_(sum_hessian),
          sum_weights_(sum_weights),
          hessian_l1_(hessian_l1),
          hessian_l2_(hessian_l2),
          monotonic_direction_(monotonic_direction),
          constraints_(constraints) {
      const double sum_gradient_l1 = l1_threshold(sum_gradient, hessian_l1);
      const auto parent_score =
          (sum_gradient_l1 * sum_gradient_l1) / (sum_hessian + hessian_l2);
      if (hessian_split_score_subtract_parent) {
        parent_score_ = parent_score;
        min_score_ = 0;
      } else {
        parent_score_ = 0;
        min_score_ = parent_score;
      }
    }

    void InitEmpty(LabelHessianNumericalScoreAccumulator* acc) const {
      acc->Clear();
      acc->SetRegularization(hessian_l1_, hessian_l2_);
      acc->constraints = constraints_;
    }

    void InitFull(LabelHessianNumericalScoreAccumulator* acc) const {
      acc->Set(sum_gradient_, sum_hessian_, sum_weights_);
      acc->SetRegularization(hessian_l1_, hessian_l2_);
      acc->constraints = constraints_;
    }

    double NormalizeScore(const double score) const {
      return score - parent_score_;
    }

    bool IsValidSplit(const LabelHessianNumericalScoreAccumulator& neg,
                      const LabelHessianNumericalScoreAccumulator& pos) const {
      if (monotonic_direction_ != 0) {
        const bool pos_is_greater =
            pos.LeafNoConstraints() >= neg.LeafNoConstraints();
        return pos_is_greater == (monotonic_direction_ == 1);
      }
      return true;
    }

    double MinimumScore() const { return min_score_; }

   private:
    const double sum_gradient_;
    const double sum_hessian_;
    const double sum_weights_;
    const double hessian_l1_;
    const double hessian_l2_;
    double parent_score_;
    double min_score_;

    // +1/-1 if the feature is monotonic increasing / decreasing. 0 if the
    // feature is not constrained.
    const int8_t monotonic_direction_ = 0;

    // Constraints on the leaf.
    const NodeConstraints& constraints_;
  };

  class Filler {
   public:
    Filler(const std::vector<float>& gradients,
           const std::vector<float>& hessians,
           const std::vector<float>& weights)
        : gradients_(gradients), hessians_(hessians), weights_(weights) {
      if constexpr (weighted) {
        DCHECK_GE(weights.size(), gradients.size());
      } else {
        DCHECK(weights.empty());
      }
    }

    void InitializeAndZero(LabelHessianNumericalOneValueBucket* acc) const {}

    void Finalize(LabelHessianNumericalOneValueBucket* acc) const {}

    void ConsumeExample(const UnsignedExampleIdx example_idx,
                        LabelHessianNumericalOneValueBucket* acc) const {
      acc->content.gradient = gradients_[example_idx];
      acc->content.hessian = hessians_[example_idx];
      if constexpr (weighted) {
        acc->content.weight = weights_[example_idx];
      }
    }

    template <typename ExampleIdx>
    void AddDirectToScoreAcc(const ExampleIdx example_idx,
                             LabelHessianNumericalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->Add(gradients_[example_idx], hessians_[example_idx],
                 weights_[example_idx]);
      } else {
        acc->Add(gradients_[example_idx], hessians_[example_idx], 1.f);
      }
    }

    template <typename ExampleIdx>
    void SubDirectToScoreAcc(const ExampleIdx example_idx,
                             LabelHessianNumericalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->Sub(gradients_[example_idx], hessians_[example_idx],
                 weights_[example_idx]);
      } else {
        acc->Sub(gradients_[example_idx], hessians_[example_idx], 1.f);
      }
    }

    template <typename ExampleIdx>
    void AddDirectToScoreAccWithDuplicates(
        const ExampleIdx example_idx, const int num_duplicates,
        LabelHessianNumericalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->Add(gradients_[example_idx], hessians_[example_idx],
                 weights_[example_idx] * num_duplicates);
      } else {
        acc->Add<float>(gradients_[example_idx], hessians_[example_idx],
                        num_duplicates);
      }
    }

    template <typename ExampleIdx>
    void SubDirectToScoreAccWithDuplicates(
        const ExampleIdx example_idx, const int num_duplicates,
        LabelHessianNumericalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->Sub(gradients_[example_idx], hessians_[example_idx],
                 weights_[example_idx] * num_duplicates);
      } else {
        acc->Sub<float>(gradients_[example_idx], hessians_[example_idx],
                        num_duplicates);
      }
    }

    template <typename ExampleIdx>
    void Prefetch(const ExampleIdx example_idx) const {
      PREFETCH(&gradients_[example_idx]);
      PREFETCH(&hessians_[example_idx]);
      if constexpr (weighted) {
        PREFETCH(&weights_[example_idx]);
      }
    }

   private:
    const std::vector<float>& gradients_;
    const std::vector<float>& hessians_;
    const std::vector<float>& weights_;
  };
};

inline std::ostream& operator<<(
    std::ostream& os, const LabelHessianNumericalOneValueBucket<false>& data) {
  os << "gradient:" << data.content.gradient
     << " hessian:" << data.content.hessian << " count:" << data.count;
  return os;
}

inline std::ostream& operator<<(
    std::ostream& os, const LabelHessianNumericalOneValueBucket<true>& data) {
  os << "gradient:" << data.content.gradient
     << " hessian:" << data.content.hessian << " weight:" << data.content.weight
     << " count:" << data.count;
  return os;
}

template <bool weighted>
struct LabelCategoricalOneValueBucket {
  typedef typename std::conditional_t<weighted, internal::IntegerValueAndWeight,
                                      internal::IntegerValueOnly>
      ValueAndMaybeWeight;
  ValueAndMaybeWeight content;

  // Not called "kCount" because this is used as a template parameter and
  // expects the name to be `count` (in other such structs it is not a
  // constant).
  static constexpr int count = 1;  // NOLINT

  void AddToScoreAcc(LabelCategoricalScoreAccumulator* acc) const {
    if constexpr (weighted) {
      acc->label.Add(content.value, content.weight);
    } else {
      acc->label.Add(content.value);
    }
  }

  void SubToScoreAcc(LabelCategoricalScoreAccumulator* acc) const {
    if constexpr (weighted) {
      acc->label.Sub(content.value, content.weight);
    } else {
      acc->label.Sub(content.value);
    }
  }

  class Initializer {
   public:
    Initializer(const utils::IntegerDistributionDouble& label_distribution)
        : label_distribution_(label_distribution),
          initial_entropy_(label_distribution.Entropy()) {}

    void InitEmpty(LabelCategoricalScoreAccumulator* acc) const {
      acc->label.SetNumClasses(label_distribution_.NumClasses());
      acc->label.Clear();
    }

    void InitFull(LabelCategoricalScoreAccumulator* acc) const {
      acc->label = label_distribution_;
    }

    double NormalizeScore(const double score) const {
      return initial_entropy_ - score;
    }

    bool IsValidSplit(const LabelCategoricalScoreAccumulator& neg,
                      const LabelCategoricalScoreAccumulator& pos) const {
      return true;
    }

    double MinimumScore() const { return 0; }

   private:
    const utils::IntegerDistributionDouble& label_distribution_;
    const double initial_entropy_;
  };

  class Filler {
   public:
    Filler(const std::vector<int>& label, const std::vector<float>& weights)
        : label_(label), weights_(weights) {
      if constexpr (weighted) {
        DCHECK_EQ(weights.size(), label.size());
      } else {
        DCHECK(weights.empty());
      }
    }

    void InitializeAndZero(
        LabelCategoricalOneValueBucket<weighted>* bucket) const {}

    void Finalize(LabelCategoricalOneValueBucket<weighted>* bucket) const {}

    void ConsumeExample(
        const UnsignedExampleIdx example_idx,
        LabelCategoricalOneValueBucket<weighted>* bucket) const {
      bucket->content.value = label_[example_idx];
      if constexpr (weighted) {
        bucket->content.weight = weights_[example_idx];
      }
    }

    template <typename ExampleIdx>
    void AddDirectToScoreAcc(const ExampleIdx example_idx,
                             LabelCategoricalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->label.Add(label_[example_idx], weights_[example_idx]);
      } else {
        acc->label.Add(label_[example_idx]);
      }
    }

    template <typename ExampleIdx>
    void SubDirectToScoreAcc(const ExampleIdx example_idx,
                             LabelCategoricalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->label.Sub(label_[example_idx], weights_[example_idx]);
      } else {
        acc->label.Sub(label_[example_idx]);
      }
    }

    template <typename ExampleIdx>
    void AddDirectToScoreAccWithDuplicates(
        const ExampleIdx example_idx, const int num_duplicates,
        LabelCategoricalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->label.Add(label_[example_idx],
                       weights_[example_idx] * num_duplicates);
      } else {
        acc->label.Add(label_[example_idx], num_duplicates);
      }
    }

    template <typename ExampleIdx>
    void SubDirectToScoreAccWithDuplicates(
        const ExampleIdx example_idx, const int num_duplicates,
        LabelCategoricalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->label.Sub(label_[example_idx],
                       weights_[example_idx] * num_duplicates);
      } else {
        acc->label.Sub(label_[example_idx], num_duplicates);
      }
    }

    template <typename ExampleIdx>
    void Prefetch(const ExampleIdx example_idx) const {
      PREFETCH(&label_[example_idx]);
      if constexpr (weighted) {
        PREFETCH(&weights_[example_idx]);
      }
    }

   private:
    const std::vector<int>& label_;
    const std::vector<float>& weights_;
  };

  friend std::ostream& operator<<(std::ostream& os,
                                  const LabelCategoricalOneValueBucket& data);
};

inline std::ostream& operator<<(
    std::ostream& os,
    const LabelCategoricalOneValueBucket</*weighted=*/true>& data) {
  os << "value:" << data.content.value << " weight:" << data.content.weight
     << " count:" << data.count;
  return os;
}

inline std::ostream& operator<<(
    std::ostream& os,
    const LabelCategoricalOneValueBucket</*weighted=*/false>& data) {
  os << "value:" << data.content.value << " count:" << data.count;
  return os;
}

template <bool weighted>
struct LabelBinaryCategoricalOneValueBucket {
  typedef typename std::conditional_t<weighted, internal::BooleanValueAndWeight,
                                      internal::BooleanValueOnly>
      ValueAndMaybeWeight;
  ValueAndMaybeWeight content;

  // Not called "kCount" because this is used as a template parameter and
  // expects the name to be `count` (in other such structs it is not a
  // constant).
  static constexpr int count = 1;  // NOLINT

  void AddToScoreAcc(LabelBinaryCategoricalScoreAccumulator* acc) const {
    if constexpr (weighted) {
      acc->AddOne(content.value, content.weight);
    } else {
      acc->AddOne(content.value);
    }
  }

  void SubToScoreAcc(LabelBinaryCategoricalScoreAccumulator* acc) const {
    if constexpr (weighted) {
      acc->SubOne(content.value, content.weight);
    } else {
      acc->SubOne(content.value);
    }
  }

  class Initializer {
   public:
    Initializer(const utils::IntegerDistributionDouble& label_distribution) {
      DCHECK_EQ(label_distribution.NumClasses(), 3);
      label_distribution_trues_ = label_distribution.count(2);
      label_distribution_weights_ = label_distribution.NumObservations();
      initial_entropy_ = utils::BinaryDistributionEntropyF(
          label_distribution_trues_ / label_distribution_weights_);
      DCHECK(std::abs(initial_entropy_ - label_distribution.Entropy()) <=
             0.0001);
    }

    void InitEmpty(LabelBinaryCategoricalScoreAccumulator* acc) const {
      acc->Clear();
    }

    void InitFull(LabelBinaryCategoricalScoreAccumulator* acc) const {
      acc->Set(label_distribution_trues_, label_distribution_weights_);
    }

    double NormalizeScore(const double score) const {
      return initial_entropy_ - score;
    }

    bool IsValidSplit(const LabelBinaryCategoricalScoreAccumulator& neg,
                      const LabelBinaryCategoricalScoreAccumulator& pos) const {
      return true;
    }

    double MinimumScore() const { return 0; }

   private:
    double label_distribution_trues_;
    double label_distribution_weights_;
    double initial_entropy_;
  };

  class Filler {
   public:
    Filler(const std::vector<int>& label, const std::vector<float>& weights)
        : label_(label), weights_(weights) {
      if constexpr (weighted) {
        DCHECK_EQ(weights.size(), label.size());
      } else {
        DCHECK(weights.empty());
      }
    }

    void InitializeAndZero(
        LabelBinaryCategoricalOneValueBucket<weighted>* bucket) const {}

    void Finalize(
        LabelBinaryCategoricalOneValueBucket<weighted>* bucket) const {}

    void ConsumeExample(
        const UnsignedExampleIdx example_idx,
        LabelBinaryCategoricalOneValueBucket<weighted>* bucket) const {
      bucket->content.value = label_[example_idx] == 2;
      if constexpr (weighted) {
        bucket->content.weight = weights_[example_idx];
      }
    }

    template <typename ExampleIdx>
    void AddDirectToScoreAcc(
        const ExampleIdx example_idx,
        LabelBinaryCategoricalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->AddOne(label_[example_idx] == 2, weights_[example_idx]);
      } else {
        acc->AddOne(label_[example_idx] == 2);
      }
    }

    template <typename ExampleIdx>
    void SubDirectToScoreAcc(
        const ExampleIdx example_idx,
        LabelBinaryCategoricalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->SubOne(label_[example_idx] == 2, weights_[example_idx]);
      } else {
        acc->SubOne(label_[example_idx] == 2);
      }
    }

    template <typename ExampleIdx>
    void AddDirectToScoreAccWithDuplicates(
        const ExampleIdx example_idx, const int num_duplicates,
        LabelBinaryCategoricalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->AddOne(label_[example_idx] == 2,
                    weights_[example_idx] * num_duplicates);
      } else {
        acc->AddOne(label_[example_idx] == 2, num_duplicates);
      }
    }

    template <typename ExampleIdx>
    void SubDirectToScoreAccWithDuplicates(
        const ExampleIdx example_idx, const int num_duplicates,
        LabelBinaryCategoricalScoreAccumulator* acc) const {
      if constexpr (weighted) {
        acc->SubOne(label_[example_idx] == 2,
                    weights_[example_idx] * num_duplicates);
      } else {
        acc->SubOne(label_[example_idx] == 2, num_duplicates);
      }
    }

    template <typename ExampleIdx>
    void Prefetch(const ExampleIdx example_idx) const {
      PREFETCH(&label_[example_idx]);
      if constexpr (weighted) {
        PREFETCH(&weights_[example_idx]);
      }
    }

   private:
    const std::vector<int>& label_;
    const std::vector<float>& weights_;
  };

  friend std::ostream& operator<<(
      std::ostream& os, const LabelBinaryCategoricalOneValueBucket& data);
};

inline std::ostream& operator<<(
    std::ostream& os, const LabelBinaryCategoricalOneValueBucket<true>& data) {
  os << "value:" << data.content.value << " weight:" << data.content.weight
     << " count:" << data.count;
  return os;
}

inline std::ostream& operator<<(
    std::ostream& os, const LabelBinaryCategoricalOneValueBucket<false>& data) {
  os << "value:" << data.content.value << " count:" << data.count;
  return os;
}

template <bool weighted>
struct LabelNumericalBucket {
  utils::NormalDistributionDouble value;
  int64_t count;

  void AddToScoreAcc(LabelNumericalScoreAccumulator* acc) const {
    acc->label.Add(value);
  }

  void SubToScoreAcc(LabelNumericalScoreAccumulator* acc) const {
    acc->label.Sub(value);
  }

  void AddToBucket(LabelNumericalBucket* dst) const {
    dst->value.Add(value);
    dst->count += count;
  }

  bool operator<(const LabelNumericalBucket& other) const {
    return value.Mean() < other.value.Mean();
  }

  class Initializer {
   public:
    Initializer(const utils::NormalDistributionDouble& label_distribution)
        : label_distribution_(label_distribution),
          initial_variance_time_weight_(
              label_distribution.VarTimesSumWeights()),
          sum_weights_(label_distribution.NumObservations()) {}

    explicit Initializer(const proto::LabelStatistics& statistics) {
      label_distribution_.Load(statistics.regression().labels());
      initial_variance_time_weight_ = label_distribution_.VarTimesSumWeights();
      sum_weights_ = label_distribution_.NumObservations();
    }

    void InitEmpty(LabelNumericalScoreAccumulator* acc) const {
      acc->label.Clear();
    }

    void InitFull(LabelNumericalScoreAccumulator* acc) const {
      acc->label = label_distribution_;
    }

    double NormalizeScore(const double score) const {
      return (initial_variance_time_weight_ - score) / sum_weights_;
    }

    bool IsValidSplit(const LabelNumericalScoreAccumulator& neg,
                      const LabelNumericalScoreAccumulator& pos) const {
      return true;
    }

    double MinimumScore() const { return 0; }

   private:
    utils::NormalDistributionDouble label_distribution_;
    double initial_variance_time_weight_;
    double sum_weights_;
  };

  class Filler {
   public:
    Filler(const std::vector<float>& label, const std::vector<float>& weights)
        : label_(label), weights_(weights) {
      if constexpr (weighted) {
        DCHECK_EQ(weights.size(), label.size());
      } else {
        DCHECK(weights.empty());
      }
    }

    void InitializeAndZero(LabelNumericalBucket* acc) const {
      acc->value.Clear();
      acc->count = 0;
    }

    void Finalize(LabelNumericalBucket* acc) const {}

    void ConsumeExample(const UnsignedExampleIdx example_idx,
                        LabelNumericalBucket* acc) const {
      if constexpr (weighted) {
        acc->value.Add(label_[example_idx], weights_[example_idx]);
      } else {
        acc->value.Add(label_[example_idx]);
      }
      acc->count++;
    }

   private:
    const std::vector<float>& label_;
    const std::vector<float>& weights_;
  };
};

inline std::ostream& operator<<(std::ostream& os,
                                const LabelNumericalBucket<false>& data) {
  os << "value:{mean:" << data.value.Mean()
     << " obs:" << data.value.NumObservations() << "} count:" << data.count;
  return os;
}
inline std::ostream& operator<<(std::ostream& os,
                                const LabelNumericalBucket<true>& data) {
  os << "value:{mean:" << data.value.Mean()
     << " obs:" << data.value.NumObservations() << "} count:" << data.count;
  return os;
}

template <bool weighted>
struct LabelNumericalWithHessianBucket {
  utils::NormalDistributionDouble value;
  double sum_hessian;
  int64_t count;

  void AddToScoreAcc(LabelNumericalWithHessianScoreAccumulator* acc) const {
    acc->label.Add(value);
    acc->sum_hessian += sum_hessian;
  }

  void SubToScoreAcc(LabelNumericalWithHessianScoreAccumulator* acc) const {
    acc->label.Sub(value);
    acc->sum_hessian -= sum_hessian;
  }

  void AddToBucket(LabelNumericalWithHessianBucket* dst) const {
    dst->value.Add(value);
    dst->count += count;
    dst->sum_hessian += sum_hessian;
  }

  bool operator<(const LabelNumericalWithHessianBucket& other) const {
    return value.Mean() < other.value.Mean();
  }

  class Initializer {
   public:
    explicit Initializer(const proto::LabelStatistics& statistics) {
      label_distribution_.Load(statistics.regression_with_hessian().labels());
      initial_variance_time_weight_ = label_distribution_.VarTimesSumWeights();
      sum_weights_ = label_distribution_.NumObservations();
      sum_hessian_ = statistics.regression_with_hessian().sum_hessian();
    }

    void InitEmpty(LabelNumericalWithHessianScoreAccumulator* acc) const {
      acc->label.Clear();
      acc->sum_hessian = 0;
    }

    void InitFull(LabelNumericalWithHessianScoreAccumulator* acc) const {
      acc->label = label_distribution_;
      acc->sum_hessian = sum_hessian_;
    }

    double NormalizeScore(const double score) const {
      return (initial_variance_time_weight_ - score) / sum_weights_;
    }

    bool IsValidSplit(
        const LabelNumericalWithHessianScoreAccumulator& neg,
        const LabelNumericalWithHessianScoreAccumulator& pos) const {
      return true;
    }

    double MinimumScore() const { return 0; }

   private:
    utils::NormalDistributionDouble label_distribution_;
    double initial_variance_time_weight_;
    double sum_weights_;
    double sum_hessian_;
  };

  friend std::ostream& operator<<(
      std::ostream& os, const LabelNumericalWithHessianBucket<weighted>& data);
};

inline std::ostream& operator<<(
    std::ostream& os, const LabelNumericalWithHessianBucket<false>& data) {
  os << "value:{mean:" << data.value.Mean()
     << " obs:" << data.value.NumObservations() << "} count:" << data.count;
  return os;
}
inline std::ostream& operator<<(
    std::ostream& os, const LabelNumericalWithHessianBucket<true>& data) {
  os << "value:{mean:" << data.value.Mean()
     << " obs:" << data.value.NumObservations() << "} count:" << data.count;
  return os;
}

template <bool weighted>
struct LabelHessianNumericalBucket {
  // The priority is defined as ~ "sum_gradient / sum_hessian" (with extra
  // regularization and check of sum_hessian = 0.
  // This value is computed from doubles, and then simply compared (i.e. this
  // is not an accumulator). In memory, it will get aligned to the feature
  // bucket of size 4 bytes.
  float priority;

  typedef
      typename std::conditional_t<weighted,
                                  internal::FloatSumGradientHessianAndWeight,
                                  internal::FloatSumGradientHessianOnly>
          SumGradientHessianAndMaybeWeights;
  SumGradientHessianAndMaybeWeights content;
  int64_t count;

  void AddToScoreAcc(LabelHessianNumericalScoreAccumulator* acc) const {
    if constexpr (weighted) {
      acc->Add(content.sum_gradient, content.sum_hessian, content.sum_weight);
    } else {
      acc->Add(content.sum_gradient, content.sum_hessian, 1.f);
    }
  }

  void SubToScoreAcc(LabelHessianNumericalScoreAccumulator* acc) const {
    if constexpr (weighted) {
      acc->Sub(content.sum_gradient, content.sum_hessian, content.sum_weight);
    } else {
      acc->Sub(content.sum_gradient, content.sum_hessian, 1.f);
    }
  }

  bool operator<(const LabelHessianNumericalBucket& other) const {
    return priority < priority;
  }

  class Initializer {
   public:
    Initializer(const double sum_gradient, const double sum_hessian,
                const double sum_weights, const double hessian_l1,
                const double hessian_l2,
                const bool hessian_split_score_subtract_parent,
                const int8_t monotonic_direction,
                const NodeConstraints& constraints)
        : sum_gradient_(sum_gradient),
          sum_hessian_(sum_hessian),
          sum_weights_(sum_weights),
          hessian_l1_(hessian_l1),
          hessian_l2_(hessian_l2),
          monotonic_direction_(monotonic_direction),
          constraints_(constraints) {
      const double sum_gradient_l1 = l1_threshold(sum_gradient, hessian_l1);
      const auto parent_score =
          (sum_gradient_l1 * sum_gradient_l1) / (sum_hessian + hessian_l2);
      if (hessian_split_score_subtract_parent) {
        parent_score_ = parent_score;
        min_score_ = 0;
      } else {
        parent_score_ = 0;
        min_score_ = parent_score;
      }
    }

    void InitEmpty(LabelHessianNumericalScoreAccumulator* acc) const {
      acc->Clear();
      acc->SetRegularization(hessian_l1_, hessian_l2_);
      acc->constraints = constraints_;
    }

    void InitFull(LabelHessianNumericalScoreAccumulator* acc) const {
      acc->Set(sum_gradient_, sum_hessian_, sum_weights_);
      acc->SetRegularization(hessian_l1_, hessian_l2_);
      acc->constraints = constraints_;
    }

    double NormalizeScore(const double score) const {
      return score - parent_score_;
    }

    bool IsValidSplit(const LabelHessianNumericalScoreAccumulator& neg,
                      const LabelHessianNumericalScoreAccumulator& pos) const {
      if (monotonic_direction_ != 0) {
        const bool pos_is_greater =
            pos.LeafNoConstraints() >= neg.LeafNoConstraints();
        return pos_is_greater == (monotonic_direction_ == 1);
      }
      return true;
    }

    double MinimumScore() const { return min_score_; }

   private:
    const double sum_gradient_;
    const double sum_hessian_;
    const double sum_weights_;
    const double hessian_l1_;
    const double hessian_l2_;
    double parent_score_;
    double min_score_;

    // +1/-1 if the feature is monotonic increasing / decreasing. 0 if the
    // feature is not constrained.
    const int8_t monotonic_direction_ = 0;

    // Constraints on the leaf.
    const NodeConstraints& constraints_;
  };

  class Filler {
   public:
    Filler(const std::vector<float>& gradients,
           const std::vector<float>& hessians,
           const std::vector<float>& weights, const double hessian_l1,
           const double hessian_l2)
        : gradients_(gradients),
          hessians_(hessians),
          weights_(weights),
          hessian_l1_(hessian_l1),
          hessian_l2_(hessian_l2) {
      if constexpr (weighted) {
        DCHECK_GE(weights.size(), gradients.size());
      } else {
        DCHECK(weights.empty());
      }
    }

    void InitializeAndZero(LabelHessianNumericalBucket* acc) const {
      acc->content.sum_gradient = 0;
      acc->content.sum_hessian = 0;
      if constexpr (weighted) {
        acc->content.sum_weight = 0;
      }
      acc->count = 0;
    }

    void Finalize(LabelHessianNumericalBucket* acc) const {
      if (acc->content.sum_hessian > 0) {
        acc->priority = l1_threshold(acc->content.sum_gradient, hessian_l1_) /
                        (acc->content.sum_hessian + hessian_l2_);
      } else {
        acc->priority = 0.;
      }
    }

    void ConsumeExample(const UnsignedExampleIdx example_idx,
                        LabelHessianNumericalBucket* acc) const {
      acc->content.sum_gradient += gradients_[example_idx];
      acc->content.sum_hessian += hessians_[example_idx];
      if constexpr (weighted) {
        acc->content.sum_weight += weights_[example_idx];
      }
      acc->count++;
    }

   private:
    const std::vector<float>& gradients_;
    const std::vector<float>& hessians_;
    const std::vector<float>& weights_;

    const double hessian_l1_;
    const double hessian_l2_;
  };
};

inline std::ostream& operator<<(
    std::ostream& os, const LabelHessianNumericalBucket<false>& data) {
  os << "value:{sum_gradient:" << data.content.sum_gradient
     << " sum_hessian:" << data.content.sum_hessian << "} count:" << data.count;
  return os;
}
inline std::ostream& operator<<(std::ostream& os,
                                const LabelHessianNumericalBucket<true>& data) {
  os << "value:{sum_gradient:" << data.content.sum_gradient
     << " sum_hessian:" << data.content.sum_hessian
     << " sum_weight:" << data.content.sum_weight << "} count:" << data.count;
  return os;
}

template <bool weighted>
struct LabelCategoricalBucket {
  utils::IntegerDistributionDouble value;
  int64_t count;

  void AddToScoreAcc(LabelCategoricalScoreAccumulator* acc) const {
    acc->label.Add(value);
  }

  void SubToScoreAcc(LabelCategoricalScoreAccumulator* acc) const {
    acc->label.Sub(value);
  }

  void AddToBucket(LabelCategoricalBucket* dst) const {
    dst->value.Add(value);
    dst->count += count;
  }

  float SafeProportionOrMinusInfinity(int idx) const {
    return value.SafeProportionOrMinusInfinity(idx);
  }

  class Initializer {
   public:
    Initializer(const utils::IntegerDistributionDouble& label_distribution)
        : non_owned_label_distribution_(&label_distribution),
          initial_entropy_(label_distribution.Entropy()) {}

    explicit Initializer(const proto::LabelStatistics& statistics) {
      owned_label_distribution_ = utils::IntegerDistributionDouble();
      owned_label_distribution_->Load(statistics.classification().labels());
      initial_entropy_ = owned_label_distribution_->Entropy();
    }

    const utils::IntegerDistributionDouble& label_distribution() const {
      if (non_owned_label_distribution_) {
        return *non_owned_label_distribution_;
      }
      return owned_label_distribution_.value();
    }

    void InitEmpty(LabelCategoricalScoreAccumulator* acc) const {
      acc->label.Clear();
      acc->label.SetNumClasses(label_distribution().NumClasses());
    }

    void InitFull(LabelCategoricalScoreAccumulator* acc) const {
      acc->label = label_distribution();
    }

    double NormalizeScore(const double score) const {
      return initial_entropy_ - score;
    }

    bool IsEmpty(const int32_t idx) const {
      return label_distribution().count(idx) == 0;
    }

    bool IsValidSplit(const LabelCategoricalScoreAccumulator& neg,
                      const LabelCategoricalScoreAccumulator& pos) const {
      return true;
    }

    double MinimumScore() const { return 0; }

   private:
    const utils::IntegerDistributionDouble* non_owned_label_distribution_ =
        nullptr;
    absl::optional<utils::IntegerDistributionDouble> owned_label_distribution_;
    double initial_entropy_;
  };

  class Filler {
   public:
    Filler(const std::vector<int>& label, const std::vector<float>& weights,
           const utils::IntegerDistributionDouble& label_distribution)
        : label_(label),
          weights_(weights),
          num_classes_(label_distribution.NumClasses()) {
      if constexpr (weighted) {
        DCHECK_EQ(weights.size(), label.size());
      } else {
        DCHECK(weights.empty());
      }
    }

    void InitializeAndZero(LabelCategoricalBucket* acc) const {
      acc->value.Clear();
      acc->value.SetNumClasses(num_classes_);
      acc->count = 0;
    }

    void Finalize(LabelCategoricalBucket* acc) const {}

    void ConsumeExample(const UnsignedExampleIdx example_idx,
                        LabelCategoricalBucket* acc) const {
      if constexpr (weighted) {
        acc->value.Add(label_[example_idx], weights_[example_idx]);
      } else {
        acc->value.Add(label_[example_idx]);
      }
      acc->count++;
    }

   private:
    const std::vector<int>& label_;
    const std::vector<float>& weights_;
    const int num_classes_;
  };

  friend std::ostream& operator<<(std::ostream& os,
                                  const LabelNumericalBucket<weighted>& data);
};

inline std::ostream& operator<<(std::ostream& os,
                                const LabelCategoricalBucket<true>& data) {
  os << "value:{obs:" << data.value.NumObservations()
     << "} count:" << data.count;
  return os;
}

inline std::ostream& operator<<(std::ostream& os,
                                const LabelCategoricalBucket<false>& data) {
  os << "value:{obs:" << data.value.NumObservations()
     << "} count:" << data.count;
  return os;
}

template <bool weighted>
struct LabelBinaryCategoricalBucket {
  typedef typename std::conditional_t<weighted, internal::SumTruesAndWeights,
                                      internal::SumTruesOnly>
      SumTruesAndMaybeWeights;
  SumTruesAndMaybeWeights content;
  int64_t count;

  void AddToScoreAcc(LabelBinaryCategoricalScoreAccumulator* acc) const {
    if constexpr (weighted) {
      acc->AddMany(content.sum_trues, content.sum_weights);
    } else {
      acc->AddMany(content.sum_trues, count);
    }
  }

  void SubToScoreAcc(LabelBinaryCategoricalScoreAccumulator* acc) const {
    if constexpr (weighted) {
      acc->SubMany(content.sum_trues, content.sum_weights);
    } else {
      acc->SubMany(content.sum_trues, count);
    }
  }

  float SafeProportionOrMinusInfinity(int idx) const {
    double sum_trues = content.sum_trues;
    double sum_weights;
    if constexpr (weighted) {
      sum_weights = content.sum_weights;
    } else {
      sum_weights = count;
    }
    if (sum_weights > 0) {
      DCHECK(idx == 1 || idx == 2);
      if (idx == 2) {
        return sum_trues / sum_weights;
      } else {
        return 1.f - sum_trues / sum_weights;
      }
    } else {
      return -std::numeric_limits<float>::infinity();
    }
  }

  class Initializer {
   public:
    Initializer(const utils::IntegerDistributionDouble& label_distribution) {
      DCHECK_EQ(label_distribution.NumClasses(), 3);
      label_distribution_trues_ = label_distribution.count(2);
      label_distribution_weights_ = label_distribution.NumObservations();
      initial_entropy_ = utils::BinaryDistributionEntropyF(
          label_distribution_trues_ / label_distribution_weights_);
      DCHECK(std::abs(initial_entropy_ - label_distribution.Entropy()) <=
             0.0001);
    }

    void InitEmpty(LabelBinaryCategoricalScoreAccumulator* acc) const {
      acc->Clear();
    }

    void InitFull(LabelBinaryCategoricalScoreAccumulator* acc) const {
      acc->Set(label_distribution_trues_, label_distribution_weights_);
    }

    double NormalizeScore(const double score) const {
      return initial_entropy_ - score;
    }

    bool IsValidSplit(const LabelBinaryCategoricalScoreAccumulator& neg,
                      const LabelBinaryCategoricalScoreAccumulator& pos) const {
      return true;
    }

    double MinimumScore() const { return 0; }

   private:
    double label_distribution_trues_;
    double label_distribution_weights_;
    double initial_entropy_;
  };

  class Filler {
   public:
    Filler(const std::vector<int>& label, const std::vector<float>& weights,
           const utils::IntegerDistributionDouble& label_distribution)
        : label_(label), weights_(weights) {
      if constexpr (weighted) {
        DCHECK_EQ(weights.size(), label.size());
      } else {
        DCHECK(weights.empty());
      }
    }

    void InitializeAndZero(
        LabelBinaryCategoricalBucket<weighted>* bucket) const {
      bucket->count = 0;
      bucket->content.sum_trues = 0;
      if constexpr (weighted) {
        bucket->content.sum_weights = 0;
      }
    }

    void Finalize(LabelBinaryCategoricalBucket<weighted>* bucket) const {}

    void ConsumeExample(const UnsignedExampleIdx example_idx,
                        LabelBinaryCategoricalBucket* bucket) const {
      static float table[] = {0.f, 1.f};
      bucket->count++;
      if constexpr (weighted) {
        bucket->content.sum_trues +=
            table[label_[example_idx] == 2] * weights_[example_idx];
        bucket->content.sum_weights += weights_[example_idx];
      } else {
        bucket->content.sum_trues += table[label_[example_idx] == 2];
      }
    }

   private:
    const std::vector<int>& label_;
    const std::vector<float>& weights_;
  };

  friend std::ostream& operator<<(std::ostream& os,
                                  const LabelNumericalBucket<weighted>& data);
};

inline std::ostream& operator<<(
    std::ostream& os, const LabelBinaryCategoricalBucket<true>& data) {
  os << "value:{trues:" << data.content.sum_trues
     << " weights:" << data.content.sum_weights << "} count:" << data.count;
  return os;
}

inline std::ostream& operator<<(
    std::ostream& os, const LabelBinaryCategoricalBucket<false>& data) {
  os << "value:{trues:" << data.content.sum_trues << "} count:" << data.count;
  return os;
}

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPLITTER_ACCUMULATOR_H_
