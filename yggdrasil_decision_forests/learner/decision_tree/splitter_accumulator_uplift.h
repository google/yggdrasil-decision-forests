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

// Classes used by the splitter to accumulate statistics for the uplift tasks.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPLITTER_ACCUMULATOR_UPLIFT_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPLITTER_ACCUMULATOR_UPLIFT_H_

#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

// Distribution of uplift classification labels.
struct UpliftLabelDistribution {
 public:
  typedef proto::DecisionTreeTrainingConfig::Uplift::SplitScore SplitScoreType;
  typedef proto::DecisionTreeTrainingConfig::Uplift::EmptyBucketOrdering
      EmptyBucketOrdering;

  void InitializeAndClearLike(const UpliftLabelDistribution& guide) {
    sum_weights_ = 0;
    sum_weights_per_treatment_.assign(guide.sum_weights_per_treatment_.size(),
                                      0);
    num_examples_per_treatment_.assign(guide.num_examples_per_treatment_.size(),
                                       0);
    sum_weights_per_treatment_and_outcome_.assign(
        guide.sum_weights_per_treatment_and_outcome_.size(), 0);
  }

  void InitializeAndClearCategoricalOutcome(
      const int num_unique_values_in_treatments_column,
      const int num_unique_in_outcomes_column) {
    // The value "0" is reserved.
    const int num_treatments = num_unique_values_in_treatments_column - 1;
    const int num_outcomes = num_unique_in_outcomes_column - 1;

    sum_weights_ = 0;
    sum_weights_per_treatment_.assign(num_treatments, 0);
    num_examples_per_treatment_.assign(num_treatments, 0);
    sum_weights_per_treatment_and_outcome_.assign(
        num_treatments * (num_outcomes - 1), 0);
  }

  void InitializeAndClearNumericalOutcome(
      const int num_unique_values_in_treatments_column) {
    // The value "0" is reserved.
    const int num_treatments = num_unique_values_in_treatments_column - 1;

    sum_weights_ = 0;
    sum_weights_per_treatment_.assign(num_treatments, 0);
    num_examples_per_treatment_.assign(num_treatments, 0);
    sum_weights_per_treatment_and_outcome_.assign(num_treatments, 0);
  }

  void AddCategoricalOutcome(int outcome_value, int treatment_value,
                             const float weight) {
    DCHECK_GE(weight, 0);
    InternalAddCategoricalOutcome(outcome_value, treatment_value, weight, 1);
  }

  void SubCategoricalOutcome(int outcome_value, int treatment_value,
                             const float weight) {
    DCHECK_GE(weight, 0);
    InternalAddCategoricalOutcome(outcome_value, treatment_value, -weight, -1);
  }

  void InternalAddCategoricalOutcome(int outcome_value, int treatment_value,
                                     const float weight,
                                     const SignedExampleIdx num_examples) {
    // Only support binary treatment and binary outcome.
    DCHECK_GE(outcome_value, 1);
    DCHECK_LE(outcome_value, 2);
    DCHECK_GE(treatment_value, 1);
    DCHECK_LE(treatment_value, 2);
    DCHECK_LE(treatment_value, sum_weights_per_treatment_.size());
    DCHECK_LE(treatment_value, num_examples_per_treatment_.size());

    treatment_value -= 1;

    sum_weights_ += weight;
    sum_weights_per_treatment_[treatment_value] += weight;
    num_examples_per_treatment_[treatment_value] += num_examples;

    // The zero outcome is ignored.
    if (outcome_value >= 2) {
      outcome_value -= 2;
      sum_weights_per_treatment_and_outcome_
          [treatment_value +
           outcome_value * sum_weights_per_treatment_.size()] += weight;
    }
  }

  void AddNumericalOutcome(float outcome_value, int treatment_value,
                           const float weight) {
    DCHECK_GE(weight, 0);
    InternalAddNumericalOutcome(outcome_value, treatment_value, weight, 1);
  }

  void SubNumericalOutcome(float outcome_value, int treatment_value,
                           const float weight) {
    DCHECK_GE(weight, 0);
    InternalAddNumericalOutcome(outcome_value, treatment_value, -weight, -1);
  }

  void InternalAddNumericalOutcome(float outcome_value, int treatment_value,
                                   const float weight,
                                   const SignedExampleIdx num_examples) {
    // Only support binary treatment and binary outcome.
    DCHECK_GE(treatment_value, 1);
    DCHECK_LE(treatment_value, 2);
    DCHECK_LE(treatment_value, sum_weights_per_treatment_.size());
    DCHECK_LE(treatment_value, num_examples_per_treatment_.size());

    treatment_value -= 1;

    sum_weights_ += weight;
    sum_weights_per_treatment_[treatment_value] += weight;
    num_examples_per_treatment_[treatment_value] += num_examples;
    sum_weights_per_treatment_and_outcome_[treatment_value] +=
        weight * outcome_value;
  }

  void Add(const UpliftLabelDistribution& src) {
    DCHECK_EQ(sum_weights_per_treatment_.size(),
              src.sum_weights_per_treatment_.size());
    DCHECK_EQ(sum_weights_per_treatment_and_outcome_.size(),
              src.sum_weights_per_treatment_and_outcome_.size());

    sum_weights_ += src.sum_weights_;

    for (size_t i = 0; i < sum_weights_per_treatment_.size(); i++) {
      sum_weights_per_treatment_[i] += src.sum_weights_per_treatment_[i];
      num_examples_per_treatment_[i] += src.num_examples_per_treatment_[i];
    }

    for (size_t i = 0; i < sum_weights_per_treatment_and_outcome_.size(); i++) {
      sum_weights_per_treatment_and_outcome_[i] +=
          src.sum_weights_per_treatment_and_outcome_[i];
    }
  }

  void Sub(const UpliftLabelDistribution& src) {
    DCHECK_EQ(sum_weights_per_treatment_.size(),
              src.sum_weights_per_treatment_.size());
    DCHECK_EQ(sum_weights_per_treatment_and_outcome_.size(),
              src.sum_weights_per_treatment_and_outcome_.size());

    sum_weights_ -= src.sum_weights_;

    for (size_t i = 0; i < sum_weights_per_treatment_.size(); i++) {
      sum_weights_per_treatment_[i] -= src.sum_weights_per_treatment_[i];
      num_examples_per_treatment_[i] -= src.num_examples_per_treatment_[i];
    }

    for (size_t i = 0; i < sum_weights_per_treatment_and_outcome_.size(); i++) {
      sum_weights_per_treatment_and_outcome_[i] -=
          src.sum_weights_per_treatment_and_outcome_[i];
    }
  }

  double MeanOutcome() const {
    DCHECK_EQ(sum_weights_per_treatment_.size(), 2);
    DCHECK_EQ(num_examples_per_treatment_.size(), 2);
    DCHECK_EQ(sum_weights_per_treatment_and_outcome_.size(), 2);
    const auto sum_weights =
        sum_weights_per_treatment_[0] + sum_weights_per_treatment_[1];
    if (sum_weights == 0) {
      return 0;
    }
    const auto sum_outcomes = sum_weights_per_treatment_and_outcome_[0] +
                              sum_weights_per_treatment_and_outcome_[1];
    return sum_outcomes / sum_weights;
  }

  double MeanOutcomePerTreatment(const int treatment) const {
    DCHECK(treatment == 0 || treatment == 1);
    if (sum_weights_per_treatment_[treatment] == 0) {
      return 0;
    }
    return sum_weights_per_treatment_and_outcome_[treatment] /
           sum_weights_per_treatment_[treatment];
  }

  // Tests if the uplift can be computed.
  bool HasUplift() const {
    return sum_weights_per_treatment_[0] != 0 &&
           sum_weights_per_treatment_[1] != 0;
  }

  // Uplift value for bucket ordering.
  double UpliftBucket(const EmptyBucketOrdering empty_bucket_ordering,
                      const UpliftLabelDistribution& parent) const {
    // Only support binary treatment and single dimension outcome.
    DCHECK_EQ(sum_weights_per_treatment_.size(), 2);
    DCHECK_EQ(num_examples_per_treatment_.size(), 2);
    DCHECK_EQ(sum_weights_per_treatment_and_outcome_.size(), 2);

    // Replacement value for missing output treatment.
    const auto missing_outcome_imputation = [&](const int treatment) -> double {
      switch (empty_bucket_ordering) {
        case proto::DecisionTreeTrainingConfig::Uplift::
            PARENT_TREATMENT_OUTCOME:
          return parent.MeanOutcomePerTreatment(treatment);
        case proto::DecisionTreeTrainingConfig::Uplift::PARENT_OUTCOME:
          return parent.MeanOutcome();
        default:
          DCHECK(false);
          break;
      }
    };

    const double response_control = (sum_weights_per_treatment_[0] != 0)
                                        ? MeanOutcomePerTreatment(0)
                                        : missing_outcome_imputation(0);
    const double response_treatment = (sum_weights_per_treatment_[1] != 0)
                                          ? MeanOutcomePerTreatment(1)
                                          : missing_outcome_imputation(1);
    return response_treatment - response_control;
  }

  // Uplift value for leafs. All the treatments should have at least one value.
  double UpliftLeaf() const {
    if (!HasUplift()) {
      // The minimum number of examples per treatment should make this case
      // not possible.
      DCHECK(false) << "Uplift leaf not available: " << *this;
      return 0;
    }

    // Only support binary treatment and single dimension outcome.
    DCHECK_EQ(sum_weights_per_treatment_.size(), 2);
    DCHECK_EQ(num_examples_per_treatment_.size(), 2);
    DCHECK_EQ(sum_weights_per_treatment_and_outcome_.size(), 2);

    const double response_control = MeanOutcomePerTreatment(0);
    const double response_treatment = MeanOutcomePerTreatment(1);
    return response_treatment - response_control;
  }

  // Returns the lower bound of the 9.7% confidence interval of the uplift.
  // Model the outcome as a normal distribution.
  double ConservativeUpliftScore() const {
    if (!HasUplift()) {
      return 0;
    }

    // Only support binary treatment and single dimension outcome.
    DCHECK_EQ(sum_weights_per_treatment_.size(), 2);
    DCHECK_EQ(num_examples_per_treatment_.size(), 2);
    DCHECK_EQ(sum_weights_per_treatment_and_outcome_.size(), 2);

    const double mean_c = MeanOutcomePerTreatment(0);
    const double mean_t = MeanOutcomePerTreatment(1);
    const auto var_c = mean_c * (1 - mean_c) / sum_weights_per_treatment_[0];
    const auto var_t = mean_t * (1 - mean_t) / sum_weights_per_treatment_[1];
    const auto mean_diff = mean_t - mean_c;
    const auto var_diff = var_c + var_t;
    const auto sd_diff = sqrt(var_diff);
    // z-value for a ~9.7% confidence bound. This value was selected to give
    // reasonable results on the train/test SimPTE dataset.
    const double z = 1.3;

    const double lb = mean_diff - z * sd_diff;
    const double ub = mean_diff + z * sd_diff;

    // Return the most conservative uplift value (i.e. the value closest to
    // zero; i.e. with the smaller absolute value) in [lb, ub]. For example, if
    // l=-0.1 and ub=0.3, return return 0.
    if (lb > 0) {
      return lb;
    }
    if (ub < 0) {
      return ub;
    }
    return 0;
  }

  // Uplift for score splits.
  double UpliftSplitScore(const SplitScoreType score) const {
    if (!HasUplift()) {
      return 0;
    }

    switch (score) {
      case proto::DecisionTreeTrainingConfig::Uplift::EUCLIDEAN_DISTANCE: {
        const double response_control = MeanOutcomePerTreatment(0);
        const double response_treatment = MeanOutcomePerTreatment(1);

        return (response_control - response_treatment) *
               (response_control - response_treatment);
      }
      case proto::DecisionTreeTrainingConfig::Uplift::KULLBACK_LEIBLER: {
        const double response_control = MeanOutcomePerTreatment(0);
        const double response_treatment = MeanOutcomePerTreatment(1);
        if (response_treatment == 0) {
          return 0;
        }

        if (response_control == 0) {
          // The returned divergence should be infinite (or very high). However,
          // this would essentially discard all the possible splits. Returning
          // 0, would enable the search for splits, but would not be great to
          // break ties. Instead, we return a correlated with the split quality
          // that would be smaller than any real divergence values.
          return response_treatment / 1000;
        }
        return response_treatment *
               std::log(response_treatment / response_control);
      }
      case proto::DecisionTreeTrainingConfig::Uplift::CHI_SQUARED: {
        const double response_control = MeanOutcomePerTreatment(0);
        const double response_treatment = MeanOutcomePerTreatment(1);
        if (response_control == 0) {
          // The returned divergence should be infinite (or very high). However,
          // this would essentially discard all the possible splits. Returning
          // 0, would enable the search for splits, but would not be great to
          // break ties. Instead, we return a correlated with the split quality
          // that would be smaller than any real divergence values.
          return response_treatment / 1000;
        }
        return (response_treatment - response_control) *
               (response_treatment - response_control) / response_control;
      }
      case proto::DecisionTreeTrainingConfig::Uplift::
          CONSERVATIVE_EUCLIDEAN_DISTANCE: {
        const auto u = ConservativeUpliftScore();
        return u * u;
      }
    }
  }

  int MinNumExamplesPerTreatment() const {
    DCHECK_EQ(num_examples_per_treatment_.size(), 2);
    return std::min(sum_weights_per_treatment_[0],
                    sum_weights_per_treatment_[1]);
  }

  double num_examples() const { return sum_weights_; }

  void ImportSetFromLeafProto(const proto::NodeUpliftOutput& leaf) {
    sum_weights_ = leaf.sum_weights();
    sum_weights_per_treatment_ = {leaf.sum_weights_per_treatment().begin(),
                                  leaf.sum_weights_per_treatment().end()};

    num_examples_per_treatment_ = {leaf.num_examples_per_treatment().begin(),
                                   leaf.num_examples_per_treatment().end()};

    sum_weights_per_treatment_and_outcome_ = {
        leaf.sum_weights_per_treatment_and_outcome().begin(),
        leaf.sum_weights_per_treatment_and_outcome().end()};
  }

  void ExportToLeafProto(proto::NodeUpliftOutput* leaf) const {
    leaf->set_sum_weights(sum_weights_);
    *leaf->mutable_sum_weights_per_treatment() = {
        sum_weights_per_treatment_.begin(), sum_weights_per_treatment_.end()};
    *leaf->mutable_num_examples_per_treatment() = {
        num_examples_per_treatment_.begin(), num_examples_per_treatment_.end()};
    *leaf->mutable_sum_weights_per_treatment_and_outcome() = {
        sum_weights_per_treatment_and_outcome_.begin(),
        sum_weights_per_treatment_and_outcome_.end()};

    auto& treatment_effect = *leaf->mutable_treatment_effect();
    treatment_effect.Clear();
    treatment_effect.Add(UpliftLeaf());
  }

 private:
  // The fields have the same definition as the fields in
  // "proto::NodeUpliftOutput". See this proto documentation for the explanation
  // about the +1/-1 in this class.
  double sum_weights_;
  absl::InlinedVector<double, 2> sum_weights_per_treatment_;
  absl::InlinedVector<double, 2> sum_weights_per_treatment_and_outcome_;
  absl::InlinedVector<SignedExampleIdx, 2> num_examples_per_treatment_;

  friend std::ostream& operator<<(std::ostream& os,
                                  const UpliftLabelDistribution& data);
};

inline std::ostream& operator<<(std::ostream& os,
                                const UpliftLabelDistribution& data) {
  os << "{sum_weights:" << data.sum_weights_ << " sum_weights_per_treatment:[";
  for (const auto x : data.sum_weights_per_treatment_) {
    os << " " << x;
  }
  os << "] sum_weights_per_treatment_and_outcome:[";
  for (const auto x : data.sum_weights_per_treatment_and_outcome_) {
    os << " " << x;
  }
  os << "] num_examples_per_treatment:[";
  for (const auto x : data.num_examples_per_treatment_) {
    os << " " << x;
  }
  os << "]}";
  return os;
}

struct LabelUpliftCategoricalScoreAccumulator {
  static constexpr bool kNormalizeByWeight = true;

  double Score() const { return label.UpliftSplitScore(score); }

  double WeightedNumExamples() const { return label.num_examples(); }

  UpliftLabelDistribution label;
  UpliftLabelDistribution::SplitScoreType score;
};

typedef LabelUpliftCategoricalScoreAccumulator
    LabelUpliftNumericalScoreAccumulator;

template <bool categorical_label>
struct LabelUpliftGenericOneValueBucket {
  typedef typename std::conditional<
      categorical_label, LabelUpliftCategoricalScoreAccumulator,
      LabelUpliftNumericalScoreAccumulator>::type Accumulator;

  typedef LabelUpliftGenericOneValueBucket<categorical_label> Bucket;

  typedef typename std::conditional<categorical_label, int32_t, float>::type
      OutcomeValue;

  int treatment;
  OutcomeValue outcome;
  float weight;

  static constexpr int count = 1;  // NOLINT

  void AddToScoreAcc(Accumulator* acc) const {
    if constexpr (categorical_label) {
      acc->label.AddCategoricalOutcome(outcome, treatment, weight);
    } else {
      acc->label.AddNumericalOutcome(outcome, treatment, weight);
    }
  }

  void SubToScoreAcc(Accumulator* acc) const {
    if constexpr (categorical_label) {
      acc->label.SubCategoricalOutcome(outcome, treatment, weight);
    } else {
      acc->label.SubNumericalOutcome(outcome, treatment, weight);
    }
  }

  class Initializer {
   public:
    Initializer(const UpliftLabelDistribution& label_distribution,
                const int min_examples_per_treatment,
                const UpliftLabelDistribution::SplitScoreType score_type)
        : label_distribution_(label_distribution),
          initial_uplift_(label_distribution.UpliftSplitScore(score_type)),
          min_examples_per_treatment_(min_examples_per_treatment),
          score_type_(score_type) {}

    void InitEmpty(Accumulator* acc) const {
      acc->label.InitializeAndClearLike(label_distribution_);
      acc->score = score_type_;
    }

    void InitFull(Accumulator* acc) const {
      acc->label = label_distribution_;
      acc->score = score_type_;
    }

    double NormalizeScore(const double score) const {
      return score - initial_uplift_;
    }

    bool IsValidSplit(const Accumulator& neg, const Accumulator& pos) const {
      if (min_examples_per_treatment_ == 0) {
        return true;
      }
      return neg.label.MinNumExamplesPerTreatment() >=
                 min_examples_per_treatment_ &&
             pos.label.MinNumExamplesPerTreatment() >=
                 min_examples_per_treatment_;
    }

    double MinimumScore() const { return 0; }

   private:
    const UpliftLabelDistribution& label_distribution_;
    const double initial_uplift_;
    const int min_examples_per_treatment_;
    const UpliftLabelDistribution::SplitScoreType score_type_;
  };

  class Filler {
   public:
    Filler(const std::vector<OutcomeValue>& outcomes,
           const std::vector<int32_t>& treatments,
           const std::vector<float>& weights)
        : outcomes_(outcomes), treatments_(treatments), weights_(weights) {}

    void InitializeAndZero(Bucket* acc) const {
      // Nothing to do.
    }

    void Finalize(Bucket* acc) const {
      // Nothing to do.
    }

    void ConsumeExample(const UnsignedExampleIdx example_idx,
                        Bucket* acc) const {
      acc->outcome = outcomes_[example_idx];
      acc->treatment = treatments_[example_idx];
      acc->weight = weights_[example_idx];
    }

   private:
    const std::vector<OutcomeValue>& outcomes_;
    const std::vector<int32_t>& treatments_;
    const std::vector<float>& weights_;
  };

  friend std::ostream& operator<<(std::ostream& os, const Bucket& data);
};

inline std::ostream& operator<<(
    std::ostream& os, const LabelUpliftGenericOneValueBucket<true>& data) {
  os << "value:{treatment:" << data.treatment << " outcome:" << data.outcome
     << " weight:" << data.weight << "} count:" << data.count;
  return os;
}

inline std::ostream& operator<<(
    std::ostream& os, const LabelUpliftGenericOneValueBucket<false>& data) {
  os << "value:{treatment:" << data.treatment << " outcome:" << data.outcome
     << " weight:" << data.weight << "} count:" << data.count;
  return os;
}

typedef LabelUpliftGenericOneValueBucket<true>
    LabelUpliftCategoricalOneValueBucket;

typedef LabelUpliftGenericOneValueBucket<false>
    LabelUpliftNumericalOneValueBucket;

template <bool categorical_label>
struct LabelUpliftGenericBucket {
  typedef typename std::conditional<
      categorical_label, LabelUpliftCategoricalScoreAccumulator,
      LabelUpliftNumericalScoreAccumulator>::type Accumulator;

  typedef LabelUpliftGenericBucket<categorical_label> Bucket;

  typedef typename std::conditional<categorical_label, int32_t, float>::type
      OutcomeValue;

  UpliftLabelDistribution distribution;
  int64_t count;
  float signed_uplift;

  void AddToScoreAcc(Accumulator* acc) const { acc->label.Add(distribution); }

  void SubToScoreAcc(Accumulator* acc) const { acc->label.Sub(distribution); }

  bool operator<(const Bucket& other) const {
    return signed_uplift < other.signed_uplift;
  }

  class Initializer {
   public:
    Initializer(const UpliftLabelDistribution& label_distribution,
                const int min_examples_per_treatment,
                UpliftLabelDistribution::SplitScoreType score)
        : label_distribution_(label_distribution),
          initial_uplift_(label_distribution.UpliftSplitScore(score)),
          min_examples_per_treatment_(min_examples_per_treatment),
          score_(score) {}

    void InitEmpty(Accumulator* acc) const {
      acc->label.InitializeAndClearLike(label_distribution_);
      acc->score = score_;
    }

    void InitFull(Accumulator* acc) const {
      acc->label = label_distribution_;
      acc->score = score_;
    }

    double NormalizeScore(const double score) const {
      return score - initial_uplift_;
    }

    bool IsValidSplit(const Accumulator& neg, const Accumulator& pos) const {
      if (min_examples_per_treatment_ == 0) {
        return true;
      }

      return neg.label.MinNumExamplesPerTreatment() >=
                 min_examples_per_treatment_ &&
             pos.label.MinNumExamplesPerTreatment() >=
                 min_examples_per_treatment_;
    }

    double MinimumScore() const { return 0; }

   private:
    const UpliftLabelDistribution& label_distribution_;
    const double initial_uplift_;
    const int min_examples_per_treatment_;
    const UpliftLabelDistribution::SplitScoreType score_;
  };

  class Filler {
   public:
    Filler(const UpliftLabelDistribution& label_distribution,
           const std::vector<OutcomeValue>& outcomes,
           const std::vector<int32_t>& treatments,
           const std::vector<float>& weights,
           const proto::DecisionTreeTrainingConfig::Uplift::EmptyBucketOrdering
               empty_bucket_ordering)
        : outcomes_(outcomes),
          treatments_(treatments),
          weights_(weights),
          label_distribution_(label_distribution),
          empty_bucket_ordering_(empty_bucket_ordering) {}

    void InitializeAndZero(Bucket* acc) const {
      acc->count = 0;
      acc->distribution.InitializeAndClearLike(label_distribution_);
    }

    void Finalize(Bucket* acc) const {
      acc->signed_uplift = acc->distribution.UpliftBucket(
          empty_bucket_ordering_, label_distribution_);
    }

    void ConsumeExample(const UnsignedExampleIdx example_idx,
                        Bucket* acc) const {
      if constexpr (categorical_label) {
        acc->distribution.AddCategoricalOutcome(outcomes_[example_idx],
                                                treatments_[example_idx],
                                                weights_[example_idx]);

      } else {
        acc->distribution.AddNumericalOutcome(outcomes_[example_idx],
                                              treatments_[example_idx],
                                              weights_[example_idx]);
      }
      acc->count++;
    }

   private:
    const std::vector<OutcomeValue>& outcomes_;
    const std::vector<int32_t>& treatments_;
    const std::vector<float>& weights_;
    const UpliftLabelDistribution& label_distribution_;
    const proto::DecisionTreeTrainingConfig::Uplift::EmptyBucketOrdering
        empty_bucket_ordering_;
  };

  friend std::ostream& operator<<(
      std::ostream& os,
      const LabelUpliftGenericBucket<categorical_label>& data);
};

inline std::ostream& operator<<(std::ostream& os,
                                const LabelUpliftGenericBucket<true>& data) {
  os << "value:{distribution: " << data.distribution
     << " signed_uplift:" << data.signed_uplift << "} count:" << data.count;
  return os;
}

inline std::ostream& operator<<(std::ostream& os,
                                const LabelUpliftGenericBucket<false>& data) {
  os << "value:{distribution: " << data.distribution
     << " signed_uplift:" << data.signed_uplift << "} count:" << data.count;
  return os;
}

typedef LabelUpliftGenericBucket<true> LabelUpliftCategoricalBucket;

typedef LabelUpliftGenericBucket<false> LabelUpliftNumericalBucket;

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPLITTER_ACCUMULATOR_UPLIFT_H_
