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

// Give access to label values.
//
// A "label filler" is a class that adds/subtracts label values from label
// statistics accumulator (called "accumulator" for short) and label statistics
// buckets (called "label bucket" for short).
//
// An accumulator and a label bucket are similar objects from the point of view
// of the label filler. In practice, a dataset is always split into two
// accumulators while there can be one bucket per examples (therefore, they can
// have different internal structures / numerical precision). Additionally,
// examples can be removed from accumulator, but not from buckets.
//
// A label filler should implement the following methods:
//   InitializeAndZeroAccumulator(accumulator): Initialize an (label statistics)
//     accumulator to the empty state.
//   InitializeAndZeroBucket: Initialize a label bucket to the empty state.
//   Add(accumulator)/Sub(accumulator): Add/subtract an example to an
//     accumulator.
//   Add(label_bucket): Add an example to an accumulator.
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_LABEL_ACCESSOR_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_LABEL_ACCESSOR_H_

#include "yggdrasil_decision_forests/learner/decision_tree/splitter_accumulator.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {

// How to represent an example index.
typedef uint64_t ExampleIndex;

// Categorical label filler. Alternative to LabelCategoricalBucket::Filler.
class ClassificationLabelFiller {
 public:
  // How to represent a label value.
  typedef int16_t Label;
  typedef decision_tree::LabelCategoricalBucket LabelBucket;
  typedef decision_tree::LabelCategoricalScoreAccumulator Accumulator;
  typedef LabelBucket::Initializer AccumulatorInitializer;

  ClassificationLabelFiller(const std::vector<Label>& labels,
                            const std::vector<float>& weights,
                            const Label num_classes)
      : labels_(labels), weights_(weights), num_classes_(num_classes) {}

  const std::vector<Label>& labels() const { return labels_; }
  const std::vector<float>& weights() const { return weights_; }
  const size_t num_examples() const { return labels_.size(); }

  void InitializeAndZeroAccumulator(Accumulator* accumulator) const {
    accumulator->label.SetNumClasses(num_classes_);
    accumulator->label.Clear();
  }

  void InitializeAndZeroBucket(LabelBucket* bucket) const {
    bucket->value.Clear();
    bucket->value.SetNumClasses(num_classes_);
    bucket->count = 0;
  }

  void Add(const ExampleIndex example_idx, Accumulator* accumulator) const {
    if (weights_.empty()) {
      accumulator->label.Add(labels_[example_idx]);
    } else {
      accumulator->label.Add(labels_[example_idx], weights_[example_idx]);
    }
  }

  void Sub(const ExampleIndex example_idx, Accumulator* accumulator) const {
    if (weights_.empty()) {
      accumulator->label.Sub(labels_[example_idx]);
    } else {
      accumulator->label.Sub(labels_[example_idx], weights_[example_idx]);
    }
  }

  void Add(const ExampleIndex example_idx, LabelBucket* bucket) const {
    if (weights_.empty()) {
      bucket->value.Add(labels_[example_idx]);
    } else {
      bucket->value.Add(labels_[example_idx], weights_[example_idx]);
    }
    bucket->count++;
  }

 private:
  const std::vector<Label>& labels_;
  const std::vector<float>& weights_;
  const int num_classes_;
};

// Regression label filler. Alternative to LabelNumericalBucket::Filler.
class RegressionLabelFiller {
 public:
  // How to represent a label value.
  typedef float Label;
  typedef decision_tree::LabelNumericalBucket LabelBucket;
  typedef decision_tree::LabelNumericalScoreAccumulator Accumulator;
  typedef LabelBucket::Initializer AccumulatorInitializer;

  RegressionLabelFiller(const std::vector<Label>& labels,
                        const std::vector<float>& weights)
      : labels_(labels), weights_(weights) {}

  const std::vector<Label>& labels() const { return labels_; }
  const std::vector<float>& weights() const { return weights_; }
  const size_t num_examples() const { return labels_.size(); }

  void InitializeAndZeroAccumulator(Accumulator* accumulator) const {
    accumulator->label.Clear();
  }

  void InitializeAndZeroBucket(LabelBucket* bucket) const {
    bucket->value.Clear();
    bucket->count = 0;
  }

  void Add(const ExampleIndex example_idx, Accumulator* accumulator) const {
    if (weights_.empty()) {
      accumulator->label.Add(labels_[example_idx]);
    } else {
      accumulator->label.Add(labels_[example_idx], weights_[example_idx]);
    }
  }

  void Sub(const ExampleIndex example_idx, Accumulator* accumulator) const {
    if (weights_.empty()) {
      accumulator->label.Sub(labels_[example_idx]);
    } else {
      accumulator->label.Sub(labels_[example_idx], weights_[example_idx]);
    }
  }

  void Add(const ExampleIndex example_idx, LabelBucket* bucket) const {
    if (weights_.empty()) {
      bucket->value.Add(labels_[example_idx]);
    } else {
      bucket->value.Add(labels_[example_idx], weights_[example_idx]);
    }
    bucket->count++;
  }

 private:
  const std::vector<Label>& labels_;
  const std::vector<float>& weights_;
};

// Regression label filler. Alternative to
// LabelNumericalWithHessianBucket::Filler.
class RegressionWithHessianLabelFiller {
 public:
  // How to represent a label value.
  typedef decision_tree::LabelNumericalWithHessianBucket LabelBucket;
  typedef decision_tree::LabelNumericalWithHessianScoreAccumulator Accumulator;
  typedef LabelBucket::Initializer AccumulatorInitializer;

  RegressionWithHessianLabelFiller(const std::vector<float>& labels,
                                   const std::vector<float>& hessians,
                                   const std::vector<float>& weights)
      : labels_(labels), hessians_(hessians), weights_(weights) {}

  const std::vector<float>& labels() const { return labels_; }
  const std::vector<float>& hessians() const { return hessians_; }
  const std::vector<float>& weights() const { return weights_; }
  const size_t num_examples() const { return labels_.size(); }

  void InitializeAndZeroAccumulator(Accumulator* accumulator) const {
    accumulator->label.Clear();
    accumulator->sum_hessian = 0;
  }

  void InitializeAndZeroBucket(LabelBucket* bucket) const {
    bucket->value.Clear();
    bucket->sum_hessian = 0;
    bucket->count = 0;
  }

  void Add(const ExampleIndex example_idx, Accumulator* accumulator) const {
    if (weights_.empty()) {
      accumulator->label.Add(labels_[example_idx]);
      accumulator->sum_hessian += hessians_[example_idx];
    } else {
      accumulator->label.Add(labels_[example_idx], weights_[example_idx]);
      accumulator->sum_hessian +=
          hessians_[example_idx] * weights_[example_idx];
    }
  }

  void Sub(const ExampleIndex example_idx, Accumulator* accumulator) const {
    if (weights_.empty()) {
      accumulator->label.Sub(labels_[example_idx]);
      accumulator->sum_hessian -= hessians_[example_idx];
    } else {
      accumulator->label.Sub(labels_[example_idx], weights_[example_idx]);
      accumulator->sum_hessian -=
          hessians_[example_idx] * weights_[example_idx];
    }
  }

  void Add(const ExampleIndex example_idx, LabelBucket* bucket) const {
    if (weights_.empty()) {
      bucket->value.Add(labels_[example_idx]);
      bucket->sum_hessian += hessians_[example_idx];
    } else {
      bucket->value.Add(labels_[example_idx], weights_[example_idx]);
      bucket->sum_hessian += hessians_[example_idx] * weights_[example_idx];
    }
    bucket->count++;
  }

 private:
  const std::vector<float>& labels_;
  const std::vector<float>& hessians_;
  const std::vector<float>& weights_;
};

// Gives access to label values.
class AbstractLabelAccessor {
 public:
  virtual ~AbstractLabelAccessor() = default;

  // Classification.
  virtual utils::StatusOr<ClassificationLabelFiller>
  CreateClassificationLabelFiller() const {
    return absl::InternalError(
        "CreateClassificationLabelFiller not implemented");
  }
  virtual utils::StatusOr<ClassificationLabelFiller::AccumulatorInitializer>
  CreateClassificationAccumulatorInitializer(
      const decision_tree::proto::LabelStatistics& statistics) const {
    return absl::InternalError(
        "CreateClassificationAccumulatorInitializer not implemented");
  }

  // Regression.
  virtual utils::StatusOr<RegressionLabelFiller> CreateRegressionLabelFiller()
      const {
    return absl::InternalError("CreateRegressionLabelFiller not implemented");
  }
  virtual utils::StatusOr<RegressionLabelFiller::AccumulatorInitializer>
  CreateRegressionAccumulatorInitializer(
      const decision_tree::proto::LabelStatistics& statistics) const {
    return absl::InternalError(
        "CreateRegressionAccumulatorInitializer not implemented");
  }

  // Regression with hessian information
  virtual utils::StatusOr<RegressionWithHessianLabelFiller>
  CreateRegressionWithHessianLabelFiller() const {
    return absl::InternalError(
        "CreateRegressionWithHessianLabelFiller not implemented");
  }
  virtual utils::StatusOr<
      RegressionWithHessianLabelFiller::AccumulatorInitializer>
  CreateRegressionWithHessianAccumulatorInitializer(
      const decision_tree::proto::LabelStatistics& statistics) const {
    return absl::InternalError(
        "CreateRegressionWithHessianAccumulatorInitializer not implemented");
  }
};

class ClassificationLabelAccessor : public AbstractLabelAccessor {
 public:
  ClassificationLabelAccessor(
      const std::vector<ClassificationLabelFiller::Label>& labels,
      const std::vector<float>& weights, const int num_classes)
      : labels_(labels), weights_(weights), num_classes_(num_classes) {}

  utils::StatusOr<ClassificationLabelFiller> CreateClassificationLabelFiller()
      const override {
    return ClassificationLabelFiller(labels_, weights_, num_classes_);
  }

  utils::StatusOr<ClassificationLabelFiller::AccumulatorInitializer>
  CreateClassificationAccumulatorInitializer(
      const decision_tree::proto::LabelStatistics& statistics) const override {
    return ClassificationLabelFiller::AccumulatorInitializer(statistics);
  }

 private:
  const std::vector<ClassificationLabelFiller::Label>& labels_;
  const std::vector<float>& weights_;
  ClassificationLabelFiller::Label num_classes_;
};

class RegressionLabelAccessor : public AbstractLabelAccessor {
 public:
  RegressionLabelAccessor(
      const std::vector<RegressionLabelFiller::Label>& labels,
      const std::vector<float>& weights)
      : labels_(labels), weights_(weights) {}

  utils::StatusOr<RegressionLabelFiller> CreateRegressionLabelFiller()
      const override {
    return RegressionLabelFiller(labels_, weights_);
  }

  utils::StatusOr<RegressionLabelFiller::AccumulatorInitializer>
  CreateRegressionAccumulatorInitializer(
      const decision_tree::proto::LabelStatistics& statistics) const override {
    return RegressionLabelFiller::AccumulatorInitializer(statistics);
  }

 private:
  const std::vector<RegressionLabelFiller::Label>& labels_;
  const std::vector<float>& weights_;
};

class RegressionWithHessianLabelAccessor : public AbstractLabelAccessor {
 public:
  RegressionWithHessianLabelAccessor(const std::vector<float>& gradients,
                                     const std::vector<float>& hessians,
                                     const std::vector<float>& weights)
      : labels_(gradients), hessians_(hessians), weights_(weights) {}

  utils::StatusOr<RegressionWithHessianLabelFiller>
  CreateRegressionWithHessianLabelFiller() const override {
    return RegressionWithHessianLabelFiller(labels_, hessians_, weights_);
  }

  utils::StatusOr<RegressionWithHessianLabelFiller::AccumulatorInitializer>
  CreateRegressionWithHessianAccumulatorInitializer(
      const decision_tree::proto::LabelStatistics& statistics) const override {
    return RegressionWithHessianLabelFiller::AccumulatorInitializer(statistics);
  }

 private:
  const std::vector<float>& labels_;
  const std::vector<float>& hessians_;
  const std::vector<float>& weights_;
};

}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif
