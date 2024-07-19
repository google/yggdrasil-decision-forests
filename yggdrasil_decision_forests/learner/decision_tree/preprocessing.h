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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_PREPROCESSING_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_PREPROCESSING_H_

#include <stdint.h>

#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"

namespace yggdrasil_decision_forests::model::decision_tree {

// Item used to pass the sorted numerical attribute values to the
// "ScanSplitsPresortedSparseDuplicateExampleTemplate" function below.
//
// Index of the example in the training dataset.
// The highest bit (MaskDeltaBit) is 1 iif. the feature value of this item is
// strictly greater than the preceding one. The other bits (MaskExampleIdx) are
// used to encode the example index.
struct SparseItemMeta {
  typedef UnsignedExampleIdx ExampleIdx;

  static constexpr auto kMaxNumExamples =
      std::numeric_limits<SignedExampleIdx>::max();

  static constexpr ExampleIdx kMaskDeltaBit =
      (SparseItemMeta::ExampleIdx{1}
       << (sizeof(SparseItemMeta::ExampleIdx) * 8 - 1));

  static constexpr ExampleIdx kMaskExampleIdx = kMaskDeltaBit - 1;
};

typedef SparseItemMeta::ExampleIdx SparseItem;

// Returns true if the strategy require for the input features to be pre-sorted.
bool StrategyRequireFeaturePresorting(
    proto::DecisionTreeTrainingConfig::Internal::SortingStrategy strategy);

// Pre-computation on the training dataset used for the training of individual
// trees. The pre-processing is computed before any tree is trained.
class Preprocessing {
 public:
  struct PresortedNumericalFeature {
    // Example index sorted in increasing order of feature values.
    // Missing values are treated as replaced by the GLOBAL_IMPUTATION strategy.
    // The high bit of each example index is set iif. the feature value is
    // different (greater) than the previous one.
    std::vector<SparseItem> items;
  };

  std::vector<PresortedNumericalFeature>*
  mutable_presorted_numerical_features() {
    return &presorted_numerical_features_;
  }

  const std::vector<PresortedNumericalFeature>& presorted_numerical_features()
      const {
    return presorted_numerical_features_;
  }

  uint64_t num_examples() const { return num_examples_; }

  void set_num_examples(const uint64_t value) { num_examples_ = value; }

 private:
  // List of presorted numerical features, indexed by feature index.
  // If feature "i" is not numerical or not presorted,
  // "presorted_numerical_features_[i]" will be an empty index.
  std::vector<PresortedNumericalFeature> presorted_numerical_features_;

  // Total number of examples.
  uint64_t num_examples_ = -1;
};

// Preprocess the dataset before any tree training.
absl::StatusOr<Preprocessing> PreprocessTrainingDataset(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config, int num_threads);

// Component of "PreprocessTrainingDataset". Computes pre-sorted numerical
// features.
absl::Status PresortNumericalFeatures(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config, int num_threads,
    Preprocessing* preprocessing);

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_PREPROCESSING_H_
