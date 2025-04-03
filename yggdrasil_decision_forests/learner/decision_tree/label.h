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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_CONFIG_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_CONFIG_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/uplift.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/distribution.h"

namespace yggdrasil_decision_forests::model::decision_tree {

// Label statistics.
struct LabelStats {
  virtual ~LabelStats() = default;
};

// Label statistics for Classification.
struct ClassificationLabelStats : LabelStats {
  explicit ClassificationLabelStats(const std::vector<int32_t>& label_data)
      : label_data(label_data) {}

  const std::vector<int32_t>& label_data;
  int32_t num_label_classes;
  utils::IntegerDistributionDouble label_distribution;
};

// Label statistics for Regression.
struct RegressionLabelStats : LabelStats {
  explicit RegressionLabelStats(const std::vector<float>& label_data)
      : label_data(label_data) {}

  const std::vector<float>& label_data;
  utils::NormalDistributionDouble label_distribution;
};

// Label statistics for Regression with hessian.
struct RegressionHessianLabelStats : LabelStats {
  RegressionHessianLabelStats(const std::vector<float>& gradient_data,
                              const std::vector<float>& hessian_data)
      : gradient_data(gradient_data), hessian_data(hessian_data) {}

  const std::vector<float>& gradient_data;
  const std::vector<float>& hessian_data;
  double sum_gradient;
  double sum_hessian;
  double sum_weights;
};

// Label statistics for uplift with categorical treatment and categorical
// outcome.
struct CategoricalUpliftLabelStats : LabelStats {
  explicit CategoricalUpliftLabelStats(
      const std::vector<int32_t>& outcome_values,
      const int num_unique_in_outcomes_column,
      const std::vector<int32_t>& treatment_values,
      const int num_unique_values_in_treatments_column)
      : outcome_values(outcome_values),
        treatment_values(treatment_values),
        num_unique_values_in_treatments_column(
            num_unique_values_in_treatments_column),
        num_unique_in_outcomes_column(num_unique_in_outcomes_column) {}

  const std::vector<int32_t>& outcome_values;
  const std::vector<int32_t>& treatment_values;
  int32_t num_unique_values_in_treatments_column;
  int32_t num_unique_in_outcomes_column;

  UpliftLabelDistribution label_distribution;
};

// Label statistics for uplift with categorical treatment and numerical outcome.
struct NumericalUpliftLabelStats : LabelStats {
  explicit NumericalUpliftLabelStats(
      const std::vector<float>& outcome_values,
      const std::vector<int32_t>& treatment_values,
      const int num_unique_values_in_treatments_column)
      : outcome_values(outcome_values),
        treatment_values(treatment_values),
        num_unique_values_in_treatments_column(
            num_unique_values_in_treatments_column) {}

  const std::vector<float>& outcome_values;
  const std::vector<int32_t>& treatment_values;
  int32_t num_unique_values_in_treatments_column;

  UpliftLabelDistribution label_distribution;
};

// Signature of a function that sets the value (i.e. the prediction) of a leaf
// from the gradient data.
typedef std::function<absl::Status(
    const dataset::VerticalDataset&, absl::Span<const UnsignedExampleIdx>,
    const std::vector<float>&, const model::proto::TrainingConfig&,
    const model::proto::TrainingConfigLinking&, NodeWithChildren* node)>
    CreateSetLeafValueFunctor;

// Similar to CreateSetLeafValueFunctor, but use pre-computed statistics instead
// of scanning the values.
typedef std::function<absl::Status(
    const decision_tree::proto::LabelStatistics& label_stats,
    decision_tree::proto::Node* leaf)>
    SetLeafValueFromLabelStatsFunctor;

// The default policy to set the value of a leaf.
// - Distribution of the labels for classification.
// - Mean of the labels for regression.
absl::Status SetLabelDistribution(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    NodeWithChildren* node);

// Copies the content on uplift categorical label distribution to the leafs.
void UpliftLabelDistToLeaf(const UpliftLabelDistribution& dist,
                           decision_tree::proto::NodeUpliftOutput* leaf);

// Copies the content on uplift categorical leaf output to a label distribution.
void UpliftLeafToLabelDist(const decision_tree::proto::NodeUpliftOutput& leaf,
                           UpliftLabelDistribution* dist);

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_CONFIG_H_
