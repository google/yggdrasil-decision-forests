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

// Implementation and extension of the "Sparse Projection Oblique Random
// Forests" 2020 JMLR paper by Tomita et al
// (https://www.jmlr.org/papers/volume21/18-664/18-664.pdf) and the
// "Classification Based on Multivariate Contrast Patterns" 2019 paper by
// Canete-Sifuentes et al.
//
// Sparse Projection Oblique Random Forests algorithm:
//
// During training, at each node, the algorithm samples multiple random sparse
// linear projections of the numerical features, and evaluate then as classical
// numerical features (i.e. looking for a split projection >= threshold).
//
// Experiments in Tomita's paper indicates that this split algorithm used with
// Random Forest can leads to improvements over classical (RF) and other
// random-projection-oblique (RR-RF, CCF) random forest algorithms. These
// results have been confirmed experimentally using this implementation.
//
// Multi-class Hellinger Linear Discriminant algorithm:
//
// The MHLD algorithm works by greedily building projections by adding features
// one after the other. Given a set of features, the projection coefficient is
// determined with Linear Discriminant Analysis (LDA). Like for the SPO
// algorithm, the regular splitting algorithm is used to select the threshold.
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/label.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

// The following three "FindBestConditionOblique" functions are searching
// for the best sparse oblique split for different objectives / loss functions.
// These methods only differ by the type of the "label_stats" argument.

// Classification.
absl::StatusOr<bool> FindBestConditionOblique(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const ClassificationLabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache);

// Regression with hessian term.
absl::StatusOr<bool> FindBestConditionOblique(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionHessianLabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache);

// Regression.
absl::StatusOr<bool> FindBestConditionOblique(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionLabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache);

// Computes the number of projections to test i.e.
// num_projections = min(max_num_projections,
// ceil(num_features ^ num_projections_exponent)).
int GetNumProjections(const proto::DecisionTreeTrainingConfig& dt_config,
                      int num_numerical_features);

// Extraction of label values. Different implementations for different types of
// labels.
std::vector<int32_t> ExtractLabels(
    const ClassificationLabelStats& labels,
    absl::Span<const UnsignedExampleIdx> selected);

std::vector<float> ExtractLabels(const RegressionLabelStats& labels,
                                 absl::Span<const UnsignedExampleIdx> selected);

struct GradientAndHessian {
  const std::vector<float> gradient_data;
  const std::vector<float> hessian_data;
};

GradientAndHessian ExtractLabels(const RegressionHessianLabelStats& labels,
                                 absl::Span<const UnsignedExampleIdx> selected);

// Extracts values using an index i.e. returns "values[selected]".
template <typename T>
std::vector<T> Extract(const std::vector<T>& values,
                       absl::Span<const UnsignedExampleIdx> selected);

// Runs a splitter to finds a "x >= t" condition on
// (projection_values,selected_labels,selected_weights).
template <typename LabelStats, typename Labels>
absl::StatusOr<SplitSearchResult> EvaluateProjection(
    const proto::DecisionTreeTrainingConfig& dt_config,
    const LabelStats& label_stats,
    absl::Span<const UnsignedExampleIdx> dense_example_idxs,
    const std::vector<float>& selected_weights, const Labels& selected_labels,
    absl::Span<const float> projection_values,
    const InternalTrainConfig& internal_config, int first_attribute_idx,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache, utils::RandomEngine* random);

namespace internal {

// A projection is defined as \sum features[projection[i].index] *
// projection[i].weight;
struct AttributeAndWeight {
  int attribute_idx;
  float weight;
};
typedef std::vector<AttributeAndWeight> Projection;

// Utility to evaluate projections.
//
// This object references the data of the vertical dataset given as input.
class ProjectionEvaluator {
 public:
  // Initialize.
  //
  // Args:
  //   train_dataset: Dataset containing the data. This object should remain
  //     valid until the "ProjectionEvaluator" is destroyed.
  //   numerical_features: Features to index. Projections can only use indexed
  //     features.
  ProjectionEvaluator(const dataset::VerticalDataset& train_dataset,
                      const google::protobuf::RepeatedField<int32_t>& numerical_features);

  // Evaluates a projection of a set of selected examples.
  //
  // "values", the output variable, contains "selected_examples.size()" values.
  // If one of the input feature of the projection is missing, this input
  // feature is replaced by the mean value of feature as computed on the
  // training dataset. This is the same logic used during inference.
  absl::Status Evaluate(const Projection& projection,
                        absl::Span<const UnsignedExampleIdx> selected_examples,
                        std::vector<float>* values) const;

  absl::Status ExtractAttribute(
      int attribute_idx, absl::Span<const UnsignedExampleIdx> selected_examples,
      std::vector<float>* values) const;

  const std::vector<float>& AttributeValues(int attribute_idx) const {
    return *numerical_attributes_[attribute_idx];
  }

  float NaReplacementValue(int attribute_idx) const {
    return na_replacement_value_[attribute_idx];
  }

 private:
  // Non-owning pointer to numerical attributes.
  // Indexed by attribute idx.
  std::vector<const std::vector<float>*> numerical_attributes_;

  // Replacement for missing values.
  // Indexed by attribute idx.
  std::vector<float> na_replacement_value_;

  // Constructor status.
  absl::Status constructor_status_;
};

// Computes the SW and SB matricies needed to solve the LDA optimization.
class LDACache {
 public:
  absl::Status ComputeClassification(
      const proto::DecisionTreeTrainingConfig& dt_config,
      const ProjectionEvaluator& projection_evaluator,
      const std::vector<int>& selected_features, int num_classes,
      const std::vector<int32_t>& labels, const std::vector<float>& weights,
      bool index_features = true);

  const std::vector<double>& FullSB() const { return sb_; }
  const std::vector<double>& FullSW() const { return sw_; }

  // Gets the SB matrice for a subset of features.
  absl::Status GetSB(const std::vector<int>& selected_features,
                     std::vector<double>* out) const;

  // Gets the SW matrice for a subset of features.
  absl::Status GetSW(const std::vector<int>& selected_features,
                     std::vector<double>* out) const;

  // Builds the feature mapping. "mapping[i]" is the index in "sw_" and "sb_" of
  // feature "selected_features[i]".
  absl::Status BuildMapping(const std::vector<int>& selected_features,
                            std::vector<int>* mapping) const;

  absl::Status Extract(const std::vector<int>& selected_features,
                       const std::vector<double>& in,
                       std::vector<double>* out) const;

 private:
  // The SB and SW square matrices for all the features.
  std::vector<double> sb_;
  std::vector<double> sw_;

  // Size of the sb and sw square matrices.
  int size_;

  // "feature_to_idx_[i]" is the index in sb_ and sw_ of the feature i.
  std::vector<int> feature_to_idx_;
};

// Computes: output += weight * (a - b) * transpose(a - b), where "a" and "b"
// are vectors. "output" should be of the size "a.size() * b.size()".
void SubtractTransposeMultiplyAdd(double weight, absl::Span<double> a,
                                  absl::Span<double> b,
                                  std::vector<double>& output);

// Computes: output += weight * (a - b) * transpose(a - b), where a=
// projection_evaluator[example_idx,:].
void SubtractTransposeMultiplyAdd(
    double weight, std::size_t example_idx,
    const std::vector<int>& selected_features,
    const ProjectionEvaluator& projection_evaluator, absl::Span<double> b,
    std::vector<double>& output);

// Randomly generates a projection from `features`. `features` should only be
// numerical. A projection cannot be empty. If the projection contains only one
// dimension, the weight is guaranteed to be 1. If the projection contains an
// input feature with monotonic constraint, monotonic_direction is set to 1
// (i.e. the projection should be monotonically increasing).
void SampleProjection(const absl::Span<const int>& features,
                      const proto::DecisionTreeTrainingConfig& dt_config,
                      const dataset::proto::DataSpecification& data_spec,
                      const model::proto::TrainingConfigLinking& config_link,
                      float projection_density,
                      internal::Projection* projection,
                      int8_t* monotonic_direction, utils::RandomEngine* random);

// Converts a Projection object + float threshold into a proto condition of the
// same semantic. `projection` cannot be empty.
absl::Status SetCondition(const Projection& projection, float threshold,
                          const dataset::proto::DataSpecification& dataspec,
                          proto::NodeCondition* condition);

}  // namespace internal
}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_H_
