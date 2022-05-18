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

// Fast inference code for Decision Forest type models. The currently supported
// models are:
//   - Random Forest
//   - Gradient Boosted Trees
//   - Extra Trees
//
// With the following constraints:
//   - Binary classification with numerical input features (see
//   *BinaryClassificationNumericalOnlyFlatModel).
//   - Binary classification with numerical and categorical (less than 32 unique
//     values) input features (see
//     *OneDimensionOutputNumericalAndCategoricalFeatureNode).
//
// To speed up inference time, models are stored as a contiguous list of nodes
// called "flat node model".
//
// Usage example:
//
//   std::unique_ptr<model::AbstractModel> generic_model = ...
//   RandomForestBinaryClassificationNumericalFeatures serving_model;
//   GenericToSpecializedModel(
//     dynamic_cast<RandomForestModel*>(generic_model.get()), &serving_model);
//   std::vector<float> examples = ... 5 examples ...;
//   std::vector<float> predictions;
//   Predict(serving_model, examples, 5, &predictions);
//
// Unless stated otherwise, the prediction functions ("Predict*" function) are
// thread safe: A given model can be used a different threads simultaneously
// without mutex protection. Note: Unless highlighted, the model object is
// consumed as a constant reference.
//
#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_H_

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/serving/decision_forest/utils.h"
#include "yggdrasil_decision_forests/serving/example_set.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace decision_forest {

// Node for numerical input features, and single dimensional output (e.g. binary
// classification and regression). Doesn't handle Na values. Has small memory
// foot print (compared to the generic model). Supports a maximum of ~64k unique
// features and ~64k node per tree.
struct OneDimensionOutputNumericalFeatureNode {
  using NodeOffset = uint16_t;
  using FeatureIdx = uint16_t;

  NodeOffset right_idx;    // Offset to the positive child node. 0 if is leaf.
  FeatureIdx feature_idx;  // Tested attribute idx.
  union {
    float label;      // Output value (if this is a leaf).
    float threshold;  // Test threshold (if this is not a leaf).
  };

  // Simple one dimension output leaf constructor.
  static OneDimensionOutputNumericalFeatureNode Leaf(NodeOffset right_idx,
                                                     FeatureIdx feature_idx,
                                                     float label) {
    OneDimensionOutputNumericalFeatureNode node;
    node.right_idx = right_idx;
    node.feature_idx = feature_idx;
    node.label = label;
    return node;
  }
};

// Node for numerical and categorical input features, and single dimensional
// output.
struct OneDimensionOutputNumericalAndCategoricalFeatureNode {
  using NodeOffset = uint16_t;
  using FeatureIdx = int16_t;

  // Offset to the positive child node. 0 if is leaf.
  NodeOffset right_idx;
  // Tested attribute idx.
  //
  // If feature_idx is positive, the attribute is numerical and the condition is
  // a threshold test:
  //   attributes[feature_idx] >= threshold.
  //
  // If feature_idx is strictly negative, the attribute is categorical and the
  // condition is a bitmap OR operation:
  //   (attributes[-(feature_idx+1)] & mask) != 0
  FeatureIdx feature_idx;
  union {
    float label;      // Output value (if this is a leaf).
    float threshold;  // Test threshold (if numerical condition and not a leaf).
    uint32_t mask;    // Test mask (if categorical condition and not a leaf).
  };

  // Simple one dimension output leaf constructor.
  static OneDimensionOutputNumericalAndCategoricalFeatureNode Leaf(
      NodeOffset right_idx, FeatureIdx feature_idx, float label) {
    OneDimensionOutputNumericalAndCategoricalFeatureNode node;
    node.right_idx = right_idx;
    node.feature_idx = feature_idx;
    node.label = label;
    return node;
  }
};

// Generic node that support all types of features and all output size.
// This structure is more generic but less efficient than other node versions.
template <typename NodeOffsetRep>
struct GenericNode {
  using NodeOffset = NodeOffsetRep;
  using FeatureIdx = int16_t;

  // Offset to the positive child node. 0 if is leaf.
  NodeOffset right_idx;

  union {
    // Index of the feature being tested. For all condition types except
    // "kNumericalObliqueProjectionIsHigher".
    FeatureIdx feature_idx;

    // Number of oblique projections. Only for
    // "kNumericalObliqueProjectionIsHigher" conditions.
    FeatureIdx num_oblique_projections;
  };

  enum class Type : uint8_t {
    kLeaf,
    // The following symbol follow the convention:
    // {attribute type}{condition}{implementation; optional}
    kNumericalIsHigher,
    kCategoricalContainsMask,
    kCategoricalContainsBufferOffset,
    kCategoricalSetContainsBufferOffset,
    kNumericalObliqueProjectionIsHigher
  };
  // Type of the node (leaf or condition type).
  Type type;

  union {
    // Output value (if this is a leaf) if the node is a leaf. Used for
    // single-dimensional output (e.G. binary classification RF, regressive
    // tree).
    float label;

    // Output value (if this is a leaf) if the node is a leaf. Used for
    // multi-dimensional output. The predictions are
    // label_buffer[label_buffer_offset+class_idx].
    uint32_t label_buffer_offset;

    // Numerical condition as "attribute >= threshold".
    // Also used for discretized numerical features.
    float numerical_is_higher_threshold;

    // Categorical condition as "categorical_contains_mask[attribute]", where
    // "categorical_contains_mask" is a bitmap. Only for attribute<32.
    uint32_t categorical_contains_mask;

    // Categorical condition as "categorical_mask_buffer[offset+attribute]",
    // where "categorical_contains_mask" is a bitmap.
    uint32_t categorical_contains_buffer_offset;

    // Offset to the first projection in
    // "oblique_{weights,feature_idxs}". The condition is
    // defined as: \sum_{i in 0..num_oblique_projections}
    // weights[offset+i]
    // * feature_values[oblique_feature_idxs[offset+i].feature_idx] >=
    // weights[offset+num_oblique_projections].
    uint32_t oblique_projection_offset;
  };

  // Simple one dimension output leaf constructor.
  static GenericNode<NodeOffsetRep> Leaf(NodeOffset right_idx,
                                         FeatureIdx feature_idx, float label) {
    static GenericNode<NodeOffsetRep> node;
    node.right_idx = right_idx;
    node.feature_idx = feature_idx;
    node.label = label;
    return node;
  }

  // Simple multi-dimensions output leaf constructor.
  static GenericNode<NodeOffsetRep> LeafMulticlassClassification(
      NodeOffset right_idx, FeatureIdx feature_idx, Type type,
      uint32_t label_buffer_offset) {
    GenericNode<NodeOffsetRep> node;
    node.right_idx = right_idx;
    node.feature_idx = feature_idx;
    node.type = type;
    node.label_buffer_offset = label_buffer_offset;
    return node;
  }

  // Simple categorical-uplift output leaf constructor.
  static GenericNode<NodeOffsetRep> LeafCategoricalUplift(
      NodeOffset right_idx, FeatureIdx feature_idx, Type type,
      uint32_t label_buffer_offset) {
    return LeafMulticlassClassification(right_idx, feature_idx, type,
                                        label_buffer_offset);
  }
};

// A generic decision forest.
template <typename Node, typename Value>
struct FlatNodeModel {
  using ValueType = Value;
  using NodeType = Node;
  using FeaturesDefinition = FeaturesDefinitionNumericalOrCategoricalFlat;

  using ExampleSet =
      ExampleSetNumericalOrCategoricalFlat<FlatNodeModel<Node, Value>,
                                           ExampleFormat::FORMAT_EXAMPLE_MAJOR>;

  const FeaturesDefinition& features() const { return internal_features; }

  FeaturesDefinition* mutable_features() { return &internal_features; }

  // The list of nodes in the model.
  std::vector<Node> nodes;
  // The indices (in "nodes") of the root nodes.
  std::vector<int32_t> root_offsets;

  FeaturesDefinition internal_features;

  // Buffer of label values. Used for multi-dimensional output trees.
  // See the description of "label_buffer_offset".
  std::vector<float> label_buffer;

  // Buffer of categorical mask to use for categorical condition.
  std::vector<bool> categorical_mask_buffer;

  // Buffer for oblique projection splits. See "GenericNode" for the
  // documentation about these fields.
  std::vector<float> oblique_weights;
  std::vector<typename Node::FeatureIdx> oblique_internal_feature_idxs;

  model::proto::Metadata metadata;
};

// Specialized models.

// Random Forest model for binary classification with numerical input features.
struct RandomForestBinaryClassificationNumericalFeatures
    : FlatNodeModel<OneDimensionOutputNumericalFeatureNode, float> {
  static constexpr model::proto::Task kTask =
      model::proto::Task::CLASSIFICATION;
};

// Random Forest model for binary classification with numerical and categorical
// (less than 32 unique categories) features.
struct RandomForestBinaryClassificationNumericalAndCategoricalFeatures
    : FlatNodeModel<OneDimensionOutputNumericalAndCategoricalFeatureNode,
                    NumericalOrCategoricalValue> {
  static constexpr model::proto::Task kTask =
      model::proto::Task::CLASSIFICATION;
};

// Random Forest model for regression with numerical input features.
struct RandomForestRegressionNumericalOnly
    : FlatNodeModel<OneDimensionOutputNumericalFeatureNode, float> {
  static constexpr model::proto::Task kTask = model::proto::Task::REGRESSION;
};

// Random Forest model for regression with numerical and categorical
// (less than 32 unique input features.
struct RandomForestRegressionNumericalAndCategorical
    : FlatNodeModel<OneDimensionOutputNumericalAndCategoricalFeatureNode,
                    NumericalOrCategoricalValue> {
  static constexpr model::proto::Task kTask = model::proto::Task::REGRESSION;
};

// Gradient Boosted Trees model for binary classification with numerical input
// features.
struct GradientBoostedTreesBinaryClassificationNumericalOnly
    : FlatNodeModel<OneDimensionOutputNumericalFeatureNode, float> {
  static constexpr model::proto::Task kTask =
      model::proto::Task::CLASSIFICATION;
  // Value to add on the predicted value.
  float initial_predictions = 0.f;
};

// Gradient Boosted Trees for binary classification with numerical and
// categorical (less than 32 unique values) input features.
struct GradientBoostedTreesBinaryClassificationNumericalAndCategorical
    : FlatNodeModel<OneDimensionOutputNumericalAndCategoricalFeatureNode,
                    NumericalOrCategoricalValue> {
  static constexpr model::proto::Task kTask =
      model::proto::Task::CLASSIFICATION;
  // Output of the model before any tree is applied, and before the final
  // activation function.
  float initial_predictions = 0.f;
};

// Gradient Boosted Trees model for regression with numerical input
// features.
struct GradientBoostedTreesRegressionNumericalOnly
    : FlatNodeModel<OneDimensionOutputNumericalFeatureNode, float> {
  static constexpr model::proto::Task kTask = model::proto::Task::REGRESSION;
  // Output of the model before any tree is applied, and before the final
  // activation function.
  float initial_predictions = 0.f;
};

// Gradient Boosted Trees for regression with numerical and
// categorical (less than 32 unique input features.
struct GradientBoostedTreesRegressionNumericalAndCategorical
    : FlatNodeModel<OneDimensionOutputNumericalAndCategoricalFeatureNode,
                    NumericalOrCategoricalValue> {
  static constexpr model::proto::Task kTask = model::proto::Task::REGRESSION;
  // Output of the model before any tree is applied, and before the final
  // activation function.
  float initial_predictions = 0.f;
};

// Gradient Boosted Trees model for ranking with numerical input
// features.
struct GradientBoostedTreesRankingNumericalOnly

    : FlatNodeModel<OneDimensionOutputNumericalFeatureNode, float> {
  static constexpr model::proto::Task kTask = model::proto::Task::RANKING;
  // Output of the model before any tree is applied, and before the final
  // activation function.
  float initial_predictions = 0.f;
};

// Gradient Boosted Trees for ranking with numerical and
// categorical (less than 32 unique input features.
struct GradientBoostedTreesRankingNumericalAndCategorical
    : FlatNodeModel<OneDimensionOutputNumericalAndCategoricalFeatureNode,
                    NumericalOrCategoricalValue> {
  static constexpr model::proto::Task kTask = model::proto::Task::RANKING;
  // Output of the model before any tree is applied, and before the final
  // activation function.
  float initial_predictions = 0.f;
};

// Models using the ExampleSet API.
// "NodeIndex" is the precision required to store a node offset in a
// single tree. This offset is bounded by the number of nodes in a tree (and
// generally ~50% smaller).
template <typename NodeOffsetRep = uint16_t>
struct ExampleSetModel
    : FlatNodeModel<GenericNode<NodeOffsetRep>, NumericalOrCategoricalValue> {
  using ExampleSet =
      ExampleSetNumericalOrCategoricalFlat<ExampleSetModel<NodeOffsetRep>,
                                           ExampleFormat::FORMAT_EXAMPLE_MAJOR>;
};

struct ExampleSetModelManyNodes : ExampleSetModel<uint32_t> {};

// Random Forest model for binary classification.
template <typename NodeOffsetRep = uint16_t>
struct GenericRandomForestBinaryClassification
    : ExampleSetModel<NodeOffsetRep> {
  static constexpr model::proto::Task kTask =
      model::proto::Task::CLASSIFICATION;
};
using RandomForestBinaryClassification =
    GenericRandomForestBinaryClassification<>;

// Random Forest model for multi-class classification.
template <typename NodeOffsetRep = uint16_t>
struct GenericRandomForestMulticlassClassification
    : ExampleSetModel<NodeOffsetRep> {
  static constexpr model::proto::Task kTask =
      model::proto::Task::CLASSIFICATION;
  int num_classes;
};
using RandomForestMulticlassClassification =
    GenericRandomForestMulticlassClassification<>;

// Random Forest model for regression.
template <typename NodeOffsetRep = uint16_t>
struct GenericRandomForestRegression : ExampleSetModel<NodeOffsetRep> {
  static constexpr model::proto::Task kTask = model::proto::Task::REGRESSION;
};
using RandomForestRegression = GenericRandomForestRegression<>;

// Random Forest model for categorical uplift.
template <typename NodeOffsetRep = uint16_t>
struct GenericRandomForestCategoricalUplift : ExampleSetModel<NodeOffsetRep> {
  static constexpr model::proto::Task kTask =
      model::proto::Task::CATEGORICAL_UPLIFT;
  int num_classes;
};
using RandomForestCategoricalUplift = GenericRandomForestCategoricalUplift<>;

// Random Forest model for numerical uplift.
template <typename NodeOffsetRep = uint16_t>
struct GenericRandomForestNumericalUplift : ExampleSetModel<NodeOffsetRep> {
  static constexpr model::proto::Task kTask =
      model::proto::Task::NUMERICAL_UPLIFT;
};
using RandomForestNumericalUplift = GenericRandomForestNumericalUplift<>;

// GBDT model for binary classification.
template <typename NodeOffsetRep = uint16_t>
struct GenericGradientBoostedTreesBinaryClassification
    : ExampleSetModel<NodeOffsetRep> {
  static constexpr model::proto::Task kTask =
      model::proto::Task::CLASSIFICATION;
  // Output of the model before any tree is applied, and before the final
  // activation function.
  float initial_predictions = 0.f;
  bool output_logits = false;
};
using GradientBoostedTreesBinaryClassification =
    GenericGradientBoostedTreesBinaryClassification<>;

// GBDT model for multi-class classification.
template <typename NodeOffsetRep = uint16_t>
struct GenericGradientBoostedTreesMulticlassClassification
    : ExampleSetModel<NodeOffsetRep> {
  static constexpr model::proto::Task kTask =
      model::proto::Task::CLASSIFICATION;
  int num_classes;
  std::vector<float> initial_predictions;
  bool output_logits = false;
};
using GradientBoostedTreesMulticlassClassification =
    GenericGradientBoostedTreesMulticlassClassification<>;

// GBDT model for regression.
template <typename NodeOffsetRep = uint16_t>
struct GenericGradientBoostedTreesRegression : ExampleSetModel<NodeOffsetRep> {
  static constexpr model::proto::Task kTask = model::proto::Task::REGRESSION;
  // Output of the model before any tree is applied, and before the final
  // activation function.
  float initial_predictions = 0.f;
};
using GradientBoostedTreesRegression = GenericGradientBoostedTreesRegression<>;

// GBDT model for ranking.
template <typename NodeOffsetRep = uint16_t>
struct GenericGradientBoostedTreesRanking : ExampleSetModel<NodeOffsetRep> {
  static constexpr model::proto::Task kTask = model::proto::Task::RANKING;
  // Output of the model before any tree is applied, and before the final
  // activation function.
  float initial_predictions = 0.f;
};
using GradientBoostedTreesRanking = GenericGradientBoostedTreesRanking<>;

template <typename AbstractModel, typename CompiledModel>
absl::Status GenericToSpecializedModel(const AbstractModel& src,
                                       CompiledModel* dst);

template <typename Model>
void Predict(const Model& model, const typename Model::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions);

// Converts a generic model into a specialized model.
//
// Returns an error if the model is not compatible.
//
// Args:
//    src: Generic Random Forest model.
//    dst: Specialized Random Forest model.

absl::Status GenericToSpecializedModel(
    const model::random_forest::RandomForestModel& src,
    RandomForestBinaryClassificationNumericalFeatures* dst);

absl::Status GenericToSpecializedModel(
    const model::random_forest::RandomForestModel& src,
    RandomForestBinaryClassificationNumericalAndCategoricalFeatures* dst);

absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesBinaryClassificationNumericalOnly* dst);

absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesBinaryClassificationNumericalAndCategorical* dst);

absl::Status GenericToSpecializedModel(
    const model::random_forest::RandomForestModel& src,
    RandomForestRegressionNumericalOnly* dst);

absl::Status GenericToSpecializedModel(
    const model::random_forest::RandomForestModel& src,
    RandomForestRegressionNumericalAndCategorical* dst);

absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesRegressionNumericalOnly* dst);

absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesRegressionNumericalAndCategorical* dst);

absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesRankingNumericalOnly* dst);

absl::Status GenericToSpecializedModel(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& src,
    GradientBoostedTreesRankingNumericalAndCategorical* dst);

// Generates the predictions of a model on a batch of examples.
//
// Args:
//    model: The model.
//    examples: A example-major set of examples. The features should be ordered
//      as defined by "model.feature_names" when building the flat node
//      model. Should contains num_examples * model.feature_names.size()
//      values.
//    num_examples: Number of examples.
//    predictions: The predictions.
//
// "Predict" is a simple inference solution that iterates over all the examples
// and all the trees iteratively.
//
// "PredictOptimizedV1" iterates over trees in batches (bathes of trees) which
// is more efficient for RAM access.
//
// "PredictOptimizedV1" is estimated to be ~ 60% faster than "Predict" (i.e.
// ~2.5x speed-up) on the QKMS dataset. This figure can be re-estimated by
// running the benchmark (:benchmark).
//
// Ram access is the bottleneck for decision tree evaluation. A node need to be
// evaluated in order to select the correct child to visit. Most of the time is
// spent waiting for the node data (e.g. threshold value and position of the
// children). "Predict" treats the trees one after another. Instead,
// "PredictOptimizedV1" evaluations "kTreeBatchSize" trees at the same time.
//
// The optimal value for "kTreeBatchSize" was estimated to 8, among {4, 6, 8,
// 10}, using the QKMS dataset with a RandomForest with max depth 16 and average
// depth 12. To avoid breaking existing test, this value was set to 5 in the
// case of "BinaryClassificationNumericalAndCategoricalOnlyFlatModel" (see the
// next to-do). This value of 5 is close to optimal.
//
void Predict(const RandomForestBinaryClassificationNumericalFeatures& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions);

void Predict(
    const RandomForestBinaryClassificationNumericalAndCategoricalFeatures&
        model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions);

void Predict(const GradientBoostedTreesBinaryClassificationNumericalOnly& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions);

void Predict(
    const GradientBoostedTreesBinaryClassificationNumericalAndCategorical&
        model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions);

void Predict(const RandomForestRegressionNumericalOnly& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions);

void Predict(const RandomForestRegressionNumericalAndCategorical& model,
             const std::vector<NumericalOrCategoricalValue>& examples,
             int num_examples, std::vector<float>* predictions);

void Predict(const GradientBoostedTreesRegressionNumericalOnly& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions);

void Predict(const GradientBoostedTreesRegressionNumericalAndCategorical& model,
             const std::vector<NumericalOrCategoricalValue>& examples,
             int num_examples, std::vector<float>* predictions);

void Predict(const GradientBoostedTreesRankingNumericalOnly& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions);

void Predict(const GradientBoostedTreesRankingNumericalAndCategorical& model,
             const std::vector<NumericalOrCategoricalValue>& examples,
             int num_examples, std::vector<float>* predictions);

template <typename Model>
void PredictWithExampleSet(const Model& model,
                           const typename Model::ExampleSet& examples,
                           int num_examples, std::vector<float>* predictions) {
  Predict(model, examples.InternalCategoricalAndNumericalValues(), num_examples,
          predictions);
}

// Note: Requires for the number of trees to be a multiple of 8.
void PredictOptimizedV1(
    const RandomForestBinaryClassificationNumericalFeatures& model,
    const std::vector<float>& examples, int num_examples,
    std::vector<float>* predictions);

// Note: Requires for the number of trees to be a multiple of 5.
void PredictOptimizedV1(
    const RandomForestBinaryClassificationNumericalAndCategoricalFeatures&
        model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions);

// Note: Requires for the number of trees to be a multiple of 5.
void PredictOptimizedV1(
    const GradientBoostedTreesBinaryClassificationNumericalOnly& model,
    const std::vector<float>& examples, int num_examples,
    std::vector<float>* predictions);

// Note: Requires for the number of trees to be a multiple of 5.
void PredictOptimizedV1(
    const GradientBoostedTreesBinaryClassificationNumericalAndCategorical&
        model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions);

// Note: Requires for the number of trees to be a multiple of 8.
void PredictOptimizedV1(const RandomForestRegressionNumericalOnly& model,
                        const std::vector<float>& examples, int num_examples,
                        std::vector<float>* predictions);

// Note: Requires for the number of trees to be a multiple of 5.
void PredictOptimizedV1(
    const RandomForestRegressionNumericalAndCategorical& model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions);

// Note: Requires for the number of trees to be a multiple of 5.
void PredictOptimizedV1(
    const GradientBoostedTreesRegressionNumericalOnly& model,
    const std::vector<float>& examples, int num_examples,
    std::vector<float>* predictions);

// Note: Requires for the number of trees to be a multiple of 5.
void PredictOptimizedV1(
    const GradientBoostedTreesRegressionNumericalAndCategorical& model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions);

// Note: Requires for the number of trees to be a multiple of 5.
void PredictOptimizedV1(const GradientBoostedTreesRankingNumericalOnly& model,
                        const std::vector<float>& examples, int num_examples,
                        std::vector<float>* predictions);

// Note: Requires for the number of trees to be a multiple of 5.
void PredictOptimizedV1(
    const GradientBoostedTreesRankingNumericalAndCategorical& model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions);

// Loads a batch a examples from a vertical dataset (i.e. column major generic
// dataset stored in memory) into a flat batch. This code is inefficient and
// should not be used for time critical applications. This functions replaces
// the Na values with "na_replacement_values".
//
// If the format is not example major, and if "batch_size" is provided, the
// output values are organized so that examples in a given batch are grouped
// together. For example, suppose: num_example=4, features={a,b}, batch_size=2.
// Let's a_i be the "a" feature value of the i_th example.
//
// The output will be ordered as follow for the different example formats:
//
// FORMAT_EXAMPLE_MAJOR:
//   a0 b0 a1 b1 a2 b2 a3 b3
//
// FORMAT_FEATURE_MAJOR:
//   a0 a1 b0 b1 a2 a3 b2 b3
//     Details:
//       - Batch 0: [a0 a1 b0 b1], Batch 1: [a2 a3, b2 b3]
//       - Feature major within batch.
//
// Args:
//    dataset: The source vertical dataset.
//    begin_example_idx: Beginning index of the batch.
//    end_example_idx: End (excluded) index of the batch.
//    feature_names: Names and order of the features to extract.
//    na_replacement_values: Replacement value for non available features.
//    Specified in the same order as "feature_names". Should have the same size
//    as "feature_names".
//    flat_examples: Output flatten batch of examples.
//    example_format: Internal format of the example.
//    batch_size: Batch size of example. If set, groups the examples into
//    batches. Only used for feature-major format.
//
absl::Status LoadFlatBatchFromDataset(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t begin_example_idx,
    dataset::VerticalDataset::row_t end_example_idx,
    const std::vector<std::string>& feature_names,
    const std::vector<NumericalOrCategoricalValue>& na_replacement_values,
    std::vector<float>* flat_examples,
    ExampleFormat example_format = ExampleFormat::FORMAT_EXAMPLE_MAJOR,
    absl::optional<int64_t> batch_size = {});

absl::Status LoadFlatBatchFromDataset(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t begin_example_idx,
    dataset::VerticalDataset::row_t end_example_idx,
    const std::vector<std::string>& feature_names,
    const std::vector<NumericalOrCategoricalValue>& na_replacement_values,
    std::vector<NumericalOrCategoricalValue>* flat_examples,
    ExampleFormat example_format = ExampleFormat::FORMAT_EXAMPLE_MAJOR,
    absl::optional<int64_t> batch_size = {});

std::vector<NumericalOrCategoricalValue> FloatToValue(
    const std::vector<float>& values);

}  // namespace decision_forest
}  // namespace serving
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_H_
