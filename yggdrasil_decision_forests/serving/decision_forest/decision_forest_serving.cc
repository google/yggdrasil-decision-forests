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

#include "yggdrasil_decision_forests/serving/decision_forest/decision_forest_serving.h"

#include "yggdrasil_decision_forests/utils/usage.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace decision_forest {

template <typename SpecializedModel>
void ActivationMultiDimIdentity(const SpecializedModel& model,
                                float* const values, const int num_values) {}

// Final function applied by a Gradient Boosted Trees with
// BINOMIAL_LOG_LIKELIHOOD loss function.
template <typename SpecializedModel>
float ActivationGradientBoostedTreesBinomialLogLikelihood(
    const SpecializedModel& model, const float value) {
  return utils::clamp(
      1.f / (1.f + std::exp(-(value + model.initial_predictions))), 0.f, 1.f);
}

// Final function applied by a Gradient Boosted Trees with
// SQUARED_ERROR loss function.
template <typename SpecializedModel>
float ActivationAddInitialPrediction(const SpecializedModel& model,
                                     const float value) {
  return value + model.initial_predictions;
}

// Final function applied by a Gradient Boosted Trees with
// MULTINOMIAL_LOG_LIKELIHOOD loss function. I.e. this is a softmax function.
template <typename SpecializedModel>
void ActivationGradientBoostedTreesMultinomialLogLikelihood(
    const SpecializedModel& model, float* const values, const int num_values) {
  float* cache = static_cast<float*>(alloca(sizeof(float) * num_values));
  float sum = 0;
  for (int i = 0; i < num_values; i++) {
    const float value = std::exp(values[i]);
    cache[i] = value;
    sum += value;
  }
  const float noramlize = 1.f / sum;
  for (int i = 0; i < num_values; i++) {
    values[i] = cache[i] * noramlize;
  }
}

// Identity transformation for the output of a decision forest model.
// Default value for the "FinalTransform" argument in "PredictHelper".
//
// Note: Lambda default segfaults clang.
template <typename Model>
float Idendity(const Model& model, const float value) {
  return value;
}

template <typename Model>
float Clamp01(const Model& model, const float value) {
  return utils::clamp(value, 0.f, 1.f);
}

// Evaluates a numerical condition.
inline bool EvalCondition(const OneDimensionOutputNumericalFeatureNode* node,
                          const float* example) {
  return example[node->feature_idx] >= node->threshold;
}

// Evaluates a numerical or categorical condition.
inline bool EvalCondition(
    const OneDimensionOutputNumericalAndCategoricalFeatureNode* node,
    const NumericalOrCategoricalValue* example) {
  if (node->feature_idx >= 0) {
    // Numerical condition.
    return example[node->feature_idx].numerical_value >= node->threshold;
  } else {
    // Categorical condition.
    const uint32_t example_mask =
        1 << example[-(node->feature_idx + 1)].categorical_value;
    return (example_mask & node->mask) != 0;
  }
}

template <typename Model>
inline bool EvalCondition(const typename Model::NodeType* node,
                          const typename Model::ExampleSet& examples,
                          const int example_idx, const Model& model) {
  using GenericNode = typename Model::NodeType;
  switch (node->type) {
    case GenericNode::Type::kNumericalIsHigherMissingIsFalse:
    case GenericNode::Type::kNumericalIsHigherMissingIsTrue: {
      const auto attribute_value =
          examples.GetNumerical(example_idx, {node->feature_idx}, model);

      // Note: It is faster to have this second check, than having a different
      // case.
      if (node->type == GenericNode::Type::kNumericalIsHigherMissingIsFalse) {
        return attribute_value >= node->numerical_is_higher_threshold;
      } else {
        return !(attribute_value < node->numerical_is_higher_threshold);
      }
    }

    case GenericNode::Type::kCategoricalContainsMask: {
      const uint32_t attribute_value =
          examples.GetCategoricalInt(example_idx, {node->feature_idx}, model);
      return ((1 << attribute_value) & node->categorical_contains_mask) != 0;
    }

    case GenericNode::Type::kCategoricalContainsBufferOffset: {
      const auto attribute_value =
          examples.GetCategoricalInt(example_idx, {node->feature_idx}, model);
      return model
          .categorical_mask_buffer[node->categorical_contains_buffer_offset +
                                   attribute_value];
    }

    case GenericNode::Type::kCategoricalSetContainsBufferOffset: {
      const auto& range_values =
          examples.InternalCategoricalSetBeginAndEnds()
              [node->feature_idx * examples.NumberOfExamples() + example_idx];
      for (int value_idx = range_values.begin; value_idx < range_values.end;
           value_idx++) {
        const auto attribute_value =
            examples.InternalCategoricalItemBuffer()[value_idx];
        if (model.categorical_mask_buffer
                [node->categorical_contains_buffer_offset + attribute_value]) {
          return true;
        }
      }
      return false;
    }

    case GenericNode::Type::kNumericalObliqueProjectionIsHigher: {
      float sum = 0;
      const auto attributes = model.oblique_internal_feature_idxs.begin() +
                              node->oblique_projection_offset;
      const auto weights =
          model.oblique_weights.begin() + node->oblique_projection_offset;

      const uint32_t num_projection = node->feature_idx;
      for (uint32_t projection_idx = 0; projection_idx < num_projection;
           projection_idx++) {
        const auto attribute_value = examples.GetNumerical(
            example_idx, {attributes[projection_idx]}, model);
        const float weight = weights[projection_idx];
        sum += weight * attribute_value;
      }
      return sum >= model.oblique_weights[node->oblique_projection_offset +
                                          num_projection];
    }

    default:
      NOTREACHED();
      return false;
  }
}

// Basic inference of a decision forest on a set of trees.
template <typename Model,
          float (*FinalTransform)(const Model&, const float) = Idendity<Model>>
inline void PredictHelper(
    const Model& model, const std::vector<typename Model::ValueType>& examples,
    int num_examples, std::vector<float>* predictions) {
  utils::usage::OnInference(num_examples, model.metadata);
  const int num_features = model.features().fixed_length_features().size();
  predictions->resize(num_examples);
  const typename Model::ValueType* sample = examples.data();
  for (int example_idx = 0; example_idx < num_examples; ++example_idx) {
    float output = 0.f;

    for (const auto root_node_idx : model.root_offsets) {
      const auto* node = &model.nodes[root_node_idx];
      while (node->right_idx) {
        node += EvalCondition(node, sample) ? node->right_idx : 1;
      }
      output += node->label;
    }

    (*predictions)[example_idx] = FinalTransform(model, output);
    sample += num_features;
  }
}

template <typename Model,
          float (*FinalTransform)(const Model&, const float) /*= Idendity*/>
inline void PredictHelper(const Model& model,
                          const typename Model::ExampleSet& examples,
                          int num_examples, std::vector<float>* predictions) {
  utils::usage::OnInference(num_examples, model.metadata);
  predictions->resize(num_examples);
  for (int example_idx = 0; example_idx < num_examples; ++example_idx) {
    float output = 0.f;
    for (const auto root_node_idx : model.root_offsets) {
      const auto* node = &model.nodes[root_node_idx];
      while (node->right_idx) {
        node += EvalCondition(node, examples, example_idx, model)
                    ? node->right_idx
                    : 1;
      }
      output += node->label;
    }
    (*predictions)[example_idx] = FinalTransform(model, output);
  }
}

template <typename Model,
          float (*FinalTransform)(const Model&, const float) /*= Idendity*/>
inline void PredictHelperMultiDimensionTrees(
    const Model& model, const typename Model::ExampleSet& examples,
    int num_examples, std::vector<float>* predictions) {
  utils::usage::OnInference(num_examples, model.metadata);
  predictions->assign(num_examples * model.num_classes, 0.f);
  float* cur_predictions = &(*predictions)[0];
  for (int example_idx = 0; example_idx < num_examples; ++example_idx) {
    for (const auto root_node_idx : model.root_offsets) {
      const auto* node = &model.nodes[root_node_idx];
      while (node->right_idx) {
        node += EvalCondition(node, examples, example_idx, model)
                    ? node->right_idx
                    : 1;
      }
      for (int class_idx = 0; class_idx < model.num_classes; class_idx++) {
        cur_predictions[class_idx] +=
            model.label_buffer[node->label_buffer_offset + class_idx];
      }
    }
    for (int class_idx = 0; class_idx < model.num_classes; class_idx++) {
      cur_predictions[class_idx] =
          FinalTransform(model, cur_predictions[class_idx]);
    }
    cur_predictions += model.num_classes;
  }
}

template <typename Model,
          void (*FinalTransform)(const Model&, float* const, const int)>
inline void PredictHelperMultiDimensionFromSingleDimensionTrees(
    const Model& model, const typename Model::ExampleSet& examples,
    int num_examples, std::vector<float>* predictions) {
  predictions->assign(num_examples * model.num_classes, 0.f);
  float* cur_predictions = &(*predictions)[0];
  for (int example_idx = 0; example_idx < num_examples; ++example_idx) {
    int class_idx = 0;
    for (const auto root_node_idx : model.root_offsets) {
      const auto* node = &model.nodes[root_node_idx];
      while (node->right_idx) {
        node += EvalCondition(node, examples, example_idx, model)
                    ? node->right_idx
                    : 1;
      }
      cur_predictions[class_idx] += node->label;
      class_idx = (class_idx + 1) % model.num_classes;
    }
    FinalTransform(model, cur_predictions, model.num_classes);
    cur_predictions += model.num_classes;
  }
}

// See the documentation of "PredictOptimizedV1".
template <typename Model,
          float (*FinalTransform)(const Model&, const float) = Idendity<Model>,
          int kTreeBatchSize = 5>
inline void PredictHelperOptimizedV1(
    const Model& model, const std::vector<typename Model::ValueType>& examples,
    int num_examples, std::vector<float>* predictions) {
  utils::usage::OnInference(num_examples, model.metadata);
  // A Group of "kTreeBatchSize" trees is called a "tree batch".
  predictions->resize(num_examples);

  if (num_examples == 0) {
    return;
  }

  const int num_tree_batches = model.root_offsets.size() / kTreeBatchSize;
  const int num_remaining_trees =
      model.root_offsets.size() - num_tree_batches * kTreeBatchSize;

  // The active nodes in the current tree batch. If "nodes[i]==nullptr", the
  // tree "i" is disabled i.e. it reached a leaf.
  const typename Model::NodeType* nodes[kTreeBatchSize];

  // The number of actives nodes.
  int num_active;

  // Number of input features for the model.
  const int num_features = model.features().fixed_length_features().size();

  // Select the first example.
  // Note: The examples are stored example-major/feature-minor.
  const typename Model::ValueType* sample = examples.data();
  for (size_t example_idx = 0; example_idx < num_examples; ++example_idx) {
    // Accumulator of the predictions for the current example.
    float output = 0.f;

    // Select the first tree bath.
    auto current_root_node_offset = &model.root_offsets[0];

    for (int tree_batch_idx = 0; tree_batch_idx < num_tree_batches;
         ++tree_batch_idx) {
      // Initialize "nodes" to the roots of the "kTreeBatchSize" trees in the
      // tree batch.
      //
      // Note: "#pragma clang loop unroll(full)" ensures that the loop is
      // unrolled.
#pragma clang loop unroll(full)
      for (int tree_in_batch_idx = 0; tree_in_batch_idx < kTreeBatchSize;
           ++tree_in_batch_idx) {
        nodes[tree_in_batch_idx] =
            &model.nodes[*(current_root_node_offset + tree_in_batch_idx)];
      }
      current_root_node_offset += kTreeBatchSize;
      num_active = kTreeBatchSize;

      // While not all nodes are disabled.
      while (num_active) {
#pragma clang loop unroll(full)
        for (int tree_in_batch_idx = 0; tree_in_batch_idx < kTreeBatchSize;
             ++tree_in_batch_idx) {
          if (nodes[tree_in_batch_idx]) {
            if (nodes[tree_in_batch_idx]->right_idx) {
              // Evaluates the node and go to the correct child.
              nodes[tree_in_batch_idx] +=
                  EvalCondition(nodes[tree_in_batch_idx], sample)
                      ? nodes[tree_in_batch_idx]->right_idx
                      : 1;
            } else {
              // Add the node value to the prediction accumulator.
              output += nodes[tree_in_batch_idx]->label;
              // Disable the node
              --num_active;
              nodes[tree_in_batch_idx] = nullptr;
            }
          }
        }
      }
    }

    for (int tree_in_batch_idx = 0; tree_in_batch_idx < num_remaining_trees;
         tree_in_batch_idx++) {
      auto node = &model.nodes[*(current_root_node_offset + tree_in_batch_idx)];
      while (node->right_idx) {
        node += EvalCondition(node, sample) ? node->right_idx : 1;
      }
      output += node->label;
    }

    // Move to the next example.
    sample += num_features;
    // Store the prediction accumulator result.
    (*predictions)[example_idx] = FinalTransform(model, output);
  }
}

void Predict(const RandomForestBinaryClassificationNumericalFeatures& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Clamp01>(
      model, examples, num_examples, predictions);
}

void Predict(
    const RandomForestBinaryClassificationNumericalAndCategoricalFeatures&
        model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Clamp01>(
      model, examples, num_examples, predictions);
}

void Predict(const GradientBoostedTreesBinaryClassificationNumericalOnly& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationGradientBoostedTreesBinomialLogLikelihood>(
      model, examples, num_examples, predictions);
}

void Predict(
    const GradientBoostedTreesBinaryClassificationNumericalAndCategorical&
        model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationGradientBoostedTreesBinomialLogLikelihood>(
      model, examples, num_examples, predictions);
}

void Predict(const RandomForestRegressionNumericalOnly& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Idendity>(
      model, examples, num_examples, predictions);
}

void Predict(const RandomForestRegressionNumericalAndCategorical& model,
             const std::vector<NumericalOrCategoricalValue>& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Idendity>(
      model, examples, num_examples, predictions);
}

void Predict(const GradientBoostedTreesRegressionNumericalOnly& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationAddInitialPrediction>(model, examples, num_examples,
                                                predictions);
}

void Predict(const GradientBoostedTreesRegressionNumericalAndCategorical& model,
             const std::vector<NumericalOrCategoricalValue>& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationAddInitialPrediction>(model, examples, num_examples,
                                                predictions);
}

void Predict(const GradientBoostedTreesRankingNumericalOnly& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationAddInitialPrediction>(model, examples, num_examples,
                                                predictions);
}

void Predict(const GradientBoostedTreesRankingNumericalAndCategorical& model,
             const std::vector<NumericalOrCategoricalValue>& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationAddInitialPrediction>(model, examples, num_examples,
                                                predictions);
}

void PredictOptimizedV1(
    const RandomForestBinaryClassificationNumericalFeatures& model,
    const std::vector<float>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<
      RandomForestBinaryClassificationNumericalFeatures,
      Idendity<RandomForestBinaryClassificationNumericalFeatures>>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(
    const RandomForestBinaryClassificationNumericalAndCategoricalFeatures&
        model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<
      RandomForestBinaryClassificationNumericalAndCategoricalFeatures,
      Idendity<
          RandomForestBinaryClassificationNumericalAndCategoricalFeatures>>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(
    const GradientBoostedTreesBinaryClassificationNumericalOnly& model,
    const std::vector<float>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<
      GradientBoostedTreesBinaryClassificationNumericalOnly,
      ActivationGradientBoostedTreesBinomialLogLikelihood>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(
    const GradientBoostedTreesBinaryClassificationNumericalAndCategorical&
        model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<
      GradientBoostedTreesBinaryClassificationNumericalAndCategorical,
      ActivationGradientBoostedTreesBinomialLogLikelihood>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(const RandomForestRegressionNumericalOnly& model,
                        const std::vector<float>& examples, int num_examples,
                        std::vector<float>* predictions) {
  PredictHelperOptimizedV1<RandomForestRegressionNumericalOnly,
                           Idendity<RandomForestRegressionNumericalOnly>>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(
    const RandomForestRegressionNumericalAndCategorical& model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<
      RandomForestRegressionNumericalAndCategorical,
      Idendity<RandomForestRegressionNumericalAndCategorical>>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(
    const GradientBoostedTreesRegressionNumericalOnly& model,
    const std::vector<float>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<GradientBoostedTreesRegressionNumericalOnly,
                           ActivationAddInitialPrediction>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(
    const GradientBoostedTreesRegressionNumericalAndCategorical& model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<
      GradientBoostedTreesRegressionNumericalAndCategorical,
      ActivationAddInitialPrediction>(model, examples, num_examples,
                                      predictions);
}

void PredictOptimizedV1(const GradientBoostedTreesRankingNumericalOnly& model,
                        const std::vector<float>& examples, int num_examples,
                        std::vector<float>* predictions) {
  PredictHelperOptimizedV1<GradientBoostedTreesRankingNumericalOnly,
                           ActivationAddInitialPrediction>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(
    const GradientBoostedTreesRankingNumericalAndCategorical& model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<GradientBoostedTreesRankingNumericalAndCategorical,
                           ActivationAddInitialPrediction>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(
    const RandomForestBinaryClassification& model,
    const typename RandomForestBinaryClassification::ExampleSet& examples,
    int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Clamp01>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(
    const RandomForestMulticlassClassification& model,
    const typename RandomForestMulticlassClassification::ExampleSet& examples,
    int num_examples, std::vector<float>* predictions) {
  PredictHelperMultiDimensionTrees<std::remove_reference<decltype(model)>::type,
                                   Clamp01>(model, examples, num_examples,
                                            predictions);
}

template <>
void Predict(const RandomForestRegression& model,
             const typename RandomForestRegression::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Idendity>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(const RandomForestCategoricalUplift& model,
             const typename RandomForestCategoricalUplift::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelperMultiDimensionTrees<std::remove_reference<decltype(model)>::type,
                                   Idendity>(model, examples, num_examples,
                                             predictions);
}

template <>
void Predict(const RandomForestNumericalUplift& model,
             const typename RandomForestNumericalUplift::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Idendity>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(const GenericRandomForestBinaryClassification<uint32_t>& model,
             const typename GenericRandomForestBinaryClassification<
                 uint32_t>::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Clamp01>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(const GenericRandomForestMulticlassClassification<uint32_t>& model,
             const typename GenericRandomForestMulticlassClassification<
                 uint32_t>::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelperMultiDimensionTrees<std::remove_reference<decltype(model)>::type,
                                   Clamp01>(model, examples, num_examples,
                                            predictions);
}

template <>
void Predict(const GenericRandomForestRegression<uint32_t>& model,
             const typename GenericRandomForestRegression<uint32_t>::ExampleSet&
                 examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Idendity>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(
    const GenericRandomForestCategoricalUplift<uint32_t>& model,
    const typename GenericRandomForestCategoricalUplift<uint32_t>::ExampleSet&
        examples,
    int num_examples, std::vector<float>* predictions) {
  PredictHelperMultiDimensionTrees<std::remove_reference<decltype(model)>::type,
                                   Idendity>(model, examples, num_examples,
                                             predictions);
}

template <>
void Predict(
    const GenericRandomForestNumericalUplift<uint32_t>& model,
    const typename GenericRandomForestNumericalUplift<uint32_t>::ExampleSet&
        examples,
    int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Idendity>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(
    const GradientBoostedTreesBinaryClassification& model,
    const typename GradientBoostedTreesBinaryClassification::ExampleSet&
        examples,
    int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationGradientBoostedTreesBinomialLogLikelihood>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(
    const GenericGradientBoostedTreesBinaryClassification<uint32_t>& model,
    const typename GenericGradientBoostedTreesBinaryClassification<
        uint32_t>::ExampleSet& examples,
    int num_examples, std::vector<float>* predictions) {
  if (model.output_logits) {
    PredictHelper<std::remove_reference<decltype(model)>::type, Idendity>(
        model, examples, num_examples, predictions);
  } else {
    PredictHelper<std::remove_reference<decltype(model)>::type,
                  ActivationGradientBoostedTreesBinomialLogLikelihood>(
        model, examples, num_examples, predictions);
  }
}

template <>
void Predict(
    const GradientBoostedTreesMulticlassClassification& model,
    const typename GradientBoostedTreesMulticlassClassification::ExampleSet&
        examples,
    int num_examples, std::vector<float>* predictions) {
  if (model.output_logits) {
    PredictHelperMultiDimensionFromSingleDimensionTrees<
        std::remove_reference<decltype(model)>::type,
        ActivationMultiDimIdentity>(model, examples, num_examples, predictions);
  } else {
    PredictHelperMultiDimensionFromSingleDimensionTrees<
        std::remove_reference<decltype(model)>::type,
        ActivationGradientBoostedTreesMultinomialLogLikelihood>(
        model, examples, num_examples, predictions);
  }
}

template <>
void Predict(
    const GradientBoostedTreesRegression& model,
    const typename GradientBoostedTreesRegression::ExampleSet& examples,
    int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationAddInitialPrediction>(model, examples, num_examples,
                                                predictions);
}

template <>
void Predict(const GradientBoostedTreesRanking& model,
             const typename GradientBoostedTreesRanking::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationAddInitialPrediction>(model, examples, num_examples,
                                                predictions);
}

}  // namespace decision_forest
}  // namespace serving
}  // namespace yggdrasil_decision_forests
