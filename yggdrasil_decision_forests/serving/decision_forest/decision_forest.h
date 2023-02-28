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
#include "yggdrasil_decision_forests/serving/decision_forest/decision_forest_serving.h"
#include "yggdrasil_decision_forests/serving/decision_forest/utils.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace decision_forest {

template <typename AbstractModel, typename CompiledModel>
absl::Status GenericToSpecializedModel(const AbstractModel& src,
                                       CompiledModel* dst);

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
