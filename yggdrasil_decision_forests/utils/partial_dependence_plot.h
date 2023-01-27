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

// This library computes and plots the partial Dependence Plot (PDP) of groups
// of input features.
//
// Because of the closeness in their computation, this library also computes and
// plots the conditional expectation of the labels given the input features. See
// chapter 13.10.2 of "The Elements of Statistical Learning"
// (https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf)
// for the details on the differences between the two measures.
//
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_PARTIAL_DEPENDENCE_PLOT_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_PARTIAL_DEPENDENCE_PLOT_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/partial_dependence_plot.pb.h"

namespace yggdrasil_decision_forests {
namespace utils {

using ConditionalExpectationPlotSet = proto::PartialDependencePlotSet;
using PartialDependencePlotSet = proto::PartialDependencePlotSet;

// Initializes a PartialDependencePlotSet, by initializing a
// PartialDependencePlot for each set of attributes specified by attribute_idxs.
// Each PartialDependencePlot is initialized by creating bins for each
// corresponding set of attributes and storing the center values of each bin for
// easy computation later.
absl::StatusOr<PartialDependencePlotSet> InitializePartialDependencePlotSet(
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<std::vector<int>>& attribute_idxs,
    const model::proto::Task& task, int label_col_idx, int num_numerical_bins,
    const dataset::VerticalDataset& dataset);

absl::StatusOr<ConditionalExpectationPlotSet>
InitializeConditionalExpectationPlotSet(
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<std::vector<int>>& attribute_idxs,
    const model::proto::Task& task, int label_col_idx, int num_numerical_bins,
    const dataset::VerticalDataset& dataset);

// Given a model and an example, updates the PartialDependencePlotSet.
// This is done by updating each bin in each PartialDependencePlot in the
// pdp_set.
// Each bin is updated by modifying the example at the attributes to reflect the
// center values from the bin. Then, based on the prediction of the model on
// this modified example, the statistics of this bin are updated.
absl::Status UpdatePartialDependencePlotSet(
    const model::AbstractModel& model, const dataset::proto::Example& example,
    proto::PartialDependencePlotSet* pdp_set);

absl::Status UpdateConditionalExpectationPlotSet(
    const model::AbstractModel& model, const dataset::proto::Example& example,
    ConditionalExpectationPlotSet* cond_set);

// Given a dataset and a model, computes a PartialDependencePlotSet locally.
absl::StatusOr<PartialDependencePlotSet> ComputePartialDependencePlotSet(
    const dataset::VerticalDataset& dataset, const model::AbstractModel& model,
    const std::vector<std::vector<int>>& attribute_idxs, int num_numerical_bins,
    float example_sampling);

absl::StatusOr<ConditionalExpectationPlotSet>
ComputeConditionalExpectationPlotSet(
    const dataset::VerticalDataset& dataset, const model::AbstractModel& model,
    const std::vector<std::vector<int>>& attribute_idxs, int num_numerical_bins,
    float example_sampling);

// Appends all the "num_dims"-dimensional combinations of input features.
absl::Status AppendAttributesCombinations(
    const model::AbstractModel& model, int num_dims,
    std::vector<std::vector<int>>* attribute_idxs);

// Appends all 2-dimensional combinations of attributes of type {type_1,
// type_2}.
absl::Status AppendAttributesCombinations2D(
    const model::AbstractModel& model, dataset::proto::ColumnType type_1,
    dataset::proto::ColumnType type_2,
    std::vector<std::vector<int>>* attribute_idxs);

// Creates a list of attribute sets from user set flags.
absl::StatusOr<std::vector<std::vector<int>>> GenerateAttributesCombinations(
    const model::AbstractModel& model, const bool flag_1d, const bool flag_2d,
    const bool flag_2d_categorical_numerical);

namespace internal {

// Given an example and an initialized pdp, returns the index of the bin that
// would contain this example. This function is essentially the inverse of
// "IndexToBinCenter".
absl::StatusOr<int> ExampleToBinIndex(
    const dataset::proto::Example& example,
    const dataset::proto::DataSpecification& data_spec,
    const proto::PartialDependencePlotSet::PartialDependencePlot& pdp);

// Returns a vector of Attribute protos storing the 'center value' of each bin
// corresponding to that attribute.
// If the attribute is categorical or boolean, then this returns a vector of
// Attribute protos containing each unique value this attribute can take.
// If the attribute is numerical, then this function divides the range of values
// this attribute takes (computed from the min and max values from the
// dataspec) into equal sized bins. The center value of each of these bins is
// stored in Attribute protos and returned.
struct BinsDefinition {
  std::vector<dataset::proto::Example::Attribute> centers;
  std::vector<float> numerical_boundaries;
  bool is_log = false;
};

absl::StatusOr<BinsDefinition> GetBinsForOneAttribute(
    const dataset::proto::DataSpecification& data_spec, int attribute_idx,
    int num_numerical_bins, const dataset::VerticalDataset& dataset);

// Computes a sorted list of unique values (and counts) in "values". NaN values
// in "values" are ignored.
std::vector<std::pair<float, int>> SortedUniqueCounts(
    std::vector<float> values);

}  // namespace internal

}  // namespace utils
}  // namespace yggdrasil_decision_forests
#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_PARTIAL_DEPENDENCE_PLOT_H_
