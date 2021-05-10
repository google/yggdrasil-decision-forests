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

// This library contains utility functions for the creation of dataspecs.
//
// Usage example:
//
//   DataSpecification data_spec;
//   CreateDataSpec("recordio+tfe:/my/dataset@10", false, {}, &data_spec);
//
#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_DATA_SPEC_INFERENCE_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_DATA_SPEC_INFERENCE_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/registration.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// Computes a data spec given a dataset and a dataspec guide.
void CreateDataSpec(absl::string_view typed_path, bool use_flume,
                    const proto::DataSpecificationGuide& guide,
                    proto::DataSpecification* data_spec);

// Creates a dataspec from a dataset.
//
// The life of a AbstractDataSpecCreator is as follow:
// 1. "InferColumnsAndTypes" is called on a compatible dataset.
// 2. "ComputeColumnStatistics" is called on the
//
class AbstractDataSpecCreator {
 public:
  virtual ~AbstractDataSpecCreator() = default;

  // Lists the columns (with name and semantic) in the dataset.
  // If the column semantic is not defined in "guide", the most likely semantic
  // should be used.
  virtual void InferColumnsAndTypes(const std::vector<std::string>& paths,
                                    const proto::DataSpecificationGuide& guide,
                                    proto::DataSpecification* data_spec) = 0;

  // Accumulate statistics about each features.
  virtual void ComputeColumnStatistics(
      const std::vector<std::string>& paths,
      const proto::DataSpecificationGuide& guide,
      proto::DataSpecification* data_spec,
      proto::DataSpecificationAccumulator* accumulator) = 0;

  // Counts the number of examples.
  virtual utils::StatusOr<int64_t> CountExamples(absl::string_view path) = 0;
};

REGISTRATION_CREATE_POOL(AbstractDataSpecCreator);

#define REGISTER_AbstractDataSpecCreator(name, key) \
  REGISTRATION_REGISTER_CLASS(name, key, AbstractDataSpecCreator)

// Finalize the computation of data spec. This steps removes infrequent
// dictionary entries, update statistics of numerical columns, etc.
void FinalizeComputeSpec(const proto::DataSpecificationGuide& guide,
                         const proto::DataSpecificationAccumulator& accumulator,
                         proto::DataSpecification* data_spec);

// Finalize the inference of the type of each features. Should be called after
// "InferColumnsAndTypes" and before "ComputeColumnStatistics".
void FinalizeInferTypes(const proto::DataSpecificationGuide& guide,
                        proto::DataSpecification* data_spec);

// Add the specified (i.e. has_XXX is true) fields of the src column guide to
// dst one.
void MergeColumnGuide(const proto::ColumnGuide& src, proto::ColumnGuide* dst);

// Initialize the accumulator from a dataspec.
void InitializeDataspecAccumulator(
    const proto::DataSpecification& data_spec,
    proto::DataSpecificationAccumulator* accumulator);

// Update a column spec with a single numerical value. Should be used between
// "InitializeDataspecAccumulator" and "FinalizeComputeSpec".
absl::Status UpdateNumericalColumnSpec(
    const float num_value, proto::Column* col,
    proto::DataSpecificationAccumulator::Column* col_acc);

absl::Status UpdateCategoricalStringColumnSpec(
    const std::string& str_value, proto::Column* col,
    proto::DataSpecificationAccumulator::Column* col_acc);

absl::Status UpdateCategoricalIntColumnSpec(
    int int_value, proto::Column* col,
    proto::DataSpecificationAccumulator::Column* col_acc);

// Counts efficiently the number of examples in a dataset.
// This method is equivalent (but much more efficient) than reading and counting
// all the examples with "CreateExampleReader".
utils::StatusOr<int64_t> CountNumberOfExamples(absl::string_view typed_path);

// Generate the column guide of a given column (by merging the default column
// guide and the specific [matched using a regex on the column name]
// column guides (if any)). Returns true if a column name specific definition
// was found. Returns false otherwise.
bool BuildColumnGuide(absl::string_view col_name,
                      const proto::DataSpecificationGuide& guide,
                      proto::ColumnGuide* col_guide);

// Update the a column in a data spec (with the newly decided and definitive
// column type) with the appropriate guide information.
absl::Status UpdateSingleColSpecWithGuideInfo(
    const proto::ColumnGuide& col_guide, proto::Column* col);

// Initialize the column of a dataspec from a vector of column names (vector of
// strings). Also returns the guides of each newly created dataspec columns.
void InitializeDataSpecFromColumnNames(
    const proto::DataSpecificationGuide& guide,
    const std::vector<std::string>& header, proto::DataSpecification* data_spec,
    std::vector<std::pair<int, proto::ColumnGuide>>*
        spec_col_idx_2_csv_col_idx);

// Update the accumulator with a numerical feature value.
void FillContentNumericalFeature(
    float num_value, proto::DataSpecificationAccumulator::Column* col_acc);

// Update the accumulator with a discretized numerical feature value.
void UpdateComputeSpecDiscretizedNumerical(
    float value, proto::Column* column,
    proto::DataSpecificationAccumulator::Column* accumulator);

// Update the accumulator with a boolean feature value with a numerical
// observation.
void UpdateComputeSpecBooleanFeature(float value, proto::Column* column);

// Update all the columns in a data spec with the appropriate guide information.
// Used when inferring the column type from a csv file.
absl::Status UpdateColSpecsWithGuideInfo(
    const std::vector<std::pair<int, proto::ColumnGuide>>&
        spec_col_idx_2_csv_col_idx,
    proto::DataSpecification* data_spec);

// Add the tokens to the dictionary of categorical column spec that is
// currently being assembled. Don't check the size of the dictionary (i.e. the
// dictionary can grow larger than "max_number_of_unique_values").
void AddTokensToCategoricalColumnSpec(const std::vector<std::string>& tokens,
                                      proto::Column* col);

// Does this value looks like to be a multi dimensional value?
bool LooksMultiDimensional(absl::string_view value,
                           const proto::Tokenizer& tokenizer);

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_DATA_SPEC_INFERENCE_H_
