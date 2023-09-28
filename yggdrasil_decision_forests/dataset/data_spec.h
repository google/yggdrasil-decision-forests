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

// This library contains utility functions for the manipulation of dataspecs.

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_DATA_SPEC_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_DATA_SPEC_H_

#include <stdint.h>

#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/utils/hash.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// The "out of dictionary" (OOD) item is a special item used to represent
// infrequent or unknown categorical values.
constexpr char kOutOfDictionaryItemKey[] = "<OOD>";
constexpr int kOutOfDictionaryItemIndex = 0;

// Format used to represent discretized numerical values.
using DiscretizedNumericalIndex = uint16_t;
// Special value reserved to represent missing value.
constexpr auto kDiscretizedNumericalMissingValue =
    std::numeric_limits<DiscretizedNumericalIndex>::max();

// How to represent a Na values in a csv (apart from leaving the field empty).
constexpr char CSV_NA[] = "na";
constexpr char CSV_NA_V2[] = "nan";

// Build the mapping from col idx to a given vector of field names.
//
// If "required_columns" is not provided, all the columns are required.
// If "required_columns" is provided, only the columns in "required_columns" are
// required. Missing and non-required column receive the field index -1.
//
// For example, if "required_columns={1,2}" and if column "3" is not in the
// fields (i.e. column  "3" is missing), the column "3 will be considered to be
// filled with missing values. However, if column "1" is not in the fields, and
// error will be raised.
//
// Another example, if "required_columns={}", andy column can be missing. Such
// missing column will be filled with missing values.
absl::Status BuildColIdxToFeatureLabelIdx(
    const proto::DataSpecification& data_spec,
    const std::vector<std::string>& fields,
    const absl::optional<std::vector<int>>& required_columns,
    std::vector<int>* col_idx_to_field_idx);

// Returns a sorted list (in increasing order of column idx) of column idxs from
// a list of regular expressions on the column name.
void GetMultipleColumnIdxFromName(
    const std::vector<std::string>& column_name_regexs,
    const dataset::proto::DataSpecification& data_spec,
    std::vector<int32_t>* column_idxs);

// Returns a column idx from a regular expression on the column name. If none or
// several columns are matching the regular expression, the function fails with
// its error message starting with `error_message_prefix` (if specified).
absl::Status GetSingleColumnIdxFromName(
    absl::string_view column_name_regex,
    const dataset::proto::DataSpecification& data_spec, int32_t* column_idx,
    absl::string_view error_message_prefix = "");

// Converts a single row from a csv into an Example.
// If col_idx_to_field_idx[i] == -1, all the values of the i-th column are
// replaced by empty values.
absl::Status CsvRowToExample(const std::vector<std::string>& csv_fields,
                             const proto::DataSpecification& data_spec,
                             const std::vector<int>& col_idx_to_field_idx,
                             proto::Example* example);

// Converts a proto::Example into an array of string that can be saved in a csv
// file. The output "csv_fields[i]" is the string representation of the "i-th"
// column of "example".
absl::Status ExampleToCsvRow(const proto::Example& example,
                             const proto::DataSpecification& data_spec,
                             std::vector<std::string>* csv_fields);

// Returns the index of the column with the corresponding name. Raise an error
// if the column does not exist.
int GetColumnIdxFromName(absl::string_view name,
                         const proto::DataSpecification& data_spec);
absl::StatusOr<int> GetColumnIdxFromNameWithStatus(
    absl::string_view name, const proto::DataSpecification& data_spec);

// Returns the index of the column with the corresponding name. Returns  an
// error if the column does not exist.
absl::optional<int> GetOptionalColumnIdxFromName(
    absl::string_view name, const proto::DataSpecification& data_spec);

// Test if the dataspec contains this column.
bool HasColumn(absl::string_view name,
               const proto::DataSpecification& data_spec);

// Test if a particular attribute value is NA -- "Not Available", also
// referred as "missing".
inline bool IsNa(const proto::Example::Attribute& value) {
  return value.type_case() == proto::Example::Attribute::TypeCase::TYPE_NOT_SET;
}

// Returns a human readable representation of the dataspec. Easier to read and
// more informative than proto's DebugString() method.
std::string PrintHumanReadable(const proto::DataSpecification& data_spec,
                               bool sort_by_column_names = false);

// Returns the integer representation of a categorical value provided as a
// string.
//
// TODO: Remove this version when external protobuffer will support
// map query with absl::string_view.
absl::StatusOr<int32_t> CategoricalStringToValueWithStatus(
    const std::string& value, const proto::Column& col_spec);

int32_t CategoricalStringToValue(const std::string& value,
                                 const proto::Column& col_spec);

// Tokenize a string. "tokens" is cleared before being filled.
absl::Status Tokenize(const absl::string_view text,
                      const proto::Tokenizer& tokenizer,
                      std::vector<std::string>* tokens);

// Extract a ngrams of tokens from a list of token i.e. extracts all the
// sub-sequences of length "n" from "tokens". Append "separator" in between the
// items.
void ExtractNGrams(const std::vector<std::string>& tokens, const int n,
                   const absl::string_view separator,
                   std::vector<std::string>* grouped_tokens);

// Returns a string representation of a categorical value.
std::string CategoricalIdxToRepresentation(const proto::Column& col_spec,
                                           int32_t value_idx,
                                           bool add_quotes = false);

// Returns a string representation of a list of categorical values. If more than
// "max_values" are available, only print the first "max_values" and "..[xyz
// left]". Example: "b, c,...[1 left]". If "max_values==-1", all the elements
// are printed.
std::string CategoricalIdxsToRepresentation(
    const proto::Column& col_spec, const std::vector<int>& elements,
    int max_values, const absl::string_view separator = ", ");

// Add a column with specified name and specified type to a dataspec.
proto::Column* AddColumn(const absl::string_view name,
                         const proto::ColumnType type,
                         proto::DataSpecification* data_spec);

// Is this column type multidimensional?
bool IsMultiDimensional(proto::ColumnType type);

// Is this column categorical?
bool IsCategorical(proto::ColumnType type);

// Is this column numerical?
bool IsNumerical(proto::ColumnType type);

// Converts a discretized numerical value into a numerical value.
absl::StatusOr<float> DiscretizedNumericalToNumerical(
    const proto::Column& col_spec, DiscretizedNumericalIndex value);

// Determines a set of histogram-like boundaries of a discretized numerical
// features.
//
// For example, if "candidates" is a list of more-of-less uniform values between
// 0 and 50, and if "maximum_num_bins"=5, the results will be the following 4
// boundaries: 10, 20, 30, 40 representing the 5 bins: ]-inf, 10[, [10,20[, [20,
// 30[, [30, 40[, [40, +inf[.
//
// The function is mostly an heuristic that assigns bins proportionally to the
// density of values. See the arg for details on the other logics.
//
// Args:
//  candidates: Pairs of <value,count> sorted by value. Values should be unique.
//  maximum_num_bins: Maximum number of bins.
//  min_obs_in_bins: Minimum number of observations (sum of counts) in each
//    bins.
//  special_values: Special values (not necessarily present in "candidates")
//    that require their own bin. E.g. if 5 is in "special_values", and if
//    "candidates" contains values both greater and smaller than 5 (e.g. -3, 1,
//    6, 10) there will be a bin [5-eps, 5+eps[.
absl::StatusOr<std::vector<float>> GenDiscretizedBoundaries(
    const std::vector<std::pair<float, int>>& candidates, int maximum_num_bins,
    int min_obs_in_bins, const std::vector<float>& special_values);

// Converts a numerical value into a discretized numerical value.
DiscretizedNumericalIndex NumericalToDiscretizedNumerical(
    const proto::Column& col_spec, float value);

// Escape a feature name to be consumed as a "feature" in a training config.
std::string EscapeTrainingConfigFeatureName(absl::string_view feature_name);

// Hashing function for HASH columns.
inline uint64_t HashColumnString(absl::string_view value) {
  return utils::hash::HashStringViewToUint64(value);
}

inline uint64_t HashColumnInteger(int64_t value) {
  return utils::hash::HashInt64ToUint64(value);
}

// Name of an unrolled column.
std::string UnstackedColumnName(absl::string_view original_name, int dim_idx);

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_DATA_SPEC_H_
