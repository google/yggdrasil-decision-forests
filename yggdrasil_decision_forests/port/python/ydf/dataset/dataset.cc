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

#include "ydf/dataset/dataset.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <string.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "pybind11_protobuf/native_proto_caster.h"  // IWYU pragma : keep
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/formats.h"
#include "yggdrasil_decision_forests/dataset/formats.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "ydf/utils/numpy_data.h"
#include "ydf/utils/status_casters.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

#ifdef _WIN32
#include <cstddef>
typedef std::ptrdiff_t ssize_t;
#endif

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {
namespace {

using NumericalColumn =
    ::yggdrasil_decision_forests::dataset::VerticalDataset::NumericalColumn;
using BooleanColumn =
    ::yggdrasil_decision_forests::dataset::VerticalDataset::BooleanColumn;
using CategoricalColumn =
    ::yggdrasil_decision_forests::dataset::VerticalDataset::CategoricalColumn;
using CategoricalSetColumn = ::yggdrasil_decision_forests::dataset::
    VerticalDataset::CategoricalSetColumn;
using HashColumn =
    ::yggdrasil_decision_forests::dataset::VerticalDataset::HashColumn;

// Checks if all columns of the dataset have the same number of rows and sets
// the dataset's number of rows accordingly. Columns with 0 rows are resized
// (i.e. filled with N/A values. If requested, also sets the
// created_num_rows parameter in the dataspec.
absl::Status SetAndCheckNumRowsAndFillMissing(dataset::VerticalDataset& self,
                                              const bool set_data_spec) {
  if (self.ncol() == 0) {
    // Nothing to do here.
    return absl::OkStatus();
  }
  dataset::VerticalDataset::row_t num_rows = 0;
  for (int col_idx = 0; col_idx < self.ncol(); col_idx++) {
    const auto col_nrows = self.column(col_idx)->nrows();
    if (col_nrows == 0) {
      continue;
    }
    if (num_rows == 0) {
      num_rows = col_nrows;
    } else if (num_rows != col_nrows) {
      return absl::InvalidArgumentError(
          "Inconsistent number of rows between the columns.");
    }
  }
  // Resize fills empty rows with N/A values.
  self.Resize(num_rows);
  self.set_nrow(num_rows);
  if (set_data_spec) {
    self.mutable_data_spec()->set_created_num_rows(num_rows);
  }
  return absl::OkStatus();
}

// Creates a column spec for a numerical column.
absl::StatusOr<dataset::proto::Column> CreateNumericalColumnSpec(
    const absl::string_view name, const StridedSpanFloat32 values) {
  size_t num_valid_values = 0;
  double sum_values = 0;
  double sum_square_values = 0;
  double min_value = 0;
  double max_value = 0;
  bool first_value = true;

  for (size_t value_idx = 0; value_idx < values.size(); value_idx++) {
    const float value = values[value_idx];
    if (std::isnan(value)) {
      continue;
    }
    if (std::isinf(value)) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Found infinite value for numerical feature $0", name));
    }

    sum_values += value;
    sum_square_values += value * value;

    if (first_value) {
      min_value = value;
      max_value = value;
      first_value = false;
    } else {
      if (value < min_value) {
        min_value = value;
      }
      if (value > max_value) {
        max_value = value;
      }
    }

    num_valid_values++;
  }

  dataset::proto::Column column;
  column.set_name(name);
  column.set_type(dataset::proto::ColumnType::NUMERICAL);
  column.set_count_nas(values.size() - num_valid_values);

  auto* colum_num = column.mutable_numerical();
  if (num_valid_values > 0) {
    const double mean = sum_values / num_valid_values;
    const double var = sum_square_values / num_valid_values - mean * mean;

    colum_num->set_min_value(min_value);
    colum_num->set_max_value(max_value);
    colum_num->set_mean(mean);
    colum_num->set_standard_deviation(std::sqrt(var));
  }
  return column;
}

// Append contents of `data` to a numerical column. If no `column_idx` is given,
// a new column is created.
//
// Note that this function only creates the columns and copies the data, it does
// not set `num_rows` on the dataset. Before using the dataset, `num_rows` has
// to be set (e.g. using SetAndCheckNumRows).
absl::Status PopulateColumnNumericalNPFloat32(
    dataset::VerticalDataset& self, const std::string& name,
    py::array_t<float>& data, std::optional<dataset::proto::DType> ydf_dtype,
    std::optional<int> column_idx) {
  StridedSpanFloat32 src_values(data);

  NumericalColumn* column;
  if (!column_idx.has_value()) {
    // Create column spec
    ASSIGN_OR_RETURN(auto column_spec,
                     CreateNumericalColumnSpec(name, src_values));

    if (ydf_dtype.has_value()) {
      column_spec.set_dtype(*ydf_dtype);
    }

    ASSIGN_OR_RETURN(auto* abstract_column, self.AddColumn(column_spec));
    // Import column data
    ASSIGN_OR_RETURN(column,
                     abstract_column->MutableCastWithStatus<NumericalColumn>());
  } else {
    // Note that when populating an existing vertical dataset column, we don't
    // compute / update the dataspec.
    ASSIGN_OR_RETURN(column,
                     self.MutableColumnWithCastWithStatus<NumericalColumn>(
                         column_idx.value()));
  }

  std::vector<float>& dst_values = *column->mutable_values();
  const size_t offset = dst_values.size();
  dst_values.resize(offset + src_values.size());
  for (size_t i = 0; i < src_values.size(); i++) {
    dst_values[i + offset] = src_values[i];
  }

  return absl::OkStatus();
}

// Creates a column spec for a boolean column.
absl::StatusOr<dataset::proto::Column> CreateBooleanColumnSpec(
    const std::string& name, const StridedSpan<bool> values) {
  // Note: A span of bool cannot represent missing values.
  const size_t num_valid_values = values.size();
  size_t count_true = 0;
  size_t count_false = 0;

  for (size_t value_idx = 0; value_idx < values.size(); value_idx++) {
    if (values[value_idx]) {
      count_true++;
    } else {
      count_false++;
    }
  }

  dataset::proto::Column column;
  column.set_name(name);
  column.set_type(dataset::proto::ColumnType::BOOLEAN);
  column.set_count_nas(values.size() - num_valid_values);

  auto* colum_bool = column.mutable_boolean();
  colum_bool->set_count_true(count_true);
  colum_bool->set_count_false(count_false);
  return column;
}

// Append contents of `data` to a boolean column. If no `column_idx` is given, a
// new column is created.
//
// Note that this function only creates the columns and copies the data, it does
// not set `num_rows` on the dataset. Before using the dataset, `num_rows` has
// to be set (e.g. using SetAndCheckNumRows).
absl::Status PopulateColumnBooleanNPBool(
    dataset::VerticalDataset& self, const std::string& name,
    py::array_t<bool>& data, std::optional<dataset::proto::DType> ydf_dtype,
    std::optional<int> column_idx) {
  StridedSpan<bool> src_values(data);

  BooleanColumn* column;
  if (!column_idx.has_value()) {
    // Create column spec
    ASSIGN_OR_RETURN(auto column_spec,
                     CreateBooleanColumnSpec(name, src_values));
    if (ydf_dtype.has_value()) {
      column_spec.set_dtype(*ydf_dtype);
    }
    ASSIGN_OR_RETURN(auto* abstract_column, self.AddColumn(column_spec));
    // Import column data
    ASSIGN_OR_RETURN(column,
                     abstract_column->MutableCastWithStatus<BooleanColumn>());
  } else {
    ASSIGN_OR_RETURN(column,
                     self.MutableColumnWithCastWithStatus<BooleanColumn>(
                         column_idx.value()));
  }

  std::vector<BooleanColumn::Format>& dst_values = *column->mutable_values();
  const size_t offset = dst_values.size();
  dst_values.resize(offset + src_values.size());
  for (size_t i = 0; i < src_values.size(); i++) {
    dst_values[i + offset] = src_values[i];
  }

  return absl::OkStatus();
}

// A dictionary.
// This structure holds the dictionary before it gets copied to the dataspec.
struct Dictionary {
  struct Item {
    size_t count = 0;
    std::string key;
  };

  std::vector<Item> items;
  size_t num_valid_values = 0;
};

// Computes a dictionary from a set of observed values.
Dictionary DictionaryFromValues(const std::vector<std::string_view>& values,
                                const int max_vocab_count,
                                const int min_vocab_frequency) {
  // Count unique values.
  // Warning: "dict_map" does not own its keys.
  absl::flat_hash_map<absl::string_view, size_t> dict_map;
  for (size_t value_idx = 0; value_idx < values.size(); value_idx++) {
    const std::string_view value = values[value_idx];
    if (value.empty()) {
      continue;  // Value is missing
    }
    dict_map[value]++;
  }

  Dictionary dictionary;
  dictionary.items.reserve(dict_map.size());
  size_t num_oov_values = 0;
  for (const auto& src_value : dict_map) {
    dictionary.num_valid_values += src_value.second;
    if (src_value.second < min_vocab_frequency) {
      num_oov_values += src_value.second;
      continue;
    }
    dictionary.items.push_back(
        {.count = src_value.second, .key = std::string(src_value.first)});
  }
  std::sort(dictionary.items.begin(), dictionary.items.end(),
            [](const Dictionary::Item& a, const Dictionary::Item& b) {
              // Sort by count
              if (a.count != b.count) {
                return a.count > b.count;  // Large counts first
              }
              // Sort by key
              return a.key < b.key;  // First keys first
            });

  // Apply maximum vocab constraint.
  if (max_vocab_count >= 0 && dictionary.items.size() > max_vocab_count) {
    for (int item_idx = max_vocab_count; item_idx < dictionary.items.size();
         item_idx++) {
      num_oov_values += dictionary.items[item_idx].count;
    }
    dictionary.items.resize(max_vocab_count);
  }

  // Create the OOV item
  dictionary.items.insert(dictionary.items.begin(),
                          Dictionary::Item{
                              .count = num_oov_values,
                              .key = dataset::kOutOfDictionaryItemKey,
                          });

  return dictionary;
}

// Computes a dictionary from a user provided dictionary.
//
// The index of the items in the output dictionary match the index of the items
// in the input dictionary.
//
// Args:
//   values: Value of each example in the dataset.
//   force_dictionary: The dictionary wanted by the user.
//
// Returns:
//   A dictionary.
//
absl::StatusOr<Dictionary> DictionaryFromUserDictionary(
    const std::vector<std::string_view>& values,
    const py::array& force_dictionary) {
  ASSIGN_OR_RETURN(const auto wrapper, NPByteArray::Create(force_dictionary));
  const auto src_items = wrapper.ToVectorNotOwned();

  // Count the number of occurrences for the items in the user dictionary.
  // Warning: "count_map" does not own its keys.
  absl::flat_hash_map<absl::string_view, size_t> count_map;
  for (const auto& item : src_items) {
    count_map[item] = 0;
  }
  size_t num_oov_value = 0;
  for (size_t value_idx = 0; value_idx < values.size(); value_idx++) {
    const std::string_view value = values[value_idx];
    if (value.empty()) {
      continue;  // Value is missing
    }
    const auto it = count_map.find(value);
    if (it == count_map.end()) {
      num_oov_value++;
    } else {
      it->second++;
    }
  }

  Dictionary dictionary;
  dictionary.num_valid_values = num_oov_value;
  for (const auto& src_value : count_map) {
    dictionary.num_valid_values += src_value.second;
  }

  dictionary.items.reserve(src_items.size());
  for (const auto& item : src_items) {
    dictionary.items.push_back(
        {.count = count_map[item], .key = std::string(item)});
  }
  return dictionary;
}

// Creates a column spec for a categorical column.
absl::StatusOr<dataset::proto::Column> CreateCategoricalColumnSpec(
    const std::string& name, const std::vector<std::string_view>& values,
    const int max_vocab_count, const int min_vocab_frequency,
    const dataset::proto::ColumnType& column_type,
    const std::optional<py::array>& force_dictionary) {
  if (max_vocab_count < -1) {
    return absl::InvalidArgumentError(
        absl::Substitute("Column $0 received invalid dataspec inference "
                         "argument max_vocab_count: $1",
                         name, max_vocab_count));
  }
  if (max_vocab_count == -1) {
    LOG(INFO) << "max_vocab_count = -1 for column " << name
              << ", the dictionary will not be pruned by size.";
  }
  if (max_vocab_count == 0) {
    LOG(WARNING) << "max_vocab_count = 0 for column " << name
                 << ", the dictionary will only contain OOD values.";
  }
  if (min_vocab_frequency < 0) {
    return absl::InvalidArgumentError(
        absl::Substitute("Column $0 received invalid dataspec inference "
                         "argument min_vocab_frequency: $1",
                         name, min_vocab_frequency));
  }

  Dictionary dictionary;
  if (force_dictionary.has_value()) {
    ASSIGN_OR_RETURN(dictionary,
                     DictionaryFromUserDictionary(values, *force_dictionary));
  } else {
    dictionary =
        DictionaryFromValues(values, max_vocab_count, min_vocab_frequency);
  }

  // Create column spec
  dataset::proto::Column column;
  column.set_type(column_type);
  column.set_name(name);
  column.set_count_nas(values.size() - dictionary.num_valid_values);

  // Copy dictionary in proto
  auto* column_cat = column.mutable_categorical();
  auto& dst_items = *column_cat->mutable_items();
  for (size_t item_idx = 0; item_idx < dictionary.items.size(); item_idx++) {
    const auto& item = dictionary.items[item_idx];
    auto& dst_item = dst_items[item.key];
    dst_item.set_count(item.count);
    // Index counting starts with the OOD item.
    dst_item.set_index(item_idx);
  }
  column_cat->set_number_of_unique_values(dictionary.items.size());

  return column;
}

// Append contents of `data` to a categorical column. If no `column_idx` is
// given, a new column is created.
//
// Note that this function only creates the columns and copies the data, it does
// not set `num_rows` on the dataset. Before using the dataset, `num_rows` has
// to be set (e.g. using SetAndCheckNumRows).
absl::Status PopulateColumnCategoricalNPBytes(
    dataset::VerticalDataset& self, const std::string& name, py::array& data,
    std::optional<dataset::proto::DType> ydf_dtype, const int max_vocab_count,
    const int min_vocab_frequency, std::optional<int> column_idx,
    const std::optional<py::array> dictionary) {
  ASSIGN_OR_RETURN(const auto values, NPByteArray::Create(data));
  const auto values_vector = values.ToVectorNotOwned();

  CategoricalColumn* column;
  ssize_t offset = 0;
  if (!column_idx.has_value()) {
    // Create column spec
    ASSIGN_OR_RETURN(
        auto column_spec,
        CreateCategoricalColumnSpec(name, values_vector, max_vocab_count,
                                    min_vocab_frequency,
                                    dataset::proto::CATEGORICAL, dictionary));

    if (ydf_dtype.has_value()) {
      column_spec.set_dtype(*ydf_dtype);
    }

    // Import column data
    ASSIGN_OR_RETURN(auto* abstract_column, self.AddColumn(column_spec));
    ASSIGN_OR_RETURN(
        column, abstract_column->MutableCastWithStatus<CategoricalColumn>());
    column_idx = self.ncol() - 1;
  } else {
    DCHECK_EQ(min_vocab_frequency, -1)
        << "`min_vocab_frequency` is ignored as column " << name << " exists";
    DCHECK_EQ(max_vocab_count, -1)
        << "`max_vocab_count` is ignored as column " << name << " exists";
    ASSIGN_OR_RETURN(column,
                     self.MutableColumnWithCastWithStatus<CategoricalColumn>(
                         column_idx.value()));
    offset = column->values().size();
  }
  const auto& column_spec = self.data_spec().columns(column_idx.value());
  column->Resize(offset + values.size());
  auto& dst_values = *column->mutable_values();

  // TODO: Check if using an absl::flat_map is significantly faster.
  const auto& items = column_spec.categorical().items();

  if (items.empty()) {
    return absl::InvalidArgumentError(
        absl::Substitute("Column \"$0\": Empty categorical dictionary. PYDF "
                         "does not support empty dictionaries",
                         name));
  }

  for (size_t value_idx = 0; value_idx < values_vector.size(); value_idx++) {
    const std::string_view value = values_vector[value_idx];
    int32_t dst_value;
    if (value.empty()) {
      dst_value = dataset::VerticalDataset::CategoricalColumn::kNaValue;
    } else {
      const auto it = items.find(value);
      if (it == items.end()) {
        dst_value = dataset::kOutOfDictionaryItemIndex;
      } else {
        dst_value = it->second.index();
      }
    }
    dst_values[offset + value_idx] = dst_value;
  }

  return absl::OkStatus();
}

// Append contents of `data` to a categorical set column. If no `column_idx` is
// given, a new column is created.
//
// Note that this function only creates the columns and copies the data, it does
// not set `num_rows` on the dataset. Before using the dataset, `num_rows` has
// to be set (e.g. using SetAndCheckNumRows).
absl::Status PopulateColumnCategoricalSetNPBytes(
    dataset::VerticalDataset& self, const std::string& name, py::array& data,
    std::optional<dataset::proto::DType> ydf_dtype, const int max_vocab_count,
    const int min_vocab_frequency, std::optional<int> column_idx,
    const std::optional<py::array> dictionary) {
  // `bank` contains all the tokens. `boundaries` contains the boundaries of the
  // individual sets.
  std::vector<std::string_view> bank;
  std::vector<dataset::UnsignedExampleIdx> boundaries;
  std::vector<std::string_view> sorted_row_values;

  size_t num_rows = data.shape(0);
  boundaries.reserve(num_rows + 1);
  boundaries.push_back(0);
  for (const auto& row_object : data) {
    const auto& row_data = py::cast<py::array>(row_object);
    ASSIGN_OR_RETURN(const auto row_values, NPByteArray::Create(row_data));

    sorted_row_values.clear();
    for (size_t i = 0; i < row_values.size(); i++) {
      sorted_row_values.push_back(row_values[i]);
    }
    std::sort(sorted_row_values.begin(), sorted_row_values.end());

    std::unique_copy(sorted_row_values.begin(), sorted_row_values.end(),
                     std::back_inserter(bank));

    boundaries.push_back(bank.size());
  }

  CategoricalSetColumn* column;
  if (!column_idx.has_value()) {
    // Create column spec
    ASSIGN_OR_RETURN(auto column_spec,
                     CreateCategoricalColumnSpec(
                         name, bank, max_vocab_count, min_vocab_frequency,
                         dataset::proto::CATEGORICAL_SET, dictionary));

    if (ydf_dtype.has_value()) {
      column_spec.set_dtype(*ydf_dtype);
    }

    // Import column data
    ASSIGN_OR_RETURN(auto* abstract_column, self.AddColumn(column_spec));
    ASSIGN_OR_RETURN(
        column, abstract_column->MutableCastWithStatus<CategoricalSetColumn>());
    column_idx = self.ncol() - 1;
  } else {
    DCHECK_EQ(min_vocab_frequency, -1)
        << "`min_vocab_frequency` is ignored as column " << name << " exists";
    DCHECK_EQ(max_vocab_count, -1)
        << "`max_vocab_count` is ignored as column " << name << " exists";
    ASSIGN_OR_RETURN(column,
                     self.MutableColumnWithCastWithStatus<CategoricalSetColumn>(
                         column_idx.value()));
  }
  const auto& column_spec = self.data_spec().columns(column_idx.value());

  // TODO: Check if using an absl::flat_map is significantly faster.
  const auto& items = column_spec.categorical().items();

  if (items.empty()) {
    return absl::InvalidArgumentError(
        "Empty categorical dictionary. PYDF does not support empty "
        "dictionaries");
  }

  std::vector<int32_t> dst_values;
  for (size_t row_idx = 0; row_idx < num_rows; row_idx++) {
    const auto left_boundary = boundaries[row_idx];
    const auto right_boundary = boundaries[row_idx + 1];

    dst_values.clear();
    for (size_t value_idx = left_boundary; value_idx < right_boundary;
         value_idx++) {
      const std::string_view value = bank[value_idx];
      const auto it = items.find(value);
      if (it == items.end()) {
        dst_values.push_back(dataset::kOutOfDictionaryItemIndex);
      } else {
        dst_values.push_back(it->second.index());
      }
    }
    std::sort(dst_values.begin(), dst_values.end());
    column->AddVector(dst_values);
  }

  return absl::OkStatus();
}

// Records, in the dataspec, which column comes from an unrolled
// multi-dimensional feature. This function can only be called once.
absl::Status SetMultiDimDataspec(
    dataset::VerticalDataset& self,
    const std::unordered_map<std::string, std::vector<std::string>>&
        unrolling) {
  auto& dataspec = *self.mutable_data_spec();

  if (dataspec.unstackeds_size() != 0) {
    return absl::InvalidArgumentError(
        "Multi-dimensional information already set");
  }

  // Index column indices
  absl::flat_hash_map<std::string, int> column_name_to_idx;
  {
    int column_idx = 0;
    for (const auto& column : dataspec.columns()) {
      column_name_to_idx[column.name()] = column_idx;
      column_idx++;
    }
    DCHECK_EQ(column_idx, dataspec.columns_size());
  }

  for (const auto& [original_name, unrolled_names] : unrolling) {
    // Column index of the first dimension
    int begin_column_idx = -1;

    for (int dim_idx = 0; dim_idx < unrolled_names.size(); dim_idx++) {
      const auto& unrolled_name = unrolled_names[dim_idx];

      // Search for the column idx of this dimension
      const auto it_column_idx = column_name_to_idx.find(unrolled_name);
      if (it_column_idx == column_name_to_idx.end()) {
        return absl::InvalidArgumentError(
            absl::Substitute("Column \"$0\" not found ", unrolled_name));
      }

      // Check that columns are contiguous
      if (dim_idx == 0) {
        begin_column_idx = it_column_idx->second;
      }
      if (it_column_idx->second != begin_column_idx + dim_idx) {
        return absl::InvalidArgumentError("Non contiguous column");
      }

      // Tag the column
      dataspec.mutable_columns(it_column_idx->second)->set_is_unstacked(true);
    }

    if (begin_column_idx == -1) {
      // Note: This should never happen.
      return absl::InvalidArgumentError(
          "Empty unrolled columns are not allowed");
    }

    auto* unstacked = dataspec.add_unstackeds();
    unstacked->set_original_name(original_name);
    unstacked->set_size(unrolled_names.size());
    unstacked->set_begin_column_idx(begin_column_idx);
    unstacked->set_type(dataspec.columns(begin_column_idx).type());
  }

  return absl::OkStatus();
}

// Append contents of `data` to a HASH column. If no `column_idx` is given, a
// new column is created.
//
// Note that this function only creates the columns and copies the data, it does
// not set `num_rows` on the dataset. Before using the dataset, `num_rows` has
// to be set (e.g. using SetAndCheckNumRows).
absl::Status PopulateColumnHashNPBytes(
    dataset::VerticalDataset& self, const std::string& name, py::array& data,
    std::optional<dataset::proto::DType> ydf_dtype,
    std::optional<int> column_idx) {
  ASSIGN_OR_RETURN(const auto values, NPByteArray::Create(data));

  HashColumn* column;
  size_t offset = 0;
  if (!column_idx.has_value()) {
    // Create column spec
    dataset::proto::Column column_spec;
    column_spec.set_name(name);
    column_spec.set_type(dataset::proto::ColumnType::HASH);

    if (ydf_dtype.has_value()) {
      column_spec.set_dtype(*ydf_dtype);
    }

    // Import column data
    ASSIGN_OR_RETURN(auto* abstract_column, self.AddColumn(column_spec));
    ASSIGN_OR_RETURN(column,
                     abstract_column->MutableCastWithStatus<HashColumn>());
    column_idx = self.ncol() - 1;
  } else {
    ASSIGN_OR_RETURN(column, self.MutableColumnWithCastWithStatus<HashColumn>(
                                 column_idx.value()));
    offset = column->values().size();
  }
  column->Resize(offset + values.size());
  auto& dst_values = *column->mutable_values();

  for (size_t value_idx = 0; value_idx < values.size(); value_idx++) {
    const std::string_view value = values[value_idx];
    uint64_t dst_value;
    if (value.empty()) {
      dst_value = dataset::VerticalDataset::HashColumn::kNaValue;
    } else {
      dst_value = dataset::HashColumnString(value);
    }
    dst_values[offset + value_idx] = dst_value;
  }

  return absl::OkStatus();
}

absl::Status CreateColumnsFromDataSpec(
    dataset::VerticalDataset& self,
    const dataset::proto::DataSpecification& data_spec) {
  self.set_data_spec(data_spec);
  return self.CreateColumnsFromDataspec();
}

// Returns the raw contents of the dataset. To be used for testing/debugging
// only.
std::string DebugString(const dataset::VerticalDataset& self) {
  return self.DebugString(/*max_displayed_rows=*/{}, /*vertical=*/true,
                          /*digit_precision=*/6);
}

// Print warning messages when reading from path.
absl::Status CheckDataSpecForPathReading(
    absl::string_view typed_path,
    const dataset::proto::DataSpecification& data_spec) {
  ASSIGN_OR_RETURN(auto path_and_type,
                   dataset::GetDatasetPathAndTypeOrStatus(typed_path));
  if (path_and_type.second == dataset::proto::FORMAT_CSV) {
    bool tokenizer_warning_shown = false;
    for (const auto& column : data_spec.columns()) {
      if (!tokenizer_warning_shown &&
          column.type() == dataset::proto::CATEGORICAL_SET) {
        if (column.tokenizer().splitter() !=
            dataset::proto::Tokenizer::NO_SPLITTING) {
          LOG(INFO)
              << "Column " << column.name()
              << " has type CATEGORICAL_SET and will be tokenized. Set the "
                 "column type to CATEGORICAL or deactivate the tokenizer in "
                 "the data spec to prevent this.";
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status CreateFromPathWithDataSpec(
    dataset::VerticalDataset& self, const std::string& path,
    const dataset::proto::DataSpecification& data_spec,
    const std::optional<std::vector<std::string>>& required_columns =
        std::nullopt) {
  dataset::LoadConfig dataset_loading_config;

  // If `required_columns` is not given, all columns are required.
  std::optional<std::vector<int>> required_column_idxs;
  if (required_columns.has_value()) {
    required_column_idxs = std::vector<int>();
    required_column_idxs->reserve(required_columns->size());
    for (const auto& column_name : required_columns.value()) {
      ASSIGN_OR_RETURN(
          const auto col_idx,
          dataset::GetColumnIdxFromNameWithStatus(column_name, data_spec));
      required_column_idxs->push_back(col_idx);
    }
  }
  ASSIGN_OR_RETURN(const auto typed_path, dataset::GetTypedPath(path));

  RETURN_IF_ERROR(CheckDataSpecForPathReading(typed_path, data_spec));

  RETURN_IF_ERROR(dataset::LoadVerticalDataset(typed_path, data_spec, &self,
                                               required_column_idxs,
                                               dataset_loading_config));
  return absl::OkStatus();
}

absl::Status CreateFromPathWithDataSpecGuide(
    dataset::VerticalDataset& self, std::string path,
    const dataset::proto::DataSpecificationGuide& data_spec_guide,
    const std::optional<std::vector<std::string>>& required_columns =
        std::nullopt) {
  dataset::LoadConfig dataset_loading_config;

  ASSIGN_OR_RETURN(const std::string typed_path, dataset::GetTypedPath(path));
  dataset::proto::DataSpecification data_spec;
  RETURN_IF_ERROR(dataset::CreateDataSpecWithStatus(
      typed_path, false, data_spec_guide, &data_spec));

  // If no required columns are given, check if all the specified columns are
  // also specified in the dataspec.
  int unused_column_idx;
  if (!required_columns.has_value()) {
    for (const auto& column : data_spec_guide.column_guides()) {
      RETURN_IF_ERROR(dataset::GetSingleColumnIdxFromName(
          column.column_name_pattern(), data_spec, &unused_column_idx));
    }
  }

  return CreateFromPathWithDataSpec(self, path, data_spec, required_columns);
}
}  // namespace

void init_dataset(py::module_& m) {
  // TODO: 310919367 - Pass strings with string_view if/when possible.
  py::class_<dataset::VerticalDataset>(m, "VerticalDataset")
      .def(py::init<>())
      .def("data_spec", &dataset::VerticalDataset::data_spec)
      .def("nrow", &dataset::VerticalDataset::nrow)
      .def("MemoryUsage", &dataset::VerticalDataset::MemoryUsage)
      .def("__repr__",
           [](const dataset::VerticalDataset& a) {
             return absl::Substitute(
                 "<dataset_cc.CCVerticalDataset occupying $0 bytes>. "
                 "Dataspec:\n$1",
                 a.MemoryUsage(), dataset::PrintHumanReadable(a.data_spec()));
           })
      .def("DebugString", &DebugString,
           "Converts a dataset's contents to a CSV-like string. To be used for "
           "debugging / testing only.")
      .def("CreateFromPathWithDataSpec", WithStatus(CreateFromPathWithDataSpec),
           py::arg("path"), py::arg("data_spec"),
           py::arg("required_columns") = std::nullopt,
           "Creates a dataset from a path, supports sharding. If the path is "
           "typed, use the type, the given type is used. Otherwise, YDF will "
           "try to determine the path and fail if this is not possible.")
      .def("CreateFromPathWithDataSpecGuide",
           WithStatus(CreateFromPathWithDataSpecGuide), py::arg("path"),
           py::arg("data_spec_guide"),
           py::arg("required_columns") = std::nullopt,
           "Creates a dataset from a path, supports sharding. If the path is "
           "typed, use the type, the given type is used. Otherwise, YDF will "
           "try to determine the path and fail if this is not possible.")
      .def("CreateColumnsFromDataSpec", WithStatus(CreateColumnsFromDataSpec),
           py::arg("data_spec"))
      .def("SetAndCheckNumRowsAndFillMissing",
           WithStatus(SetAndCheckNumRowsAndFillMissing),
           py::arg("set_data_spec"))
      // Data setters
      .def("PopulateColumnCategoricalNPBytes",
           WithStatus(PopulateColumnCategoricalNPBytes), py::arg("name"),
           py::arg("data").noconvert(), py::arg("ydf_dtype"),
           py::arg("max_vocab_count") = -1, py::arg("min_vocab_frequency") = -1,
           py::arg("column_idx") = std::nullopt,
           py::arg("dictionary") = std::nullopt)
      .def("PopulateColumnNumericalNPFloat32",
           WithStatus(PopulateColumnNumericalNPFloat32), py::arg("name"),
           py::arg("data").noconvert(), py::arg("ydf_dtype"),
           py::arg("column_idx") = std::nullopt)
      .def("PopulateColumnBooleanNPBool",
           WithStatus(PopulateColumnBooleanNPBool), py::arg("name"),
           py::arg("data").noconvert(), py::arg("ydf_dtype"),
           py::arg("column_idx") = std::nullopt)
      .def("PopulateColumnHashNPBytes", WithStatus(PopulateColumnHashNPBytes),
           py::arg("name"), py::arg("data").noconvert(), py::arg("ydf_dtype"),
           py::arg("column_idx") = std::nullopt)
      .def("PopulateColumnCategoricalSetNPBytes",
           WithStatus(PopulateColumnCategoricalSetNPBytes), py::arg("name"),
           py::arg("data").noconvert(), py::arg("ydf_dtype"),
           py::arg("max_vocab_count") = -1, py::arg("min_vocab_frequency") = -1,
           py::arg("column_idx") = std::nullopt,
           py::arg("dictionary") = std::nullopt)
      .def("SetMultiDimDataspec", WithStatus(SetMultiDimDataspec),
           py::arg("unrolling"),
           "Records, in the dataspec, which columns have been unrolled from "
           "multi-dimensional features.");
}

}  // namespace yggdrasil_decision_forests::port::python
