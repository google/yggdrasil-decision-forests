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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "pybind11_protobuf/native_proto_caster.h"  // IWYU pargma : keep
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/formats.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
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
using HashColumn =
    ::yggdrasil_decision_forests::dataset::VerticalDataset::HashColumn;

// A collection of values, somehow similar to a "Span<const float>", but with a
// stride (i.e., items are evenly spaced, but possibly with a constant gap).
// Only support what is needed in this file.
//
// "StridedSpanFloat32" does not own the underlying data and relies on the data
// in "data". The initialization "py::array_t data" object should not be
// destroyed before "StridedSpanFloat32".
class StridedSpanFloat32 {
 public:
  // Build a "StridedSpanFloat32".
  //
  // Args:
  //   data: A one dimentionnal array of float32 values.
  StridedSpanFloat32(py::array_t<float>& data)
      : item_stride_(data.strides(0) / data.itemsize()),
        size_(static_cast<size_t>(data.shape(0))),
        values_(data.data()) {
    DCHECK_EQ(data.strides(0) % data.itemsize(), 0);
  }

  // Number of values.
  size_t size() const { return size_; }

  // Accesses to a value. "index" should be in [0, size).
  float operator[](const size_t index) const {
    DCHECK_LT(index, size_);
    return values_[index * item_stride_];
  }

 private:
  const size_t item_stride_;
  const size_t size_;
  const float* const values_;
};

// Checks if all columns of the dataset have the same number of rows and sets
// the dataset's number of rows accordingly. If requested, also modifies the
// data spec.
absl::Status SetAndCheckNumRows(dataset::VerticalDataset& self,
                                const bool set_data_spec) {
  if (self.ncol() == 0) {
    // Nothing to do here.
    return absl::OkStatus();
  }
  const auto num_rows = self.column(0)->nrows();
  for (int i = 1; i < self.ncol(); i++) {
    if (num_rows != self.column(i)->nrows()) {
      return absl::InvalidArgumentError(
          "Inconsitent number of rows between the columns.");
    }
  }
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

// Append contents of `data` to a numerical column. If no `column_idx` is not
// given, a new column is created.
//
// Note that this function only creates the columns and copies the data, but it
// does not set `num_rows` on the dataset. Before using the dataset, `num_rows
// has to be set (e.g. using SetAndCheckNumRows).
absl::Status PopulateColumnNumericalNPFloat32(dataset::VerticalDataset& self,
                                              const std::string& name,
                                              py::array_t<float>& data,
                                              std::optional<int> column_idx) {
  StridedSpanFloat32 src_values(data);

  const auto get_mutable_column =
      [column_idx, &self, &name,
       &src_values]() -> absl::StatusOr<NumericalColumn* const> {
    if (!column_idx.has_value()) {
      // Create column spec
      ASSIGN_OR_RETURN(const auto column_spec,
                       CreateNumericalColumnSpec(name, src_values));
      ASSIGN_OR_RETURN(auto* abstract_column, self.AddColumn(column_spec));
      // Import column data
      return abstract_column->MutableCastWithStatus<NumericalColumn>();
    } else {
      // Note that when populating an existing vertical dataset column, we don't
      // compute / update the dataspec.
      return self.MutableColumnWithCastWithStatus<NumericalColumn>(
          column_idx.value());
    }
  };
  ASSIGN_OR_RETURN(NumericalColumn* const column, get_mutable_column());
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
    const std::string& name, absl::Span<const bool> values) {
  // Note: A span of bool cannot represent missing values.
  const size_t num_valid_values = values.size();
  size_t count_true = 0;
  size_t count_false = 0;

  for (const bool value : values) {
    if (value) {
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

// Append contents of `data` to a boolean column. If no `column_idx` is not
// given, a new column is created.
//
// Note that this function only creates the columns and copies the data, but it
// does not set `num_rows` on the dataset. Before using the dataset, `num_rows
// has to be set (e.g. using SetAndCheckNumRows).
absl::Status PopulateColumnBooleanNPBool(dataset::VerticalDataset& self,
                                         const std::string& name,
                                         py::array_t<bool>& data,
                                         std::optional<int> column_idx) {
  const auto unchecked = data.unchecked<1>();

  if (data.strides(0) != data.itemsize()) {
    return absl::InternalError("Expecting non-strided np.bool_ array.");
  }
  const auto values = absl::Span<const bool>(data.data(), unchecked.shape(0));

  if (!column_idx) {
    // Create column spec
    ASSIGN_OR_RETURN(const auto column_spec,
                     CreateBooleanColumnSpec(name, values));
    ASSIGN_OR_RETURN(auto* abstract_column, self.AddColumn(column_spec));
    // Import column data
    ASSIGN_OR_RETURN(auto* column,
                     abstract_column->MutableCastWithStatus<BooleanColumn>());
    column->mutable_values()->assign(values.data(),
                                     values.data() + values.size());
  } else {
    ASSIGN_OR_RETURN(auto* column,
                     self.MutableColumnWithCastWithStatus<BooleanColumn>(
                         column_idx.value()));
    column->mutable_values()->insert(column->mutable_values()->end(),
                                     values.data(),
                                     values.data() + values.size());
  }

  return absl::OkStatus();
}

// Removes '\0' at the end of a string_view. Returns a string_view without the
// zeroes.
std::string_view remove_tailing_zeros(std::string_view src) {
  int i = static_cast<int>(src.size()) - 1;
  while (i >= 0 && src[i] == 0) {
    i--;
  }
  return src.substr(0, i + 1);
}

// Non owning accessor to a np bytes array.
struct NPByteArray {
  // Wraps a single dimensional np::array of bytes.
  static absl::StatusOr<NPByteArray> Create(const py::array& data) {
    if (data.dtype().kind() != 'S') {
      return absl::InternalError(
          absl::StrCat("Expecting a np.bytes (i.e. |S) array. Got |",
                       std::string(1, data.dtype().kind()), " instead"));
    }
    if (data.ndim() != 1) {
      return absl::InternalError("Wrong shape");
    }
    py::buffer_info info = data.request();
    return NPByteArray{
        /*_data=*/(char*)info.ptr,
        /*_stride=*/(size_t)info.strides[0],
        /*_itemsize=*/(size_t)info.itemsize,
        /*_size=*/(size_t)info.shape[0],
    };
  }

  // Number of items.
  size_t size() const { return _size; }

  // Value accessor.
  std::string_view operator[](size_t i) const {
    return remove_tailing_zeros({_data + i * _stride, _itemsize});
  }

  // Extracts the content of the numpy array into a string vector.
  std::vector<std::string> ToVector() const {
    std::vector<std::string> dst(_size);
    for (size_t i = 0; i < _size; i++) {
      dst[i] = (*this)[i];
    }
    return dst;
  }

  const char* _data;
  const size_t _stride;
  const size_t _itemsize;
  const size_t _size;
};

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
Dictionary DictionaryFromValues(const NPByteArray& values,
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
        {/*count=*/src_value.second, /*key=*/std::string(src_value.first)});
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
    const NPByteArray& values, const py::array& force_dictionary) {
  ASSIGN_OR_RETURN(const auto wrapper, NPByteArray::Create(force_dictionary));
  const auto src_items = wrapper.ToVector();

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
    dictionary.items.push_back({.count = count_map[item], .key = item});
  }
  return dictionary;
}

// Creates a column spec for a categorical column.
absl::StatusOr<dataset::proto::Column> CreateCategoricalColumnSpec(
    const std::string& name, const NPByteArray& values,
    const int max_vocab_count, const int min_vocab_frequency,
    const std::optional<py::array>& force_dictionary) {
  if (max_vocab_count < -1) {
    return absl::InvalidArgumentError(
        absl::Substitute("Column $0 received invalid dataspec inference "
                         "argument max_vocab_count: $1",
                         name, max_vocab_count));
  }
  if (max_vocab_count == -1) {
    YDF_LOG(INFO) << "max_vocab_count = -1 for column " << name
                  << ", the dictionary will not be pruned by size.";
  }
  if (max_vocab_count == 0) {
    YDF_LOG(WARNING) << "max_vocab_count = 0 for column " << name
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
  column.set_name(name);
  column.set_type(dataset::proto::ColumnType::CATEGORICAL);
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

// Append contents of `data` to a categorical column. If no `column_idx` is not
// given, a new column is created.
//
// Note that this function only creates the columns and copies the data, but it
// does not set `num_rows` on the dataset. Before using the dataset, `num_rows
// has to be set (e.g. using SetAndCheckNumRows).
absl::Status PopulateColumnCategoricalNPBytes(
    dataset::VerticalDataset& self, const std::string& name, py::array& data,
    const int max_vocab_count, const int min_vocab_frequency,
    std::optional<int> column_idx, const std::optional<py::array> dictionary) {
  ASSIGN_OR_RETURN(const auto values, NPByteArray::Create(data));

  CategoricalColumn* column;
  ssize_t offset = 0;
  if (!column_idx.has_value()) {
    // Create column spec
    ASSIGN_OR_RETURN(
        const auto& column_spec,
        CreateCategoricalColumnSpec(name, values, max_vocab_count,
                                    min_vocab_frequency, dictionary));

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
  for (size_t value_idx = 0; value_idx < values.size(); value_idx++) {
    const auto value = values[value_idx];
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

// Append contents of `data` to a HASH column. If no `column_idx` is not
// given, a new column is created.
//
// Note that this function only creates the columns and copies the data, but it
// does not set `num_rows` on the dataset. Before using the dataset, `num_rows
// has to be set (e.g. using SetAndCheckNumRows).
absl::Status PopulateColumnHashNPBytes(dataset::VerticalDataset& self,
                                       const std::string& name, py::array& data,
                                       std::optional<int> column_idx) {
  ASSIGN_OR_RETURN(const auto values, NPByteArray::Create(data));

  HashColumn* column;
  size_t offset = 0;
  if (!column_idx.has_value()) {
    // Create column spec
    dataset::proto::Column column_spec;
    column_spec.set_name(name);
    column_spec.set_type(dataset::proto::ColumnType::HASH);

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
    const auto value = values[value_idx];
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
  std::string ds_as_string = "";
  {
    for (int col_idx = 0; col_idx < self.ncol(); col_idx++) {
      if (col_idx > 0) {
        absl::StrAppend(&ds_as_string, ",");
      }
      absl::StrAppend(&ds_as_string, self.column(col_idx)->name());
    }
    absl::StrAppend(&ds_as_string, "\n");
  }
  // Body
  for (dataset::VerticalDataset::row_t example_idx = 0;
       example_idx < self.nrow(); example_idx++) {
    for (int col_idx = 0; col_idx < self.ncol(); col_idx++) {
      const auto& col_spec = self.data_spec().columns(col_idx);
      if (col_idx > 0) {
        absl::StrAppend(&ds_as_string, ",");
      }
      if (!self.column(col_idx)->IsNa(example_idx)) {
        absl::StrAppend(&ds_as_string,
                        self.column(col_idx)->ToStringWithDigitPrecision(
                            example_idx, col_spec, /*digit_precision=*/6));
      }
    }
    absl::StrAppend(&ds_as_string, "\n");
  }
  return ds_as_string;
}

absl::Status CreateFromPathWithDataSpec(
    dataset::VerticalDataset& self, const std::string& path,
    const dataset::proto::DataSpecification& data_spec) {
  dataset::LoadConfig dataset_loading_config;
  // TODO: Should all columns be listed in the required columns?

  ASSIGN_OR_RETURN(const auto typed_path, dataset::GetTypedPath(path));

  RETURN_IF_ERROR(dataset::LoadVerticalDataset(
      typed_path, data_spec, &self,
      /*required_columns=*/absl::nullopt, dataset_loading_config));
  return absl::OkStatus();
}

absl::Status CreateFromPathWithDataSpecGuide(
    dataset::VerticalDataset& self, std::string path,
    const dataset::proto::DataSpecificationGuide& data_spec_guide) {
  dataset::LoadConfig dataset_loading_config;
  // TODO: Should all columns be listed in the required columns?

  ASSIGN_OR_RETURN(const std::string typed_path, dataset::GetTypedPath(path));
  dataset::proto::DataSpecification data_spec;
  RETURN_IF_ERROR(dataset::CreateDataSpecWithStatus(
      typed_path, false, data_spec_guide, &data_spec));

  return CreateFromPathWithDataSpec(self, path, data_spec);
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
           "Creates a dataset from a path, supports sharding. If the path is "
           "typed, use the type, the given type is used. Otherwise, YDF will "
           "try to determine the path and fail if this is not possible.")
      .def("CreateFromPathWithDataSpecGuide",
           WithStatus(CreateFromPathWithDataSpecGuide), py::arg("path"),
           py::arg("data_spec_guide"),
           "Creates a dataset from a path, supports sharding. If the path is "
           "typed, use the type, the given type is used. Otherwise, YDF will "
           "try to determine the path and fail if this is not possible.")
      .def("CreateColumnsFromDataSpec", WithStatus(CreateColumnsFromDataSpec),
           py::arg("data_spec"))
      .def("SetAndCheckNumRows", WithStatus(SetAndCheckNumRows),
           py::arg("set_data_spec"))
      // Data setters
      .def("PopulateColumnCategoricalNPBytes",
           WithStatus(PopulateColumnCategoricalNPBytes), py::arg("name"),
           py::arg("data").noconvert(), py::arg("max_vocab_count") = -1,
           py::arg("min_vocab_frequency") = -1,
           py::arg("column_idx") = std::nullopt,
           py::arg("dictionary") = std::nullopt)
      .def("PopulateColumnNumericalNPFloat32",
           WithStatus(PopulateColumnNumericalNPFloat32), py::arg("name"),
           py::arg("data").noconvert(), py::arg("column_idx") = std::nullopt)
      .def("PopulateColumnBooleanNPBool",
           WithStatus(PopulateColumnBooleanNPBool), py::arg("name"),
           py::arg("data").noconvert(), py::arg("column_idx") = std::nullopt)
      .def("PopulateColumnHashNPBytes", WithStatus(PopulateColumnHashNPBytes),
           py::arg("name"), py::arg("data").noconvert(),
           py::arg("column_idx") = std::nullopt);
}

}  // namespace yggdrasil_decision_forests::port::python
