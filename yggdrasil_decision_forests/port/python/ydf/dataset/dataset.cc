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
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "pybind11_abseil/status_casters.h"  // IWYU pargma : keep
#include "pybind11_protobuf/native_proto_caster.h"  // IWYU pargma : keep
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
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
    const std::string& name, absl::Span<const float> values) {
  size_t num_valid_values = 0;
  double sum_values = 0;
  double sum_square_values = 0;
  double min_value = 0;
  double max_value = 0;
  bool first_value = true;

  for (const float value : values) {
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
  const auto unchecked = data.unchecked<1>();

  if (data.strides(0) != data.itemsize()) {
    return absl::InternalError("Expecting non-strided np.float32 array.");
  }
  const auto values = absl::Span<const float>(data.data(), unchecked.shape(0));

  if (!column_idx.has_value()) {
    // Create column spec
    ASSIGN_OR_RETURN(const auto column_spec,
                     CreateNumericalColumnSpec(name, values));
    ASSIGN_OR_RETURN(auto* abstract_column, self.AddColumn(column_spec));
    // Import column data
    ASSIGN_OR_RETURN(auto* column,
                     abstract_column->MutableCastWithStatus<NumericalColumn>());
    column->mutable_values()->assign(values.data(),
                                     values.data() + values.size());
  } else {
    ASSIGN_OR_RETURN(auto* column,
                     self.MutableColumnWithCastWithStatus<NumericalColumn>(
                         column_idx.value()));
    column->mutable_values()->insert(column->mutable_values()->end(),
                                     values.data(),
                                     values.data() + values.size());
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
  static absl::StatusOr<NPByteArray> Create(py::array& data) {
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

  const char* _data;
  const size_t _stride;
  const size_t _itemsize;
  const size_t _size;
};

// Creates a column spec for a categorical column.
absl::StatusOr<dataset::proto::Column> CreateCategoricalColumnSpec(
    const std::string& name, const NPByteArray& values,
    const int max_vocab_count, const int min_vocab_frequency) {
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
  absl::flat_hash_map<absl::string_view, int> dict_map;

  // Count unique values.
  for (size_t value_idx = 0; value_idx < values.size(); value_idx++) {
    const auto value = values[value_idx];
    if (value.empty()) {
      continue;  // Value is missing
    }
    dict_map[value]++;
  }

  // Reduce dictionary
  struct Item {
    int count;
    absl::string_view key;
  };
  std::vector<Item> dict_list;
  dict_list.reserve(dict_map.size());
  size_t num_valid_values = 0;
  size_t num_oov_values = 0;
  for (const auto& src_value : dict_map) {
    num_valid_values += src_value.second;
    if (src_value.second < min_vocab_frequency) {
      num_oov_values += src_value.second;
      continue;
    }
    dict_list.push_back({/*count=*/src_value.second, /*key=*/src_value.first});
  }
  std::sort(dict_list.begin(), dict_list.end(),
            [](const Item& a, const Item& b) {
              // Sort by count
              if (a.count != b.count) {
                return a.count > b.count;  // Large counts first
              }
              // Sort by key
              return a.key < b.key;  // First keys first
            });

  if (max_vocab_count >= 0 && dict_list.size() > max_vocab_count) {
    for (int item_idx = max_vocab_count; item_idx < dict_list.size();
         item_idx++) {
      num_oov_values += dict_list[item_idx].count;
    }
    dict_list.resize(max_vocab_count);
  }

  // Create column spec
  dataset::proto::Column column;
  column.set_name(name);
  column.set_type(dataset::proto::ColumnType::CATEGORICAL);
  column.set_count_nas(values.size() - num_valid_values);

  // Copy dictionary in proto
  auto* column_cat = column.mutable_categorical();
  auto& dst_items = *column_cat->mutable_items();
  for (size_t item_idx = 0; item_idx < dict_list.size(); item_idx++) {
    const auto& item = dict_list[item_idx];
    auto& dst_item = dst_items[item.key];
    dst_item.set_count(item.count);
    // Index counting starts with the OOD item.
    dst_item.set_index(dataset::kOutOfDictionaryItemIndex + 1 + item_idx);
  }

  // Create OOV item
  auto& dst_oov_item = dst_items[dataset::kOutOfDictionaryItemKey];
  dst_oov_item.set_count(num_oov_values);
  dst_oov_item.set_index(dataset::kOutOfDictionaryItemIndex);

  // Set number of unique values. One value is the OOV item.
  column_cat->set_number_of_unique_values(dict_list.size() + 1);

  return column;
}

// Append contents of `data` to a categorical column. If no `column_idx` is not
// given, a new column is created.
//
// Note that this function only creates the columns and copies the data, but it
// does not set `num_rows` on the dataset. Before using the dataset, `num_rows
// has to be set (e.g. using SetAndCheckNumRows).
absl::Status PopulateColumnCategoricalNPBytes(dataset::VerticalDataset& self,
                                              const std::string& name,
                                              py::array& data,
                                              const int max_vocab_count,
                                              const int min_vocab_frequency,
                                              std::optional<int> column_idx) {
  ASSIGN_OR_RETURN(const auto values, NPByteArray::Create(data));

  CategoricalColumn* column;
  ssize_t offset = 0;
  if (!column_idx.has_value()) {
    // Create column spec
    ASSIGN_OR_RETURN(const auto& column_spec,
                     CreateCategoricalColumnSpec(name, values, max_vocab_count,
                                                 min_vocab_frequency));

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

}  // namespace

void init_dataset(py::module_& m) {
  py::class_<dataset::VerticalDataset>(m, "VerticalDataset")
      .def(py::init<>())
      .def("data_spec", &dataset::VerticalDataset::data_spec)
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
      .def("CreateColumnsFromDataSpec", &CreateColumnsFromDataSpec,
           py::arg("data_spec"))
      .def("SetAndCheckNumRows", &SetAndCheckNumRows, py::arg("set_data_spec"))
      // Data setters
      .def("PopulateColumnCategoricalNPBytes",
           &PopulateColumnCategoricalNPBytes, py::arg("name"),
           py::arg("data").noconvert(), py::arg("max_vocab_count") = -1,
           py::arg("min_vocab_frequency") = -1,
           py::arg("column_idx") = std::nullopt)
      .def("PopulateColumnNumericalNPFloat32",
           &PopulateColumnNumericalNPFloat32, py::arg("name"),
           py::arg("data").noconvert(), py::arg("column_idx") = std::nullopt)
      .def("PopulateColumnBooleanNPBool", &PopulateColumnBooleanNPBool,
           py::arg("name"), py::arg("data").noconvert(),
           py::arg("column_idx") = std::nullopt);
}

}  // namespace yggdrasil_decision_forests::port::python
