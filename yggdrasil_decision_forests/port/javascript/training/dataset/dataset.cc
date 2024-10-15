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

#include "yggdrasil_decision_forests/port/javascript/training/dataset/dataset.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#include <emscripten/emscripten.h>
#endif  // __EMSCRIPTEN__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/port/javascript/training/util/status_casters.h"

namespace yggdrasil_decision_forests::port::javascript {

namespace {

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
Dictionary DictionaryFromValues(const std::vector<std::string>& values,
                                const int max_vocab_count,
                                const int min_vocab_frequency) {
  // Count unique values.
  // Warning: "dict_map" does not own its keys.
  absl::flat_hash_map<std::string_view, size_t> dict_map;
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

// Creates a column spec for a categorical column.
absl::StatusOr<dataset::proto::Column> CreateCategoricalColumnSpec(
    const std::string& name, const std::vector<std::string>& values,
    const bool is_label) {
  // Create column spec
  dataset::proto::Column column;
  column.set_type(dataset::proto::CATEGORICAL);
  column.set_name(name);

  int max_num_values =
      is_label ? -1 : column.categorical().max_number_of_unique_values();
  int min_value_count = is_label ? 0 : column.categorical().min_value_count();

  Dictionary dictionary =
      DictionaryFromValues(values, max_num_values, min_value_count);
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

// Creates a column spec for a numerical column.
absl::StatusOr<dataset::proto::Column> CreateNumericalColumnSpec(
    const absl::string_view name, const absl::Span<const float>& values) {
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

}  // namespace

void Dataset::AddNumericalColumn(const std::string& name,
                                 const std::vector<float>& data) {
  auto column_spec = CreateNumericalColumnSpec(name, absl::MakeSpan(data));
  if (!column_spec.ok()) {
    CheckOrThrowError(column_spec.status());
  }
  auto abstract_column = dataset_.AddColumn(column_spec.value());
  if (!abstract_column.ok()) {
    CheckOrThrowError(abstract_column.status());
  }
  auto column =
      abstract_column.value()
          ->MutableCastWithStatus<dataset::VerticalDataset::NumericalColumn>();
  if (!column.ok()) {
    CheckOrThrowError(abstract_column.status());
  }
  std::vector<float>& dst_values = *column.value()->mutable_values();
  dst_values.resize(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    dst_values[i] = data[i];
  }
  if (dataset_.nrow() == 0) {
    dataset_.set_nrow(data.size());
  } else {
    if (dataset_.nrow() != data.size()) {
      CheckOrThrowError(absl::InvalidArgumentError(
          "All columns must have the same number of items"));
    }
  }
}

void Dataset::AddCategoricalColumn(const std::string& name,
                                   const std::vector<std::string>& data,
                                   bool is_label) {
  auto column_spec = CreateCategoricalColumnSpec(name, data, is_label);
  if (!column_spec.ok()) {
    CheckOrThrowError(column_spec.status());
  }
  auto abstract_column = dataset_.AddColumn(column_spec.value());
  if (!abstract_column.ok()) {
    CheckOrThrowError(abstract_column.status());
  }
  auto column = abstract_column.value()
                    ->MutableCastWithStatus<
                        dataset::VerticalDataset::CategoricalColumn>();
  if (!column.ok()) {
    CheckOrThrowError(abstract_column.status());
  }
  column.value()->Resize(data.size());
  auto& dst_values = *column.value()->mutable_values();

  const auto& items = column_spec.value().categorical().items();

  if (items.empty()) {
    CheckOrThrowError(absl::InternalError(absl::Substitute(
        "Column \"$0\": Empty categorical dictionary.", name)));
  }

  for (size_t value_idx = 0; value_idx < data.size(); value_idx++) {
    const std::string_view value = data[value_idx];
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
    dst_values[value_idx] = dst_value;
  }
  if (dataset_.nrow() == 0) {
    dataset_.set_nrow(data.size());
  } else {
    if (dataset_.nrow() != data.size()) {
      CheckOrThrowError(absl::InvalidArgumentError(
          "All columns must have the same number of items"));
    }
  }
}

void init_dataset() {
#ifdef __EMSCRIPTEN__
  emscripten::class_<Dataset>("Dataset")
      // TODO: Check if a different memory management is useful here.
      .smart_ptr_constructor("Dataset", &std::make_shared<Dataset>)
      .function("addCategoricalColumn", &Dataset::AddCategoricalColumn)
      .function("addNumericalColumn", &Dataset::AddNumericalColumn);
#endif  // __EMSCRIPTEN__
}
}  // namespace yggdrasil_decision_forests::port::javascript
