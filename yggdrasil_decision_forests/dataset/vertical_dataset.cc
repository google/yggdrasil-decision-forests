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

#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace dataset {

constexpr float VerticalDataset::NumericalColumn::kNaValue;
constexpr int8_t VerticalDataset::BooleanColumn::kNaValue;
constexpr int VerticalDataset::CategoricalColumn::kNaValue;
constexpr VerticalDataset::DiscretizedNumericalColumn::Format
    VerticalDataset::DiscretizedNumericalColumn::kNaValue;
constexpr uint64_t VerticalDataset::HashColumn::kNaValue;

namespace {

// Symbol used for the string representation of column values.
constexpr char kNaSymbol[] = "NA";  // NA=non-available i.e. missing value.
constexpr char kEmptySymbol[] = "EMPTY";

absl::StatusOr<std::unique_ptr<VerticalDataset::AbstractColumn>> CreateColumn(
    const proto::ColumnType type, const absl::string_view column_name) {
  std::unique_ptr<VerticalDataset::AbstractColumn> col;
  switch (type) {
    case proto::ColumnType::UNKNOWN:
      return absl::InvalidArgumentError(
          absl::StrCat("Impossible to create a column \"", column_name,
                       "\" of type UNKNOWN. If you "
                       "created the dataspec manually, make sure the \"type\" "
                       "fields are set for all the columns."));
    case proto::ColumnType::NUMERICAL:
      col = absl::make_unique<VerticalDataset::NumericalColumn>();
      break;
    case proto::ColumnType::NUMERICAL_SET:
      col = absl::make_unique<VerticalDataset::NumericalSetColumn>();
      break;
    case proto::ColumnType::NUMERICAL_LIST:
      col = absl::make_unique<VerticalDataset::NumericalListColumn>();
      break;
    case proto::ColumnType::CATEGORICAL:
      col = absl::make_unique<VerticalDataset::CategoricalColumn>();
      break;
    case proto::ColumnType::CATEGORICAL_SET:
      col = absl::make_unique<VerticalDataset::CategoricalSetColumn>();
      break;
    case proto::ColumnType::CATEGORICAL_LIST:
      col = absl::make_unique<VerticalDataset::CategoricalListColumn>();
      break;
    case proto::ColumnType::BOOLEAN:
      col = absl::make_unique<VerticalDataset::BooleanColumn>();
      break;
    case proto::ColumnType::STRING:
      col = absl::make_unique<VerticalDataset::StringColumn>();
      break;
    case proto::ColumnType::DISCRETIZED_NUMERICAL:
      col = absl::make_unique<VerticalDataset::DiscretizedNumericalColumn>();
      break;
    case proto::ColumnType::HASH:
      col = absl::make_unique<VerticalDataset::HashColumn>();
      break;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Column type ", proto::ColumnType_Name(type),
          " provided for column \"", column_name, "\" not implemented"));
  }
  col->set_name(column_name);
  return std::move(col);
}

// Ensure that two solumn specs are compatible i.e. "src_spec" can be converted
// into "dst_spec".
absl::Status CheckCompatibleCategocialColumnSpec(
    const proto::Column& src_spec, const proto::Column& dst_spec) {
  if (src_spec.categorical().is_already_integerized() !=
      dst_spec.categorical().is_already_integerized()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Non matching \"is_already_integerized\" for column \"",
                     src_spec.name(), "\"."));
  }
  if (src_spec.categorical().is_already_integerized()) {
    STATUS_CHECK_LE(dst_spec.categorical().number_of_unique_values(),
                    src_spec.categorical().number_of_unique_values());
  }
  return absl::OkStatus();
}

}  // namespace

int VerticalDataset::ColumnNameToColumnIdx(absl::string_view name) const {
  for (int col_idx = 0; col_idx < columns_.size(); col_idx++) {
    if (column(col_idx)->name() == name) {
      return col_idx;
    }
  }
  return -1;
}

void VerticalDataset::AppendExample(
    const proto::Example& example,
    const absl::optional<std::vector<int>>& load_columns) {
  // TODO: Update.
  CHECK_OK(AppendExampleWithStatus(example, load_columns));
}

absl::Status VerticalDataset::AppendExampleWithStatus(
    const proto::Example& example,
    const absl::optional<std::vector<int>>& load_columns) {
  DCHECK_EQ(columns_.size(), example.attributes_size());
  if (load_columns.has_value()) {
    for (int col_idx : load_columns.value()) {
      mutable_column(col_idx)->AddFromExample(example.attributes(col_idx));
    }
  } else {
    for (int col_idx = 0; col_idx < columns_.size(); col_idx++) {
      mutable_column(col_idx)->AddFromExample(example.attributes(col_idx));
    }
  }
  nrow_++;
  return absl::OkStatus();
}

void VerticalDataset::ExtractExample(const row_t example_idx,
                                     proto::Example* example) const {
  DCHECK_GE(example_idx, 0);
  DCHECK_LT(example_idx, nrow_);
  example->mutable_attributes()->Clear();
  for (int col_idx = 0; col_idx < columns_.size(); col_idx++) {
    const auto& col = *column(col_idx);
    auto* attribute = example->add_attributes();
    if (col.nrows() > 0) {
      col.ExtractExample(example_idx, attribute);
    }
  }
}

void VerticalDataset::PushBackOwnedColumn(
    std::unique_ptr<VerticalDataset::AbstractColumn>&& column) {
  const auto* raw_pointer = column.get();
  columns_.push_back(ColumnContainer{raw_pointer, std::move(column)});
}

void VerticalDataset::PushBackNotOwnedColumn(
    const VerticalDataset::AbstractColumn* column) {
  columns_.push_back(ColumnContainer{column, nullptr});
}

bool VerticalDataset::OwnsColumn(int col) const {
  return columns_[col].owned_column != nullptr;
}

VerticalDataset VerticalDataset::ShallowNonOwningClone() const {
  VerticalDataset clone;
  clone.set_data_spec(data_spec());
  clone.set_nrow(nrow());
  for (int col_idx = 0; col_idx < data_spec_.columns_size(); col_idx++) {
    clone.PushBackNotOwnedColumn(column(col_idx));
  }
  return clone;
}

absl::StatusOr<VerticalDataset::AbstractColumn*> VerticalDataset::AddColumn(
    const proto::Column& column_spec) {
  if (ColumnNameToColumnIdx(column_spec.name()) != -1) {
    return absl::InvalidArgumentError("The column already exists");
  }
  *data_spec_.add_columns() = column_spec;
  ASSIGN_OR_RETURN(auto new_column,
                   CreateColumn(column_spec.type(), column_spec.name()));
  PushBackOwnedColumn(std::move(new_column));
  auto* column = mutable_column(columns_.size() - 1);
  column->Resize(nrow_);
  column->set_name(column_spec.name());
  return column;
}

absl::StatusOr<proto::Column*> VerticalDataset::AddColumn(
    const absl::string_view name, const proto::ColumnType type) {
  if (ColumnNameToColumnIdx(name) != -1) {
    return absl::InvalidArgumentError("The column already exists");
  }
  auto* column_spec = data_spec_.add_columns();
  column_spec->set_name(std::string(name));
  column_spec->set_type(type);
  ASSIGN_OR_RETURN(auto new_column,
                   CreateColumn(column_spec->type(), column_spec->name()));
  PushBackOwnedColumn(std::move(new_column));
  auto* column = mutable_column(columns_.size() - 1);
  column->Resize(nrow_);
  column->set_name(column_spec->name());
  return column_spec;
}

absl::StatusOr<VerticalDataset::AbstractColumn*> VerticalDataset::ReplaceColumn(
    int column_idx, const proto::Column& column_spec) {
  DCHECK_GE(column_idx, 0);
  DCHECK_LT(column_idx, columns_.size());
  *data_spec_.mutable_columns(column_idx) = column_spec;
  ASSIGN_OR_RETURN(auto new_column,
                   CreateColumn(column_spec.type(), column_spec.name()));
  auto* raw_pointer = new_column.get();
  columns_[column_idx] = ColumnContainer{raw_pointer, std::move(new_column)};
  raw_pointer->Resize(nrow_);
  return raw_pointer;
}

absl::Status VerticalDataset::CreateColumnsFromDataspec() {
  columns_.clear();
  columns_.reserve(data_spec_.columns_size());
  for (int col_idx = 0; col_idx < data_spec_.columns_size(); col_idx++) {
    const auto& col_spec = data_spec_.columns(col_idx);
    ASSIGN_OR_RETURN(auto new_column,
                     CreateColumn(col_spec.type(), col_spec.name()));
    PushBackOwnedColumn(std::move(new_column));
    columns_.back().owned_column->set_name(col_spec.name());
    DCHECK_EQ(columns_.back().column->type(), col_spec.type());
  }
  return absl::OkStatus();
}

absl::Status VerticalDataset::NumericalColumn::ConvertToGivenDataspec(
    AbstractColumn* dst, const proto::Column& src_spec,
    const proto::Column& dst_spec) const {
  auto* cast_dst = dst->MutableCast<NumericalColumn>();
  *cast_dst->mutable_values() = values();
  return absl::OkStatus();
}

absl::Status VerticalDataset::HashColumn::ConvertToGivenDataspec(
    AbstractColumn* dst, const proto::Column& src_spec,
    const proto::Column& dst_spec) const {
  auto* cast_dst = dst->MutableCast<HashColumn>();
  *cast_dst->mutable_values() = values();
  return absl::OkStatus();
}

absl::Status VerticalDataset::BooleanColumn::ConvertToGivenDataspec(
    AbstractColumn* dst, const proto::Column& src_spec,
    const proto::Column& dst_spec) const {
  auto* cast_dst = dst->MutableCast<BooleanColumn>();
  *cast_dst->mutable_values() = values();
  return absl::OkStatus();
}

absl::Status VerticalDataset::NumericalSetColumn::ConvertToGivenDataspec(
    AbstractColumn* dst, const proto::Column& src_spec,
    const proto::Column& dst_spec) const {
  auto* cast_dst = dst->MutableCast<NumericalSetColumn>();
  cast_dst->mutable_values() = values();
  cast_dst->mutable_bank() = bank();
  return absl::OkStatus();
}

absl::Status VerticalDataset::NumericalListColumn::ConvertToGivenDataspec(
    AbstractColumn* dst, const proto::Column& src_spec,
    const proto::Column& dst_spec) const {
  auto* cast_dst = dst->MutableCast<NumericalListColumn>();
  cast_dst->mutable_values() = values();
  cast_dst->mutable_bank() = bank();
  return absl::OkStatus();
}

absl::Status VerticalDataset::StringColumn::ConvertToGivenDataspec(
    AbstractColumn* dst, const proto::Column& src_spec,
    const proto::Column& dst_spec) const {
  auto* cast_dst = dst->MutableCast<StringColumn>();
  *cast_dst->mutable_values() = values();
  return absl::OkStatus();
}

absl::Status VerticalDataset::CategoricalColumn::ConvertToGivenDataspec(
    AbstractColumn* dst, const proto::Column& src_spec,
    const proto::Column& dst_spec) const {
  auto* cast_dst = dst->MutableCast<CategoricalColumn>();
  RETURN_IF_ERROR(CheckCompatibleCategocialColumnSpec(src_spec, dst_spec));
  if (src_spec.categorical().is_already_integerized()) {
    *cast_dst->mutable_values() = values();
  } else {
    for (row_t example_idx = 0; example_idx < values().size(); example_idx++) {
      if (IsNa(example_idx)) {
        cast_dst->AddNA();
        continue;
      }
      const int src_value_idx = values()[example_idx];
      const std::string value =
          CategoricalIdxToRepresentation(src_spec, src_value_idx, false);
      ASSIGN_OR_RETURN(const int dst_value_idx,
                       CategoricalStringToValueWithStatus(value, dst_spec));
      cast_dst->Add(dst_value_idx);
    }
  }
  return absl::OkStatus();
}

absl::Status
VerticalDataset::DiscretizedNumericalColumn::ConvertToGivenDataspec(
    AbstractColumn* dst, const proto::Column& src_spec,
    const proto::Column& dst_spec) const {
  return absl::InvalidArgumentError(
      "The conversion of dataspecs with DISCRETIZED_NUMERICAL columns is "
      "not supported (yet). Use NUMERICAL columns or re-use the same "
      "dataspec.");
}

absl::Status VerticalDataset::CategoricalSetColumn::ConvertToGivenDataspec(
    AbstractColumn* dst, const proto::Column& src_spec,
    const proto::Column& dst_spec) const {
  auto* cast_dst = dst->MutableCast<CategoricalSetColumn>();
  cast_dst->mutable_values() = values();
  if (src_spec.categorical().is_already_integerized()) {
    cast_dst->mutable_bank() = bank();
  } else {
    for (row_t bank_idx = 0; bank_idx < bank().size(); bank_idx++) {
      const int src_value_idx = bank()[bank_idx];
      const std::string value =
          CategoricalIdxToRepresentation(src_spec, src_value_idx, false);
      ASSIGN_OR_RETURN(const int dst_value_idx,
                       CategoricalStringToValueWithStatus(value, dst_spec));
      cast_dst->mutable_bank().push_back(dst_value_idx);
    }
  }
  return absl::OkStatus();
}

absl::Status VerticalDataset::CategoricalListColumn::ConvertToGivenDataspec(
    AbstractColumn* dst, const proto::Column& src_spec,
    const proto::Column& dst_spec) const {
  auto* cast_dst = dst->MutableCast<CategoricalListColumn>();
  cast_dst->mutable_values() = values();
  if (src_spec.categorical().is_already_integerized()) {
    cast_dst->mutable_bank() = bank();
  } else {
    for (row_t bank_idx = 0; bank_idx < bank().size(); bank_idx++) {
      const int src_value_idx = bank()[bank_idx];
      const std::string value =
          CategoricalIdxToRepresentation(src_spec, src_value_idx, false);
      ASSIGN_OR_RETURN(const int dst_value_idx,
                       CategoricalStringToValueWithStatus(value, dst_spec));
      cast_dst->mutable_bank().push_back(dst_value_idx);
    }
  }
  return absl::OkStatus();
}

void VerticalDataset::NumericalColumn::ExtractExample(
    const row_t example_idx, proto::Example::Attribute* attribute) const {
  if (IsNa(example_idx)) {
    return;
  }
  attribute->set_numerical(values()[example_idx]);
}

void VerticalDataset::HashColumn::ExtractExample(
    const row_t example_idx, proto::Example::Attribute* attribute) const {
  if (IsNa(example_idx)) {
    return;
  }
  attribute->set_hash(values()[example_idx]);
}

void VerticalDataset::BooleanColumn::ExtractExample(
    const row_t example_idx, proto::Example::Attribute* attribute) const {
  if (IsNa(example_idx)) {
    return;
  }
  attribute->set_boolean(values()[example_idx] == kTrueValue);
}

void VerticalDataset::CategoricalColumn::ExtractExample(
    const row_t example_idx, proto::Example::Attribute* attribute) const {
  if (IsNa(example_idx)) {
    return;
  }
  attribute->set_categorical(values()[example_idx]);
}

void VerticalDataset::DiscretizedNumericalColumn::ExtractExample(
    const row_t example_idx, proto::Example::Attribute* attribute) const {
  if (IsNa(example_idx)) {
    return;
  }
  attribute->set_discretized_numerical(values()[example_idx]);
}

void VerticalDataset::NumericalSetColumn::ExtractExample(
    const row_t example_idx, proto::Example::Attribute* attribute) const {
  if (IsNa(example_idx)) {
    return;
  }
  *attribute->mutable_numerical_set()->mutable_values() = {
      bank().begin() + values()[example_idx].first,
      bank().begin() + values()[example_idx].second};
}

void VerticalDataset::CategoricalSetColumn::ExtractExample(
    const row_t example_idx, proto::Example::Attribute* attribute) const {
  if (IsNa(example_idx)) {
    return;
  }
  *attribute->mutable_categorical_set()->mutable_values() = {
      bank().begin() + values()[example_idx].first,
      bank().begin() + values()[example_idx].second};
}

void VerticalDataset::NumericalListColumn::ExtractExample(
    const row_t example_idx, proto::Example::Attribute* attribute) const {
  if (IsNa(example_idx)) {
    return;
  }
  *attribute->mutable_numerical_list()->mutable_values() = {
      bank().begin() + values()[example_idx].first,
      bank().begin() + values()[example_idx].second};
}

void VerticalDataset::CategoricalListColumn::ExtractExample(
    const row_t example_idx, proto::Example::Attribute* attribute) const {
  if (IsNa(example_idx)) {
    return;
  }
  *attribute->mutable_categorical_list()->mutable_values() = {
      bank().begin() + values()[example_idx].first,
      bank().begin() + values()[example_idx].second};
}

void VerticalDataset::StringColumn::ExtractExample(
    const row_t example_idx, proto::Example::Attribute* attribute) const {
  if (IsNa(example_idx)) {
    return;
  }
  attribute->set_text(values()[example_idx]);
}

void VerticalDataset::NumericalColumn::AddFromExample(
    const proto::Example::Attribute& attribute) {
  if (ABSL_PREDICT_FALSE(dataset::IsNa(attribute))) {
    AddNA();
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kNumerical);
    Add(attribute.numerical());
  }
}

void VerticalDataset::HashColumn::AddFromExample(
    const proto::Example::Attribute& attribute) {
  if (ABSL_PREDICT_FALSE(dataset::IsNa(attribute))) {
    AddNA();
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kHash);
    Add(attribute.hash());
  }
}

void VerticalDataset::BooleanColumn::AddFromExample(
    const proto::Example::Attribute& attribute) {
  if (ABSL_PREDICT_FALSE(dataset::IsNa(attribute))) {
    Add(kNaValue);
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kBoolean);
    Add(attribute.boolean());
  }
}

void VerticalDataset::CategoricalColumn::AddFromExample(
    const proto::Example::Attribute& attribute) {
  if (ABSL_PREDICT_FALSE(dataset::IsNa(attribute))) {
    Add(kNaValue);
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kCategorical);
    Add(attribute.categorical());
  }
}

void VerticalDataset::DiscretizedNumericalColumn::AddFromExample(
    const proto::Example::Attribute& attribute) {
  if (ABSL_PREDICT_FALSE(dataset::IsNa(attribute))) {
    Add(kNaValue);
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kDiscretizedNumerical);
    Add(attribute.discretized_numerical());
  }
}

void VerticalDataset::NumericalSetColumn::AddFromExample(
    const proto::Example::Attribute& attribute) {
  if (ABSL_PREDICT_FALSE(dataset::IsNa(attribute))) {
    AddNA();
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kNumericalSet);
    Add(attribute.numerical_set().values().begin(),
        attribute.numerical_set().values().end());
  }
}

void VerticalDataset::CategoricalSetColumn::AddFromExample(
    const proto::Example::Attribute& attribute) {
  if (ABSL_PREDICT_FALSE(dataset::IsNa(attribute))) {
    AddNA();
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kCategoricalSet);
    Add(attribute.categorical_set().values().begin(),
        attribute.categorical_set().values().end());
  }
}

void VerticalDataset::NumericalListColumn::AddFromExample(
    const proto::Example::Attribute& attribute) {
  if (ABSL_PREDICT_FALSE(dataset::IsNa(attribute))) {
    AddNA();
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kNumericalList);
    Add(attribute.numerical_list().values().begin(),
        attribute.numerical_list().values().end());
  }
}

void VerticalDataset::CategoricalListColumn::AddFromExample(
    const proto::Example::Attribute& attribute) {
  if (ABSL_PREDICT_FALSE(dataset::IsNa(attribute))) {
    AddNA();
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kCategoricalList);
    Add(attribute.categorical_list().values().begin(),
        attribute.categorical_list().values().end());
  }
}

void VerticalDataset::StringColumn::AddFromExample(
    const proto::Example::Attribute& attribute) {
  if (ABSL_PREDICT_FALSE(dataset::IsNa(attribute))) {
    AddNA();
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kText);
    Add(attribute.text());
  }
}

void VerticalDataset::NumericalColumn::Set(
    row_t example_idx, const proto::Example::Attribute& attribute) {
  if (dataset::IsNa(attribute)) {
    Set(example_idx, kNaValue);
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kNumerical);
    Set(example_idx, attribute.numerical());
  }
}

void VerticalDataset::NumericalColumn::Set(row_t example_idx, float value) {
  (*mutable_values())[example_idx] = value;
}

void VerticalDataset::HashColumn::Set(row_t example_idx, const uint64_t value) {
  (*mutable_values())[example_idx] = value;
}

void VerticalDataset::HashColumn::Set(
    row_t example_idx, const proto::Example::Attribute& attribute) {
  if (dataset::IsNa(attribute)) {
    Set(example_idx, kNaValue);
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kHash);
    Set(example_idx, attribute.hash());
  }
}

void VerticalDataset::BooleanColumn::Set(
    row_t example_idx, const proto::Example::Attribute& attribute) {
  if (dataset::IsNa(attribute)) {
    (*mutable_values())[example_idx] = kNaValue;
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kBoolean);
    (*mutable_values())[example_idx] = attribute.boolean();
  }
}

void VerticalDataset::CategoricalColumn::Set(
    row_t example_idx, const proto::Example::Attribute& attribute) {
  if (dataset::IsNa(attribute)) {
    (*mutable_values())[example_idx] = kNaValue;
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kCategorical);
    (*mutable_values())[example_idx] = attribute.categorical();
  }
}

void VerticalDataset::CategoricalColumn::Set(row_t example_idx, int32_t value) {
  (*mutable_values())[example_idx] = value;
}

void VerticalDataset::DiscretizedNumericalColumn::Set(
    row_t example_idx, const proto::Example::Attribute& attribute) {
  if (dataset::IsNa(attribute)) {
    (*mutable_values())[example_idx] = kNaValue;
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kDiscretizedNumerical);
    (*mutable_values())[example_idx] = attribute.discretized_numerical();
  }
}

void VerticalDataset::NumericalSetColumn::Set(
    row_t example_idx, const proto::Example::Attribute& attribute) {
  if (dataset::IsNa(attribute)) {
    SetNA(example_idx);
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kNumericalSet);
    SetIter(example_idx, attribute.numerical_set().values().begin(),
            attribute.numerical_set().values().end());
  }
}

void VerticalDataset::CategoricalSetColumn::Set(
    row_t example_idx, const proto::Example::Attribute& attribute) {
  if (dataset::IsNa(attribute)) {
    SetNA(example_idx);
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kCategoricalSet);
    SetIter(example_idx, attribute.categorical_set().values().begin(),
            attribute.categorical_set().values().end());
  }
}

void VerticalDataset::NumericalListColumn::Set(
    row_t example_idx, const proto::Example::Attribute& attribute) {
  if (dataset::IsNa(attribute)) {
    SetNA(example_idx);
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kNumericalList);
    SetIter(example_idx, attribute.numerical_list().values().begin(),
            attribute.numerical_list().values().end());
  }
}

void VerticalDataset::CategoricalListColumn::Set(
    row_t example_idx, const proto::Example::Attribute& attribute) {
  if (dataset::IsNa(attribute)) {
    SetNA(example_idx);
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kCategoricalList);
    SetIter(example_idx, attribute.categorical_list().values().begin(),
            attribute.categorical_list().values().end());
  }
}

void VerticalDataset::StringColumn::Set(
    row_t example_idx, const proto::Example::Attribute& attribute) {
  if (dataset::IsNa(attribute)) {
    is_na_[example_idx] = true;
    (*mutable_values())[example_idx].clear();
  } else {
    DCHECK_EQ(attribute.type_case(),
              proto::Example::Attribute::TypeCase::kText);
    Set(example_idx, attribute.text());
  }
}

void VerticalDataset::StringColumn::Set(row_t example_idx,
                                        const absl::string_view value) {
  (*mutable_values())[example_idx] = std::string(value);
  is_na_[example_idx] = false;
}

std::string VerticalDataset::NumericalColumn::ToStringWithDigitPrecision(
    const row_t row, const proto::Column& col_spec, int digit_precision) const {
  return absl::StrFormat("%.*g", digit_precision, values()[row]);
}

std::string VerticalDataset::StringColumn::ToStringWithDigitPrecision(
    const row_t row, const proto::Column& col_spec, int digit_precision) const {
  return values()[row];
}

std::string VerticalDataset::BooleanColumn::ToStringWithDigitPrecision(
    const row_t row, const proto::Column& col_spec, int digit_precision) const {
  if (values()[row] == 0) {
    return "0";
  }
  if (values()[row] == 1) {
    return "1";
  }
  if (values()[row] == kNaValue) {
    return kNaSymbol;
  }
  return "Invalid";
}

std::string VerticalDataset::NumericalSetColumn::ToStringWithDigitPrecision(
    const row_t row, const proto::Column& col_spec, int digit_precision) const {
  if (IsNa(row)) {
    return kNaSymbol;
  }
  const auto& indices = values()[row];
  if (indices.first == indices.second) {
    return kEmptySymbol;
  }
  const std::string format_mask = absl::StrCat("%.", digit_precision, "g");
  std::string rep;
  for (size_t bank_idx = indices.first; bank_idx < indices.second; bank_idx++) {
    if (bank_idx != indices.first) {
      absl::StrAppend(&rep, ", ");
    }
    absl::StrAppendFormat(&rep, "%.*g", digit_precision, bank()[bank_idx]);
  }
  return rep;
}

std::string VerticalDataset::NumericalListColumn::ToStringWithDigitPrecision(
    const row_t row, const proto::Column& col_spec, int digit_precision) const {
  if (IsNa(row)) {
    return kNaSymbol;
  }
  const auto& indices = values()[row];
  if (indices.first == indices.second) {
    return kEmptySymbol;
  }
  std::string rep;
  for (size_t bank_idx = indices.first; bank_idx < indices.second; bank_idx++) {
    if (bank_idx != indices.first) {
      absl::StrAppend(&rep, ", ");
    }
    absl::StrAppendFormat(&rep, "%.*g", digit_precision, bank()[bank_idx]);
  }
  return rep;
}

std::string VerticalDataset::CategoricalColumn::ToStringWithDigitPrecision(
    const row_t row, const proto::Column& col_spec, int digit_precision) const {
  if (IsNa(row)) {
    return kNaSymbol;
  }
  if (col_spec.categorical().is_already_integerized()) {
    return absl::StrCat(values()[row]);
  } else {
    return CategoricalIdxToRepresentation(col_spec, values()[row]);
  }
}

std::string
VerticalDataset::DiscretizedNumericalColumn::ToStringWithDigitPrecision(
    const row_t row, const proto::Column& col_spec, int digit_precision) const {
  if (IsNa(row)) {
    return kNaSymbol;
  }
  // TODO: Update.
  const float value =
      DiscretizedNumericalToNumerical(col_spec, values()[row]).value();
  return absl::StrFormat("%.*g", digit_precision, value);
}

std::string VerticalDataset::CategoricalSetColumn::ToStringWithDigitPrecision(
    const row_t row, const proto::Column& col_spec, int digit_precision) const {
  if (IsNa(row)) {
    return kNaSymbol;
  }
  const auto& indices = values()[row];
  if (indices.first == indices.second) {
    return kEmptySymbol;
  }
  std::string rep;
  for (size_t bank_idx = indices.first; bank_idx < indices.second; bank_idx++) {
    if (bank_idx != indices.first) {
      absl::StrAppend(&rep, ", ");
    }
    absl::StrAppend(&rep,
                    CategoricalIdxToRepresentation(col_spec, bank()[bank_idx]));
  }
  return rep;
}

std::string VerticalDataset::CategoricalListColumn::ToStringWithDigitPrecision(
    const row_t row, const proto::Column& col_spec, int digit_precision) const {
  if (IsNa(row)) {
    return kNaSymbol;
  }
  const auto& indices = values()[row];
  if (indices.first == indices.second) {
    return kEmptySymbol;
  }
  std::string rep;
  for (size_t bank_idx = indices.first; bank_idx < indices.second; bank_idx++) {
    if (bank_idx != indices.first) {
      absl::StrAppend(&rep, ", ");
    }
    absl::StrAppend(&rep,
                    CategoricalIdxToRepresentation(col_spec, bank()[bank_idx]));
  }
  return rep;
}

absl::StatusOr<VerticalDataset> VerticalDataset::ConvertToGivenDataspec(
    const proto::DataSpecification& new_data_spec,
    const std::vector<int>& required_column_idxs) const {
  VerticalDataset new_dataset;
  new_dataset.data_spec_ = new_data_spec;
  new_dataset.nrow_ = nrow_;
  RETURN_IF_ERROR(new_dataset.CreateColumnsFromDataspec());

  for (int dst_col_idx = 0; dst_col_idx < new_dataset.ncol(); dst_col_idx++) {
    auto* dst_col = new_dataset.mutable_column(dst_col_idx);
    if (!HasColumn(dst_col->name(), data_spec_)) {
      if (std::find(required_column_idxs.begin(), required_column_idxs.end(),
                    dst_col_idx) != required_column_idxs.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Source dataspec doesn't contains the required column \"",
            dst_col->name(), "\"."));
      } else {
        for (row_t example_idx = 0; example_idx < nrow_; example_idx++) {
          dst_col->AddNA();
        }
        continue;
      }
    }
    const int src_col_idx = GetColumnIdxFromName(dst_col->name(), data_spec_);
    const auto& src_col = *column(src_col_idx);
    if (src_col.type() != dst_col->type()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Source and destination dataspec types don't match for column \"",
          dst_col->name(), "\"."));
    }
    RETURN_IF_ERROR(
        src_col.ConvertToGivenDataspec(dst_col, data_spec_.columns(src_col_idx),
                                       new_data_spec.columns(dst_col_idx)));
  }
  return std::move(new_dataset);
}

absl::Status VerticalDataset::Append(const VerticalDataset& src) {
  std::vector<row_t> indices(src.nrow());
  std::iota(indices.begin(), indices.end(), 0);
  return Append(src, indices);
}

absl::Status VerticalDataset::Append(const VerticalDataset& src,
                                     const std::vector<row_t>& indices) {
  if (columns_.empty()) {
    data_spec_ = src.data_spec();
    RETURN_IF_ERROR(CreateColumnsFromDataspec());
  }
  // Note: "MessageDifferencer" is too slow.
  if (src.data_spec().ShortDebugString() != data_spec().ShortDebugString()) {
    return absl::InvalidArgumentError(
        "The source and destination datasets should have the same dataspec.");
  }
  nrow_ += indices.size();
  for (int col_idx = 0; col_idx < ncol(); col_idx++) {
    RETURN_IF_ERROR(src.column(col_idx)->ExtractAndAppend(
        indices, mutable_column(col_idx)));
  }
  return absl::OkStatus();
}

void MapExampleToProtoExample(
    const std::unordered_map<std::string, std::string>& src,
    const proto::DataSpecification& data_spec, proto::Example* dst) {
  // TODO: Update.
  CHECK_OK(MapExampleToProtoExampleWithStatus(src, data_spec, dst));
}

absl::Status MapExampleToProtoExampleWithStatus(
    const std::unordered_map<std::string, std::string>& src,
    const proto::DataSpecification& data_spec, proto::Example* dst) {
  std::vector<std::string> flat_values;
  std::vector<int> col_idx_to_field_idx(data_spec.columns_size(), -1);
  for (const auto& src_value : src) {
    const int col_idx = GetColumnIdxFromName(src_value.first, data_spec);
    col_idx_to_field_idx[col_idx] = flat_values.size();
    flat_values.push_back(src_value.second);
  }
  return CsvRowToExample(flat_values, data_spec, col_idx_to_field_idx, dst);
}

absl::StatusOr<std::unordered_map<std::string, std::string>>
ProtoExampleToMapExample(const proto::Example& src,
                         const proto::DataSpecification& data_spec) {
  std::unordered_map<std::string, std::string> dst;
  VerticalDataset ds;
  ds.set_data_spec(data_spec);
  RETURN_IF_ERROR(ds.CreateColumnsFromDataspec());
  RETURN_IF_ERROR(ds.AppendExampleWithStatus(src));

  for (int col_idx = 0; col_idx < data_spec.columns_size(); col_idx++) {
    auto& dst_value = dst[data_spec.columns(col_idx).name()];
    dst_value = ds.column(col_idx)->ToString(0, data_spec.columns(col_idx));
  }
  return dst;
}

void VerticalDataset::AppendExample(
    const std::unordered_map<std::string, std::string>& example) {
  // TODO: Update.
  CHECK_OK(AppendExampleWithStatus(example));
}

absl::Status VerticalDataset::AppendExampleWithStatus(
    const std::unordered_map<std::string, std::string>& example) {
  proto::Example proto_example;
  RETURN_IF_ERROR(
      MapExampleToProtoExampleWithStatus(example, data_spec(), &proto_example));
  return AppendExampleWithStatus(proto_example);
}

std::string VerticalDataset::ValueToString(const row_t row,
                                           const int col) const {
  return column(col)->ToString(row, data_spec().columns(col));
}

std::string VerticalDataset::DebugString(
    const absl::optional<row_t> max_displayed_rows, const bool vertical,
    const int digit_precision) const {
  // Maximum number of rows to display.
  row_t num_displayed_rows = nrow();
  if (max_displayed_rows.has_value() &&
      *max_displayed_rows < num_displayed_rows) {
    num_displayed_rows = *max_displayed_rows;
  }

  std::string rep;
  if (vertical) {
    // Header
    for (int col_idx = 0; col_idx < ncol(); col_idx++) {
      if (col_idx > 0) {
        absl::StrAppend(&rep, ",");
      }
      absl::StrAppend(&rep, column(col_idx)->name());
    }
    absl::StrAppend(&rep, "\n");

    // Body
    for (row_t row_idx = 0; row_idx < num_displayed_rows; row_idx++) {
      for (int col_idx = 0; col_idx < ncol(); col_idx++) {
        const auto& col_spec = data_spec().columns(col_idx);
        if (col_idx > 0) {
          absl::StrAppend(&rep, ",");
        }
        absl::StrAppend(&rep, column(col_idx)->ToStringWithDigitPrecision(
                                  row_idx, col_spec, digit_precision));
      }
      absl::StrAppend(&rep, "\n");
    }
  } else {
    for (int col_idx = 0; col_idx < ncol(); col_idx++) {
      const auto* col = column(col_idx);
      const auto& col_spec = data_spec().columns(col_idx);
      absl::StrAppend(&rep, col->name(), ": ");
      for (row_t row_idx = 0; row_idx < num_displayed_rows; row_idx++) {
        if (row_idx > 0) {
          absl::StrAppend(&rep, ",");
        }
        absl::StrAppend(&rep, col->ToStringWithDigitPrecision(row_idx, col_spec,
                                                              digit_precision));
      }
      absl::StrAppend(&rep, "\n");
    }
  }
  return rep;
}

void VerticalDataset::Set(const row_t row, const int col,
                          const proto::Example::Attribute& value) {
  mutable_column(col)->Set(row, value);
}

void VerticalDataset::Reserve(
    const row_t num_rows,
    const absl::optional<std::vector<int>>& load_columns) {
  if (load_columns.has_value()) {
    for (int col_idx : load_columns.value()) {
      mutable_column(col_idx)->Reserve(num_rows);
    }
  } else {
    for (int col_idx = 0; col_idx < columns_.size(); col_idx++) {
      mutable_column(col_idx)->Reserve(num_rows);
    }
  }
}

void VerticalDataset::Resize(const row_t num_rows) {
  nrow_ = num_rows;
  for (int col_idx = 0; col_idx < columns_.size(); col_idx++) {
    mutable_column(col_idx)->Resize(num_rows);
  }
}

uint64_t VerticalDataset::MemoryUsage() const {
  uint64_t reserved = 0;
  for (int col_idx = 0; col_idx < ncol(); col_idx++) {
    const auto col_mem = column(col_idx)->memory_usage();
    reserved += col_mem.second;
  }
  return reserved;
}

std::string VerticalDataset::MemorySummary() const {
  uint64_t usage = 0;
  uint64_t reserved = 0;
  for (int col_idx = 0; col_idx < ncol(); col_idx++) {
    const auto col_mem = column(col_idx)->memory_usage();
    usage += col_mem.first;
    reserved += col_mem.second;
  }
  const uint64_t scale = 1e6;
  return absl::StrFormat("usage:%dMB allocated:%dMB", usage / scale,
                         reserved / scale);
}

void VerticalDataset::ShrinkToFit() {
  for (int col_idx = 0; col_idx < ncol(); col_idx++) {
    mutable_column(col_idx)->ShrinkToFit();
  }
}

absl::StatusOr<const VerticalDataset::CategoricalColumn*>
VerticalDataset::categorical_column(int col_idx) const {
  return ColumnWithCastWithStatus<dataset::VerticalDataset::CategoricalColumn>(
      col_idx);
}

absl::StatusOr<VerticalDataset::CategoricalColumn*>
VerticalDataset::mutable_categorical_column(int col_idx) {
  return MutableColumnWithCastWithStatus<
      dataset::VerticalDataset::CategoricalColumn>(col_idx);
}

absl::StatusOr<const VerticalDataset::NumericalColumn*>
VerticalDataset::numerical_column(int col_idx) const {
  return ColumnWithCastWithStatus<dataset::VerticalDataset::NumericalColumn>(
      col_idx);
}

absl::StatusOr<VerticalDataset::NumericalColumn*>
VerticalDataset::mutable_numerical_column(int col_idx) {
  return MutableColumnWithCastWithStatus<
      dataset::VerticalDataset::NumericalColumn>(col_idx);
}

absl::StatusOr<const VerticalDataset::BooleanColumn*>
VerticalDataset::boolean_column(int col_idx) const {
  return ColumnWithCastWithStatus<dataset::VerticalDataset::BooleanColumn>(
      col_idx);
}

absl::StatusOr<VerticalDataset::BooleanColumn*>
VerticalDataset::mutable_boolean_column(int col_idx) {
  return MutableColumnWithCastWithStatus<
      dataset::VerticalDataset::BooleanColumn>(col_idx);
}

absl::StatusOr<const VerticalDataset::DiscretizedNumericalColumn*>
VerticalDataset::discretized_numerical_column(int col_idx) const {
  return ColumnWithCastWithStatus<
      dataset::VerticalDataset::DiscretizedNumericalColumn>(col_idx);
}

absl::StatusOr<VerticalDataset::DiscretizedNumericalColumn*>
VerticalDataset::mutable_discretized_numerical_column(int col_idx) {
  return MutableColumnWithCastWithStatus<
      dataset::VerticalDataset::DiscretizedNumericalColumn>(col_idx);
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
