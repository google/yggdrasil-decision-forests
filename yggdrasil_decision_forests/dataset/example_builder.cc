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

#include "yggdrasil_decision_forests/dataset/example_builder.h"

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"

namespace yggdrasil_decision_forests::dataset {

ExampleProtoBuilder::ExampleProtoBuilder(
    const proto::DataSpecification* data_spec)
    : data_spec_(data_spec) {
  // Allocate the example.
  for (size_t i = 0; i < data_spec_->columns_size(); i++) {
    example_.add_attributes();
  }
}

void ExampleProtoBuilder::SetNumericalValue(int col_idx, float value) {
  CheckColumn(col_idx, proto::ColumnType::NUMERICAL);
  example_.mutable_attributes(col_idx)->set_numerical(value);
}

void ExampleProtoBuilder::SetDiscretizedNumericalValue(int col_idx,
                                                       int value_idx) {
  CheckColumn(col_idx, proto::ColumnType::DISCRETIZED_NUMERICAL);
  example_.mutable_attributes(col_idx)->set_discretized_numerical(value_idx);
}

void ExampleProtoBuilder::SetCategoricalValue(int col_idx, int value) {
  CheckColumn(col_idx, proto::ColumnType::CATEGORICAL);
  example_.mutable_attributes(col_idx)->set_categorical(value);
}

void ExampleProtoBuilder::SetCategoricalValue(int col_idx,
                                              absl::string_view value) {
  CheckColumn(col_idx, proto::ColumnType::CATEGORICAL);
  const int int_value = NonintegerizedCategoricalStringToValue(
      value, data_spec_->columns(col_idx));
  example_.mutable_attributes(col_idx)->set_categorical(int_value);
}

void ExampleProtoBuilder::SetBooleanValue(int col_idx, bool value) {
  CheckColumn(col_idx, proto::ColumnType::BOOLEAN);
  example_.mutable_attributes(col_idx)->set_boolean(value);
}

void ExampleProtoBuilder::SetCategoricalSetValue(int col_idx,
                                                 absl::Span<const int> values) {
  CheckColumn(col_idx, proto::ColumnType::CATEGORICAL_SET);
  *example_.mutable_attributes(col_idx)
       ->mutable_categorical_set()
       ->mutable_values() = {values.begin(), values.end()};
}

void ExampleProtoBuilder::SetCategoricalSetValue(
    int col_idx, absl::Span<const absl::string_view> values) {
  CheckColumn(col_idx, proto::ColumnType::CATEGORICAL_SET);
  for (const absl::string_view value : values) {
    const int int_value = NonintegerizedCategoricalStringToValue(
        value, data_spec_->columns(col_idx));
    example_.mutable_attributes(col_idx)->mutable_categorical_set()->add_values(
        int_value);
  }
}

}  // namespace yggdrasil_decision_forests::dataset
