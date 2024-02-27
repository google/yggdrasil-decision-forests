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

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_PROTO_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_PROTO_H_

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests::dataset {

// Utility to create a ydf::dataset::proto::Example.
class ExampleProtoBuilder {
 public:
  ExampleProtoBuilder(const proto::DataSpecification* data_spec);

  // Get the proto::Example.
  const proto::Example& example() const { return example_; }

  // Set attribute values.
  void SetNumericalValue(int col_idx, float value);
  void SetDiscretizedNumericalValue(int col_idx, int value_idx);
  void SetCategoricalValue(int col_idx, int value);
  void SetCategoricalValue(int col_idx, absl::string_view value);
  void SetBooleanValue(int col_idx, bool value);
  void SetCategoricalSetValue(int col_idx, absl::Span<const int> values);
  void SetCategoricalSetValue(int col_idx,
                              absl::Span<const absl::string_view> values);

 private:
  void CheckColumn(int col_idx, const proto::ColumnType type) {
    DCHECK_GE(col_idx, 0);
    DCHECK_LT(col_idx, data_spec_->columns_size());
    DCHECK_EQ(data_spec_->columns(col_idx).type(), type);
  }

  const proto::DataSpecification* data_spec_;  // Non owned.
  proto::Example example_;
};

}  // namespace yggdrasil_decision_forests::dataset

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_EXAMPLE_PROTO_H_
