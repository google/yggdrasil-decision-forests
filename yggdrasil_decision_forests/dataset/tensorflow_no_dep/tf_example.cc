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

#include "yggdrasil_decision_forests/dataset/tensorflow_no_dep/tf_example.h"

#include <string>

#include "absl/strings/string_view.h"

namespace yggdrasil_decision_forests::tensorflow {

void SetFeatureValues(std::initializer_list<double> values,
                      absl::string_view key, tensorflow::Example* example) {
  auto& feature = (*example->mutable_features()->mutable_feature())[key];
  auto& dst_values = *feature.mutable_float_list()->mutable_value();
  dst_values.Clear();
  dst_values.Reserve(values.size());
  for (const auto value : values) {
    dst_values.Add(value);
  }
}

void SetFeatureValues(std::initializer_list<int> values, absl::string_view key,
                      tensorflow::Example* example) {
  auto& feature = (*example->mutable_features()->mutable_feature())[key];
  auto& dst_values = *feature.mutable_int64_list()->mutable_value();
  dst_values.Clear();
  dst_values.Reserve(values.size());
  for (const auto value : values) {
    dst_values.Add(value);
  }
}

void SetFeatureValues(std::initializer_list<absl::string_view> values,
                      absl::string_view key, tensorflow::Example* example) {
  auto& feature = (*example->mutable_features()->mutable_feature())[key];
  auto& dst_values = *feature.mutable_bytes_list()->mutable_value();
  dst_values.Clear();
  dst_values.Reserve(values.size());
  for (const auto& value : values) {
    dst_values.Add(std::string(value));
  }
}

}  // namespace yggdrasil_decision_forests::tensorflow