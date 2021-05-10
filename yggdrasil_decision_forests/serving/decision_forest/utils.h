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

// Utility function for the serving of decision forests.
//
#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_UTILS_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_UTILS_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace decision_forest {

// Initialize the "feature_name", and "na_replacement_values" fields of a
// "FlatNodeModel".
template <typename GenericModel, typename SpecializedModel>
absl::Status InitializeFlatNodeModel(const GenericModel& src_model,
                                     SpecializedModel* dst_model);

// Get the list of input features used by the model.
absl::Status GetInputFeatures(
    const model::AbstractModel& src_model,
    std::vector<int>* input_features_idxs,
    std::vector<int>* feature_idx_to_local_feature_idx);

// Converts a feature name into the model's internal feature index. Returns "-1"
// if the model does not use this feature.
template <typename SpecializedModel>
int GetSpecializedModelFeatureIdx(const SpecializedModel& model,
                                  absl::string_view feature_name);

// Returns a mapping of feature_name -> model's internal feature index.
template <typename SpecializedModel>
absl::flat_hash_map<std::string, int> GetFeatureIndexMap(
    const SpecializedModel& model);

// =======================================
//   Below are the template definitions.
// =======================================

template <typename GenericModel, typename SpecializedModel>
absl::Status InitializeFlatNodeModel(const GenericModel& src_model,
                                     SpecializedModel* dst_model) {
  // List the model input features.
  std::vector<int> all_input_features;
  RETURN_IF_ERROR(GetInputFeatures(src_model, &all_input_features, nullptr));

  RETURN_IF_ERROR(dst_model->mutable_features()->Initialize(
      all_input_features, src_model.data_spec()));

  return absl::OkStatus();
}

template <typename SpecializedModel>
int GetSpecializedModelFeatureIdx(const SpecializedModel& model,
                                  absl::string_view feature_name) {
  const auto feature_it = std::find(model.feature_names.begin(),
                                    model.feature_names.end(), feature_name);
  if (feature_it == model.feature_names.end()) {
    return -1;
  }
  return static_cast<int>(
      std::distance(model.feature_names.begin(), feature_it));
}

template <typename SpecializedModel>
absl::flat_hash_map<std::string, int> GetFeatureIndexMap(
    const SpecializedModel& model) {
  absl::flat_hash_map<std::string, int> feature_index_map;
  for (int local_feature_idx = 0;
       local_feature_idx < model.feature_names.size(); ++local_feature_idx) {
    feature_index_map[model.feature_names[local_feature_idx]] =
        local_feature_idx;
  }
  return feature_index_map;
}

}  // namespace decision_forest
}  // namespace serving
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_DECISION_FOREST_UTILS_H_
