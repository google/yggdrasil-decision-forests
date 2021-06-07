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

#include "yggdrasil_decision_forests/serving/decision_forest/utils.h"

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace decision_forest {


// Get the list of input features used by the model.
//
// The order of the input feature is deterministic.
absl::Status GetInputFeatures(
    const model::AbstractModel& src_model,
    std::vector<int>* input_features_idxs,
    std::vector<int>* feature_idx_to_local_feature_idx) {
  // Map from the dataspec feature index to the node array feature index (i.e.
  // the inverse of "node_feature_idx_to_spec_feature_idx").
  if (feature_idx_to_local_feature_idx != nullptr) {
    feature_idx_to_local_feature_idx->assign(
        src_model.data_spec().columns_size(), -1);
  }

  std::unordered_map<int32_t, int64_t> feature_usage;
  const auto* rf_model =
      dynamic_cast<const model::random_forest::RandomForestModel*>(&src_model);
  const auto* gbt_model = dynamic_cast<
      const model::gradient_boosted_trees::GradientBoostedTreesModel*>(
      &src_model);
  if (rf_model) {
    rf_model->CountFeatureUsage(&feature_usage);
  } else if (gbt_model) {
    gbt_model->CountFeatureUsage(&feature_usage);
  } else {
    return absl::InvalidArgumentError("Unsupported decision forest model type");
  }

  input_features_idxs->clear();
  for (const auto& feature : feature_usage) {
    input_features_idxs->push_back(feature.first);
  }

  std::sort(input_features_idxs->begin(), input_features_idxs->end());

  for (int node_feature_idx = 0; node_feature_idx < input_features_idxs->size();
       ++node_feature_idx) {
    const int spec_feature_idx = (*input_features_idxs)[node_feature_idx];
    if (feature_idx_to_local_feature_idx != nullptr) {
      (*feature_idx_to_local_feature_idx)[spec_feature_idx] = node_feature_idx;
    }
  }

  if (input_features_idxs->empty()) {
    LOG(WARNING) << "The model does not have any input features i.e. the model "
                    "is constant and will always return the same prediction.";
  }

  return absl::OkStatus();
}

std::vector<NumericalOrCategoricalValue> FloatToValue(
    const std::vector<float>& values) {
  std::vector<NumericalOrCategoricalValue> result;
  result.reserve(values.size());
  for (const auto item : values) {
    result.push_back(NumericalOrCategoricalValue::Numerical(item));
  }
  return result;
}

}  // namespace decision_forest
}  // namespace serving
}  // namespace yggdrasil_decision_forests
