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

#include "yggdrasil_decision_forests/serving/tf_example.h"

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/tensorflow_no_dep/tf_example.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/serving/example_set.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace {

using FeaturesDefinition = FeaturesDefinitionNumericalOrCategoricalFlat;

absl::Status ParseBaseFeatureFromTfExample(
    const int example_idx, const serving::FeatureDef& feature_def,
    const absl::string_view feature_name,
    const tensorflow::Feature& feature_value,
    const serving::FeaturesDefinition& features,
    serving::AbstractExampleSet* dst) {
  switch (feature_def.type) {
    case dataset::proto::ColumnType::NUMERICAL:
    case dataset::proto::ColumnType::BOOLEAN:
    case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL: {
      switch (feature_value.kind_case()) {
        case tensorflow::Feature::KindCase::kFloatList: {
          const int value_size = feature_value.float_list().value_size();
          if (value_size == 1) {
            dst->SetNumerical(example_idx,
                              FeaturesDefinition::NumericalFeatureId{
                                  feature_def.internal_idx},
                              feature_value.float_list().value(0), features);
          } else if (value_size > 1) {
            return absl::InvalidArgumentError(
                absl::StrCat("Too many values for feature: ", feature_name));
          }
        } break;
        case tensorflow::Feature::KindCase::kInt64List: {
          const int value_size = feature_value.int64_list().value_size();
          if (value_size == 1) {
            dst->SetNumerical(example_idx,
                              FeaturesDefinition::NumericalFeatureId{
                                  feature_def.internal_idx},
                              feature_value.int64_list().value(0), features);
          } else if (value_size > 1) {
            return absl::InvalidArgumentError(
                absl::StrCat("Too many values for feature: ", feature_name));
          }
        } break;
        default:
          return absl::InvalidArgumentError(
              absl::StrCat("Feature ", feature_name, " is not numerical."));
      }
    } break;
    case dataset::proto::ColumnType::CATEGORICAL: {
      switch (feature_value.kind_case()) {
        case tensorflow::Feature::KindCase::kBytesList: {
          const int value_size = feature_value.bytes_list().value_size();
          if (value_size == 1) {
            dst->SetCategorical(example_idx,
                                FeaturesDefinition::CategoricalFeatureId{
                                    feature_def.internal_idx},
                                feature_value.bytes_list().value(0), features);
          } else if (value_size > 1) {
            return absl::InvalidArgumentError(
                absl::StrCat("Too many values for feature: ", feature_name));
          }
        } break;
        case tensorflow::Feature::KindCase::kInt64List: {
          const int value_size = feature_value.int64_list().value_size();
          if (value_size == 1) {
            dst->SetCategorical(example_idx,
                                FeaturesDefinition::CategoricalFeatureId{
                                    feature_def.internal_idx},
                                feature_value.int64_list().value(0), features);
          } else if (value_size > 1) {
            return absl::InvalidArgumentError(
                absl::StrCat("Too many values for feature: ", feature_name));
          }
        } break;
        default:
          return absl::InvalidArgumentError(
              absl::StrCat("Feature ", feature_name, " is not categorical."));
      }
    } break;
    case dataset::proto::ColumnType::CATEGORICAL_SET: {
      switch (feature_value.kind_case()) {
        case tensorflow::Feature::KindCase::kBytesList: {
          if (feature_value.bytes_list().value_size() > 0) {
            std::vector<std::string> values_copy(
                feature_value.bytes_list().value().begin(),
                feature_value.bytes_list().value().end());
            dst->SetCategoricalSet(example_idx,
                                   FeaturesDefinition::CategoricalSetFeatureId{
                                       feature_def.internal_idx},
                                   values_copy, features);
          }
        } break;
        case tensorflow::Feature::KindCase::kInt64List: {
          if (feature_value.int64_list().value_size() > 0) {
            std::vector<int> values_copy(
                feature_value.int64_list().value().begin(),
                feature_value.int64_list().value().end());
            dst->SetCategoricalSet(example_idx,
                                   FeaturesDefinition::CategoricalSetFeatureId{
                                       feature_def.internal_idx},
                                   values_copy.cbegin(), values_copy.cend(),
                                   features);
          }
        } break;
        default:
          return absl::InvalidArgumentError(absl::StrCat(
              "Feature ", feature_name, " is not a categorical set."));
      }
    } break;
    default:
      return absl::InvalidArgumentError("Non supported feature type.");
  }
  return absl::OkStatus();
}

absl::Status ParseUnstackedFeatureFromTfExample(
    const int example_idx, const serving::UnstackedFeature& feature_def,
    const absl::string_view feature_name,
    const tensorflow::Feature& feature_value,
    const serving::FeaturesDefinition& features,
    serving::AbstractExampleSet* dst) {
  switch (feature_value.kind_case()) {
    case tensorflow::Feature::KindCase::kFloatList: {
      return dst->SetMultiDimNumerical(
          example_idx,
          FeaturesDefinition::MultiDimNumericalFeatureId{
              feature_def.unstacked_index},
          feature_value.float_list().value(), features);
    }

    case tensorflow::Feature::KindCase::kInt64List: {
      std::vector<float> float_values = {
          feature_value.int64_list().value().begin(),
          feature_value.int64_list().value().end()};
      return dst->SetMultiDimNumerical(
          example_idx,
          FeaturesDefinition::MultiDimNumericalFeatureId{
              feature_def.unstacked_index},
          float_values, features);
    }

    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Feature ", feature_name, " is not numerical."));
  }
}

}  // namespace

absl::Status TfExampleToExampleSet(const tensorflow::Example& src,
                                   const int example_idx,
                                   const serving::FeaturesDefinition& features,
                                   serving::AbstractExampleSet* dst) {
  // Iterate over the source features.
  for (const auto& fname_and_value : src.features().feature()) {
    // If the feature is not used by any of the "Parse*" function, this
    // indicates that the feature is not used by the model and ignored.

    const auto unstacked_feature_def =
        features.FindUnstackedFeatureDefByName(fname_and_value.first);
    if (unstacked_feature_def.ok()) {
      // Parse the unstacked feature.
      RETURN_IF_ERROR(ParseUnstackedFeatureFromTfExample(
          example_idx, *unstacked_feature_def.value(), fname_and_value.first,
          fname_and_value.second, features, dst));
    } else {
      const auto base_feature_def =
          features.FindFeatureDefByName(fname_and_value.first);
      if (base_feature_def.ok()) {
        // Parse the base feature.
        RETURN_IF_ERROR(ParseBaseFeatureFromTfExample(
            example_idx, *base_feature_def.value(), fname_and_value.first,
            fname_and_value.second, features, dst));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace serving
}  // namespace yggdrasil_decision_forests
