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

#include "yggdrasil_decision_forests/serving/example_set.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace serving {

using dataset::proto::ColumnType;

// Creates the default value for a numerical only model.
template <>
absl::StatusOr<float> GetDefaultValue<float>(
    const dataset::proto::Column& col_spec,
    const bool missing_numerical_is_na) {
  if (col_spec.type() == ColumnType::NUMERICAL ||
      col_spec.type() == ColumnType::DISCRETIZED_NUMERICAL) {
    if (missing_numerical_is_na) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return col_spec.numerical().mean();
  } else if (col_spec.type() == ColumnType::BOOLEAN) {
    // Note: Boolean can only be compared with 0.5f.
    // The {2,-1} values (instead of the common {1,0}) make it possible to
    // detect missing values in serialization+deserialization.
    // Note: That serialization+deserialization is not guaranteed to produce
    // the same values; only to give the same predictions through a model.
    return (col_spec.boolean().count_true() >= col_spec.boolean().count_false())
               ? 2.f
               : -1.f;
  }
  return absl::InvalidArgumentError(
      absl::StrCat(col_spec.name(), " is not numerical."));
}

// Creates the default value for a numerical or categorical only model.
template <>
absl::StatusOr<NumericalOrCategoricalValue>
GetDefaultValue<NumericalOrCategoricalValue>(
    const dataset::proto::Column& col_spec,
    const bool missing_numerical_is_na) {
  switch (col_spec.type()) {
    case ColumnType::NUMERICAL:
    case ColumnType::DISCRETIZED_NUMERICAL:
      if (missing_numerical_is_na) {
        return NumericalOrCategoricalValue::Numerical(
            std::numeric_limits<float>::quiet_NaN());
      }
      return NumericalOrCategoricalValue::Numerical(
          col_spec.numerical().mean());
    case ColumnType::BOOLEAN:
      return NumericalOrCategoricalValue::Boolean(
          col_spec.boolean().count_true() >= col_spec.boolean().count_false());
    case ColumnType::CATEGORICAL:
      return NumericalOrCategoricalValue::Categorical(
          col_spec.categorical().most_frequent_value());
    default:
      return absl::InvalidArgumentError(
          absl::StrCat(col_spec.name(), " is not numerical nor categorical."));
  }
}

const FeatureDef* FindFeatureDefFromInternalIndex(
    const std::vector<FeatureDef>& defs, const int internal_index) {
  for (const auto& def : defs) {
    if (def.internal_idx == internal_index) {
      return &def;
    }
  }
  return nullptr;
}

std::vector<std::string> FeatureNames(const std::vector<FeatureDef>& defs) {
  std::vector<std::string> names;
  names.reserve(defs.size());
  for (const auto& feature : defs) {
    names.push_back(feature.name);
  }
  return names;
}

std::vector<FeatureDef>
FeaturesDefinitionNumericalOrCategoricalFlat::input_features() const {
  std::vector<FeatureDef> features;
  features.insert(features.end(), fixed_length_features_.begin(),
                  fixed_length_features_.end());
  features.insert(features.end(), categorical_set_features_.begin(),
                  categorical_set_features_.end());
  features.insert(features.end(), numerical_vector_sequence_features_.begin(),
                  numerical_vector_sequence_features_.end());
  return features;
}

const std::vector<UnstackedFeature>&
FeaturesDefinitionNumericalOrCategoricalFlat::unstacked_features() const {
  return unstacked_features_;
}

bool FeaturesDefinitionNumericalOrCategoricalFlat::HasInputFeature(
    const absl::string_view name) const {
  return feature_def_cache_.find(std::string(name)) !=
             feature_def_cache_.end() ||
         indexed_unstacked_features_.find(std::string(name)) !=
             indexed_unstacked_features_.end();
}

absl::Status FeaturesDefinitionNumericalOrCategoricalFlat::Initialize(
    const std::vector<int>& input_features, const DataSpecification& dataspec,
    const bool missing_numerical_is_na) {
  column_input_features_ = input_features;
  RETURN_IF_ERROR(InitializeNormalFeatures(input_features, dataspec,
                                           missing_numerical_is_na));
  RETURN_IF_ERROR(InitializeUnstackedFeatures(input_features, dataspec,
                                              missing_numerical_is_na));

  // At this point, feature definitions (e.g. fixed_length_features) should not
  // be modified anymore.

  // Index of the feature definitions by name.
  for (const auto& feature_def : fixed_length_features()) {
    feature_def_cache_[feature_def.name] = &feature_def;
  }
  for (const auto& feature_def : categorical_set_features()) {
    feature_def_cache_[feature_def.name] = &feature_def;
  }
  for (const auto& feature_def : numerical_vector_sequence_features()) {
    feature_def_cache_[feature_def.name] = &feature_def;
  }

  return absl::OkStatus();
}

absl::Status
FeaturesDefinitionNumericalOrCategoricalFlat::InitializeNormalFeatures(
    const std::vector<int>& input_features, const DataSpecification& dataspec,
    const bool missing_numerical_is_na) {
  data_spec_ = dataspec;

  // Index the input features.
  for (const int spec_feature_idx : input_features) {
    const auto& col_spec = data_spec_.columns(spec_feature_idx);

    if (col_spec.is_unstacked()) {
      continue;
    }

    switch (col_spec.type()) {
      case ColumnType::NUMERICAL:
      case ColumnType::DISCRETIZED_NUMERICAL:
      case ColumnType::CATEGORICAL:
      case ColumnType::BOOLEAN: {
        const int internal_idx = fixed_length_features_.size();
        fixed_length_features_.push_back({/*.name =*/col_spec.name(),
                                          /*.type =*/col_spec.type(),
                                          /*.spec_idx =*/spec_feature_idx,
                                          /*.internal_idx =*/internal_idx});
        ASSIGN_OR_RETURN(auto default_value,
                         GetDefaultValue<NumericalOrCategoricalValue>(
                             col_spec, missing_numerical_is_na));
        fixed_length_feature_missing_values_.push_back(default_value);
      } break;
      case ColumnType::CATEGORICAL_SET: {
        const int internal_idx = categorical_set_features_.size();
        categorical_set_features_.push_back({/*.name =*/col_spec.name(),
                                             /*.type =*/col_spec.type(),
                                             /*.spec_idx =*/spec_feature_idx,
                                             /*.internal_idx =*/internal_idx});
      } break;
      case ColumnType::NUMERICAL_VECTOR_SEQUENCE: {
        const int internal_idx = numerical_vector_sequence_features_.size();
        numerical_vector_sequence_features_.push_back({
            {/*.name =*/col_spec.name(),
             /*.type =*/col_spec.type(),
             /*.spec_idx =*/spec_feature_idx,
             /*.internal_idx =*/internal_idx},
            /*.vector_length =*/
            col_spec.numerical_vector_sequence().vector_length(),
        });
      } break;
      default:
        return absl::InvalidArgumentError(
            absl::Substitute("Unsupported feature type $0",
                             dataset::proto::ColumnType_Name(col_spec.type())));
    }
  }

  return absl::OkStatus();
}

absl::Status
FeaturesDefinitionNumericalOrCategoricalFlat::InitializeUnstackedFeatures(
    const std::vector<int>& input_features, const DataSpecification& dataspec,
    const bool missing_numerical_is_na) {
  // List the sub-set of input features which are unstacked.
  std::vector<int> unstacked_input_features;
  for (const int spec_feature_idx : input_features) {
    const auto& col_spec = data_spec_.columns(spec_feature_idx);
    if (col_spec.is_unstacked()) {
      unstacked_input_features.push_back(spec_feature_idx);
    }
  }

  // Index the unstacked features.
  for (const auto& unstacked : data_spec_.unstackeds()) {
    // Check if the unstacked is used by the model.
    const bool used =
        std::find_if(unstacked_input_features.begin(),
                     unstacked_input_features.end(),
                     [&](const int feature_idx) {
                       return feature_idx >= unstacked.begin_column_idx() &&
                              feature_idx < unstacked.begin_column_idx() +
                                                unstacked.size();
                     }) != unstacked_input_features.end();
    if (!used) {
      continue;
    }

    // Index the unstacking information.
    const int begin_internal_idx = fixed_length_features_.size();
    const int unstacked_index = unstacked_features_.size();
    indexed_unstacked_features_[unstacked.original_name()] = unstacked_index;
    unstacked_features_.push_back(
        {/*.begin_internal_idx =*/begin_internal_idx,
         /*.begin_spec_idx =*/unstacked.begin_column_idx(),
         /*.size =*/unstacked.size(),
         /*.unstacked_index =*/unstacked_index});

    // Index the unstacked units.
    for (int dim_idx = 0; dim_idx < unstacked.size(); dim_idx++) {
      const int spec_feature_idx = unstacked.begin_column_idx() + dim_idx;
      const auto& col_spec = data_spec_.columns(spec_feature_idx);
      if (!col_spec.is_unstacked()) {
        return absl::InternalError("Unexpected non-unstacked feature.");
      }
      fixed_length_features_.push_back(
          {/*.name =*/col_spec.name(),
           /*.type =*/col_spec.type(),
           /*.spec_idx =*/spec_feature_idx,
           /*.internal_idx =*/begin_internal_idx + dim_idx});
      ASSIGN_OR_RETURN(auto default_value,
                       GetDefaultValue<NumericalOrCategoricalValue>(
                           col_spec, missing_numerical_is_na));
      fixed_length_feature_missing_values_.push_back(default_value);
    }
  }

  return absl::OkStatus();
}

std::ostream& operator<<(std::ostream& os, const FeatureDef& feature) {
  os << "\"" << feature.name << "\" type:" << ColumnType_Name(feature.type)
     << " spec_idx:" << feature.spec_idx
     << " internal_idx:" << feature.internal_idx;
  return os;
}

absl::Status CopyVerticalDatasetToAbstractExampleSet(
    const dataset::VerticalDataset& dataset,
    const dataset::VerticalDataset::row_t begin_example_idx,
    const dataset::VerticalDataset::row_t end_example_idx,
    const FeaturesDefinition& features, AbstractExampleSet* examples) {
  using row_t = dataset::VerticalDataset::row_t;
  using NumericalColumn = dataset::VerticalDataset::NumericalColumn;
  using BooleanColumn = dataset::VerticalDataset::BooleanColumn;
  using DiscretizedNumericalColumn =
      dataset::VerticalDataset::DiscretizedNumericalColumn;
  using CategoricalColumn = dataset::VerticalDataset::CategoricalColumn;
  using CategoricalSetColumn = dataset::VerticalDataset::CategoricalSetColumn;
  using NumericalVectorSequenceColumn =
      dataset::VerticalDataset::NumericalVectorSequenceColumn;

  const auto num_examples = end_example_idx - begin_example_idx;

  const auto CopyNumericalFeature = [&](const int feature_idx) -> absl::Status {
    ASSIGN_OR_RETURN(
        const auto& feature_data,
        dataset.ColumnWithCastWithStatus<NumericalColumn>(feature_idx));
    ASSIGN_OR_RETURN(const auto feature_id,
                     features.GetNumericalFeatureId(feature_data->name()));
    for (int example_idx = 0; example_idx < num_examples; example_idx++) {
      const row_t row_idx = begin_example_idx + example_idx;
      if (feature_data->IsNa(row_idx)) {
        examples->SetMissingNumerical(example_idx, feature_id, features);
      } else {
        examples->SetNumerical(example_idx, feature_id,
                               feature_data->values()[row_idx], features);
      }
    }
    return absl::OkStatus();
  };

  const auto CopyBooleanFeature = [&](const int feature_idx) -> absl::Status {
    ASSIGN_OR_RETURN(
        const auto& feature_data,
        dataset.ColumnWithCastWithStatus<BooleanColumn>(feature_idx));
    ASSIGN_OR_RETURN(const auto feature_id,
                     features.GetBooleanFeatureId(feature_data->name()));
    for (int example_idx = 0; example_idx < num_examples; example_idx++) {
      const row_t row_idx = begin_example_idx + example_idx;
      if (feature_data->IsNa(row_idx)) {
        examples->SetMissingBoolean(example_idx, feature_id, features);
      } else {
        examples->SetBoolean(example_idx, feature_id,
                             feature_data->values()[row_idx], features);
      }
    }
    return absl::OkStatus();
  };

  const auto CopyDiscretizedNumericalFeature =
      [&](const int feature_idx) -> absl::Status {
    ASSIGN_OR_RETURN(
        const auto& feature_data,
        dataset.ColumnWithCastWithStatus<DiscretizedNumericalColumn>(
            feature_idx));
    ASSIGN_OR_RETURN(const auto feature_id,
                     features.GetNumericalFeatureId(feature_data->name()));
    for (int example_idx = 0; example_idx < num_examples; example_idx++) {
      const row_t row_idx = begin_example_idx + example_idx;
      if (feature_data->IsNa(row_idx)) {
        examples->SetMissingNumerical(example_idx, feature_id, features);
      } else {
        ASSIGN_OR_RETURN(auto value,
                         dataset::DiscretizedNumericalToNumerical(
                             features.data_spec().columns(feature_idx),
                             feature_data->values()[row_idx]));
        examples->SetNumerical(example_idx, feature_id, value, features);
      }
    }
    return absl::OkStatus();
  };

  const auto CopyCategoricalFeature =
      [&](const int feature_idx) -> absl::Status {
    ASSIGN_OR_RETURN(
        const auto& feature_data,
        dataset.ColumnWithCastWithStatus<CategoricalColumn>(feature_idx));
    ASSIGN_OR_RETURN(const auto feature_id,
                     features.GetCategoricalFeatureId(feature_data->name()));
    for (int example_idx = 0; example_idx < num_examples; example_idx++) {
      const row_t row_idx = begin_example_idx + example_idx;
      if (feature_data->IsNa(row_idx)) {
        examples->SetMissingCategorical(example_idx, feature_id, features);
      } else {
        examples->SetCategorical(example_idx, feature_id,
                                 feature_data->values()[row_idx], features);
      }
    }
    return absl::OkStatus();
  };

  const auto CopyCategoricalSetFeature =
      [&](const int feature_idx) -> absl::Status {
    ASSIGN_OR_RETURN(
        const auto& feature_data,
        dataset.ColumnWithCastWithStatus<CategoricalSetColumn>(feature_idx));
    ASSIGN_OR_RETURN(const auto feature_id,
                     features.GetCategoricalSetFeatureId(feature_data->name()));
    for (int example_idx = 0; example_idx < num_examples; example_idx++) {
      const row_t row_idx = begin_example_idx + example_idx;
      if (feature_data->IsNa(row_idx)) {
        examples->SetMissingCategoricalSet(example_idx, feature_id, features);
      } else {
        std::vector<int32_t> values = {feature_data->begin(row_idx),
                                       feature_data->end(row_idx)};
        examples->SetCategoricalSet(example_idx, feature_id, values, features);
      }
    }
    return absl::OkStatus();
  };

  const auto CopyNumericalVectorSequenceFeature =
      [&](const int feature_idx) -> absl::Status {
    ASSIGN_OR_RETURN(
        const auto& feature_data,
        dataset.ColumnWithCastWithStatus<NumericalVectorSequenceColumn>(
            feature_idx));
    ASSIGN_OR_RETURN(
        const auto feature_id,
        features.GetNumericalVectorSequenceFeatureId(feature_data->name()));
    for (int example_idx = 0; example_idx < num_examples; example_idx++) {
      const row_t row_idx = begin_example_idx + example_idx;
      if (feature_data->IsNa(row_idx)) {
        examples->SetMissingNumericalVectorSequence(example_idx, feature_id,
                                                    features);
      } else {
        std::vector<float> values;
        const int num_vectors = feature_data->SequenceLength(row_idx);
        for (int vector_idx = 0; vector_idx < num_vectors; vector_idx++) {
          const auto vector = feature_data->GetVector(row_idx, vector_idx);
          values.insert(values.end(), vector->begin(), vector->end());
        }
        examples->SetNumericalVectorSequence(example_idx, feature_id, values,
                                             features);
      }
    }
    return absl::OkStatus();
  };

  for (const auto& feature : features.input_features()) {
    if (feature.spec_idx < 0 || feature.spec_idx >= dataset.ncol()) {
      return absl::InvalidArgumentError(
          absl::Substitute("Invalid feature index $0. Feature indices should"
                           "be between 0 and $1",
                           feature.spec_idx, dataset.ncol()));
    }
    if (num_examples > 0 && dataset.column(feature.spec_idx)->nrows() == 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Feature \"", feature.name, "\" is empty. Cannot extract it."));
    }

    switch (feature.type) {
      case dataset::proto::ColumnType::NUMERICAL:
        RETURN_IF_ERROR(CopyNumericalFeature(feature.spec_idx));
        break;
      case dataset::proto::ColumnType::BOOLEAN:
        RETURN_IF_ERROR(CopyBooleanFeature(feature.spec_idx));
        break;
      case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL:
        RETURN_IF_ERROR(CopyDiscretizedNumericalFeature(feature.spec_idx));
        break;
      case dataset::proto::ColumnType::CATEGORICAL:
        RETURN_IF_ERROR(CopyCategoricalFeature(feature.spec_idx));
        break;
      case dataset::proto::ColumnType::CATEGORICAL_SET:
        RETURN_IF_ERROR(CopyCategoricalSetFeature(feature.spec_idx));
        break;
      case dataset::proto::ColumnType::NUMERICAL_VECTOR_SEQUENCE:
        RETURN_IF_ERROR(CopyNumericalVectorSequenceFeature(feature.spec_idx));
        break;
      default:
        return absl::InvalidArgumentError("Non supported feature type.");
    }
  }
  return absl::OkStatus();
}

}  // namespace serving
}  // namespace yggdrasil_decision_forests
