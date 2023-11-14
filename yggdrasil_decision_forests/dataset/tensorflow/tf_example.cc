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

#include "yggdrasil_decision_forests/dataset/tensorflow/tf_example.h"

#include <stdint.h>

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "yggdrasil_decision_forests/dataset/tensorflow_no_dep/tf_example.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/serving/tf_example.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace dataset {
namespace {
using proto::ColumnType;
}  // namespace

absl::Status TfExampleToYdfExample(const tensorflow::Example& tf_example,
                                   const proto::DataSpecification& data_spec,
                                   proto::Example* example) {
  example->mutable_attributes()->Clear();
  example->mutable_attributes()->Reserve(data_spec.columns_size());
  for (int col_idx = 0; col_idx < data_spec.columns_size(); col_idx++) {
    example->mutable_attributes()->Add();
  }

  // Stacked columns.
  for (const auto& unstacked : data_spec.unstackeds()) {
    const auto it_feature =
        tf_example.features().feature().find(unstacked.original_name());
    if (it_feature == tf_example.features().feature().end() ||
        it_feature->second.kind_case() ==
            tensorflow::Feature::KindCase::KIND_NOT_SET) {
      // NAs
      continue;
    }

    switch (it_feature->second.kind_case()) {
      case tensorflow::Feature::KindCase::kFloatList: {
        const auto& feature_value = it_feature->second.float_list();
        if (feature_value.value_size() != unstacked.size()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Wrong number of elements for feature %s", it_feature->first));
        }
        for (int dim_idx = 0; dim_idx < unstacked.size(); dim_idx++) {
          const auto col_idx = unstacked.begin_column_idx() + dim_idx;
          const auto& col_spec = data_spec.columns(col_idx);
          switch (unstacked.type()) {
            case proto::NUMERICAL:
              example->mutable_attributes(col_idx)->set_numerical(
                  feature_value.value(dim_idx));
              break;
            case proto::DISCRETIZED_NUMERICAL:
              example->mutable_attributes(col_idx)->set_discretized_numerical(
                  NumericalToDiscretizedNumerical(
                      col_spec, feature_value.value(dim_idx)));
              break;
            default:
              return absl::InvalidArgumentError(absl::StrFormat(
                  "%s's type is not supported for stacked feature.",
                  it_feature->first));
          }
        }
      } break;

      case tensorflow::Feature::KindCase::kInt64List: {
        const auto& feature_value = it_feature->second.int64_list();
        if (feature_value.value_size() != unstacked.size()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Wrong number of elements for feature %s", it_feature->first));
        }
        for (int dim_idx = 0; dim_idx < unstacked.size(); dim_idx++) {
          const int col_idx = unstacked.begin_column_idx() + dim_idx;
          const auto& col_spec = data_spec.columns(col_idx);
          switch (unstacked.type()) {
            case proto::NUMERICAL:
              example->mutable_attributes(col_idx)->set_numerical(
                  feature_value.value(dim_idx));
              break;
            case proto::DISCRETIZED_NUMERICAL:
              example->mutable_attributes(col_idx)->set_discretized_numerical(
                  NumericalToDiscretizedNumerical(
                      col_spec, feature_value.value(dim_idx)));
              break;
            default:
              return absl::InvalidArgumentError(absl::StrFormat(
                  "%s's type is not supported for stacked feature.",
                  it_feature->first));
          }
        }
      } break;

      default:
        return absl::InvalidArgumentError(absl::StrFormat(
            "Feature %s is not stored as float or int64.", it_feature->first));
    }
  }

  // Base columns.
  for (int col_idx = 0; col_idx < data_spec.columns_size(); col_idx++) {
    const auto& col_spec = data_spec.columns(col_idx);
    if (col_spec.is_unstacked()) {
      continue;
    }
    auto* dst_value = example->mutable_attributes(col_idx);
    const auto it_feature =
        tf_example.features().feature().find(col_spec.name());
    if (it_feature == tf_example.features().feature().end() ||
        it_feature->second.kind_case() ==
            tensorflow::Feature::KindCase::KIND_NOT_SET) {
      // NAs
      continue;
    }

    switch (col_spec.type()) {
      case ColumnType::UNKNOWN:
        break;
      case ColumnType::NUMERICAL: {
        ASSIGN_OR_RETURN(const float num_value,
                         internal::GetSingleFloatFromTFFeature(
                             it_feature->second, col_spec));
        dst_value->set_numerical(num_value);
      } break;
      case ColumnType::DISCRETIZED_NUMERICAL: {
        ASSIGN_OR_RETURN(const float num_value,
                         internal::GetSingleFloatFromTFFeature(
                             it_feature->second, col_spec));
        dst_value->set_discretized_numerical(
            NumericalToDiscretizedNumerical(col_spec, num_value));
      } break;
      case ColumnType::NUMERICAL_SET:
      case ColumnType::NUMERICAL_LIST: {
        std::vector<float> values;
        RETURN_IF_ERROR(internal::GetNumericalValuesFromTFFeature(
            it_feature->second, col_spec, &values));

        google::protobuf::RepeatedField<float>* dst;
        if (col_spec.type() == ColumnType::NUMERICAL_SET) {
          dst = dst_value->mutable_numerical_set()->mutable_values();
        } else {
          dst = dst_value->mutable_numerical_list()->mutable_values();
        }
        dst->Reserve(values.size());
        for (const float& value : values) {
          dst->Add(value);
        }
        if (col_spec.type() == ColumnType::NUMERICAL_SET) {
          // Sets are expected to be sorted.
          std::sort(dst->begin(), dst->end());
          dst->erase(std::unique(dst->begin(), dst->end()), dst->end());
        }
      } break;
      case ColumnType::CATEGORICAL: {
        std::vector<std::string> tokens;
        RETURN_IF_ERROR(internal::GetCategoricalTokensFromTFFeature(
            it_feature->second, col_spec, &tokens));
        if (tokens.empty()) {
          // NA.
        } else if (tokens.size() == 1) {
          ASSIGN_OR_RETURN(auto value, CategoricalStringToValueWithStatus(
                                           tokens[0], col_spec));
          dst_value->set_categorical(value);
        } else {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Categorical attribute with more than one value for feature %s",
              it_feature->first));
        }
      } break;
      case ColumnType::CATEGORICAL_SET:
      case ColumnType::CATEGORICAL_LIST: {
        std::vector<std::string> tokens;
        RETURN_IF_ERROR(internal::GetCategoricalTokensFromTFFeature(
            it_feature->second, col_spec, &tokens));

        google::protobuf::RepeatedField<int32_t>* dst;
        if (col_spec.type() == ColumnType::CATEGORICAL_SET) {
          dst = dst_value->mutable_categorical_set()->mutable_values();
        } else {
          dst = dst_value->mutable_categorical_list()->mutable_values();
        }

        dst->Reserve(tokens.size());
        for (const std::string& token : tokens) {
          ASSIGN_OR_RETURN(auto value,
                           CategoricalStringToValueWithStatus(token, col_spec));
          dst->Add(value);
        }
        if (col_spec.type() == ColumnType::CATEGORICAL_SET) {
          // Sets are expected to be sorted.
          std::sort(dst->begin(), dst->end());
          dst->erase(std::unique(dst->begin(), dst->end()), dst->end());
        }
      } break;
      case ColumnType::BOOLEAN: {
        ASSIGN_OR_RETURN(const float num_value,
                         internal::GetSingleFloatFromTFFeature(
                             it_feature->second, col_spec));
        dst_value->set_boolean(num_value >= 0.5f);
      } break;
      case ColumnType::STRING:
        STATUS_CHECK_EQ(it_feature->second.kind_case(),
                        tensorflow::Feature::KindCase::kBytesList);
        if (it_feature->second.bytes_list().value().empty()) {
          // NA
        } else if (it_feature->second.bytes_list().value().size() == 1) {
          *dst_value->mutable_text() =
              it_feature->second.bytes_list().value()[0];
        } else {
          return absl::InvalidArgumentError(absl::StrFormat(
              "String attribute with more than one value for feature %s",
              it_feature->first));
        }
        break;

      case ColumnType::HASH: {
        std::vector<std::string> tokens;
        RETURN_IF_ERROR(internal::GetCategoricalTokensFromTFFeature(
            it_feature->second, col_spec, &tokens));
        if (tokens.empty()) {
          // NA.
        } else if (tokens.size() == 1) {
          dst_value->set_hash(HashColumnString(tokens[0]));
        } else {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Hash attribute with more than one value for feature %s",
              it_feature->first));
        }
      } break;
    }
  }
  return absl::OkStatus();
}

absl::Status YdfExampleToTfExample(const proto::Example& example,
                                   const proto::DataSpecification& data_spec,
                                   tensorflow::Example* tf_example) {
  tf_example->clear_features();

  for (const auto& unstacked : data_spec.unstackeds()) {
    auto& dst_value = (*tf_example->mutable_features()
                            ->mutable_feature())[unstacked.original_name()];
    for (int dim_idx = 0; dim_idx < unstacked.size(); dim_idx++) {
      const int col_idx = unstacked.begin_column_idx() + dim_idx;
      const auto& src_value = example.attributes(col_idx);
      const auto& col_spec = data_spec.columns(col_idx);
      switch (unstacked.type()) {
        case proto::NUMERICAL:
          dst_value.mutable_float_list()->add_value(src_value.numerical());
          break;
        case proto::DISCRETIZED_NUMERICAL: {
          ASSIGN_OR_RETURN(auto value,
                           DiscretizedNumericalToNumerical(
                               col_spec, src_value.discretized_numerical()));
          dst_value.mutable_float_list()->add_value(value);
        } break;
        default:
          return absl::InvalidArgumentError(
              absl::StrFormat("%s's type is not supported for stacked feature.",
                              unstacked.original_name()));
      }
    }
  }

  for (int col_idx = 0; col_idx < data_spec.columns_size(); col_idx++) {
    const auto& col_spec = data_spec.columns(col_idx);
    if (col_spec.is_unstacked()) {
      continue;
    }
    const auto& src_value = example.attributes(col_idx);
    auto& dst_value =
        (*tf_example->mutable_features()->mutable_feature())[col_spec.name()];

    switch (src_value.type_case()) {
      case proto::Example::Attribute::TypeCase::TYPE_NOT_SET:
        break;
      case proto::Example::Attribute::TypeCase::kBoolean:
        dst_value.mutable_float_list()->add_value(src_value.boolean());
        break;
      case proto::Example::Attribute::TypeCase::kNumerical:
        dst_value.mutable_float_list()->add_value(src_value.numerical());
        break;
      case proto::Example::Attribute::TypeCase::kDiscretizedNumerical: {
        ASSIGN_OR_RETURN(auto value,
                         DiscretizedNumericalToNumerical(
                             col_spec, src_value.discretized_numerical()));
        dst_value.mutable_float_list()->add_value(value);
      } break;
      case proto::Example::Attribute::TypeCase::kCategorical:
        if (col_spec.categorical().is_already_integerized()) {
          dst_value.mutable_int64_list()->add_value(src_value.categorical());
        } else {
          dst_value.mutable_bytes_list()->add_value(
              CategoricalIdxToRepresentation(col_spec,
                                             src_value.categorical()));
        }
        break;
      case proto::Example::Attribute::TypeCase::kText:
        dst_value.mutable_bytes_list()->add_value(src_value.text());
        break;
      case proto::Example::Attribute::TypeCase::kCategoricalList:
        if (col_spec.categorical().is_already_integerized()) {
          *dst_value.mutable_int64_list()->mutable_value() = {
              src_value.categorical_list().values().begin(),
              src_value.categorical_list().values().end()};
        } else {
          for (const auto int_value : src_value.categorical_list().values()) {
            dst_value.mutable_bytes_list()->add_value(
                CategoricalIdxToRepresentation(col_spec, int_value));
          }
        }
        break;
      case proto::Example::Attribute::TypeCase::kCategoricalSet:
        if (col_spec.categorical().is_already_integerized()) {
          *dst_value.mutable_int64_list()->mutable_value() = {
              src_value.categorical_set().values().begin(),
              src_value.categorical_set().values().end()};
        } else {
          for (const auto int_value : src_value.categorical_set().values()) {
            dst_value.mutable_bytes_list()->add_value(
                CategoricalIdxToRepresentation(col_spec, int_value));
          }
        }
        break;
      case proto::Example::Attribute::TypeCase::kNumericalList:
        *dst_value.mutable_float_list()->mutable_value() =
            src_value.numerical_list().values();
        break;
      case proto::Example::Attribute::TypeCase::kNumericalSet:
        *dst_value.mutable_float_list()->mutable_value() =
            src_value.numerical_set().values();
        break;
      case proto::Example::Attribute::TypeCase::kHash:
        dst_value.mutable_int64_list()->add_value(src_value.hash());
        break;
    }
  }
  return absl::OkStatus();
}

absl::Status TfExampleToExampleSet(const tensorflow::Example& src,
                                   int example_idx,
                                   const serving::FeaturesDefinition& features,
                                   serving::AbstractExampleSet* dst) {
  return serving::TfExampleToExampleSet(src, example_idx, features, dst);
}

namespace internal {

absl::StatusOr<float> GetSingleFloatFromTFFeature(
    const tensorflow::Feature& feature, const proto::Column& col) {
  float num_value;
  switch (feature.kind_case()) {
    case tensorflow::Feature::KindCase::KIND_NOT_SET:
      num_value = std::numeric_limits<float>::quiet_NaN();
      break;
    case tensorflow::Feature::KindCase::kFloatList:
      if (feature.float_list().value_size() == 0) {
        num_value = std::numeric_limits<float>::quiet_NaN();
      } else {
        if (feature.float_list().value_size() != 1) {
          return absl::InvalidArgumentError(absl::StrCat(
              "[Error #1] Example found with \"", col.name(),
              "\" having several values while this feature is univariate. ",
              feature.DebugString()));
        }
        num_value = feature.float_list().value(0);
      }
      break;
    case tensorflow::Feature::KindCase::kInt64List:
      if (feature.int64_list().value_size() == 0) {
        num_value = std::numeric_limits<float>::quiet_NaN();
      } else {
        if (feature.int64_list().value_size() != 1) {
          return absl::InvalidArgumentError(
              absl::StrCat("[Error #1] Example found with \"", col.name(),
                           "\" having several values while this "
                           "feature is univariate. ",
                           feature.DebugString()));
        }
        num_value = static_cast<float>(feature.int64_list().value(0));
      }
      break;
    case tensorflow::Feature::KindCase::kBytesList:
      if (feature.bytes_list().value_size() == 0) {
        num_value = std::numeric_limits<float>::quiet_NaN();
      } else {
        if (feature.bytes_list().value_size() != 1) {
          return absl::InvalidArgumentError(absl::StrCat(
              "[Error #1] Example found with \"", col.name(),
              "\" having several values while this feature is univariate. ",
              feature.DebugString()));
        }
        STATUS_CHECK(
            absl::SimpleAtof(feature.bytes_list().value(0), &num_value));
      }
      break;
  }
  return num_value;
}

absl::Status GetNumericalValuesFromTFFeature(const tensorflow::Feature& feature,
                                             const proto::Column& col,
                                             std::vector<float>* values) {
  if (feature.kind_case() == tensorflow::Feature::KindCase::kFloatList) {
    values->assign(feature.float_list().value().begin(),
                   feature.float_list().value().end());
  } else if (feature.kind_case() == tensorflow::Feature::KindCase::kInt64List) {
    values->assign(feature.int64_list().value().begin(),
                   feature.int64_list().value().end());
  } else {
    return absl::InvalidArgumentError(
        "Non supported values for set of numerical values.");
  }
  return absl::OkStatus();
}

absl::Status GetCategoricalTokensFromTFFeature(
    const tensorflow::Feature& feature, const proto::Column& col,
    std::vector<std::string>* tokens) {
  switch (feature.kind_case()) {
    case tensorflow::Feature::KindCase::KIND_NOT_SET:
      break;
    case tensorflow::Feature::KindCase::kFloatList:
      for (const auto value : feature.float_list().value()) {
        tokens->push_back(absl::StrCat(value));
      }
      break;
    case tensorflow::Feature::KindCase::kInt64List:
      for (const auto value : feature.int64_list().value()) {
        tokens->push_back(absl::StrCat(value));
      }
      break;
    case tensorflow::Feature::KindCase::kBytesList:
      if (col.has_tokenizer()) {
        if (feature.bytes_list().value_size() > 1) {
          STATUS_FATALS(
              "The feature \"", col.name(),
              "\" configured with a tokenizer contains multiple entries. "
              "Either disable the tokenizer, or make sure each example does "
              "not contains more than one entry.");
        }
        if (feature.bytes_list().value_size()) {
          RETURN_IF_ERROR(
              Tokenize(feature.bytes_list().value(0), col.tokenizer(), tokens));
        }
      } else {
        for (const auto& value : feature.bytes_list().value()) {
          tokens->push_back(value);
        }
      }
      break;
  }
  if (!IsMultiDimensional(col.type())) {
    if (tokens->size() > 1) {
      STATUS_FATALS(
          "[Error #1] Feature \"", col.name(),
          "\" having several values while this feature is defined as a "
          "univariate feature (",
          proto::ColumnType_Name(col.type()),
          ").\nFeature value: ", feature.DebugString());
    }
  }
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace dataset
}  // namespace yggdrasil_decision_forests
