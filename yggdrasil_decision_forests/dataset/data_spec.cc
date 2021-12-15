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

#include "yggdrasil_decision_forests/dataset/data_spec.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <regex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/regex.h"
namespace yggdrasil_decision_forests {
namespace dataset {
namespace {

using proto::ColumnType;

// Margin added or removed to the extrem boundaries in the conversion from
// discretized to non-discretized numerical values. This value is (currently)
// only used for display. In case of a modeling use, the dataspec will be
// customized for the user / algorithm to choose this value.
const float kEpsDiscretizedToNonDiscretizedNumerical = 1.0f;

// Display "count" followed by the percentage "count/total". If safe to call
// with total=0.
std::string PrettyPercent(uint64_t count, uint64_t total) {
  std::string result;
  absl::StrAppend(&result, count);
  if (total == 0 || count == 0) {
    return result;
  }
  const double p = 100 * static_cast<double>(count) / total;
  absl::StrAppend(&result, " (", absl::SixDigits(p), "%)");
  return result;
}

// Add a bucket [value-eps,value+eps] to boundaries.
// "boundaries" is not expected to be sorted, and can become unsorted.
void AddBucket(const float value, std::vector<float>* boundaries) {
  const float lower_bound = std::nextafter(value, value - 1.f);
  const float upper_bound = std::nextafter(value, value + 1.f);

  if (boundaries->empty()) {
    boundaries->push_back(lower_bound);
    boundaries->push_back(upper_bound);
    return;
  }

  boundaries->erase(std::remove_if(boundaries->begin(), boundaries->end(),
                                   [lower_bound, upper_bound](const float v) {
                                     return v >= lower_bound &&
                                            v <= upper_bound;
                                   }),
                    boundaries->end());

  const float min_value =
      *std::min_element(boundaries->begin(), boundaries->end());
  const float max_value =
      *std::max_element(boundaries->begin(), boundaries->end());

  if (min_value < upper_bound) {
    boundaries->push_back(lower_bound);
  }

  if (max_value > lower_bound) {
    boundaries->push_back(upper_bound);
  }
}

}  // namespace

float GetSingleFloatFromTFFeature(const tensorflow::Feature& feature,
                                  const proto::Column& col) {
  float num_value;
  switch (feature.kind_case()) {
    case tensorflow::Feature::KindCase::KIND_NOT_SET:
      num_value = std::numeric_limits<float>::quiet_NaN();
      break;
    case tensorflow::Feature::KindCase::kFloatList:
      if (feature.float_list().value_size() == 0) {
        num_value = std::numeric_limits<float>::quiet_NaN();
      } else {
        CHECK_EQ(feature.float_list().value_size(), 1)
            << "[Error #1] Example found with \"" << col.name()
            << "\" having several values while this feature is univariate. "
            << feature.DebugString();
        num_value = feature.float_list().value(0);
      }
      break;
    case tensorflow::Feature::KindCase::kInt64List:
      if (feature.int64_list().value_size() == 0) {
        num_value = std::numeric_limits<float>::quiet_NaN();
      } else {
        CHECK_EQ(feature.int64_list().value_size(), 1)
            << "[Error #1] Example found with \"" << col.name()
            << "\" having several values while this feature is univariate. "
            << feature.DebugString();
        num_value = static_cast<float>(feature.int64_list().value(0));
      }
      break;
    case tensorflow::Feature::KindCase::kBytesList:
      if (feature.bytes_list().value_size() == 0) {
        num_value = std::numeric_limits<float>::quiet_NaN();
      } else {
        CHECK_EQ(feature.bytes_list().value_size(), 1)
            << "[Error #1] Example found with \"" << col.name()
            << "\" having several values while this feature is univariate. "
            << feature.DebugString();
        CHECK(absl::SimpleAtof(feature.bytes_list().value(0), &num_value));
      }
      break;
  }
  return num_value;
}

void GetNumericalValuesFromTFFeature(const tensorflow::Feature& feature,
                                     const proto::Column& col,
                                     std::vector<float>* values) {
  if (feature.kind_case() == tensorflow::Feature::KindCase::kFloatList) {
    values->assign(feature.float_list().value().begin(),
                   feature.float_list().value().end());
  } else if (feature.kind_case() == tensorflow::Feature::KindCase::kInt64List) {
    values->assign(feature.int64_list().value().begin(),
                   feature.int64_list().value().end());
  } else {
    LOG(FATAL) << "Non supported values for set of numerical values.";
  }
}

void GetCategoricalTokensFromTFFeature(const tensorflow::Feature& feature,
                                       const proto::Column& col,
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
        CHECK_LE(feature.bytes_list().value_size(), 1)
            << "The feature \"" << col.name()
            << "\" configured with a tokenizer contains multiple entries. "
               "Either disable the tokenizer, or make sure each example does "
               "not contains more than one entry.";
        if (feature.bytes_list().value_size()) {
          Tokenize(feature.bytes_list().value(0), col.tokenizer(), tokens);
        }
      } else {
        for (const auto& value : feature.bytes_list().value()) {
          tokens->push_back(value);
        }
      }
      break;
  }
  if (!IsMultiDimensional(col.type())) {
    CHECK_LE(tokens->size(), 1)
        << "[Error #1] Feature \"" << col.name()
        << "\" having several values while this feature is defined as a "
           "univariate feature ("
        << proto::ColumnType_Name(col.type())
        << ").\nFeature value: " << feature.DebugString();
  }
}

bool IsMultiDimensional(ColumnType type) {
  return type == ColumnType::CATEGORICAL_SET ||
         type == ColumnType::NUMERICAL_SET ||
         type == ColumnType::NUMERICAL_LIST ||
         type == ColumnType::CATEGORICAL_LIST;
}

bool IsCategorical(ColumnType type) {
  return type == ColumnType::CATEGORICAL_SET ||
         type == ColumnType::CATEGORICAL ||
         type == ColumnType::CATEGORICAL_LIST;
}

bool IsNumerical(ColumnType type) {
  return type == ColumnType::NUMERICAL_SET || type == ColumnType::NUMERICAL ||
         type == ColumnType::NUMERICAL_LIST ||
         type == ColumnType::DISCRETIZED_NUMERICAL;
}

int32_t CategoricalStringToValue(const std::string& value,
                                 const proto::Column& col_spec) {
  if (col_spec.categorical().is_already_integerized()) {
    int32_t int_value;
    CHECK(absl::SimpleAtoi(value, &int_value))
        << "Cannot parse the string \"" << value
        << "\" as an integer for columns \"" << col_spec.name() << "\".";
    CHECK_GE(int_value, 0);
    CHECK_LT(int_value, col_spec.categorical().number_of_unique_values());
    return int_value;
  } else {
    auto value_in_dict = col_spec.categorical().items().find(value);
    if (value_in_dict == col_spec.categorical().items().end()) {
      return kOutOfDictionaryItemIndex;
    } else {
      return value_in_dict->second.index();
    }
  }
}

absl::Status BuildColIdxToFeatureLabelIdx(
    const proto::DataSpecification& data_spec,
    const std::vector<std::string>& fields,
    std::vector<int>* col_idx_to_field_idx) {
  col_idx_to_field_idx->resize(data_spec.columns_size());
  for (int col_idx = 0; col_idx < data_spec.columns_size(); col_idx++) {
    const auto& col_name = data_spec.columns(col_idx).name();
    auto it_col_in_feature_names =
        std::find(fields.begin(), fields.end(), col_name);
    if (it_col_in_feature_names == fields.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The column \"", col_name,
          "\" specified in the datasetspec was not found in the csv."));
    }
    (*col_idx_to_field_idx)[col_idx] = it_col_in_feature_names - fields.begin();
  }
  return absl::OkStatus();
}

void GetMultipleColumnIdxFromName(
    const std::vector<std::string>& column_name_regexs,
    const dataset::proto::DataSpecification& data_spec,
    std::vector<int32_t>* column_idxs) {
  // Compile the regex patterns.
  std::vector<std::regex> patterns;
  patterns.reserve(column_name_regexs.size());
  for (const auto& regex : column_name_regexs) {
    patterns.emplace_back(regex);
  }

  column_idxs->clear();
  for (int32_t col_idx = 0; col_idx < data_spec.columns().size(); col_idx++) {
    const auto& col_name = data_spec.columns(col_idx).name();
    // Find if one of the pattern matches the column name.
    bool found_match = false;
    for (const auto& pattern : patterns) {
      if (std::regex_match(col_name, pattern)) {
        found_match = true;
        break;
      }
    }
    if (found_match) {
      column_idxs->push_back(col_idx);
    }
  }

  // Sort the column indices in increasing order and remove the duplicates.
  std::sort(column_idxs->begin(), column_idxs->end());
  column_idxs->erase(std::unique(column_idxs->begin(), column_idxs->end()),
                     column_idxs->end());
}

absl::Status GetSingleColumnIdxFromName(
    const absl::string_view column_name_regex,
    const dataset::proto::DataSpecification& data_spec, int32_t* column_idx) {
  std::vector<std::string> tmp_pattern_vector{std::string(column_name_regex)};
  std::vector<int32_t> tmp_column_idxs;
  GetMultipleColumnIdxFromName(tmp_pattern_vector, data_spec, &tmp_column_idxs);
  if (tmp_column_idxs.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "\"", column_name_regex, "\" does not match any column names."));
  }
  if (tmp_column_idxs.size() > 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "\"", column_name_regex, "\" matches more than one column names."));
  }
  *column_idx = tmp_column_idxs[0];
  return absl::OkStatus();
}

int GetColumnIdxFromName(absl::string_view name,
                         const proto::DataSpecification& data_spec) {
  return GetColumnIdxFromNameWithStatus(name, data_spec).value();
}

utils::StatusOr<int> GetColumnIdxFromNameWithStatus(
    absl::string_view name, const proto::DataSpecification& data_spec) {
  for (int col_idx = 0; col_idx < data_spec.columns_size(); col_idx++) {
    if (data_spec.columns(col_idx).name() == name) {
      return col_idx;
    }
  }
  return absl::InvalidArgumentError(
      absl::Substitute("Unknown column $0", name));
}

absl::optional<int> GetOptionalColumnIdxFromName(
    absl::string_view name, const proto::DataSpecification& data_spec) {
  for (int col_idx = 0; col_idx < data_spec.columns_size(); col_idx++) {
    if (data_spec.columns(col_idx).name() == name) {
      return col_idx;
    }
  }
  return {};
}

bool HasColumn(absl::string_view name,
               const proto::DataSpecification& data_spec) {
  for (int col_idx = 0; col_idx < data_spec.columns_size(); col_idx++) {
    if (data_spec.columns(col_idx).name() == name) {
      return true;
    }
  }
  return false;
}


absl::Status CsvRowToExample(const std::vector<std::string>& csv_fields,
                             const proto::DataSpecification& data_spec,
                             const std::vector<int>& col_idx_to_field_idx,
                             proto::Example* example) {
  CHECK_EQ(col_idx_to_field_idx.size(), data_spec.columns_size());
  example->mutable_attributes()->Clear();
  example->mutable_attributes()->Reserve(data_spec.columns_size());
  for (int col_idx = 0; col_idx < data_spec.columns_size(); col_idx++) {
    const auto& col_spec = data_spec.columns(col_idx);
    auto* dst_value = example->mutable_attributes()->Add();
    // Skip NAs
    const auto field_idx = col_idx_to_field_idx[col_idx];
    if (field_idx == -1) {
      continue;
    }
    const auto& value = csv_fields[field_idx];
    const auto lower_case = absl::AsciiStrToLower(value);
    if (lower_case == CSV_NA || lower_case == CSV_NA_V2) {
      continue;
    }

    switch (col_spec.type()) {
      case ColumnType::UNKNOWN:
        break;
      case ColumnType::NUMERICAL: {
        if (value.empty()) {
          break;
        }
        float num_value;
        if (!absl::SimpleAtof(value, &num_value)) {
          return absl::InvalidArgumentError(
              absl::StrCat("Cannot parse: ", value));
        }

        dst_value->set_numerical(num_value);
      } break;
      case ColumnType::DISCRETIZED_NUMERICAL: {
        if (value.empty()) {
          break;
        }
        float num_value;
        if (!absl::SimpleAtof(value, &num_value)) {
          return absl::InvalidArgumentError(
              absl::StrCat("Cannot parse: ", value));
        }
        dst_value->set_discretized_numerical(
            NumericalToDiscretizedNumerical(col_spec, num_value));
      } break;
      case ColumnType::NUMERICAL_SET:
      case ColumnType::NUMERICAL_LIST: {
        google::protobuf::RepeatedField<float>* dst;
        if (col_spec.type() == ColumnType::NUMERICAL_SET) {
          dst = dst_value->mutable_numerical_set()->mutable_values();
        } else {
          dst = dst_value->mutable_numerical_list()->mutable_values();
        }
        std::vector<std::string> tokens;
        Tokenize(value, col_spec.tokenizer(), &tokens);
        dst->Reserve(tokens.size());
        for (const std::string& token : tokens) {
          float num_value;
          if (!absl::SimpleAtof(token, &num_value)) {
            return absl::InvalidArgumentError(
                absl::StrCat("Cannot parse: ", token));
          }
          dst->Add(num_value);
        }
        if (col_spec.type() == ColumnType::NUMERICAL_SET) {
          // Sets are expected to be sorted.
          std::sort(dst->begin(), dst->end());
          dst->erase(std::unique(dst->begin(), dst->end()), dst->end());
        }
      } break;
      case ColumnType::CATEGORICAL:
        if (value.empty()) {
          break;
        }
        dst_value->set_categorical(CategoricalStringToValue(value, col_spec));
        break;
      case ColumnType::CATEGORICAL_SET:
      case ColumnType::CATEGORICAL_LIST: {
        google::protobuf::RepeatedField<int32_t>* dst;
        if (col_spec.type() == ColumnType::CATEGORICAL_SET) {
          dst = dst_value->mutable_categorical_set()->mutable_values();
        } else {
          dst = dst_value->mutable_categorical_list()->mutable_values();
        }
        std::vector<std::string> tokens;
        Tokenize(value, col_spec.tokenizer(), &tokens);
        dst->Reserve(tokens.size());
        for (const std::string& token : tokens) {
          dst->Add(CategoricalStringToValue(token, col_spec));
        }
        if (col_spec.type() == ColumnType::CATEGORICAL_SET) {
          // Sets are expected to be sorted.
          std::sort(dst->begin(), dst->end());
          dst->erase(std::unique(dst->begin(), dst->end()), dst->end());
        }
      } break;
      case ColumnType::BOOLEAN: {
        if (value.empty()) {
          break;
        }
        float num_value;
        if (!absl::SimpleAtof(value, &num_value)) {
          return absl::InvalidArgumentError(
              absl::StrCat("Cannot parse: ", value));
        }
        dst_value->set_boolean(num_value >= 0.5f);
      } break;
      case ColumnType::STRING:
        *dst_value->mutable_text() = value;
        break;
      case ColumnType::HASH: {
        if (value.empty()) {
          break;
        }
        dst_value->set_hash(HashColumnString(value));
      } break;
    }
  }
  return absl::OkStatus();
}

void ExampleToCsvRow(const proto::Example& example,
                     const proto::DataSpecification& data_spec,
                     std::vector<std::string>* csv_fields) {
  csv_fields->resize(data_spec.columns_size());
  for (int col_idx = 0; col_idx < data_spec.columns_size(); col_idx++) {
    const auto& col_spec = data_spec.columns(col_idx);
    const auto& src_value = example.attributes(col_idx);
    auto& dst_value = (*csv_fields)[col_idx];

    switch (src_value.type_case()) {
      case proto::Example::Attribute::TypeCase::TYPE_NOT_SET:
        dst_value = "NA";
        break;
      case proto::Example::Attribute::TypeCase::kBoolean:
        dst_value = src_value.boolean() ? '1' : '0';
        break;
      case proto::Example::Attribute::TypeCase::kNumerical:
        dst_value = absl::StrCat(src_value.numerical());
        break;
      case proto::Example::Attribute::TypeCase::kDiscretizedNumerical:
        dst_value = absl::StrCat(DiscretizedNumericalToNumerical(
            col_spec, src_value.discretized_numerical()));
        break;
      case proto::Example::Attribute::TypeCase::kCategorical:
        dst_value =
            CategoricalIdxToRepresentation(col_spec, src_value.categorical());
        break;
      case proto::Example::Attribute::TypeCase::kText:
        dst_value = src_value.text();
        break;
      case proto::Example::Attribute::TypeCase::kCategoricalList:
        dst_value = CategoricalIdxsToRepresentation(
            col_spec,
            std::vector<int>(src_value.categorical_list().values().begin(),
                             src_value.categorical_list().values().end()),
            /*max_values = */ -1, /*separator = */ " ");
        break;
      case proto::Example::Attribute::TypeCase::kCategoricalSet:
        dst_value = CategoricalIdxsToRepresentation(
            col_spec,
            std::vector<int>(src_value.categorical_set().values().begin(),
                             src_value.categorical_set().values().end()),
            /*max_values = */ -1, /*separator = */ " ");
        break;
      case proto::Example::Attribute::TypeCase::kNumericalList:
        dst_value = absl::StrJoin(src_value.numerical_list().values(),
                                  /*separator = */ " ");
        break;
      case proto::Example::Attribute::TypeCase::kNumericalSet:
        dst_value = absl::StrJoin(src_value.numerical_set().values(),
                                  /*separator = */ " ");
        break;
      case proto::Example::Attribute::TypeCase::kHash:
        dst_value = absl::StrCat(src_value.hash());
        break;
    }
  }
}

absl::Status TfExampleToExample(const tensorflow::Example& tf_example,
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
        const float num_value =
            GetSingleFloatFromTFFeature(it_feature->second, col_spec);
        dst_value->set_numerical(num_value);
      } break;
      case ColumnType::DISCRETIZED_NUMERICAL: {
        const float num_value =
            GetSingleFloatFromTFFeature(it_feature->second, col_spec);
        dst_value->set_discretized_numerical(
            NumericalToDiscretizedNumerical(col_spec, num_value));
      } break;
      case ColumnType::NUMERICAL_SET:
      case ColumnType::NUMERICAL_LIST: {
        std::vector<float> values;
        GetNumericalValuesFromTFFeature(it_feature->second, col_spec, &values);

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
        GetCategoricalTokensFromTFFeature(it_feature->second, col_spec,
                                          &tokens);
        if (tokens.empty()) {
          // NA.
        } else if (tokens.size() == 1) {
          dst_value->set_categorical(
              CategoricalStringToValue(tokens[0], col_spec));
        } else {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Categorical attribute with more than one value for feature %s",
              it_feature->first));
        }
      } break;
      case ColumnType::CATEGORICAL_SET:
      case ColumnType::CATEGORICAL_LIST: {
        std::vector<std::string> tokens;
        GetCategoricalTokensFromTFFeature(it_feature->second, col_spec,
                                          &tokens);

        google::protobuf::RepeatedField<int32_t>* dst;
        if (col_spec.type() == ColumnType::CATEGORICAL_SET) {
          dst = dst_value->mutable_categorical_set()->mutable_values();
        } else {
          dst = dst_value->mutable_categorical_list()->mutable_values();
        }

        dst->Reserve(tokens.size());
        for (const std::string& token : tokens) {
          dst->Add(CategoricalStringToValue(token, col_spec));
        }
        if (col_spec.type() == ColumnType::CATEGORICAL_SET) {
          // Sets are expected to be sorted.
          std::sort(dst->begin(), dst->end());
          dst->erase(std::unique(dst->begin(), dst->end()), dst->end());
        }
      } break;
      case ColumnType::BOOLEAN: {
        const float num_value =
            GetSingleFloatFromTFFeature(it_feature->second, col_spec);
        dst_value->set_boolean(num_value >= 0.5f);
      } break;
      case ColumnType::STRING:
        CHECK_EQ(it_feature->second.kind_case(),
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
        GetCategoricalTokensFromTFFeature(it_feature->second, col_spec,
                                          &tokens);
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

void ExampleToTfExample(const proto::Example& example,
                        const proto::DataSpecification& data_spec,
                        tensorflow::Example* tf_example) {
  CHECK_OK(ExampleToTfExampleWithStatus(example, data_spec, tf_example));
}

absl::Status ExampleToTfExampleWithStatus(
    const proto::Example& example, const proto::DataSpecification& data_spec,
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
        case proto::DISCRETIZED_NUMERICAL:
          dst_value.mutable_float_list()->add_value(
              DiscretizedNumericalToNumerical(
                  col_spec, src_value.discretized_numerical()));
          break;
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
      case proto::Example::Attribute::TypeCase::kDiscretizedNumerical:
        dst_value.mutable_float_list()->add_value(
            DiscretizedNumericalToNumerical(col_spec,
                                            src_value.discretized_numerical()));
        break;
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

std::string PrintHumanReadable(const proto::DataSpecification& data_spec,
                               const bool sort_by_column_names) {
  std::string result;

  // Compute the display order of the columns.
  std::vector<int> display_idxs(data_spec.columns_size());
  if (sort_by_column_names) {
    // Sort the column indices by name.
    std::vector<std::pair<std::string, int>> index;
    index.reserve(data_spec.columns_size());
    for (int col_idx = 0; col_idx < data_spec.columns_size(); col_idx++) {
      index.emplace_back(data_spec.columns(col_idx).name(), col_idx);
    }
    std::sort(index.begin(), index.end());
    std::transform(
        index.begin(), index.end(), display_idxs.begin(),
        [](const std::pair<std::string, int>& e) { return e.second; });
  } else {
    // Use the native column ordering.
    std::iota(display_idxs.begin(), display_idxs.end(), 0);
  }

  absl::StrAppendFormat(&result, "Number of records: %i\n",
                        data_spec.created_num_rows());
  absl::StrAppendFormat(&result, "Number of columns: %i\n",
                        data_spec.columns_size());
  absl::StrAppend(&result, "\n");

  // Compute the number of columns of every type.
  std::map<proto::ColumnType, int> column_type_count_map;
  for (const auto& col : data_spec.columns()) {
    column_type_count_map[col.type()]++;
  }
  std::vector<std::pair<int, proto::ColumnType>> column_type_count;
  column_type_count.reserve(column_type_count_map.size());
  for (const auto& type_and_count : column_type_count_map) {
    column_type_count.emplace_back(type_and_count.second, type_and_count.first);
  }
  std::sort(column_type_count.begin(), column_type_count.end(),
            std::greater<std::pair<int, proto::ColumnType>>());

  // Display the number of columns for every type.
  absl::StrAppend(&result, "Number of columns by type:\n");
  for (const auto& count_and_type : column_type_count) {
    absl::StrAppendFormat(
        &result, "\t%s: %s\n", ColumnType_Name(count_and_type.second),
        PrettyPercent(count_and_type.first, data_spec.columns_size()));
  }
  absl::StrAppend(&result, "\n");

  // Display individual column information. Columns are grouped by type.
  absl::StrAppend(&result, "Columns:\n\n");
  for (const auto& count_and_type : column_type_count) {
    // Column type.
    absl::StrAppendFormat(
        &result, "%s: %s\n", ColumnType_Name(count_and_type.second),
        PrettyPercent(count_and_type.first, data_spec.columns_size()));
    for (const int col_idx : display_idxs) {
      const auto& col = data_spec.columns(col_idx);
      if (col.type() != count_and_type.second) {
        continue;
      }
      // Column index.
      absl::StrAppendFormat(&result, "\t%i: ", col_idx);
      // Column name.
      absl::StrAppendFormat(&result, "\"%s\" %s", col.name(),
                            ColumnType_Name(col.type()));
      // Column tags.
      if (col.is_manual_type()) {
        absl::StrAppend(&result, " manually-defined");
      }
      if (col.has_tokenizer()) {
        absl::StrAppend(&result, " tokenized");
      }

      if (col.count_nas() > 0) {
        absl::StrAppend(
            &result, " num-nas:",
            PrettyPercent(col.count_nas(), data_spec.created_num_rows()));
      }

      if (col.has_numerical()) {
        // Numerical information.
        absl::SubstituteAndAppend(
            &result, " mean:$0 min:$1 max:$2 sd:$3", col.numerical().mean(),
            col.numerical().min_value(), col.numerical().max_value(),
            col.numerical().standard_deviation());
      }

      if (col.has_discretized_numerical()) {
        absl::SubstituteAndAppend(
            &result, " discretized bins:$0 orig-bins:$1",
            col.discretized_numerical().boundaries_size() + 1,
            col.discretized_numerical().original_num_unique_values());
      }

      if (col.has_boolean()) {
        // Boolean information.
        absl::SubstituteAndAppend(&result, " true_count:$0 false_count:$1",
                                  col.boolean().count_true(),
                                  col.boolean().count_false());
      }

      if (col.has_categorical()) {
        // Categorical information.
        if (col.categorical().items_size() > 0) {
          absl::StrAppend(&result, " has-dict");
        }
        if (col.categorical().is_already_integerized()) {
          absl::StrAppend(&result, " integerized");
        }
        if (col.categorical().number_of_unique_values() > 0) {
          absl::StrAppend(&result, " vocab-size:",
                          col.categorical().number_of_unique_values());

          // Check for the "out-of-dictionary" item.
          const auto it_rare =
              col.categorical().items().find(kOutOfDictionaryItemKey);
          if (it_rare != col.categorical().items().end()) {
            if (it_rare->second.count() == 0) {
              absl::StrAppend(&result, " zero-ood-items");
            } else {
              absl::StrAppend(&result, " num-oods:",
                              PrettyPercent(it_rare->second.count(),
                                            data_spec.created_num_rows() -
                                                col.count_nas()));
            }
          } else {
            absl::StrAppend(&result, " no-ood-item");
          }

          // Display the number and the most frequent item in the dictionary.
          // In case of ties, use the lexical ordering.
          if (col.categorical().number_of_unique_values() >= 2) {
            int64_t highest_count = -1;
            std::string most_frequent_item;
            for (auto& item : col.categorical().items()) {
              if (item.second.count() > highest_count ||
                  (item.second.count() == highest_count &&
                   item.first < most_frequent_item)) {
                highest_count = item.second.count();
                most_frequent_item = item.first;
              }
            }
            if (highest_count > 0) {
              absl::StrAppendFormat(
                  &result, " most-frequent:\"%s\" %s", most_frequent_item,
                  PrettyPercent(highest_count, data_spec.created_num_rows() -
                                                   col.count_nas()));
            }
          }
        }
      }
      absl::StrAppend(&result, "\n");
    }
    absl::StrAppend(&result, "\n");
  }

  absl::StrAppend(
      &result,
      "Terminology:\n"
      "\tnas: Number of non-available (i.e. missing) values.\n"
      "\tood: Out of dictionary.\n"
      "\tmanually-defined: Attribute which type is manually defined by the "
      "user i.e. the type was not automatically inferred.\n"
      "\ttokenized: The attribute value is obtained through tokenization.\n"
      "\thas-dict: The attribute is attached to a string dictionary e.g. a "
      "categorical attribute stored as a string.\n"
      "\tvocab-size: Number of unique values.\n");
  return result;
}

void Tokenize(const absl::string_view text, const proto::Tokenizer& tokenizer,
              std::vector<std::string>* tokens) {
  tokens->clear();
  if (text.empty()) return;
  std::string cased_text;
  // Optional string lower casing.
  if (tokenizer.to_lower_case()) {
    cased_text = absl::AsciiStrToLower(text);
  } else {
    cased_text = std::string(text);
  }
  // Split the string into tokens.
  std::vector<std::string> unit_tokens;
  switch (tokenizer.splitter()) {
    case proto::Tokenizer::INVALID:
      LOG(FATAL) << "Unsupported INVALID tokenizer type.";
      break;
    case proto::Tokenizer::SEPARATOR:
      unit_tokens =
          absl::StrSplit(cased_text, absl::ByAnyChar(tokenizer.separator()));
      break;
    case proto::Tokenizer::REGEX_MATCH: {
      std::string remaining = cased_text;
      std::regex re(tokenizer.regex());
      std::smatch sm;
      while (std::regex_search(remaining, sm, re)) {
        unit_tokens.emplace_back(sm.str());
        remaining = sm.suffix();
      }
    } break;
    case proto::Tokenizer::CHARACTER:
      for (char c : cased_text) {
        unit_tokens.emplace_back(1, c);
      }
      break;
  }
  // Remove empty tokens.
  unit_tokens.erase(
      std::remove_if(unit_tokens.begin(), unit_tokens.end(),
                     [](const std::string& key) { return key.empty(); }),
      unit_tokens.end());
  // Ground the tokens.
  const absl::string_view group_separator =
      (tokenizer.splitter() == proto::Tokenizer::CHARACTER) ? "" : "_";
  if (tokenizer.grouping().unigrams()) {
    tokens->insert(tokens->end(), unit_tokens.begin(), unit_tokens.end());
  }
  if (tokenizer.grouping().bigrams()) {
    ExtractNGrams(unit_tokens, 2, group_separator, tokens);
  }
  if (tokenizer.grouping().trigrams()) {
    ExtractNGrams(unit_tokens, 3, group_separator, tokens);
  }
}

void ExtractNGrams(const std::vector<std::string>& tokens, const int n,
                   const absl::string_view separator,
                   std::vector<std::string>* grouped_tokens) {
  const int expected_size =
      static_cast<int>(grouped_tokens->size()) + tokens.size() - n;
  grouped_tokens->reserve(std::max(0, expected_size));
  const int end_new_token_idx = static_cast<int>(tokens.size()) - n + 1;
  for (int new_token_idx = 0; new_token_idx < end_new_token_idx;
       new_token_idx++) {
    std::string new_token =
        absl::StrJoin(tokens.begin() + new_token_idx,
                      tokens.begin() + new_token_idx + n, separator);
    grouped_tokens->push_back(std::move(new_token));
  }
}

std::string CategoricalIdxToRepresentation(const proto::Column& col_spec,
                                           const int32_t value_idx,
                                           const bool add_quotes) {
  if (value_idx == -1) {
    return "NA";
  } else if (col_spec.categorical().is_already_integerized()) {
    return absl::StrCat(value_idx);
  } else {
    for (auto& item : col_spec.categorical().items()) {
      if (item.second.index() == value_idx) {
        // If the string contains spaces, we add quotes.
        if (add_quotes && std::find(item.first.begin(), item.first.end(),
                                    ' ') != item.first.end()) {
          return absl::StrCat("\"", item.first, "\"");
        } else {
          return item.first;
        }
      }
    }
    return absl::StrCat("NA(", value_idx, ")");
  }
}

std::string CategoricalIdxsToRepresentation(const proto::Column& col_spec,
                                            const std::vector<int>& elements,
                                            const int max_values,
                                            const absl::string_view separator) {
  std::string result;
  for (int idx = 0; idx < elements.size(); idx++) {
    if (max_values >= 0 && idx >= max_values) {
      absl::StrAppend(&result, separator, "...[", elements.size() - idx,
                      " left]");
      break;
    }
    if (idx > 0) {
      absl::StrAppend(&result, separator);
    }
    absl::StrAppend(&result,
                    CategoricalIdxToRepresentation(col_spec, elements[idx]));
  }
  return result;
}

proto::Column* AddColumn(const absl::string_view name,
                         const proto::ColumnType type,
                         proto::DataSpecification* data_spec) {
  auto* col = data_spec->add_columns();
  col->set_name(std::string(name));
  col->set_type(type);
  return col;
}

std::vector<float> GenDiscretizedBoundaries(
    const std::vector<std::pair<float, int>>& candidates, int maximum_num_bins,
    int min_obs_in_bins, const std::vector<float>& special_values) {
  // Algorithm:
  //
  // The goal is to create "maximum_num_bins" bins, somehow uniformly by
  // quantile in the count space and without roll-backs on the decision.
  //
  // If the number of candidate is smaller than the maximum_num_bins, the
  // boundaries are built directly (while making sure "min_obs_in_bins" is
  // respected).
  //
  // Otherwise:
  //   - Each value in "special_values" will have one bin is 2*eps surface.
  //   - Each value with count >= total_count / maximum_num_bins will have one
  //     bin. Those are called "large candidates".
  //   - Bins are created iteratively from lowest to highest values. A bin is
  //     created if a value count is >= remaining_count / num_remaining_bins.
  //     Those are called "late large candidates".
  //
  // Terminology:
  //
  // remaining counts: Observations not already assigned to a bin and not
  // reserved by a large candidate.

  CHECK_LT(maximum_num_bins, kDiscretizedNumericalMissingValue);
  CHECK_GE(min_obs_in_bins, 1);

  // Reserve bins for special values.
  int num_special_values_in_bounds = 0;
  if (!candidates.empty()) {
    for (const auto special_value : special_values) {
      if (special_value > candidates.front().first &&
          special_value < candidates.back().first) {
        num_special_values_in_bounds++;
      }
    }
  }
  maximum_num_bins =
      std::max<std::size_t>(1, maximum_num_bins - special_values.size() -
                                   num_special_values_in_bounds);

  const int max_num_boundaries = maximum_num_bins - 1;
  std::vector<float> boundaries;
  if (candidates.size() > maximum_num_bins) {
    int64_t sum_counts = 0;
    for (const auto& candidate : candidates) {
      sum_counts += candidate.second;
    }

    // Effective maximum number of bins
    maximum_num_bins = std::min(maximum_num_bins,
                                static_cast<int>(sum_counts / min_obs_in_bins));
    // Candidates with more than "maximum_num_bins" elements are "large".
    const int large_count = sum_counts / maximum_num_bins;

    // Count of elements not already added and not in the "large" count.
    int remaining_num_bins = maximum_num_bins;
    int64_t remaining_count = sum_counts;

    // Remove the "large" count from the remaining counts.
    std::vector<bool> is_large_count(candidates.size(), false);
    for (int candidate_idx = 0; candidate_idx < candidates.size();
         candidate_idx++) {
      if (candidates[candidate_idx].second >= large_count) {
        is_large_count[candidate_idx] = true;
        remaining_num_bins--;
        remaining_count -= candidates[candidate_idx].second;
      }
    }

    // What is considered "large". Will be updated as new bins are created.
    if (remaining_num_bins < 1) {
      remaining_num_bins = 1;
    }
    int64_t current_large_count = remaining_count / remaining_num_bins;

    int running_count = 0;
    int next_bin_idx = 0;
    for (int candidate_idx = 0;
         candidate_idx < static_cast<int>(candidates.size()) - 1;
         candidate_idx++) {
      if (!is_large_count[candidate_idx]) {
        remaining_count -= candidates[candidate_idx].second;
      }
      running_count += candidates[candidate_idx].second;
      if (is_large_count[candidate_idx] ||
          running_count >= current_large_count ||
          (is_large_count[candidate_idx + 1] &&
           running_count >= std::max(int64_t{1}, current_large_count / 2))) {
        const float boundary = (candidates[candidate_idx].first +
                                candidates[candidate_idx + 1].first) /
                               2;
        boundaries.push_back(boundary);

        next_bin_idx++;
        if (next_bin_idx >= max_num_boundaries) {
          break;
        }
        running_count = 0;
        if (!is_large_count[candidate_idx]) {
          remaining_num_bins--;
          if (remaining_num_bins < 1) {
            remaining_num_bins = 1;
          }
          current_large_count = remaining_count / remaining_num_bins;
        }
      }
    }
  } else {
    // Set to boundaries to be the center in between each bins.
    int running_count = 0;
    for (int candidate_idx = 0;
         candidate_idx < static_cast<int>(candidates.size()) - 1;
         candidate_idx++) {
      running_count += candidates[candidate_idx].second;
      if (running_count >= min_obs_in_bins) {
        const float boundary = (candidates[candidate_idx].first +
                                candidates[candidate_idx + 1].first) /
                               2;
        boundaries.push_back(boundary);
        running_count = 0;
      }
    }
  }

  for (const auto& special_value : special_values) {
    AddBucket(special_value, &boundaries);
  }
  std::sort(boundaries.begin(), boundaries.end());

  return boundaries;
}

float DiscretizedNumericalToNumerical(const proto::Column& col_spec,
                                      const DiscretizedNumericalIndex value) {
  if (value == kDiscretizedNumericalMissingValue) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const auto& boundaries = col_spec.discretized_numerical().boundaries();
  DCHECK_GT(boundaries.size(), 0);
  CHECK_LE(value, boundaries.size());
  if (value == 0) {
    return boundaries[0] - kEpsDiscretizedToNonDiscretizedNumerical;
  }
  if (value == boundaries.size()) {
    return boundaries[boundaries.size() - 1] +
           kEpsDiscretizedToNonDiscretizedNumerical;
  }
  return (boundaries[value] + boundaries[value - 1]) / 2;
}

DiscretizedNumericalIndex NumericalToDiscretizedNumerical(
    const proto::Column& col_spec, const float value) {
  if (std::isnan(value)) {
    return kDiscretizedNumericalMissingValue;
  }
  const auto& boundaries = col_spec.discretized_numerical().boundaries();
  const auto it = std::upper_bound(boundaries.begin(), boundaries.end(), value);
  return std::distance(boundaries.begin(), it);
}

std::string EscapeTrainingConfigFeatureName(absl::string_view feature_name) {
  return utils::QuoteRegex(feature_name);
}

std::string UnstackedColumnName(const absl::string_view original_name,
                                const int dim_idx) {
  return absl::StrFormat("%s__%05d", original_name, dim_idx);
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
