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

#include "yggdrasil_decision_forests/dataset/weight.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/weight.pb.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace dataset {

utils::StatusOr<proto::WeightDefinition> GetUnlinkedWeightDefinition(
    const proto::LinkedWeightDefinition& linked_def,
    const proto::DataSpecification& data_spec) {
  proto::WeightDefinition weights;
  weights.set_attribute(data_spec.columns(linked_def.attribute_idx()).name());
  switch (linked_def.type_case()) {
    case proto::LinkedWeightDefinition::kNumerical:
      weights.mutable_numerical();
      break;
    case proto::LinkedWeightDefinition::kCategorical: {
      auto& dst_items = *weights.mutable_categorical()->mutable_items();
      const auto& spec = data_spec.columns(linked_def.attribute_idx());
      for (int value = 0; value < spec.categorical().number_of_unique_values();
           value++) {
        auto& item = *dst_items.Add();
        item.set_value(dataset::CategoricalIdxToRepresentation(spec, value));
        item.set_weight(
            linked_def.categorical().categorical_value_idx_2_weight(value));
      }
    } break;
    default:
      return absl::InvalidArgumentError("Unknown weight type");
  }
  return weights;
}

absl::Status GetLinkedWeightDefinition(
    const proto::WeightDefinition& def,
    const proto::DataSpecification& data_spec,
    proto::LinkedWeightDefinition* linked_def) {
  int32_t attribute_idx;
  RETURN_IF_ERROR(
      GetSingleColumnIdxFromName(def.attribute(), data_spec, &attribute_idx));
  linked_def->set_attribute_idx(attribute_idx);

  switch (def.type_case()) {
    case dataset::proto::WeightDefinition::TypeCase::kCategorical: {
      const auto& data_spec_categorical =
          data_spec.columns(attribute_idx).categorical().items();
      auto cat_value_idx_2_weight =
          linked_def->mutable_categorical()
              ->mutable_categorical_value_idx_2_weight();
      cat_value_idx_2_weight->Clear();
      cat_value_idx_2_weight->Resize(data_spec_categorical.size(), -1);
      for (const auto& weight_item : def.categorical().items()) {
        const auto it_item = data_spec_categorical.find(weight_item.value());
        if (it_item == data_spec_categorical.end()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "The categorical weight value \"", weight_item.value(),
              "\" is not defined in the column dataspec of \"", def.attribute(),
              "\"."));
        }
        const int32_t cat_value_idx = it_item->second.index();
        if ((*cat_value_idx_2_weight)[cat_value_idx] != -1) {
          return absl::InvalidArgumentError(absl::StrCat(
              "The categorical weight value \"", weight_item.value(),
              "\" is defined several times in the weight specification."));
        };
        if (weight_item.weight() < 0) {
          return absl::InvalidArgumentError(absl::StrCat(
              "The categorical weight value \"", weight_item.value(),
              "\" is defined with a negative weight."));
        };
        linked_def->mutable_categorical()->set_categorical_value_idx_2_weight(
            cat_value_idx, weight_item.weight());
      }

      // The OOD item has a weight of 1 if not specified by the user.
      if (!cat_value_idx_2_weight->empty() &&
          (*cat_value_idx_2_weight)[kOutOfDictionaryItemIndex] < 0) {
        (*cat_value_idx_2_weight)[kOutOfDictionaryItemIndex] = 1.f;
      }
      for (int weight_idx = 0; weight_idx < cat_value_idx_2_weight->size();
           weight_idx++) {
        if ((*cat_value_idx_2_weight)[weight_idx] < 0) {
          return absl::InvalidArgumentError(absl::StrCat(
              "The categorical weight value \"",
              CategoricalIdxToRepresentation(data_spec.columns(attribute_idx),
                                             weight_idx),
              "\" does not have any defined weight or the defined weight is "
              "negative."));
        }
      }
      break;
    }
    case dataset::proto::WeightDefinition::TypeCase::kNumerical:
      linked_def->mutable_numerical();
      break;
    case dataset::proto::WeightDefinition::TypeCase::TYPE_NOT_SET:
      return absl::InvalidArgumentError(
          "The type of weighting is not defined.");
      break;
  }
  return absl::OkStatus();
}

float GetWeight(const proto::Example& example,
                const proto::LinkedWeightDefinition& weight_definition) {
  switch (weight_definition.type_case()) {
    case proto::LinkedWeightDefinition::kCategorical: {
      const int cat_value =
          example.attributes(weight_definition.attribute_idx()).categorical();
      if (cat_value == VerticalDataset::CategoricalColumn::kNaValue) {
        LOG(FATAL) << "Found NA value for weighting attribute";
      }
      return weight_definition.categorical().categorical_value_idx_2_weight(
          cat_value);
    }
    case proto::LinkedWeightDefinition::kNumerical: {
      const float num_value =
          example.attributes(weight_definition.attribute_idx()).numerical();
      if (std::isnan(num_value)) {
        LOG(FATAL) << "Found NA value for weighting attribute";
      }
      if (num_value < 0) {
        LOG(FATAL) << "Found negative weight value";
      }
      return num_value;
    }
    default:
      LOG(FATAL) << "Non implemented";
  }
  return 1.f;
}

float GetWeight(const VerticalDataset& dataset, VerticalDataset::row_t row,
                const proto::LinkedWeightDefinition& weight_definition) {
  switch (weight_definition.type_case()) {
    case proto::LinkedWeightDefinition::kCategorical: {
      // "weight_col" is the data about the categorical attribute that controls
      // the weighting of example i.e. the vector of value used for weighting
      // indexed by the row index. These values are categorical and they are
      // indexed with integers. the mapping from these integer to their value is
      // done with the column_spec dictionary.
      //
      // "categorical_value_idx_2_weight" maps the categorical value indices
      // (i.e. what "weight_col" contains) to the float weight values.
      //
      // Example:
      //   std::vector<int> weight_col = {1,2,1}
      //   column_spec_dictionary = {0 : "riri", 1: "fifi", 2: "loulou"}
      //   std::vector<float> categorical_value_idx_2_weight = {2.f,3.f,4.f}
      //
      // In this example, the weight of the first example is 3, and the second
      // example is 4.
      const auto* weight_col =
          dataset.ColumnWithCast<VerticalDataset::CategoricalColumn>(
              weight_definition.attribute_idx());
      const int cat_value = weight_col->values()[row];
      if (cat_value == VerticalDataset::CategoricalColumn::kNaValue) {
        LOG(FATAL) << "Found NA value for weighting attribute in example #"
                   << row;
      }
      return weight_definition.categorical().categorical_value_idx_2_weight(
          cat_value);
    }
    case proto::LinkedWeightDefinition::kNumerical: {
      const auto* weight_col =
          dataset.ColumnWithCast<VerticalDataset::NumericalColumn>(
              weight_definition.attribute_idx());
      const float num_value = weight_col->values()[row];
      if (std::isnan(num_value)) {
        LOG(FATAL) << "Found NA value for weighting attribute in example #"
                   << row;
      }
      if (num_value < 0) {
        LOG(FATAL) << "Found negative weight value in example #" << row;
      }
      return num_value;
    }
    default:
      LOG(FATAL) << "Non implemented";
  }
  return 1.f;
}

absl::Status GetWeights(const VerticalDataset& dataset,
                        const proto::LinkedWeightDefinition& weight_definition,
                        std::vector<float>* weights) {
  switch (weight_definition.type_case()) {
    case proto::LinkedWeightDefinition::kCategorical: {
      const auto* weight_col =
          dataset.ColumnWithCast<VerticalDataset::CategoricalColumn>(
              weight_definition.attribute_idx());
      weights->resize(dataset.nrow());
      for (VerticalDataset::row_t row_idx = 0; row_idx < dataset.nrow();
           row_idx++) {
        const int cat_value = weight_col->values()[row_idx];
        if (cat_value == VerticalDataset::CategoricalColumn::kNaValue) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Found NA value for weighting attribute in example #", row_idx));
        }
        (*weights)[row_idx] =
            weight_definition.categorical().categorical_value_idx_2_weight(
                cat_value);
      }
    } break;
    case proto::LinkedWeightDefinition::kNumerical: {
      const auto* weight_col =
          dataset.ColumnWithCast<VerticalDataset::NumericalColumn>(
              weight_definition.attribute_idx());
      *weights = weight_col->values();
      if (std::find_if(weights->begin(), weights->end(), [](const float value) {
            return std::isnan(value);
          }) != weights->end()) {
        return absl::InvalidArgumentError(
            "Found NA value for weighting attribute.");
      }
      if (std::find_if(weights->begin(), weights->end(), [](const float value) {
            return value < 0;
          }) != weights->end()) {
        return absl::InvalidArgumentError("Found negative weight value.");
      }
    } break;
    default:
      return absl::InvalidArgumentError("Non implemented");
  }
  return absl::OkStatus();
}

absl::Status GetWeights(
    const VerticalDataset& dataset,
    const model::proto::TrainingConfigLinking& train_config_link,
    std::vector<float>* weights, bool use_optimized_unit_weights) {
  if (train_config_link.has_weight_definition()) {
    RETURN_IF_ERROR(
        GetWeights(dataset, train_config_link.weight_definition(), weights));
    // Check if all values are identical.
    if (use_optimized_unit_weights &&
        std::all_of(weights->cbegin(), weights->cend(),
                    [](const float value) { return value == 1.f; })) {
      weights->clear();
    }
  } else {
    if (use_optimized_unit_weights) {
      weights->clear();
    } else {
      weights->assign(dataset.nrow(), 1.f);
    }
  }
  return absl::OkStatus();
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
