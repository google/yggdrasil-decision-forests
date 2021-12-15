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

#include "yggdrasil_decision_forests/utils/hyper_parameters.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace utils {

GenericHyperParameterConsumer::GenericHyperParameterConsumer(
    const model::proto::GenericHyperParameters& generic_hyper_parameters) {
  for (const auto& field : generic_hyper_parameters.fields()) {
    if (generic_hyper_parameters_.find(field.name()) !=
        generic_hyper_parameters_.end()) {
      LOG(FATAL) << "The field \"" << field.name()
                 << "\" is defined several times.";
    }
    generic_hyper_parameters_[field.name()] = field;
  }
}

absl::optional<model::proto::GenericHyperParameters::Field>
GenericHyperParameterConsumer::Get(const absl::string_view key) {
  if (consumed_values_.find(key) != consumed_values_.end()) {
    LOG(FATAL) << absl::StrCat("Already consumed hyper-parameter \"", key,
                               "\".");
  }
  consumed_values_.insert(std::string(key));
  const auto value_it = generic_hyper_parameters_.find(key);
  if (value_it == generic_hyper_parameters_.end()) {
    return {};
  }
  return value_it->second;
}

absl::Status
GenericHyperParameterConsumer::CheckThatAllHyperparametersAreConsumed() const {
  for (const auto& field : generic_hyper_parameters_) {
    if (consumed_values_.find(field.first) == consumed_values_.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unused hyper-parameter \"", field.first, "\"."));
    }
  }
  return absl::OkStatus();
}

utils::StatusOr<bool> SatisfyDefaultCondition(
    const model::proto::GenericHyperParameterSpecification::Value& value,
    const model::proto::GenericHyperParameterSpecification::Conditional&
        condition) {
  switch (condition.constraint_case()) {
    case model::proto::GenericHyperParameterSpecification::Conditional::
        kCategorical:
      if (!value.has_categorical()) {
        return absl::InvalidArgumentError("The value is not categorical.");
      }
      return std::find(condition.categorical().values().begin(),
                       condition.categorical().values().end(),
                       value.categorical().default_value()) !=
             condition.categorical().values().end();
    default:
      return absl::InvalidArgumentError("Invalid condition");
  }
}

bool HyperParameterIsBoolean(
    const model::proto::GenericHyperParameterSpecification::Value& def) {
  return def.has_categorical() &&
         def.categorical().possible_values_size() == 2 &&
         ((def.categorical().possible_values(0) == "false" &&
           def.categorical().possible_values(1) == "true") ||
          (def.categorical().possible_values(0) == "true" &&
           def.categorical().possible_values(1) == "false"));
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
