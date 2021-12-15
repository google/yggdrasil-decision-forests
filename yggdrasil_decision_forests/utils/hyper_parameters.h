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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_HYPER_PARAMETERS_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_HYPER_PARAMETERS_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Helper function to consume generic hyper-parameters.
class GenericHyperParameterConsumer {
 public:
  explicit GenericHyperParameterConsumer(
      const model::proto::GenericHyperParameters& generic_hyper_parameters);

  // Returns a hparam if present.
  absl::optional<model::proto::GenericHyperParameters::Field> Get(
      absl::string_view key);

  // Ensures that all the fields have been consumed.
  // Returns OK if all the hyper-parameters have been consumed.
  // Returns a InvalidArgumentError if at least one of the hyper-parameter has
  // not been consumed.
  absl::Status CheckThatAllHyperparametersAreConsumed() const;

 private:
  // Set of hyper-parameters.
  absl::flat_hash_map<std::string, model::proto::GenericHyperParameters::Field>
      generic_hyper_parameters_;
  // Already consumed hyper-parameters.
  absl::flat_hash_set<std::string> consumed_values_;
};

// Tests if the default value of a field satisfy a condition.
utils::StatusOr<bool> SatisfyDefaultCondition(
    const model::proto::GenericHyperParameterSpecification::Value& value,
    const model::proto::GenericHyperParameterSpecification::Conditional&
        condition);

// Tests if a field is boolean.
bool HyperParameterIsBoolean(
    const model::proto::GenericHyperParameterSpecification::Value& def);

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_HYPER_PARAMETERS_H_
