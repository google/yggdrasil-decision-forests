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

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_DESCRIBE_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_DESCRIBE_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/utils/html.h"

namespace yggdrasil_decision_forests::model {

// Creates a Html report of a model.
// "block_id" is a prefix used to generate the unique ID of html elements.
// If not given, a unique id will be generated.
absl::StatusOr<std::string> DescribeModelHtml(const model::AbstractModel& model,
                                              absl::string_view block_id = {});

// Creates a Html report with variable importances. "block_id" is used to
// generate unique html id.
absl::StatusOr<utils::html::Html> VariableImportance(
    const absl::flat_hash_map<std::string,
                              std::vector<proto::VariableImportance>>&
        variable_importances,
    const dataset::proto::DataSpecification& data_spec,
    absl::string_view block_id);

// Html header for VariableImportance.
std::string Header();

}  // namespace yggdrasil_decision_forests::model

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_DESCRIBE_H_
