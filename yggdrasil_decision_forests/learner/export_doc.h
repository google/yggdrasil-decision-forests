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

// Export the learner's documentation (e.g. hparams) to various documentation
// formats.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_EXPORT_DOC_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_EXPORT_DOC_H_

#include <string>

#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"

namespace yggdrasil_decision_forests {
namespace model {

// Signature of a function to create the url to the code source code.
using DocumentationUrlFunctor = std::function<std::string(
    absl::string_view file_path, absl::string_view keyword)>;

// Creates the Markdown documentation of the linked learners.
//
// Args:
//   learners: Name of the learners to export. Those learners need to be linked.
//     Use "AllRegisteredLearners()" for the list of all linked learners.
//   order: Order in which to display the learners. Remaining learners are
//     ordered alphabetically after the ones specified in "order".
utils::StatusOr<std::string> ExportSeveralLearnersToMarkdown(
    std::vector<std::string> learners,
    const DocumentationUrlFunctor& gen_doc_url,
    const std::vector<std::string>& ordering = {});

// Create the Markdown documentation for a set of hyper-parameters.
utils::StatusOr<std::string> ExportHParamSpecToMarkdown(
    absl::string_view learner_key,
    const proto::GenericHyperParameterSpecification& hparams,
    const DocumentationUrlFunctor& gen_doc_url);

}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_EXPORT_DOC_H_
