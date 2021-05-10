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

// Generic the Markdown documentation of the learning algorithms.
// The result of this binary should be exported to: documentation/learners.md
//
#include <iostream>

#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/learner/export_doc.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/utils/logging.h"

std::string url(absl::string_view path, absl::string_view keyword) {
  return std::string(path);
}

int main(int argc, char** argv) {
  InitLogging(argv[0], &argc, &argv, true);
  auto content_or =
      yggdrasil_decision_forests::model::ExportSeveralLearnersToMarkdown(
          yggdrasil_decision_forests::model::AllRegisteredLearners(), url,
          {"GRADIENT_BOOSTED_TREES", "RANDOM_FOREST"});
  std::cout << content_or.value();
  return 0;
}
