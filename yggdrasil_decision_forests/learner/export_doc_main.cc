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

// Generic the Markdown documentation of the learning algorithms.
// The result of this binary should be exported to: documentation/learners.md
//
#include <iostream>

#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/learner/export_doc.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, url_type, "GITHUB",
          "Type of url to the protobuffer definition.");

// Converts a source file path (relative to the ydf directory) and search
// keyword into an url. When opening this url, the user expects to see the
// source file content.
std::string url(absl::string_view path, absl::string_view keyword) {
  const auto& url_type = absl::GetFlag(FLAGS_url_type);
  if (url_type == "GITHUB") {
    // Local github path.
    return std::string(path);
  }
  else if (url_type == "READ_THE_DOCS") {
    // Use absolute github paths.
    return absl::StrCat(
        "https://github.com/google/yggdrasil-decision-forests/blob/main/"
        "yggdrasil_decision_forests/",
        path);
  } else {
    YDF_LOG(FATAL) << "Unknown --url_type value";
  }
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
