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

// Exports a vertical dataset into a html table.

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_VERTICAL_DATASET_HTML_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_VERTICAL_DATASET_HTML_H_

#include <string>

#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// Print the dataset content as a html table. "id" is the html id of the
// table. The id should be non empty for the sorting to be enabled.

struct AppendHtmlOptions {
  int digit_precision = 4;
  // If true, the html should include sorttable.js defined in
  // "yggdrasil_decision_forests/data:html_report".
  bool interactive_column_sorting = false;
  // If true, the html should include vertical_dataset.js defined in
  // "yggdrasil_decision_forests/data:html_report".
  bool interactive_column_selection = false;
  // Html id of the table.
  std::string id = "";
};

void AppendVerticalDatasetToHtml(const VerticalDataset& dataset,
                                 const AppendHtmlOptions& options,
                                 std::string* html);

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_VERTICAL_DATASET_HTML_H_
