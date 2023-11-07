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

// Documentation URLs for C++ code.
//
// See also:
// yggdrasil_decision_forests/port/python/utils/documentation.py
//

#include <string>

#include "absl/strings/str_cat.h"

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_DOCUMENTATION_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_DOCUMENTATION_H_

namespace yggdrasil_decision_forests::utils::documentation {

constexpr char URL_DOCUMENTATION[] = "https://ydf.readthedocs.io/en/latest";

#define DOC(NAME, PATH) \
  inline std::string NAME() { return absl::StrCat(URL_DOCUMENTATION, PATH); }

DOC(Glossary, "/glossary");
DOC(GlossaryConfusionMatrix, "/glossary#confusion-matrix");
DOC(GlossaryNumExamples, "/glossary#number-of-examples");
DOC(GlossaryWeightedNumExamples, "/glossary#weighted-number-of-examples");
DOC(GlossaryTuner, "/glossary#tuner");  // TODO: Write
DOC(VariableImportance, "/cli_user_manual#variable-importances");

}  // namespace yggdrasil_decision_forests::utils::documentation

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_DOCUMENTATION_H_
