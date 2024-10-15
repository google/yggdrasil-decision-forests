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

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#include <emscripten/emscripten.h>
#endif  // __EMSCRIPTEN__

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/port/javascript/training/dataset/dataset.h"
#include "yggdrasil_decision_forests/port/javascript/training/learner/learner.h"
#include "yggdrasil_decision_forests/port/javascript/training/model/model.h"
#include "yggdrasil_decision_forests/port/javascript/training/util/status_casters.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests::port::javascript {

namespace {

std::vector<std::string> CreateVectorString(size_t reserved) {
  std::vector<std::string> v;
  v.reserve(reserved);
  return v;
}

std::vector<int> CreateVectorInt(size_t reserved) {
  std::vector<int> v;
  v.reserve(reserved);
  return v;
}

std::vector<float> CreateVectorFloat(size_t reserved) {
  std::vector<float> v;
  v.reserve(reserved);
  return v;
}
}  // namespace

#ifdef __EMSCRIPTEN__
// Expose some of the class/functions to JS.
EMSCRIPTEN_BINDINGS(m) {
  init_dataset();
  init_model();
  init_learner();

  emscripten::function("CreateVectorString", &CreateVectorString);
  emscripten::function("CreateVectorInt", &CreateVectorInt);
  emscripten::function("CreateVectorFloat", &CreateVectorFloat);

  emscripten::register_vector<float>("vectorFloat");
  emscripten::register_vector<int>("vectorInt");
  emscripten::register_vector<std::vector<float>>("vectorVectorFloat");
  emscripten::register_vector<std::string>("vectorString");
}
#endif  // __EMSCRIPTEN__
}  // namespace yggdrasil_decision_forests::port::javascript
