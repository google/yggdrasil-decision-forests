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

// A minimal version of the usage monitoring in google.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_MINI_USAGE_GOOGLE_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_MINI_USAGE_GOOGLE_H_

#include <cstdint>
#include <string_view>

namespace yggdrasil_decision_forests::utils::usage {

struct OnInferenceArg {
  int64_t num_examples = 1;
  std::string_view owner = {};
  uint64_t created_date = 0;
  uint64_t uid = 0;
  std::string_view framework = {};
};

// Called at model inference time.
void OnInference(const OnInferenceArg& arg);

}  // namespace yggdrasil_decision_forests::utils::usage

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_MINI_USAGE_GOOGLE_H_
