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

#include "yggdrasil_decision_forests/port/javascript/training/util/status_casters.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif  // __EMSCRIPTEN__

#include <string>

#include "absl/status/status.h"

namespace yggdrasil_decision_forests::port::javascript {

namespace {
#ifdef __EMSCRIPTEN__
EM_JS(void, ThrowErrorWithMessage, (const char* message),
      { throw new Error(UTF8ToString(message)); });
#else
void ThrowErrorWithMessage(const char* message) {}
#endif  // __EMSCRIPTEN__
}  // namespace

void CheckOrThrowError(const absl::Status status) {
  if (!status.ok()) {
    std::string message = std::string(status.message());
    ThrowErrorWithMessage(message.c_str());
  }
}

}  // namespace yggdrasil_decision_forests::port::javascript
