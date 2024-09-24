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

// Outputs Python source code defining Python wrapper around Yggdrasil Decision
// Forest learners.

#include "ydf/learner/wrapper/wrapper_generator.h"
#include "yggdrasil_decision_forests/utils/logging.h"

int main(int argc, char* argv[]) {
  // Enable the logging. Optional.
  InitLogging(argv[0], &argc, &argv, true);
  std::cout << yggdrasil_decision_forests::GenAllLearnersWrapper().value();
  return 0;
}
