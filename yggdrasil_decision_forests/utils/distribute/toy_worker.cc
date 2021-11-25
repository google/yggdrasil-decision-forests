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

#include "yggdrasil_decision_forests/utils/distribute/toy_worker.h"

namespace yggdrasil_decision_forests {
namespace distribute {

std::atomic<int> ToyWorker::num_existing_toy_workers_{0};
std::atomic<int> ToyWorker::max_num_existing_toy_workers_{0};

}  // namespace distribute
}  // namespace yggdrasil_decision_forests