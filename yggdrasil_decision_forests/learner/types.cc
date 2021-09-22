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

#include "yggdrasil_decision_forests/learner/types.h"

#include "absl/status/status.h"
#include "absl/strings/substitute.h"

namespace yggdrasil_decision_forests {
namespace model {

absl::Status CheckNumExamples(size_t num_examples) {
  const auto max = std::numeric_limits<SignedExampleIdx>::max();
  if (num_examples > max) {
    return absl::InvalidArgumentError(
        absl::Substitute("Too many training example ($0 > $1). Recompile the "
                         "binary with --define=example_idx_num_bits=64.",
                         num_examples, max));
  }
  return absl::OkStatus();
}

}  // namespace model
}  // namespace yggdrasil_decision_forests