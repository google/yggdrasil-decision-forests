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

#include "yggdrasil_decision_forests/dataset/types.h"

#include "absl/status/status.h"
#include "absl/strings/substitute.h"

namespace yggdrasil_decision_forests {
namespace dataset {

absl::Status CheckNumExamples(size_t num_examples) {
  const auto max = std::numeric_limits<SignedExampleIdx>::max();
  if (num_examples > max) {
    return absl::InvalidArgumentError(absl::Substitute(
        "The dataset contains to many example ($0 > $1). Compile Yggdrasil "
        "Decision Forests with support for 64-bits example index with the "
        "following flag to train on more example: "
        "--define=ydf_example_idx_num_bits=64. Warning: 64-bits example index "
        "can increase up to 2x the RAM usage of YDF. Don't use it for datasets "
        "with less than 2^31 i.e. ~2B examples.",
        num_examples, max));
  }
  return absl::OkStatus();
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
