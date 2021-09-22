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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPLITTER_STRUCTURE_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPLITTER_STRUCTURE_H_

#include <stdint.h>

#include <limits>

#include "yggdrasil_decision_forests/learner/types.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

// Item used to pass the sorted numerical attribute values to the
// "ScanSplitsPresortedSparseDuplicateExampleTemplate" function below.
struct SparseItem {
  typedef UnsignedExampleIdx ExampleIdx;
  static constexpr auto kMaxNumExamples =
      std::numeric_limits<SignedExampleIdx>::max();

  // Index of the example in the training dataset.
  // The highest bit is 1 iif. the feature value of this item is strictly
  // greater than the preceding one.
  ExampleIdx example_idx_and_extra;
};

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_SPLITTER_STRUCTURE_H_
