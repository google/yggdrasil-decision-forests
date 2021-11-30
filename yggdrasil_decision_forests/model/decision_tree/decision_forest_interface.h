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

// Interface (pure abstract) class for decision forest models.

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_FOREST_INTERFACE_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_FOREST_INTERFACE_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"

namespace yggdrasil_decision_forests {
namespace model {

class DecisionForestInterface {
 public:
  virtual ~DecisionForestInterface() = default;

  // Number of trees in the models.
  virtual int num_trees() const = 0;

  // Apply the model on an example in a VerticalDataset. Instead of the raw
  // prediction (like "AbstractModel::Predict"), this method returns the index
  // of the active leaf in each tree of the model.
  //
  // This method should be called with "leaves" containing exactly "num_trees"
  // elements.
  virtual absl::Status PredictGetLeaves(const dataset::VerticalDataset& dataset,
                                        dataset::VerticalDataset::row_t row_idx,
                                        absl::Span<int32_t> leaves) const = 0;
};

}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_FOREST_INTERFACE_H_
