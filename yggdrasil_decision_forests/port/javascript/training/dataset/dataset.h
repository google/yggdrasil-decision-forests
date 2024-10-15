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

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_JAVASCRIPT_DATASET_DATASET_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_JAVASCRIPT_DATASET_DATASET_H_

#include <string>
#include <vector>

#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"

namespace yggdrasil_decision_forests::port::javascript {

// Wrapper for a dataset::VerticalDataset.
class Dataset {
 public:
  Dataset() {}

  void AddNumericalColumn(const std::string& name,
                          const std::vector<float>& data);

  void AddCategoricalColumn(const std::string& name,
                            const std::vector<std::string>& data,
                            bool is_label);

  const dataset::VerticalDataset& dataset() const { return dataset_; }

 private:
  dataset::VerticalDataset dataset_;
};

void init_dataset();

}  // namespace yggdrasil_decision_forests::port::javascript

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_JAVASCRIPT_DATASET_DATASET_H_
