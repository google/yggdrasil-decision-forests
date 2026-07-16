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

#include "yggdrasil_decision_forests/model/postprocessor/abstract_postprocessor.h"

#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace postprocessor {

void AbstractPostprocessor::Process(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t row_idx,
    yggdrasil_decision_forests::model::proto::Prediction* prediction) const {
  if (enabled_) {
    ProcessImpl(dataset, row_idx, prediction);
  }
}

void AbstractPostprocessor::Process(
    const dataset::proto::Example& example,
    yggdrasil_decision_forests::model::proto::Prediction* prediction) const {
  if (enabled_) {
    ProcessImpl(example, prediction);
  }
}

void AbstractPostprocessor::ExportProto(
    proto::AbstractPostprocessor* proto) const {
  proto->set_enabled(enabled_);
  // TODO: Pass in only the one-of field here.
  ExportProtoImpl(proto);
}

}  // namespace postprocessor
}  // namespace model
}  // namespace yggdrasil_decision_forests
