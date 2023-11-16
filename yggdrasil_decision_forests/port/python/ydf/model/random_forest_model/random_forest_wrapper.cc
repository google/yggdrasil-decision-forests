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

#include "ydf/model/random_forest_model/random_forest_wrapper.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"

namespace yggdrasil_decision_forests::port::python {

absl::StatusOr<std::unique_ptr<RandomForestCCModel>>
RandomForestCCModel::Create(std::unique_ptr<model::AbstractModel>& model_ptr) {
  auto* rf_model = dynamic_cast<YDFModel*>(model_ptr.get());
  if (rf_model == nullptr) {
    return absl::InvalidArgumentError(
        "This model is not a random forest model.");
  }
  // Both release and the unique_ptr constructor are noexcept.
  model_ptr.release();
  std::unique_ptr<YDFModel> new_model_ptr(rf_model);

  return std::make_unique<RandomForestCCModel>(std::move(new_model_ptr),
                                               rf_model);
}

}  // namespace yggdrasil_decision_forests::port::python
