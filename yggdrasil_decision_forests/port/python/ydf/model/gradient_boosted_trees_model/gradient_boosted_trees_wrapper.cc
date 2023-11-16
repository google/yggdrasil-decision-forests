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

#include "ydf/model/gradient_boosted_trees_model/gradient_boosted_trees_wrapper.h"

#include <pybind11/numpy.h>

#include <cstring>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"

namespace yggdrasil_decision_forests::port::python {

absl::StatusOr<std::unique_ptr<GradientBoostedTreesCCModel>>
GradientBoostedTreesCCModel::Create(
    std::unique_ptr<model::AbstractModel>& model_ptr) {
  auto* gbt_model = dynamic_cast<YDFModel*>(model_ptr.get());
  if (gbt_model == nullptr) {
    return absl::InvalidArgumentError(
        "This model is not a gradient boosted trees model.");
  }
  // Both release and the unique_ptr constructor are noexcept.
  model_ptr.release();
  std::unique_ptr<YDFModel> new_model_ptr(gbt_model);

  return std::make_unique<GradientBoostedTreesCCModel>(std::move(new_model_ptr),
                                                       gbt_model);
}

py::array_t<float> GradientBoostedTreesCCModel::initial_predictions() const {
  py::array_t<float, py::array::c_style | py::array::forcecast>
      initial_predictions;
  const auto& gbt_initial_predictions = gbt_model_->initial_predictions();
  static_assert(initial_predictions.itemsize() == sizeof(float),
                "A C++ float should have the same size as a numpy float");
  initial_predictions.resize({gbt_initial_predictions.size()});
  std::memcpy(initial_predictions.mutable_data(),
              gbt_initial_predictions.data(),
              initial_predictions.size() * sizeof(float));
  return initial_predictions;
}

}  // namespace yggdrasil_decision_forests::port::python
