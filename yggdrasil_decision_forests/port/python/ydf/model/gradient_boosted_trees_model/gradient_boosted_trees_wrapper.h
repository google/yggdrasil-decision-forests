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

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_MODEL_GRADIENT_BOOSTED_TREES_MODEL_GRADIENT_BOOSTED_TREES_WRAPPER_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_MODEL_GRADIENT_BOOSTED_TREES_MODEL_GRADIENT_BOOSTED_TREES_WRAPPER_H_

#include <pybind11/numpy.h>

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "ydf/model/decision_forest_model/decision_forest_wrapper.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests::port::python {

class GradientBoostedTreesCCModel : public DecisionForestCCModel {
  using YDFModel = ::yggdrasil_decision_forests::model::gradient_boosted_trees::
      GradientBoostedTreesModel;

 public:
  // Creates a GradientBoostedTreesCCModel if `model_ptr` refers to a
  // GradientBoostedTreesModel.
  //
  // If this method returns an invalid status, "model_ptr" is not modified.
  // If this method returns an ok status, the content of "model_ptr" is
  // moved (and "model_ptr" becomes empty).
  static absl::StatusOr<std::unique_ptr<GradientBoostedTreesCCModel>> Create(
      std::unique_ptr<model::AbstractModel>& model_ptr);

  // `model` and `rf_model` must point to the same object. Prefer using
  // GradientBoostedTreesCCModel::Compute for construction.
  GradientBoostedTreesCCModel(std::unique_ptr<YDFModel> model,
                              YDFModel* gbt_model)
      : DecisionForestCCModel(std::move(model), gbt_model),
        gbt_model_(gbt_model) {
    DCHECK_EQ(model_.get(), gbt_model_);
  }

  // Return's the model's validation loss.
  float validation_loss() const { return gbt_model_->validation_loss(); }

  py::array_t<float> initial_predictions() const;

 private:
  // This is a non-owning pointer to the model held by `model_`.
  YDFModel* gbt_model_;
};

}  // namespace yggdrasil_decision_forests::port::python

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_MODEL_GRADIENT_BOOSTED_TREES_MODEL_GRADIENT_BOOSTED_TREES_WRAPPER_H_
