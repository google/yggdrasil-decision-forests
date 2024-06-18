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

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_MODEL_ISOLATION_FOREST_MODEL_RANDOM_FOREST_WRAPPER_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_MODEL_ISOLATION_FOREST_MODEL_RANDOM_FOREST_WRAPPER_H_

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/isolation_forest/isolation_forest.h"
#include "ydf/model/decision_forest_model/decision_forest_wrapper.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests::port::python {

class IsolationForestCCModel : public DecisionForestCCModel {
  using YDFModel = ::yggdrasil_decision_forests::model::isolation_forest::
      IsolationForestModel;

 public:
  // Creates a IsolationForestCCModel if `model_ptr` refers to a
  // IsolationForestModel.
  //
  // If this method returns an invalid status, "model_ptr" is not modified.
  // If this method returns an ok status, the content of "model_ptr" is moved
  // (and "model_ptr" becomes empty).
  static absl::StatusOr<std::unique_ptr<IsolationForestCCModel>> Create(
      std::unique_ptr<model::AbstractModel>& model_ptr);

  // `model` and `if_model` must point to the same object. Prefer using
  // IsolationForestCCModel::Compute for construction.
  IsolationForestCCModel(std::unique_ptr<YDFModel> model, YDFModel* if_model)
      : DecisionForestCCModel(std::move(model), if_model), if_model_(if_model) {
    DCHECK_EQ(model_.get(), if_model_);
  }

 private:
  // This is a non-owning pointer to the model held by `model_`.
  YDFModel* if_model_;
};

}  // namespace yggdrasil_decision_forests::port::python
#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_MODEL_ISOLATION_FOREST_MODEL_RANDOM_FOREST_WRAPPER_H_
