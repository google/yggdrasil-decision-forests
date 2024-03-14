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

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_MODEL_CUSTOM_LOSS_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_MODEL_CUSTOM_LOSS_H_

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <functional>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_custom_binary_classification.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_custom_multi_classification.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_custom_regression.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {

struct CCRegressionLoss {
  using InitFunc = std::function<float(const py::array_t<float>&,
                                       const py::array_t<float>&)>;
  using LossFunc =
      std::function<float(const py::array_t<float>&, const py::array_t<float>&,
                          const py::array_t<float>&)>;
  using GradFunc = std::function<py::sequence(const py::array_t<float>&,
                                              const py::array_t<float>&)>;
  InitFunc initial_predictions;
  LossFunc loss;
  GradFunc gradient_and_hessian;
  bool may_trigger_gc;

  absl::StatusOr<model::gradient_boosted_trees::CustomRegressionLossFunctions>
  ToCustomRegressionLossFunctions() const;
};

struct CCBinaryClassificationLoss {
  using InitFunc = std::function<float(const py::array_t<int32_t>&,
                                       const py::array_t<float>&)>;
  using LossFunc = std::function<float(const py::array_t<int32_t>&,
                                       const py::array_t<float>&,
                                       const py::array_t<float>&)>;
  using GradFunc = std::function<py::sequence(const py::array_t<int32_t>&,
                                              const py::array_t<float>&)>;
  InitFunc initial_predictions;
  LossFunc loss;
  GradFunc gradient_and_hessian;
  bool may_trigger_gc;

  absl::StatusOr<
      model::gradient_boosted_trees::CustomBinaryClassificationLossFunctions>
  ToCustomBinaryClassificationLossFunctions() const;
};

struct CCMultiClassificationLoss {
  using InitFunc = std::function<py::array_t<float>(const py::array_t<int32_t>&,
                                                    const py::array_t<float>&)>;
  using LossFunc = std::function<float(const py::array_t<int32_t>&,
                                       const py::array_t<float>&,
                                       const py::array_t<float>&)>;
  using GradFunc = std::function<py::sequence(const py::array_t<int32_t>&,
                                              const py::array_t<float>&)>;
  InitFunc initial_predictions;
  LossFunc loss;
  GradFunc gradient_and_hessian;
  bool may_trigger_gc;

  absl::StatusOr<
      model::gradient_boosted_trees::CustomMultiClassificationLossFunctions>
  ToCustomMultiClassificationLossFunctions() const;
};
using CCCustomLoss =
    std::variant<std::monostate, CCRegressionLoss, CCBinaryClassificationLoss,
                 CCMultiClassificationLoss>;

// Adds a custom loss to the given (not owned) learner.
absl::Status ApplyCustomLoss(const CCCustomLoss& custom_loss,
                             model::AbstractLearner* learner);

}  // namespace yggdrasil_decision_forests::port::python

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_MODEL_CUSTOM_LOSS_H_
