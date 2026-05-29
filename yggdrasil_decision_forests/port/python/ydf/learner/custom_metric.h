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

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_LEARNER_CUSTOM_METRIC_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_LEARNER_CUSTOM_METRIC_H_

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {

struct CCBinaryClassificationMetric {
  using MetricFunc = std::function<float(const py::array_t<int32_t>&,
                                         const py::array_t<float>&,
                                         const py::array_t<float>&)>;

  std::string name;
  MetricFunc evaluation_func;

  absl::StatusOr<model::gradient_boosted_trees::CustomMetric> ToCustomMetric()
      const;
};

struct CCMultiClassificationMetric {
  using MetricFunc = std::function<float(const py::array_t<int32_t>&,
                                         const py::array_t<float>&,
                                         const py::array_t<float>&)>;

  std::string name;
  MetricFunc evaluation_func;

  absl::StatusOr<model::gradient_boosted_trees::CustomMetric> ToCustomMetric()
      const;
};

struct CCRegressionMetric {
  using MetricFunc =
      std::function<float(const py::array_t<float>&, const py::array_t<float>&,
                          const py::array_t<float>&)>;

  std::string name;
  MetricFunc evaluation_func;

  absl::StatusOr<model::gradient_boosted_trees::CustomMetric> ToCustomMetric()
      const;
};

using CCCustomMetric =
    std::variant<std::monostate, CCBinaryClassificationMetric,
                 CCMultiClassificationMetric, CCRegressionMetric>;

// Adds a custom metric to the given (not owned) learner.
absl::Status ApplyCustomMetric(
    const std::vector<CCCustomMetric>& custom_metrics,
    model::AbstractLearner* learner);

}  // namespace yggdrasil_decision_forests::port::python

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_LEARNER_CUSTOM_METRIC_H_
