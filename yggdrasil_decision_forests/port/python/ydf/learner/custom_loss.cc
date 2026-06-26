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

#include "ydf/learner/custom_loss.h"

#include <pybind11/numpy.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_custom_binary_classification.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_custom_multi_classification.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_custom_regression.h"
#include "ydf/utils/numpy_data.h"
#include "ydf/utils/pybind.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

#ifdef _WIN32
typedef std::ptrdiff_t ssize_t;
#endif

namespace yggdrasil_decision_forests::port::python {

namespace {

// Pybind11 automatically casts Numpy arrays of shape (2, x) to pairs /
// sequences of length 2. This can lead to subtle user error, so we explicitly
// disallow such arrays.
absl::Status CheckGradientAndHessianShape(const py::sequence& seq) {
  if (py::isinstance<py::array>(seq)) {
    return absl::InvalidArgumentError(
        "The gradient_and_hessian function returned a numpy array, expected a "
        "Sequence of two numpy arrays.");
  }
  if (seq.size() != 2) {
    return absl::InvalidArgumentError(absl::Substitute(
        "The gradient_and_hessian function returned a sequence of length $0. "
        "Expected a sequence of length 2",
        seq.size()));
  }
  return absl::OkStatus();
}

absl::Status Check1DArrayShape(const py::array_t<float>& arr,
                               absl::string_view arr_name,
                               size_t expected_size) {
  py::buffer_info py_result_buf = arr.request();
  if (py_result_buf.ndim != 1 || py_result_buf.shape.size() != 1 ||
      py_result_buf.shape[0] != expected_size) {
    return absl::InvalidArgumentError(absl::Substitute(
        "The $0 must be a one-dimensional "
        "Numpy array of $1 elements. Got $2-dimensional array of $3 elements.",
        arr_name, expected_size, py_result_buf.ndim, py_result_buf.size));
  }
  return absl::OkStatus();
}

absl::Status Check2DArrayShape(const py::array_t<float>& arr,
                               absl::string_view arr_name,
                               size_t expected_size_x, size_t expected_size_y) {
  py::buffer_info py_result_buf = arr.request();
  if (py_result_buf.ndim != 2 || py_result_buf.shape.size() != 2 ||
      py_result_buf.shape[0] != expected_size_x ||
      py_result_buf.shape[1] != expected_size_y) {
    return absl::InvalidArgumentError(absl::Substitute(
        "The $0 must be a two-dimensional Numpy array of shape ($1, $2)."
        "Got $3-dimensional array of $4 elements.",
        arr_name, expected_size_x, expected_size_y, py_result_buf.ndim,
        py_result_buf.size));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<model::gradient_boosted_trees::CustomRegressionLossFunctions>
CCRegressionLoss::ToCustomRegressionLossFunctions() const {
  auto safe_initial_predictions = MakeSafeGilHolder(initial_predictions);
  auto cc_initial_predictions =
      [safe_initial_predictions](
          const absl::Span<const float>& labels,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    return SafePythonCall<absl::StatusOr<float>>(
        "initial_predictions", [&]() -> absl::StatusOr<float> {
          auto pylabels = SpanToSafeCopy(labels);
          auto pyweights = SpanToSafeCopy(weights);
          return (*safe_initial_predictions)(pylabels, pyweights);
        });
  };
  auto safe_loss = MakeSafeGilHolder(loss);
  auto cc_loss =
      [safe_loss](
          const absl::Span<const float>& labels,
          const absl::Span<const float>& predictions,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    return SafePythonCall<absl::StatusOr<float>>(
        "loss", [&]() -> absl::StatusOr<float> {
          auto pylabels = SpanToSafeCopy(labels);
          auto pyweights = SpanToSafeCopy(weights);
          auto pypredictions = SpanToSafeCopy(predictions);
          return (*safe_loss)(pylabels, pypredictions, pyweights);
        });
  };
  auto safe_gradient_and_hessian = MakeSafeGilHolder(gradient_and_hessian);
  auto cc_gradient_and_hessian =
      [safe_gradient_and_hessian](
          const absl::Span<const float>& labels,
          const absl::Span<const float>& predictions,
          absl::Span<float> gradient,
          absl::Span<float> hessian) mutable -> absl::Status {
    return SafePythonCall("gradient_and_hessian", [&]() -> absl::Status {
      auto pylabels = SpanToSafeCopy(labels);
      auto pypredictions = SpanToSafeCopy(predictions);
      auto py_result = (*safe_gradient_and_hessian)(pylabels, pypredictions);
      RETURN_IF_ERROR(CheckGradientAndHessianShape(py_result));
      auto py_gradient =
          py_result[0].cast<py::array_t<float, py::array::forcecast>>();
      auto py_hessian =
          py_result[1].cast<py::array_t<float, py::array::forcecast>>();
      RETURN_IF_ERROR(
          Check1DArrayShape(py_gradient, "gradient", gradient.size()));
      RETURN_IF_ERROR(Check1DArrayShape(py_hessian, "hessian", hessian.size()));

      StridedSpanFloat32 py_gradient_accessor(py_gradient);
      StridedSpanFloat32 py_hessian_accessor(py_hessian);
      for (ssize_t example_idx = 0; example_idx < gradient.size();
           ++example_idx) {
        // TODO: Consider removing this copy (also for other losses).
        gradient[example_idx] = -py_gradient_accessor[example_idx];
        hessian[example_idx] = -py_hessian_accessor[example_idx];
      }
      return absl::OkStatus();
    });
  };
  return model::gradient_boosted_trees::CustomRegressionLossFunctions{
      cc_initial_predictions, cc_loss, cc_gradient_and_hessian};
}

absl::StatusOr<
    model::gradient_boosted_trees::CustomBinaryClassificationLossFunctions>
CCBinaryClassificationLoss::ToCustomBinaryClassificationLossFunctions() const {
  // Wrap the Python functions in a std::shared_ptr with a custom deleter
  // that acquires the GIL before deleting the underlying Python function.
  // This avoids dangling references and ensures safe out-of-scope destruction.
  auto safe_initial_predictions = MakeSafeGilHolder(initial_predictions);
  auto cc_initial_predictions =
      [safe_initial_predictions](
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    return SafePythonCall<absl::StatusOr<float>>(
        "initial_predictions", [&]() -> absl::StatusOr<float> {
          auto pylabels = SpanToSafeCopy(labels);
          auto pyweights = SpanToSafeCopy(weights);
          return (*safe_initial_predictions)(pylabels, pyweights);
        });
  };
  auto safe_loss = MakeSafeGilHolder(loss);
  auto cc_loss =
      [safe_loss](
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& predictions,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    return SafePythonCall<absl::StatusOr<float>>(
        "loss", [&]() -> absl::StatusOr<float> {
          auto pylabels = SpanToSafeCopy(labels);
          auto pyweights = SpanToSafeCopy(weights);
          auto pypredictions = SpanToSafeCopy(predictions);
          return (*safe_loss)(pylabels, pypredictions, pyweights);
        });
  };
  auto safe_gradient_and_hessian = MakeSafeGilHolder(gradient_and_hessian);
  auto cc_gradient_and_hessian =
      [safe_gradient_and_hessian](
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& predictions,
          absl::Span<float> gradient,
          absl::Span<float> hessian) mutable -> absl::Status {
    return SafePythonCall("gradient_and_hessian", [&]() -> absl::Status {
      auto pylabels = SpanToSafeCopy(labels);
      auto pypredictions = SpanToSafeCopy(predictions);
      auto py_result = (*safe_gradient_and_hessian)(pylabels, pypredictions);
      RETURN_IF_ERROR(CheckGradientAndHessianShape(py_result));
      auto py_gradient =
          py_result[0].cast<py::array_t<float, py::array::forcecast>>();
      auto py_hessian =
          py_result[1].cast<py::array_t<float, py::array::forcecast>>();
      RETURN_IF_ERROR(
          Check1DArrayShape(py_gradient, "gradient", gradient.size()));
      RETURN_IF_ERROR(Check1DArrayShape(py_hessian, "hessian", hessian.size()));

      StridedSpanFloat32 py_gradient_accessor(py_gradient);
      StridedSpanFloat32 py_hessian_accessor(py_hessian);
      for (ssize_t example_idx = 0; example_idx < gradient.size();
           ++example_idx) {
        gradient[example_idx] = -py_gradient_accessor[example_idx];
        hessian[example_idx] = -py_hessian_accessor[example_idx];
      }
      return absl::OkStatus();
    });
  };
  return model::gradient_boosted_trees::CustomBinaryClassificationLossFunctions{
      cc_initial_predictions, cc_loss, cc_gradient_and_hessian};
}

absl::StatusOr<
    model::gradient_boosted_trees::CustomMultiClassificationLossFunctions>
CCMultiClassificationLoss::ToCustomMultiClassificationLossFunctions() const {
  auto safe_initial_predictions = MakeSafeGilHolder(initial_predictions);
  auto cc_initial_predictions =
      [safe_initial_predictions](
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& weights,
          absl::Span<float> cc_initial_predictions) -> absl::Status {
    return SafePythonCall("initial_predictions", [&]() -> absl::Status {
      auto pylabels = SpanToSafeCopy(labels);
      auto pyweights = SpanToSafeCopy(weights);
      py::array_t<float> py_initial_predictions =
          (*safe_initial_predictions)(pylabels, pyweights);
      RETURN_IF_ERROR(Check1DArrayShape(py_initial_predictions,
                                        "initial_predictions",
                                        cc_initial_predictions.size()));
      StridedSpanFloat32 accessor(py_initial_predictions);
      for (ssize_t example_idx = 0; example_idx < cc_initial_predictions.size();
           ++example_idx) {
        cc_initial_predictions[example_idx] = accessor[example_idx];
      }
      return absl::OkStatus();
    });
  };
  auto safe_loss = MakeSafeGilHolder(loss);
  auto cc_loss =
      [safe_loss](
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& predictions,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    int num_examples = labels.size();
    DCHECK_GT(num_examples, 0);
    int dimension = predictions.size() / num_examples;
    return SafePythonCall<absl::StatusOr<float>>(
        "loss", [&]() -> absl::StatusOr<float> {
          auto pylabels = SpanToSafeCopy(labels);
          auto pyweights = SpanToSafeCopy(weights);
          auto pypredictions = SpanToSafeCopy(predictions);
          pypredictions = pypredictions.reshape({num_examples, dimension});
          return (*safe_loss)(pylabels, pypredictions, pyweights);
        });
  };
  auto safe_gradient_and_hessian = MakeSafeGilHolder(gradient_and_hessian);
  auto cc_gradient_and_hessian =
      [safe_gradient_and_hessian](
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& predictions,
          absl::Span<const absl::Span<float>> gradient,
          absl::Span<const absl::Span<float>> hessian) mutable -> absl::Status {
    int num_examples = labels.size();
    DCHECK_GT(num_examples, 0);
    int dimension = predictions.size() / num_examples;
    return SafePythonCall("gradient_and_hessian", [&]() -> absl::Status {
      auto pylabels = SpanToSafeCopy(labels);
      auto pypredictions = SpanToSafeCopy(predictions);
      pypredictions = pypredictions.reshape({num_examples, dimension});
      auto py_result = (*safe_gradient_and_hessian)(pylabels, pypredictions);
      RETURN_IF_ERROR(CheckGradientAndHessianShape(py_result));
      auto py_gradient =
          py_result[0].cast<py::array_t<float, py::array::forcecast>>();
      auto py_hessian =
          py_result[1].cast<py::array_t<float, py::array::forcecast>>();
      RETURN_IF_ERROR(
          Check2DArrayShape(py_gradient, "gradient", dimension, num_examples));
      RETURN_IF_ERROR(
          Check2DArrayShape(py_hessian, "hessian", dimension, num_examples));

      auto py_gradient_unchecked = py_gradient.unchecked<2>();
      auto py_hessian_unchecked = py_hessian.unchecked<2>();
      for (ssize_t example_idx = 0; example_idx < num_examples; ++example_idx) {
        for (int grad_idx = 0; grad_idx < dimension; grad_idx++) {
          gradient[grad_idx][example_idx] =
              -py_gradient_unchecked(grad_idx, example_idx);
          hessian[grad_idx][example_idx] =
              -py_hessian_unchecked(grad_idx, example_idx);
        }
      }
      return absl::OkStatus();
    });
  };
  return model::gradient_boosted_trees::CustomMultiClassificationLossFunctions{
      cc_initial_predictions, cc_loss, cc_gradient_and_hessian};
}

absl::Status ApplyCustomLoss(const CCCustomLoss& custom_loss,
                             model::AbstractLearner* learner) {
  if (std::holds_alternative<std::monostate>(custom_loss)) {
    return absl::OkStatus();
  }
  auto* gbt_learner =
      dynamic_cast<model::gradient_boosted_trees::GradientBoostedTreesLearner*>(
          learner);
  if (!gbt_learner) {
    return absl::InvalidArgumentError(
        "Custom losses are only compatible with Gradient Boosted Trees.");
  }
  if (std::holds_alternative<CCRegressionLoss>(custom_loss)) {
    ASSIGN_OR_RETURN(auto loss_func, std::get<CCRegressionLoss>(custom_loss)
                                         .ToCustomRegressionLossFunctions());
    gbt_learner->SetCustomLossFunctions(loss_func);
  } else if (std::holds_alternative<CCBinaryClassificationLoss>(custom_loss)) {
    ASSIGN_OR_RETURN(auto loss_func,
                     std::get<CCBinaryClassificationLoss>(custom_loss)
                         .ToCustomBinaryClassificationLossFunctions());
    gbt_learner->SetCustomLossFunctions(loss_func);
  } else if (std::holds_alternative<CCMultiClassificationLoss>(custom_loss)) {
    ASSIGN_OR_RETURN(auto loss_func,
                     std::get<CCMultiClassificationLoss>(custom_loss)
                         .ToCustomMultiClassificationLossFunctions());
    gbt_learner->SetCustomLossFunctions(loss_func);
  } else {
    NOTREACHED();
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::port::python
