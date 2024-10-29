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
#include <exception>
#include <string>
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
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

#ifdef _WIN32
#include <cstddef>
typedef std::ptrdiff_t ssize_t;
#endif

namespace yggdrasil_decision_forests::port::python {

namespace {
absl::Status CheckRefCountIsNull(const py::object& py_ref,
                                 const std::string& ref_name) {
  if (py_ref.ref_count() > 1) {
    // Trigger GC - maybe we haven't collected yet?
    py::module_::import("gc").attr("collect")();
    if (py_ref.ref_count() > 1) {
      return absl::InternalError(absl::Substitute(
          "Cannot hold a reference to \"$0\" outside of a "
          "custom loss function. "
          "Currently holding $1 references. If this variable "
          "is required outside "
          "of the function, create a copy with np.copy($0). This check can be "
          "deactivated by setting `may_trigger_gc=False` on the custom loss "
          "object.",
          ref_name, py_ref.ref_count()));
    }
  }
  return absl::OkStatus();
};

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

// Returns a read-only Numpy array referencing the data in `data`. The array
// does not own the data. Its lifetime is tied to an no-op capsule. This is
// dangerous, since the C++ might delete `data`, which would lead to undefined
// behaviour. The caller must therefore make sure the returned array is no
// longer referenced when `data` is removed.
template <typename T>
py::array_t<T> SpanToUnsafeNumpyArray(absl::Span<T> data) {
  auto arr = py::array_t<T>(/*shape=*/{data.size()}, /*strides=*/{sizeof(T)},
                            /*ptr=*/data.data(),
                            py::capsule(data.data(), [](void* v) {}));
  py::detail::array_proxy(arr.ptr())->flags &=
      ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
  return arr;
}
}  // namespace

absl::StatusOr<model::gradient_boosted_trees::CustomRegressionLossFunctions>
CCRegressionLoss::ToCustomRegressionLossFunctions() const {
  // Pass the functions by copy to avoid dangling references here.
  // The Python functions only receive a view of the data and should therefore
  // not access the data after returning.
  auto cc_initial_predictions =
      [initial_predictions = initial_predictions](
          const absl::Span<const float>& labels,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    float current_initial_predictions;
    {
      py::gil_scoped_acquire acquire;
      auto pylabels = SpanToUnsafeNumpyArray(labels);
      auto pyweights = SpanToUnsafeNumpyArray(weights);
      try {
        current_initial_predictions = initial_predictions(pylabels, pyweights);
      } catch (const std::exception& e) {
        return absl::UnknownError(
            absl::Substitute("initial predictions raised: $0", e.what()));
      }
    }
    return current_initial_predictions;
  };
  auto cc_loss =
      [loss = loss](
          const absl::Span<const float>& labels,
          const absl::Span<const float>& predictions,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    float current_loss;
    {
      py::gil_scoped_acquire acquire;
      auto pylabels = SpanToUnsafeNumpyArray(labels);
      auto pyweights = SpanToUnsafeNumpyArray(weights);
      auto pypredictions = SpanToUnsafeNumpyArray(predictions);
      try {
        current_loss = loss(pylabels, pypredictions, pyweights);
      } catch (const std::exception& e) {
        return absl::UnknownError(
            absl::Substitute("loss raised: $0", e.what()));
      }
    }
    return current_loss;
  };
  auto cc_gradient_and_hessian =
      [gradient_and_hessian = gradient_and_hessian,
       may_trigger_gc = may_trigger_gc](
          const absl::Span<const float>& labels,
          const absl::Span<const float>& predictions,
          absl::Span<float> gradient,
          absl::Span<float> hessian) mutable -> absl::Status {
    {
      py::gil_scoped_acquire acquire;
      auto pylabels = SpanToUnsafeNumpyArray(labels);
      auto pypredictions = SpanToUnsafeNumpyArray(predictions);
      py::sequence py_result;
      try {
        py_result = gradient_and_hessian(pylabels, pypredictions);
      } catch (const std::exception& e) {
        return absl::UnknownError(
            absl::Substitute("gradient_and_hessian raised: $0", e.what()));
      }
      RETURN_IF_ERROR(CheckGradientAndHessianShape(py_result));
      auto py_gradient = py_result[0].cast<py::array_t<float>>();
      auto py_hessian = py_result[1].cast<py::array_t<float>>();
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
      if (may_trigger_gc) {
        RETURN_IF_ERROR(CheckRefCountIsNull(pylabels, "labels"));
        RETURN_IF_ERROR(CheckRefCountIsNull(pypredictions, "predictions"));
        // Only trigger GC once.
        may_trigger_gc = false;
      }
    }
    return absl::OkStatus();
  };
  return model::gradient_boosted_trees::CustomRegressionLossFunctions{
      cc_initial_predictions, cc_loss, cc_gradient_and_hessian};
}

absl::StatusOr<
    model::gradient_boosted_trees::CustomBinaryClassificationLossFunctions>
CCBinaryClassificationLoss::ToCustomBinaryClassificationLossFunctions() const {
  // Pass the functions by copy to avoid dangling references here.
  // The Python functions only receive a view of the data and should therefore
  // not access the data after returning.
  auto cc_initial_predictions =
      [initial_predictions = initial_predictions](
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    float current_initial_predictions;
    {
      py::gil_scoped_acquire acquire;
      auto pylabels = SpanToUnsafeNumpyArray(labels);
      auto pyweights = SpanToUnsafeNumpyArray(weights);
      try {
        current_initial_predictions = initial_predictions(pylabels, pyweights);
      } catch (const std::exception& e) {
        return absl::AbortedError(e.what());
      }
    }
    return current_initial_predictions;
  };
  auto cc_loss =
      [loss = loss](
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& predictions,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    float current_loss;
    {
      py::gil_scoped_acquire acquire;
      auto pylabels = SpanToUnsafeNumpyArray(labels);
      auto pyweights = SpanToUnsafeNumpyArray(weights);
      auto pypredictions = SpanToUnsafeNumpyArray(predictions);
      try {
        current_loss = loss(pylabels, pypredictions, pyweights);
      } catch (const std::exception& e) {
        return absl::AbortedError(e.what());
      }
    }
    return current_loss;
  };
  auto cc_gradient_and_hessian =
      [gradient_and_hessian = gradient_and_hessian,
       may_trigger_gc = may_trigger_gc](
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& predictions,
          absl::Span<float> gradient,
          absl::Span<float> hessian) mutable -> absl::Status {
    {
      py::gil_scoped_acquire acquire;
      auto pylabels = SpanToUnsafeNumpyArray(labels);
      auto pypredictions = SpanToUnsafeNumpyArray(predictions);
      py::sequence py_result;
      try {
        py_result = gradient_and_hessian(pylabels, pypredictions);
      } catch (const std::exception& e) {
        return absl::AbortedError(e.what());
      }
      RETURN_IF_ERROR(CheckGradientAndHessianShape(py_result));
      auto py_gradient = py_result[0].cast<py::array_t<float>>();
      auto py_hessian = py_result[1].cast<py::array_t<float>>();
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
      if (may_trigger_gc) {
        RETURN_IF_ERROR(CheckRefCountIsNull(pylabels, "labels"));
        RETURN_IF_ERROR(CheckRefCountIsNull(pypredictions, "predictions"));
        // Only trigger GC once.
        may_trigger_gc = false;
      }
    }
    return absl::OkStatus();
  };
  return model::gradient_boosted_trees::CustomBinaryClassificationLossFunctions{
      cc_initial_predictions, cc_loss, cc_gradient_and_hessian};
}

absl::StatusOr<
    model::gradient_boosted_trees::CustomMultiClassificationLossFunctions>
CCMultiClassificationLoss::ToCustomMultiClassificationLossFunctions() const {
  // Pass the functions by copy to avoid dangling references here.
  // The Python functions only receive a view of the data and should therefore
  // not access the data after returning.
  auto cc_initial_predictions =
      [initial_predictions = initial_predictions](
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& weights,
          absl::Span<float> cc_initial_predictions) -> absl::Status {
    {
      py::gil_scoped_acquire acquire;
      auto pylabels = SpanToUnsafeNumpyArray(labels);
      auto pyweights = SpanToUnsafeNumpyArray(weights);

      py::array_t<float> py_initial_predictions;
      try {
        py_initial_predictions = initial_predictions(pylabels, pyweights);
      } catch (const std::exception& e) {
        return absl::AbortedError(e.what());
      }

      RETURN_IF_ERROR(Check1DArrayShape(py_initial_predictions,
                                        "initial_predictions",
                                        cc_initial_predictions.size()));
      StridedSpanFloat32 accessor(py_initial_predictions);
      for (ssize_t example_idx = 0; example_idx < cc_initial_predictions.size();
           ++example_idx) {
        cc_initial_predictions[example_idx] = accessor[example_idx];
      }
    }
    return absl::OkStatus();
  };
  auto cc_loss =
      [loss = loss](
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& predictions,
          const absl::Span<const float>& weights) -> absl::StatusOr<float> {
    int num_examples = labels.size();
    DCHECK_GT(num_examples, 0);
    int dimension = predictions.size() / num_examples;
    float current_loss;
    {
      py::gil_scoped_acquire acquire;
      auto pylabels = SpanToUnsafeNumpyArray(labels);
      auto pyweights = SpanToUnsafeNumpyArray(weights);
      auto pypredictions = SpanToUnsafeNumpyArray(predictions);
      pypredictions = pypredictions.reshape({num_examples, dimension});
      try {
        current_loss = loss(pylabels, pypredictions, pyweights);
      } catch (const std::exception& e) {
        return absl::AbortedError(e.what());
      }
    }
    return current_loss;
  };
  auto cc_gradient_and_hessian =
      [gradient_and_hessian = gradient_and_hessian,
       may_trigger_gc = may_trigger_gc](
          const absl::Span<const int32_t>& labels,
          const absl::Span<const float>& predictions,
          absl::Span<const absl::Span<float>> gradient,
          absl::Span<const absl::Span<float>> hessian) mutable -> absl::Status {
    int num_examples = labels.size();
    DCHECK_GT(num_examples, 0);
    int dimension = predictions.size() / num_examples;
    {
      py::gil_scoped_acquire acquire;
      auto pylabels = SpanToUnsafeNumpyArray(labels);
      auto pypredictions = SpanToUnsafeNumpyArray(predictions);
      pypredictions = pypredictions.reshape({num_examples, dimension});
      py::sequence py_result;
      try {
        py_result = gradient_and_hessian(pylabels, pypredictions);
      } catch (const std::exception& e) {
        return absl::AbortedError(e.what());
      }
      RETURN_IF_ERROR(CheckGradientAndHessianShape(py_result));
      auto py_gradient = py_result[0].cast<py::array_t<float>>();
      auto py_hessian = py_result[1].cast<py::array_t<float>>();
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
      if (may_trigger_gc) {
        RETURN_IF_ERROR(CheckRefCountIsNull(pylabels, "labels"));
        RETURN_IF_ERROR(CheckRefCountIsNull(pypredictions, "predictions"));
        // Only trigger GC once.
        may_trigger_gc = false;
      }
    }
    return absl::OkStatus();
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
