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

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_UTILS_PY_FUNC_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_UTILS_PY_FUNC_H_

#include <pybind11/pybind11.h>

#include <exception>
#include <memory>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"

namespace yggdrasil_decision_forests::port::python {

namespace py = ::pybind11;

// Executes a callable holding the Python GIL and catches any C++ or Pybind11
// exceptions, wrapping them into an absl::Status or absl::StatusOr with the
// function name as context.
template <typename ReturnType = absl::Status, typename Func>
ReturnType SafePythonCall(absl::string_view func_name, Func&& func) {
  py::gil_scoped_acquire acquire;
  try {
    return func();
  } catch (const std::exception& e) {
    return absl::UnknownError(absl::Substitute(
        "Python function '$0' raised: $1", func_name, e.what()));
  } catch (...) {
    return absl::UnknownError(absl::Substitute(
        "Python function '$0' raised an unknown C++ exception", func_name));
  }
}

// Holder for objects that require the Python GIL to be held upon destruction
// (such as py::function or pybind11 wrapped std::functions).
template <typename Func>
std::shared_ptr<Func> MakeSafeGilHolder(const Func& func) {
  return std::shared_ptr<Func>(new Func(func), [](Func* ptr) {
    if (ptr) {
      if (Py_IsInitialized()) {
        py::gil_scoped_acquire acquire;
        delete ptr;
      } else {
        // Intentionally leak the object to avoid a segfault.
      }
    }
  });
}

}  // namespace yggdrasil_decision_forests::port::python

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_UTILS_PY_FUNC_H_
