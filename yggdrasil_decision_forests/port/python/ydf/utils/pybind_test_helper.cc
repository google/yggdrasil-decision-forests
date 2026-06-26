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

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <stdexcept>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ydf/utils/pybind.h"
#include "ydf/utils/status_casters.h"

namespace yggdrasil_decision_forests::port::python {
namespace {

namespace py = ::pybind11;

// Helper that calls SafePythonCall on a python callback.
// In the callback, if it throws a Python exception, SafePythonCall catches it
// and returns absl::Status. WithStatus will then throw a std::runtime_error if
// the returned status is not OK.
absl::Status CallSafePythonCallStatus(py::function func) {
  return SafePythonCall("test_func", [&]() {
    func();
    return absl::OkStatus();
  });
}

absl::StatusOr<int> CallSafePythonCallStatusOr(py::function func) {
  return SafePythonCall<absl::StatusOr<int>>("test_func_or", [&]() {
    return func().cast<int>();
  });
}

// Helper to test std::exception handling.
absl::Status CallSafePythonCallThrowStd() {
  return SafePythonCall("test_throw_std", []() -> absl::Status {
    throw std::runtime_error("C++ error");
  });
}

// Helper to test unknown exception handling.
absl::Status CallSafePythonCallThrowUnknown() {
  return SafePythonCall("test_throw_unknown", []() -> absl::Status {
    throw 42;
  });
}

// Helper to test MakeSafeGilHolder
class GilCheckNotifier {
 public:
  explicit GilCheckNotifier(py::function callback)
      : callback_(MakeSafeGilHolder(callback)) {}

  void Destroy() { callback_.reset(); }

 private:
  std::shared_ptr<py::function> callback_;
};

}  // namespace

PYBIND11_MODULE(pybind_test_helper, m) {
  m.def("call_safe_python_call_status", WithStatus(CallSafePythonCallStatus));
  m.def("call_safe_python_call_status_or",
        WithStatusOr(CallSafePythonCallStatusOr));
  m.def("call_safe_python_call_throw_std",
        WithStatus(CallSafePythonCallThrowStd));
  m.def("call_safe_python_call_throw_unknown",
        WithStatus(CallSafePythonCallThrowUnknown));

  py::class_<GilCheckNotifier>(m, "GilCheckNotifier")
      .def(py::init<py::function>())
      .def("destroy", &GilCheckNotifier::Destroy);
}

}  // namespace yggdrasil_decision_forests::port::python
