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

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_UTILS_NUMPY_DATA_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_UTILS_NUMPY_DATA_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/utils/strided_span.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {

template <typename T>
using StridedSpan = utils::StridedSpan<T>;
using StridedSpanFloat = utils::StridedSpan<float>;
using StridedSpanInt = utils::StridedSpan<int>;

// Creates a non-owning 1D "utils::StridedSpan" view over "data". "name"
// identifies the array (e.g., a column name) in error messages.
//
// Returns an error if "data" is not one-dimensional, if its stride is not a
// multiple of its item size, or if its data is not aligned for "T".
//
// The "py::array_t" object and its underlying buffer must outlive the returned
// span.
template <typename T, int ExtraFlags>
absl::StatusOr<utils::StridedSpan<T>> StridedSpanFromNumpy(
    const py::array_t<T, ExtraFlags>& data, const absl::string_view name) {
  if (data.ndim() != 1) {
    return absl::InvalidArgumentError(absl::Substitute(
        "Invalid numpy array for \"$0\": Expected a 1-dimensional array, got "
        "$1 dimensions.",
        name, data.ndim()));
  }
  constexpr absl::string_view kCopyHint =
      " This can happen for fields of packed structured arrays, or for arrays "
      "created from a raw buffer (e.g., with np.frombuffer or np.ndarray). "
      "Pass a copy of the array instead, e.g., `x.copy()`.";
  if (data.strides(0) % data.itemsize() != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        absl::Substitute("Invalid numpy array for \"$0\": The stride ($1 "
                         "bytes) is not a multiple of the item size ($2 "
                         "bytes).",
                         name, data.strides(0), data.itemsize()),
        kCopyHint));
  }
  if (data.shape(0) > 0 &&
      reinterpret_cast<std::uintptr_t>(data.data()) % alignof(T) != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        absl::Substitute("Invalid numpy array for \"$0\": The data is not "
                         "aligned to $1 bytes.",
                         name, alignof(T)),
        kCopyHint));
  }
  return utils::StridedSpan<T>(data.data(), static_cast<size_t>(data.shape(0)),
                               data.strides(0) / data.itemsize());
}

template <typename T, int ExtraFlags>
absl::StatusOr<utils::StridedSpan<T>> StridedSpanFromNumpy(
    const py::array_t<T, ExtraFlags>&&, absl::string_view) = delete;

// Non owning accessor to a Numpy bytes array.
struct NPByteArray {
  // Wraps a single dimensional np::array of bytes.
  static absl::StatusOr<NPByteArray> Create(const py::array& data);

  // Number of items.
  size_t size() const { return _size; }

  // Value accessor.
  std::string_view operator[](size_t i) const;

  // Extracts the content of the numpy array into a string_view vector.
  std::vector<std::string_view> ToVectorNotOwned() const;

  const char* _data;
  const size_t _stride;
  const size_t _itemsize;
  const size_t _size;
};

template <typename T>
py::array_t<T> SpanToSafeCopy(absl::Span<const T> span) {
  py::array_t<T> arr({static_cast<py::ssize_t>(span.size())});
  if (!span.empty()) {
    std::memcpy(arr.mutable_data(), span.data(), span.size() * sizeof(T));
  }
  return arr;
}

}  // namespace yggdrasil_decision_forests::port::python

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_UTILS_NUMPY_DATA_H_
