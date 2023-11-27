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
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {

// A collection of values, somehow similar to a "Span<const float>", but with a
// stride (i.e., items are evenly spaced, but possibly with a constant gap).
// Only support what is needed in this file.
//
// "StridedSpanFloat32" does not own the underlying data and relies on the data
// in "data". The initialization "py::array_t data" object should not be
// destroyed before "StridedSpanFloat32".
class StridedSpanFloat32 {
 public:
  // Build a "StridedSpanFloat32".
  //
  // Args:
  //   data: A one dimentionnal array of float32 values.
  StridedSpanFloat32(py::array_t<float>& data)
      : item_stride_(data.strides(0) / data.itemsize()),
        size_(static_cast<size_t>(data.shape(0))),
        values_(data.data()) {
    DCHECK_EQ(data.strides(0) % data.itemsize(), 0);
  }

  // Number of values.
  size_t size() const { return size_; }

  // Accesses to a value. "index" should be in [0, size).
  float operator[](const size_t index) const {
    DCHECK_LT(index, size_);
    return values_[index * item_stride_];
  }

 private:
  const size_t item_stride_;
  const size_t size_;
  const float* const values_;
};

// Non owning accessor to a Numpy bytes array.
struct NPByteArray {
  // Wraps a single dimensional np::array of bytes.
  static absl::StatusOr<NPByteArray> Create(const py::array& data);

  // Number of items.
  size_t size() const { return _size; }

  // Value accessor.
  std::string_view operator[](size_t i) const;

  // Extracts the content of the numpy array into a string vector.
  std::vector<std::string> ToVector() const;

  const char* _data;
  const size_t _stride;
  const size_t _itemsize;
  const size_t _size;
};

}  // namespace yggdrasil_decision_forests::port::python

#endif  // YGGDRASIL_DECISION_FORESTS_PORT_PYTHON_YDF_UTILS_NUMPY_DATA_H_
