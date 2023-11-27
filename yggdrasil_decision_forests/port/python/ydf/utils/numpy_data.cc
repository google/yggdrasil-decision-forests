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

#include "ydf/utils/numpy_data.h"

#include <pybind11/numpy.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {

namespace {

// Removes '\0' at the end of a string_view. Returns a string_view without the
// zeroes.
std::string_view remove_tailing_zeros(std::string_view src) {
  int i = static_cast<int>(src.size()) - 1;
  while (i >= 0 && src[i] == 0) {
    i--;
  }
  return src.substr(0, i + 1);
}

}  // namespace

absl::StatusOr<NPByteArray> NPByteArray::Create(const py::array& data) {
  if (data.dtype().kind() != 'S') {
    return absl::InternalError(
        absl::StrCat("Expecting a np.bytes (i.e. |S) array. Got |",
                     std::string(1, data.dtype().kind()), " instead"));
  }
  if (data.ndim() != 1) {
    return absl::InternalError("Wrong shape");
  }
  py::buffer_info info = data.request();
  return NPByteArray{
      /*_data=*/(char*)info.ptr,
      /*_stride=*/(size_t)info.strides[0],
      /*_itemsize=*/(size_t)info.itemsize,
      /*_size=*/(size_t)info.shape[0],
  };
}

std::string_view NPByteArray::operator[](size_t i) const {
  return remove_tailing_zeros({_data + i * _stride, _itemsize});
}

// Extracts the content of the numpy array into a string vector.
std::vector<std::string> NPByteArray::ToVector() const {
  std::vector<std::string> dst(_size);
  for (size_t i = 0; i < _size; i++) {
    dst[i] = (*this)[i];
  }
  return dst;
}

}  // namespace yggdrasil_decision_forests::port::python
