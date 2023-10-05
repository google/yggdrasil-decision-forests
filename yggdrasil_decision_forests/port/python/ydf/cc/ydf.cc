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


#include <pybind11/pybind11.h>

#include "pybind11_abseil/import_status_module.h"
#include "pybind11_protobuf/native_proto_caster.h"
#include "ydf/dataset/dataset.h"
#include "ydf/learner/learner.h"
#include "ydf/model/model.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {

PYBIND11_MODULE(ydf, m) {
  pybind11_protobuf::ImportNativeProtoCasters();
  py::google::ImportStatusModule();
  m.doc() =
      "Wrappers for Yggdrasil Decision Forests, a library for training, "
      "serving, analyzing and evaluating decision forest models.";
  init_dataset(m);
  init_model(m);
  init_learner(m);
}

}  // namespace yggdrasil_decision_forests::port::python
