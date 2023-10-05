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

#include "ydf/learner/learner.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/native_proto_caster.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/hyperparameter.pb.h"
#include "ydf/model/model_wrapper.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {
namespace {

class GenericCCLearner {
 public:
  GenericCCLearner(std::unique_ptr<model::AbstractLearner>&& learner)
      : learner_(std::move(learner)) {}

  virtual ~GenericCCLearner() {}

  absl::StatusOr<std::unique_ptr<GenericCCModel>> Train(
      const dataset::VerticalDataset& dataset) const {
    ASSIGN_OR_RETURN(std::unique_ptr<model::AbstractModel> raw_model,
                     learner_->TrainWithStatus(dataset));
    return CreateCCModel(std::move(raw_model));
  }

  absl::Status SetHyperParameters(
      const model::proto::GenericHyperParameters& generic_hyper_params) {
    return learner_->SetHyperParameters(generic_hyper_params);
  }

 protected:
  std::unique_ptr<model::AbstractLearner> learner_;
};

// Create a learner just from the training config and the
absl::StatusOr<std::unique_ptr<GenericCCLearner>> GetLearner(
    const model::proto::TrainingConfig& train_config,
    const model::proto::GenericHyperParameters& hyperparameters,
    const model::proto::DeploymentConfig& deployment_config) {
  std::unique_ptr<model::AbstractLearner> learner_ptr;
  RETURN_IF_ERROR(
      model::GetLearner(train_config, &learner_ptr, deployment_config));
  RETURN_IF_ERROR(learner_ptr->SetHyperParameters(hyperparameters));
  return std::make_unique<GenericCCLearner>(std::move(learner_ptr));
}

}  // namespace

void init_learner(py::module_& m) {
  m.def("GetLearner", &GetLearner);
  py::class_<GenericCCLearner>(m, "GenericCCLearner")
      .def("__repr__",
           [](const GenericCCLearner& a) {
             return "<learner_cc.GenericCCLearner";
           })
      .def("Train", &GenericCCLearner::Train);
}

}  // namespace yggdrasil_decision_forests::port::python
