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

#include <atomic>
#include <csignal>
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
#include "yggdrasil_decision_forests/utils/logging.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {
namespace {

// Number of learners training.
std::atomic<int> active_learners{0};

// If true, all stop all the active trainings.
std::atomic<bool> stop_training = false;

// Existing interruption signal handler (if any).
void (*existing_signal_handler_int)(int) = nullptr;

#ifndef _WIN32
void (*existing_signal_handler_alarm)(int) = nullptr;
#endif

void ReceiveSigna(int signal) {
  if (!stop_training) {
    YDF_LOG(INFO) << "Stopping all active trainings";
    stop_training = true;
  } else {
    // Pass the signal to any existing handler.
    if (signal == SIGINT && existing_signal_handler_int) {
      existing_signal_handler_int(signal);
    }
    // SIGALRM does not exist on Windows.
#ifndef _WIN32
    if (signal == SIGALRM && existing_signal_handler_alarm) {
      existing_signal_handler_alarm(signal);
    }
#endif
  }
}

void EnableUserInterruption() {
  // One more model is training
  if (active_learners.fetch_add(1) == 0) {
    existing_signal_handler_int = std::signal(SIGINT, ReceiveSigna);
    if (existing_signal_handler_int == SIG_ERR) {
      YDF_LOG(WARNING) << "Cannot set SIGINT handler";
    }
#ifndef _WIN32
    existing_signal_handler_alarm = std::signal(SIGALRM, ReceiveSigna);
    if (existing_signal_handler_alarm == SIG_ERR) {
      YDF_LOG(WARNING) << "Cannot set SIGALRM handler";
    }
#endif
  }
}

void DisableUserInterruption() {
  // Reset the stop training flag and remove the signal handler if all the
  // models are done training
  if (active_learners.fetch_sub(1) != 1) {
    return;
  }

  stop_training = false;
  if (existing_signal_handler_int &&
      std::signal(SIGINT, existing_signal_handler_int) == SIG_ERR) {
    YDF_LOG(WARNING) << "Cannot unset SIGINT handler";
  }
#ifndef _WIN32
  if (existing_signal_handler_alarm &&
      std::signal(SIGALRM, existing_signal_handler_alarm) == SIG_ERR) {
    YDF_LOG(WARNING) << "Cannot unset SIGALRM handler";
  }
#endif
}

class GenericCCLearner {
 public:
  GenericCCLearner(std::unique_ptr<model::AbstractLearner>&& learner)
      : learner_(std::move(learner)) {}

  virtual ~GenericCCLearner() {}

  absl::StatusOr<std::unique_ptr<GenericCCModel>> Train(
      const dataset::VerticalDataset& dataset) const {
    EnableUserInterruption();
    ASSIGN_OR_RETURN(auto model, learner_->TrainWithStatus(dataset));
    DisableUserInterruption();
    return CreateCCModel(std::move(model));
  }

  absl::Status SetHyperParameters(
      const model::proto::GenericHyperParameters& generic_hyper_params) {
    return learner_->SetHyperParameters(generic_hyper_params);
  }

  absl::StatusOr<metric::proto::EvaluationResults> Evaluate(
      const dataset::VerticalDataset& dataset,
      const utils::proto::FoldGenerator& fold_generator,
      const metric::proto::EvaluationOptions& evaluation_options,
      const model::proto::DeploymentConfig& deployment_evaluation) const {
    EnableUserInterruption();
    ASSIGN_OR_RETURN(auto evaluation,
                     model::EvaluateLearnerOrStatus(
                         *learner_, dataset, fold_generator, evaluation_options,
                         deployment_evaluation));
    DisableUserInterruption();
    return evaluation;
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
  learner_ptr->set_stop_training_trigger(&stop_training);
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
      .def("Train", &GenericCCLearner::Train, py::arg("dataset"))
      .def("Evaluate", &GenericCCLearner::Evaluate, py::arg("dataset"),
           py::arg("fold_generator"), py::arg("evaluation_options"),
           py::arg("deployment_evaluation"));
}

}  // namespace yggdrasil_decision_forests::port::python
