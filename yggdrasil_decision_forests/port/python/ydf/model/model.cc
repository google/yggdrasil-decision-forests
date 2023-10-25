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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/native_proto_caster.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "ydf/model/model_wrapper.h"
#include "yggdrasil_decision_forests/utils/benchmark/inference.h"
#include "yggdrasil_decision_forests/utils/model_analysis.h"
#include "yggdrasil_decision_forests/utils/model_analysis.pb.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {
namespace {

absl::StatusOr<std::unique_ptr<GenericCCModel>> LoadModel(
    const std::string& directory,
    const std::optional<std::string>& file_prefix) {
  std::unique_ptr<model::AbstractModel> model_ptr;
  RETURN_IF_ERROR(model::LoadModel(directory, &model_ptr, {file_prefix}));
  return CreateCCModel(std::move(model_ptr));
}

absl::StatusOr<std::string> ModelAnalysisCreateHtmlReport(
    const utils::model_analysis::proto::StandaloneAnalysisResult& analysis,
    const utils::model_analysis::proto::Options& options = {}) {
  return utils::model_analysis::CreateHtmlReport(analysis, options);
}

}  // namespace

void init_model(py::module_& m) {
  m.def("LoadModel", LoadModel, py::arg("directory"), py::arg("file_prefix"));
  m.def("ModelAnalysisCreateHtmlReport", ModelAnalysisCreateHtmlReport,
        py::arg("analysis"), py::arg("options"));
  py::class_<GenericCCModel>(m, "GenericCCModel")
      .def("__repr__",
           [](const GenericCCModel& a) {
             return absl::Substitute("<model_cc.GenericCCModel of type $0.",
                                     a.name());
           })
      .def("Predict", &GenericCCModel::Predict, py::arg("dataset"))
      .def("Evaluate", &GenericCCModel::Evaluate, py::arg("dataset"),
           py::arg("options"))
      .def("Analyze", &GenericCCModel::Analyze, py::arg("dataset"),
           py::arg("options"))
      .def("Save", &GenericCCModel::Save, py::arg("directory"),
           py::arg("file_prefix"))
      .def("name", &GenericCCModel::name)
      .def("task", &GenericCCModel::task)
      .def("data_spec", &GenericCCModel::data_spec)
      .def("Describe", &GenericCCModel::Describe, py::arg("full_details"))
      .def("input_features", &GenericCCModel::input_features)
      .def("hyperparameter_optimizer_logs",
           &GenericCCModel::hyperparameter_optimizer_logs)
      .def("Benchmark", &GenericCCModel::Benchmark, py::arg("dataset"),
           py::arg("benchmark_duration"), py::arg("warmup_duration"),
           py::arg("batch_size"));

  py::class_<BenchmarkInferenceCCResult>(m, "BenchmarkInferenceCCResult")
      .def_readwrite("duration_per_example",
                     &BenchmarkInferenceCCResult::duration_per_example)
      .def_readwrite("benchmark_duration",
                     &BenchmarkInferenceCCResult::benchmark_duration)
      .def_readwrite("num_runs",
                     &BenchmarkInferenceCCResult::num_runs)
      .def_readwrite("batch_size", &BenchmarkInferenceCCResult::batch_size)
      .def("__repr__",
           [](const BenchmarkInferenceCCResult& a) { return a.ToString(); })
      .doc() = R"(Results of the inference benchmark.

  Attributes:
      duration_per_example: Average duration per example in seconds.
      benchmark_duration: Total duration of the benchmark run without warmup
        runs in seconds.
      num_runs: Number of times the benchmark fully ran over all
        the examples of the dataset. Warmup runs are not included.
      batch_size: Number of examples per batch used when benchmarking.)";

  py::class_<DecisionForestCCModel,
             /*parent class*/ GenericCCModel>(m, "DecisionForestCCModel")
      .def("__repr__",
           [](const DecisionForestCCModel& a) {
             return absl::Substitute(
                 "<model_cc.DecisionForestCCModel of type $0.", a.name());
           })
      .def("num_trees", &DecisionForestCCModel::num_trees)
      .def("PredictLeaves", &DecisionForestCCModel::PredictLeaves)
      .def("Distance", &DecisionForestCCModel::Distance, py::arg("dataset1"),
           py::arg("dataset2"));

  py::class_<RandomForestCCModel,
             /*parent class*/ DecisionForestCCModel>(m, "RandomForestCCModel")
      .def("__repr__",
           [](const GenericCCModel& a) {
             return absl::Substitute(
                 "<model_cc.RandomForestCCModel of type $0.", a.name());
           })
      .def_property_readonly_static(
          "kRegisteredName", [](py::object /* self */) {
            return model::random_forest::RandomForestModel::kRegisteredName;
          });

  py::class_<GradientBoostedTreesCCModel,
             /*parent class*/ DecisionForestCCModel>(
      m, "GradientBoostedTreesCCModel")
      .def("__repr__",
           [](const GenericCCModel& a) {
             return absl::Substitute(
                 "<model_cc.GradientBoostedTreesCCModel of type $0.", a.name());
           })
      .def("validation_loss", &GradientBoostedTreesCCModel::validation_loss)
      .def_property_readonly_static(
          "kRegisteredName", [](py::object /* self */) {
            return model::gradient_boosted_trees::GradientBoostedTreesModel::
                kRegisteredName;
          });
}

}  // namespace yggdrasil_decision_forests::port::python
