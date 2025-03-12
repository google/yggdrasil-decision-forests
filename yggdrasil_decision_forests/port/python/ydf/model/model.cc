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

#include "ydf/model/model.h"

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
#include "pybind11_protobuf/native_proto_caster.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/isolation_forest/isolation_forest.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "ydf/model/decision_forest_model/decision_forest_wrapper.h"
#include "ydf/model/gradient_boosted_trees_model/gradient_boosted_trees_wrapper.h"
#include "ydf/model/isolation_forest_model/isolation_forest_wrapper.h"
#include "ydf/model/model_wrapper.h"
#include "ydf/model/random_forest_model/random_forest_wrapper.h"
#include "ydf/utils/custom_casters.h"
#include "ydf/utils/status_casters.h"
#include "yggdrasil_decision_forests/utils/benchmark/inference.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/model_analysis.h"
#include "yggdrasil_decision_forests/utils/model_analysis.pb.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {
namespace {

absl::StatusOr<std::unique_ptr<GenericCCModel>> LoadModel(
    const std::string& directory,
    const std::optional<std::string>& file_prefix) {
  py::gil_scoped_release release;
  ASSIGN_OR_RETURN(auto model, model::LoadModel(directory, {file_prefix}));
  return CreateCCModel(std::move(model));
}

absl::StatusOr<std::unique_ptr<GenericCCModel>> DeserializeModel(
    const py::bytes& data) {
  ASSIGN_OR_RETURN(auto model, model::DeserializeModel(data));
  return CreateCCModel(std::move(model));
}

absl::StatusOr<std::string> ModelAnalysisCreateHtmlReport(
    const utils::model_analysis::proto::StandaloneAnalysisResult& analysis,
    const utils::model_analysis::proto::Options& options = {}) {
  return utils::model_analysis::CreateHtmlReport(analysis, options);
}

absl::StatusOr<std::string> PredictionAnalysisCreateHtmlReport(
    const utils::model_analysis::proto::PredictionAnalysisResult& analysis,
    const utils::model_analysis::proto::PredictionAnalysisOptions& options =
        {}) {
  return utils::model_analysis::CreateHtmlReport(analysis, options);
}

}  // namespace

std::unique_ptr<GenericCCModel> CreateCCModel(
    std::unique_ptr<model::AbstractModel> model_ptr) {
  auto rf_model = RandomForestCCModel::Create(model_ptr);
  if (rf_model.ok()) {
    // `model_ptr` is now invalid.
    return std::move(rf_model.value());
  }
  auto gbt_model = GradientBoostedTreesCCModel::Create(model_ptr);
  if (gbt_model.ok()) {
    // `model_ptr` is now invalid.
    return std::move(gbt_model.value());
  }
  auto if_model = IsolationForestCCModel::Create(model_ptr);
  if (if_model.ok()) {
    // `model_ptr` is now invalid.
    return std::move(if_model.value());
  }
  // `model_ptr` is still valid.
  return std::make_unique<GenericCCModel>(std::move(model_ptr));
}

void init_model(py::module_& m) {
  // WARNING: This method releases the Global Interpreter Lock.
  m.def("LoadModel", WithStatusOr(LoadModel), py::arg("directory"),
        py::arg("file_prefix"));
  m.def("DeserializeModel", WithStatusOr(DeserializeModel), py::arg("data"));
  m.def("ModelAnalysisCreateHtmlReport",
        WithStatusOr(ModelAnalysisCreateHtmlReport), py::arg("analysis"),
        py::arg("options"));
  m.def("PredictionAnalysisCreateHtmlReport",
        WithStatusOr(PredictionAnalysisCreateHtmlReport), py::arg("analysis"),
        py::arg("options"));
  py::class_<GenericCCModel>(m, "GenericCCModel")
      .def("__repr__",
           [](const GenericCCModel& a) {
             return absl::Substitute("<model_cc.GenericCCModel of type $0.",
                                     a.name());
           })
      // WARNING: This method releases the Global Interpreter Lock.
      .def("Predict", WithStatusOr(&GenericCCModel::Predict),
           py::arg("dataset"), py::arg("use_slow_engine"),
           py::arg("num_threads"))
      // WARNING: This method releases the Global Interpreter Lock.
      .def("Evaluate", WithStatusOr(&GenericCCModel::Evaluate),
           py::arg("dataset"), py::arg("options"), py::arg("weighted"),
           py::arg("label_col_idx"), py::arg("group_col_idx"),
           py::arg("use_slow_engine"), py::arg("num_threads"))
      // WARNING: This method releases the Global Interpreter Lock.
      .def("Analyze", WithStatusOr(&GenericCCModel::Analyze),
           py::arg("dataset"), py::arg("options"))
      // WARNING: This method releases the Global Interpreter Lock.
      .def("AnalyzePrediction",
           WithStatusOr(&GenericCCModel::AnalyzePrediction), py::arg("dataset"),
           py::arg("options"))
      // WARNING: This method releases the Global Interpreter Lock.
      .def("Save", WithStatus(&GenericCCModel::Save), py::arg("directory"),
           py::arg("file_prefix"), py::arg("pure_serving"))
      .def("Serialize", WithStatusOr(&GenericCCModel::Serialize))
      .def("name", &GenericCCModel::name)
      .def("task", &GenericCCModel::task)
      .def("data_spec", &GenericCCModel::data_spec)
      .def("set_data_spec", &GenericCCModel::set_data_spec,
           py::arg("data_spec"))
      .def("label_col_idx", &GenericCCModel::label_col_idx)
      .def("group_col_idx", &GenericCCModel::group_col_idx)
      .def("weight_col_idx", &GenericCCModel::weight_col_idx)
      .def("metadata", &GenericCCModel::metadata)
      .def("set_metadata", &GenericCCModel::set_metadata, py::arg("metadata"))
      .def("Describe", WithStatusOr(&GenericCCModel::Describe),
           py::arg("full_details"), py::arg("text_format"))
      .def("input_features", &GenericCCModel::input_features)
      .def("hyperparameter_optimizer_logs",
           &GenericCCModel::hyperparameter_optimizer_logs)
      // WARNING: This method releases the Global Interpreter Lock.
      .def("Benchmark", WithStatusOr(&GenericCCModel::Benchmark),
           py::arg("dataset"), py::arg("benchmark_duration"),
           py::arg("warmup_duration"), py::arg("batch_size"),
           py::arg("num_threads"))
      .def("VariableImportances",
           WithStatusOr(&GenericCCModel::VariableImportances))
      .def("ForceEngine", &GenericCCModel::ForceEngine, py::arg("engine_name"))
      .def("ListCompatibleEngines", &GenericCCModel::ListCompatibleEngines)
      // TODO: Remove when solved.
      .def("weighted_training", &GenericCCModel::weighted_training)
      .def("set_feature_selection_logs",
           &GenericCCModel::set_feature_selection_logs)
      .def("feature_selection_logs", &GenericCCModel::feature_selection_logs);

  py::class_<BenchmarkInferenceCCResult>(m, "BenchmarkInferenceCCResult")
      .def_readwrite("duration_per_example",
                     &BenchmarkInferenceCCResult::duration_per_example)
      .def_readwrite("benchmark_duration",
                     &BenchmarkInferenceCCResult::benchmark_duration)
      .def_readwrite("num_runs", &BenchmarkInferenceCCResult::num_runs)
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
      .def("set_node_format", &DecisionForestCCModel::set_node_format,
           py::arg("node_format"))
      .def("PredictLeaves", WithStatusOr(&DecisionForestCCModel::PredictLeaves),
           py::arg("dataset"))
      .def("Distance", WithStatusOr(&DecisionForestCCModel::Distance),
           py::arg("dataset1"), py::arg("dataset2"))
      .def("GetTree", WithStatusOr(&DecisionForestCCModel::GetTree),
           py::arg("tree_idx"))
      .def("SetTree", WithStatus(&DecisionForestCCModel::SetTree),
           py::arg("tree_idx"), py::arg("nodes"))
      .def("AddTree", WithStatus(&DecisionForestCCModel::AddTree),
           py::arg("nodes"))
      .def("RemoveTree", WithStatus(&DecisionForestCCModel::RemoveTree),
           py::arg("tree_idx"));

  py::class_<RandomForestCCModel,
             /*parent class*/ DecisionForestCCModel>(m, "RandomForestCCModel")
      .def("__repr__",
           [](const GenericCCModel& a) {
             return absl::Substitute(
                 "<model_cc.RandomForestCCModel of type $0.", a.name());
           })
      .def("out_of_bag_evaluations",
           &RandomForestCCModel::out_of_bag_evaluations)
      .def("winner_takes_all", &RandomForestCCModel::winner_takes_all)
      .def_property_readonly_static(
          "kRegisteredName", [](py::object /* self */) {
            return model::random_forest::RandomForestModel::kRegisteredName;
          });

  py::class_<IsolationForestCCModel,
             /*parent class*/ DecisionForestCCModel>(m,
                                                     "IsolationForestCCModel")
      .def("__repr__",
           [](const GenericCCModel& a) {
             return absl::Substitute(
                 "<model_cc.IsolationForestCCModel of type $0.", a.name());
           })
      .def("num_examples_per_tree",
           &IsolationForestCCModel::num_examples_per_tree)
      .def("set_num_examples_per_tree",
           &IsolationForestCCModel::set_num_examples_per_tree)
      .def_property_readonly_static("kRegisteredName",
                                    [](py::object /* self */) {
                                      return model::isolation_forest::
                                          IsolationForestModel::kRegisteredName;
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
      .def("initial_predictions",
           &GradientBoostedTreesCCModel::initial_predictions)
      .def("set_initial_predictions",
           &GradientBoostedTreesCCModel::set_initial_predictions)
      .def("validation_evaluation",
           &GradientBoostedTreesCCModel::validation_evaluation)
      .def("loss", &GradientBoostedTreesCCModel::loss)
      .def("num_trees_per_iter",
           &GradientBoostedTreesCCModel::num_trees_per_iter)
      .def_property_readonly_static(
          "kRegisteredName", [](py::object /* self */) {
            return model::gradient_boosted_trees::GradientBoostedTreesModel::
                kRegisteredName;
          });
}

}  // namespace yggdrasil_decision_forests::port::python
