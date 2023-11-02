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

#include <string>

#include "absl/status/statusor.h"
#include "pybind11_protobuf/native_proto_caster.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "ydf/utils/status_casters.h"

namespace py = ::pybind11;

namespace yggdrasil_decision_forests::port::python {
namespace {

absl::StatusOr<std::string> EvaluationToStr(
    const metric::proto::EvaluationResults& evaluation) {
  return metric::TextReport(evaluation);
}

absl::StatusOr<std::string> EvaluationPlotToHtml(
    const metric::proto::EvaluationResults& evaluation) {
  std::string html;
  metric::HtmlReportOptions options;
  options.plot_width = 500;
  options.plot_height = 400;
  options.include_text_report = false;
  options.include_title = false;
  options.num_plots_per_columns = 2;
  RETURN_IF_ERROR(metric::AppendHtmlReport(evaluation, &html, options));
  return html;
}

}  // namespace

void init_metric(py::module_& m) {
  m.def("EvaluationToStr", WithStatusOr(EvaluationToStr),
        py::arg("evaluation"));
  m.def("EvaluationPlotToHtml", WithStatusOr(EvaluationPlotToHtml),
        py::arg("evaluation"));
}

}  // namespace yggdrasil_decision_forests::port::python
