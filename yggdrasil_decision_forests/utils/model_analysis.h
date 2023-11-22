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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_MODEL_ANALYSIS_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_MODEL_ANALYSIS_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/utils/model_analysis.pb.h"
#include "yggdrasil_decision_forests/utils/partial_dependence_plot.h"
#include "yggdrasil_decision_forests/utils/plot.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace model_analysis {

// Analyses a model with regard to a dataset.
absl::StatusOr<proto::AnalysisResult> Analyse(
    const model::AbstractModel& model, const dataset::VerticalDataset& dataset,
    const proto::Options& options = {});

// Analyses the prediction of a model.
absl::StatusOr<proto::PredictionAnalysisResult> AnalyzePrediction(
    const model::AbstractModel& model, const dataset::proto::Example& example,
    const proto::PredictionAnalysisOptions& options = {});

// Creates a HTML report of a prediction analysis.
absl::StatusOr<std::string> CreateHtmlReport(
    const proto::PredictionAnalysisResult& analysis,
    const proto::PredictionAnalysisOptions& options = {});

// Creates a HTML report of an analysis. The report also contains the  human
// model description and dataspec specification. This functions does not run any
// computation not read files.
absl::Status CreateHtmlReport(const model::AbstractModel& model,
                              const dataset::VerticalDataset& dataset,
                              absl::string_view model_path,
                              absl::string_view dataset_path,
                              const proto::AnalysisResult& analysis,
                              absl::string_view output_directory,
                              const proto::Options& options = {});

// Same as "CreateHtmlReport" above. Return the HTML as a string.
absl::StatusOr<std::string> CreateHtmlReport(
    const model::AbstractModel& model, const dataset::VerticalDataset& dataset,
    absl::string_view model_path, absl::string_view dataset_path,
    const proto::AnalysisResult& analysis, const proto::Options& options = {});

// Same as "CreateHtmlReport" from a standalone analysis.
absl::StatusOr<std::string> CreateHtmlReport(
    const proto::StandaloneAnalysisResult& analysis,
    const proto::Options& options = {});

// Assemble an analysis into a standalone analysis.
proto::StandaloneAnalysisResult CreateStandaloneAnalysis(
    const model::AbstractModel& model, const dataset::VerticalDataset& dataset,
    absl::string_view model_path, absl::string_view dataset_path,
    const proto::AnalysisResult& analysis);

// Combines the model analysis and html creation i.e. Analyse +
// CreateHtmlReport.
absl::Status AnalyseAndCreateHtmlReport(const model::AbstractModel& model,
                                        const dataset::VerticalDataset& dataset,
                                        absl::string_view model_path,
                                        absl::string_view dataset_path,
                                        absl::string_view output_directory,
                                        const proto::Options& options = {});

// Similar to "AnalyseAndCreateHtmlReport" above, but takes a dataset path
// instead of a vertical dataset.
absl::Status AnalyseAndCreateHtmlReport(const model::AbstractModel& model,
                                        absl::string_view model_path,
                                        absl::string_view dataset_path,
                                        absl::string_view output_directory,
                                        const proto::Options& options = {});

namespace internal {

absl::StatusOr<utils::plot::MultiPlot> PlotConditionalExpectationPlotSet(
    const dataset::proto::DataSpecification& data_spec,
    const ConditionalExpectationPlotSet& cond_set,
    const model::proto::Task task, const int label_col_idx,
    const proto::Options& options, int* recommended_width_px,
    int* recommended_height_px);

absl::StatusOr<utils::plot::MultiPlot> PlotPartialDependencePlotSet(
    const dataset::proto::DataSpecification& data_spec,
    const PartialDependencePlotSet& pdp_set, const model::proto::Task task,
    const int label_col_idx, const proto::Options& options,
    int* recommended_width_px, int* recommended_height_px);

}  // namespace internal
}  // namespace model_analysis
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_MODEL_ANALYSIS_H_
