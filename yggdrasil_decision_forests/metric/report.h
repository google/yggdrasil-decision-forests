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

// Creation of report containing metrics and metric comparisons.

#ifndef YGGDRASIL_DECISION_FORESTS_METRIC_REPORT_H_
#define YGGDRASIL_DECISION_FORESTS_METRIC_REPORT_H_

#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace metric {

// Append a textual report of the evaluation.
absl::StatusOr<std::string> TextReport(const proto::EvaluationResults& eval);

void AppendTextReport(const proto::EvaluationResults& eval,
                      std::string* report);

absl::Status AppendTextReportWithStatus(const proto::EvaluationResults& eval,
                                        std::string* report);

// Specialization for specific task.
absl::Status AppendTextReportClassification(
    const proto::EvaluationResults& eval, std::string* report);
absl::Status AppendTextReportRegression(const proto::EvaluationResults& eval,
                                        std::string* report);
absl::Status AppendTextReportRanking(const proto::EvaluationResults& eval,
                                     std::string* report);
absl::Status AppendTextReportUplift(const proto::EvaluationResults& eval,
                                    std::string* report);

// Add the report in a html format.
struct HtmlReportOptions {
  bool include_title = true;
  bool include_text_report = true;

  // Size of the plots.
  int plot_width = 600;
  int plot_height = 400;

  // Maximum number of plots side by side.
  int num_plots_per_columns = 3;
};
absl::Status AppendHtmlReport(const proto::EvaluationResults& eval,
                              std::string* html_report,
                              const HtmlReportOptions& options = {});

}  // namespace metric
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_METRIC_REPORT_H_
