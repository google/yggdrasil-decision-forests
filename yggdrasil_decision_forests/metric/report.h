/*
 * Copyright 2021 Google LLC.
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

#include "yggdrasil_decision_forests/metric/metric.pb.h"

namespace yggdrasil_decision_forests {
namespace metric {

// Append a textual report of the evaluation.
std::string TextReport(const proto::EvaluationResults& eval);
void AppendTextReport(const proto::EvaluationResults& eval,
                      std::string* report);

// Specialization for specific task.
void AppendTextReportClassification(const proto::EvaluationResults& eval,
                                    std::string* report);
void AppendTextReportRegression(const proto::EvaluationResults& eval,
                                std::string* report);
void AppendTextReportRanking(const proto::EvaluationResults& eval,
                             std::string* report);
void AppendTextReportUplift(const proto::EvaluationResults& eval,
                            std::string* report);

}  // namespace metric
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_METRIC_REPORT_H_
