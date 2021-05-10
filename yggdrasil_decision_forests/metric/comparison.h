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

// Metric comparison of machine learning models and learners.

#ifndef YGGDRASIL_DECISION_FORESTS_METRIC_COMPARISON_H_
#define YGGDRASIL_DECISION_FORESTS_METRIC_COMPARISON_H_

#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"

namespace yggdrasil_decision_forests {
namespace metric {

// Create a dataspec for a table comparing models. The output dataspec matches
// the output of "ExtractFlatMetrics".
dataset::proto::DataSpecification CreateDataSpecForComparisonTable(
    const proto::EvaluationOptions& option,
    const proto::EvaluationResults& example_of_evaluation);

// Computes the p-value of the one-sided McNemar between the evaluation of two
// classifier models. More precisely, evaluates the p-value of the
// null-hypothesis: The accuracy of the "model_1" at threshold "threshold_1" is
// not greater than the accuracy of the "model_2" at threshold "threshold_2".
// A small p-value (e.g. <0.05) indicates that it is safe to reject the
// null-hypothesis i.e. deciding that the model 1 is better than the model 2.
float OneSidedMcNemarTest(const proto::EvaluationResults& eval_results1,
                          const proto::EvaluationResults& eval_results2,
                          int roc_idx, float threshold1, float threshold2);

// Computes the p-value of the one-sided McNemar between the evaluation of two
// classifier models at a variety of thresholds. The thresholds used for each
// model are extracted from the XAtYMetrics fields in the Rocs. The output
// returned is a vector of pairs of labels (which pertain to a certain threshold
// and the XAtYMetric they were taken from) and their corresponding p-values.
// An example of a label : '1_vs_the_others@Recall=0.5', which corresponds to
// comparing the two models at the thresholds where the class '1' has a recall
// of 0.5 .
std::vector<std::pair<std::string, float>> OneSidedMcNemarTest(
    const proto::EvaluationResults& eval_results1,
    const proto::EvaluationResults& eval_results2);

// Computes the p-value of the pairwise comparison on the mean of the
// difference of residual of two regressive models.
//
// H1: expected_value(residual(baseline) - residual(candidate)) > 0 i.e.
//
// See https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm for more
// details.
float PairwiseRegressiveResidualTest(
    const proto::EvaluationResults& eval_baseline,
    const proto::EvaluationResults& eval_candidate);

// Computes the p-value of the pairewise comparison on the mean of the
// difference of ndcg of two ranking models.
//
// H1: expected_value(residual(candidate) - residual(baseline)) > 0
float PairwiseRankingNDCG5Test(const proto::EvaluationResults& eval_baseline,
                               const proto::EvaluationResults& eval_candidate);

}  // namespace metric
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_METRIC_COMPARISON_H_
