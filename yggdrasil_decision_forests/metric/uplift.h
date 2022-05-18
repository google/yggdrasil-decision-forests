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

#ifndef YGGDRASIL_DECISION_FORESTS_METRIC_UPLIFT_H_
#define YGGDRASIL_DECISION_FORESTS_METRIC_UPLIFT_H_

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace metric {
namespace uplift {
// Implementation of "InitializeEvaluation", "AddUpPrediction" and
// "FinalizeMetricsFromSampledPredictions" in the case of Uplift evaluation.

absl::Status InitializeCategoricalUpliftEvaluation(
    const proto::EvaluationOptions& option,
    const dataset::proto::Column& label_column, proto::EvaluationResults* eval);

absl::Status InitializeNumericalUpliftEvaluation(
    const proto::EvaluationOptions& option,
    const dataset::proto::Column& label_column, proto::EvaluationResults* eval);

absl::Status AddUpliftPredictionImp(const proto::EvaluationOptions& option,
                                    const model::proto::Prediction& pred,
                                    utils::RandomEngine* rnd,
                                    proto::EvaluationResults* eval);

absl::Status FinalizeUpliftMetricsFromSampledPredictions(
    const proto::EvaluationOptions& option,
    const dataset::proto::Column& label_column, proto::EvaluationResults* eval);

namespace internal {

struct Example {
  float predicted_uplift;
  float outcome;
  float weight;
  int treatment;

  bool operator<(const Example& a) const {
    return predicted_uplift < a.predicted_uplift;
  }
  bool operator>(const Example& a) const {
    return predicted_uplift > a.predicted_uplift;
  }
};

// Area under a curve. Used on the cumulative response curves.
struct AreaUnderCurve {
  double auc;

  // Maximum of the curve e.g. right side of the curve in the case of a
  // cumulative curve.
  double max_uplift_curve;
};

AreaUnderCurve ComputeAuuc(const std::vector<Example>& sorted_items,
                           int positive_treatment);

}  // namespace internal

}  // namespace uplift
}  // namespace metric
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_METRIC_UPLIFT_H_
