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

#include "yggdrasil_decision_forests/metric/uplift.h"

#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace metric {
namespace uplift {
namespace {

// Gets the "outcome" of a prediction.
utils::StatusOr<float> GetOutcome(const model::proto::Prediction& prediction) {
  switch (prediction.uplift().outcome_type_case()) {
    case model::proto::Prediction::Uplift::kOutcomeCategorical:
      if (prediction.uplift().outcome_categorical() < 1 ||
          prediction.uplift().outcome_categorical() > 2) {
        return absl::InvalidArgumentError("Invalid categorical outcome value");
      }
      return prediction.uplift().outcome_categorical() == 2;
    case model::proto::Prediction::Uplift::kOutcomeNumerical:
      return prediction.uplift().outcome_numerical();
    default:
      return absl::UnimplementedError("");
  }
}

}  // namespace

absl::Status InitializeCategoricalUpliftEvaluation(
    const proto::EvaluationOptions& option,
    const dataset::proto::Column& label_column,
    proto::EvaluationResults* eval) {
  if (label_column.type() != dataset::proto::ColumnType::CATEGORICAL) {
    return absl::InvalidArgumentError(
        "Categorical uplift requires a categorical label (i.e. response or "
        "outcome). ");
  }
  if (label_column.categorical().number_of_unique_values() != 3) {
    return absl::InvalidArgumentError(
        "Uplift categorical response should be binary (i.e. have two values).");
  }
  eval->mutable_uplift();
  return absl::OkStatus();
}

absl::Status InitializeNumericalUpliftEvaluation(
    const proto::EvaluationOptions& option,
    const dataset::proto::Column& label_column,
    proto::EvaluationResults* eval) {
  if (label_column.type() != dataset::proto::ColumnType::NUMERICAL) {
    return absl::InvalidArgumentError(
        "Numerical uplift requires a numerical label (i.e. response or "
        "outcome). ");
  }
  eval->mutable_uplift();
  return absl::OkStatus();
}

absl::Status AddUpliftPredictionImp(const proto::EvaluationOptions& option,
                                    const model::proto::Prediction& pred,
                                    utils::RandomEngine* rnd,
                                    proto::EvaluationResults* eval) {
  if (!pred.has_uplift()) {
    return absl::InvalidArgumentError("Missing uplift field in prediction");
  }
  auto* eval_uplift = eval->mutable_uplift();
  eval_uplift->set_num_treatments(
      std::max<int>(eval_uplift->num_treatments(), pred.uplift().treatment()));
  return absl::OkStatus();
}

absl::Status FinalizeUpliftMetricsFromSampledPredictions(
    const proto::EvaluationOptions& option,
    const dataset::proto::Column& label_column,
    proto::EvaluationResults* eval) {
  const int num_treatments = eval->uplift().num_treatments();
  if (num_treatments < 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("There should be at least two treatments (including"
                     "control). Found ",
                     num_treatments, " treatments"));
  }

  if (num_treatments != 2) {
    return absl::InvalidArgumentError(
        "Only binary treatment is currently supported.");
    // TODO(gbm): Implement the Expected Response metric for the
    // multi-treatment.
  }

  // Split and accumulate the examples per treatment or control.
  typedef std::vector<internal::Example> ExampleList;
  ExampleList examples;
  examples.reserve(eval->sampled_predictions_size());

  for (const auto& sample : eval->sampled_predictions()) {
    // The treatment_value 0 is reserved for the OOV item and is not a valid
    // treatment. The treatment_value 1 indicates this is a control. The
    // treatment_value in [2, 1+num_treatments) indicates an effective
    // treatment.

    ASSIGN_OR_RETURN(const float outcome, GetOutcome(sample));

    if (sample.uplift().treatment_effect_size() != num_treatments - 1) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Wrong prediction shape. num_treatments:$0 prediction:$1 "
          "expected_predictions:$2",
          num_treatments, sample.uplift().treatment_effect_size(),
          num_treatments - 1));
    }

    if (sample.uplift().treatment() == 0) {
      return absl::InvalidArgumentError(
          "treatment value of 0 (i.e. OOV) is not allowed.");
    } else {
      if (sample.uplift().treatment() > 2) {
        return absl::InvalidArgumentError("Only binary treatment supported");
      }
      examples.push_back(
          internal::Example{/*effect=*/sample.uplift().treatment_effect(0),
                            /*outcome=*/outcome,
                            /*weight=*/sample.weight(),
                            /*treatment=*/sample.uplift().treatment() - 1});
    }
  }

  // Sort the effect/outcome/treatment values by increasing effect.
  std::sort(examples.begin(), examples.end(), std::greater<>());

  const auto auuc_result = internal::ComputeAuuc(examples, /*treatment=*/1);
  const auto qini = auuc_result.auc - auuc_result.max_uplift_curve / 2;

  auto& uplift = *eval->mutable_uplift();
  uplift.set_auuc(auuc_result.auc);
  uplift.set_qini(qini);
  return absl::OkStatus();
}

namespace internal {

AreaUnderCurve ComputeAuuc(const std::vector<Example>& sorted_items,
                           int positive_treatment) {
  if (sorted_items.empty()) {
    return AreaUnderCurve{0, 0};
  }

  // Count the number of examples in control and treatment.
  double sum_treatment_outcomes = 0;
  double sum_control_outcomes = 0;

  double sum_treatments = 0;
  double sum_controls = 0;

  double sum_weights = 0;
  for (const auto& example : sorted_items) {
    if (example.treatment == positive_treatment) {
      sum_treatment_outcomes += example.weight * example.outcome;
      sum_treatments += example.weight;
    } else {
      sum_control_outcomes += example.weight * example.outcome;
      sum_controls += example.weight;
    }
    sum_weights += example.weight;
  }

  double auuc = 0;
  double acc_sum_treatment_outcomes = 0;
  double acc_sum_control_outcomes = 0;
  double acc_sum_weights = 0;

  double last_net_lift = 0;
  double last_acc_sum_weights = 0;

  for (size_t example_idx = 0; example_idx < sorted_items.size();
       example_idx++) {
    const auto& example = sorted_items[example_idx];

    if (example.treatment == positive_treatment) {
      acc_sum_treatment_outcomes += example.weight * example.outcome;
    } else {
      acc_sum_control_outcomes += example.weight * example.outcome;
    }
    acc_sum_weights += example.weight;

    // Making sure tied predictions are grouped together.
    if (example_idx + 1 == sorted_items.size() ||
        example.predicted_uplift !=
            sorted_items[example_idx + 1].predicted_uplift) {
      // Following the notation in section 7.2 of "Optimal personalized
      // treatment learning models with insurance applications" by "Leo
      // Guelman".
      const double r_treatment = acc_sum_treatment_outcomes / sum_treatments;
      const double r_control = acc_sum_control_outcomes / sum_controls;
      const double net_lift = r_treatment - r_control;
      auuc += (acc_sum_weights - last_acc_sum_weights) / sum_weights *
              (net_lift + last_net_lift) / 2;

      last_net_lift = net_lift;
      last_acc_sum_weights = acc_sum_weights;
    }
  }

  return {auuc, last_net_lift};
}

}  // namespace internal
}  // namespace uplift
}  // namespace metric
}  // namespace yggdrasil_decision_forests
