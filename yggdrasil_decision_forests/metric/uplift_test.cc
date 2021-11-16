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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace metric {
namespace {

proto::EvaluationResults GenEvaluationBinaryOutcome(
    const std::vector<int>& treatments, const std::vector<int>& outcomes,
    const std::vector<std::vector<float>>& predicted_uplift,
    const std::vector<float>& weights, const int num_treatments) {
  dataset::proto::Column effect_column = PARSE_TEST_PROTO(
      R"pb(
        name: "effect"
        type: CATEGORICAL
        categorical { number_of_unique_values: 3 }
      )pb");

  dataset::proto::Column treatment_column = PARSE_TEST_PROTO(absl::Substitute(
      R"pb(
        name: "treatment"
        type: CATEGORICAL
        categorical { number_of_unique_values: $0 }
      )pb",
      num_treatments + 1));

  CHECK_EQ(treatments.size(), outcomes.size());
  CHECK_EQ(treatments.size(), predicted_uplift.size());
  CHECK(weights.size() == treatments.size() || weights.empty());

  // Configure the evaluation.
  utils::RandomEngine rnd;
  proto::EvaluationOptions option;
  option.set_task(model::proto::Task::CATEGORICAL_UPLIFT);

  proto::EvaluationResults eval;
  InitializeEvaluation(option, effect_column, &eval);

  double sum_weights = 0;
  for (int example_idx = 0; example_idx < treatments.size(); example_idx++) {
    model::proto::Prediction pred;
    auto& pred_uplift = *pred.mutable_uplift();

    float weight = 1;

    if (!weights.empty()) {
      weight = weights[example_idx];
      pred.set_weight(weight);
    }

    *pred_uplift.mutable_treatment_effect() = {
        predicted_uplift[example_idx].begin(),
        predicted_uplift[example_idx].end()};
    pred_uplift.set_outcome_categorical(outcomes[example_idx]);
    pred_uplift.set_treatment(treatments[example_idx]);
    AddPrediction(option, pred, &rnd, &eval);

    sum_weights += weight;
  }

  FinalizeEvaluation(option, effect_column, &eval);

  EXPECT_NEAR(eval.count_predictions(), sum_weights, 0.00001);
  EXPECT_EQ(eval.count_predictions_no_weight(), treatments.size());
  EXPECT_EQ(eval.count_sampled_predictions(), treatments.size());
  EXPECT_EQ(eval.task(), model::proto::Task::CATEGORICAL_UPLIFT);

  // Create reports.
  std::string report;
  AppendTextReport(eval, &report);
  LOG(INFO) << "Report :\n" << report;

  return eval;
}

TEST(Uplift, BinaryTreatmentBinaryEffect) {
  auto evaluation = GenEvaluationBinaryOutcome(
      /*treatments=*/{1, 1, 2, 2}, /*outcomes=*/{2, 1, 1, 2},
      /*predicted_uplift=*/{{0.8}, {0.5}, {0.9}, {0.4}},
      /*weights=*/{}, /*num_treatments=*/2);

  // Checked with R's uplifteval lib.
  //  e = plUpliftEval(
  //      treatment = c(1, 1, 2, 2) - 1,
  //      outcome = c(2, 1, 1, 2) - 1,
  //      prediction = c(0.8, 0.5, 0.9, 0.4)
  //  )
  EXPECT_NEAR(AUUC(evaluation), -0.25, 0.0001);
  EXPECT_NEAR(Qini(evaluation), -0.25, 0.0001);

  evaluation = GenEvaluationBinaryOutcome(
      /*treatments=*/{1, 1, 1, 2, 2, 2}, /*outcomes=*/{1, 2, 1, 1, 2, 2},
      /*predicted_uplift=*/{{0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}},
      /*weights=*/{}, /*num_treatments=*/2);

  // Checked with R's uplifteval lib.
  //  e = plUpliftEval(
  //      treatment = c(1, 1, 1, 2, 2, 2) - 1,
  //      outcome = c(1, 2, 1, 1, 2, 2) - 1,
  //      prediction = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
  //  )
  EXPECT_NEAR(AUUC(evaluation), 0.4722222, 0.0001);
  EXPECT_NEAR(Qini(evaluation), 0.3055556, 0.0001);
}

TEST(Uplift, DuplicateEffectValues) {
  const auto evaluation = GenEvaluationBinaryOutcome(
      /*treatments=*/{1, 1, 2, 2}, /*outcomes=*/{2, 1, 1, 2},
      /*predicted_uplift=*/{{0.8}, {0.5}, {0.5}, {0.5}},
      /*weights=*/{}, /*num_treatments=*/2);

  // Note "uplifteval" handles predictions ties differently.
  EXPECT_NEAR(AUUC(evaluation), -0.25, 0.0001);
  EXPECT_NEAR(Qini(evaluation), -0.25, 0.0001);
}

}  // namespace
}  // namespace metric
}  // namespace yggdrasil_decision_forests
