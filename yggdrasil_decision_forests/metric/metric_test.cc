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

#include "yggdrasil_decision_forests/metric/metric.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/metric/comparison.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace metric {
namespace {

using test::EqualsProto;
using testing::ElementsAre;

TEST(Metric, EvaluationOfClassification) {
  // Create a fake column specification.
  dataset::proto::Column label_column;
  label_column.set_type(dataset::proto::ColumnType::CATEGORICAL);
  label_column.set_name("label");
  label_column.mutable_categorical()->set_number_of_unique_values(3);
  label_column.mutable_categorical()->set_most_frequent_value(1);
  label_column.mutable_categorical()->set_is_already_integerized(false);
  auto& vocab = *label_column.mutable_categorical()->mutable_items();
  vocab["a"].set_index(0);
  vocab["b"].set_index(1);
  vocab["c"].set_index(2);

  // Configure the evaluation.
  utils::RandomEngine rnd;
  proto::EvaluationOptions option;
  option.set_task(model::proto::Task::CLASSIFICATION);

  // Initialize.
  proto::EvaluationResults eval;
  InitializeEvaluation(option, label_column, &eval);
  model::proto::Prediction pred;
  auto* pred_proba = pred.mutable_classification()->mutable_distribution();
  pred_proba->mutable_counts()->Resize(3, 0);
  pred_proba->set_sum(1);

  // Add some predictions.
  pred.mutable_classification()->set_value(1);
  pred_proba->set_counts(0, 0.2f);
  pred_proba->set_counts(1, 0.8f);
  pred_proba->set_counts(2, 0.2f);
  pred.mutable_classification()->set_ground_truth(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_classification()->set_value(1);
  pred_proba->set_counts(0, 0.2f);
  pred_proba->set_counts(1, 0.6f);
  pred_proba->set_counts(2, 0.2f);
  pred.mutable_classification()->set_ground_truth(2);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_classification()->set_value(2);
  pred_proba->set_counts(0, 0.0f);
  pred_proba->set_counts(1, 0.1f);
  pred_proba->set_counts(2, 0.9f);
  pred.mutable_classification()->set_ground_truth(2);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_classification()->set_value(0);
  pred_proba->set_counts(0, 0.5f);
  pred_proba->set_counts(1, 0.2f);
  pred_proba->set_counts(2, 0.3f);
  pred.mutable_classification()->set_ground_truth(1);
  AddPrediction(option, pred, &rnd, &eval);

  // Finalize.
  FinalizeEvaluation(option, label_column, &eval);

  EXPECT_EQ(eval.count_predictions(), 4);
  EXPECT_EQ(eval.count_predictions_no_weight(), 4);
  EXPECT_EQ(eval.count_sampled_predictions(), 4);
  EXPECT_EQ(eval.task(), model::proto::Task::CLASSIFICATION);
  EXPECT_NEAR(Accuracy(eval), 0.5f, 0.0001f);
  EXPECT_NEAR(ErrorRate(eval), 0.5f, 0.0001f);
  EXPECT_NEAR(LogLoss(eval), (-log(0.8) - log(0.2) - log(0.9) - log(0.2)) / 4,
              0.0001f);

  EXPECT_NEAR(DefaultAccuracy(eval), 0.5f, 0.0001f);
  EXPECT_NEAR(DefaultErrorRate(eval), 0.5f, 0.0001f);
  EXPECT_NEAR(DefaultLogLoss(eval), -2 * 0.5 * log(0.5), 0.0001f);

  EXPECT_EQ(eval.classification().rocs_size(), 3);

  // Create reports.
  std::string report;
  AppendTextReport(eval, &report);
  LOG(INFO) << "Report :\n " << report;
}

TEST(Metric, EvaluationOfClassificationWithNumericalWeights) {
  // Create a fake column specification.
  dataset::proto::Column label_column;
  label_column.set_type(dataset::proto::ColumnType::CATEGORICAL);
  label_column.set_name("label");
  label_column.mutable_categorical()->set_number_of_unique_values(3);
  label_column.mutable_categorical()->set_most_frequent_value(1);
  label_column.mutable_categorical()->set_is_already_integerized(false);
  auto& vocab = *label_column.mutable_categorical()->mutable_items();
  vocab["a"].set_index(0);
  vocab["b"].set_index(1);
  vocab["c"].set_index(2);

  // Configure the evaluation.
  utils::RandomEngine rnd;
  proto::EvaluationOptions option;
  option.set_task(model::proto::Task::CLASSIFICATION);
  option.mutable_weights()->set_attribute("toto");

  // Initialize.
  proto::EvaluationResults eval;
  InitializeEvaluation(option, label_column, &eval);
  model::proto::Prediction pred;
  auto* pred_proba = pred.mutable_classification()->mutable_distribution();
  pred_proba->mutable_counts()->Resize(3, 0);
  pred_proba->set_sum(1);

  // Add some predictions.
  pred.mutable_classification()->set_value(1);
  pred_proba->set_counts(0, 0.2f);
  pred_proba->set_counts(1, 0.8f);
  pred_proba->set_counts(2, 0.2f);
  pred.mutable_classification()->set_ground_truth(1);
  pred.set_weight(2.5f);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_classification()->set_value(1);
  pred_proba->set_counts(0, 0.2f);
  pred_proba->set_counts(1, 0.6f);
  pred_proba->set_counts(2, 0.2f);
  pred.mutable_classification()->set_ground_truth(2);
  pred.set_weight(1.f);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_classification()->set_value(2);
  pred_proba->set_counts(0, 0.0f);
  pred_proba->set_counts(1, 0.1f);
  pred_proba->set_counts(2, 0.9f);
  pred.mutable_classification()->set_ground_truth(2);
  pred.set_weight(2.f);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_classification()->set_value(0);
  pred_proba->set_counts(0, 0.5f);
  pred_proba->set_counts(1, 0.2f);
  pred_proba->set_counts(2, 0.3f);
  pred.mutable_classification()->set_ground_truth(1);
  pred.set_weight(4.f);
  AddPrediction(option, pred, &rnd, &eval);

  // Finalize.
  FinalizeEvaluation(option, label_column, &eval);

  const float sum_weights = 2.5f + 1.f + 2.f + 4.f;
  EXPECT_EQ(eval.count_predictions(), sum_weights);
  EXPECT_EQ(eval.count_predictions_no_weight(), 4);
  EXPECT_EQ(eval.count_sampled_predictions(), sum_weights);
  EXPECT_EQ(eval.task(), model::proto::Task::CLASSIFICATION);
  EXPECT_NEAR(Accuracy(eval), (2.5f + 2.f) / sum_weights, 0.0001f);
  EXPECT_NEAR(ErrorRate(eval), 1.f - (2.5f + 2.f) / sum_weights, 0.0001f);
  EXPECT_NEAR(
      LogLoss(eval),
      (-log(0.8) * 2.5 - log(0.2) * 1. - log(0.9) * 2. - log(0.2) * 4.) /
          sum_weights,
      0.0001f);

  EXPECT_NEAR(DefaultAccuracy(eval), (2.5f + 4.f) / sum_weights, 0.0001f);
  EXPECT_NEAR(DefaultErrorRate(eval), 1.f - (2.5f + 4.f) / sum_weights,
              0.0001f);
  const float relative_weight_label_1 = (2.5f + 4.f) / sum_weights;
  const float relative_weight_label_2 = (1.f + 2.f) / sum_weights;
  EXPECT_NEAR(DefaultLogLoss(eval),
              -relative_weight_label_1 * std::log(relative_weight_label_1) -
                  relative_weight_label_2 * std::log(relative_weight_label_2),
              0.0001f);
}

TEST(Metric, EvaluationOfRegression) {
  // Create a fake column specification.
  dataset::proto::Column label_column;
  label_column.set_type(dataset::proto::ColumnType::NUMERICAL);
  label_column.set_name("label");

  // Configure the evaluation.
  utils::RandomEngine rnd;
  proto::EvaluationOptions option;
  option.set_task(model::proto::Task::REGRESSION);

  // Initialize.
  proto::EvaluationResults eval;
  InitializeEvaluation(option, label_column, &eval);
  model::proto::Prediction pred;

  // Add some predictions.
  pred.mutable_regression()->set_value(1);
  pred.mutable_regression()->set_ground_truth(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_regression()->set_value(1);
  pred.mutable_regression()->set_ground_truth(2);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_regression()->set_value(2);
  pred.mutable_regression()->set_ground_truth(2);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_regression()->set_value(0);
  pred.mutable_regression()->set_ground_truth(1);
  AddPrediction(option, pred, &rnd, &eval);

  // Finalize.
  FinalizeEvaluation(option, label_column, &eval);

  EXPECT_EQ(eval.count_predictions(), 4);
  EXPECT_EQ(eval.count_predictions_no_weight(), 4);
  EXPECT_EQ(eval.count_sampled_predictions(), 4);
  EXPECT_EQ(eval.task(), model::proto::Task::REGRESSION);
  EXPECT_NEAR(RMSE(eval), sqrt(0.5), 0.0001);
  EXPECT_NEAR(DefaultRMSE(eval), 0.5, 0.0001);

  // Create reports.
  std::string report;
  AppendTextReport(eval, &report);
  LOG(INFO) << "Report :\n " << report;
}

TEST(Metric, ComputeRoc) {
  // Create a fake column specification.
  dataset::proto::Column label_column;
  label_column.set_type(dataset::proto::ColumnType::CATEGORICAL);
  label_column.set_name("label");
  label_column.mutable_categorical()->set_number_of_unique_values(2);
  label_column.mutable_categorical()->set_most_frequent_value(1);
  label_column.mutable_categorical()->set_is_already_integerized(false);
  auto& vocab = *label_column.mutable_categorical()->mutable_items();
  vocab["a"].set_index(0);
  vocab["b"].set_index(1);

  const int num_predictions = 10000;

  for (const float label_correlation : std::vector<float>({0.f, 0.5f, 1.f})) {
    for (const int max_roc_samples : {100, 100000}) {
      // Configure the evaluation.
      const proto::EvaluationOptions option =
          PARSE_TEST_PROTO(absl::Substitute(R"(
task: CLASSIFICATION
classification {
  max_roc_samples: $0
  precision_at_recall: 0.90
  recall_at_precision: 0.90
  precision_at_volume: 0.10
  recall_at_false_positive_rate: 0.10
  false_positive_rate_at_recall: 0.90
})",
                                            max_roc_samples));

      proto::EvaluationResults eval;
      InitializeEvaluation(option, label_column, &eval);

      // Generate some random predictions.
      utils::RandomEngine rnd;
      std::normal_distribution<float> unit_normal_dist;
      std::uniform_int_distribution<int> label_dist(0, 1);
      model::proto::Prediction pred;
      auto* pred_proba = pred.mutable_classification()->mutable_distribution();
      pred_proba->mutable_counts()->Resize(2, 0);
      for (int pred_idx = 0; pred_idx < num_predictions; pred_idx++) {
        const int ground_truth = label_dist(rnd);
        float prediction = unit_normal_dist(rnd) * (1 - label_correlation) +
                           ground_truth * label_correlation;
        prediction = utils::clamp(prediction / 100.f + 0.5f, 0.f, 1.f);
        pred.mutable_classification()->set_value(
            static_cast<int>(prediction > 0.5f));
        pred_proba->set_counts(0, 1 - prediction);
        pred_proba->set_counts(1, prediction);
        pred_proba->set_sum(prediction + (1 - prediction));
        pred.mutable_classification()->set_ground_truth(ground_truth);
        AddPrediction(option, pred, &rnd, &eval);
      }

      // Finalize.
      FinalizeEvaluation(option, label_column, &eval);

      for (auto& roc : *eval.mutable_classification()->mutable_rocs()) {
        if (label_correlation == 0) {
          EXPECT_NEAR(roc.auc(), 0.5f, 0.01f);
          EXPECT_NEAR(roc.pr_auc(), 0.5f, 0.02f);
        } else if (label_correlation == 1) {
          EXPECT_NEAR(roc.auc(), 1.0f, 0.01f);
          EXPECT_NEAR(roc.pr_auc(), 1.0f, 0.02f);
        } else {
          EXPECT_NEAR(roc.auc(), 0.75f, 0.02f);
          EXPECT_NEAR(roc.pr_auc(), 0.75f, 0.05f);

          EXPECT_EQ(roc.precision_at_recall().size(), 1);
          EXPECT_EQ(roc.recall_at_precision().size(), 1);
          EXPECT_EQ(roc.precision_at_volume().size(), 1);
          EXPECT_EQ(roc.recall_at_false_positive_rate().size(), 1);
          EXPECT_EQ(roc.false_positive_rate_at_recall().size(), 1);

          // These values should match the configuration.
          EXPECT_NEAR(roc.precision_at_recall(0).y_metric_constraint(), 0.90,
                      0.01);
          EXPECT_NEAR(roc.recall_at_precision(0).y_metric_constraint(), 0.90,
                      0.01);
          EXPECT_NEAR(roc.precision_at_volume(0).y_metric_constraint(), 0.10,
                      0.01);
          EXPECT_NEAR(
              roc.recall_at_false_positive_rate(0).y_metric_constraint(), 0.10,
              0.01);
          EXPECT_NEAR(
              roc.false_positive_rate_at_recall(0).y_metric_constraint(), 0.90,
              0.01);

          // The following number have been validated visually using SimplePlot.
          EXPECT_NEAR(roc.precision_at_recall(0).x_metric_value(), 0.594, 0.01);
          // The error margin (0.03) is high because the precision has a high
          // variance for high recall.
          EXPECT_NEAR(roc.recall_at_precision(0).x_metric_value(), 0.136, 0.04);
          EXPECT_NEAR(roc.precision_at_volume(0).x_metric_value(), 0.875, 0.01);
          EXPECT_NEAR(roc.recall_at_false_positive_rate(0).x_metric_value(),
                      0.393, 0.02);
          EXPECT_NEAR(roc.false_positive_rate_at_recall(0).x_metric_value(),
                      0.619, 0.02);
        }
        EXPECT_LE(roc.curve_size(), max_roc_samples);

        // Matching between closed-form and bootstrapping estimated confidence
        // intervals.

        // AUC
        const auto auc_ci_close_form = AUCConfidenceInterval(roc);
        EXPECT_NEAR(auc_ci_close_form.first,
                    roc.bootstrap_lower_bounds_95p().auc(), 0.01);
        EXPECT_NEAR(auc_ci_close_form.second,
                    roc.bootstrap_upper_bounds_95p().auc(), 0.01);

        // PR AUC
        const auto pr_auc_ci_close_form = PRAUCConfidenceInterval(roc);
        EXPECT_NEAR(pr_auc_ci_close_form.first,
                    roc.bootstrap_lower_bounds_95p().pr_auc(), 0.01);
        EXPECT_NEAR(pr_auc_ci_close_form.second,
                    roc.bootstrap_upper_bounds_95p().pr_auc(), 0.01);
      }
    }
  }
}

TEST(Metric, CreateDataSpecForComparisonTable) {
  proto::EvaluationOptions option;
  option.set_task(model::proto::Task::CLASSIFICATION);
  proto::EvaluationResults example_of_evaluation;
  example_of_evaluation.set_task(model::proto::Task::CLASSIFICATION);
  const auto data_spec =
      CreateDataSpecForComparisonTable(option, example_of_evaluation);
  dataset::proto::DataSpecification expected = PARSE_TEST_PROTO(
      R"=(
        columns { type: STRING name: "Model" }
        columns { type: NUMERICAL name: "Training duration (s)" }
        columns { type: NUMERICAL name: "Accuracy" }
        columns { type: NUMERICAL_SET name: "Accuracy CI95% [W]" }
      )=");
  EXPECT_THAT(data_spec, EqualsProto(expected));
}

TEST(Metric, ExtractFlatMetrics) {
  proto::EvaluationResults evaluation = PARSE_TEST_PROTO(
      R"pb(
        task: CLASSIFICATION
        classification {
          rocs {
            auc: 1
            pr_auc: 2
            ap: 3
            precision_at_recall { x_metric_value: 4 y_metric_constraint: 5 }
            precision_at_recall { x_metric_value: 6 y_metric_constraint: 7 }
            curve { tp: 10 tn: 10 fp: 10 fn: 10 }
          }
          rocs {
            auc: 10
            pr_auc: 11
            ap: 12
            precision_at_recall { x_metric_value: 13 y_metric_constraint: 14 }
            precision_at_recall { x_metric_value: 15 y_metric_constraint: 16 }
            curve { tp: 10 tn: 10 fp: 10 fn: 10 }
          }
        }
        label_column {
          categorical {
            number_of_unique_values: 2
            is_already_integerized: true
          }
        }
      )pb");

  auto& confusion = *evaluation.mutable_classification()->mutable_confusion();
  utils::InitializeConfusionMatrixProto(2, 2, &confusion);
  utils::AddToConfusionMatrixProto(0, 0, 5, &confusion);
  utils::AddToConfusionMatrixProto(0, 1, 5, &confusion);
  evaluation.set_count_predictions(10);
  const auto flat_metrics = ExtractFlatMetrics("test", evaluation);
  const std::unordered_map<std::string, std::string> expected{
      {"Model", "test"},
      {"Accuracy", "0.5"},
      {"Accuracy CI95% [W]", "0 1"},
      {"AUC (0 vs others)", "1"},
      {"AUC CI95% [H] (1 vs others)", "nan nan"},
      {"AUC (1 vs others)", "10"},
      {"AUC CI95% [H] (0 vs others)", "1 1"},
      {"P/R AUC (0 vs others)", "2"},
      {"P/R AUC CI95% [L] (0 vs others)", "nan nan"},
      {"P/R AUC (1 vs others)", "11"},
      {"P/R AUC CI95% [L] (1 vs others)", "nan nan"},
      {"AP (0 vs others)", "3"},
      {"AP (1 vs others)", "12"},
      {"Precision@Recall=5 (0 vs others)", "4"},
      {"Precision@Recall=7 (0 vs others)", "6"},
      {"Precision@Recall=14 (1 vs others)", "13"},
      {"Precision@Recall=16 (1 vs others)", "15"}};
  EXPECT_EQ(flat_metrics, expected);
}

TEST(Metric, ComputeXAtYMetrics) {
  const proto::EvaluationOptions options = PARSE_TEST_PROTO(
      R"pb(
        task: CLASSIFICATION
        classification {
          precision_at_recall: 0.90
          precision_at_recall: 0.70
          precision_at_recall: 0.50
          precision_at_recall: 0.30

          recall_at_precision: 0.90
          recall_at_precision: 0.60
          recall_at_precision: 0.40

          precision_at_volume: 0.50
          precision_at_volume: 0.90

          recall_at_false_positive_rate: 0.50
          recall_at_false_positive_rate: 0.90

          false_positive_rate_at_recall: 0.50
          false_positive_rate_at_recall: 0.20
        }
      )pb");
  proto::Roc roc = PARSE_TEST_PROTO(
      R"pb(
        # Recall: 0.8 Prec: 0.5 Vol: 0.8 fpr: 0.8
        curve { tp: 160 tn: 40 fp: 160 fn: 40 }
        # Recall: 0.5 Prec: 0.8 vol:~0.31 fpr: ~0.21
        curve { tp: 100 tn: 175 fp: 25 fn: 100 }
      )pb");
  ComputeXAtYMetrics(options, roc.curve(), &roc);

  // Precision @ fixed Recall
  EXPECT_EQ(roc.precision_at_recall(0).y_metric_constraint(), 0.9);
  EXPECT_TRUE(std::isnan(roc.precision_at_recall(0).x_metric_value()));
  EXPECT_TRUE(std::isnan(roc.precision_at_recall(0).threshold()));

  EXPECT_THAT(
      roc.precision_at_recall(1),
      EqualsProto(PARSE_TEST_PROTO_WITH_TYPE(proto::Roc::XAtYMetric, R"pb(
        y_metric_constraint: 0.7
        x_metric_value: 0.5
        threshold: 0)pb")));

  EXPECT_THAT(
      roc.precision_at_recall(2),
      EqualsProto(PARSE_TEST_PROTO_WITH_TYPE(proto::Roc::XAtYMetric, R"pb(
        y_metric_constraint: 0.5
        x_metric_value: 0.8
        threshold: 0)pb")));

  EXPECT_THAT(
      roc.precision_at_recall(3),
      EqualsProto(PARSE_TEST_PROTO_WITH_TYPE(proto::Roc::XAtYMetric, R"pb(
        y_metric_constraint: 0.3
        x_metric_value: 0.8
        threshold: 0)pb")));

  // Recall @ fixed Precision
  EXPECT_EQ(roc.recall_at_precision(0).y_metric_constraint(), 0.9);
  EXPECT_TRUE(std::isnan(roc.recall_at_precision(0).x_metric_value()));
  EXPECT_TRUE(std::isnan(roc.recall_at_precision(0).threshold()));

  EXPECT_THAT(
      roc.recall_at_precision(1),
      EqualsProto(PARSE_TEST_PROTO_WITH_TYPE(proto::Roc::XAtYMetric, R"pb(
        y_metric_constraint: 0.6
        x_metric_value: 0.5
        threshold: 0)pb")));

  EXPECT_THAT(
      roc.recall_at_precision(2),
      EqualsProto(PARSE_TEST_PROTO_WITH_TYPE(proto::Roc::XAtYMetric, R"pb(
        y_metric_constraint: 0.4
        x_metric_value: 0.8
        threshold: 0)pb")));

  // Precision @ fixed Volume
  EXPECT_THAT(
      roc.precision_at_volume(0),
      EqualsProto(PARSE_TEST_PROTO_WITH_TYPE(proto::Roc::XAtYMetric, R"pb(
        y_metric_constraint: 0.5
        x_metric_value: 0.5
        threshold: 0)pb")));

  EXPECT_EQ(roc.precision_at_volume(1).y_metric_constraint(), 0.9);
  EXPECT_TRUE(std::isnan(roc.recall_at_precision(0).x_metric_value()));
  EXPECT_TRUE(std::isnan(roc.recall_at_precision(0).threshold()));

  // Recall @ fixed False positive rate
  EXPECT_THAT(
      roc.recall_at_false_positive_rate(0),
      EqualsProto(PARSE_TEST_PROTO_WITH_TYPE(proto::Roc::XAtYMetric, R"pb(
        y_metric_constraint: 0.5
        x_metric_value: 0.5
        threshold: 0)pb")));

  EXPECT_THAT(
      roc.recall_at_false_positive_rate(1),
      EqualsProto(PARSE_TEST_PROTO_WITH_TYPE(proto::Roc::XAtYMetric, R"pb(
        y_metric_constraint: 0.9
        x_metric_value: 0.8
        threshold: 0)pb")));

  // False positive rate @ fixed Recall
  EXPECT_THAT(
      roc.false_positive_rate_at_recall(0),
      EqualsProto(PARSE_TEST_PROTO_WITH_TYPE(proto::Roc::XAtYMetric, R"pb(
        y_metric_constraint: 0.5
        x_metric_value: 0.125
        threshold: 0)pb")));

  EXPECT_THAT(
      roc.false_positive_rate_at_recall(1),
      EqualsProto(PARSE_TEST_PROTO_WITH_TYPE(proto::Roc::XAtYMetric, R"pb(
        y_metric_constraint: 0.2
        x_metric_value: 0.125
        threshold: 0)pb")));
}

TEST(Metric, AccuracyConfidenceInterval) {
  proto::EvaluationResults eval;
  eval.set_task(model::proto::Task::CLASSIFICATION);
  auto& confusion = *eval.mutable_classification()->mutable_confusion();
  utils::InitializeConfusionMatrixProto(2, 2, &confusion);
  utils::AddToConfusionMatrixProto(0, 0, 100, &confusion);
  utils::AddToConfusionMatrixProto(0, 1, 50, &confusion);
  utils::AddToConfusionMatrixProto(1, 1, 50, &confusion);
  eval.set_count_predictions(200);
  eval.set_count_predictions_no_weight(200);

  // Ground truth values have been obtained with the "Hmisc" R package:
  // >> binconf(150, 200, alpha=0.05, method="wilson")
  // PointEst    Lower     Upper
  //     0.75 0.685659 0.8049183
  const auto ci = AccuracyConfidenceInterval(eval, 0.95);
  EXPECT_NEAR(ci.first, 0.686, 0.01);
  EXPECT_NEAR(ci.second, 0.805, 0.01);
}

TEST(Metric, AUCConfidenceInterval) {
  const proto::Roc roc = PARSE_TEST_PROTO(
      R"pb(
        curve { tp: 100 fp: 100 tn: 0 fn: 0 }
        auc: 0.75
      )pb");
  const auto ci = AUCConfidenceInterval(roc, 0.95);
  EXPECT_NEAR(ci.first, 0.683, 0.01);
  EXPECT_NEAR(ci.second, 0.817, 0.01);
}

TEST(Metric, PRAUCConfidenceInterval) {
  const proto::Roc roc = PARSE_TEST_PROTO(
      R"pb(
        curve { tp: 100 fp: 100 tn: 0 fn: 0 }
        pr_auc: 0.75
      )pb");
  const auto ci = PRAUCConfidenceInterval(roc, 0.95);
  EXPECT_NEAR(ci.first, 0.656f, 0.01f);
  EXPECT_NEAR(ci.second, 0.825f, 0.01f);
}

TEST(Metric, XAtYMetricsAccessors) {
  const auto accessors = XAtYMetricsAccessors();
  EXPECT_EQ(accessors.size(), 5);
  proto::Roc roc;
  for (const auto& accessor : accessors) {
    const google::protobuf::RepeatedPtrField<proto::Roc::XAtYMetric>* a1 =
        accessor.mutable_access(&roc);
    const google::protobuf::RepeatedPtrField<proto::Roc::XAtYMetric>* a2 =
        &accessor.const_access(roc);
    EXPECT_EQ(a1, a2);
  }
}

TEST(Metric, GetQuantiles) {
  std::vector<proto::Roc> samples;
  const int n = 1000;
  for (int i = 0; i < n; i++) {
    proto::Roc roc;
    roc.set_auc(static_cast<float>(i) / n);
    samples.push_back(roc);
  }
  const auto quantiles = internal::GetQuantiles(
      samples, [](const proto::Roc& roc) -> double { return roc.auc(); }, 0.5f,
      0.975f);
  EXPECT_NEAR(quantiles.first, 0.5f, 0.01);
  EXPECT_NEAR(quantiles.second, 0.975f, 0.01);
}

TEST(Metric, ComputeRocConfidenceIntervalsUsingBootstrapping) {
  const proto::EvaluationOptions options = PARSE_TEST_PROTO(
      R"pb(
        task: CLASSIFICATION
        classification { precision_at_recall: 0.50 recall_at_precision: 0.50 }
      )pb");
  const std::vector<BinaryPrediction> sorted_predictions = {
      {0.0, false, 1}, {0.1, false, 1}, {0.2, false, 1}, {0.3, true, 1},
      {0.4, false, 1}, {0.5, false, 1}, {0.6, true, 1},  {0.7, true, 1},
      {0.8, false, 1}, {0.9, true, 1},  {1.0, true, 1}};
  proto::Roc roc;
  internal::ComputeRocConfidenceIntervalsUsingBootstrapping(
      options, sorted_predictions, &roc);

  EXPECT_NEAR(roc.bootstrap_lower_bounds_95p().auc(), 0.50, 0.10);
  EXPECT_NEAR(roc.bootstrap_upper_bounds_95p().auc(), 1.0, 0.01);

  EXPECT_NEAR(roc.bootstrap_lower_bounds_95p().pr_auc(), 0.30, 0.10);
  EXPECT_NEAR(roc.bootstrap_upper_bounds_95p().pr_auc(), 1.00, 0.01);

  EXPECT_EQ(roc.bootstrap_lower_bounds_95p().precision_at_recall_size(), 1);
  EXPECT_EQ(roc.bootstrap_upper_bounds_95p().precision_at_recall_size(), 1);
  EXPECT_NEAR(
      roc.bootstrap_lower_bounds_95p().precision_at_recall(0).x_metric_value(),
      0.4, 0.02);
  EXPECT_NEAR(
      roc.bootstrap_upper_bounds_95p().precision_at_recall(0).x_metric_value(),
      1.00, 0.02);

  EXPECT_EQ(roc.bootstrap_lower_bounds_95p().recall_at_precision_size(), 1);
  EXPECT_EQ(roc.bootstrap_upper_bounds_95p().recall_at_precision_size(), 1);
  EXPECT_NEAR(
      roc.bootstrap_lower_bounds_95p().recall_at_precision(0).x_metric_value(),
      0.33, 0.02);
  EXPECT_NEAR(
      roc.bootstrap_upper_bounds_95p().recall_at_precision(0).x_metric_value(),
      1.00, 0.02);
}

TEST(Metric, ExtractFlatMetricsRegression) {
  proto::EvaluationResults evaluation;
  evaluation.set_task(model::proto::Task::REGRESSION);
  evaluation.mutable_regression()->set_sum_square_error(10 * 4 * 4);
  evaluation.set_count_predictions(10);
  evaluation.set_count_predictions_no_weight(10);
  const auto flat_metrics = ExtractFlatMetrics("test", evaluation);
  const std::unordered_map<std::string, std::string> expected{
      {"Model", "test"}, {"Rmse", "4"}, {"Rmse CI95% [X2]", "2.79487 7.01973"}};
  EXPECT_EQ(flat_metrics, expected);
}

TEST(Metric, DefaultRMSE) {
  // Labels: 1 2 3 4
  // Label sd: sqrt(mean(l^2)-mean(l)^2) = 1.118
  const proto::EvaluationResults eval = PARSE_TEST_PROTO(
      R"pb(
        task: REGRESSION
        count_predictions: 4
        regression { sum_label: 10 sum_square_label: 30 }
      )pb");
  EXPECT_NEAR(DefaultRMSE(eval), 1.118, 0.02f);
}

// We generate "num_runs" dataset, each composed of "num_predictions" example.
// The residual is sampled with a normal distribution of known variance. The
// test checks that approximately 95% of the time, the real RMSE is in the
// confidence interval.
//
// Evaluates both bootstrapping and closed form estimators.
TEST(Metric, RMSEConfidenceIntervals) {
  const proto::EvaluationOptions options = PARSE_TEST_PROTO(
      R"pb(
        task: REGRESSION
      )pb");
  utils::RandomEngine rnd;
  std::normal_distribution<float> dist_residual;
  std::uniform_real_distribution<float> dist_label(0, 10);

  const float real_rmse = 3.0;
  const int num_runs = 1000;
  const int num_predictions = 100;

  dataset::proto::Column label_column;
  label_column.set_type(dataset::proto::ColumnType::NUMERICAL);

  int count_in_confidence_interval_boot = 0;
  int count_in_confidence_interval_closed = 0;
  for (int run = 0; run < num_runs; run++) {
    // Simulate a dataset evaluation.
    proto::EvaluationResults eval;
    InitializeEvaluation(options, label_column, &eval);
    for (int pred_idx = 0; pred_idx < num_predictions; pred_idx++) {
      model::proto::Prediction pred;
      const float label = dist_label(rnd);
      const float residual = dist_residual(rnd) * real_rmse;
      pred.mutable_regression()->set_ground_truth(label + residual);
      pred.mutable_regression()->set_value(label);
      AddPrediction(options, pred, &rnd, &eval);
    }
    FinalizeEvaluation(options, label_column, &eval);
    // Check validity of confidence intervals.
    const bool in_confidence_interval_boot =
        (real_rmse >= eval.regression().bootstrap_rmse_lower_bounds_95p()) &&
        (real_rmse <= eval.regression().bootstrap_rmse_upper_bounds_95p());
    if (in_confidence_interval_boot) {
      count_in_confidence_interval_boot++;
    }

    const auto closed_form_ci = RMSEConfidenceInterval(eval);
    const bool in_confidence_interval_closed =
        (real_rmse >= closed_form_ci.first) &&
        (real_rmse <= closed_form_ci.second);
    if (in_confidence_interval_closed) {
      count_in_confidence_interval_closed++;
    }
  }
  // ~95% of the CI should be valid.
  EXPECT_NEAR(static_cast<double>(count_in_confidence_interval_boot) / num_runs,
              0.95, 0.03);
  EXPECT_NEAR(
      static_cast<double>(count_in_confidence_interval_closed) / num_runs, 0.95,
      0.03);
}

TEST(Metric, GetMetric) {
  const proto::EvaluationResults results_regression = PARSE_TEST_PROTO(R"pb(
    task: REGRESSION
    label_column { type: NUMERICAL }
    regression { sum_square_error: 10 }
    count_predictions: 10
  )pb");
  EXPECT_NEAR(
      GetMetric(results_regression, PARSE_TEST_PROTO("regression { rmse {}}")),
      RMSE(results_regression), 0.0001);

  const proto::EvaluationResults results_classification = PARSE_TEST_PROTO(R"pb(
    task: CLASSIFICATION
    label_column {
      type: CATEGORICAL
      categorical { number_of_unique_values: 2 is_already_integerized: 1 }
    }
    classification {
      confusion {
        sum: 10
        nrow: 2
        ncol: 2
        counts: 1
        counts: 4
        counts: 2
        counts: 3
      }
      rocs {
        auc: 0.1
        pr_auc: 0.2
        ap: 0.3
        precision_at_recall { x_metric_value: 0.4 y_metric_constraint: 0.5 }
        recall_at_precision { x_metric_value: 0.6 y_metric_constraint: 0.7 }
        precision_at_volume { x_metric_value: 0.8 y_metric_constraint: 0.9 }
        recall_at_false_positive_rate {
          x_metric_value: 1.0
          y_metric_constraint: 1.1
        }
        false_positive_rate_at_recall {
          x_metric_value: 1.2
          y_metric_constraint: 1.3
        }
      }
      rocs { auc: 1.4 pr_auc: 1.5 ap: 1.6 }
    }
    count_predictions: 10
  )pb");

  EXPECT_NEAR(GetMetric(results_classification,
                        PARSE_TEST_PROTO("classification { accuracy {}}")),
              Accuracy(results_classification), 0.0001);
  EXPECT_NEAR(GetMetric(results_classification,
                        PARSE_TEST_PROTO("classification { logloss {}}")),
              LogLoss(results_classification), 0.0001);

  EXPECT_NEAR(GetMetric(results_classification, PARSE_TEST_PROTO(R"pb(
                          classification {
                            one_vs_other {
                              positive_class: "1"
                              auc {}
                            }
                          })pb")),
              results_classification.classification().rocs(1).auc(), 0.0001);

  EXPECT_NEAR(GetMetric(results_classification, PARSE_TEST_PROTO(R"pb(
                          classification {
                            one_vs_other {
                              positive_class: "0"
                              auc {}
                            }
                          })pb")),
              results_classification.classification().rocs(0).auc(), 0.0001);

  EXPECT_NEAR(GetMetric(results_classification, PARSE_TEST_PROTO(R"pb(
                          classification {
                            one_vs_other {
                              positive_class: "0"
                              ap {}
                            }
                          })pb")),
              results_classification.classification().rocs(0).ap(), 0.0001);

  EXPECT_NEAR(GetMetric(results_classification, PARSE_TEST_PROTO(R"pb(
                          classification {
                            one_vs_other {
                              positive_class: "0"
                              pr_auc {}
                            }
                          })pb")),
              results_classification.classification().rocs(0).pr_auc(), 0.0001);

  EXPECT_NEAR(GetMetric(results_classification, PARSE_TEST_PROTO(R"pb(
                          classification {
                            one_vs_other {
                              positive_class: "0"
                              precision_at_recall { recall: 0.5 }
                            }
                          })pb")),
              results_classification.classification()
                  .rocs(0)
                  .precision_at_recall(0)
                  .x_metric_value(),
              0.0001);

  EXPECT_NEAR(GetMetric(results_classification, PARSE_TEST_PROTO(R"pb(
                          classification {
                            one_vs_other {
                              positive_class: "0"
                              recall_at_precision { precision: 0.7 }
                            }
                          })pb")),
              results_classification.classification()
                  .rocs(0)
                  .recall_at_precision(0)
                  .x_metric_value(),
              0.0001);

  EXPECT_NEAR(GetMetric(results_classification, PARSE_TEST_PROTO(R"pb(
                          classification {
                            one_vs_other {
                              positive_class: "0"
                              precision_at_volume { volume: 0.9 }
                            }
                          })pb")),
              results_classification.classification()
                  .rocs(0)
                  .precision_at_volume(0)
                  .x_metric_value(),
              0.0001);

  EXPECT_NEAR(
      GetMetric(results_classification, PARSE_TEST_PROTO(R"pb(
                  classification {
                    one_vs_other {
                      positive_class: "0"
                      recall_at_false_positive_rate { false_positive_rate: 1.1 }
                    }
                  })pb")),
      results_classification.classification()
          .rocs(0)
          .recall_at_false_positive_rate(0)
          .x_metric_value(),
      0.0001);

  EXPECT_NEAR(GetMetric(results_classification, PARSE_TEST_PROTO(R"pb(
                          classification {
                            one_vs_other {
                              positive_class: "0"
                              false_positive_rate_at_recall { recall: 1.3 }
                            }
                          })pb")),
              results_classification.classification()
                  .rocs(0)
                  .false_positive_rate_at_recall(0)
                  .x_metric_value(),
              0.0001);
}

TEST(Metric, MinMaxStream) {
  MinMaxStream<int> bounds;
  EXPECT_TRUE(bounds.empty());

  bounds.visit(5);
  EXPECT_FALSE(bounds.empty());
  EXPECT_EQ(bounds.max(), 5);
  EXPECT_EQ(bounds.min(), 5);

  bounds.visit(2);
  EXPECT_FALSE(bounds.empty());
  EXPECT_EQ(bounds.max(), 5);
  EXPECT_EQ(bounds.min(), 2);
}

TEST(Metric, BinaryClassificationEvaluationHelper) {
  const float label_correlation = 0.7;
  const int num_examples = 10000;

  // Generate a random set of predictions.
  utils::RandomEngine rnd;
  std::normal_distribution<float> unit_normal_dist;
  std::uniform_int_distribution<int> label_dist(0, 1);

  std::vector<bool> binary_labels;
  std::vector<float> prediction_positive;
  binary_labels.reserve(num_examples);
  prediction_positive.reserve(num_examples);

  for (int example_idx = 0; example_idx < num_examples; example_idx++) {
    const bool label = label_dist(rnd);
    const float prediction =
        utils::clamp(unit_normal_dist(rnd) * (1 - label_correlation) +
                         label * label_correlation,
                     0.f, 1.f);
    binary_labels.push_back(label);
    prediction_positive.push_back(prediction);
  }

  // Evaluate the predictions.
  proto::EvaluationOptions options;
  // 10'000 examples is enough for the closed form estimation of the confidence
  // intervals.
  options.set_bootstrapping_samples(0);
  const proto::EvaluationResults eval = BinaryClassificationEvaluationHelper(
      binary_labels, prediction_positive, options, &rnd);

  // String readable report.
  std::string report;
  AppendTextReport(eval, &report);
  LOG(INFO) << "Report :\n " << report;
}

TEST(Metric, EvaluationOfRanking) {
  // Create a fake column specification.
  dataset::proto::Column label_column;
  label_column.set_type(dataset::proto::ColumnType::NUMERICAL);
  label_column.set_name("label");

  // Configure the evaluation.
  utils::RandomEngine rnd;
  proto::EvaluationOptions option;
  option.set_task(model::proto::Task::RANKING);
  option.mutable_ranking()->set_allow_only_one_group(true);

  // Initialize.
  proto::EvaluationResults eval;
  InitializeEvaluation(option, label_column, &eval);
  model::proto::Prediction pred;

  // Add some predictions.
  pred.mutable_ranking()->set_relevance(10);
  pred.mutable_ranking()->set_ground_truth_relevance(3);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(9);
  pred.mutable_ranking()->set_ground_truth_relevance(2);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(8);
  pred.mutable_ranking()->set_ground_truth_relevance(3);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(7);
  pred.mutable_ranking()->set_ground_truth_relevance(0);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(6);
  pred.mutable_ranking()->set_ground_truth_relevance(1);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(5);
  pred.mutable_ranking()->set_ground_truth_relevance(2);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  // Finalize.
  FinalizeEvaluation(option, label_column, &eval);

  EXPECT_EQ(eval.count_predictions(), 6);
  EXPECT_EQ(eval.count_predictions_no_weight(), 6);
  EXPECT_NEAR(eval.ranking().mean_num_items_in_group(), 6, 0.01);
  EXPECT_EQ(eval.count_sampled_predictions(), 6);
  EXPECT_EQ(eval.task(), model::proto::Task::RANKING);
  // R> 0.8755944 = sum((2^c(3,2,3,0,1)-1)/log2(seq(5)+1)) /
  //   sum((2^c(3,3,2,2,1)-1)/log2(seq(5)+1))
  EXPECT_NEAR(NDCG(eval), 0.8755944, 0.01);

  // R=1
  EXPECT_NEAR(MRR(eval), 1.0, 0.01);

  EXPECT_NEAR(PrecisionAt1(eval), 1.0, 0.01);

  EXPECT_EQ(eval.ranking().num_groups(), 1);

  // Since there is only one group, the bootstrap metrics are equal to the mean.
  EXPECT_NEAR(eval.ranking().ndcg().bootstrap_based_95p().lower(), NDCG(eval),
              0.01);
  EXPECT_NEAR(eval.ranking().ndcg().bootstrap_based_95p().upper(), NDCG(eval),
              0.01);

  EXPECT_NEAR(eval.ranking().mrr().bootstrap_based_95p().lower(), MRR(eval),
              0.01);
  EXPECT_NEAR(eval.ranking().mrr().bootstrap_based_95p().upper(), MRR(eval),
              0.01);

  // R> sum(mean(2^c(3,3,2,2,1,0)-1)/log2(seq(5)+1)) /
  //   sum((2^c(3,3,2,2,1)-1)/log2(seq(5)+1))
  // 0.7070456
  EXPECT_NEAR(DefaultNDCG(eval), 0.7070456, 0.01);

  // Create reports.
  std::string report;
  AppendTextReport(eval, &report);
  LOG(INFO) << "Report :\n " << report;
}

TEST(Metric, EvaluationOfRankingMRR) {
  // Create a fake column specification.
  dataset::proto::Column label_column;
  label_column.set_type(dataset::proto::ColumnType::NUMERICAL);
  label_column.set_name("label");

  // Configure the evaluation.
  utils::RandomEngine rnd;
  proto::EvaluationOptions option;
  option.mutable_ranking()->set_allow_only_one_group(true);
  option.set_task(model::proto::Task::RANKING);

  // Initialize.
  proto::EvaluationResults eval;
  InitializeEvaluation(option, label_column, &eval);
  model::proto::Prediction pred;

  // Add some predictions.
  pred.mutable_ranking()->set_relevance(1);
  pred.mutable_ranking()->set_ground_truth_relevance(0);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(2);
  pred.mutable_ranking()->set_ground_truth_relevance(0);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(4);
  pred.mutable_ranking()->set_ground_truth_relevance(0);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(3);
  pred.mutable_ranking()->set_ground_truth_relevance(1);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  // Finalize.
  FinalizeEvaluation(option, label_column, &eval);

  EXPECT_EQ(eval.task(), model::proto::Task::RANKING);

  EXPECT_NEAR(MRR(eval), 0.5, 0.01);
}

TEST(Metric, EvaluationOfRankingWithTies) {
  // Create a fake column specification.
  dataset::proto::Column label_column;
  label_column.set_type(dataset::proto::ColumnType::NUMERICAL);
  label_column.set_name("label");

  // Configure the evaluation.
  utils::RandomEngine rnd;
  proto::EvaluationOptions option;
  option.set_task(model::proto::Task::RANKING);
  option.mutable_ranking()->set_allow_only_one_group(true);

  // Initialize.
  proto::EvaluationResults eval;
  InitializeEvaluation(option, label_column, &eval);
  model::proto::Prediction pred;

  // Add some predictions.
  // Note: All six examples have the same model prediction. The trucation is set
  // at 5. Therefore, the ties should convider all 6 examples (instead of the
  // first 5).
  pred.mutable_ranking()->set_relevance(1);
  pred.mutable_ranking()->set_ground_truth_relevance(3);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(1);
  pred.mutable_ranking()->set_ground_truth_relevance(2);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(1);
  pred.mutable_ranking()->set_ground_truth_relevance(3);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(1);
  pred.mutable_ranking()->set_ground_truth_relevance(0);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(1);
  pred.mutable_ranking()->set_ground_truth_relevance(1);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(1);
  pred.mutable_ranking()->set_ground_truth_relevance(2);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  // Finalize.
  FinalizeEvaluation(option, label_column, &eval);

  EXPECT_EQ(eval.count_predictions(), 6);
  EXPECT_EQ(eval.count_predictions_no_weight(), 6);
  EXPECT_EQ(eval.count_sampled_predictions(), 6);
  EXPECT_EQ(eval.task(), model::proto::Task::RANKING);

  // A ties-less evaluation would give a random NDCG between 0 and 1.
  //
  // Instead, the tried-aware evaluation should return:
  // R> 0.7070456 = sum(mean(2^c(3,2,3,0,1,2)-1)/log2(seq(5)+1)) /
  // sum((2^c(3,3,2,2,1)-1)/log2(seq(5)+1))
  EXPECT_NEAR(NDCG(eval), 0.7070456, 0.01);

  EXPECT_EQ(eval.ranking().num_groups(), 1);
}

TEST(Metric, EvaluationOfRankingWithTiesV2) {
  // Create a fake column specification.
  dataset::proto::Column label_column;
  label_column.set_type(dataset::proto::ColumnType::NUMERICAL);
  label_column.set_name("label");

  // Configure the evaluation.
  utils::RandomEngine rnd;
  proto::EvaluationOptions option;
  option.set_task(model::proto::Task::RANKING);
  option.mutable_ranking()->set_allow_only_one_group(true);

  // Initialize.
  proto::EvaluationResults eval;
  InitializeEvaluation(option, label_column, &eval);
  model::proto::Prediction pred;

  // Add some predictions.
  // Note: There are three groups of examples with equal predictions but
  // different label values. The ties are computed per group.
  pred.mutable_ranking()->set_relevance(3);
  pred.mutable_ranking()->set_ground_truth_relevance(3);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(3);
  pred.mutable_ranking()->set_ground_truth_relevance(2);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(2);
  pred.mutable_ranking()->set_ground_truth_relevance(3);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(1);
  pred.mutable_ranking()->set_ground_truth_relevance(0);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(1);
  pred.mutable_ranking()->set_ground_truth_relevance(1);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(1);
  pred.mutable_ranking()->set_ground_truth_relevance(2);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(0);
  pred.mutable_ranking()->set_ground_truth_relevance(4);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  // Finalize.
  FinalizeEvaluation(option, label_column, &eval);

  EXPECT_EQ(eval.count_predictions(), 7);
  EXPECT_EQ(eval.count_predictions_no_weight(), 7);
  EXPECT_EQ(eval.count_sampled_predictions(), 7);
  EXPECT_EQ(eval.task(), model::proto::Task::RANKING);

  // A ties-less evaluation would give a random NDCG between 0 and 1.
  //
  // Instead, the tried-aware evaluation should return:
  // R> m1 = mean(2^c(3,2)-1)
  // R> m2 = mean(2^c(3)-1)
  // R> m3 = mean(2^c(0,1,2)-1)
  // R> 0.5023706 = sum(c(m1,m1,m2,m3,m3)/log2(seq(5)+1)) /
  // sum((2^c(4,3,3,2,2)-1)/log2(seq(5)+1))
  EXPECT_NEAR(NDCG(eval), 0.5023706, 0.01);

  EXPECT_EQ(eval.ranking().num_groups(), 1);
}

TEST(Metric, EvaluationOfRankingPrecisionAt1) {
  // Create a fake column specification.
  dataset::proto::Column label_column;
  label_column.set_type(dataset::proto::ColumnType::NUMERICAL);
  label_column.set_name("label");

  // Configure the evaluation.
  utils::RandomEngine rnd;
  proto::EvaluationOptions option;
  option.mutable_ranking()->set_allow_only_one_group(true);
  option.set_task(model::proto::Task::RANKING);

  // Initialize.
  proto::EvaluationResults eval;
  InitializeEvaluation(option, label_column, &eval);
  model::proto::Prediction pred;

  // Add some predictions.
  pred.mutable_ranking()->set_relevance(1);
  pred.mutable_ranking()->set_ground_truth_relevance(0);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(2);
  pred.mutable_ranking()->set_ground_truth_relevance(1);
  pred.mutable_ranking()->set_group_id(1);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(2);
  pred.mutable_ranking()->set_ground_truth_relevance(0);
  pred.mutable_ranking()->set_group_id(2);
  AddPrediction(option, pred, &rnd, &eval);

  pred.mutable_ranking()->set_relevance(1);
  pred.mutable_ranking()->set_ground_truth_relevance(1);
  pred.mutable_ranking()->set_group_id(2);
  AddPrediction(option, pred, &rnd, &eval);

  // Finalize.
  FinalizeEvaluation(option, label_column, &eval);

  EXPECT_EQ(eval.task(), model::proto::Task::RANKING);

  EXPECT_NEAR(PrecisionAt1(eval), 0.5, 0.01);
}

TEST(Metric, EvaluationOfRankingCI) {
  // Create a fake column specification.
  dataset::proto::Column label_column;
  label_column.set_type(dataset::proto::ColumnType::NUMERICAL);
  label_column.set_name("label");

  // Configure the evaluation.
  utils::RandomEngine rnd;
  proto::EvaluationOptions option;
  option.set_task(model::proto::Task::RANKING);
  std::uniform_real_distribution<float> prediction_dist(0, 1);

  // Initialize.
  proto::EvaluationResults eval;
  InitializeEvaluation(option, label_column, &eval);
  model::proto::Prediction pred;

  const int num_groups = 500;
  const int num_items_per_groups = 5;

  // Add some predictions.
  // Note: The predictions are generated randomly. Therefore, the NDCG should be
  // close to the default NDCG. In addition, we generate enought predictions for
  // the CI to be somehow close to a closed form CI computed with a Studient
  // distribution.
  for (int group_idx = 0; group_idx < num_groups; group_idx++) {
    for (int item_idx = 0; item_idx < num_items_per_groups; item_idx++) {
      pred.mutable_ranking()->set_relevance(prediction_dist(rnd));
      pred.mutable_ranking()->set_ground_truth_relevance(item_idx);
      pred.mutable_ranking()->set_group_id(group_idx);
      AddPrediction(option, pred, &rnd, &eval);
    }
  }

  // Finalize.
  FinalizeEvaluation(option, label_column, &eval);

  EXPECT_EQ(eval.count_predictions(), num_groups * num_items_per_groups);
  EXPECT_EQ(eval.count_predictions_no_weight(),
            num_groups * num_items_per_groups);
  EXPECT_EQ(eval.count_sampled_predictions(),
            num_groups * num_items_per_groups);
  EXPECT_EQ(eval.task(), model::proto::Task::RANKING);
  EXPECT_EQ(eval.ranking().num_groups(), num_groups);
  EXPECT_NEAR(eval.ranking().mean_num_items_in_group(), num_items_per_groups,
              0.01);
  EXPECT_EQ(eval.ranking().min_num_items_in_group(), num_items_per_groups);
  EXPECT_EQ(eval.ranking().max_num_items_in_group(), num_items_per_groups);

  // We assume these values are correct from the previous tests.
  EXPECT_NEAR(DefaultNDCG(eval), 0.71822, 0.005);
  EXPECT_NEAR(NDCG(eval), 0.716362, 0.005);

  // Note: Standard deviation was measured at ~0.14379.
  // NDCG(eval) - 1.96 * 0.14379 / sqrt(500) = 0.7037583 ~= 0.703178
  // NDCG(eval) + 1.96 * 0.14379 / sqrt(500) = 0.7289657 ~= 0.729018
  //
  // These values correspond to the observed values. Note: the student
  // approximation seems optimistic.
  EXPECT_NEAR(eval.ranking().ndcg().bootstrap_based_95p().lower(), 0.703178,
              0.005);
  EXPECT_NEAR(eval.ranking().ndcg().bootstrap_based_95p().upper(), 0.729018,
              0.005);

  // Create reports.
  std::string report;
  AppendTextReport(eval, &report);
  LOG(INFO) << "Report :\n " << report;
}

TEST(Metric, RMSE) {
  // R> sqrt(mean((c(1,2,3)-c(1,3,4))^2))
  // 0.8164966
  EXPECT_NEAR(RMSE(/*labels=*/{1, 2, 3}, /*predictions=*/{1, 3, 4},
                   /*weights=*/{1, 1, 1}),
              0.8164966, 0.0001);

  // R> sqrt(mean((c(1,2,2,3,3,3)-c(1,3,3,4,4,4))^2))
  // 0.9128709
  EXPECT_NEAR(RMSE(/*labels=*/{1, 2, 3}, /*predictions=*/{1, 3, 4},
                   /*weights=*/{1, 2, 3}),
              0.9128709, 0.0001);

  EXPECT_NEAR(RMSE(/*labels=*/{1, 2, 3}, /*predictions=*/{1, 3, 4}), 0.8164966,
              0.0001);
}

TEST(DefaultMetrics, Classification) {
  const dataset::proto::Column label = PARSE_TEST_PROTO(
      R"pb(
        type: CATEGORICAL
        categorical { is_already_integerized: true number_of_unique_values: 3 }
      )pb");
  const auto metrics =
      DefaultMetrics(model::proto::Task::CLASSIFICATION, label);
  EXPECT_EQ(metrics.size(), 4);
  EXPECT_EQ(metrics[0].name, "ACCURACY");
  proto::MetricAccessor expected_0 = PARSE_TEST_PROTO(
      R"pb(
        classification: { accuracy {} }
      )pb");
  EXPECT_THAT(metrics[0].accessor, EqualsProto(expected_0));

  EXPECT_EQ(metrics[1].name, "AUC_2_VS_OTHERS");
  proto::MetricAccessor expected_1 = PARSE_TEST_PROTO(
      R"pb(
        classification {
          one_vs_other {
            positive_class: "2"
            auc {}
          }
        }
      )pb");
  EXPECT_THAT(metrics[1].accessor, EqualsProto(expected_1));

  EXPECT_EQ(metrics[2].name, "PRAUC_2_VS_OTHERS");
  proto::MetricAccessor expected_2 = PARSE_TEST_PROTO(
      R"pb(
        classification {
          one_vs_other {
            positive_class: "2"
            pr_auc {}
          }
        }
      )pb");
  EXPECT_THAT(metrics[2].accessor, EqualsProto(expected_2));

  EXPECT_EQ(metrics[3].name, "AP_2_VS_OTHERS");
  proto::MetricAccessor expected_3 = PARSE_TEST_PROTO(
      R"pb(
        classification {
          one_vs_other {
            positive_class: "2"
            ap {}
          }
        }
      )pb");
  EXPECT_THAT(metrics[3].accessor, EqualsProto(expected_3));
}

TEST(DefaultMetrics, Regression) {
  const auto metrics = DefaultMetrics(model::proto::Task::REGRESSION, {});
  EXPECT_EQ(metrics.size(), 1);
  EXPECT_EQ(metrics[0].name, "RMSE");
  proto::MetricAccessor expected = PARSE_TEST_PROTO(
      R"pb(
        regression { rmse {} }
      )pb");
  EXPECT_THAT(metrics[0].accessor, EqualsProto(expected));
}

TEST(DefaultMetrics, Ranking) {
  const auto metrics = DefaultMetrics(model::proto::Task::RANKING, {});
  EXPECT_EQ(metrics.size(), 1);
  EXPECT_EQ(metrics[0].name, "NDCG");
  proto::MetricAccessor expected = PARSE_TEST_PROTO(
      R"pb(
        ranking { ndcg {} }
      )pb");
  EXPECT_THAT(metrics[0].accessor, EqualsProto(expected));
}

TEST(Metric, MergeEvaluationClassification) {
  const proto::EvaluationResults src = PARSE_TEST_PROTO(
      R"pb(
        count_predictions: 1
        count_predictions_no_weight: 2
        sampled_predictions { example_key: "a" }
        count_sampled_predictions: 3
        training_duration_in_seconds: 4
        num_folds: 5
        classification {
          sum_log_loss: 6
          confusion { nrow: 1 ncol: 1 counts: 7 sum: 7 }
        }
      )pb");
  proto::EvaluationResults dst = PARSE_TEST_PROTO(
      R"pb(
        count_predictions: 10
        count_predictions_no_weight: 20
        sampled_predictions { example_key: "b" }
        count_sampled_predictions: 30
        training_duration_in_seconds: 40
        num_folds: 50
        classification {
          sum_log_loss: 60
          confusion { nrow: 1 ncol: 1 counts: 70 sum: 70 }
        }
      )pb");
  MergeEvaluation({}, src, &dst);
  proto::EvaluationResults expected_dst = PARSE_TEST_PROTO(
      R"pb(
        count_predictions: 11
        count_predictions_no_weight: 22
        sampled_predictions { example_key: "b" }
        sampled_predictions { example_key: "a" }
        count_sampled_predictions: 33
        training_duration_in_seconds: 44
        num_folds: 55
        classification {
          sum_log_loss: 66
          confusion { nrow: 1 ncol: 1 counts: 77 sum: 77 }
        }
      )pb");
  EXPECT_THAT(dst, EqualsProto(expected_dst));
}

TEST(Metric, HigherIsBetter) {
  {
    const proto::MetricAccessor accessor = PARSE_TEST_PROTO(
        R"pb(
          classification { accuracy {} }
        )pb");
    EXPECT_TRUE(HigherIsBetter(accessor).value());
  }

  {
    const proto::MetricAccessor accessor = PARSE_TEST_PROTO(
        R"pb(
          loss {}
        )pb");
    EXPECT_FALSE(HigherIsBetter(accessor).value());
  }
}

}  // namespace
}  // namespace metric
}  // namespace yggdrasil_decision_forests
