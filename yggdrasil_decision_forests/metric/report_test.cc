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

#include "yggdrasil_decision_forests/metric/report.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace metric {
namespace {

TEST(Report, HtmlReportClassification) {
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
  CHECK_OK(InitializeEvaluation(option, label_column, &eval));
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
  CHECK_OK(AddPrediction(option, pred, &rnd, &eval));

  pred.mutable_classification()->set_value(1);
  pred_proba->set_counts(0, 0.2f);
  pred_proba->set_counts(1, 0.6f);
  pred_proba->set_counts(2, 0.2f);
  pred.mutable_classification()->set_ground_truth(2);
  CHECK_OK(AddPrediction(option, pred, &rnd, &eval));

  pred.mutable_classification()->set_value(2);
  pred_proba->set_counts(0, 0.0f);
  pred_proba->set_counts(1, 0.1f);
  pred_proba->set_counts(2, 0.9f);
  pred.mutable_classification()->set_ground_truth(2);
  CHECK_OK(AddPrediction(option, pred, &rnd, &eval));

  pred.mutable_classification()->set_value(0);
  pred_proba->set_counts(0, 0.5f);
  pred_proba->set_counts(1, 0.2f);
  pred_proba->set_counts(2, 0.3f);
  pred.mutable_classification()->set_ground_truth(1);
  CHECK_OK(AddPrediction(option, pred, &rnd, &eval));

  // Finalize.
  CHECK_OK(FinalizeEvaluation(option, label_column, &eval));

  std::string html_report;
  CHECK_OK(AppendHtmlReport(eval, &html_report));

  const auto path =
      file::JoinPath(test::TmpDirectory(), "report_classification.html");
  YDF_LOG(INFO) << "path: " << path;
  CHECK_OK(file::SetContent(path, html_report));
}

TEST(Report, HtmlReportRegression) {
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
  CHECK_OK(InitializeEvaluation(option, label_column, &eval));
  model::proto::Prediction pred;
  auto* pred_reg = pred.mutable_regression();

  // Add some predictions.
  pred_reg->set_value(1);
  pred_reg->set_ground_truth(1);
  CHECK_OK(AddPrediction(option, pred, &rnd, &eval));

  pred_reg->set_value(1.1);
  pred_reg->set_ground_truth(2);
  CHECK_OK(AddPrediction(option, pred, &rnd, &eval));

  pred_reg->set_value(0.9);
  pred_reg->set_ground_truth(0);
  CHECK_OK(AddPrediction(option, pred, &rnd, &eval));

  pred_reg->set_value(2);
  pred_reg->set_ground_truth(1.1);
  CHECK_OK(AddPrediction(option, pred, &rnd, &eval));

  // Finalize.
  CHECK_OK(FinalizeEvaluation(option, label_column, &eval));

  std::string html_report;
  CHECK_OK(AppendHtmlReport(eval, &html_report));

  const auto path =
      file::JoinPath(test::TmpDirectory(), "report_regression.html");
  YDF_LOG(INFO) << "path: " << path;
  CHECK_OK(file::SetContent(path, html_report));
}

}  // namespace
}  // namespace metric
}  // namespace yggdrasil_decision_forests
