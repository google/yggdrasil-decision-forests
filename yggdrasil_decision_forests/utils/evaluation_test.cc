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

// clang-format off
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#ifdef YDF_EVAL_TFRECORD
#include "yggdrasil_decision_forests/utils/sharded_io_tfrecord.h"
#endif
// clang-format on

#include "yggdrasil_decision_forests/utils/evaluation.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/example_reader.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

using test::EqualsProto;
using ::testing::StrEq;

TEST(Evaluation, PredictionToExampleClassification) {
  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns {
      type: CATEGORICAL
      name: "label"
      categorical { is_already_integerized: true number_of_unique_values: 3 }
    }
  )pb");
  dataset::proto::DataSpecification expected_dataspec = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "1" }
    columns { type: NUMERICAL name: "2" }
  )pb");
  EXPECT_THAT(PredictionDataspec(model::proto::Task::CLASSIFICATION,
                                 dataspec.columns(0))
                  .value(),
              EqualsProto(expected_dataspec));

  model::proto::Prediction prediction = PARSE_TEST_PROTO(R"pb(
    classification {
      value: 1
      ground_truth: 2
      distribution { counts: 0 counts: 8 counts: 2 sum: 10 }
    }
  )pb");
  dataset::proto::Example prediction_as_example;
  EXPECT_OK(PredictionToExample(model::proto::Task::CLASSIFICATION,
                                dataspec.columns(0), prediction,
                                &prediction_as_example));
  dataset::proto::Example expected_prediction_as_example = PARSE_TEST_PROTO(
      R"pb(
        attributes { numerical: 0.8 }
        attributes { numerical: 0.2 }
      )pb");
  EXPECT_THAT(prediction_as_example,
              EqualsProto(expected_prediction_as_example));
}

TEST(Evaluation, PredictionToExampleRegression) {
  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "label" }
  )pb");
  dataset::proto::DataSpecification expected_dataspec = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "label" }
  )pb");
  EXPECT_THAT(
      PredictionDataspec(model::proto::Task::REGRESSION, dataspec.columns(0))
          .value(),
      EqualsProto(expected_dataspec));

  model::proto::Prediction prediction = PARSE_TEST_PROTO(R"pb(
    regression { value: 5 }
  )pb");
  dataset::proto::Example prediction_as_example;
  EXPECT_OK(PredictionToExample(model::proto::Task::REGRESSION,
                                dataspec.columns(0), prediction,
                                &prediction_as_example));
  dataset::proto::Example expected_prediction_as_example = PARSE_TEST_PROTO(
      R"pb(
        attributes { numerical: 5 }
      )pb");
  EXPECT_THAT(prediction_as_example,
              EqualsProto(expected_prediction_as_example));
}

TEST(Evaluation, ExportRegressionPredictionsToDatasetRaw) {
  std::vector<model::proto::Prediction> predictions;
  predictions.push_back(PARSE_TEST_PROTO("regression { value: 1 }"));
  predictions.push_back(PARSE_TEST_PROTO("regression { value: 2 }"));
  predictions.push_back(PARSE_TEST_PROTO("regression { value: 3 }"));

  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "label" }
  )pb");

  const auto prediction_path =
      file::JoinPath(test::TmpDirectory(), "predictions.csv");
  EXPECT_OK(ExportPredictions(predictions, model::proto::Task::REGRESSION,
                              dataspec.columns(0),
                              absl::StrCat("csv:", prediction_path),
                              /*num_records_by_shard_in_output=*/-1,
                              /*prediction_key=*/{}, PredictionFormat::kRaw));

  std::string csv_content = file::GetContent(prediction_path).value();
  EXPECT_EQ(csv_content, "label\n1\n2\n3\n");
}

TEST(Evaluation, ExportPredictionsWithKeyToDataset) {
  std::vector<model::proto::Prediction> predictions;
  predictions.push_back(
      PARSE_TEST_PROTO("regression { value: 1 } example_key: \"k1\" "));
  predictions.push_back(
      PARSE_TEST_PROTO("regression { value: 2 } example_key: \"k2\" "));
  predictions.push_back(
      PARSE_TEST_PROTO("regression { value: 3 } example_key: \"k3\" "));

  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "label" }
  )pb");

  const auto prediction_path =
      file::JoinPath(test::TmpDirectory(), "predictions.csv");
  EXPECT_OK(ExportPredictions(predictions, model::proto::Task::REGRESSION,
                              dataspec.columns(0),
                              absl::StrCat("csv:", prediction_path),
                              /*num_records_by_shard_in_output=*/-1,
                              /*prediction_key=*/"key"));

  std::string csv_content = file::GetContent(prediction_path).value();
  EXPECT_EQ(csv_content, "label,key\n1,k1\n2,k2\n3,k3\n");
}

TEST(Evaluation, ExportClassificationPredictionsToDatasetRaw) {
  std::vector<model::proto::Prediction> predictions;
  predictions.push_back(PARSE_TEST_PROTO(R"pb(
    classification {
      value: 1
      distribution { counts: 0 counts: 8 counts: 2 sum: 10 }
    }
  )pb"));
  predictions.push_back(PARSE_TEST_PROTO(R"pb(
    classification {
      value: 2
      distribution { counts: 0 counts: 0 counts: 5 sum: 5 }
    }
  )pb"));

  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns {
      type: CATEGORICAL
      name: "label"
      categorical {
        number_of_unique_values: 3
        items {
          key: "OOD"
          value { index: 0 }
        }
        items {
          key: "X"
          value { index: 1 }
        }
        items {
          key: "Y"
          value { index: 2 }
        }
      }
    }
  )pb");

  const auto prediction_path =
      file::JoinPath(test::TmpDirectory(), "predictions.csv");
  EXPECT_OK(ExportPredictions(predictions, model::proto::Task::CLASSIFICATION,
                              dataspec.columns(0),
                              absl::StrCat("csv:", prediction_path),
                              /*num_records_by_shard_in_output=*/-1,
                              /*prediction_key=*/{}, PredictionFormat::kRaw));

  std::string csv_content = file::GetContent(prediction_path).value();
  EXPECT_THAT(csv_content, StrEq("X,Y\n0.8,0.2\n0,1\n"));
}

TEST(Evaluation, ExportClassificationPredictionsToDatasetSimple) {
  std::vector<model::proto::Prediction> predictions;
  predictions.push_back(PARSE_TEST_PROTO(R"pb(
    classification {
      value: 1
      distribution { counts: 0 counts: 8 counts: 2 sum: 10 }
    }
  )pb"));
  predictions.push_back(PARSE_TEST_PROTO(R"pb(
    classification {
      value: 2
      distribution { counts: 0 counts: 0 counts: 5 sum: 5 }
    }
  )pb"));

  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns {
      type: CATEGORICAL
      name: "label"
      categorical {
        number_of_unique_values: 3
        items {
          key: "OOD"
          value { index: 0 }
        }
        items {
          key: "X"
          value { index: 1 }
        }
        items {
          key: "Y"
          value { index: 2 }
        }
      }
    }
  )pb");

  const auto prediction_path =
      file::JoinPath(test::TmpDirectory(), "predictions.csv");
  EXPECT_OK(ExportPredictions(
      predictions, model::proto::Task::CLASSIFICATION, dataspec.columns(0),
      absl::StrCat("csv:", prediction_path),
      /*num_records_by_shard_in_output=*/-1, /*prediction_key=*/{},
      PredictionFormat::kSimple));

  std::string csv_content = file::GetContent(prediction_path).value();
  EXPECT_THAT(csv_content, StrEq("label\nX\nY\n"));
}

TEST(Evaluation, ExportClassificationPredictionsToDatasetRich) {
  std::vector<model::proto::Prediction> predictions;
  predictions.push_back(PARSE_TEST_PROTO(R"pb(
    classification {
      value: 1
      distribution { counts: 0 counts: 8 counts: 2 sum: 10 }
    }
  )pb"));
  predictions.push_back(PARSE_TEST_PROTO(R"pb(
    classification {
      value: 2
      distribution { counts: 0 counts: 0 counts: 5 sum: 5 }
    }
  )pb"));

  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns {
      type: CATEGORICAL
      name: "label"
      categorical {
        number_of_unique_values: 3
        items {
          key: "OOD"
          value { index: 0 }
        }
        items {
          key: "X"
          value { index: 1 }
        }
        items {
          key: "Y"
          value { index: 2 }
        }
      }
    }
  )pb");

  const auto prediction_path =
      file::JoinPath(test::TmpDirectory(), "predictions.csv");
  EXPECT_OK(ExportPredictions(predictions, model::proto::Task::CLASSIFICATION,
                              dataspec.columns(0),
                              absl::StrCat("csv:", prediction_path),
                              /*num_records_by_shard_in_output=*/-1,
                              /*prediction_key=*/{}, PredictionFormat::kRich));

  std::string csv_content = file::GetContent(prediction_path).value();
  EXPECT_THAT(csv_content, StrEq("label,Conf.label\nX,0.8\nY,1\n"));
}

TEST(Evaluation, ExportClassificationPredictionsToDatasetFull) {
  std::vector<model::proto::Prediction> predictions;
  predictions.push_back(PARSE_TEST_PROTO(R"pb(
    classification {
      value: 1
      distribution { counts: 0 counts: 8 counts: 2 sum: 10 }
    }
  )pb"));
  predictions.push_back(PARSE_TEST_PROTO(R"pb(
    classification {
      value: 2
      distribution { counts: 0 counts: 0 counts: 5 sum: 5 }
    }
  )pb"));

  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns {
      type: CATEGORICAL
      name: "label"
      categorical {
        number_of_unique_values: 3
        items {
          key: "OOD"
          value { index: 0 }
        }
        items {
          key: "X"
          value { index: 1 }
        }
        items {
          key: "Y"
          value { index: 2 }
        }
      }
    }
  )pb");

  const auto prediction_path =
      file::JoinPath(test::TmpDirectory(), "predictions.csv");
  EXPECT_OK(ExportPredictions(predictions, model::proto::Task::CLASSIFICATION,
                              dataspec.columns(0),
                              absl::StrCat("csv:", prediction_path),
                              /*num_records_by_shard_in_output=*/-1,
                              /*prediction_key=*/{}, PredictionFormat::kFull));

  std::string csv_content = file::GetContent(prediction_path).value();
  EXPECT_THAT(csv_content, StrEq("label,X,Y\nX,0.8,0.2\nY,0,1\n"));
}

#ifdef YDF_EVAL_TFRECORD
TEST(Evaluation, ExportPredictionsToTFRecord) {
  std::vector<model::proto::Prediction> predictions;
  predictions.push_back(PARSE_TEST_PROTO("regression { value: 1 }"));
  predictions.push_back(PARSE_TEST_PROTO("regression { value: 2 }"));
  predictions.push_back(PARSE_TEST_PROTO("regression { value: 3 }"));

  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "label" }
  )pb");

  const auto path =
      file::JoinPath(test::TmpDirectory(), "predictions.tfrecord-pred");
  const auto typed_path = absl::StrCat("tfrecord+pred:", path);
  EXPECT_OK(ExportPredictions(predictions, model::proto::Task::REGRESSION,
                              dataspec.columns(0), typed_path, -1));

  auto reader =
      absl::make_unique<TFRecordShardedReader<model::proto::Prediction>>();
  EXPECT_OK(reader->Open(path));

  model::proto::Prediction prediction;
  EXPECT_TRUE(reader->Next(&prediction).value());
  model::proto::Prediction tmp = PARSE_TEST_PROTO("regression { value: 1 }");
  EXPECT_THAT(prediction, EqualsProto(tmp));
  EXPECT_TRUE(reader->Next(&prediction).value());
  tmp = PARSE_TEST_PROTO("regression { value: 2 }");
  EXPECT_THAT(prediction, EqualsProto(tmp));
  EXPECT_TRUE(reader->Next(&prediction).value());
  tmp = PARSE_TEST_PROTO("regression { value: 3 }");
  EXPECT_THAT(prediction, EqualsProto(tmp));
  EXPECT_FALSE(reader->Next(&prediction).value());
}
#endif

TEST(Evaluation, ParsePredictionFormat) {
  EXPECT_THAT(ParsePredictionFormat("non existing value").status(),
              test::StatusIs(absl::StatusCode::kInvalidArgument));

  for (const auto& [str, value] :
       std::vector<std::pair<std::string, PredictionFormat>>{
           {"kRaw", PredictionFormat::kRaw},
           {"kSimple", PredictionFormat::kSimple},
           {"kRich", PredictionFormat::kRich},
           {"kFull", PredictionFormat::kFull},
       }) {
    EXPECT_EQ(ParsePredictionFormat(str).value(), value);
  }
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
