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

#include "yggdrasil_decision_forests/model/abstract_model.h"

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/evaluate_on_disk.h"
#include "yggdrasil_decision_forests/model/fast_engine_factory.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/model_testing.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace {

using test::EqualsProto;
using test::StatusIs;

std::string TestDataDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/test_data");
}

class FakeModelWithEngine : public FakeModel {
 public:
  FakeModelWithEngine() : FakeModel() {}
};

class Engine1 : public serving::FastEngine {
 public:
  std::unique_ptr<serving::AbstractExampleSet> AllocateExamples(
      int num_examples) const override {
    LOG(FATAL) << "Not implemented";
    return {};
  }

  void Predict(const serving::AbstractExampleSet& examples, int num_examples,
               std::vector<float>* predictions) const override {
    LOG(FATAL) << "Not implemented";
  }

  int NumPredictionDimension() const override {
    LOG(FATAL) << "Not implemented";
    return 1;
  }

  const serving::FeaturesDefinition& features() const override {
    LOG(FATAL) << "Not implemented";
    return features_;
  }

 private:
  serving::FeaturesDefinition features_;
};

class EngineFactory1 : public model::FastEngineFactory {
 public:
  std::string name() const override { return "engine1"; }

  bool IsCompatible(const AbstractModel* const model) const override {
    auto* casted_model = dynamic_cast<const FakeModelWithEngine*>(model);
    if (casted_model == nullptr) {
      return false;
    }
    return true;
  }

  std::vector<std::string> IsBetterThan() const override { return {}; }

  absl::StatusOr<std::unique_ptr<serving::FastEngine>> CreateEngine(
      const AbstractModel* const model) const override {
    auto* casted_model = dynamic_cast<const FakeModelWithEngine*>(model);
    if (!casted_model) {
      return absl::InvalidArgumentError(
          "The model is not a FakeModelWithEngine.");
    }
    return std::make_unique<Engine1>();
  }
};

REGISTER_FastEngineFactory(EngineFactory1, "engine1");

class Engine2 : public serving::FastEngine {
 public:
  std::unique_ptr<serving::AbstractExampleSet> AllocateExamples(
      int num_examples) const override {
    LOG(FATAL) << "Not implemented";
    return {};
  }

  void Predict(const serving::AbstractExampleSet& examples, int num_examples,
               std::vector<float>* predictions) const override {
    LOG(FATAL) << "Not implemented";
  }

  int NumPredictionDimension() const override {
    LOG(FATAL) << "Not implemented";
    return 1;
  }

  const serving::FeaturesDefinition& features() const override {
    LOG(FATAL) << "Not implemented";
    return features_;
  }

 private:
  serving::FeaturesDefinition features_;
};

class EngineFactory2 : public model::FastEngineFactory {
 public:
  std::string name() const override { return "engine2"; }

  bool IsCompatible(const AbstractModel* const model) const override {
    auto* casted_model = dynamic_cast<const FakeModelWithEngine*>(model);
    if (casted_model == nullptr) {
      return false;
    }
    return true;
  }

  std::vector<std::string> IsBetterThan() const override { return {"engine1"}; }

  absl::StatusOr<std::unique_ptr<serving::FastEngine>> CreateEngine(
      const AbstractModel* const model) const override {
    auto* casted_model = dynamic_cast<const FakeModelWithEngine*>(model);
    if (!casted_model) {
      return absl::InvalidArgumentError(
          "The model is not a FakeModelWithEngine.");
    }
    return std::make_unique<Engine2>();
  }
};

REGISTER_FastEngineFactory(EngineFactory2, "engine2");

class FakeModelWithoutEngine : public FakeModel {
 public:
  FakeModelWithoutEngine() : FakeModel() {}
};

TEST(AbstractLearner, MergeVariableImportance) {
  std::vector<proto::VariableImportance> a;
  a.push_back(PARSE_TEST_PROTO("attribute_idx:0 importance:2"));
  a.push_back(PARSE_TEST_PROTO("attribute_idx:1 importance:1"));

  std::vector<proto::VariableImportance> b;
  b.push_back(PARSE_TEST_PROTO("attribute_idx:1 importance:4"));
  b.push_back(PARSE_TEST_PROTO("attribute_idx:2 importance:3"));

  MergeVariableImportance(a, 0.1, &b);

  EXPECT_EQ(b.size(), 3);

  EXPECT_EQ(b[0].attribute_idx(), 1);
  EXPECT_EQ(b[0].importance(), 4 * 0.9 + 1 * 0.1);

  EXPECT_EQ(b[1].attribute_idx(), 2);
  EXPECT_EQ(b[1].importance(), 3 * 0.9 + 0 * 0.1);

  EXPECT_EQ(b[2].attribute_idx(), 0);
  EXPECT_EQ(b[2].importance(), 0 * 0.9 + 2 * 0.1);
}

TEST(AbstractLearner, MergeAddPredictionsRegression) {
  proto::Prediction src = PARSE_TEST_PROTO(R"pb(regression { value: 1 })pb");
  proto::Prediction dst;
  PredictionMerger merger(&dst);

  merger.Add(src, 0.25f);
  EXPECT_THAT(dst, EqualsProto(utils::ParseTextProto<proto::Prediction>(
                                   "regression {value:0.25 }")
                                   .value()));

  merger.Add(src, 0.25f);
  EXPECT_THAT(dst, EqualsProto(utils::ParseTextProto<proto::Prediction>(
                                   "regression { value: 0.5 }")
                                   .value()));

  merger.Add(src, 0.50f);
  EXPECT_THAT(dst, EqualsProto(utils::ParseTextProto<proto::Prediction>(
                                   "regression { value: 1.0 }")
                                   .value()));
}

TEST(AbstractLearner, MergeAddPredictionsClassification) {
  proto::Prediction src = PARSE_TEST_PROTO(
      R"pb(classification { distribution { counts: 1 counts: 3 sum: 4 } })pb");
  proto::Prediction dst;
  PredictionMerger merger(&dst);

  merger.Add(src, 0.25f);
  EXPECT_THAT(dst, EqualsProto(utils::ParseTextProto<proto::Prediction>(R"(
                classification {
                  distribution { counts: 0.0625 counts: 0.1875 sum: 0.25 }
                })")
                                   .value()));

  merger.Add(src, 0.25f);
  EXPECT_THAT(
      dst,
      EqualsProto(utils::ParseTextProto<proto::Prediction>(R"(classification {
                       distribution { counts: 0.125 counts: 0.375 sum: 0.5 }
                     })")
                      .value()));

  merger.Add(src, 0.50f);
  EXPECT_THAT(
      dst,
      EqualsProto(utils::ParseTextProto<proto::Prediction>(R"(classification {
                               distribution { counts: 0.25 counts: 0.75 sum: 1 }
                             })")
                      .value()));

  merger.Merge();
  EXPECT_THAT(
      dst,
      EqualsProto(utils::ParseTextProto<proto::Prediction>(R"(classification {
                               distribution { counts: 0.25 counts: 0.75 sum: 1 }
                               value: 1
                             })")
                      .value()));
}

TEST(AbstractLearner, MergeAddPredictionsAnomalyDetection) {
  proto::Prediction src =
      PARSE_TEST_PROTO(R"pb(anomaly_detection { value: 1 })pb");
  proto::Prediction dst;
  PredictionMerger merger(&dst);

  merger.Add(src, 0.25f);
  EXPECT_THAT(dst, EqualsProto(utils::ParseTextProto<proto::Prediction>(
                                   "anomaly_detection {value:0.25 }")
                                   .value()));

  merger.Add(src, 0.25f);
  EXPECT_THAT(dst, EqualsProto(utils::ParseTextProto<proto::Prediction>(
                                   "anomaly_detection { value: 0.5 }")
                                   .value()));

  merger.Add(src, 0.50f);
  EXPECT_THAT(dst, EqualsProto(utils::ParseTextProto<proto::Prediction>(
                                   "anomaly_detection { value: 1.0 }")
                                   .value()));
}

TEST(AbstractModel, BuildFastEngine) {
  FakeModelWithoutEngine model_without_engine;
  EXPECT_THAT(model_without_engine.BuildFastEngine().status(),
              StatusIs(absl::StatusCode::kNotFound));

  FakeModelWithEngine model_with_engine;
  const auto engine_or_status = model_with_engine.BuildFastEngine();
  EXPECT_OK(engine_or_status.status());
  const serving::FastEngine* engine = engine_or_status.value().get();
  EXPECT_TRUE(dynamic_cast<const Engine2*>(engine) != nullptr);
}

TEST(FloatToProtoPrediction, Base) {
  proto::Prediction prediction;

  FloatToProtoPrediction({0, 0.5, 1}, /*example_idx=*/0,
                         proto::Task::CLASSIFICATION,
                         /*num_prediction_dimensions=*/1, &prediction);
  EXPECT_THAT(
      prediction,
      EqualsProto(utils::ParseTextProto<proto::Prediction>(R"pb(
                    classification {
                      value: 1
                      distribution { counts: 0 counts: 1 counts: 0 sum: 1 }
                    }
                  )pb")
                      .value()));

  FloatToProtoPrediction({0, 0.5, 1}, /*example_idx=*/1,
                         proto::Task::CLASSIFICATION,
                         /*num_prediction_dimensions=*/1, &prediction);
  EXPECT_THAT(prediction,
              EqualsProto(utils::ParseTextProto<proto::Prediction>(R"(
                classification {
                  value: 1
                  distribution { counts: 0 counts: 0.5 counts: 0.5 sum: 1 }
                }
              )")
                              .value()));

  FloatToProtoPrediction({0, 0.5, 1}, /*example_idx=*/2,
                         proto::Task::CLASSIFICATION,
                         /*num_prediction_dimensions=*/1, &prediction);
  EXPECT_THAT(prediction,
              EqualsProto(utils::ParseTextProto<proto::Prediction>(R"(
                classification {
                  value: 2
                  distribution { counts: 0 counts: 0 counts: 1 sum: 1 }
                }
              )")
                              .value()));

  FloatToProtoPrediction({0.2, 0.8, 0.4, 0.6}, /*example_idx=*/0,
                         proto::Task::CLASSIFICATION,
                         /*num_prediction_dimensions=*/2, &prediction);
  EXPECT_THAT(prediction,
              EqualsProto(utils::ParseTextProto<proto::Prediction>(R"(
                classification {
                  value: 2
                  distribution { counts: 0 counts: 0.2 counts: 0.8 sum: 1 }
                }
              )")
                              .value()));

  FloatToProtoPrediction({0.2, 0.8, 0.4, 0.6}, /*example_idx=*/1,
                         proto::Task::CLASSIFICATION,
                         /*num_prediction_dimensions=*/2, &prediction);
  EXPECT_THAT(prediction,
              EqualsProto(utils::ParseTextProto<proto::Prediction>(R"(
                classification {
                  value: 2
                  distribution { counts: 0 counts: 0.4 counts: 0.6 sum: 1 }
                }
              )")
                              .value()));

  FloatToProtoPrediction({0.2, 0.4}, /*example_idx=*/0, proto::Task::REGRESSION,
                         /*num_prediction_dimensions=*/1, &prediction);
  EXPECT_THAT(prediction, EqualsProto(utils::ParseTextProto<proto::Prediction>(
                                          R"(regression { value: 0.2 })")
                                          .value()));

  FloatToProtoPrediction({0.2, 0.4}, /*example_idx=*/0, proto::Task::RANKING,
                         /*num_prediction_dimensions=*/1, &prediction);
  EXPECT_THAT(prediction, EqualsProto(utils::ParseTextProto<proto::Prediction>(
                                          R"(ranking { relevance: 0.2 })")
                                          .value()));

  FloatToProtoPrediction({0.2, 0.4, 0.5, 0.6}, /*example_idx=*/1,
                         proto::Task::CATEGORICAL_UPLIFT,
                         /*num_prediction_dimensions=*/1, &prediction);
  EXPECT_THAT(prediction, EqualsProto(utils::ParseTextProto<proto::Prediction>(
                                          R"(uplift { treatment_effect: 0.4 })")
                                          .value()));

  FloatToProtoPrediction({0.2, 0.4}, /*example_idx=*/0,
                         proto::Task::ANOMALY_DETECTION,
                         /*num_prediction_dimensions=*/1, &prediction);
  EXPECT_THAT(prediction, EqualsProto(utils::ParseTextProto<proto::Prediction>(
                                          R"(anomaly_detection { value: 0.2 })")
                                          .value()));
}

TEST(ProtoToFloatPrediction, BinaryClassification) {
  std::vector<float> dst_prediction(1);
  proto::Prediction prediction;

  // example_idx = 0
  ASSERT_OK_AND_ASSIGN(
      prediction,
      utils::ParseTextProto<proto::Prediction>(
          R"pb(classification {
                 value: 1
                 distribution { counts: 0 counts: 1 counts: 0 sum: 1 }
               })pb"));
  ProtoToFloatPrediction(prediction, proto::Task::CLASSIFICATION,
                         absl::Span<float>(dst_prediction));
  EXPECT_FLOAT_EQ(dst_prediction[0], 0.0f);

  // example_idx = 1
  ASSERT_OK_AND_ASSIGN(
      prediction,
      utils::ParseTextProto<proto::Prediction>(
          R"pb(classification {
                 value: 1
                 distribution { counts: 0 counts: 0.5 counts: 0.5 sum: 1 }
               })pb"));
  ProtoToFloatPrediction(prediction, proto::Task::CLASSIFICATION,
                         absl::Span<float>(dst_prediction));
  EXPECT_FLOAT_EQ(dst_prediction[0], 0.5f);

  // example_idx = 2
  ASSERT_OK_AND_ASSIGN(
      prediction,
      utils::ParseTextProto<proto::Prediction>(
          R"pb(classification {
                 value: 2
                 distribution { counts: 0 counts: 0 counts: 1 sum: 1 }
               })pb"));
  ProtoToFloatPrediction(prediction, proto::Task::CLASSIFICATION,
                         absl::Span<float>(dst_prediction));
  EXPECT_FLOAT_EQ(dst_prediction[0], 1.0f);
}

TEST(ProtoToFloatPrediction, MulticlassClassification) {
  std::vector<float> dst_prediction(2);

  // example_idx = 0
  proto::Prediction prediction;

  // example_idx = 0
  ASSERT_OK_AND_ASSIGN(
      prediction,
      utils::ParseTextProto<proto::Prediction>(
          R"pb(classification {
                 value: 2
                 distribution { counts: 0 counts: 0.2 counts: 0.8 sum: 1 }
               })pb"));
  ProtoToFloatPrediction(prediction, proto::Task::CLASSIFICATION,
                         absl::Span<float>(dst_prediction));
  EXPECT_FLOAT_EQ(dst_prediction[0], 0.2f);
  EXPECT_FLOAT_EQ(dst_prediction[1], 0.8f);

  // example_idx = 1
  ASSERT_OK_AND_ASSIGN(
      prediction,
      utils::ParseTextProto<proto::Prediction>(
          R"pb(classification {
                 value: 2
                 distribution { counts: 0 counts: 0.4 counts: 0.6 sum: 1 }
               })pb"));
  ProtoToFloatPrediction(prediction, proto::Task::CLASSIFICATION,
                         absl::Span<float>(dst_prediction));
  EXPECT_FLOAT_EQ(dst_prediction[0], 0.4f);
  EXPECT_FLOAT_EQ(dst_prediction[1], 0.6f);
}

TEST(ProtoToFloatPrediction, Regression) {
  std::vector<float> dst_prediction(1);
  proto::Prediction prediction;
  ASSERT_OK_AND_ASSIGN(prediction, utils::ParseTextProto<proto::Prediction>(
                                       R"pb(regression { value: 0.2 })pb"));
  ProtoToFloatPrediction(prediction, proto::Task::REGRESSION,
                         absl::Span<float>(dst_prediction));
  EXPECT_FLOAT_EQ(dst_prediction[0], 0.2f);
}

TEST(ProtoToFloatPrediction, Ranking) {
  std::vector<float> dst_prediction(1);
  proto::Prediction prediction;
  ASSERT_OK_AND_ASSIGN(prediction, utils::ParseTextProto<proto::Prediction>(
                                       R"pb(ranking { relevance: 0.2 })pb"));
  ProtoToFloatPrediction(prediction, proto::Task::RANKING,
                         absl::Span<float>(dst_prediction));
  EXPECT_FLOAT_EQ(dst_prediction[0], 0.2f);
}

TEST(ProtoToFloatPrediction, Uplift) {
  std::vector<float> dst_prediction(1);
  proto::Prediction prediction;
  ASSERT_OK_AND_ASSIGN(prediction,
                       utils::ParseTextProto<proto::Prediction>(
                           R"pb(uplift { treatment_effect: 0.4 })pb"));
  ProtoToFloatPrediction(prediction, proto::Task::CATEGORICAL_UPLIFT,
                         absl::Span<float>(dst_prediction));
  EXPECT_FLOAT_EQ(dst_prediction[0], 0.4f);
}

TEST(ProtoToFloatPrediction, AnomalyDetection) {
  std::vector<float> dst_prediction(1);
  proto::Prediction prediction;
  ASSERT_OK_AND_ASSIGN(prediction,
                       utils::ParseTextProto<proto::Prediction>(
                           R"pb(anomaly_detection { value: 0.2 })pb"));
  ProtoToFloatPrediction(prediction, proto::Task::ANOMALY_DETECTION,
                         absl::Span<float>(dst_prediction));
  EXPECT_FLOAT_EQ(dst_prediction[0], 0.2f);
}

TEST(Evaluate, FromVerticalDataset) {
  std::unique_ptr<model::AbstractModel> model;
  EXPECT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_gbdt"),
      &model));

  dataset::VerticalDataset dataset;
  EXPECT_OK(LoadVerticalDataset(
      absl::StrCat("csv:",
                   file::JoinPath(TestDataDir(), "dataset", "adult_test.csv")),
      model->data_spec(), &dataset));

  utils::RandomEngine rnd;
  const auto evaluation = model->Evaluate(dataset, {}, &rnd);
  EXPECT_NEAR(metric::Accuracy(evaluation), 0.8723513, 0.000001);
}

TEST(Evaluate, FromDisk) {
  std::unique_ptr<model::AbstractModel> model;
  EXPECT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_gbdt"),
      &model));

  utils::RandomEngine rnd;
  const auto evaluation =
      EvaluateOnDisk(
          *model,
          absl::StrCat("csv:", file::JoinPath(TestDataDir(), "dataset",
                                              "adult_test.csv")),
          {}, &rnd)
          .value();
  EXPECT_NEAR(metric::Accuracy(evaluation), 0.8723513, 0.000001);
}

TEST(Evaluate, FailsWithNoRowsFails) {
  std::unique_ptr<model::AbstractModel> model;
  EXPECT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_gbdt"),
      &model));

  dataset::VerticalDataset dataset;

  utils::RandomEngine rnd;
  const auto evaluation = model->EvaluateWithStatus(dataset, {}, &rnd);
  EXPECT_FALSE(evaluation.ok());
}

TEST(Model, AbstractAttributesSizeInBytes) {
  FakeModelWithEngine model;
  // The model size is compiler+arch dependent.
  EXPECT_GT(model.AbstractAttributesSizeInBytes(), 0);
}

TEST(AppendPredictions, FromVerticalDataset) {
  std::unique_ptr<model::AbstractModel> model;
  EXPECT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_gbdt"),
      &model));

  dataset::VerticalDataset dataset;
  EXPECT_OK(LoadVerticalDataset(
      absl::StrCat("csv:",
                   file::JoinPath(TestDataDir(), "dataset", "adult_test.csv")),
      model->data_spec(), &dataset));

  utils::RandomEngine rnd;
  std::vector<model::proto::Prediction> predictions;
  EXPECT_OK(model->AppendPredictions(dataset, /*add_ground_truth=*/true,
                                     &predictions));
  EXPECT_EQ(predictions.size(), 9769);

  EXPECT_THAT(predictions.front(),
              EqualsProto(utils::ParseTextProto<proto::Prediction>(
                              R"pb(
                                classification {
                                  value: 1
                                  distribution {
                                    counts: 0
                                    counts: 0.9878693
                                    counts: 0.012130676
                                    sum: 1
                                  }
                                  ground_truth: 1
                                })pb")
                              .value()));
}

TEST(CompareModel, Base) {
  std::unique_ptr<model::AbstractModel> model1;
  std::unique_ptr<model::AbstractModel> model2;

  EXPECT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_gbdt"),
      &model1));
  EXPECT_OK(model::LoadModel(
      file::JoinPath(TestDataDir(), "model", "adult_binary_class_rf"),
      &model2));
  EXPECT_EQ(
      model1->DebugCompare(*model2),
      "name_ != other.name_ i.e. GRADIENT_BOOSTED_TREES != RANDOM_FOREST");
}

}  // namespace
}  // namespace model
}  // namespace yggdrasil_decision_forests
