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


#include "yggdrasil_decision_forests/model/isolation_forest/isolation_forest.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/decision_tree/builder.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::model::isolation_forest {
namespace {

const float kEpsilon = 0.00001f;

using ::testing::ElementsAre;
using ::yggdrasil_decision_forests::test::EqualsProto;

std::string TestDataDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/test_data");
}

std::unique_ptr<IsolationForestModel> CreateToyModel() {
  auto model = std::make_unique<IsolationForestModel>();
  model->set_task(model::proto::Task::ANOMALY_DETECTION);
  model->set_num_examples_per_trees(10);

  dataset::AddNumericalColumn("a", model->mutable_data_spec());
  dataset::AddNumericalColumn("b", model->mutable_data_spec());
  dataset::AddNumericalColumn("c", model->mutable_data_spec());

  model->mutable_input_features()->push_back(0);
  model->mutable_input_features()->push_back(1);
  model->mutable_input_features()->push_back(2);

  // Tree 1
  {
    auto tree = std::make_unique<decision_tree::DecisionTree>();
    decision_tree::TreeBuilder root(tree.get());
    auto [nl1, l1] = root.ConditionIsGreater(0, 0);
    auto [l2, nl2] = nl1.ConditionIsGreater(1, 1);
    auto [l3, l4] = nl2.ConditionIsGreater(0, 1);
    l1.LeafAnomalyDetection(2);
    l2.LeafAnomalyDetection(4);
    l3.LeafAnomalyDetection(2);
    l4.LeafAnomalyDetection(2);
    tree->SetLeafIndices();
    model->mutable_decision_trees()->push_back(std::move(tree));
  }

  // Tree 2
  {
    auto tree = std::make_unique<decision_tree::DecisionTree>();
    decision_tree::TreeBuilder root(tree.get());
    auto [l1, l2] = root.ConditionIsGreater(1, 1);
    l1.LeafAnomalyDetection(5);
    l2.LeafAnomalyDetection(5);
    tree->SetLeafIndices();
    model->mutable_decision_trees()->push_back(std::move(tree));
  }

  return model;
}

dataset::VerticalDataset CreateToyDataset(
    const dataset::proto::DataSpecification& dataspec) {
  dataset::VerticalDataset dataset;
  dataset.set_data_spec(dataspec);
  EXPECT_OK(dataset.CreateColumnsFromDataspec());
  dataset.AppendExample({{"a", "-1"}, {"b", "0"}, {"c", "0"}});
  dataset.AppendExample({{"a", "0.5"}, {"b", "0"}, {"c", "0"}});
  dataset.AppendExample({{"a", "1.5"}, {"b", "0"}, {"c", "0"}});
  dataset.AppendExample({{"a", "1.5"}, {"b", "2"}, {"c", "1"}});
  return dataset;
}

TEST(IsolationForest, PreissAveragePathLength) {
  EXPECT_EQ(PreissAveragePathLength(1), 0.f);
  EXPECT_EQ(PreissAveragePathLength(2), 1.f);
  /*
  H = lambda x: math.log(x) + 0.5772156649
  2 * H(3 - 1) - 2 * (3 - 1) / 3
  >> 1.207392357586557
  */
  EXPECT_NEAR(PreissAveragePathLength(3), 1.20739233f, kEpsilon);
}

TEST(IsolationForest, IsolationForestPrediction) {
  EXPECT_NEAR(IsolationForestPrediction(0, 4), 1.f, kEpsilon);

  /*
  2**( - 1 / (2 * H(4 - 1) - 2 * (4 - 1) / 4))
  >> 0.6877436677784063
  2**( - 2 / (2 * H(4 - 1) - 2 * (4 - 1) / 4))
  >> 0.472991352569295
  2**( - 2 / (2 * H(8 - 1) - 2 * (8 - 1) / 8))
  >> 0.6566744390877336
  */
  EXPECT_NEAR(IsolationForestPrediction(1, 4), 0.687743664f, kEpsilon);
  EXPECT_NEAR(IsolationForestPrediction(2, 4), 0.472991377f, kEpsilon);
  EXPECT_NEAR(IsolationForestPrediction(2, 8), 0.656674445f, kEpsilon);
}

TEST(IsolationForest, Description) {
  const auto model = CreateToyModel();
  const std::string description = model->DescriptionAndStatistics(true);
  EXPECT_THAT(description, testing::HasSubstr(R"(
Tree #0:
    "a">=0 [s:0 n:0 np:0 miss:0]
        ├─(pos)─ "b">=1 [s:0 n:0 np:0 miss:0]
        |        ├─(pos)─ count:4
        |        └─(neg)─ "a">=1 [s:0 n:0 np:0 miss:0]
        |                 ├─(pos)─ count:2
        |                 └─(neg)─ count:2
        └─(neg)─ count:2

Tree #1:
    "b">=1 [s:0 n:0 np:0 miss:0]
        ├─(pos)─ count:5
        └─(neg)─ count:5
)"));
}

TEST(IsolationForest, Serialize) {
  const auto original_model = CreateToyModel();
  ASSERT_OK_AND_ASSIGN(std::string serialized_model,
                       SerializeModel(*original_model));
  ASSERT_OK_AND_ASSIGN(const auto loaded_model,
                       DeserializeModel(serialized_model));
  EXPECT_EQ(original_model->DebugCompare(*loaded_model), "");
}

TEST(IsolationForest, PredictGetLeaves) {
  const auto model = CreateToyModel();
  const auto dataset = CreateToyDataset(model->data_spec());
  std::vector<int32_t> leaves(model->num_trees());
  EXPECT_OK(model->PredictGetLeaves(dataset, 0, absl::MakeSpan(leaves)));
  EXPECT_THAT(leaves, ElementsAre(0, 0));

  EXPECT_OK(model->PredictGetLeaves(dataset, 1, absl::MakeSpan(leaves)));
  EXPECT_THAT(leaves, ElementsAre(1, 0));

  EXPECT_OK(model->PredictGetLeaves(dataset, 2, absl::MakeSpan(leaves)));
  EXPECT_THAT(leaves, ElementsAre(2, 0));

  EXPECT_OK(model->PredictGetLeaves(dataset, 3, absl::MakeSpan(leaves)));
  EXPECT_THAT(leaves, ElementsAre(3, 1));
}

TEST(IsolationForest, PredictVerticalDataset) {
  const auto model = CreateToyModel();
  const auto dataset = CreateToyDataset(model->data_spec());
  model::proto::Prediction prediction;

  model->Predict(dataset, 0, &prediction);
  EXPECT_THAT(prediction,
              EqualsProto(utils::ParseTextProto<model::proto::Prediction>(
                              "anomaly_detection { value: 0.6111162 }")
                              .value()));

  model->Predict(dataset, 1, &prediction);
  EXPECT_THAT(prediction,
              EqualsProto(utils::ParseTextProto<model::proto::Prediction>(
                              "anomaly_detection { value: 0.5079549 }")
                              .value()));

  model->Predict(dataset, 2, &prediction);
  EXPECT_THAT(prediction,
              EqualsProto(utils::ParseTextProto<model::proto::Prediction>(
                              "anomaly_detection { value: 0.5079549 }")
                              .value()));

  model->Predict(dataset, 3, &prediction);
  EXPECT_THAT(prediction,
              EqualsProto(utils::ParseTextProto<model::proto::Prediction>(
                              "anomaly_detection { value: 0.51496893 }")
                              .value()));
}

TEST(IsolationForest, PredictExampleProto) {
  const auto model = CreateToyModel();
  const auto dataset = CreateToyDataset(model->data_spec());

  dataset::proto::Example example;
  model::proto::Prediction prediction;

  dataset.ExtractExample(0, &example);
  model->Predict(example, &prediction);
  EXPECT_THAT(prediction,
              EqualsProto(utils::ParseTextProto<model::proto::Prediction>(
                              "anomaly_detection { value: 0.6111162 }")
                              .value()));

  dataset.ExtractExample(1, &example);
  model->Predict(example, &prediction);
  EXPECT_THAT(prediction,
              EqualsProto(utils::ParseTextProto<model::proto::Prediction>(
                              "anomaly_detection { value: 0.5079549 }")
                              .value()));

  dataset.ExtractExample(2, &example);
  model->Predict(example, &prediction);
  EXPECT_THAT(prediction,
              EqualsProto(utils::ParseTextProto<model::proto::Prediction>(
                              "anomaly_detection { value: 0.5079549 }")
                              .value()));

  dataset.ExtractExample(3, &example);
  model->Predict(example, &prediction);
  EXPECT_THAT(prediction,
              EqualsProto(utils::ParseTextProto<model::proto::Prediction>(
                              "anomaly_detection { value: 0.51496893 }")
                              .value()));
}

TEST(IsolationForest, Distance) {
  const auto model = CreateToyModel();
  const auto dataset = CreateToyDataset(model->data_spec());
  ASSERT_OK_AND_ASSIGN(const auto dataset_extract,
                       dataset.Extract(std::vector<int>{0, 1}));

  std::vector<float> distances(4);
  ASSERT_OK(model->Distance(dataset_extract, dataset_extract,
                            absl::MakeSpan(distances)));
  EXPECT_THAT(distances, ElementsAre(0, 0.5, 0.5, 0));
}

TEST(IsolationForest, PredictGolden) {
  // The model, dataset, and golden predictions have been generated in the test
  // "test_import_anomaly_detection_model" in
  // ydf/model/sklearn_model_test.py
  ASSERT_OK_AND_ASSIGN(const auto model,
                       model::LoadModel(file::JoinPath(
                           TestDataDir(), "model", "gaussians_anomaly_if")));
  dataset::VerticalDataset dataset;
  ASSERT_OK(dataset::LoadVerticalDataset(
      absl::StrCat("csv:", file::JoinPath(TestDataDir(), "dataset",
                                          "gaussians_test.csv")),
      model->data_spec(), &dataset));

  YDF_LOG(INFO) << "Model:\n" << model->DescriptionAndStatistics(true);

  // Those predictions have been checked with sklearn implementation.
  model::proto::Prediction prediction;

  model->Predict(dataset, 0, &prediction);
  EXPECT_NEAR(prediction.anomaly_detection().value(), 4.192874686491115943e-01,
              kEpsilon);

  model->Predict(dataset, 1, &prediction);
  EXPECT_NEAR(prediction.anomaly_detection().value(), 4.414360433426349206e-01,
              kEpsilon);

  model->Predict(dataset, 2, &prediction);
  EXPECT_NEAR(prediction.anomaly_detection().value(), 5.071637878193088200e-01,
              kEpsilon);

  model->Predict(dataset, 3, &prediction);
  EXPECT_NEAR(prediction.anomaly_detection().value(), 4.252762996248650729e-01,
              kEpsilon);

  model->Predict(dataset, 4, &prediction);
  EXPECT_NEAR(prediction.anomaly_detection().value(), 3.864382268322048009e-01,
              kEpsilon);
}

}  // namespace
}  // namespace yggdrasil_decision_forests::model::isolation_forest
