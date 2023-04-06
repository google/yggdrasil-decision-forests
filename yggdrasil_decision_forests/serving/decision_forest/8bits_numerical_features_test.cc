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

#include "yggdrasil_decision_forests/serving/decision_forest/8bits_numerical_features.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_join.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace decision_forest {
namespace num_8bits {
namespace {

using model::decision_tree::DecisionTree;
using model::decision_tree::NodeWithChildren;
using model::gradient_boosted_trees::GradientBoostedTreesModel;
using model::gradient_boosted_trees::proto::Loss;
using testing::ElementsAre;

std::string TestDataDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/test_data");
}

void BuildToyModelAndToyDataset(
    model::gradient_boosted_trees::GradientBoostedTreesModel* model,
    dataset::VerticalDataset* dataset) {
  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns {
      type: CATEGORICAL
      name: "label"
      categorical { is_already_integerized: true number_of_unique_values: 3 }
    }
    columns {
      type: DISCRETIZED_NUMERICAL
      name: "f1"
      discretized_numerical {
        boundaries: 0.5
        boundaries: 1.5
        boundaries: 2.5
        boundaries: 3.5
      }
    }
    columns {
      type: DISCRETIZED_NUMERICAL
      name: "f2"
      discretized_numerical { boundaries: 0.5 boundaries: 1.5 boundaries: 2.5 }
    }
  )pb");

  dataset->set_data_spec(dataspec);
  CHECK_OK(dataset->CreateColumnsFromDataspec());

  CHECK_OK(dataset->AppendExampleWithStatus(
      {{"label", "1"}, {"f1", "0"}, {"f2", "2"}}));

  struct NodeHelper {
    NodeWithChildren* pos;
    NodeWithChildren* neg;
  };

  const auto split = [](NodeWithChildren* node, const int attribute,
                        const uint8_t threshold) -> NodeHelper {
    node->CreateChildren();
    node->mutable_node()->mutable_condition()->set_attribute(attribute);
    node->mutable_node()
        ->mutable_condition()
        ->mutable_condition()
        ->mutable_discretized_higher_condition()
        ->set_threshold(threshold);
    return {/*.pos =*/node->mutable_pos_child(),
            /*.neg =*/node->mutable_neg_child()};
  };

  model->set_task(model::proto::Task::CLASSIFICATION);
  model->set_label_col_idx(0);
  model->set_data_spec(dataspec);
  model->set_loss(Loss::BINOMIAL_LOG_LIKELIHOOD);
  model->mutable_initial_predictions()->push_back(1.f);
  *model->mutable_input_features() = {1, 2};
  // Avoid an access to uninitialized memory when printing the model.
  model->set_num_trees_per_iter(1);

  {
    // Tree #0:
    //     "f1".index >= 2 i.e. "f1" >= 1.5
    //         ├─(pos)─ "f1".index >= 3 i.e. "f1" >= 2.5
    //         |        ├─(pos)─ pred:1
    //         |        └─(neg)─ pred:2
    //         └─(neg)─ "f2".index >= 1 i.e. "f2" >= 0.5
    //                  ├─(pos)─ pred:3
    //                  └─(neg)─ pred:4
    auto tree = absl::make_unique<DecisionTree>();
    tree->CreateRoot();
    auto n1 = split(tree->mutable_root(), 1, 2);

    auto n2 = split(n1.pos, 1, 3);
    n2.pos->mutable_node()->mutable_regressor()->set_top_value(1.f);
    n2.neg->mutable_node()->mutable_regressor()->set_top_value(2.f);

    auto n3 = split(n1.neg, 2, 1);
    n3.pos->mutable_node()->mutable_regressor()->set_top_value(3.f);
    n3.neg->mutable_node()->mutable_regressor()->set_top_value(4.f);

    model->mutable_decision_trees()->push_back(std::move(tree));
  }

  {
    // Tree #1:
    //     "f2".index >= 2 i.e. "f2" >= 1.5
    //         ├─(pos)─ pred:10
    //         └─(neg)─ pred:11
    auto tree = absl::make_unique<DecisionTree>();
    tree->CreateRoot();
    auto n1 = split(tree->mutable_root(), 2, 2);
    n1.pos->mutable_node()->mutable_regressor()->set_top_value(10.f);
    n1.neg->mutable_node()->mutable_regressor()->set_top_value(11.f);

    model->mutable_decision_trees()->push_back(std::move(tree));
  }
}

float logistic(const float value) {
  return utils::clamp(1.f / (1.f + std::exp(-value)), 0.f, 1.f);
}

TEST(Num8Bits, ToyExample) {
  GradientBoostedTreesModel model;
  dataset::VerticalDataset dataset;
  BuildToyModelAndToyDataset(&model, &dataset);
  YDF_LOG(INFO) << "Model:\n" << model.DescriptionAndStatistics(true);

  num_8bits::GradientBoostedTreesBinaryClassificationModel engine;
  CHECK_OK(GenericToSpecializedModel(model, &engine));

  YDF_LOG(INFO) << "Engine:\n" << EngineDetails(engine);

  EXPECT_EQ(engine.num_trees, 2);
  EXPECT_EQ(engine.num_features, 2);
  EXPECT_EQ(engine.initial_prediction, 1);
  // EXPECT_EQ(engine.masks.size(), 18); // V0
  // EXPECT_THAT(engine.masks_feature_index, ElementsAre(0, 10)); // V0
  EXPECT_THAT(engine.num_buckets, ElementsAre(5, 4));
  EXPECT_THAT(engine.leaves, ElementsAre(4, 3, 2, 1, 11, 10));
  EXPECT_THAT(engine.leaves_tree_index, ElementsAre(0, 4));
  EXPECT_THAT(engine.features, ElementsAre(1, 2));

  std::vector<uint8_t> examples = {1, 0, 2, 0, 3, 0, 1, 1, 1, 2, 1, 3};
  std::vector<float> predictions;
  CHECK_OK(Predict(engine, examples, /*num_examples=*/6, &predictions));

  EXPECT_THAT(predictions,
              ElementsAre(logistic(16), logistic(14), logistic(13),
                          logistic(15), logistic(14), logistic(14)));
}

TEST(Num8Bits, CompareToSlowEngine) {
  const std::string model_dir = file::JoinPath(
      TestDataDir(), "model", "8bits_numerical_binary_class_gbdt");
  std::unique_ptr<model::AbstractModel> model;
  CHECK_OK(model::LoadModel(model_dir, &model));

  const std::string ds_typed_path = absl::StrCat(
      "csv:",
      file::JoinPath(TestDataDir(), "dataset", "8bits_numerical_test.csv"));
  dataset::VerticalDataset dataset;
  CHECK_OK(LoadVerticalDataset(ds_typed_path, model->data_spec(), &dataset));

  // Ground truth prediction
  std::vector<float> ground_truth_predictions;
  for (int example_idx = 0; example_idx < dataset.nrow(); example_idx++) {
    model::proto::Prediction prediction;
    model->Predict(dataset, example_idx, &prediction);
    ground_truth_predictions.push_back(
        prediction.classification().distribution().counts(2) /
        prediction.classification().distribution().sum());
  }

  // Compile model
  auto* gbt_model =
      dynamic_cast<model::gradient_boosted_trees::GradientBoostedTreesModel*>(
          model.get());
  CHECK(gbt_model);
  GradientBoostedTreesBinaryClassificationModel engine;
  CHECK_OK(GenericToSpecializedModel(*gbt_model, &engine));
  YDF_LOG(INFO) << "Engine:\n" << EngineDetails(engine);

  // Prediction with new inference engine.
  std::vector<uint8_t> examples(engine.num_features * dataset.nrow());
  for (int example_idx = 0; example_idx < dataset.nrow(); example_idx++) {
    for (int local_feature_idx = 0; local_feature_idx < engine.num_features;
         local_feature_idx++) {
      const auto feature_index_values =
          dataset
              .ColumnWithCast<
                  dataset::VerticalDataset::DiscretizedNumericalColumn>(
                  engine.features[local_feature_idx])
              ->values();

      // Note: For this model, the discretized feature indices and the
      // corresponding feature values are equal.
      examples[example_idx * engine.num_features + local_feature_idx] =
          feature_index_values[example_idx];
    }
  }
  std::vector<float> engine_predictions;
  CHECK_OK(Predict(engine, examples, dataset.nrow(), &engine_predictions));

  EXPECT_EQ(engine_predictions.size(), ground_truth_predictions.size());
  for (int i = 0; i < engine_predictions.size(); i++) {
    EXPECT_NEAR(engine_predictions[i], ground_truth_predictions[i], 0.0001f);
  }
}

}  // namespace
}  // namespace num_8bits
}  // namespace decision_forest
}  // namespace serving
}  // namespace yggdrasil_decision_forests
