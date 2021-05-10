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

#include "yggdrasil_decision_forests/learner/cart/cart.h"

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/test_utils.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace cart {
namespace {

class CartOnAdult : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(CartLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    train_config_.set_label("income");
    train_config_.add_features(".*");
    dataset_filename_ = "adult.csv";
  }
};

TEST_F(CartOnAdult, Base) {
  TrainAndEvaluateModel();
  // Random Forest has an accuracy of ~0.860.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8560, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.4373, 0.04);

  // Show the tree structure.
  std::string description;
  model_->AppendDescriptionAndStatistics(true, &description);
  LOG(INFO) << description;
}

// Similar as "CartOnAdult", but use the pre-splitted train and test dataset (
// instead of generating the splits).
class CartOnAdultPreSplit : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(CartLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    train_config_.set_label("income");
    train_config_.add_features(".*");
    dataset_filename_ = "adult_train.csv";
    dataset_test_filename_ = "adult_test.csv";
  }
};

TEST_F(CartOnAdultPreSplit, Base) {
  TrainAndEvaluateModel();
  // Random Forest has an accuracy of ~0.860.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8560, 0.01);
}

class CartOnAbalone : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(CartLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::REGRESSION);
    train_config_.set_label("Rings");
    dataset_filename_ = "abalone.csv";
  }
};

TEST_F(CartOnAbalone, Base) {
  TrainAndEvaluateModel();
  // Random Forest has an rmse of ~2.10.
  EXPECT_NEAR(metric::RMSE(evaluation_), 3.2047, 0.01);
}

class CartOnIris : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(CartLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    train_config_.set_label("class");
    dataset_filename_ = "iris.csv";
  }
};

TEST_F(CartOnIris, Base) {
  TrainAndEvaluateModel();
  // Random Forest has an accuracy of ~0.947.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.9333, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 1.0883, 0.04);
}

TEST(Cart, SetHyperParameters) {
  CartLearner learner{model::proto::TrainingConfig()};
  const auto hparam_spec =
      learner.GetGenericHyperParameterSpecification().value();
  EXPECT_OK(learner.SetHyperParameters(model::proto::GenericHyperParameters()));
  EXPECT_OK(learner.SetHyperParameters(PARSE_TEST_PROTO(
      "fields { name: \"validation_ratio\" value { real: 0.5 } }")));
  const auto& cart_config =
      learner.training_config().GetExtension(cart::proto::cart_config);
  EXPECT_NEAR(cart_config.validation_ratio(), 0.5, 0.001);
}

TEST(Cart, PruneTree) {
  // The classification tree is as follow:
  // (a>0.5; pred=1)
  //   │
  //   ├─[pred=1]
  //   └─(a>1.5; pred=2)
  //         │
  //         ├─[pred=1]
  //         └─[pred=2]
  //
  // The dataset is:
  //   a  l
  //   0  1
  //   1  2
  //   2  2
  //
  // The node "a>1.5" should be pruned.

  dataset::VerticalDataset dataset;
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "a" }
    columns {
      type: CATEGORICAL
      name: "l"
      categorical { number_of_unique_values: 3 is_already_integerized: true }
    }
  )pb");
  EXPECT_OK(dataset.CreateColumnsFromDataspec());
  dataset.AppendExample({{"a", "0"}, {"l", "1"}});
  dataset.AppendExample({{"a", "1"}, {"l", "2"}});
  dataset.AppendExample({{"a", "2"}, {"l", "2"}});

  std::vector<float> weights = {1.f, 1.f, 1.f};
  std::vector<dataset::VerticalDataset::row_t> example_idxs = {0, 1, 2};
  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(1);

  model::decision_tree::DecisionTree tree;
  tree.CreateRoot();
  tree.mutable_root()->CreateChildren();
  tree.mutable_root()->mutable_node()->mutable_condition()->set_attribute(0);
  tree.mutable_root()
      ->mutable_node()
      ->mutable_condition()
      ->mutable_condition()
      ->mutable_higher_condition()
      ->set_threshold(0.5);

  auto* neg_child = tree.mutable_root()->mutable_neg_child();
  auto* pos_child = tree.mutable_root()->mutable_pos_child();

  neg_child->mutable_node()->mutable_classifier()->set_top_value(1);
  pos_child->mutable_node()->mutable_classifier()->set_top_value(2);

  pos_child->CreateChildren();
  pos_child->mutable_node()->mutable_condition()->set_attribute(0);
  pos_child->mutable_node()
      ->mutable_condition()
      ->mutable_condition()
      ->mutable_higher_condition()
      ->set_threshold(1.5);

  pos_child->mutable_neg_child()
      ->mutable_node()
      ->mutable_classifier()
      ->set_top_value(1);
  pos_child->mutable_pos_child()
      ->mutable_node()
      ->mutable_classifier()
      ->set_top_value(2);

  EXPECT_EQ(tree.NumNodes(), 5);

  EXPECT_OK(internal::PruneTree(dataset, weights, example_idxs, config,
                                config_link, &tree));

  // Note: There is only one way to prune the tree and make it having 3 nodes.
  EXPECT_EQ(tree.NumNodes(), 3);
}

}  // namespace
}  // namespace cart
}  // namespace model
}  // namespace yggdrasil_decision_forests
