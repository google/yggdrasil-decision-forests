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
  // The default RMSE (always retuning the label mean) is ~3.204.
  // Note: This test show a lot of variance with RMSE ranging from 2.333
  // to 2.649 from only changes in the random generator output.
  EXPECT_LT(metric::RMSE(evaluation_), 2.67);
  EXPECT_GT(metric::RMSE(evaluation_), 2.31);

  auto* rf_model =
      dynamic_cast<const random_forest::RandomForestModel*>(model_.get());
  EXPECT_GT(rf_model->num_pruned_nodes().value(), 50);
  EXPECT_GT(rf_model->NumNodes(), 10);
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

class CartPruningTest : public ::testing::Test {
 protected:
  void SetUp() override {
    *dataset_.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
      columns { type: NUMERICAL name: "a" }
      columns {
        type: CATEGORICAL
        name: "l"
        categorical { number_of_unique_values: 3 is_already_integerized: true }
      }
    )pb");
    EXPECT_OK(dataset_.CreateColumnsFromDataspec());
    dataset_.AppendExample({{"a", "0"}, {"l", "1"}});
    dataset_.AppendExample({{"a", "1"}, {"l", "2"}});
    dataset_.AppendExample({{"a", "2"}, {"l", "2"}});

    config_link_.set_label(1);

    tree_.CreateRoot();
    tree_.mutable_root()->CreateChildren();
    tree_.mutable_root()->mutable_node()->mutable_condition()->set_attribute(0);
    tree_.mutable_root()
        ->mutable_node()
        ->mutable_condition()
        ->mutable_condition()
        ->mutable_higher_condition()
        ->set_threshold(0.5);

    auto* neg_child = tree_.mutable_root()->mutable_neg_child();
    auto* pos_child = tree_.mutable_root()->mutable_pos_child();

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
  }

  dataset::VerticalDataset dataset_;
  model::decision_tree::DecisionTree tree_;
  model::proto::TrainingConfig config_;
  model::proto::TrainingConfigLinking config_link_;
};

TEST_F(CartPruningTest, PruneTree) {
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

  EXPECT_EQ(tree_.NumNodes(), 5);

  std::vector<float> weights = {1.f, 1.f, 1.f};
  std::vector<dataset::VerticalDataset::row_t> example_idxs = {0, 1, 2};
  EXPECT_OK(internal::PruneTree(dataset_, weights, example_idxs, config_,
                                config_link_, &tree_));

  // Note: There is only one way to prune the tree and make it having 3 nodes.
  EXPECT_EQ(tree_.NumNodes(), 3);
}

class CartOnSimPTE : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(CartLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::CATEGORICAL_UPLIFT);
    train_config_.set_label("y");
    train_config_.set_uplift_treatment("treat");

    guide_ = PARSE_TEST_PROTO(
        R"pb(
          column_guides {
            column_name_pattern: "(y|treat)"
            type: CATEGORICAL
            categorial { is_already_integerized: true }
          }
          detect_boolean_as_numerical: true
        )pb");

    dataset_filename_ = "sim_pte_train.csv";
    dataset_test_filename_ = "sim_pte_test.csv";
  }
};

TEST_F(CartOnSimPTE, Base) {
  TrainAndEvaluateModel();
  // Note: A Qini of ~0.1 is expected with a simple Random Forest model.
  EXPECT_NEAR(metric::Qini(evaluation_), 0.06724, 0.01);

  auto* rf_model =
      dynamic_cast<const random_forest::RandomForestModel*>(model_.get());
  EXPECT_GT(rf_model->num_pruned_nodes().value(), 50);
  EXPECT_GT(rf_model->NumNodes(), 20);
}

TEST_F(CartOnSimPTE, Honest) {
  auto* config = train_config_.MutableExtension(cart::proto::cart_config);
  config->mutable_decision_tree()->mutable_honest();

  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Qini(evaluation_), 0.05047, 0.01);
}

}  // namespace
}  // namespace cart
}  // namespace model
}  // namespace yggdrasil_decision_forests
