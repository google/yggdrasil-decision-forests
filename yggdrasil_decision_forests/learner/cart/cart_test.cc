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

#include "yggdrasil_decision_forests/learner/cart/cart.h"

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "absl/log/log.h"
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

using Internal = ::yggdrasil_decision_forests::model::decision_tree::proto::
    DecisionTreeTrainingConfig::Internal;

void SetExpectedSortingStrategy(Internal::SortingStrategy expected,
                                model::proto::TrainingConfig* train_config) {
  auto* cart_config = train_config->MutableExtension(cart::proto::cart_config);
  cart_config->mutable_decision_tree()
      ->mutable_internal()
      ->set_ensure_effective_sorting_strategy(expected);
}

class CartOnAdult : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(CartLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    train_config_.set_label("income");
    train_config_.add_features(".*");
    dataset_filename_ = "adult.csv";

    // CART only uses IN_NODE sorting.
    SetExpectedSortingStrategy(Internal::IN_NODE, &train_config_);
  }
};

TEST_F(CartOnAdult, Base) {
  TrainAndEvaluateModel();
  // Random Forest has an accuracy of ~0.860.
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8546, 0.011, 0.8552);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.4392, 0.156, 0.4082);

  // Show the tree structure.
  std::string description;
  model_->AppendDescriptionAndStatistics(true, &description);
  LOG(INFO) << description;

  utils::ExpectEqualGoldenModel(*model_, "cart_adult");
}

// Similar as "CartOnAdult", but use the pre-split train and test dataset (
// instead of generating the splits).
class CartOnAdultPreSplit : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(CartLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    train_config_.set_label("income");
    train_config_.add_features(".*");
    dataset_filename_ = "adult_train.csv";
    dataset_test_filename_ = "adult_test.csv";

    // CART only use IN_NODE sorting.
    SetExpectedSortingStrategy(Internal::IN_NODE, &train_config_);
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

    // CART only use IN_NODE sorting.
    auto* cart_config =
        train_config_.MutableExtension(cart::proto::cart_config);
    cart_config->mutable_decision_tree()
        ->mutable_internal()
        ->set_ensure_effective_sorting_strategy(
            decision_tree::proto::DecisionTreeTrainingConfig::Internal::
                IN_NODE);
  }
};

TEST_F(CartOnAbalone, Base) {
  TrainAndEvaluateModel();
  // Random Forest has an rmse of ~2.10.
  // The default RMSE (always retuning the label mean) is ~3.204.
  YDF_TEST_METRIC(metric::RMSE(evaluation_), 2.3728, 0.1566, 2.3054);

  auto* rf_model =
      dynamic_cast<const random_forest::RandomForestModel*>(model_.get());
  EXPECT_GT(rf_model->num_pruned_nodes().value(), 50);
  EXPECT_GT(rf_model->NumNodes(), 10);

  utils::ExpectEqualGoldenModel(*model_, "cart_abalone");
}

class CartOnIris : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(CartLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    train_config_.set_label("class");
    dataset_filename_ = "iris.csv";

    // CART only uses IN_NODE sorting.
    SetExpectedSortingStrategy(Internal::IN_NODE, &train_config_);
  }
};

TEST_F(CartOnIris, Base) {
  TrainAndEvaluateModel();
  // Random Forest has an accuracy of ~0.947.

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.82, 0.23, 0.9333);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 1.0176, 1.3571, 1.0883);

  const auto valid_acc = metric::Accuracy(model_->ValidationEvaluation());
  YDF_TEST_METRIC(valid_acc, 0.8, 0.3, 0.8333);
}

TEST_F(CartOnIris, WithManualValidation) {
  pass_validation_dataset_ = true;
  TrainAndEvaluateModel();
  // This test should have no variance.
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.973, 0.0001, 0.973);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.1135, 0.0001, 0.1135);
  const auto valid_acc = metric::Accuracy(model_->ValidationEvaluation());
  YDF_TEST_METRIC(valid_acc, 0.9211, 0.0001, 0.9211);
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
    CHECK_OK(dataset_.AppendExampleWithStatus({{"a", "0"}, {"l", "1"}}));
    CHECK_OK(dataset_.AppendExampleWithStatus({{"a", "1"}, {"l", "2"}}));
    CHECK_OK(dataset_.AppendExampleWithStatus({{"a", "2"}, {"l", "2"}}));

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
  std::vector<UnsignedExampleIdx> example_idxs = {0, 1, 2};
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

    // CART only use IN_NODE sorting.
    SetExpectedSortingStrategy(Internal::IN_NODE, &train_config_);
  }
};

TEST_F(CartOnSimPTE, Base) {
  TrainAndEvaluateModel();
  // Note: A Qini of ~0.1 is expected with a simple Random Forest model.
  YDF_TEST_METRIC(metric::Qini(evaluation_), 0.058, 0.041, 0.0825);

  auto* rf_model =
      dynamic_cast<const random_forest::RandomForestModel*>(model_.get());

  YDF_TEST_METRIC(rf_model->num_pruned_nodes().value(), 49.0, 46.5, 58.0);
  YDF_TEST_METRIC(rf_model->NumNodes(), 37.0, 42.0, 33.0);
}

TEST_F(CartOnSimPTE, Honest) {
  auto* config = train_config_.MutableExtension(cart::proto::cart_config);
  config->mutable_decision_tree()->mutable_honest();

  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::Qini(evaluation_), 0.044, 0.0521, 0.0276);
}

}  // namespace
}  // namespace cart
}  // namespace model
}  // namespace yggdrasil_decision_forests
