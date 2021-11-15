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

#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.h"

#include "gmock/gmock.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/test_utils.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_gradient_boosted_trees {
namespace {

using testing::ElementsAre;

class DatasetAdult : public utils::TrainAndTestTester {
 public:
  void SetNumWorkers(const int num_workers) {
    *deployment_config_.mutable_distribute() =
        PARSE_TEST_PROTO(absl::Substitute(
            R"pb(
              implementation_key: "MULTI_THREAD"
              [yggdrasil_decision_forests.distribute.proto.multi_thread] {
                num_workers: $0
              }
            )pb",
            num_workers));
  }

 private:
  void SetUp() override {
    train_config_ = PARSE_TEST_PROTO(R"pb(
      learner: "DISTRIBUTED_GRADIENT_BOOSTED_TREES"
      task: CLASSIFICATION
      label: "income"
      [yggdrasil_decision_forests.model.distributed_gradient_boosted_trees.proto
           .distributed_gradient_boosted_trees_config] {
        checkpoint_interval_trees: 50
        worker_logs: false
        gbt { export_logs_during_training_in_trees: 30 }
      }
    )pb");

    dataset_filename_ = "adult_train.csv";
    dataset_test_filename_ = "adult_test.csv";
    preferred_format_type = "csv";
    pass_training_dataset_as_path_ = true;
    num_shards_ = 20;

    deployment_config_.set_cache_path(
        file::JoinPath(test::TmpDirectory(), "working_directory"));

    SetNumWorkers(5);
  }
};

// Train and test a model on the adult dataset (with various number of workers).
TEST_F(DatasetAdult, Base_1worker) {
  SetNumWorkers(1);
  TrainAndEvaluateModel();
  // Note: This result does not take early stopping into account.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8748, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.2765, 0.04);
}

TEST_F(DatasetAdult, Base_2workers) {
  SetNumWorkers(2);
  TrainAndEvaluateModel();
  // Note: This result does not take early stopping into account.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8748, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.2765, 0.04);
}

TEST_F(DatasetAdult, Base_5workers) {
  SetNumWorkers(5);
  TrainAndEvaluateModel();
  // Note: This result does not take early stopping into account.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8748, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.2765, 0.04);
}

// Train and test a model on the adult dataset with workers continuously
// failing and requiring checkpoint restoration.
TEST_F(DatasetAdult, BaseWithFailure) {
  auto* spe_config = train_config_.MutableExtension(
      distributed_gradient_boosted_trees::proto::
          distributed_gradient_boosted_trees_config);
  spe_config->mutable_internal()->set_simulate_worker_failure(true);
  spe_config->set_checkpoint_interval_trees(5);
  TrainAndEvaluateModel();
  // Note: This result does not take early stopping into account.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8748, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.2765, 0.04);
}

// Train and test a model on the adult dataset with aggressive (64 unique
// values) forced discretization of the numerical features.
TEST_F(DatasetAdult, ForceNumericalDiscretization) {
  auto* spe_config = train_config_.MutableExtension(
      distributed_gradient_boosted_trees::proto::
          distributed_gradient_boosted_trees_config);
  spe_config->mutable_create_cache()->set_force_numerical_discretization(true);
  spe_config->mutable_create_cache()
      ->set_max_unique_values_for_discretized_numerical(16);
  TrainAndEvaluateModel();
  // Note: This result does not take early stopping into account.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8748, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.2765, 0.04);
}

// Train and test a model on the adult dataset.
TEST_F(DatasetAdult, Weighted) {
  train_config_.mutable_weight_definition()->set_attribute("hours_per_week");
  train_config_.mutable_weight_definition()->mutable_numerical();

  eval_options_.mutable_weights()->set_attribute("hours_per_week");
  eval_options_.mutable_weights()->mutable_numerical();

  TrainAndEvaluateModel();
  // Note: This result does not take early stopping into account.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.857905, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.2765, 0.04);
}

// Train and test a model on the adult dataset.
TEST_F(DatasetAdult, AttributeSampling) {
  auto* spe_config = train_config_.MutableExtension(
      distributed_gradient_boosted_trees::proto::
          distributed_gradient_boosted_trees_config);
  spe_config->mutable_gbt()
      ->mutable_decision_tree()
      ->set_num_candidate_attributes(4);
  TrainAndEvaluateModel();
  // Note: This result does not take early stopping into account.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8748, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.2765, 0.04);
}

// Makes sure the exact same tree structure is learned by the distributed and
// classical algorithms. Note: We remove the features "education_num" as it is a
// duplication of "education" (i.e. same split scores) and can lead to different
// models.
TEST_F(DatasetAdult, CompareWithClassicalAlgorithm) {
  // Classical algorithm.
  train_config_ = PARSE_TEST_PROTO(R"pb(
    learner: "GRADIENT_BOOSTED_TREES"
    task: CLASSIFICATION
    label: "income"
    features: "^(?!.*education).*$"
    [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
         .gradient_boosted_trees_config] {
      num_trees: 3
      validation_set_ratio: 0.0
      decision_tree { max_depth: 3 }
    }
  )pb");
  LOG(INFO) << "Training classical algorithm";
  TrainAndEvaluateModel();
  // Note: The description includes a details of the tree structure.
  const auto description_classical = model_->DescriptionAndStatistics(true);

  // Distributed algorithm.
  train_config_ = PARSE_TEST_PROTO(R"pb(
    learner: "DISTRIBUTED_GRADIENT_BOOSTED_TREES"
    task: CLASSIFICATION
    label: "income"
    features: "^(?!.*education).*$"
    [yggdrasil_decision_forests.model.distributed_gradient_boosted_trees.proto
         .distributed_gradient_boosted_trees_config] {
      worker_logs: false
      gbt {
        num_trees: 3
        decision_tree { max_depth: 3 }
      }
      create_cache {
        # Disable the discretization of numerical features.
        #
        # Enabling this feature leads to a one difference between the two
        # models: A threshold value of 5095.5 vs 5119.0 for the attribute
        # "capital_gain" in one of the nodes. The score and statistics of those
        # nodes are the same.
        #
        # The difference is dues to the classical algorithm applying a mean
        # in the feature domain while the distributed algorithm makes a mean
        # in the bucket domain.
        max_unique_values_for_discretized_numerical: 0
      }
    }
  )pb");
  LOG(INFO) << "Training distributed algorithm";
  TrainAndEvaluateModel();
  // Note: The description includes a details of the tree structure.
  const auto description_distributed = model_->DescriptionAndStatistics(true);

  EXPECT_EQ(description_classical, description_distributed);
}

// The load balancer continuously change the worker<->feature mapping.
TEST_F(DatasetAdult, ContinuousRebalancing) {
  auto* spe_config = train_config_.MutableExtension(
      distributed_gradient_boosted_trees::proto::
          distributed_gradient_boosted_trees_config);
  spe_config->mutable_load_balancer()
      ->mutable_internal()
      ->set_random_dynamic_balancing(true);
  TrainAndEvaluateModel();
  // Note: This result does not take early stopping into account.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8748, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.2765, 0.04);
}

class DatasetIris : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_ = PARSE_TEST_PROTO(R"pb(
      learner: "DISTRIBUTED_GRADIENT_BOOSTED_TREES"
      task: CLASSIFICATION
      label: "class"
      [yggdrasil_decision_forests.model.distributed_gradient_boosted_trees.proto
           .distributed_gradient_boosted_trees_config] { worker_logs: false }
    )pb");

    deployment_config_ = PARSE_TEST_PROTO(R"pb(
      distribute {
        implementation_key: "MULTI_THREAD"
        [yggdrasil_decision_forests.distribute.proto.multi_thread] {
          num_workers: 5
        }
      }
    )pb");

    dataset_filename_ = "iris.csv";
    preferred_format_type = "csv";
    pass_training_dataset_as_path_ = true;
    num_shards_ = 20;

    deployment_config_.set_cache_path(
        file::JoinPath(test::TmpDirectory(), "working_directory"));
  }
};

// Train and test a model on the adult dataset.
TEST_F(DatasetIris, Base) {
  TrainAndEvaluateModel();
  // Note: This result does not take early stopping into account.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.9733, 0.02);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.2762, 0.04);
  // Note: R RandomForest has an OOB accuracy of 0.9467.
}

class DatasetDna : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_ = PARSE_TEST_PROTO(R"pb(
      learner: "DISTRIBUTED_GRADIENT_BOOSTED_TREES"
      task: CLASSIFICATION
      label: "LABEL"
      [yggdrasil_decision_forests.model.distributed_gradient_boosted_trees.proto
           .distributed_gradient_boosted_trees_config] { worker_logs: false }
    )pb");

    deployment_config_ = PARSE_TEST_PROTO(R"pb(
      distribute {
        implementation_key: "MULTI_THREAD"
        [yggdrasil_decision_forests.distribute.proto.multi_thread] {
          num_workers: 5
        }
      }
    )pb");

    dataset_filename_ = "dna.csv";
    preferred_format_type = "csv";
    pass_training_dataset_as_path_ = true;
    num_shards_ = 20;

    deployment_config_.set_cache_path(
        file::JoinPath(test::TmpDirectory(), "working_directory"));
  }
};

// Train and test a model on the adult dataset.
TEST_F(DatasetDna, Base) {
  TrainAndEvaluateModel();
  // Note: This result does not take early stopping into account.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.95291, 0.02);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.27166, 0.04);
  // Note: R RandomForest has an OOB accuracy of 0.909.
}

class DatasetAbalone : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_ = PARSE_TEST_PROTO(R"pb(
      learner: "DISTRIBUTED_GRADIENT_BOOSTED_TREES"
      task: REGRESSION
      label: "Rings"
      [yggdrasil_decision_forests.model.distributed_gradient_boosted_trees.proto
           .distributed_gradient_boosted_trees_config] { worker_logs: false }
    )pb");

    deployment_config_ = PARSE_TEST_PROTO(R"pb(
      distribute {
        implementation_key: "MULTI_THREAD"
        [yggdrasil_decision_forests.distribute.proto.multi_thread] {
          num_workers: 5
        }
      }
    )pb");

    dataset_filename_ = "abalone.csv";
    preferred_format_type = "csv";
    pass_training_dataset_as_path_ = true;
    num_shards_ = 20;

    deployment_config_.set_cache_path(
        file::JoinPath(test::TmpDirectory(), "working_directory"));
  }
};

// Train and test a model on the adult dataset.
TEST_F(DatasetAbalone, Base) {
  TrainAndEvaluateModel();
  // Note: This result does not take early stopping into account.
  EXPECT_NEAR(metric::RMSE(evaluation_), 2.205387, 0.01);
}

}  // namespace
}  // namespace distributed_gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
