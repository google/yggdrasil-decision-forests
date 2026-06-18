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

#include "yggdrasil_decision_forests/learner/hyperparameters_optimizer/hyperparameters_optimizer.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/test_utils.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace hyperparameters_optimizer_v2 {
namespace {

using ::testing::HasSubstr;

class OnAdult : public utils::TrainAndTestTester {
  void SetUp() override {
    SetTrainConfig();
    dataset_filename_ = "adult.csv";

    deployment_config_.set_cache_path(
        file::JoinPath(test::TmpDirectory(), "working_directory"));
  }

 public:
  void SetTrainConfig(const absl::string_view optimizer_key = "RANDOM",
                      const absl::string_view optimizer_proto = "random",
                      const int num_trials = 25) {
    train_config_ = PARSE_TEST_PROTO(absl::Substitute(
        R"pb(
          task: CLASSIFICATION
          learner: "HYPERPARAMETER_OPTIMIZER"
          label: "income"

          [yggdrasil_decision_forests.model.hyperparameters_optimizer_v2.proto
               .hyperparameters_optimizer_config] {

            retrain_final_model: true

            optimizer {
              optimizer_key: "$0"
              [yggdrasil_decision_forests.model.hyperparameters_optimizer_v2
                   .proto.$2] { num_trials: $1 }
            }

            base_learner {
              learner: "GRADIENT_BOOSTED_TREES"
              [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
                   .gradient_boosted_trees_config] { num_trees: 50 }
            }

            base_learner_deployment {
              # The multi-threading is done at the optimizer level.
              num_threads: 1
            }

            search_space {
              fields {
                name: "num_candidate_attributes_ratio"
                discrete_candidates {
                  possible_values { real: 1.0 }
                  possible_values { real: 0.8 }
                  possible_values { real: 0.6 }
                }
              }

              fields {
                name: "use_hessian_gain"
                discrete_candidates {
                  possible_values { categorical: "true" }
                  possible_values { categorical: "false" }
                }
              }

              fields {
                name: "growing_strategy"
                discrete_candidates {
                  possible_values { categorical: "LOCAL" }
                  possible_values { categorical: "BEST_FIRST_GLOBAL" }
                }

                children {
                  parent_discrete_values {
                    possible_values { categorical: "LOCAL" }
                  }
                  name: "max_depth"
                  discrete_candidates {
                    possible_values { integer: 4 }
                    possible_values { integer: 5 }
                    possible_values { integer: 6 }
                    possible_values { integer: 7 }
                  }
                }

                children {
                  parent_discrete_values {
                    possible_values { categorical: "BEST_FIRST_GLOBAL" }
                  }
                  name: "max_num_nodes"
                  discrete_candidates {
                    possible_values { integer: 16 }
                    possible_values { integer: 32 }
                    possible_values { integer: 64 }
                    possible_values { integer: 128 }
                  }
                }
              }
            }
          }
        )pb",
        optimizer_key, num_trials, optimizer_proto));
  }

  void SetLocalTraining() {
    deployment_config_.set_num_threads(4);
    deployment_config_.clear_distribute();
  }

  void SetDistributedTraining(const int num_workers = 2) {
    // Note: This test effectively using local multi-threaded training. However,
    // because we use the "distribute" API, this would behave the same way as
    // with distributed training.
    deployment_config_.set_num_threads(2);
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
};

TEST_F(OnAdult, RandomTuner_MemoryDataset_LocalTraining) {
  SetLocalTraining();
  TrainAndEvaluateModel();
  EXPECT_GE(metric::Accuracy(evaluation_), 0.865);
  EXPECT_LT(metric::LogLoss(evaluation_), 0.30);
  EXPECT_EQ(model_->hyperparameter_optimizer_logs()->steps_size(), 25);
}

TEST_F(OnAdult, RandomTuner_MemoryDataset_LocalTrainingCrossValidation) {
  SetLocalTraining();
  auto* spe_config = train_config_.MutableExtension(
      hyperparameters_optimizer_v2::proto::hyperparameters_optimizer_config);
  spe_config->mutable_evaluation()->mutable_cross_validation();
  SetTrainConfig("RANDOM", "random", 10);
  TrainAndEvaluateModel();
  EXPECT_GE(metric::Accuracy(evaluation_), 0.865);
  EXPECT_LT(metric::LogLoss(evaluation_), 0.30);
  EXPECT_EQ(model_->hyperparameter_optimizer_logs()->steps_size(), 10);
}

TEST_F(OnAdult, RandomTuner_FileDataset_LocalTraining) {
  pass_training_dataset_as_path_ = true;
  SetLocalTraining();
  TrainAndEvaluateModel();
  EXPECT_GE(metric::Accuracy(evaluation_), 0.865);
  EXPECT_LT(metric::LogLoss(evaluation_), 0.30);
  EXPECT_EQ(model_->hyperparameter_optimizer_logs()->steps_size(), 25);
}

TEST_F(OnAdult, RandomTuner_MemoryDataset_DistributedTraining) {
  SetDistributedTraining();
  TrainAndEvaluateModel();
  EXPECT_GE(metric::Accuracy(evaluation_), 0.865);
  EXPECT_LT(metric::LogLoss(evaluation_), 0.30);
  EXPECT_EQ(model_->hyperparameter_optimizer_logs()->steps_size(), 25);
}

TEST_F(OnAdult, RandomTuner_MemoryDataset_DistributedTrainingCrossValidation) {
  SetDistributedTraining();
  auto* spe_config = train_config_.MutableExtension(
      hyperparameters_optimizer_v2::proto::hyperparameters_optimizer_config);
  spe_config->mutable_evaluation()->mutable_cross_validation();
  SetTrainConfig("RANDOM", "random", 10);
  TrainAndEvaluateModel();
  EXPECT_GE(metric::Accuracy(evaluation_), 0.865);
  EXPECT_LT(metric::LogLoss(evaluation_), 0.30);
  EXPECT_EQ(model_->hyperparameter_optimizer_logs()->steps_size(), 10);
}

TEST_F(OnAdult, RandomTuner_FileDataset_DistributedTraining) {
  SetDistributedTraining();
  pass_training_dataset_as_path_ = true;
  TrainAndEvaluateModel();
  EXPECT_GE(metric::Accuracy(evaluation_), 0.865);
  EXPECT_LT(metric::LogLoss(evaluation_), 0.30);
  EXPECT_EQ(model_->hyperparameter_optimizer_logs()->steps_size(), 25);
}

TEST_F(OnAdult, RandomTuner_MemoryDataset_LocalTraining_NoRetrain) {
  auto* spe_config = train_config_.MutableExtension(
      hyperparameters_optimizer_v2::proto::hyperparameters_optimizer_config);
  spe_config->set_retrain_final_model(false);

  SetLocalTraining();
  TrainAndEvaluateModel();
  EXPECT_GE(metric::Accuracy(evaluation_), 0.865);
  EXPECT_LT(metric::LogLoss(evaluation_), 0.30);
  EXPECT_EQ(model_->hyperparameter_optimizer_logs()->steps_size(), 25);
}

TEST_F(OnAdult, RandomTuner_FileDataset_DistributedTraining_NoRetrain) {
  auto* spe_config = train_config_.MutableExtension(
      hyperparameters_optimizer_v2::proto::hyperparameters_optimizer_config);
  spe_config->set_retrain_final_model(false);

  SetDistributedTraining();
  pass_training_dataset_as_path_ = true;
  TrainAndEvaluateModel();
  EXPECT_GE(metric::Accuracy(evaluation_), 0.865);
  EXPECT_LT(metric::LogLoss(evaluation_), 0.30);
  EXPECT_EQ(model_->hyperparameter_optimizer_logs()->steps_size(), 25);
}

TEST_F(OnAdult, RandomTuner_MemoryDataset_LocalTraining_DefaultHPSpace) {
  SetLocalTraining();
  TrainAndEvaluateModel();
  EXPECT_GE(metric::Accuracy(evaluation_), 0.865);
  EXPECT_LT(metric::LogLoss(evaluation_), 0.30);
  EXPECT_EQ(model_->hyperparameter_optimizer_logs()->steps_size(), 25);
}

TEST_F(OnAdult, DefaultTargetMetric) {
  metric::proto::EvaluationResults evaluation;
  EXPECT_FALSE(internal::DefaultTargetMetric(evaluation).ok());

  evaluation.mutable_uplift()->set_qini(1);
  EXPECT_TRUE(
      internal::DefaultTargetMetric(evaluation).value().uplift().has_qini());

  evaluation.mutable_ranking()->mutable_ndcg()->set_value(1);
  EXPECT_TRUE(
      internal::DefaultTargetMetric(evaluation).value().ranking().has_ndcg());

  evaluation.mutable_regression()->set_sum_square_error(1);
  EXPECT_TRUE(internal::DefaultTargetMetric(evaluation)
                  .value()
                  .regression()
                  .has_rmse());

  evaluation.mutable_classification()->set_accuracy(1);
  EXPECT_TRUE(internal::DefaultTargetMetric(evaluation)
                  .value()
                  .classification()
                  .has_accuracy());

  evaluation.mutable_classification()->add_rocs();
  evaluation.mutable_classification()->add_rocs()->set_auc(1);
  evaluation.mutable_classification()->add_rocs()->set_auc(1);
  EXPECT_TRUE(internal::DefaultTargetMetric(evaluation)
                  .value()
                  .classification()
                  .one_vs_other()
                  .has_auc());

  evaluation.set_loss_value(1);
  EXPECT_TRUE(internal::DefaultTargetMetric(evaluation).value().has_loss());
}

TEST_F(OnAdult, GBT_MonotonicConstraints) {
  SetTrainConfig("RANDOM", "random", 5);

  auto* constrain_1 = train_config_.add_monotonic_constraints();
  constrain_1->set_feature("^age$");
  constrain_1->set_direction(model::proto::MonotonicConstraint::INCREASING);

  auto* spe_config = train_config_.MutableExtension(
      hyperparameters_optimizer_v2::proto::hyperparameters_optimizer_config);

  auto* gbt_config = spe_config->mutable_base_learner()->MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_use_hessian_gain(true);
  gbt_config->mutable_decision_tree()
      ->mutable_internal()
      ->set_check_monotonic_constraints(true);

  auto* search_space = spe_config->mutable_search_space();
  model::proto::HyperParameterSpace new_space;
  for (const auto& field : search_space->fields()) {
    if (field.name() != "use_hessian_gain" &&
        field.name() != "growing_strategy") {
      *new_space.add_fields() = field;
    }
  }
  *search_space = new_space;

  auto* max_depth_field = search_space->add_fields();
  max_depth_field->set_name("max_depth");
  auto* candidates = max_depth_field->mutable_discrete_candidates();
  candidates->add_possible_values()->set_integer(4);
  candidates->add_possible_values()->set_integer(6);

  SetLocalTraining();
  TrainAndEvaluateModel();

  EXPECT_GE(metric::Accuracy(evaluation_), 0.80);
  EXPECT_EQ(model_->hyperparameter_optimizer_logs()->steps_size(), 5);
  EXPECT_EQ(model_->name(), "GRADIENT_BOOSTED_TREES");
}

TEST_F(OnAdult, RF_MonotonicConstraints_Fail) {
  SetTrainConfig("RANDOM", "random", 5);

  auto* constrain_1 = train_config_.add_monotonic_constraints();
  constrain_1->set_feature("^age$");
  constrain_1->set_direction(model::proto::MonotonicConstraint::INCREASING);

  auto* spe_config = train_config_.MutableExtension(
      hyperparameters_optimizer_v2::proto::hyperparameters_optimizer_config);
  spe_config->mutable_base_learner()->set_learner("RANDOM_FOREST");
  spe_config->mutable_base_learner()->ClearExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);

  spe_config->mutable_search_space()->clear_fields();
  auto* field = spe_config->mutable_search_space()->add_fields();
  field->set_name("num_trees");
  field->mutable_discrete_candidates()->add_possible_values()->set_integer(10);

  SetLocalTraining();

  PrepareDataset();

  std::unique_ptr<model::AbstractLearner> learner;
  ASSERT_OK(model::GetLearner(train_config_, &learner, deployment_config_));

  auto model_or = learner->TrainWithStatus(train_dataset_);
  EXPECT_FALSE(model_or.ok());
  EXPECT_EQ(model_or.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(model_or.status().message(),
              HasSubstr("does not support monotonic constraints"));
}

TEST_F(OnAdult, EmptyBaseLearner_Fail) {
  SetTrainConfig("RANDOM", "random", 5);
  auto* spe_config = train_config_.MutableExtension(
      hyperparameters_optimizer_v2::proto::hyperparameters_optimizer_config);
  spe_config->clear_base_learner();

  SetLocalTraining();
  PrepareDataset();

  std::unique_ptr<model::AbstractLearner> learner;
  ASSERT_OK(model::GetLearner(train_config_, &learner, deployment_config_));

  auto model_or = learner->TrainWithStatus(train_dataset_);
  EXPECT_FALSE(model_or.ok());
  EXPECT_EQ(model_or.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(model_or.status().message(),
              HasSubstr("base_learner config or learner name is missing"));
}

TEST_F(OnAdult, InvalidParallelTrials_Fail) {
  SetTrainConfig("RANDOM", "random", 5);
  auto* spe_config = train_config_.MutableExtension(
      hyperparameters_optimizer_v2::proto::hyperparameters_optimizer_config);
  spe_config->mutable_optimizer()->set_parallel_trials(0);

  SetLocalTraining();
  PrepareDataset();

  std::unique_ptr<model::AbstractLearner> learner;
  ASSERT_OK(model::GetLearner(train_config_, &learner, deployment_config_));

  auto model_or = learner->TrainWithStatus(train_dataset_);
  EXPECT_FALSE(model_or.ok());
  EXPECT_EQ(model_or.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(model_or.status().message(),
              HasSubstr("parallel_trials must be at least 1"));
}

TEST_F(OnAdult, EmptySearchSpace_Fail) {
  SetTrainConfig("RANDOM", "random", 5);
  auto* spe_config = train_config_.MutableExtension(
      hyperparameters_optimizer_v2::proto::hyperparameters_optimizer_config);
  spe_config->mutable_search_space()->clear_fields();

  SetLocalTraining();
  PrepareDataset();

  std::unique_ptr<model::AbstractLearner> learner;
  ASSERT_OK(model::GetLearner(train_config_, &learner, deployment_config_));

  auto model_or = learner->TrainWithStatus(train_dataset_);
  EXPECT_FALSE(model_or.ok());
  EXPECT_EQ(model_or.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(model_or.status().message(), HasSubstr("search space is empty"));
}

}  // namespace
}  // namespace hyperparameters_optimizer_v2
}  // namespace model
}  // namespace yggdrasil_decision_forests
