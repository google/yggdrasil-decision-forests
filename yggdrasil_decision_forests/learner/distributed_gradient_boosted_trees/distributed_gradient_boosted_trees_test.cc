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

#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.h"

#include <random>
#include <string>

#include "gmock/gmock.h"
#include "absl/debugging/leak_check.h"
#include "absl/strings/str_format.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/common.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/csv.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/test_utils.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_gradient_boosted_trees {
namespace {

using test::EqualsProto;

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
        # With 10 total workers, 2 are used for evaluation.
        ratio_evaluation_workers: 0.2
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

// Train and test a model on the adult dataset with workers continuously
// failing and requiring checkpoint restoration for both the training and
// validation workers.
TEST_F(DatasetAdult, BaseWithFailureAndManualValidation) {
  auto* spe_config = train_config_.MutableExtension(
      distributed_gradient_boosted_trees::proto::
          distributed_gradient_boosted_trees_config);
  spe_config->mutable_internal()->set_simulate_worker_failure(true);
  spe_config->set_checkpoint_interval_trees(5);
  pass_validation_dataset_ = true;
  TrainAndEvaluateModel();
  // Note: This result does not take early stopping into account.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8748, 0.012);
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

TEST_F(DatasetAdult, ManualValidation) {
  SetNumWorkers(10);
  pass_validation_dataset_ = true;
  TrainAndEvaluateModel();
  EXPECT_GT(metric::Accuracy(evaluation_), 0.863);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.2765, 0.04);

  auto* gbt_model =
      dynamic_cast<const gradient_boosted_trees::GradientBoostedTreesModel*>(
          model_.get());

  // Note: With early stopping, the non-distributed implementation of GBT has a
  // validation loss of 0.57404.
  EXPECT_NEAR(gbt_model->validation_loss(), 0.5859, 0.04);
  // (currently) There is not any early stopping.
  EXPECT_EQ(gbt_model->training_logs().number_of_trees_in_final_model(), 300);
  // (currently) There is one evaluation for each iteration.
  EXPECT_EQ(gbt_model->training_logs().entries_size(), 300);
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
  YDF_LOG(INFO) << "Training classical algorithm";
  TrainAndEvaluateModel();
  // Note: The description includes a details of the tree structure.
  auto* gbt_model =
      dynamic_cast<gradient_boosted_trees::GradientBoostedTreesModel*>(
          model_.get());
  gbt_model->mutable_training_logs()->Clear();
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
  YDF_LOG(INFO) << "Training distributed algorithm";
  TrainAndEvaluateModel();
  // Note: The description includes a details of the tree structure.
  auto* distributed_gbt_model =
      dynamic_cast<gradient_boosted_trees::GradientBoostedTreesModel*>(
          model_.get());
  distributed_gbt_model->mutable_training_logs()->Clear();
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
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.2762, 0.042);
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

TEST(ProtoIO, Base) {
  const decision_tree::proto::Node node_1 = PARSE_TEST_PROTO(R"pb(
    num_pos_training_examples_without_weight: 5
  )pb");
  const decision_tree::proto::Node node_2 = PARSE_TEST_PROTO(R"pb(
    num_pos_training_examples_without_weight: 10
  )pb");

  proto::WorkerRequest::EndIter::Tree dst;
  EndIterTreeProtoWriter writer(&dst);
  EXPECT_OK(writer.Write(node_1));
  EXPECT_OK(writer.Write(node_2));

  decision_tree::proto::Node read_node;
  EndIterTreeProtoReader reader(dst);
  EXPECT_TRUE(reader.Next(&read_node).value());
  EXPECT_THAT(read_node, EqualsProto(node_1));
  EXPECT_TRUE(reader.Next(&read_node).value());
  EXPECT_THAT(read_node, EqualsProto(node_2));
  EXPECT_FALSE(reader.Next(&read_node).value());
}

// Test training on 2.2B examples (i.e. more than 2^31). The test will fail if
// YDF is compiled with 32-bits example index.
//
// This test is disabled in presubmit as it requires a large amount of disk
// (~100GB) and RAM. The test takes ~15 minutes to run on a workstation. You can
// make the test run fast / require less resource by reducing the number of
// examples.
//
// The test can be run manually as follow:
//
// bazel test -c opt --test_strategy=local --test_output=streamed --test_ \
// arg=--alsologtostderr --copt=-mfma --copt=-mavx2 \
// --define=ydf_example_idx_num_bits=64 \
// //yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees:distributed_gradient_boosted_trees_test\
// --test_filter=LargeDataset.Base --test_timeout=1000000
//
// This test creates a large amount of data in the
// "/tmp/ydf_large_distributed_test" directory. Executing the test multiple time
// will re-use this data possibly skipping some of the computation (e.g.,
// creation of the synthetic dataset, creation of the dataset cache, training of
// the model) and run significantly faster.
//
// This test can be run on a large (but still smaller than 2B examples) dataset
// by decreasing the "num_shards" value.
TEST(DISABLED_LargeDataset, Base) {
  absl::LeakCheckDisabler disabler;

  // Create a large csv dataset.
  const size_t num_shards = 110;
  const size_t num_examples_per_shards = 20000000;  // 20M examples per shard.
  CHECK_GT(num_shards * num_examples_per_shards, size_t(1) << 31);

  const auto tmp_dir = "/tmp/ydf_large_distributed_test";

  const auto ds_dir = file::JoinPath(tmp_dir, "ds");
  const auto ds_path = file::JoinPath(ds_dir, absl::StrCat("ds@", num_shards));
  std::vector<std::string> shard_paths;
  file::GenerateShardedFilenames(ds_path, &shard_paths);
  if (file::FileExists(shard_paths.back()).value()) {
    YDF_LOG(INFO) << "Dataset already present";
  } else {
    YDF_LOG(INFO) << "Create dataset: " << ds_path;
    CHECK_OK(file::RecursivelyCreateDir(ds_dir, file::Defaults()));

    const auto create_shard = [&](const int shard_idx) {
      YDF_LOG(INFO) << "Write shard " << shard_idx << " of " << num_shards;
      const auto& path = shard_paths[shard_idx];
      auto writer = file::OpenOutputFile(path).value();
      std::minstd_rand0 rnd(shard_idx);
      std::uniform_real_distribution<float> unif_dist;
      CHECK_OK(writer->Write("f1,f2,l\n"));
      std::string buffer;
      for (int example_idx = 0; example_idx < num_examples_per_shards;
           example_idx++) {
        const float f1 = unif_dist(rnd);
        const float f2 = unif_dist(rnd);
        const float l = unif_dist(rnd);
        buffer.clear();
        // Note: f1 will be discretized, hence faster to train than f2.
        absl::StrAppendFormat(&buffer, "%.3f,%f,%.3f\n", f1, f2, l);
        CHECK_OK(writer->Write(buffer));
      }
      CHECK_OK(writer->Close());
    };

    {
      ThreadPool pool("create_dataset", /*num_threads=*/5);
      pool.StartWorkers();
      for (int shard_idx = 0; shard_idx < num_shards; shard_idx++) {
        pool.Schedule([shard_idx, create_shard]() { create_shard(shard_idx); });
      }
    }
  }

  YDF_LOG(INFO) << "Create dataspec";
  const auto dataspec =
      dataset::CreateDataSpec(absl::StrCat("csv:", shard_paths.front()))
          .value();
  std::string dataspec_report = dataset::PrintHumanReadable(dataspec);
  YDF_LOG(INFO) << "Dataspec:\n" << dataspec_report;

  YDF_LOG(INFO) << "Train model";
  model::proto::TrainingConfig train_config = PARSE_TEST_PROTO(R"pb(
    learner: "DISTRIBUTED_GRADIENT_BOOSTED_TREES"
    task: REGRESSION
    label: "l"
    [yggdrasil_decision_forests.model.distributed_gradient_boosted_trees.proto
         .distributed_gradient_boosted_trees_config] {
      checkpoint_interval_trees: 10
      worker_logs: true
      gbt { export_logs_during_training_in_trees: 10 num_trees: 2 }
      create_cache {
        # Significantly speed-up training of f2 feature.
        force_numerical_discretization: true
      }
    }
  )pb");

  model::proto::DeploymentConfig deploy_config =
      PARSE_TEST_PROTO(absl::Substitute(
          R"pb(
            num_threads: 4
            num_io_threads: 4
            try_resume_training: true
            cache_path: "$0"
            distribute {
              implementation_key: "MULTI_THREAD"
              [yggdrasil_decision_forests.distribute.proto.multi_thread] {
                num_workers: 2
              }
            }
          )pb",
          file::JoinPath(tmp_dir, "cache")));

  const auto learner = model::GetLearner(train_config, deploy_config,
                                         file::JoinPath(tmp_dir, "logs"))
                           .value();
  auto model =
      learner->TrainWithStatus(absl::StrCat("csv:", ds_path), dataspec).value();

  YDF_LOG(INFO) << "Model trained";
  YDF_LOG(INFO) << model->DescriptionAndStatistics(true);
}

}  // namespace
}  // namespace distributed_gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
