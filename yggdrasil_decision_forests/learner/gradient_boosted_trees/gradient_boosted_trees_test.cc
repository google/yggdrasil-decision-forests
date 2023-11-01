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

// Gradient Boosted Trees algorithms tests.

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/btree_set.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/early_stopping/early_stopping.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_library.h"
#include "yggdrasil_decision_forests/learner/hyperparameters_optimizer/hyperparameters_optimizer.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/csv.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/model_analysis.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/snapshot.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/test_utils.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

using test::EqualsProto;
using ::testing::ElementsAre;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

std::string DatasetDir() {
  return file::JoinPath(
      test::DataRootDirectory(),
      "yggdrasil_decision_forests/test_data/dataset");
}

absl::StatusOr<dataset::VerticalDataset> CreateToyDataset() {
  dataset::VerticalDataset dataset;
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "a" }
    columns {
      type: CATEGORICAL
      name: "b"
      categorical { number_of_unique_values: 3 is_already_integerized: true }
    }
  )pb");
  RETURN_IF_ERROR(dataset.CreateColumnsFromDataspec());
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "1"}, {"b", "1"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "2"}, {"b", "2"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "3"}, {"b", "1"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "4"}, {"b", "2"}}));
  return dataset;
}

// Shards a dataset. Returns the sharded path in the temp directory.
std::string ShardDataset(const dataset::VerticalDataset& dataset,
                         const int num_shards, const float sampling) {
  const auto sharded_dir = file::JoinPath(test::TmpDirectory(), "sharded");
  const auto sharded_path =
      file::JoinPath(sharded_dir, absl::StrCat("dataset@", num_shards));
  const auto typed_sharded_path = absl::StrCat("csv:", sharded_path);
  CHECK_OK(file::RecursivelyCreateDir(sharded_dir, file::Defaults()));
  std::vector<std::string> shards;
  CHECK_OK(utils::ExpandOutputShards(sharded_path, &shards));

  // Down-sample the number of examples.
  std::vector<UnsignedExampleIdx> examples(dataset.nrow());
  std::iota(examples.begin(), examples.end(), 0);
  std::mt19937 rnd;
  std::shuffle(examples.begin(), examples.end(), rnd);
  examples.resize(std::lround(sampling * dataset.nrow()));

  for (int shard_idx = 0; shard_idx < num_shards; shard_idx++) {
    std::vector<UnsignedExampleIdx> idxs;
    for (int i = shard_idx; i < examples.size(); i += num_shards) {
      idxs.push_back(examples[i]);
    }
    CHECK_OK(
        dataset::SaveVerticalDataset(dataset.Extract(idxs).value(),
                                     absl::StrCat("csv:", shards[shard_idx])));
  }
  return typed_sharded_path;
}

TEST(GradientBoostedTrees, ExtractValidationDataset) {
  const std::string ds_typed_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "adult.csv"));
  dataset::proto::DataSpecification data_spec;
  dataset::proto::DataSpecificationGuide guide;
  dataset::CreateDataSpec(ds_typed_path, false, guide, &data_spec);
  dataset::VerticalDataset dataset;
  CHECK_OK(LoadVerticalDataset(ds_typed_path, data_spec, &dataset));

  utils::RandomEngine random(1234);

  dataset::VerticalDataset training_1;
  dataset::VerticalDataset validation_1;
  CHECK_OK(internal::ExtractValidationDataset(dataset, 0.5f, -1, &training_1,
                                              &validation_1, &random));
  EXPECT_NEAR(training_1.nrow(), 0.5f * dataset.nrow(), dataset.nrow() * 0.05);
  EXPECT_EQ(training_1.nrow() + validation_1.nrow(), dataset.nrow());
  EXPECT_TRUE(training_1.OwnsColumn(0));

  dataset::VerticalDataset training_2;
  dataset::VerticalDataset validation_2;
  CHECK_OK(internal::ExtractValidationDataset(dataset, 0.0f, {}, &training_2,
                                              &validation_2, &random));
  EXPECT_EQ(training_2.nrow(), dataset.nrow());
  EXPECT_EQ(validation_2.nrow(), 0);
  EXPECT_FALSE(training_2.OwnsColumn(0));
}

TEST(GradientBoostedTrees, ExtractValidationDatasetWithGroup) {
  const std::string ds_typed_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "adult.csv"));
  dataset::proto::DataSpecification data_spec;
  dataset::proto::DataSpecificationGuide guide;
  dataset::CreateDataSpec(ds_typed_path, false, guide, &data_spec);
  dataset::VerticalDataset dataset;
  CHECK_OK(LoadVerticalDataset(ds_typed_path, data_spec, &dataset));

  utils::RandomEngine random(1234);

  const int group_feature_idx =
      dataset::GetColumnIdxFromName("education", data_spec);

  const float validation_set_ratio = 0.2f;
  dataset::VerticalDataset training;
  dataset::VerticalDataset validation;
  CHECK_OK(internal::ExtractValidationDataset(dataset, validation_set_ratio,
                                              group_feature_idx, &training,
                                              &validation, &random));
  // Note: The most popular group contains ~32% of the dataset.
  EXPECT_NEAR(training.nrow(), (1.f - validation_set_ratio) * dataset.nrow(),
              dataset.nrow() * 0.20);
  EXPECT_EQ(training.nrow() + validation.nrow(), dataset.nrow());
  EXPECT_TRUE(training.OwnsColumn(0));

  const auto train_group =
      training
          .ColumnWithCastWithStatus<
              dataset::VerticalDataset::CategoricalColumn>(group_feature_idx)
          .value();
  const auto validation_group =
      validation
          .ColumnWithCastWithStatus<
              dataset::VerticalDataset::CategoricalColumn>(group_feature_idx)
          .value();

  // Ensure the intersection of the groups is empty.
  absl::btree_set<int> train_group_values(train_group->values().begin(),
                                          train_group->values().end());
  absl::btree_set<int> validation_group_values(
      validation_group->values().begin(), validation_group->values().end());
  std::vector<int> group_intersection;
  std::set_intersection(train_group_values.begin(), train_group_values.end(),
                        validation_group_values.begin(),
                        validation_group_values.end(),
                        std::back_inserter(group_intersection));
  EXPECT_TRUE(group_intersection.empty());
}

TEST(GradientBoostedTrees, CreateGradientDataset) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;

  const auto loss_imp = CreateLoss(proto::Loss::BINOMIAL_LOG_LIKELIHOOD,
                                   model::proto::Task::CLASSIFICATION,
                                   dataset.data_spec().columns(1), {})
                            .value();
  CHECK_OK(internal::CreateGradientDataset(dataset,
                                           /* label_col_idx= */ 1,
                                           /*hessian_splits=*/false, *loss_imp,
                                           &gradient_dataset, &gradients,
                                           &predictions));
  EXPECT_EQ(gradient_dataset.nrow(), 4);
  const dataset::proto::DataSpecification expected_data_spec = PARSE_TEST_PROTO(
      R"pb(
        columns { type: NUMERICAL name: "a" }
        columns {
          type: CATEGORICAL
          name: "b"
          categorical {
            number_of_unique_values: 3
            is_already_integerized: true
          }
        }
        columns { type: NUMERICAL name: "__gradient__0" }
        columns { type: NUMERICAL name: "__hessian__0" }
      )pb");
  EXPECT_THAT(gradient_dataset.data_spec(), EqualsProto(expected_data_spec));

  EXPECT_EQ(gradients.size(), 1);
  EXPECT_EQ(gradients[0].gradient_column_name, "__gradient__0");
  EXPECT_EQ(gradients[0].gradient.size(), 4);
  EXPECT_EQ(predictions.size(), 4);
}

TEST(GradientBoostedTrees, SetInitialPredictions) {
  std::vector<float> predictions;
  internal::SetInitialPredictions({1, 2}, 3, &predictions);
  EXPECT_THAT(predictions, ElementsAre(1, 2, 1, 2, 1, 2));
}

TEST(GradientBoostedTrees, SampleTrainingExamplesWithGoss) {
  const UnsignedExampleIdx num_rows = 4;
  std::vector<float> weights(num_rows, 1.f);

  std::vector<float> dim1_values = {0.8, 2.0, -0.1, -3.2};
  std::vector<float> hessian_values = {1, 1, 1, 1, 1};
  GradientData dim1{.gradient = dim1_values, .hessian = hessian_values};
  std::vector<GradientData> gradients = {dim1};

  utils::RandomEngine random(1234);
  std::vector<UnsignedExampleIdx> selected_examples;

  internal::SampleTrainingExamplesWithGoss(gradients, num_rows, /*alpha=*/1.,
                                           /*beta=*/0., &random,
                                           &selected_examples, &weights);
  EXPECT_THAT(selected_examples, ElementsAre(3, 1, 0, 2));
  EXPECT_THAT(weights, ElementsAre(1, 1, 1, 1));

  selected_examples.clear();
  std::fill(weights.begin(), weights.end(), 1.f);
  internal::SampleTrainingExamplesWithGoss(gradients, num_rows, /*alpha=*/0.2,
                                           /*beta=*/0., &random,
                                           &selected_examples, &weights);
  EXPECT_THAT(selected_examples, ElementsAre(3));
  EXPECT_THAT(weights, ElementsAre(1, 1, 1, 1));

  selected_examples.clear();
  std::fill(weights.begin(), weights.end(), 1.f);
  internal::SampleTrainingExamplesWithGoss(gradients, num_rows, /*alpha=*/0.5,
                                           /*beta=*/0.2, &random,
                                           &selected_examples, &weights);
  EXPECT_THAT(selected_examples, ElementsAre(3, 1, 0));
  EXPECT_THAT(weights, ElementsAre(2.5, 1, 1, 1));
}

TEST(GradientBoostedTrees, SampleTrainingExamplesWithSelGB) {
  dataset::VerticalDataset dataset;
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "a" }
    columns {
      type: CATEGORICAL
      name: "b"
      categorical { number_of_unique_values: 3 is_already_integerized: true }
    }
  )pb");
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "1"}, {"b", "1"}}));
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "2"}, {"b", "2"}}));
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "0"}, {"b", "2"}}));
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "3"}, {"b", "1"}}));
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "0"}, {"b", "2"}}));
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "4"}, {"b", "2"}}));
  CHECK_OK(dataset.AppendExampleWithStatus({{"a", "0"}, {"b", "2"}}));

  RankingGroupsIndices index;
  EXPECT_OK(index.Initialize(dataset, 0, 1));
  EXPECT_EQ(index.groups().size(), 2);

  std::vector<float> predictions = {0.8, -0.1, -4.0, 2.0, 1.2, -3.2, -0.3};
  std::vector<UnsignedExampleIdx> selected_examples;

  CHECK_OK(internal::SampleTrainingExamplesWithSelGB(
      model::proto::Task::RANKING, dataset.nrow(), &index, predictions,
      /*ratio=*/1., &selected_examples));
  EXPECT_THAT(selected_examples, ElementsAre(0, 1, 2, 3, 4, 5, 6));

  selected_examples.clear();
  CHECK_OK(internal::SampleTrainingExamplesWithSelGB(
      model::proto::Task::RANKING, dataset.nrow(), &index, predictions,
      /*ratio=*/0., &selected_examples));
  EXPECT_THAT(selected_examples, ElementsAre(3, 0, 5, 1));

  selected_examples.clear();
  CHECK_OK(internal::SampleTrainingExamplesWithSelGB(
      model::proto::Task::RANKING, dataset.nrow(), &index, predictions,
      /*ratio=*/0.1, &selected_examples));
  EXPECT_THAT(selected_examples, ElementsAre(3, 0, 5, 1, 4));
}

// Helper for the training and testing on two non-overlapping samples from the
// adult dataset.
class GradientBoostedTreesOnAdult : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(GradientBoostedTreesLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    train_config_.set_label("income");
    dataset_filename_ = "adult.csv";
    dataset_sampling_ = 0.2f;
    deployment_config_.set_num_threads(1);
    auto* gbt_config = train_config_.MutableExtension(
        gradient_boosted_trees::proto::gradient_boosted_trees_config);
    gbt_config->mutable_decision_tree()
        ->set_internal_error_on_wrong_splitter_statistics(true);
  }
};

// Train and test a model on the adult dataset.
TEST_F(GradientBoostedTreesOnAdult, BaseDeprecated) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->set_subsample(0.9f);
  TrainAndEvaluateModel();

  // Note: Accuracy is similar as RF (see :random_forest_test). However logloss
  // is significantly better (which is expected as, unlike RF,  GBT is
  // calibrated).
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8644, 0.0099, 0.8658);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2979, 0.0127, 0.294);

  auto* gbt_model =
      dynamic_cast<const GradientBoostedTreesModel*>(model_.get());
  EXPECT_TRUE(gbt_model->CheckStructure(
      decision_tree::CheckStructureOptions::GlobalImuptation()));
}

// Train and test a model on the adult dataset.
TEST_F(GradientBoostedTreesOnAdult, Base) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->mutable_stochastic_gradient_boosting()->set_ratio(0.9f);
  TrainAndEvaluateModel();

  // Note: Accuracy is similar as RF (see :random_forest_test). However, logloss
  // is significantly better (which is expected as, unlike RF, GBT is
  // calibrated).
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8647, 0.0099, 0.8658);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2984, 0.0162, 0.294);

  auto* gbt_model =
      dynamic_cast<const GradientBoostedTreesModel*>(model_.get());
  EXPECT_TRUE(gbt_model->CheckStructure(
      decision_tree::CheckStructureOptions::GlobalImuptation()));
}

TEST_F(GradientBoostedTreesOnAdult, MonotonicConstraints) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);

  gbt_config->set_use_hessian_gain(true);

  // Ensures that the tree is monotonic.
  gbt_config->mutable_decision_tree()
      ->mutable_internal()
      ->set_check_monotonic_constraints(true);

  auto* constrain_1 = train_config_.mutable_monotonic_constraints()->Add();
  constrain_1->set_feature("^age$");
  constrain_1->set_direction(model::proto::MonotonicConstraint::INCREASING);

  auto* constrain_2 = train_config_.mutable_monotonic_constraints()->Add();
  constrain_2->set_feature("^hours_per_week$");
  constrain_2->set_direction(model::proto::MonotonicConstraint::INCREASING);

  auto* constrain_3 = train_config_.mutable_monotonic_constraints()->Add();
  constrain_3->set_feature("^education_num$");
  constrain_3->set_direction(model::proto::MonotonicConstraint::INCREASING);

  TrainAndEvaluateModel();

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8641, 0.009, 0.8627);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2972, 0.0181, 0.2902);

  // Show the tree structure.
  std::string description;
  model_->AppendDescriptionAndStatistics(true, &description);
  YDF_LOG(INFO) << description;

  QCHECK_OK(utils::model_analysis::AnalyseAndCreateHtmlReport(
      *model_, test_dataset_, "", "",
      file::JoinPath(test::TmpDirectory(), "analysis"), {}));
}

TEST_F(GradientBoostedTreesOnAdult, DecreasingMonotonicConstraints) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);

  gbt_config->set_use_hessian_gain(true);

  // Ensures that the tree is monotonic.
  gbt_config->mutable_decision_tree()
      ->mutable_internal()
      ->set_check_monotonic_constraints(true);

  auto* constrain_1 = train_config_.mutable_monotonic_constraints()->Add();
  constrain_1->set_feature("^age$");
  constrain_1->set_direction(model::proto::MonotonicConstraint::DECREASING);

  auto* constrain_2 = train_config_.mutable_monotonic_constraints()->Add();
  constrain_2->set_feature("^hours_per_week$");
  constrain_2->set_direction(model::proto::MonotonicConstraint::INCREASING);

  auto* constrain_3 = train_config_.mutable_monotonic_constraints()->Add();
  constrain_3->set_feature("^education_num$");
  constrain_3->set_direction(model::proto::MonotonicConstraint::DECREASING);

  TrainAndEvaluateModel();

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8586, 0.0094, 0.8587);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.309, 0.0156, 0.3028);

  // Show the tree structure.
  std::string description;
  model_->AppendDescriptionAndStatistics(true, &description);
  YDF_LOG(INFO) << description;

  QCHECK_OK(utils::model_analysis::AnalyseAndCreateHtmlReport(
      *model_, test_dataset_, "", "",
      file::JoinPath(test::TmpDirectory(), "analysis"), {}));
}

TEST_F(GradientBoostedTreesOnAdult, ObliqueMonotonicConstraints) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);

  gbt_config->mutable_decision_tree()->mutable_sparse_oblique_split();

  gbt_config->set_use_hessian_gain(true);

  auto* constrain_1 = train_config_.mutable_monotonic_constraints()->Add();
  constrain_1->set_feature("^age$");
  constrain_1->set_direction(model::proto::MonotonicConstraint::INCREASING);

  auto* constrain_2 = train_config_.mutable_monotonic_constraints()->Add();
  constrain_2->set_feature("^hours_per_week$");
  constrain_2->set_direction(model::proto::MonotonicConstraint::INCREASING);

  auto* constrain_3 = train_config_.mutable_monotonic_constraints()->Add();
  constrain_3->set_feature("^education_num$");
  constrain_3->set_direction(model::proto::MonotonicConstraint::INCREASING);

  TrainAndEvaluateModel();

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.861, 0.009, 0.8596);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3049, 0.0195, 0.3057);

  // Show the tree structure.
  std::string description;
  model_->AppendDescriptionAndStatistics(true, &description);
  YDF_LOG(INFO) << description;

  QCHECK_OK(utils::model_analysis::AnalyseAndCreateHtmlReport(
      *model_, test_dataset_, "", "",
      file::JoinPath(test::TmpDirectory(), "analysis"), {}));
}

// Train and test a model on the adult dataset with focal loss, but with gamma
// equals zero, which equals to log loss.
TEST_F(GradientBoostedTreesOnAdult, FocalLossWithGammaZero) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_loss(proto::Loss::BINARY_FOCAL_LOSS);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->mutable_stochastic_gradient_boosting()->set_ratio(0.9f);
  gbt_config->mutable_binary_focal_loss_options()->set_misprediction_exponent(
      0.0f);
  TrainAndEvaluateModel();

  // Similar metrics as with log loss.
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8647, 0.0122, 0.8658);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2969, 0.0107, 0.2949);
}

// Train and test a model on the adult dataset with focal loss, now with
// an effective gamma of 0.5.
TEST_F(GradientBoostedTreesOnAdult, FocalLossWithGammaHalf) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_loss(proto::Loss::BINARY_FOCAL_LOSS);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->mutable_stochastic_gradient_boosting()->set_ratio(0.9f);
  gbt_config->mutable_binary_focal_loss_options()->set_misprediction_exponent(
      0.5f);
  TrainAndEvaluateModel();

  // Slightly better accuracy, but worse log loss; we are not
  // optimizing for log loss directly any more.
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8653, 0.0094, 0.8624);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3226, 0.0218, 0.3145);
}

// Train and test a model on the adult dataset with focal loss, now with
// an effective gamma of 2.
TEST_F(GradientBoostedTreesOnAdult, FocalLossWithGammaTwo) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_loss(proto::Loss::BINARY_FOCAL_LOSS);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->mutable_stochastic_gradient_boosting()->set_ratio(0.9f);
  gbt_config->mutable_binary_focal_loss_options()->set_misprediction_exponent(
      2.0f);
  TrainAndEvaluateModel();

  // Even slightly better accuracy (could be just noise, but illustrative),
  // log loss deviates even more
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8638, 0.0094, 0.8643);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.4107, 0.0351, 0.3924);
}

// Train and test a model on the adult dataset with focal loss, adding a
// non-default, 0.25 alpha parameter to the gamma of 2.0
TEST_F(GradientBoostedTreesOnAdult, FocalLossWithGammaTwoAlphaQuarter) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_loss(proto::Loss::BINARY_FOCAL_LOSS);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->mutable_stochastic_gradient_boosting()->set_ratio(0.9f);
  gbt_config->mutable_binary_focal_loss_options()->set_misprediction_exponent(
      2.0f);
  gbt_config->mutable_binary_focal_loss_options()
      ->set_positive_sample_coefficient(0.25f);
  TrainAndEvaluateModel();

  // Worse accuracy but smaller log loss due to low alpha
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8503, 0.0177, 0.8553);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3907, 0.036, 0.3753);
}

// Separate the examples used for the structure and the leaves of the model.
TEST_F(GradientBoostedTreesOnAdult, Honest) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->mutable_decision_tree()->mutable_honest();
  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8589, 0.0131, 0.8609);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3095, 0.015, 0.3075);
}
// Train a GBT with a validation dataset provided as a VerticalDataset.
TEST_F(GradientBoostedTreesOnAdult, ValidVerticalDataset) {
  pass_validation_dataset_ = true;
  inject_random_noise_ = true;
  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8732, 0.0023, 0.8747);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2794, 0.0027, 0.2776);
}

// Train a GBT with a validation dataset provided as a path.
TEST_F(GradientBoostedTreesOnAdult, ValidPathDataset) {
  pass_training_dataset_as_path_ = true;
  pass_validation_dataset_ = true;
  inject_random_noise_ = true;
  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8732, 0.0023, 0.8747);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2794, 0.0057, 0.2776);
}

TEST_F(GradientBoostedTreesOnAdult, DISABLED_VariableImportance) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_compute_permutation_variable_importance(true);

  TrainAndEvaluateModel();

  EXPECT_THAT(model_->AvailableVariableImportances(),
              UnorderedElementsAre("MEAN_DECREASE_IN_ACCURACY",
                                   "MEAN_DECREASE_IN_AP_>50K_VS_OTHERS",
                                   "MEAN_DECREASE_IN_AUC_>50K_VS_OTHERS",
                                   "MEAN_DECREASE_IN_PRAUC_>50K_VS_OTHERS",
                                   "INV_MEAN_MIN_DEPTH", "NUM_AS_ROOT",
                                   "NUM_NODES", "SUM_SCORE"));

  const auto mean_decrease_accuracy =
      model_->GetVariableImportance("MEAN_DECREASE_IN_ACCURACY").value();

  // Top 3 variables.
  const int rank_capital_gain = utils::GetVariableImportanceRank(
      "capital_gain", model_->data_spec(), mean_decrease_accuracy);
  const int rank_relationship = utils::GetVariableImportanceRank(
      "relationship", model_->data_spec(), mean_decrease_accuracy);
  const int rank_occupation = utils::GetVariableImportanceRank(
      "occupation", model_->data_spec(), mean_decrease_accuracy);

  EXPECT_LE(rank_capital_gain, 3);
  EXPECT_LE(rank_relationship, 3);
  EXPECT_LE(rank_occupation, 7);
}

class PerShardSamplingOnAdult : public ::testing::Test {
 public:
  void SetUp() override {
    // Load the datasets.
    const auto train_ds_path = absl::StrCat(
        "csv:", file::JoinPath(
                    test::DataRootDirectory(),
                    "yggdrasil_decision_forests/test_data/dataset/"
                    "adult_train.csv"));
    const auto test_ds_path = absl::StrCat(
        "csv:", file::JoinPath(
                    test::DataRootDirectory(),
                    "yggdrasil_decision_forests/test_data/dataset/"
                    "adult_test.csv"));
    dataset::CreateDataSpec(train_ds_path, false, {}, &data_spec_);
    CHECK_OK(LoadVerticalDataset(train_ds_path, data_spec_, &train_ds_));
    CHECK_OK(LoadVerticalDataset(test_ds_path, data_spec_, &test_ds_));
  }

  std::unique_ptr<model::AbstractLearner> BuildBaseLearner() {
    // Configure model training.
    model::proto::DeploymentConfig deployment_config;
    model::proto::TrainingConfig train_config;
    train_config.set_learner(GradientBoostedTreesLearner::kRegisteredName);
    train_config.set_task(model::proto::Task::CLASSIFICATION);
    train_config.set_label("income");
    train_config.add_features(".*");
    std::unique_ptr<model::AbstractLearner> learner;
    CHECK_OK(model::GetLearner(train_config, &learner, deployment_config));
    return learner;
  }

  dataset::proto::DataSpecification data_spec_;
  dataset::VerticalDataset train_ds_;
  dataset::VerticalDataset test_ds_;
};

// Training a model with the shard sampler algorithm, but with all the shards
// used for each tree.
TEST_F(PerShardSamplingOnAdult, PerShardSamplingExact) {
  auto learner = BuildBaseLearner();
  auto* gbt_config = learner->mutable_training_config()->MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);

  // Shard the training dataset.
  const auto sharded_path = ShardDataset(train_ds_, 20, 0.3);

  YDF_LOG(INFO) << "Train sharded model";
  gbt_config->mutable_sample_with_shards();

  const auto model = learner->TrainWithStatus(sharded_path, data_spec_).value();

  YDF_LOG(INFO) << "Evaluate models";
  // Evaluate the models.
  utils::RandomEngine rnd(1234);
  const auto evaluation = model->Evaluate(test_ds_, {}, &rnd);
  YDF_LOG(INFO) << "Evaluation:" << metric::TextReport(evaluation).value();

  // Sharded model is "good".
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  YDF_TEST_METRIC(metric::Accuracy(evaluation), 0.8667, 0.008, nan);
}

// Model trained with the sharded algorithm and sampling.
TEST_F(PerShardSamplingOnAdult, PerShardSamplingSampling) {
  auto learner = BuildBaseLearner();
  auto* gbt_config = learner->mutable_training_config()->MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);

  // Shard the training dataset.
  const auto sharded_path = ShardDataset(train_ds_, 20, 1.0);

  // Model trained with the sharded algorithm and sampling.
  gbt_config->mutable_sample_with_shards();
  gbt_config->set_subsample(0.1f);
  const auto sharded_sampled_model =
      learner->TrainWithStatus(sharded_path, data_spec_).value();

  // Evaluate the models.
  utils::RandomEngine rnd(1234);
  const auto eval = sharded_sampled_model->Evaluate(test_ds_, {}, &rnd);

  YDF_TEST_METRIC(metric::Accuracy(eval), 0.8633, 0.006, 0.8633);
}

// Model trained with the sharded algorithm and sampling.
TEST_F(PerShardSamplingOnAdult, PerShardSamplingSamplingRecycle) {
  auto learner = BuildBaseLearner();
  auto* gbt_config = learner->mutable_training_config()->MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);

  // Shard the training dataset.
  const auto sharded_path = ShardDataset(train_ds_, 20, 1.0);

  // Model trained with the sharded algorithm and sampling.
  gbt_config->set_subsample(0.1f);
  gbt_config->mutable_sample_with_shards()->set_num_recycling(5);
  const auto sharded_sampled_model =
      learner->TrainWithStatus(sharded_path, data_spec_).value();

  // Evaluate the models.
  utils::RandomEngine rnd(1234);
  const auto eval = sharded_sampled_model->Evaluate(test_ds_, {}, &rnd);

  YDF_TEST_METRIC(metric::Accuracy(eval), 0.8589, 0.005, 0.8589);
}

// Train and test a model on the adult dataset using random categorical splits.
TEST_F(GradientBoostedTreesOnAdult, RandomCategorical) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->mutable_stochastic_gradient_boosting()->set_ratio(0.9f);
  gbt_config->mutable_decision_tree()->mutable_categorical()->mutable_random();
  TrainAndEvaluateModel();

  // Note: Accuracy is similar as RF (see :random_forest_test). However logloss
  // is significantly better (which is expected as, unlike RF,  GBT is
  // calibrated).
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8642, 0.0097, 0.863);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2954, 0.0095, 0.294);

  auto* gbt_model =
      dynamic_cast<const GradientBoostedTreesModel*>(model_.get());
  EXPECT_TRUE(gbt_model->CheckStructure(
      decision_tree::CheckStructureOptions::GlobalImuptation()));
}

// Train and test a model on the adult dataset with too much nodes for the
// QuickScorer serving algorithm.
TEST_F(GradientBoostedTreesOnAdult, BaseNoQuickScorer) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(10);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->set_subsample(0.9f);
  TrainAndEvaluateModel();

  // Note: Accuracy is similar as RF (see :random_forest_test). However logloss
  // is significantly better (which is expected as, unlike RF,  GBT is
  // calibrated).
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8596, 0.0134, 0.8569);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3146, 0.0212, 0.3111);

  auto* gbt_model =
      dynamic_cast<const GradientBoostedTreesModel*>(model_.get());
  EXPECT_TRUE(gbt_model->CheckStructure(
      decision_tree::CheckStructureOptions::GlobalImuptation()));
}

// Train and test a model on the adult dataset.
TEST_F(GradientBoostedTreesOnAdult, BaseConcurrentDeprecated) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->set_subsample(1.0f);
  deployment_config_.set_num_threads(4);
  TrainAndEvaluateModel();

  // Note: Accuracy is similar as RF (see :random_forest_test). However logloss
  // is significantly better (which is expected as, unlike RF,  GBT is
  // calibrated).
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8662, 0.0094, 0.8664);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2966, 0.0145, 0.2942);
}

// Train and test a model on the adult dataset.
TEST_F(GradientBoostedTreesOnAdult, BaseConcurrent) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->mutable_stochastic_gradient_boosting()->set_ratio(1.0f);
  deployment_config_.set_num_threads(4);
  TrainAndEvaluateModel();

  // Note: Accuracy is similar as RF (see :random_forest_test). However logloss
  // is significantly better (which is expected as, unlike RF,  GBT is
  // calibrated).
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8656, 0.0094, 0.8664);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.296, 0.0117, 0.2942);
}

// Train and test a model on the adult dataset with Goss sampling.
TEST_F(GradientBoostedTreesOnAdult, GossDeprecated) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->set_use_goss(true);
  TrainAndEvaluateModel();

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.86, 0.012, 0.86);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3106, 0.0168, 0.3074);
}

// Train and test a model on the adult dataset with Goss sampling.
TEST_F(GradientBoostedTreesOnAdult, Goss) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->mutable_gradient_one_side_sampling();
  TrainAndEvaluateModel();

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8601, 0.0127, 0.86);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3095, 0.0138, 0.3074);
}

// Train and test a model on the adult dataset.
TEST_F(GradientBoostedTreesOnAdult, BaseDiscretizedNumerical) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->set_subsample(0.9f);
  guide_.set_detect_numerical_as_discretized_numerical(true);

  TrainAndEvaluateModel();

  // Note: Accuracy is similar as RF (see :random_forest_test). However logloss
  // is significantly better (which is expected as, unlike RF,  GBT is
  // calibrated).
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8649, 0.0097, 0.8618);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2986, 0.0148, 0.2957);
}

// Train and test a model on the adult dataset.
TEST_F(GradientBoostedTreesOnAdult, BaseAggressiveDiscretizedNumerical) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->set_subsample(0.9f);
  guide_.set_detect_numerical_as_discretized_numerical(true);
  guide_.mutable_default_column_guide()
      ->mutable_discretized_numerical()
      ->set_maximum_num_bins(16);

  TrainAndEvaluateModel();

  // Note: Accuracy is similar as RF (see :random_forest_test). However logloss
  // is significantly better (which is expected as, unlike RF,  GBT is
  // calibrated).
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8607, 0.0131, nan);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3099, 0.0183, nan);
}

TEST_F(GradientBoostedTreesOnAdult, BaseWithWeights) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->set_subsample(0.9f);

  TrainAndEvaluateModel(/*numerical_weight_attribute=*/"age");

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8388, 0.0146, 0.8375);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3614, 0.0313, 0.3533);
}

TEST_F(GradientBoostedTreesOnAdult, NumCandidateAttributeRatio) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->mutable_decision_tree()->set_num_candidate_attributes_ratio(0.5f);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->set_subsample(0.9f);

  TrainAndEvaluateModel();

  // Note: Accuracy is similar as RF (see :random_forest_test). However logloss
  // is significantly better (which is expected as, unlike RF,  GBT is
  // calibrated).
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8644, 0.0108, 0.8649);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3011, 0.0151, 0.2972);
}

// Train and test a model on the adult dataset.
TEST_F(GradientBoostedTreesOnAdult, LeafWiseGrow) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->set_subsample(0.9f);
  gbt_config->mutable_decision_tree()
      ->mutable_growing_strategy_best_first_global();

  TrainAndEvaluateModel();

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8662, 0.0094, 0.8646);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3026, 0.0218, 0.2958);
}

// Train and test a model on the adult dataset with L2 regularization.
TEST_F(GradientBoostedTreesOnAdult, L2Regularization) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->set_subsample(0.9f);
  gbt_config->set_l2_regularization(0.1f);

  TrainAndEvaluateModel();

  // Note: Accuracy is similar as RF (see :random_forest_test). However logloss
  // is significantly better (which is expected as, unlike RF,  GBT is
  // calibrated).
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8639, 0.0097, 0.8621);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2977, 0.011, 0.2952);
}

// Multiclass version of the algorithm on the binary class adult dataset.
TEST_F(GradientBoostedTreesOnAdult, FakeMulticlass) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_loss(proto::Loss::MULTINOMIAL_LOG_LIKELIHOOD);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->set_subsample(0.9f);

  TrainAndEvaluateModel();

  // Note: As expected, the results are similar to the binary class
  // implementation.
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8641, 0.0099, 0.8646);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2979, 0.0126, 0.2966);
}

// Multiclass version of the algorithm on the binary class adult dataset with L2
// regularization.
TEST_F(GradientBoostedTreesOnAdult, FakeMulticlassL2Regularization) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_loss(proto::Loss::MULTINOMIAL_LOG_LIKELIHOOD);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_shrinkage(0.1f);
  gbt_config->set_subsample(0.9f);
  gbt_config->set_l2_regularization(0.1f);

  TrainAndEvaluateModel();

  // Note: As expected, the results are similar to the binary class
  // implementation.
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8649, 0.0092, 0.8725);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3002, 0.0152, 0.2953);
}

// Train and test a model on the adult dataset for a maximum given duration.
TEST_F(GradientBoostedTreesOnAdult, MaximumDuration) {
  dataset_sampling_ = 1.0f;
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100000);  // Would train for a long time.
  train_config_.set_maximum_training_duration_seconds(10);

  TrainAndEvaluateModel();
  // Note: The "TrainAndEvaluateModel" function last a bit more because it is
  // also preparing the dataset and evaluating the final model.
#if !defined(MEMORY_SANITIZER) && !defined(THREAD_SANITIZER) && \
    !defined(ADDRESS_SANITIZER) && !defined(SKIP_TIMING_TESTS)
  EXPECT_LE(absl::ToDoubleSeconds(training_duration_), 15);
  EXPECT_GE(metric::Accuracy(evaluation_), 0.840);
#endif
}

TEST_F(GradientBoostedTreesOnAdult, MaximumDurationInTreeLocalGrowth) {
  dataset_sampling_ = 1.0f;
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(1);
  // Would take a very long time without a timeout inside of the tree building.
  gbt_config->mutable_decision_tree()
      ->mutable_sparse_oblique_split()
      ->set_max_num_projections(10000);
  gbt_config->mutable_decision_tree()
      ->mutable_sparse_oblique_split()
      ->set_num_projections_exponent(10);

  const double kMaximumGrowthDurationSec = 10;
  train_config_.set_maximum_training_duration_seconds(
      kMaximumGrowthDurationSec);

  TrainAndEvaluateModel();
  // Note: The "TrainAndEvaluateModel" function last a bit more because it is
  // also preparing the dataset and evaluating the final model.
#if !defined(MEMORY_SANITIZER) && !defined(THREAD_SANITIZER) && \
    !defined(ADDRESS_SANITIZER) && !defined(SKIP_TIMING_TESTS)
  EXPECT_LE(absl::ToDoubleSeconds(training_duration_),
            2 * kMaximumGrowthDurationSec);
#endif
}

TEST_F(GradientBoostedTreesOnAdult, MaximumDurationInTreeGlobalGrowth) {
  dataset_sampling_ = 1.0f;
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->mutable_decision_tree()
      ->mutable_growing_strategy_best_first_global();
  gbt_config->set_num_trees(1);
  // Would take a very long time without a timeout inside of the tree building.
  gbt_config->mutable_decision_tree()
      ->mutable_sparse_oblique_split()
      ->set_max_num_projections(10000);
  gbt_config->mutable_decision_tree()
      ->mutable_sparse_oblique_split()
      ->set_num_projections_exponent(10);

  const double kMaximumGrowthDurationSec = 10;
  train_config_.set_maximum_training_duration_seconds(
      kMaximumGrowthDurationSec);

  TrainAndEvaluateModel();
  // Note: The "TrainAndEvaluateModel" function last a bit more because it is
  // also preparing the dataset and evaluating the final model.
#if !defined(MEMORY_SANITIZER) && !defined(THREAD_SANITIZER) && \
    !defined(ADDRESS_SANITIZER) && !defined(SKIP_TIMING_TESTS)
  EXPECT_LE(absl::ToDoubleSeconds(training_duration_),
            3 * kMaximumGrowthDurationSec);
#endif
}

TEST_F(GradientBoostedTreesOnAdult, MaximumDurationAdaptSubsample) {
  dataset_sampling_ = 1.0f;
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(1000);
  gbt_config->set_early_stopping(
      proto::GradientBoostedTreesTrainingConfig::NONE);
  train_config_.set_maximum_training_duration_seconds(10);
  gbt_config->set_adapt_subsample_for_maximum_training_duration(true);

  TrainAndEvaluateModel();
  // Note: The "TrainAndEvaluateModel" function last a bit more because it is
  // also preparing the dataset and evaluating the final model.
#if !defined(MEMORY_SANITIZER) && !defined(THREAD_SANITIZER) && \
    !defined(ADDRESS_SANITIZER) && !defined(SKIP_TIMING_TESTS)
  EXPECT_LE(absl::ToDoubleSeconds(training_duration_), 20);
  EXPECT_GE(metric::Accuracy(evaluation_), 0.80);
#endif
}

TEST_F(GradientBoostedTreesOnAdult,
       DisableEarlyStoppingBecauseOfZeroValidationRatio) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->set_validation_set_ratio(0.f);
  TrainAndEvaluateModel();
}

// Train and test a model on the adult dataset.
TEST_F(GradientBoostedTreesOnAdult, Dart) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_dart()->set_dropout_rate(0.1f);
  gbt_config->mutable_decision_tree()->set_num_candidate_attributes(8);

  dataset_sampling_ = 1.0f;

  TrainAndEvaluateModel();

  // Note: Dart seems to be unstable.
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8459, 0.0449, nan);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3293, 0.0727, nan);
}

TEST_F(GradientBoostedTreesOnAdult, Hessian) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_subsample(0.9f);
  gbt_config->set_use_hessian_gain(true);

  TrainAndEvaluateModel();

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8664, 0.0101, 0.8661);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2962, 0.0159, 0.2907);
}

TEST_F(GradientBoostedTreesOnAdult, HessianRandomCategorical) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_subsample(0.9f);
  gbt_config->set_use_hessian_gain(true);
  gbt_config->mutable_decision_tree()->mutable_categorical()->mutable_random();

  TrainAndEvaluateModel();

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8636, 0.0092, 0.859);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2924, 0.0112, 0.2934);
}

TEST_F(GradientBoostedTreesOnAdult, HessianDiscretizedNumerical) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_subsample(0.9f);
  gbt_config->set_use_hessian_gain(true);
  guide_.set_detect_numerical_as_discretized_numerical(true);

  TrainAndEvaluateModel();

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8662, 0.0104, 0.8642);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2938, 0.0116, 0.2930);
}

TEST_F(GradientBoostedTreesOnAdult, HessianL2Categorical) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_decision_tree()->set_max_depth(4);
  gbt_config->set_subsample(0.9f);
  gbt_config->set_use_hessian_gain(true);
  gbt_config->set_l2_regularization_categorical(10.f);

  TrainAndEvaluateModel();

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8652, 0.0124, 0.867);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2962, 0.0125, 0.2922);
}

TEST_F(GradientBoostedTreesOnAdult, PureServingModel) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  train_config_.set_pure_serving_model(true);
  TrainAndEvaluateModel();

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8661, 0.0134, 0.8615);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3001, 0.0204, 0.2977);
}

TEST_F(GradientBoostedTreesOnAdult, MakingAModelPurePureServingModel) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  TrainAndEvaluateModel();

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8676, 0.0129, 0.8615);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2977, 0.0167, 0.2977);
  const auto pre_pruning_size = model_->ModelSizeInBytes().value();
  YDF_LOG(INFO) << "pre_pruning_size:" << pre_pruning_size;

  CHECK_OK(model_->MakePureServing());

  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.8676, 0.0129, 0.8615);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2977, 0.0167, 0.2977);

  const auto post_pruning_size = model_->ModelSizeInBytes().value();
  YDF_LOG(INFO) << "post_pruning_size:" << post_pruning_size;
  EXPECT_LE(static_cast<float>(post_pruning_size) / pre_pruning_size, 0.80);
}

// Helper for the training and testing on two non-overlapping samples from the
// Abalone dataset.
class GradientBoostedTreesOnAbalone : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(GradientBoostedTreesLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::REGRESSION);
    train_config_.set_label("Rings");
    dataset_filename_ = "abalone.csv";
    deployment_config_.set_num_threads(1);
    auto* gbt_config = train_config_.MutableExtension(
        gradient_boosted_trees::proto::gradient_boosted_trees_config);
    gbt_config->mutable_decision_tree()
        ->set_internal_error_on_wrong_splitter_statistics(true);
  }
};

TEST_F(GradientBoostedTreesOnAbalone, Base) {
  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::RMSE(evaluation_), 2.1684, 0.0979, 2.1138);
}

TEST_F(GradientBoostedTreesOnAbalone, L2Regularization) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_l2_regularization(0.1f);
  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::RMSE(evaluation_), 2.1593, 0.0776, 2.1241);
}

TEST_F(GradientBoostedTreesOnAbalone, SparseOblique) {
  deployment_config_.set_num_threads(5);
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->mutable_decision_tree()->mutable_sparse_oblique_split();
  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::RMSE(evaluation_), 2.1155, 0.0988, 2.1001);
}

TEST_F(GradientBoostedTreesOnAbalone, PoissonLoss) {
  deployment_config_.set_num_threads(5);
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_loss(proto::Loss::POISSON);

  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::RMSE(evaluation_), 2.1563, 0.0852, 2.1491);
}

TEST_F(GradientBoostedTreesOnAbalone, MAELoss) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_loss(proto::Loss::MEAN_AVERAGE_ERROR);
  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::MAE(evaluation_), 1.5155, 0.0599, 1.4994);
  YDF_TEST_METRIC(metric::RMSE(evaluation_), 2.2608, 0.1464, 2.2136);
}

class GradientBoostedTreesOnIris : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(GradientBoostedTreesLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    train_config_.set_label("class");
    dataset_filename_ = "iris.csv";
    deployment_config_.set_num_threads(1);
    auto* gbt_config = train_config_.MutableExtension(
        gradient_boosted_trees::proto::gradient_boosted_trees_config);
    gbt_config->mutable_decision_tree()
        ->set_internal_error_on_wrong_splitter_statistics(true);
  }
};

TEST_F(GradientBoostedTreesOnIris, Base) {
  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.9533, 0.03, 0.96);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.2988, 0.2562, 0.2255);
  // Note: R RandomForest has an OOB accuracy of 0.9467.
}

TEST_F(GradientBoostedTreesOnIris, Hessian) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_use_hessian_gain(true);
  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.94, 0.05, 0.9733);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.3225, 0.3002, 0.1462);
}

TEST_F(GradientBoostedTreesOnIris, Dart) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100);
  gbt_config->mutable_dart()->set_dropout_rate(0.1f);
  gbt_config->mutable_decision_tree()->set_num_candidate_attributes(8);
  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.9467, 0.04, 0.9733);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.1925, 0.1226, 0.2019);
  // Note: R RandomForest has an OOB accuracy of 0.9467.
}

class GradientBoostedTreesOnDNA : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(GradientBoostedTreesLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    train_config_.set_label("LABEL");
    dataset_filename_ = "dna.csv";
    deployment_config_.set_num_threads(1);
    auto* gbt_config = train_config_.MutableExtension(
        gradient_boosted_trees::proto::gradient_boosted_trees_config);
    gbt_config->mutable_decision_tree()
        ->set_internal_error_on_wrong_splitter_statistics(true);
  }
};

TEST_F(GradientBoostedTreesOnDNA, Base) {
  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.9507, 0.0108, 0.9542);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.1933, 0.08, 0.1452);
  // Note: R RandomForest has an OOB accuracy of 0.909.
}

TEST_F(GradientBoostedTreesOnDNA, Hessian) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_use_hessian_gain(true);
  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.9539, 0.0099, 0.9554);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.1831, 0.0743, 0.1442);
}

TEST_F(GradientBoostedTreesOnDNA, BaseBooleanAsNumerical) {
  guide_filename_ = "dna_guide.pbtxt";
  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.9514, 0.0118, 0.9542);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.1757, 0.0568, 0.1452);
  // Note: R RandomForest has an OOB accuracy of 0.909.
}

TEST_F(GradientBoostedTreesOnDNA, HessianBooleanAsNumerical) {
  auto* gbt_config = train_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_use_hessian_gain(true);
  guide_filename_ = "dna_guide.pbtxt";
  TrainAndEvaluateModel();
  YDF_TEST_METRIC(metric::Accuracy(evaluation_), 0.9532, 0.0118, 0.9567);
  YDF_TEST_METRIC(metric::LogLoss(evaluation_), 0.1813, 0.0716, 0.1442);
}

TEST(GradientBoostedTrees, SetHyperParameters) {
  GradientBoostedTreesLearner learner{model::proto::TrainingConfig()};
  const auto hparam_spec =
      learner.GetGenericHyperParameterSpecification().value();
  const auto& gbdt_config =
      *learner.mutable_training_config()->MutableExtension(
          gradient_boosted_trees::proto::gradient_boosted_trees_config);
  const float epsilon = 0.0001f;

  // Defaults
  EXPECT_EQ(gbdt_config.num_trees(), 300);
  EXPECT_FALSE(
      learner.training_config().has_maximum_training_duration_seconds());
  EXPECT_NEAR(gbdt_config.validation_set_ratio(), 0.1f, epsilon);
  EXPECT_EQ(gbdt_config.early_stopping_num_trees_look_ahead(), 30);
  EXPECT_EQ(
      gbdt_config.early_stopping(),
      proto::GradientBoostedTreesTrainingConfig::VALIDATION_LOSS_INCREASE);

  EXPECT_OK(learner.SetHyperParameters(PARSE_TEST_PROTO(R"pb(
    fields {
      name: "num_trees"
      value { integer: 10 }
    }
    fields {
      name: "maximum_training_duration_seconds"
      value { real: 10 }
    }
    fields {
      name: "validation_ratio"
      value { real: 0.2 }
    }
    fields {
      name: "early_stopping_num_trees_look_ahead"
      value { integer: 10 }
    }
    fields {
      name: "early_stopping"
      value { categorical: "NONE" }
    }
  )pb")));

  // Set
  EXPECT_EQ(gbdt_config.num_trees(), 10);
  EXPECT_NEAR(learner.training_config().maximum_training_duration_seconds(),
              10., epsilon);
  EXPECT_NEAR(gbdt_config.validation_set_ratio(), 0.2, epsilon);
  EXPECT_EQ(gbdt_config.early_stopping_num_trees_look_ahead(), 10);
  EXPECT_EQ(gbdt_config.early_stopping(),
            proto::GradientBoostedTreesTrainingConfig::NONE);
  EXPECT_TRUE(gbdt_config.has_stochastic_gradient_boosting());
  EXPECT_NEAR(gbdt_config.stochastic_gradient_boosting().ratio(), 1.0, epsilon);
  EXPECT_FALSE(gbdt_config.has_subsample());

  // Extra test for the sampling.

  // Old / simple way to specify Random sampling.
  EXPECT_OK(learner.SetHyperParameters(PARSE_TEST_PROTO(R"pb(
    fields {
      name: "subsample"
      value { real: 0.2 }
    }
  )pb")));
  EXPECT_FALSE(gbdt_config.has_subsample());
  EXPECT_TRUE(gbdt_config.has_stochastic_gradient_boosting());
  EXPECT_NEAR(gbdt_config.stochastic_gradient_boosting().ratio(), 0.2, epsilon);

  // Goss
  EXPECT_OK(learner.SetHyperParameters(PARSE_TEST_PROTO(R"pb(
    fields {
      name: "sampling_method"
      value { categorical: "GOSS" }
    }
    fields {
      name: "goss_alpha"
      value { real: 0.4 }
    }
    fields {
      name: "goss_beta"
      value { real: 0.5 }
    }
  )pb")));
  EXPECT_TRUE(gbdt_config.has_gradient_one_side_sampling());
  EXPECT_NEAR(gbdt_config.gradient_one_side_sampling().alpha(), 0.4, epsilon);
  EXPECT_NEAR(gbdt_config.gradient_one_side_sampling().beta(), 0.5, epsilon);

  // Random sampling
  EXPECT_OK(learner.SetHyperParameters(PARSE_TEST_PROTO(R"pb(
    fields {
      name: "sampling_method"
      value { categorical: "RANDOM" }
    }
    fields {
      name: "subsample"
      value { real: 0.3 }
    }
  )pb")));
  EXPECT_TRUE(gbdt_config.has_stochastic_gradient_boosting());
  EXPECT_NEAR(gbdt_config.stochastic_gradient_boosting().ratio(), 0.3, epsilon);

  // No sampling.
  EXPECT_OK(learner.SetHyperParameters(PARSE_TEST_PROTO(R"pb(
    fields {
      name: "sampling_method"
      value { categorical: "NONE" }
    }
  )pb")));
  EXPECT_FALSE(gbdt_config.has_stochastic_gradient_boosting());

  // Ransom sampling with compatibility mode
  EXPECT_OK(learner.SetHyperParameters(PARSE_TEST_PROTO(R"pb(
    fields {
      name: "sampling_method"
      value { categorical: "NONE" }
    }
    fields {
      name: "subsample"
      value { real: 0.4 }
    }
  )pb")));
  EXPECT_TRUE(gbdt_config.has_stochastic_gradient_boosting());
  EXPECT_NEAR(gbdt_config.stochastic_gradient_boosting().ratio(), 0.4, epsilon);
}

TEST(DartPredictionAccumulator, Base) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  std::vector<float> weights(dataset.nrow(), 1.f);

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  const auto loss_imp =
      CreateLoss(proto::Loss::SQUARED_ERROR, model::proto::Task::REGRESSION,
                 dataset.data_spec().columns(0), {})
          .value();
  CHECK_OK(internal::CreateGradientDataset(dataset,
                                           /* label_col_idx= */ 0,
                                           /*hessian_splits=*/false, *loss_imp,
                                           &gradient_dataset, &gradients,
                                           &predictions));
  const auto initial_predictions =
      loss_imp
          ->InitialPredictions(dataset,
                               /* label_col_idx =*/0, weights)
          .value();
  internal::SetInitialPredictions(initial_predictions, dataset.nrow(),
                                  &predictions);
  EXPECT_NEAR(initial_predictions[0], 2.5f, 0.001f);

  utils::RandomEngine random(1234);
  CHECK_OK(loss_imp->UpdateGradients(gradient_dataset,
                                     /* label_col_idx= */ 0, predictions,
                                     /*ranking_index=*/nullptr, &gradients,
                                     &random));

  utils::RandomEngine rnd(12345);
  internal::DartPredictionAccumulator acc;
  acc.Initialize(initial_predictions, dataset.nrow());
  EXPECT_TRUE(acc.SampleIterIndices(0.5f, &rnd).empty());

  CHECK_OK(acc.GetSampledPredictions({}, &predictions));
  EXPECT_NEAR(predictions[0], 2.5f, 0.0001f);

  CHECK_OK(acc.GetAllPredictions(&predictions));
  EXPECT_NEAR(predictions[0], 2.5f, 0.0001f);

  auto tree = absl::make_unique<decision_tree::DecisionTree>();
  tree->CreateRoot();
  tree->mutable_root()->CreateChildren();
  tree->mutable_root()->mutable_node()->mutable_condition()->set_attribute(0);
  tree->mutable_root()
      ->mutable_node()
      ->mutable_condition()
      ->mutable_condition()
      ->mutable_higher_condition()
      ->set_threshold(2.5f);

  tree->mutable_root()
      ->mutable_pos_child()
      ->mutable_node()
      ->mutable_regressor()
      ->set_top_value(2.0f);
  tree->mutable_root()
      ->mutable_neg_child()
      ->mutable_node()
      ->mutable_regressor()
      ->set_top_value(1.0f);

  std::vector<std::unique_ptr<decision_tree::DecisionTree>> trees;
  trees.push_back(std::move(tree));

  double mean_abs_prediction;
  CHECK_OK(acc.UpdateWithNewIteration(
      {}, proto::Loss::SQUARED_ERROR, *loss_imp, trees, gradient_dataset,
      /* num_gradient_dimensions= */ 1, &mean_abs_prediction));

  CHECK_OK(acc.GetSampledPredictions({}, &predictions));
  EXPECT_NEAR(predictions[0], 3.5f, 0.0001f);

  CHECK_OK(acc.GetAllPredictions(&predictions));
  EXPECT_NEAR(predictions[0], 3.5f, 0.0001f);

  EXPECT_EQ(acc.SampleIterIndices(0.5f, &rnd).size(), 1);
  CHECK_OK(acc.UpdateWithNewIteration(
      {0}, proto::Loss::SQUARED_ERROR, *loss_imp, trees, gradient_dataset,
      /* num_gradient_dimensions= */ 1, &mean_abs_prediction));

  CHECK_OK(acc.GetSampledPredictions({}, &predictions));
  EXPECT_NEAR(predictions[0], 2.5 + 1. * 1. * 1. / 2. + 1. * 1. / 2., 0.0001f);

  // Same value as above.
  CHECK_OK(acc.GetAllPredictions(&predictions));
  EXPECT_NEAR(predictions[0], 3.5f, 0.0001f);

  CHECK_OK(acc.GetSampledPredictions({0}, &predictions));
  EXPECT_NEAR(predictions[0], 3.f, 0.0001f);  // 2.5

  CHECK_OK(acc.GetSampledPredictions({1}, &predictions));
  EXPECT_NEAR(predictions[0], 3.f, 0.0001f);  // 2.5

  const auto scaling = acc.TreeOutputScaling();
  EXPECT_EQ(scaling.size(), 2);
  EXPECT_NEAR(scaling[0], 0.5f, 0.0001f);
  EXPECT_NEAR(scaling[1], 0.5f, 0.0001f);
}

TEST(GradientBoostedTrees, PredefinedHyperParametersClassification) {
  model::proto::TrainingConfig train_config;
  train_config.set_learner(GradientBoostedTreesLearner::kRegisteredName);
  utils::TestPredefinedHyperParametersAdultDataset(train_config, 2, 0.86);
}

TEST(GradientBoostedTrees, PredefinedHyperParametersRanking) {
  model::proto::TrainingConfig train_config;
  train_config.set_learner(GradientBoostedTreesLearner::kRegisteredName);
  utils::TestPredefinedHyperParametersRankingDataset(train_config, 2,
                                                     absl::nullopt);
}

TEST_F(GradientBoostedTreesOnAdult, InterruptAndResumeTraining) {
  // Train a model for a few seconds, interrupt its training, and resume it.

  deployment_config_.set_cache_path(
      file::JoinPath(test::TmpDirectory(), "cache"));
  deployment_config_.set_try_resume_training(true);
  deployment_config_.set_resume_training_snapshot_interval_seconds(1);

  // Configure a training that would take a long time.
  // Note: The quality of this model will be poor as it will overfit strongly
  // the training dataset.
  auto* gbt_config =
      train_config_.MutableExtension(proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100000);
  gbt_config->set_early_stopping(
      proto::GradientBoostedTreesTrainingConfig::NONE);

  // Train for 10 seconds.
  interrupt_training_after = absl::Seconds(10);
  check_model = false;
  TrainAndEvaluateModel();
  auto interrupted_model = std::move(model_);

  const auto get_gbt = [](const std::unique_ptr<model::AbstractModel>& mdl) {
    return dynamic_cast<const GradientBoostedTreesModel*>(mdl.get());
  };

  // Check snapshots
  ASSERT_OK_AND_ASSIGN(std::deque<int> snapshots,
                       utils::GetSnapshots(file::JoinPath(
                           deployment_config_.cache_path(), "snapshot")));
  // The training will create 9 or 10 snapshots, but we guarantee that only 3 of
  // them will be there.
  ASSERT_THAT(snapshots, SizeIs(3));
  EXPECT_EQ(snapshots.back(), get_gbt(interrupted_model)->NumTrees());

  // Resume the training with 100 extra trees.
  gbt_config->set_num_trees(get_gbt(interrupted_model)->NumTrees() + 100);
  interrupt_training_after = {};
  check_model = true;
  TrainAndEvaluateModel();
  auto resumed_model = std::move(model_);

  // Ensures that the final model contains +100 iterations from the interrupted
  // model.
  EXPECT_EQ(get_gbt(interrupted_model)->NumTrees() + 100,
            get_gbt(resumed_model)->NumTrees());

  // Check snapshots again.
  ASSERT_OK_AND_ASSIGN(snapshots,
                       utils::GetSnapshots(file::JoinPath(
                           deployment_config_.cache_path(), "snapshot")));
  ASSERT_THAT(snapshots, SizeIs(3));
  EXPECT_EQ(snapshots.back(), get_gbt(resumed_model)->NumTrees());
}

TEST_F(GradientBoostedTreesOnAdult, EarlyStoppingInitialIteration) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  // Configure model training.
  model::proto::DeploymentConfig deployment_config;
  model::proto::TrainingConfig train_config;
  train_config.set_learner(GradientBoostedTreesLearner::kRegisteredName);
  train_config.set_task(model::proto::Task::CLASSIFICATION);
  train_config.set_label("b");
  train_config.add_features("a");

  std::unique_ptr<model::AbstractLearner> learner;
  auto* gbt_config = train_config.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  gbt_config->set_early_stopping_num_trees_look_ahead(1);
  gbt_config->set_validation_set_ratio(0.3f);
  gbt_config->set_early_stopping_initial_iteration(0);
  ASSERT_OK(model::GetLearner(train_config, &learner, deployment_config));
  ASSERT_OK_AND_ASSIGN(const std::unique_ptr<AbstractModel> model,
                       learner->TrainWithStatus(dataset));
  const GradientBoostedTreesModel* gbt_model =
      dynamic_cast<const GradientBoostedTreesModel*>(model.get());
  EXPECT_EQ(gbt_model->NumTrees(), 1);
}

TEST_F(GradientBoostedTreesOnIris, InterruptAndResumeTraining) {
  // Train a model for a few seconds, interrupt its training, and resume it.

  deployment_config_.set_cache_path(
      file::JoinPath(test::TmpDirectory(), "cache_iris"));
  deployment_config_.set_try_resume_training(true);
  deployment_config_.set_resume_training_snapshot_interval_seconds(1);

  // Configure a training that would take a long time.
  // Note: The quality of this model will be poor as it will overfit strongly
  // the training dataset.
  auto* gbt_config =
      train_config_.MutableExtension(proto::gradient_boosted_trees_config);
  gbt_config->set_num_trees(100000);
  gbt_config->set_early_stopping(
      proto::GradientBoostedTreesTrainingConfig::NONE);

  // Train for 5 seconds.
  interrupt_training_after = absl::Seconds(5);
  check_model = false;
  TrainAndEvaluateModel();
  auto interrupted_model = std::move(model_);

  const auto get_gbt = [](const std::unique_ptr<model::AbstractModel>& mdl) {
    return dynamic_cast<const GradientBoostedTreesModel*>(mdl.get());
  };

  const int num_classes = 3;  // The dataset contains 3 classes.
  EXPECT_EQ(get_gbt(interrupted_model)->num_trees_per_iter(), num_classes);

  // Resume the training with 100 extra trees.
  // There are 3 trees per iterations (because the dataset contains 3 classes).
  gbt_config->set_num_trees(
      get_gbt(interrupted_model)->NumTrees() / num_classes + 100);
  interrupt_training_after = {};
  check_model = true;
  TrainAndEvaluateModel();
  auto resumed_model = std::move(model_);

  // Ensures that the final model contains +100 iterations from the interrupted
  // model.
  EXPECT_EQ(get_gbt(interrupted_model)->NumTrees() + 100 * num_classes,
            get_gbt(resumed_model)->NumTrees());
}

// Make sure that the confusion table of the training logs agrees with the
// confusion table computed a posteriori on the validation dataset.
TEST_F(GradientBoostedTreesOnAdult, BinomialLogLikelihoodConfusionTablesAgree) {
  pass_training_dataset_as_path_ = false;
  pass_validation_dataset_ = true;
  TrainAndEvaluateModel();

  ASSERT_TRUE(model_->ValidationEvaluation().has_classification());
  ASSERT_TRUE(model_->ValidationEvaluation().classification().has_confusion());
  auto training_logs_confusion_table =
      model_->ValidationEvaluation().classification().confusion();

  utils::RandomEngine rnd(1234);
  const auto a_posteriori_evaluation =
      model_->Evaluate(valid_dataset_, {}, &rnd);

  ASSERT_TRUE(a_posteriori_evaluation.has_classification());
  ASSERT_TRUE(a_posteriori_evaluation.classification().has_confusion());
  auto evaluation_confusion_table =
      a_posteriori_evaluation.classification().confusion();
  EXPECT_THAT(training_logs_confusion_table,
              EqualsProto(evaluation_confusion_table));
}

// Make sure that the confusion table of the training logs agrees with the
// confusion table computed a posteriori on the validation dataset.
// This test specifically tests sharded datasets.
TEST_F(GradientBoostedTreesOnAdult,
       BinomialLogLikelihoodConfusionTablesAgreeSharded) {
  pass_training_dataset_as_path_ = true;
  pass_validation_dataset_ = true;
  TrainAndEvaluateModel();

  ASSERT_TRUE(model_->ValidationEvaluation().has_classification());
  ASSERT_TRUE(model_->ValidationEvaluation().classification().has_confusion());
  auto training_logs_confusion_table =
      model_->ValidationEvaluation().classification().confusion();

  utils::RandomEngine rnd(1234);
  const auto a_posteriori_evaluation =
      model_->Evaluate(valid_dataset_, {}, &rnd);

  ASSERT_TRUE(a_posteriori_evaluation.has_classification());
  ASSERT_TRUE(a_posteriori_evaluation.classification().has_confusion());
  auto evaluation_confusion_table =
      a_posteriori_evaluation.classification().confusion();
  EXPECT_THAT(training_logs_confusion_table,
              EqualsProto(evaluation_confusion_table));
}

class AutotunedGradientBoostedTreesOnAdult : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_ = PARSE_TEST_PROTO(R"pb(
      task: CLASSIFICATION
      learner: "HYPERPARAMETER_OPTIMIZER"
      label: "income"

      [yggdrasil_decision_forests.model.hyperparameters_optimizer_v2.proto
           .hyperparameters_optimizer_config] {

        retrain_final_model: true

        optimizer {
          optimizer_key: "RANDOM"
          [yggdrasil_decision_forests.model.hyperparameters_optimizer_v2.proto
               .random] { num_trials: 25 }
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

        predefined_search_space {}
      }
    )pb");

    train_config_.set_learner(
        hyperparameters_optimizer_v2::HyperParameterOptimizerLearner::
            kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    train_config_.set_label("income");
    train_config_.add_features(".*");
    dataset_filename_ = "adult.csv";

    deployment_config_.set_cache_path(
        file::JoinPath(test::TmpDirectory(), "working_directory"));
  }
};

TEST_F(AutotunedGradientBoostedTreesOnAdult,
       RandomTuner_MemoryDataset_LocalTraining) {
  TrainAndEvaluateModel();
  EXPECT_GE(metric::Accuracy(evaluation_), 0.87);
  EXPECT_LT(metric::LogLoss(evaluation_), 0.30);
  EXPECT_EQ(model_->hyperparameter_optimizer_logs()->steps_size(), 25);
}

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
