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

#include "yggdrasil_decision_forests/learner/random_forest/random_forest.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/dataset/weight.pb.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/test_utils.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace random_forest {
namespace {

std::string DatasetDir() {
  return file::JoinPath(
      test::DataRootDirectory(),
      "yggdrasil_decision_forests/test_data/dataset");
}

// Build a forest with two decision trees as follow:
// [a>1]
//   ├── [b=0] (pos)
//   └── [b=1] (neg)
// [a>3]
//   ├── [b=2] (pos)
//   └── [b=1] (neg)
//
// Build the dataset:
// "a" : {0, 2, 4}
// "b" : {1, 2, 1}
void BuildToyModelAndToyDataset(const model::proto::Task task,
                                RandomForestModel* model,
                                dataset::VerticalDataset* dataset) {
  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "a" }
    columns {
      type: CATEGORICAL
      name: "b"
      categorical { is_already_integerized: true number_of_unique_values: 3 }
    }
  )pb");

  dataset->set_data_spec(dataspec);
  CHECK_OK(dataset->CreateColumnsFromDataspec());
  auto* col_1 =
      dataset->MutableColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
          0);
  col_1->Add(0);
  col_1->Add(2);
  col_1->Add(4);

  auto* col_2 =
      dataset
          ->MutableColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
              1);
  col_2->Add(1);
  col_2->Add(2);
  col_2->Add(1);
  dataset->set_nrow(3);

  // Create a tree of the form.
  // [a> alpha]
  //   ├── [b=beta]
  //   └── [b=gamma]
  auto create_tree = [&task](const float alpha, const int beta,
                             const int gamma) {
    auto tree = absl::make_unique<decision_tree::DecisionTree>();
    tree->CreateRoot();
    tree->mutable_root()->CreateChildren();
    tree->mutable_root()->mutable_node()->mutable_condition()->set_attribute(0);
    tree->mutable_root()
        ->mutable_node()
        ->mutable_condition()
        ->mutable_condition()
        ->mutable_higher_condition()
        ->set_threshold(alpha);

    auto* pos_child = tree->mutable_root()->mutable_pos_child()->mutable_node();
    auto* neg_child = tree->mutable_root()->mutable_neg_child()->mutable_node();

    switch (task) {
      case model::proto::Task::CLASSIFICATION:
        pos_child->mutable_classifier()->set_top_value(beta);
        neg_child->mutable_classifier()->set_top_value(gamma);
        break;
      case model::proto::Task::REGRESSION:
        pos_child->mutable_regressor()->set_top_value(beta);
        neg_child->mutable_regressor()->set_top_value(gamma);
        break;
      default:
        CHECK(false);
    }
    return tree;
  };

  model->AddTree(create_tree(1, 0, 1));
  model->AddTree(create_tree(3, 2, 1));

  model->set_task(task);
  model->set_label_col_idx(1);
  model->set_data_spec(dataspec);
}

// Generate a dataset similar as the figure 10 in the Extremely Randomized Trees
// paper.
void ExtremelyRandomizeTreesFigure10Dataset(const int num_examples,
                                            dataset::VerticalDataset* dataset,
                                            utils::RandomEngine* random) {
  *dataset->mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "x" }
    columns { type: NUMERICAL name: "y" }
  )pb");
  CHECK_OK(dataset->CreateColumnsFromDataspec());
  std::uniform_real_distribution<float> x_dist(0, 3);
  for (int example_idx = 0; example_idx < num_examples; example_idx++) {
    const float x = x_dist(*random);
    float y = 0;
    if (x < 1) {
      y = 1 - x;
    } else if (x < 2) {
      y = -1 + x;
    }
    dataset::proto::Example example;
    example.add_attributes()->set_numerical(x);
    example.add_attributes()->set_numerical(y);
    dataset->AppendExample(example);
  }
}

// Returns the rank of importance of an attribute.
int GetVariableImportanceRank(
    const absl::string_view attribute,
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<model::proto::VariableImportance>& variable_importance) {
  const int attribute_idx = dataset::GetColumnIdxFromName(attribute, data_spec);
  const auto found_iterator = std::find_if(
      variable_importance.begin(), variable_importance.end(),
      [attribute_idx](const model::proto::VariableImportance& var) {
        return var.attribute_idx() == attribute_idx;
      });
  CHECK(found_iterator != variable_importance.end());
  return std::distance(variable_importance.begin(), found_iterator);
}

// Helper for the training and testing on two non-overlapping samples from the
// adult dataset.
class RandomForestOnAdult : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(RandomForestLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    train_config_.set_label("income");
    train_config_.add_features(".*");
    dataset_filename_ = "adult.csv";
    dataset_sampling_ = 0.2f;

    auto* rf_config = train_config_.MutableExtension(
        random_forest::proto::random_forest_config);
    rf_config->mutable_decision_tree()
        ->set_internal_error_on_wrong_splitter_statistics(true);
  }
};

// Train and test a model on the adult dataset.
TEST_F(RandomForestOnAdult, Base) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_compute_oob_variable_importances(true);
  rf_config->mutable_decision_tree()->set_allow_na_conditions(true);
  const auto oob_prediction_path =
      file::JoinPath(test::TmpDirectory(), "oob_predictions.csv");
  rf_config->set_export_oob_prediction_path(
      absl::StrCat("csv:", oob_prediction_path));

  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.860, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.333, 0.04);

  const auto mean_decrease_accuracy =
      model_
          ->GetVariableImportance(random_forest::RandomForestModel::
                                      kVariableImportanceMeanDecreaseInAccuracy)
          .value();

  // The top and worst variables have been computed using the "randomForest" R
  // package. This (simple ml) and the R randomForest implementation work
  // differently for categorical attributes. Since this dataset has a lot of
  // categorical attributes, the reported orders of variable importance are not
  // exactly similar among both libraries. However, the overall ranking is still
  // close.

  // Top 3 variables.
  const int rank_capital_gain = GetVariableImportanceRank(
      "capital_gain", model_->data_spec(), mean_decrease_accuracy);
  const int rank_relationship = GetVariableImportanceRank(
      "relationship", model_->data_spec(), mean_decrease_accuracy);
  const int rank_occupation = GetVariableImportanceRank(
      "occupation", model_->data_spec(), mean_decrease_accuracy);

  EXPECT_LE(rank_capital_gain, 5);
  EXPECT_LE(rank_relationship, 5);
  EXPECT_LE(rank_occupation, 5);

  // Worst 2 variables.
  const int rank_fnlwgt = GetVariableImportanceRank(
      "fnlwgt", model_->data_spec(), mean_decrease_accuracy);
  const int rank_education = GetVariableImportanceRank(
      "education", model_->data_spec(), mean_decrease_accuracy);

  EXPECT_GE(rank_fnlwgt, 7);
  EXPECT_GE(rank_education, 4);

  std::string description;
  model_->AppendDescriptionAndStatistics(false, &description);
  LOG(INFO) << description;

  CHECK_NE(description.find("MEAN_DECREASE_IN_ACCURACY"), -1);
  CHECK_NE(description.find("Out-of-bag evaluation: accuracy:"), -1);
  CHECK_NE(description.find("Type: \"RANDOM_FOREST\""), -1);
  CHECK_NE(description.find("Task: CLASSIFICATION"), -1);
  CHECK_NE(description.find("Label: \"income\""), -1);

  // Check the oob predictions.
  const auto oob_predictions = file::GetContent(oob_prediction_path).value();
  EXPECT_TRUE(absl::StartsWith(oob_predictions, "<=50K,>50K\n"));
  EXPECT_EQ(std::count(oob_predictions.begin(), oob_predictions.end(), '\n'),
            train_dataset_.nrow() + 1 /*the header*/);
}

// Separate the examples used for the structure and the leaves of the model.
TEST_F(RandomForestOnAdult, Honest) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->mutable_decision_tree()->mutable_honest();
  rf_config->set_sampling_with_replacement(false);
  rf_config->set_bootstrap_size_ratio(0.5);

  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8504, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.333, 0.04);
}

// Extremely Randomize Trees on Adult.
TEST_F(RandomForestOnAdult, ExtremelyRandomizeTrees) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_bootstrap_training_dataset(false);
  rf_config->set_num_trees(300);
  rf_config->mutable_decision_tree()->mutable_numerical_split()->set_type(
      decision_tree::proto::NumericalSplit::HISTOGRAM_RANDOM);
  rf_config->mutable_decision_tree()->set_max_depth(-1);
  rf_config->mutable_decision_tree()->set_min_examples(3);
  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8415, 0.01);
}

// Equal width histogram split on Adult.
TEST_F(RandomForestOnAdult, EqualWidthHistogramNoWinnerTakeAll) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_winner_take_all_inference(false);
  rf_config->mutable_decision_tree()->mutable_numerical_split()->set_type(
      decision_tree::proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH);
  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8525, 0.01);
}

TEST_F(RandomForestOnAdult, NoWinnerTakeAllWithWeights) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_winner_take_all_inference(false);
  TrainAndEvaluateModel(/*numerical_weight_attribute=*/"age",
                        /*emulate_weight_with_duplication=*/true);
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8415, 0.012);
}

TEST_F(RandomForestOnAdult, NoWinnerTakeAll) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_winner_take_all_inference(false);
  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.862, 0.015);
  // Disabling winner take all reduce the logloss (as expected).
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.310, 0.04);
}

TEST_F(RandomForestOnAdult, NoWinnerTakeAllDiscretizedNumerical) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_winner_take_all_inference(false);
  guide_.set_detect_numerical_as_discretized_numerical(true);
  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.862, 0.015);
  // Disabling winner take all reduce the logloss (as expected).
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.310, 0.04);
}

// Use the one-hot splitter for the categorical features.
TEST_F(RandomForestOnAdult, NoWinnerTakeAllOneHotCategorical) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);

  // Add the categorical features.
  train_config_.clear_features();
  train_config_.add_features("workclass");
  train_config_.add_features("education");
  train_config_.add_features("marital_status");
  train_config_.add_features("occupation");
  train_config_.add_features("relationship");
  train_config_.add_features("race");
  train_config_.add_features("sex");
  train_config_.add_features("native_country");

  rf_config->set_winner_take_all_inference(false);
  rf_config->mutable_decision_tree()->set_max_depth(64);
  rf_config->mutable_decision_tree()->mutable_categorical()->mutable_one_hot();
  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.8175, 0.015);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.4126, 0.04);

  // Note: Enabling "mutable_one_hot, increases the average tree depth
  // from 10.5 to 13.2, and reduce the accuracy from 0.825 to 0.817.
}

// Train a Random Forest model using only the categorical features, and using
// the Random Categorical splitter.
TEST_F(RandomForestOnAdult, NoWinnerTakeAllRandomCategorical) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);

  // Add the categorical features.
  train_config_.clear_features();
  train_config_.add_features("workclass");
  train_config_.add_features("education");
  train_config_.add_features("marital_status");
  train_config_.add_features("occupation");
  train_config_.add_features("relationship");
  train_config_.add_features("race");
  train_config_.add_features("sex");
  train_config_.add_features("native_country");

  rf_config->set_winner_take_all_inference(false);
  rf_config->mutable_decision_tree()->mutable_categorical()->mutable_random();
  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.82618, 0.005);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.40623, 0.02);
}

TEST_F(RandomForestOnAdult, NoWinnerTakeAllExampleSampling) {
  dataset_sampling_ = 1.0f;
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_winner_take_all_inference(false);
  rf_config->set_bootstrap_size_ratio(0.2f);
  TrainAndEvaluateModel();
  // Similar (should be slighly better) to the dataset sampling.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.862, 0.015);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.310, 0.04);
}

TEST_F(RandomForestOnAdult, BaseNumericalWeightedEval) {
  auto* weights = eval_options_.mutable_weights();
  weights->set_attribute("age");
  weights->mutable_numerical();

  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.833, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.380, 0.045);
}

TEST_F(RandomForestOnAdult, BaseCategoricalWeightedEval) {
  auto* weights = eval_options_.mutable_weights();
  weights->set_attribute("sex");
  auto* weight_male = weights->mutable_categorical()->add_items();
  weight_male->set_value("Male");
  weight_male->set_weight(2);
  auto* weight_female = weights->mutable_categorical()->add_items();
  weight_female->set_value("Female");
  weight_female->set_weight(3);

  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.869, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.319, 0.045);
}

TEST_F(RandomForestOnAdult, BaseCategoricalWeightedEvalAndWeightedTraining) {
  auto* weights = eval_options_.mutable_weights();
  weights->set_attribute("sex");
  auto* weight_male = weights->mutable_categorical()->add_items();
  weight_male->set_value("Male");
  weight_male->set_weight(2);
  auto* weight_female = weights->mutable_categorical()->add_items();
  weight_female->set_value("Female");
  weight_female->set_weight(3);

  *train_config_.mutable_weight_definition() = *weights;

  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.870, 0.01);
  // Training and evaluating with the same weight reduce the logloss (as
  // expected).
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.320, 0.04);
}

// Train and test a model on the adult dataset for a maximum given duration.
TEST_F(RandomForestOnAdult, MaximumDuration) {
  dataset_sampling_ = 1.0f;
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_num_trees(100000);  // Would take a very long time.
  rf_config->set_winner_take_all_inference(false);
  train_config_.set_maximum_training_duration_seconds(10);

  TrainAndEvaluateModel();
  // Note: The "TrainAndEvaluateModel" function last a bit more because it is
  // also preparing the dataset and evaluating the final model.
  EXPECT_LE(absl::ToDoubleSeconds(training_duration_), 10 + 20);

  EXPECT_GT(metric::Accuracy(evaluation_), 0.840);
}

// Train a model with a maximum size in RAM.
TEST_F(RandomForestOnAdult, MaximumSize) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_num_trees(100000);  // Would be a very big model.
  rf_config->set_winner_take_all_inference(false);

  const int max_size = 2 * 1024 * 1024;  // 2MB
  // Note: Each tree takes ~200k of RAM; the majority caused by proto overhead
  // and pointers. The serialized model will be ~5x smaller.
  train_config_.set_maximum_model_size_in_memory_in_bytes(max_size);

  TrainAndEvaluateModel();
  // Add an extra 3kB to help with the test flakiness.
  // Note: the model can be slighly larger than the
  // "set_maximum_model_size_in_memory_in_bytes" directive.
  EXPECT_LT(model_->ModelSizeInBytes().value(), max_size + 3 * 1024);

  EXPECT_GT(metric::Accuracy(evaluation_), 0.840);
}

TEST_F(RandomForestOnAdult, MaximumDurationInTree) {
  dataset_sampling_ = 1.0f;
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_num_trees(1);
  // Would take a very long time without a timeout inside of the tree building.
  rf_config->mutable_decision_tree()
      ->mutable_sparse_oblique_split()
      ->set_max_num_projections(10000);
  rf_config->mutable_decision_tree()
      ->mutable_sparse_oblique_split()
      ->set_num_projections_exponent(10);
  rf_config->set_winner_take_all_inference(false);
  train_config_.set_maximum_training_duration_seconds(10);

  TrainAndEvaluateModel();
  // Note: The "TrainAndEvaluateModel" function last a bit more because it is
  // also preparing the dataset and evaluating the final model.
#if !defined(MEMORY_SANITIZER) && !defined(THREAD_SANITIZER) && \
    !defined(ADDRESS_SANITIZER) && !defined(SKIP_TIMING_TESTS)
  EXPECT_LE(absl::ToDoubleSeconds(training_duration_), 10 + 20);
#endif
}

TEST_F(RandomForestOnAdult, InterruptTraining) {
  dataset_sampling_ = 1.0f;
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_num_trees(100000);  // Would take a very long time.
  rf_config->set_winner_take_all_inference(false);

  // Train for 5 seconds.
  interrupt_training_after = absl::Seconds(5);
  TrainAndEvaluateModel();

  // Note: The "TrainAndEvaluateModel" function last a bit more because it is
  // also preparing the dataset and evaluating the final model.
#if !defined(MEMORY_SANITIZER) && !defined(THREAD_SANITIZER) && \
    !defined(ADDRESS_SANITIZER) && !defined(SKIP_TIMING_TESTS)
  EXPECT_LE(absl::ToDoubleSeconds(training_duration_), 10 + 20);
  // Note: the model trained with a sanitizer might be small / poor.
  EXPECT_GT(metric::Accuracy(evaluation_), 0.840);
#endif
}

// Train and test a model on the adult dataset for a maximum given duration.
TEST_F(RandomForestOnAdult, MaximumDurationAdaptSampling) {
  dataset_sampling_ = 1.0f;
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_num_trees(1000);  // Would take a very long time.
  train_config_.set_maximum_training_duration_seconds(10);
  rf_config->set_adapt_bootstrap_size_ratio_for_maximum_training_duration(true);

  TrainAndEvaluateModel();
  // Note: The "TrainAndEvaluateModel" function last a bit more because it is
  // also preparing the dataset and evaluating the final model.
#if !defined(MEMORY_SANITIZER) && !defined(THREAD_SANITIZER) && \
    !defined(ADDRESS_SANITIZER) && !defined(SKIP_TIMING_TESTS)
  EXPECT_LE(absl::ToDoubleSeconds(training_duration_), 10 + 20);
  EXPECT_GE(metric::Accuracy(evaluation_), 0.84);
#endif
}

TEST_F(RandomForestOnAdult, MaxNumNodes) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  const int max_nodes = 50000;  // The model has ~100k nodes.
  rf_config->set_total_max_num_nodes(max_nodes);
  TrainAndEvaluateModel();

  auto* rf_model = dynamic_cast<const RandomForestModel*>(model_.get());
  EXPECT_LE(rf_model->NumNodes(), max_nodes);

  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.862, 0.015);
  // Disabling winner take all reduce the logloss (as expected).
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.368, 0.045);
}

TEST_F(RandomForestOnAdult, SparseOblique) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_winner_take_all_inference(false);
  rf_config->mutable_decision_tree()->mutable_sparse_oblique_split();
  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.855, 0.01);
}

// Helper for the training and testing on two non-overlapping samples from the
// Abalone dataset.
class RandomForestOnAbalone : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(RandomForestLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::REGRESSION);
    train_config_.set_label("Rings");
    dataset_filename_ = "abalone.csv";
    auto* rf_config = train_config_.MutableExtension(
        random_forest::proto::random_forest_config);
    rf_config->mutable_decision_tree()
        ->set_internal_error_on_wrong_splitter_statistics(true);
  }
};

TEST_F(RandomForestOnAbalone, Base) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  const auto oob_prediction_path =
      file::JoinPath(test::TmpDirectory(), "oob_predictions.csv");
  rf_config->set_export_oob_prediction_path(
      absl::StrCat("csv:", oob_prediction_path));

  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::RMSE(evaluation_), 2.0825, 0.01);

  // Check the oob predictions.
  const auto oob_predictions = file::GetContent(oob_prediction_path).value();
  EXPECT_TRUE(absl::StartsWith(oob_predictions, "Rings\n"));
  EXPECT_EQ(std::count(oob_predictions.begin(), oob_predictions.end(), '\n'),
            train_dataset_.nrow() + 1 /*the header*/);
}

TEST_F(RandomForestOnAbalone, ExtremelyRandomizeTrees) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_bootstrap_training_dataset(false);
  rf_config->set_num_trees(1000);
  rf_config->mutable_decision_tree()->mutable_numerical_split()->set_type(
      decision_tree::proto::NumericalSplit::HISTOGRAM_RANDOM);
  rf_config->mutable_decision_tree()->set_max_depth(-1);
  rf_config->mutable_decision_tree()->set_min_examples(3);
  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::RMSE(evaluation_), 2.114, 0.01);
}

TEST_F(RandomForestOnAbalone, SparseOblique) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->mutable_decision_tree()->mutable_sparse_oblique_split();
  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::RMSE(evaluation_), 2.054, 0.01);
}

TEST(RandomForest, SetHyperParameters) {
  RandomForestLearner learner{model::proto::TrainingConfig()};
  const auto hparam_spec =
      learner.GetGenericHyperParameterSpecification().value();
  EXPECT_OK(learner.SetHyperParameters(model::proto::GenericHyperParameters()));
  EXPECT_OK(learner.SetHyperParameters(PARSE_TEST_PROTO(
      "fields { name: \"num_trees\" value { integer: 10 } }")));
  EXPECT_OK(learner.SetHyperParameters(PARSE_TEST_PROTO(
      "fields { name: \"missing_value_policy\" value { categorical: "
      "\"LOCAL_IMPUTATION\" } }")));
  EXPECT_OK(learner.SetHyperParameters(
      PARSE_TEST_PROTO("fields { name: \"maximum_training_duration_seconds\" "
                       "value { real:10 } }")));
  EXPECT_THAT(
      learner.SetHyperParameters(PARSE_TEST_PROTO(
          "fields { name: \"missing_value_policy\" value { categorical: "
          "\"NON_EXISTING\" } }")),
      test::StatusIs(absl::StatusCode::kInvalidArgument));
  const auto& rf_config = learner.training_config().GetExtension(
      random_forest::proto::random_forest_config);
  EXPECT_EQ(rf_config.num_trees(), 10);
  EXPECT_NEAR(learner.training_config().maximum_training_duration_seconds(),
              10., 0.001);
  EXPECT_EQ(rf_config.decision_tree().missing_value_policy(),
            decision_tree::proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION);
}

TEST(RandomForest, OOBPredictions) {
  const model::proto::TrainingConfig config = PARSE_TEST_PROTO(R"pb(
    task: CLASSIFICATION
  )pb");
  const model::proto::TrainingConfigLinking config_link = PARSE_TEST_PROTO(R"pb(
    label: 1
    num_label_classes: 3
  )pb");

  utils::RandomEngine rnd;
  RandomForestModel model;
  dataset::VerticalDataset dataset;
  BuildToyModelAndToyDataset(model::proto::Task::CLASSIFICATION, &model,
                             &dataset);

  std::vector<internal::PredictionAccumulator> predictions;
  internal::InitializeOOBPredictionAccumulators(
      dataset.nrow(), config, config_link, dataset.data_spec(), &predictions);
  EXPECT_EQ(predictions.size(), dataset.nrow());

  std::vector<dataset::VerticalDataset::row_t> sorted_non_oob_example_indices =
      {1};
  internal::UpdateOOBPredictionsWithNewTree(
      dataset, config, sorted_non_oob_example_indices, true,
      *model.decision_trees()[0].get(), {}, &rnd, &predictions);
  EXPECT_EQ(predictions[0].num_trees, 1);
  EXPECT_EQ(predictions[0].classification.NumObservations(), 1);
  EXPECT_EQ(predictions[0].classification.TopClass(), 1);

  EXPECT_EQ(predictions[1].num_trees, 0);

  EXPECT_EQ(predictions[2].num_trees, 1);
  EXPECT_EQ(predictions[2].classification.NumObservations(), 1);
  EXPECT_EQ(predictions[2].classification.TopClass(), 0);

  const auto evaluation_1 = internal::EvaluateOOBPredictions(
      dataset, config.task(), config_link.label(), -1, {}, predictions);
  EXPECT_EQ(internal::EvaluationSnippet(evaluation_1),
            "accuracy:0.5 logloss:18.0218");

  internal::UpdateOOBPredictionsWithNewTree(
      dataset, config, sorted_non_oob_example_indices, true,
      *model.decision_trees()[1].get(), {}, &rnd, &predictions);
  EXPECT_EQ(predictions[0].num_trees, 2);
  EXPECT_EQ(predictions[0].classification.NumObservations(), 2);
  EXPECT_EQ(predictions[0].classification.TopClass(), 1);

  EXPECT_EQ(predictions[1].num_trees, 0);

  EXPECT_EQ(predictions[2].num_trees, 2);
  EXPECT_EQ(predictions[2].classification.NumObservations(), 2);
  EXPECT_EQ(predictions[2].classification.TopClass(), 0);

  const auto evaluation_2 = internal::EvaluateOOBPredictions(
      dataset, config.task(), config_link.label(), -1, {}, predictions);
  EXPECT_EQ(internal::EvaluationSnippet(evaluation_2),
            "accuracy:0.5 logloss:18.0218");
}

TEST(RandomForest, ComputeVariableImportancesFromAccumulatedPredictions) {
  const model::proto::TrainingConfig config = PARSE_TEST_PROTO(R"pb(
    task: CLASSIFICATION
  )pb");
  const model::proto::TrainingConfigLinking config_link = PARSE_TEST_PROTO(R"pb(
    label: 1
    num_label_classes: 3
  )pb");

  utils::RandomEngine rnd;
  RandomForestModel model;
  dataset::VerticalDataset dataset;
  BuildToyModelAndToyDataset(model::proto::Task::CLASSIFICATION, &model,
                             &dataset);

  std::vector<internal::PredictionAccumulator> oob_predictions;
  std::vector<std::vector<internal::PredictionAccumulator>>
      oob_predictions_per_input_features(2);

  internal::InitializeOOBPredictionAccumulators(
      dataset.nrow(), config, config_link, dataset.data_spec(),
      &oob_predictions);
  internal::InitializeOOBPredictionAccumulators(
      dataset.nrow(), config, config_link, dataset.data_spec(),
      &oob_predictions_per_input_features[0]);

  std::vector<dataset::VerticalDataset::row_t> sorted_non_oob_example_indices =
      {1};

  // Baseline
  internal::UpdateOOBPredictionsWithNewTree(
      dataset, config, sorted_non_oob_example_indices, true,
      *model.decision_trees()[0].get(), {}, &rnd, &oob_predictions);
  internal::UpdateOOBPredictionsWithNewTree(
      dataset, config, sorted_non_oob_example_indices, true,
      *model.decision_trees()[1].get(), {}, &rnd, &oob_predictions);

  // Shuffled
  for (int repetition = 0; repetition < 100; repetition++) {
    internal::UpdateOOBPredictionsWithNewTree(
        dataset, config, sorted_non_oob_example_indices, true,
        *model.decision_trees()[0].get(), 0, &rnd,
        &oob_predictions_per_input_features[0]);
    internal::UpdateOOBPredictionsWithNewTree(
        dataset, config, sorted_non_oob_example_indices, true,
        *model.decision_trees()[1].get(), 0, &rnd,
        &oob_predictions_per_input_features[0]);
  }

  // Compute importance.
  internal::ComputeVariableImportancesFromAccumulatedPredictions(
      oob_predictions, oob_predictions_per_input_features, dataset, &model);

  // Ground truth: 1, 1
  // Baseline prediction: 1, 0
  //
  // Baseline has 0.5 accuracy.
  // Shuffled has 1.0 accuracy (per chance).
  const auto importance =
      model.GetVariableImportance("MEAN_DECREASE_IN_ACCURACY").value();
  EXPECT_EQ(importance.size(), 1);
  EXPECT_EQ(importance[0].attribute_idx(), 0);
  EXPECT_EQ(importance[0].importance(), -0.5);
}

// We train a 100-trees regressive RF and ERT on 20 examples. The RF predictions
// are expected to be very "stairy" while the ERT predictions smoothly
// interpolate the training examples.
//
// This test is similar to the figure 10 in the Extremely Randomized Trees
// paper.
//
// The test also checks the RMSE between the model predictions and a linear
// interpolation.
TEST(ExtremelyRandomizeTrees, Figure10) {
  // Generate the training and testing datasets.
  dataset::VerticalDataset train_dataset, test_dataset;
  utils::RandomEngine random(123456);
  ExtremelyRandomizeTreesFigure10Dataset(20, &train_dataset, &random);
  ExtremelyRandomizeTreesFigure10Dataset(1000, &test_dataset, &random);

  const model::proto::TrainingConfig config_rf = PARSE_TEST_PROTO(R"pb(
    learner: "RANDOM_FOREST"
    task: REGRESSION
    label: "y"
    [yggdrasil_decision_forests.model.random_forest.proto
         .random_forest_config] {
      num_trees: 100
      decision_tree { max_depth: -1 min_examples: 1 }
    }
  )pb");
  const model::proto::TrainingConfig config_ert = PARSE_TEST_PROTO(R"pb(
    learner: "RANDOM_FOREST"
    task: REGRESSION
    label: "y"
    [yggdrasil_decision_forests.model.random_forest.proto
         .random_forest_config] {
      num_trees: 100
      bootstrap_training_dataset: false
      decision_tree {
        numerical_split { type: HISTOGRAM_RANDOM }
        max_depth: -1
        min_examples: 1
      }
    }
  )pb");

  // Train the models.
  std::unique_ptr<model::AbstractLearner> rf_learner, ert_learner;
  CHECK_OK(model::GetLearner(config_rf, &rf_learner));
  CHECK_OK(model::GetLearner(config_ert, &ert_learner));
  const auto ert_model = ert_learner->TrainWithStatus(train_dataset).value();
  const auto rf_model = rf_learner->TrainWithStatus(train_dataset).value();

  std::string ert_description, rf_description;
  ert_model->AppendDescriptionAndStatistics(false, &ert_description);
  rf_model->AppendDescriptionAndStatistics(false, &rf_description);
  LOG(INFO) << "ERT model:\n" << ert_description;
  LOG(INFO) << "RF model:\n" << rf_description;

  dataset::VerticalDataset test_and_preds_dataset =
      test_dataset.ShallowNonOwningClone();
  auto* rf_preds = dynamic_cast<dataset::VerticalDataset::NumericalColumn*>(
      test_and_preds_dataset
          .AddColumn(PARSE_TEST_PROTO(R"pb(
            type: NUMERICAL name: "rf"
          )pb"))
          .value());
  auto* ert_preds = dynamic_cast<dataset::VerticalDataset::NumericalColumn*>(
      test_and_preds_dataset
          .AddColumn(PARSE_TEST_PROTO(R"pb(
            type: NUMERICAL name: "ert"
          )pb"))
          .value());

  // Generate the predictions.
  for (dataset::VerticalDataset::row_t example_idx = 0;
       example_idx < test_dataset.nrow(); example_idx++) {
    model::proto::Prediction pred;
    rf_model->Predict(test_dataset, example_idx, &pred);
    (*rf_preds->mutable_values())[example_idx] = pred.regression().value();
    ert_model->Predict(test_dataset, example_idx, &pred);
    (*ert_preds->mutable_values())[example_idx] = pred.regression().value();
  }

  // Evaluate the RMSE between the predictions and a linear interpolation.
  const auto& train_xs =
      train_dataset
          .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
              train_dataset.ColumnNameToColumnIdx("x"))
          ->values();
  const auto& train_ys =
      train_dataset
          .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
              train_dataset.ColumnNameToColumnIdx("y"))
          ->values();
  const auto& test_xs =
      test_dataset
          .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
              test_dataset.ColumnNameToColumnIdx("x"))
          ->values();

  // List of <input,output> of the training examples sorted by increasing input
  // value (so we can use a binary search).
  std::vector<std::pair<float, float>> sorted_training_examples;
  sorted_training_examples.reserve(train_xs.size());
  for (dataset::VerticalDataset::row_t example_idx = 0;
       example_idx < train_xs.size(); example_idx++) {
    sorted_training_examples.push_back(
        {train_xs[example_idx], train_ys[example_idx]});
  }
  std::sort(sorted_training_examples.begin(), sorted_training_examples.end());

  // Computes the RMSE between a model predictions and a linear interpolation
  // (of the training examples).
  auto get_rmse = [&](const std::vector<float>& predictions) -> double {
    double sum_squared_error = 0;
    // Only the testing examples in between two training examples (i.e. not
    // outside the bounds) are valid.
    int num_valid_examples = 0;
    for (dataset::VerticalDataset::row_t example_idx = 0;
         example_idx < test_dataset.nrow(); example_idx++) {
      const float prediction = predictions[example_idx];
      // The input feature value of the  example "example_idx" is between
      // "(it_upper_bound-1).first" and "it_upper_bound.first".
      const auto it_upper_bound = std::upper_bound(
          sorted_training_examples.begin(), sorted_training_examples.end(),
          std::pair<float, float>{test_xs[example_idx], 0.f});
      if (it_upper_bound == sorted_training_examples.end() ||
          it_upper_bound == sorted_training_examples.begin()) {
        continue;
      }
      // Linear interpolation between "it_lower_bound" and "it_upper_bound".
      const auto it_lower_bound = it_upper_bound - 1;
      const float x_interpolation =
          (test_xs[example_idx] - it_lower_bound->first) /
          (it_upper_bound->first - it_lower_bound->first);
      const float y_interpolation =
          it_upper_bound->second * x_interpolation +
          it_lower_bound->second * (1 - x_interpolation);
      const float error = prediction - y_interpolation;
      sum_squared_error += error * error;
      num_valid_examples++;
    }
    return std::sqrt(sum_squared_error / num_valid_examples);
  };

  // Compute the RMSE between the model prediction and a linear interpolation.
  const double rf_rmse = get_rmse(rf_preds->values());
  const double ert_rmse = get_rmse(ert_preds->values());

  // ERT's RMSE to the linear interpolation model is significantly lower than
  // RF's RMSE (~5x less).
  EXPECT_NEAR(rf_rmse, 0.0582, 0.005);
  EXPECT_NEAR(ert_rmse, 0.0121, 0.005);

  // Export the predictions for the plotting.
  CHECK_OK(SaveVerticalDataset(
      test_and_preds_dataset,
      absl::StrCat("csv:", file::JoinPath(test::TmpDirectory(),
                                          "ert_figure10_test.csv")),
      -1));
  CHECK_OK(SaveVerticalDataset(
      train_dataset,
      absl::StrCat("csv:", file::JoinPath(test::TmpDirectory(),
                                          "ert_figure10_train.csv")),
      -1));

  // The plot can then be generated by running the following commands in R:
  //  test = read.csv("ert_figure10_test.csv")
  //  train = read.csv("ert_figure10_train.csv")
  //  test = test[order(test$x), ]
  //  train = train[order(train), ]
  //  plot(train$x, train$y, xlab = "input", ylab = "output", ylim = c(0, 1))
  //  lines(test$x, test$rf, col = "blue")
  //  lines(test$x, test$ert, col = "red")
  //  lines(test$x, test$y, col = "green")
  //  legend( 2.3, 0.6, legend = c("RF", "ERT", "True"), col = c("blue", "red",
  //    "green"), lty = c(1, 1, 1))
}

class RandomForestOnIris : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(RandomForestLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    train_config_.set_label("class");
    dataset_filename_ = "iris.csv";
    auto* rf_config = train_config_.MutableExtension(
        random_forest::proto::random_forest_config);
    rf_config->mutable_decision_tree()
        ->set_internal_error_on_wrong_splitter_statistics(true);
  }
};

TEST_F(RandomForestOnIris, Base) {
  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.9466, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.1072, 0.04);
  // Note: R RandomForest has an OOB (out-of-bag) accuracy of 0.9467.
}

// Helper for the training and testing on the DNA dataset.
class RandomForestOnDNA : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(RandomForestLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    train_config_.set_label("LABEL");
    dataset_filename_ = "dna.csv";
  }
};

TEST_F(RandomForestOnDNA, Boolean) {
  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.9466, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.2973, 0.04);
}

TEST_F(RandomForestOnDNA, BooleanAsNumerical) {
  guide_filename_ = "dna_guide.pbtxt";
  TrainAndEvaluateModel();
  // For comparison, the RF model learned in R yields an accuracy of 0.909.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.946641, 0.01);
  EXPECT_NEAR(metric::LogLoss(evaluation_), 0.2973, 0.04);
}

class RandomForestOnSyntheticClassification : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(RandomForestLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::CLASSIFICATION);
    synthetic_dataset_.set_num_examples(10000);
    ConfigureForSyntheticDataset();
  }
};

TEST_F(RandomForestOnSyntheticClassification, Base) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->set_winner_take_all_inference(false);
  rf_config->mutable_decision_tree()->mutable_categorical()->mutable_random();
  TrainAndEvaluateModel();
  // This test has a lot of variance because of the small number of examples
  // , large size of the feature space, and procedurally generated dataset.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.764, 0.03);
}

TEST(RandomForest, PredefinedHyperParameters) {
  model::proto::TrainingConfig train_config;
  train_config.set_learner(RandomForestLearner::kRegisteredName);
  utils::TestPredefinedHyperParametersAdultDataset(train_config, 2, 0.86);
}

class RandomForestOnSimPTE : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(RandomForestLearner::kRegisteredName);
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

TEST_F(RandomForestOnSimPTE, Base) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  const auto oob_prediction_path =
      file::JoinPath(test::TmpDirectory(), "oob_predictions.csv");
  rf_config->set_export_oob_prediction_path(
      absl::StrCat("csv:", oob_prediction_path));

  TrainAndEvaluateModel();
  // Note: A Qini of ~0.1 is expected with a simple Random Forest model.
  EXPECT_NEAR(metric::Qini(evaluation_), 0.105709, 0.001);

  // Export the labels+predictions for external evaluation.
  const auto uplift_pred_csv_path =
      file::JoinPath(test::TmpDirectory(), "uplift_pred.csv");
  CHECK_OK(utils::ExportUpliftPredictionsToTFUpliftCsvFormat(
      *model_, test_dataset_, uplift_pred_csv_path));

  // Check the oob predictions.
  const auto oob_predictions =
      file::GetContents(oob_prediction_path, file::Defaults()).value();
  EXPECT_TRUE(absl::StartsWith(oob_predictions, "2\n"));
  EXPECT_EQ(std::count(oob_predictions.begin(), oob_predictions.end(), '\n'),
            train_dataset_.nrow() + 1 /*the header*/);
}

TEST_F(RandomForestOnSimPTE, Honest) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->mutable_decision_tree()->mutable_honest();
  rf_config->set_sampling_with_replacement(false);
  rf_config->set_bootstrap_size_ratio(0.5);

  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Qini(evaluation_), 0.106705, 0.002);
}

TEST_F(RandomForestOnSimPTE, LowerBound) {
  auto* rf_config = train_config_.MutableExtension(
      random_forest::proto::random_forest_config);
  rf_config->mutable_decision_tree()->mutable_uplift()->set_split_score(
      decision_tree::proto::DecisionTreeTrainingConfig::Uplift::
          CONSERVATIVE_EUCLIDEAN_DISTANCE);
  TrainAndEvaluateModel();
  EXPECT_NEAR(metric::Qini(evaluation_), 0.10889, 0.002);
}

TEST(SampleTrainingExamples, WithReplacement) {
  utils::RandomEngine random;
  std::vector<dataset::VerticalDataset::row_t> examples;
  internal::SampleTrainingExamples(100, 50, /*with_replacement=*/true, &random,
                                   &examples);
  EXPECT_EQ(examples.size(), 50);
  EXPECT_TRUE(std::is_sorted(examples.begin(), examples.end()));
}

TEST(SampleTrainingExamples, WithoutReplacement) {
  utils::RandomEngine random;
  std::vector<dataset::VerticalDataset::row_t> examples;
  internal::SampleTrainingExamples(100, 50, /*with_replacement=*/false, &random,
                                   &examples);
  EXPECT_EQ(examples.size(), 50);
  EXPECT_TRUE(std::is_sorted(examples.begin(), examples.end()));
  // Values are unique.
  EXPECT_TRUE(std::adjacent_find(examples.begin(), examples.end()) ==
              examples.end());
}

class RandomForestOnRegressiveSimPTE : public utils::TrainAndTestTester {
  void SetUp() override {
    train_config_.set_learner(RandomForestLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::NUMERICAL_UPLIFT);
    train_config_.set_label("y");
    train_config_.set_uplift_treatment("treat");

    guide_ = PARSE_TEST_PROTO(
        R"pb(
          column_guides {
            column_name_pattern: "treat"
            type: CATEGORICAL
            categorial { is_already_integerized: true }
          }
          detect_boolean_as_numerical: true
        )pb");

    dataset_filename_ = "sim_pte_train.csv";
    dataset_test_filename_ = "sim_pte_test.csv";
  }
};

TEST_F(RandomForestOnRegressiveSimPTE, Base) {
  TrainAndEvaluateModel();
  // Note: The labels of this dataset are in {1,2}. Therefore, regressive
  // uplift (this test) is not perfectly equal to categorical uplift
  // (RandomForestOnSimPTE.Base) test, and the Qini score of this test is not
  // exactly the same as the Qini score of categorical uplift test. If the
  // labels of this dataset were to be replaced with {0,1}, the scores would be
  // equal (tested).
  EXPECT_NEAR(metric::Qini(evaluation_), 0.095192, 0.008);
}

}  // namespace
}  // namespace random_forest
}  // namespace model
}  // namespace yggdrasil_decision_forests
