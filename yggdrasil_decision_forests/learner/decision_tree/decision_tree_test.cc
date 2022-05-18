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

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/generic_parameters.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {
namespace {

using test::EqualsProto;

using row_t = dataset::VerticalDataset::row_t;

std::string DatasetDir() {
  return file::JoinPath(
      test::DataRootDirectory(),
      "yggdrasil_decision_forests/test_data/dataset");
}

struct FakeLabelStats : LabelStats {};

// A fake consumer that persistently fails to find a valid attribute.
SplitterWorkResponse FakeFindBestConditionConcurrentConsumerAlwaysInvalid(
    SplitterWorkRequest request) {
  return SplitterWorkResponse{
      .status_idx = request.status_idx,
      .condition = request.dst_condition,
      .status = SplitSearchResult::kInvalidAttribute,
  };
}

// A fake consumer that sets the split score to 10 times the request index.
SplitterWorkResponse FakeFindBestConditionConcurrentConsumerMultiplicative(
    SplitterWorkRequest request) {
  SplitterWorkResponse response{
      .status_idx = request.status_idx,
      .condition = request.dst_condition,
      .status = SplitSearchResult::kNoBetterSplitFound,
  };
  response.condition->set_split_score(request.attribute_idx * 10.f);
  return response;
}

// A fake consumer that fails if the attribute_idx is even, and set the score to
// 10 times the attribute_idx otherwise.
SplitterWorkResponse FakeFindBestConditionConcurrentConsumerAlternate(
    SplitterWorkRequest request) {
  auto response = SplitterWorkResponse{
      .status_idx = request.status_idx,
      .condition = request.dst_condition,
      .status = SplitSearchResult::kNoBetterSplitFound,
  };
  if (request.attribute_idx % 2 == 0) {
    response.status = SplitSearchResult::kInvalidAttribute;
  }
  response.condition->set_split_score(request.attribute_idx * 10.f);
  return response;
}

TEST(DecisionTree, FakeTrain) {
  const std::string ds_typed_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "adult.csv"));
  dataset::proto::DataSpecification data_spec;
  dataset::proto::DataSpecificationGuide guide;
  dataset::CreateDataSpec(ds_typed_path, false, guide, &data_spec);

  dataset::VerticalDataset train_dataset;
  CHECK_OK(LoadVerticalDataset(ds_typed_path, data_spec, &train_dataset));

  std::vector<row_t> selected_examples(train_dataset.nrow());
  std::iota(selected_examples.begin(), selected_examples.end(), 0);

  const std::vector<float> weights(train_dataset.nrow(), 1.f);

  model::proto::TrainingConfig config;
  config.set_task(model::proto::Task::CLASSIFICATION);
  config.set_label("income");
  config.add_features(".*");

  const model::proto::DeploymentConfig deployment;

  model::proto::TrainingConfigLinking config_link;
  CHECK_OK(
      AbstractLearner::LinkTrainingConfig(config, data_spec, &config_link));

  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.set_internal_error_on_wrong_splitter_statistics(true);

  utils::RandomEngine random;
  DecisionTree dt;
  CHECK_OK(Train(train_dataset, selected_examples, config, config_link,
                 dt_config, deployment, weights, &random, &dt, {}));
}

TEST(DecisionTree, FindBestNumericalSplitCartBase) {
  const std::vector<row_t> selected_examples = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  const float na = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> attributes = {2, 3, 0, 1, na, na};
  const std::vector<int32_t> labels = {1, 1, 0, 0, 1, 0};
  const int32_t num_label_classes = 2;
  const float na_replacement = 2;
  const row_t min_num_obs = 1;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::IN_NODE);
  utils::IntegerDistributionDouble label_distribution;
  label_distribution.SetNumClasses(num_label_classes);
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }
  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  EXPECT_EQ(FindSplitLabelClassificationFeatureNumericalCart(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, {}, &best_condition, &cache),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().higher_condition().threshold(), 1.5f);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 6);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 4);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 4);
  EXPECT_EQ(best_condition.na_value(), true);

  EXPECT_EQ(FindSplitLabelClassificationFeatureNumericalCart(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, {}, &best_condition, &cache),
            SplitSearchResult::kNoBetterSplitFound);

  attributes = {1, 1, 1, 1, 1, 1};
  EXPECT_EQ(FindSplitLabelClassificationFeatureNumericalCart(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, {}, &best_condition, &cache),
            SplitSearchResult::kInvalidAttribute);
}

TEST(DecisionTree, FindSplitLabelClassificationFeatureNumericalHistogram) {
  const std::vector<row_t> selected_examples = {0, 1, 2};
  const std::vector<float> weights = {1.f, 1.f, 1.f};
  const float na = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> attributes = {0, 1, na};
  const std::vector<int32_t> labels = {0, 1, 0};
  const int32_t num_label_classes = 2;
  const float na_replacement = 2;
  const row_t min_num_obs = 1;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_numerical_split()->set_type(
      proto::NumericalSplit::HISTOGRAM_RANDOM);
  dt_config.mutable_numerical_split()->set_num_candidates(1);
  utils::IntegerDistributionDouble label_distribution;
  label_distribution.SetNumClasses(num_label_classes);
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }
  proto::NodeCondition best_condition;
  utils::RandomEngine random(123456);
  const auto saved_state = random;

  std::uniform_real_distribution<float> threshold_distribution(0, 1);
  const float threshold = threshold_distribution(random);

  random = saved_state;
  EXPECT_EQ(FindSplitLabelClassificationFeatureNumericalHistogram(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, &random, &best_condition),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().higher_condition().threshold(),
            threshold);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 3);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 3);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 2);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 2);
  EXPECT_EQ(best_condition.na_value(), true);

  random = saved_state;
  EXPECT_EQ(FindSplitLabelClassificationFeatureNumericalHistogram(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, &random, &best_condition),
            SplitSearchResult::kNoBetterSplitFound);

  attributes = {1, 1, 1, 1, 1, 1};
  random = saved_state;
  EXPECT_EQ(FindSplitLabelClassificationFeatureNumericalHistogram(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, &random, &best_condition),
            SplitSearchResult::kInvalidAttribute);
}

TEST(DecisionTree, FindBestNumericalSplitCartWeighted) {
  const std::vector<row_t> selected_examples = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights = {2.f, 1.f, 1.f, 1.f, 1.f, 3.f};
  const float na = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> attributes = {2, 3, 0, 1, na, na};
  const std::vector<int32_t> labels = {1, 1, 0, 0, 1, 0};
  const int32_t num_label_classes = 2;
  const float na_replacement = 2;
  const row_t min_num_obs = 1;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::IN_NODE);
  utils::IntegerDistributionDouble label_distribution;
  label_distribution.SetNumClasses(num_label_classes);
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }
  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  EXPECT_EQ(FindSplitLabelClassificationFeatureNumericalCart(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, {}, &best_condition, &cache),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().higher_condition().threshold(), 1.5f);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 9);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 4);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 7);
  EXPECT_TRUE(best_condition.na_value());
}

// Tests the case of two distinct attribute values with the mean of the two
// values being equal to the first value because of float limited precision:
// i.e. a == (a+b)/2 with a<b.
TEST(DecisionTree, FindBestNumericalSplitCartVeryClose) {
  const float a = 0.1234567;
  const float b = nextafterf(a, std::numeric_limits<float>::infinity());
  const std::vector<row_t> selected_examples = {0, 1};
  const std::vector<float> weights = {1, 1};
  const std::vector<float> attributes = {a, b};
  const std::vector<int32_t> labels = {0, 1};
  const int32_t num_label_classes = 2;
  const float na_replacement = 2;
  const row_t min_num_obs = 1;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::IN_NODE);
  utils::IntegerDistributionDouble label_distribution;
  label_distribution.SetNumClasses(num_label_classes);
  for (const int32_t label : labels) {
    label_distribution.Add(label);
  }
  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  EXPECT_EQ(FindSplitLabelClassificationFeatureNumericalCart(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, {}, &best_condition, &cache),
            SplitSearchResult::kBetterSplitFound);
  EXPECT_EQ(best_condition.condition().higher_condition().threshold(), b);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 2);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 2);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 1);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 1);
  EXPECT_EQ(best_condition.na_value(), true);
}

TEST(DecisionTree, FindBestCategoricalSplitCartBaseBasic) {
  // Small basic dataset.
  const std::vector<row_t> selected_examples = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights = {1, 1, 1, 1, 1, 1};
  std::vector<int32_t> attributes = {2, 3, 0, 1, -1, -1};
  const std::vector<int32_t> labels = {1, 1, 0, 0, 1, 0};
  const int32_t num_label_classes = 2;
  const int32_t num_attribute_classes = 4;

  // Configuration.
  const int32_t na_replacement = 1;
  const row_t min_num_obs = 1;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::IN_NODE);

  // Compute the label distribution.
  utils::IntegerDistributionDouble label_distribution;
  label_distribution.SetNumClasses(num_label_classes);
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }

  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  utils::RandomEngine rnd;
  EXPECT_EQ(
      FindSplitLabelClassificationFeatureCategorical(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, -1, &rnd, &best_condition, &cache),
      SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().type_case(),
            proto::Condition::kContainsBitmapConditionFieldNumber);
  // The expected element map is "0011".
  EXPECT_EQ(
      best_condition.condition().contains_bitmap_condition().elements_bitmap(),
      "\003");
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 6);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 4);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 4);
  EXPECT_EQ(best_condition.na_value(), true);
  // R> entropy(c(3,3)) - 4/6*entropy(c(3,1)) - 2/6*entropy(c(2)) = 0.3182571
  EXPECT_NEAR(best_condition.split_score(), 0.3182571, 0.0001);
  EXPECT_EQ(
      FindSplitLabelClassificationFeatureCategorical(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, -1, &rnd, &best_condition, &cache),
      SplitSearchResult::kNoBetterSplitFound);
  LOG(INFO) << "Condition:\n" << best_condition.DebugString();

  // Since all the attributes have the same value, there are no valid splits.
  attributes = {1, 1, 1, 1, 1, 1};

  EXPECT_EQ(
      FindSplitLabelClassificationFeatureCategorical(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, -1, &rnd, &best_condition, &cache),
      SplitSearchResult::kInvalidAttribute);
  LOG(INFO) << "Condition:\n" << best_condition.DebugString();
}

TEST(DecisionTree, FindBestCategoricalSplitCartBaseWithWeights) {
  // Small basic dataset.
  const std::vector<row_t> selected_examples = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights = {2, 1, 1, 1, 1, 3};
  std::vector<int32_t> attributes = {2, 3, 0, 1, -1, -1};
  const std::vector<int32_t> labels = {1, 1, 0, 0, 1, 0};
  const int32_t num_label_classes = 2;
  const int32_t num_attribute_classes = 4;

  // Configuration.
  const int32_t na_replacement = 1;
  const row_t min_num_obs = 1;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::IN_NODE);

  // Compute the label distribution.
  utils::IntegerDistributionDouble label_distribution;
  label_distribution.SetNumClasses(num_label_classes);
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }

  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  utils::RandomEngine rnd;
  EXPECT_EQ(
      FindSplitLabelClassificationFeatureCategorical(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, -1, &rnd, &best_condition, &cache),
      SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().type_case(),
            proto::Condition::kContainsBitmapConditionFieldNumber);
  // The expected element map is "0011".
  EXPECT_EQ(
      best_condition.condition().contains_bitmap_condition().elements_bitmap(),
      "\003");
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 9);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 4);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 6);
  EXPECT_EQ(best_condition.na_value(), true);
  // R> entropy(c(4,5)) - 3/9*entropy(c(3)) - 6/9*entropy(c(1,5)) = 0.3865874
  EXPECT_NEAR(best_condition.split_score(), 0.3865874, 0.0001);

  EXPECT_EQ(
      FindSplitLabelClassificationFeatureCategorical(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, -1, &rnd, &best_condition, &cache),
      SplitSearchResult::kNoBetterSplitFound);
  LOG(INFO) << "Condition:\n" << best_condition.DebugString();

  // Since all the attributes have the same value, there are no valid splits.
  attributes = {1, 1, 1, 1, 1, 1};

  EXPECT_EQ(
      FindSplitLabelClassificationFeatureCategorical(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, -1, &rnd, &best_condition, &cache),
      SplitSearchResult::kInvalidAttribute);
  LOG(INFO) << "Condition:\n" << best_condition.DebugString();
}

TEST(DecisionTree, FindBestCategoricalSplitCartBaseAdvances) {
  // Configuration.
  for (const int32_t num_label_classes : {2, 4, 8}) {
    for (const int32_t num_attribute_classes : {2, 4, 200, 1000}) {
      const int32_t na_replacement = 1;
      const row_t min_num_obs = 1;
      const row_t num_examples = 10000;
      proto::DecisionTreeTrainingConfig dt_config;
      dt_config.mutable_internal()->set_sorting_strategy(
          proto::DecisionTreeTrainingConfig::Internal::IN_NODE);

      // Generate random attribute and label values.
      std::uniform_int_distribution<int32_t> attribute_dist(
          0, num_attribute_classes - 1);
      std::uniform_int_distribution<int32_t> label_dist(0,
                                                        num_label_classes - 1);
      utils::RandomEngine rnd;

      std::vector<row_t> selected_examples;
      std::vector<float> weights;
      std::vector<int32_t> attributes;
      std::vector<int32_t> labels;
      utils::IntegerDistributionDouble label_distribution;
      label_distribution.SetNumClasses(num_label_classes);
      selected_examples.reserve(num_examples);
      attributes.reserve(num_examples);
      labels.reserve(num_examples);
      for (row_t example_idx = 0; example_idx < num_examples; example_idx++) {
        selected_examples.push_back(example_idx);
        attributes.push_back(attribute_dist(rnd));
        const auto label = label_dist(rnd);
        labels.push_back(label);
        const float weight = 1.f;
        weights.push_back(weight);
        label_distribution.Add(label, weight);
      }

      // Look for the best condition.
      proto::NodeCondition best_condition;
      SplitterPerThreadCache cache;
      EXPECT_EQ(FindSplitLabelClassificationFeatureCategorical(
                    selected_examples, weights, attributes, labels,
                    num_attribute_classes, num_label_classes, na_replacement,
                    min_num_obs, dt_config, label_distribution, -1, &rnd,
                    &best_condition, &cache),
                SplitSearchResult::kBetterSplitFound);
      LOG(INFO) << "num_label_classes:" << num_label_classes
                << " num_attribute_classes:" << num_attribute_classes;
      LOG(INFO) << "Condition:\n" << best_condition.DebugString();
    }
  }
}

// The random categorical splitter is non deterministic (unless the see is
// fixed). This test (1) ensure that the random categorical splitter runs, (2)
// that the categorical splitter return splits, (3) and that the categorical
// splitter splits are always worst that the CART splitter (in the case of
// binary classification).
TEST(DecisionTree, FindBestCategoricalSplitRandom) {
  SplitterPerThreadCache cache;
  utils::RandomEngine rnd;

  const int num_attribute_classes = 100;
  const int num_label_classes = 2;
  const int32_t na_replacement = 1;
  const row_t min_num_obs = 1;
  const row_t num_examples = 10000;

  proto::DecisionTreeTrainingConfig random_dt_config;
  random_dt_config.mutable_categorical()->mutable_random();

  // Generate random attribute and label values.
  std::uniform_int_distribution<int32_t> attribute_dist(
      0, num_attribute_classes - 1);
  std::uniform_int_distribution<int32_t> label_dist(0, num_label_classes - 1);

  for (int trial_idx = 0; trial_idx < 10; trial_idx++) {
    std::vector<row_t> selected_examples;
    std::vector<float> weights;
    std::vector<int32_t> attributes;
    std::vector<int32_t> labels;
    utils::IntegerDistributionDouble label_distribution;
    label_distribution.SetNumClasses(num_label_classes);
    selected_examples.reserve(num_examples);
    attributes.reserve(num_examples);
    labels.reserve(num_examples);
    for (row_t example_idx = 0; example_idx < num_examples; example_idx++) {
      selected_examples.push_back(example_idx);
      attributes.push_back(attribute_dist(rnd));
      const auto label = label_dist(rnd);
      labels.push_back(label);
      const float weight = 1.f;
      weights.push_back(weight);
      label_distribution.Add(label, weight);
    }

    // Look for the best condition.
    proto::NodeCondition cart_condition;
    EXPECT_EQ(FindSplitLabelClassificationFeatureCategorical(
                  selected_examples, weights, attributes, labels,
                  num_attribute_classes, num_label_classes, na_replacement,
                  min_num_obs, {}, label_distribution, -1, &rnd,
                  &cart_condition, &cache),
              SplitSearchResult::kBetterSplitFound);

    proto::NodeCondition random_condition;
    EXPECT_EQ(FindSplitLabelClassificationFeatureCategorical(
                  selected_examples, weights, attributes, labels,
                  num_attribute_classes, num_label_classes, na_replacement,
                  min_num_obs, random_dt_config, label_distribution, -1, &rnd,
                  &random_condition, &cache),
              SplitSearchResult::kBetterSplitFound);

    EXPECT_LE(random_condition.split_score(), cart_condition.split_score());
  }
}

TEST(DecisionTree, CategoricalSplitConditionStorageVector) {
  const size_t num_elements = 500;
  std::vector<std::pair<float, int32_t>> ratio_true_label_by_attr_value;
  ratio_true_label_by_attr_value.reserve(num_elements);
  for (int i = 0; i < num_elements; i++) {
    // The smaller the element index, the larger the ratio of positive label.
    ratio_true_label_by_attr_value.push_back({i, num_elements - i - 1});
  }
  EXPECT_TRUE(std::is_sorted(ratio_true_label_by_attr_value.begin(),
                             ratio_true_label_by_attr_value.end()));
  proto::NodeCondition condition;
  SetPositiveAttributeSetOfCategoricalContainsCondition(
      ratio_true_label_by_attr_value, 2, num_elements, &condition);
  EXPECT_EQ(condition.condition().type_case(),
            proto::Condition::kContainsBitmapConditionFieldNumber);

  SetPositiveAttributeSetOfCategoricalContainsCondition(
      ratio_true_label_by_attr_value, 251, num_elements, &condition);
  EXPECT_EQ(condition.condition().type_case(),
            proto::Condition::kContainsBitmapConditionFieldNumber);

  SetPositiveAttributeSetOfCategoricalContainsCondition(
      ratio_true_label_by_attr_value, 496, num_elements, &condition);
  EXPECT_EQ(condition.condition().type_case(),
            proto::Condition::kContainsConditionFieldNumber);
  EXPECT_TRUE(std::is_sorted(
      condition.condition().contains_condition().elements().begin(),
      condition.condition().contains_condition().elements().end()));
}

TEST(DecisionTree, GetCandidateAttributes) {
  model::proto::TrainingConfig config;
  proto::DecisionTreeTrainingConfig dt_config;
  utils::RandomEngine random;
  model::proto::TrainingConfigLinking config_link;
  config_link.add_features(0);
  config_link.add_features(1);
  config_link.add_features(2);
  config_link.add_features(3);

  int num_attributes_to_test;
  std::vector<int32_t> candidate_attributes;

  GetCandidateAttributes(config, config_link, dt_config,
                         &num_attributes_to_test, &candidate_attributes,
                         &random);

  CHECK_EQ(candidate_attributes.size(), 4);
  CHECK_EQ(num_attributes_to_test, 2);
}

TEST(DecisionTree, FindBestConditionClassification) {
  dataset::VerticalDataset dataset;
  dataset::proto::DataSpecification dataspec = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "a" }
    columns {
      type: CATEGORICAL
      name: "b"
      categorical { is_already_integerized: true number_of_unique_values: 3 }
    }
    columns {
      type: CATEGORICAL
      name: "label"
      categorical { is_already_integerized: true number_of_unique_values: 3 }
    }
  )pb");

  dataset.set_data_spec(dataspec);
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  auto* col_1 =
      dataset.MutableColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
          0);
  col_1->Add(0);
  col_1->Add(2);

  auto* col_2 =
      dataset
          .MutableColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
              1);
  col_2->Add(1);
  col_2->Add(2);

  auto* col_3 =
      dataset
          .MutableColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
              2);
  col_3->Add(1);
  col_3->Add(2);

  const std::vector<row_t> selected_examples = {0, 1};
  const std::vector<float> weights = {1.f, 1.f};
  DecisionTree tree;
  tree.CreateRoot();
  auto* label_distribution = tree.mutable_root()
                                 ->mutable_node()
                                 ->mutable_classifier()
                                 ->mutable_distribution();
  label_distribution->set_sum(2);
  label_distribution->add_counts(0);
  label_distribution->add_counts(1);
  label_distribution->add_counts(1);

  model::proto::TrainingConfig config;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.set_min_examples(1);
  utils::RandomEngine random(1234);
  model::proto::TrainingConfigLinking config_link;
  config_link.add_features(0);
  config_link.add_features(1);
  config_link.set_label(2);

  proto::NodeCondition condition;
  PerThreadCache cache;
  EXPECT_TRUE(FindBestCondition(dataset, selected_examples, weights, config,
                                config_link, dt_config, {}, tree.root().node(),
                                {}, &condition, &random, &cache)
                  .value());

  // We test that a condition was created on attribute 0 or 1 (non
  // deterministic). If the attribute 0 is selected, the condition should be
  // x>=1. If the second attribute is selected, the condition should be "x \in
  // {1}" or (exclusive) "x \in {2}" (non deterministic).
  EXPECT_TRUE(condition.attribute() == 0 || condition.attribute() == 1);
  if (condition.attribute() == 0) {
    EXPECT_NEAR(1.0f, condition.condition().higher_condition().threshold(),
                0.01f);
  } else if (condition.attribute() == 1) {
    const auto& bitmap_condition =
        condition.condition().contains_bitmap_condition().elements_bitmap();
    EXPECT_EQ(bitmap_condition.size(), 1);
    EXPECT_TRUE((bitmap_condition[0] == 2) != (bitmap_condition[0] == 4));
  }
}

TEST(DecisionTree, FindBestCategoricalSplitCartIsNaForClassification) {
  // Small basic dataset.
  const std::vector<row_t> selected_examples = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights = {1, 1, 1, 1, 1, 1};
  const std::vector<int32_t> labels = {1, 1, 0, 0, 1, 0};
  const int32_t num_label_classes = 2;

  dataset::VerticalDataset::NumericalColumn attributes;
  attributes.Add(0.f);
  attributes.Add(1.f);
  attributes.Add(std::numeric_limits<float>::quiet_NaN());
  attributes.Add(std::numeric_limits<float>::quiet_NaN());
  attributes.Add(std::numeric_limits<float>::quiet_NaN());
  attributes.Add(std::numeric_limits<float>::quiet_NaN());

  // Configuration.
  const row_t min_num_obs = 1;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.set_allow_na_conditions(true);

  // Compute the label distribution.
  utils::IntegerDistributionDouble label_distribution;
  label_distribution.SetNumClasses(num_label_classes);
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }

  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  EXPECT_EQ(FindSplitLabelClassificationFeatureNA(
                selected_examples, weights, &attributes, labels,
                num_label_classes, min_num_obs, dt_config, label_distribution,
                -1, &best_condition, &cache),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().type_case(),
            proto::Condition::kNaCondition);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 6);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 4);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 4);
  // R> entropy(c(3,3)) - 4/6*entropy(c(3,1)) - 2/6*entropy(c(2)) = 0.3182571
  EXPECT_NEAR(best_condition.split_score(), 0.3182571, 0.0001);

  EXPECT_EQ(FindSplitLabelClassificationFeatureNA(
                selected_examples, weights, &attributes, labels,
                num_label_classes, min_num_obs, dt_config, label_distribution,
                -1, &best_condition, &cache),
            SplitSearchResult::kNoBetterSplitFound);
  LOG(INFO) << "Condition:\n" << best_condition.DebugString();
}

TEST(DecisionTree, FindBestCategoricalSplitCartIsNaForRegression) {
  // Small basic dataset.
  const std::vector<row_t> selected_examples = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights = {1, 1, 1, 1, 1, 1};
  const std::vector<float> labels = {1, 1, 0, 0, 1, 0};

  dataset::VerticalDataset::NumericalColumn attributes;
  attributes.Add(0.f);
  attributes.Add(1.f);
  attributes.Add(std::numeric_limits<float>::quiet_NaN());
  attributes.Add(std::numeric_limits<float>::quiet_NaN());
  attributes.Add(std::numeric_limits<float>::quiet_NaN());
  attributes.Add(std::numeric_limits<float>::quiet_NaN());

  // Configuration.
  const row_t min_num_obs = 1;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.set_allow_na_conditions(true);

  // Compute the label distribution.
  utils::NormalDistributionDouble label_distribution;
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }

  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  EXPECT_EQ(FindSplitLabelRegressionFeatureNA(
                selected_examples, weights, &attributes, labels, min_num_obs,
                dt_config, label_distribution, -1, &best_condition, &cache),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().type_case(),
            proto::Condition::kNaCondition);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 6);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 4);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 4);
  // > varb = function (x) { mean(x*x) - mean(x)^2 }
  // > varb(c(1,1,0,0,1,0)) - 4/6*varb(c(1,1,1,0)) - 2/6*varb(c(0,0))
  EXPECT_NEAR(best_condition.split_score(), 0.125, 0.01);

  EXPECT_EQ(FindSplitLabelRegressionFeatureNA(
                selected_examples, weights, &attributes, labels, min_num_obs,
                dt_config, label_distribution, -1, &best_condition, &cache),
            SplitSearchResult::kNoBetterSplitFound);
  LOG(INFO) << "Condition:\n" << best_condition.DebugString();
}

TEST(DecisionTree, FindBestNumericalSplitHistogramForRegression) {
  const std::vector<row_t> selected_examples = {0, 1, 2};
  const std::vector<float> weights = {1.f, 1.f, 1.f};
  const float na = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> attributes = {0, 1, na};
  const std::vector<float> labels = {0, 1, 0};
  const float na_replacement = 2;
  const row_t min_num_obs = 1;
  // Compute the label distribution.
  utils::NormalDistributionDouble label_distribution;
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_numerical_split()->set_type(
      proto::NumericalSplit::HISTOGRAM_RANDOM);
  dt_config.mutable_numerical_split()->set_num_candidates(1);
  proto::NodeCondition best_condition;
  utils::RandomEngine random(123456);
  const auto saved_state = random;

  std::uniform_real_distribution<float> threshold_distribution(0, 1);
  const float threshold = threshold_distribution(random);

  random = saved_state;
  EXPECT_EQ(FindSplitLabelRegressionFeatureNumericalHistogram(
                selected_examples, weights, attributes, labels, na_replacement,
                min_num_obs, dt_config, label_distribution, -1, &random,
                &best_condition),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().higher_condition().threshold(),
            threshold);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 3);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 3);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 2);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 2);
  EXPECT_EQ(best_condition.na_value(), true);

  random = saved_state;
  EXPECT_EQ(FindSplitLabelRegressionFeatureNumericalHistogram(
                selected_examples, weights, attributes, labels, na_replacement,
                min_num_obs, dt_config, label_distribution, -1, &random,
                &best_condition),
            SplitSearchResult::kNoBetterSplitFound);

  attributes = {1, 1, 1, 1, 1, 1};
  random = saved_state;
  EXPECT_EQ(FindSplitLabelRegressionFeatureNumericalHistogram(
                selected_examples, weights, attributes, labels, na_replacement,
                min_num_obs, dt_config, label_distribution, -1, &random,
                &best_condition),
            SplitSearchResult::kInvalidAttribute);
}

TEST(DecisionTree, FindBestNumericalSplitCartNumericalLabelBase) {
  const std::vector<row_t> selected_examples = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  const float na = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> attributes = {2, 3, 0, 1, na, na};
  const std::vector<float> labels = {1.f, 1.f, 0.f, 0.f, 1.f, 0.f};
  const float na_replacement = 2;
  const row_t min_num_obs = 1;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::IN_NODE);
  utils::NormalDistributionDouble label_distribution;
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }
  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  EXPECT_EQ(FindSplitLabelRegressionFeatureNumericalCart(
                selected_examples, weights, attributes, labels, na_replacement,
                min_num_obs, dt_config, label_distribution, -1, {},
                &best_condition, &cache),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().higher_condition().threshold(), 1.5f);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 6);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 4);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 4);
  // > varb = function (x) { mean(x*x) - mean(x)^2 }
  // > varb(c(1,1,0,0,1,0)) - 4/6*varb(c(1,1,1,0)) - 2/6*varb(c(0,0))
  EXPECT_NEAR(best_condition.split_score(), 0.125, 0.01);
  EXPECT_EQ(best_condition.na_value(), true);

  EXPECT_EQ(FindSplitLabelRegressionFeatureNumericalCart(
                selected_examples, weights, attributes, labels, na_replacement,
                min_num_obs, dt_config, label_distribution, -1, {},
                &best_condition, &cache),
            SplitSearchResult::kNoBetterSplitFound);

  attributes = {1, 1, 1, 1, 1, 1};
  EXPECT_EQ(FindSplitLabelRegressionFeatureNumericalCart(
                selected_examples, weights, attributes, labels, na_replacement,
                min_num_obs, dt_config, label_distribution, -1, {},
                &best_condition, &cache),
            SplitSearchResult::kInvalidAttribute);
}

class FindBestNumericalSplitCartNumericalLabelBasePresortedTest
    : public testing::TestWithParam<bool> {};

TEST_P(FindBestNumericalSplitCartNumericalLabelBasePresortedTest,
       WithDuplicates) {
  const bool duplicated_selected_examples = GetParam();

  // Similar examples as for the
  // DecisionTree.FindBestNumericalSplitCartNumericalLabelBase test.
  const std::vector<row_t> selected_examples = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  const float na = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> attributes = {2, 3, 0, 1, na, na};
  const std::vector<float> labels = {1.f, 1.f, 0.f, 0.f, 1.f, 0.f};
  const float na_replacement = 2;
  const row_t min_num_obs = 1;

  // Computes the preprocessing.
  Preprocessing preprocessing;
  {
    dataset::VerticalDataset dataset;
    dataset.set_data_spec(PARSE_TEST_PROTO(
        R"pb(
          columns {
            type: NUMERICAL
            name: "a"
            numerical { mean: 2 }
          }
        )pb"));
    CHECK_OK(dataset.CreateColumnsFromDataspec());
    for (const auto attribute : attributes) {
      dataset::proto::Example example;
      if (std::isnan(attribute)) {
        example.add_attributes();
      } else {
        example.add_attributes()->set_numerical(attribute);
      }
      dataset.AppendExample(example);
    }
    model::proto::TrainingConfigLinking config_link;
    config_link.add_features(0);
    CHECK_OK(PresortNumericalFeatures(dataset, config_link, 6, &preprocessing));
    preprocessing.set_num_examples(dataset.nrow());
  }

  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED);
  utils::NormalDistributionDouble label_distribution;
  for (const auto example_idx : selected_examples) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }
  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  InternalTrainConfig internal_config;
  internal_config.preprocessing = &preprocessing;
  internal_config.duplicated_selected_examples = duplicated_selected_examples;
  EXPECT_EQ(FindSplitLabelRegressionFeatureNumericalCart(
                selected_examples, weights, attributes, labels, na_replacement,
                min_num_obs, dt_config, label_distribution, 0, internal_config,
                &best_condition, &cache),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().higher_condition().threshold(), 1.5f);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 6);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 4);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 4);
  // > varb = function (x) { mean(x*x) - mean(x)^2 }
  // > varb(c(1,1,0,0,1,0)) - 4/6*varb(c(1,1,1,0)) - 2/6*varb(c(0,0))
  EXPECT_NEAR(best_condition.split_score(), 0.125, 0.01);
  EXPECT_EQ(best_condition.na_value(), true);

  EXPECT_EQ(FindSplitLabelRegressionFeatureNumericalCart(
                selected_examples, weights, attributes, labels, na_replacement,
                min_num_obs, dt_config, label_distribution, 0, internal_config,
                &best_condition, &cache),
            SplitSearchResult::kNoBetterSplitFound);
}
INSTANTIATE_TEST_SUITE_P(
    DuplicatedSelectedExamples,
    FindBestNumericalSplitCartNumericalLabelBasePresortedTest, testing::Bool());

TEST(FindBestNumericalSplitCartNumericalLabelBasePresortedTestManual, Base) {
  const std::vector<row_t> selected_examples = {0, 1, 3, 4, 9};
  const std::vector<float> weights(11, 1.f);
  std::vector<float> attributes = {0, 0, 1, 1, 1, 1, 2, 2, 5, 5, 5};
  const std::vector<float> labels = {0, 0, 1, 1, 1, 1, 1, 1, 1000, 1000, 1000};
  const float na_replacement = 2;
  const row_t min_num_obs = 1;

  // Computes the preprocessing.
  Preprocessing preprocessing;
  {
    dataset::VerticalDataset dataset;
    dataset.set_data_spec(PARSE_TEST_PROTO(
        R"pb(
          columns {
            type: NUMERICAL
            name: "a"
            numerical { mean: 2 }
          }
        )pb"));
    CHECK_OK(dataset.CreateColumnsFromDataspec());
    for (const auto attribute : attributes) {
      dataset::proto::Example example;
      example.add_attributes()->set_numerical(attribute);
      dataset.AppendExample(example);
    }
    model::proto::TrainingConfigLinking config_link;
    config_link.add_features(0);
    CHECK_OK(PresortNumericalFeatures(dataset, config_link, 6, &preprocessing));
    preprocessing.set_num_examples(dataset.nrow());
  }

  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED);
  utils::NormalDistributionDouble label_distribution;
  for (const auto example_idx : selected_examples) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }
  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  InternalTrainConfig internal_config;
  internal_config.preprocessing = &preprocessing;
  internal_config.duplicated_selected_examples = false;
  EXPECT_EQ(FindSplitLabelRegressionFeatureNumericalCart(
                selected_examples, weights, attributes, labels, na_replacement,
                min_num_obs, dt_config, label_distribution, 0, internal_config,
                &best_condition, &cache),
            SplitSearchResult::kBetterSplitFound);

  LOG(INFO) << "Condition: " << best_condition.condition().DebugString();

  EXPECT_EQ(best_condition.condition().higher_condition().threshold(), 3.0f);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 5);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 5);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 1);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 1);
}

TEST(DecisionTree, FindBestCategoricalSplitCartNumericalLabels) {
  // Small basic dataset.
  const std::vector<row_t> selected_examples = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights = {1, 1, 1, 1, 1, 1};
  std::vector<int32_t> attributes = {2, 3, 0, 1, -1, -1};
  const std::vector<float> labels = {1.f, 1.f, 0.f, 0.f, 1.f, 0.f};
  const int32_t num_attribute_classes = 4;
  utils::RandomEngine rnd;

  // Configuration.
  const int32_t na_replacement = 1;
  const row_t min_num_obs = 1;
  const proto::DecisionTreeTrainingConfig dt_config;

  // Compute the label distribution.
  utils::NormalDistributionDouble label_distribution;
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }

  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  EXPECT_EQ(FindSplitLabelRegressionFeatureCategorical(
                selected_examples, weights, attributes, labels,
                num_attribute_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, &best_condition, &cache, &rnd),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().type_case(),
            proto::Condition::kContainsBitmapConditionFieldNumber);
  // The expected element map is "1100".
  EXPECT_EQ(
      best_condition.condition().contains_bitmap_condition().elements_bitmap(),
      "\014");
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 6);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 2);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 2);
  EXPECT_EQ(best_condition.na_value(), false);
  // > varb = function (x) { mean(x*x) - mean(x)^2 }
  // > varb(c(1,1,0,0,1,0)) - 4/6*varb(c(0,0,1,0)) - 2/6*varb(c(1,1))
  EXPECT_NEAR(best_condition.split_score(), 0.125, 0.0001);

  EXPECT_EQ(FindSplitLabelRegressionFeatureCategorical(
                selected_examples, weights, attributes, labels,
                num_attribute_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, &best_condition, &cache, &rnd),
            SplitSearchResult::kNoBetterSplitFound);
  LOG(INFO) << "Condition:\n" << best_condition.DebugString();

  // Since all the attributes have the same value, there are no valid splits.
  attributes = {1, 1, 1, 1, 1, 1};

  EXPECT_EQ(FindSplitLabelRegressionFeatureCategorical(
                selected_examples, weights, attributes, labels,
                num_attribute_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, &best_condition, &cache, &rnd),
            SplitSearchResult::kInvalidAttribute);
  LOG(INFO) << "Condition:\n" << best_condition.DebugString();
}

TEST(DecisionTree, FindBestCategoricalSplitCartBooleanForClassification) {
  // Small basic dataset.
  const std::vector<row_t> selected_examples = {0, 1, 2, 3, 4, 5};
  std::vector<char> attributes = {0, 1, 0, 1, 0, 0};
  const std::vector<float> weights = {1, 1, 1, 1, 1, 1};
  const std::vector<int32_t> labels = {1, 0, 0, 0, 0, 1};
  const int32_t num_label_classes = 2;

  // Configuration.
  const row_t min_num_obs = 1;
  proto::DecisionTreeTrainingConfig dt_config;

  // Compute the label distribution.
  utils::IntegerDistributionDouble label_distribution;
  label_distribution.SetNumClasses(num_label_classes);
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }

  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  EXPECT_EQ(FindSplitLabelClassificationFeatureBoolean(
                selected_examples, weights, attributes, labels,
                num_label_classes, false, min_num_obs, dt_config,
                label_distribution, -1, &best_condition, &cache),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().type_case(),
            proto::Condition::kTrueValueCondition);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 6);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 2);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 2);
  // R> entropy(c(2,4)) - 4/6*entropy(c(2,2)) - 2/6*entropy(c(2)) = 0.1744
  EXPECT_NEAR(best_condition.split_score(), 0.1744, 0.0001);

  EXPECT_EQ(FindSplitLabelClassificationFeatureBoolean(
                selected_examples, weights, attributes, labels,
                num_label_classes, false, min_num_obs, dt_config,
                label_distribution, -1, &best_condition, &cache),
            SplitSearchResult::kNoBetterSplitFound);
  LOG(INFO) << "Condition:\n" << best_condition.DebugString();
}

TEST(DecisionTree, FindBestCategoricalSplitCartBooleanForRegression) {
  // Small basic dataset.
  const std::vector<row_t> selected_examples = {0, 1, 2, 3, 4, 5};
  std::vector<char> attributes = {0, 1, 0, 1, 0, 0};
  const std::vector<float> weights = {1, 1, 1, 1, 1, 1};
  const std::vector<float> labels = {1, 0, 0, 0, 0, 1};

  // Configuration.
  const row_t min_num_obs = 1;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::IN_NODE);

  // Compute the label distribution.
  utils::NormalDistributionDouble label_distribution;
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }

  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  EXPECT_EQ(
      FindSplitLabelRegressionFeatureBoolean(
          selected_examples, weights, attributes, labels, false, min_num_obs,
          dt_config, label_distribution, -1, &best_condition, &cache),
      SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().type_case(),
            proto::Condition::kTrueValueCondition);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 6);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 2);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 2);
  // > varb = function (x) { mean(x*x) - mean(x)^2 }
  // > varb(c(1,0,0,0,0,1)) - 4/6*varb(c(1,0,0,1)) - 2/6*varb(c(0,0))
  EXPECT_NEAR(best_condition.split_score(), 0.055, 0.01);

  EXPECT_EQ(
      FindSplitLabelRegressionFeatureBoolean(
          selected_examples, weights, attributes, labels, false, min_num_obs,
          dt_config, label_distribution, -1, &best_condition, &cache),
      SplitSearchResult::kNoBetterSplitFound);
  LOG(INFO) << "Condition:\n" << best_condition.DebugString();
}

TEST(DecisionTree, LocalImputationForNumericalAttribute) {
  const std::vector<row_t> selected_examples = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  const float na = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> attributes = {2, 3, 0, 1, na, na};
  const std::vector<int32_t> labels = {1, 1, 0, 0, 1, 0};
  const int32_t num_label_classes = 2;

  // This "na_replacement" value will be ignored and replaced with
  // "mean(attributes) = 1.5".
  const float na_replacement = -1;
  const row_t min_num_obs = 1;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::IN_NODE);
  dt_config.set_missing_value_policy(
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION);

  utils::IntegerDistributionDouble label_distribution;
  label_distribution.SetNumClasses(num_label_classes);
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }

  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  EXPECT_EQ(FindSplitLabelClassificationFeatureNumericalCart(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, {}, &best_condition, &cache),
            SplitSearchResult::kBetterSplitFound);

  // > mean(c(1, 1.5)) = 1.25
  EXPECT_EQ(best_condition.condition().higher_condition().threshold(), 1.25f);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 6);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 4);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 4);
  EXPECT_EQ(best_condition.na_value(), true);

  EXPECT_EQ(FindSplitLabelClassificationFeatureNumericalCart(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, {}, &best_condition, &cache),
            SplitSearchResult::kNoBetterSplitFound);

  attributes = {1, 1, 1, 1, 1, 1};
  EXPECT_EQ(FindSplitLabelClassificationFeatureNumericalCart(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, {}, &best_condition, &cache),
            SplitSearchResult::kInvalidAttribute);
}

TEST(DecisionTree, LocalImputationForCategoricalAttribute) {
  // Small basic dataset.
  const std::vector<row_t> selected_examples = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights = {1, 1, 1, 1, 1, 1};
  std::vector<int32_t> attributes = {2, 2, 0, 1, -1, -1};
  const std::vector<int32_t> labels = {1, 1, 0, 0, 1, 0};
  const int32_t num_label_classes = 2;
  const int32_t num_attribute_classes = 8;

  // This "na_replacement" value will be ignored and replaced with
  // the most frequent item = 2.
  const int32_t na_replacement = 0;

  // Configuration.
  const row_t min_num_obs = 1;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.set_missing_value_policy(
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION);

  // Compute the label distribution.
  utils::IntegerDistributionDouble label_distribution;
  label_distribution.SetNumClasses(num_label_classes);
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }

  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  utils::RandomEngine rnd;
  EXPECT_EQ(
      FindSplitLabelClassificationFeatureCategorical(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, -1, &rnd, &best_condition, &cache),
      SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().type_case(),
            proto::Condition::kContainsBitmapConditionFieldNumber);
  // The expected element map is "0011".
  EXPECT_EQ(
      best_condition.condition().contains_bitmap_condition().elements_bitmap(),
      "\003");
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 6);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 2);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 2);
  EXPECT_EQ(best_condition.na_value(), false);
  // R> entropy(c(3,3)) - 4/6*entropy(c(3,1)) - 2/6*entropy(c(2)) = 0.3182571
  EXPECT_NEAR(best_condition.split_score(), 0.3182571, 0.0001);

  EXPECT_EQ(
      FindSplitLabelClassificationFeatureCategorical(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, -1, &rnd, &best_condition, &cache),
      SplitSearchResult::kNoBetterSplitFound);
  LOG(INFO) << "Condition:\n" << best_condition.DebugString();

  // Since all the attributes have the same value, there are no valid splits.
  attributes = {1, 1, 1, 1, 1, 1};
  EXPECT_EQ(
      FindSplitLabelClassificationFeatureCategorical(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, -1, &rnd, &best_condition, &cache),
      SplitSearchResult::kInvalidAttribute);
  LOG(INFO) << "Condition:\n" << best_condition.DebugString();

  // All the attribute value are NA.
  attributes = {-1, -1, -1, -1, -1, -1};
  EXPECT_EQ(
      FindSplitLabelClassificationFeatureCategorical(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, -1, &rnd, &best_condition, &cache),
      SplitSearchResult::kInvalidAttribute);
  LOG(INFO) << "Condition:\n" << best_condition.DebugString();
}

TEST(DecisionTree, LocalImputationForBooleanAttribute) {
  const std::vector<row_t> selected_examples = {0, 1, 2, 3, 4, 5};
  const std::vector<float> weights = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  const char na = dataset::VerticalDataset::BooleanColumn::kNaValue;
  std::vector<char> attributes = {0, 1, 0, 0, na, na};
  const std::vector<int32_t> labels = {1, 1, 0, 0, 1, 0};
  const int32_t num_label_classes = 2;

  // This "na_replacement" value will be ignored and replaced with
  // the most frequent attribute (0).
  const bool na_replacement = true;
  const row_t min_num_obs = 1;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.set_missing_value_policy(
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION);

  utils::IntegerDistributionDouble label_distribution;
  label_distribution.SetNumClasses(num_label_classes);
  for (int example_idx = 0; example_idx < selected_examples.size();
       example_idx++) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }

  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  EXPECT_EQ(FindSplitLabelClassificationFeatureBoolean(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, &best_condition, &cache),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().type_case(),
            proto::Condition::kTrueValueCondition);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 6);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 1);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 1);
  EXPECT_EQ(best_condition.na_value(), false);

  EXPECT_EQ(FindSplitLabelClassificationFeatureBoolean(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, &best_condition, &cache),
            SplitSearchResult::kNoBetterSplitFound);

  attributes = {1, 1, 1, 1, 1, 1};
  EXPECT_EQ(FindSplitLabelClassificationFeatureBoolean(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, &best_condition, &cache),
            SplitSearchResult::kInvalidAttribute);

  // Test majority positive case.
  attributes = {1, 1, 1, 0, na, na};
  proto::NodeCondition best_condition_pos_na;
  SplitterPerThreadCache cache_pos_na;
  EXPECT_EQ(FindSplitLabelClassificationFeatureBoolean(
                selected_examples, weights, attributes, labels,
                num_label_classes, na_replacement, min_num_obs, dt_config,
                label_distribution, -1, &best_condition_pos_na, &cache_pos_na),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition_pos_na.condition().type_case(),
            proto::Condition::kTrueValueCondition);
  EXPECT_EQ(best_condition_pos_na.num_training_examples_without_weight(), 6);
  EXPECT_EQ(best_condition_pos_na.num_training_examples_with_weight(), 6);
  EXPECT_EQ(best_condition_pos_na.num_pos_training_examples_without_weight(),
            5);
  EXPECT_EQ(best_condition_pos_na.num_pos_training_examples_with_weight(), 5);
  EXPECT_EQ(best_condition_pos_na.na_value(), true);
}

TEST(DecisionTree, GenerateRandomImputation) {
  // The test defines 4 example: 0 and 2 are valid non-na examples, 1 is an
  // example filled with na-values, and 3 is a forbidden example.

  dataset::VerticalDataset dataset;
  dataset.set_data_spec(PARSE_TEST_PROTO(
      R"pb(
        columns { type: NUMERICAL name: "a" }
        columns { type: NUMERICAL_SET name: "b" }
        columns { type: CATEGORICAL name: "d" }
        columns { type: STRING name: "h" }
        columns { type: NUMERICAL name: "non_copied" }
      )pb"));
  CHECK_OK(dataset.CreateColumnsFromDataspec());

  const dataset::proto::Example example_1 = PARSE_TEST_PROTO(
      R"pb(
        attributes { numerical: 1 }
        attributes { numerical_set: { values: 1 } }
        attributes { categorical: 1 }
        attributes { text: "riri" }
        attributes { numerical: 1 }
      )pb");
  dataset.AppendExample(example_1);

  const dataset::proto::Example example_2 = PARSE_TEST_PROTO(
      R"pb(
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
      )pb");
  dataset.AppendExample(example_2);

  const dataset::proto::Example example_3 = PARSE_TEST_PROTO(
      R"pb(
        attributes { numerical: 2 }
        attributes { numerical_set: { values: 2 values: 3 } }
        attributes { categorical: 2 }
        attributes { text: "fifi" }
        attributes { numerical: 1 }
      )pb");
  dataset.AppendExample(example_3);

  const dataset::proto::Example example_4 = PARSE_TEST_PROTO(
      R"pb(
        attributes { numerical: 3 }
        attributes { numerical_set: { values: 4 values: 5 } }
        attributes { categorical: 3 }
        attributes { text: "loulou" }
        attributes { numerical: 1 }
      )pb");
  dataset.AppendExample(example_4);

  utils::RandomEngine rnd(1324);
  dataset::VerticalDataset imputed;
  GenerateRandomImputation(dataset, {0, 1, 2, 3}, {0, 1, 2}, &imputed, &rnd);

  EXPECT_EQ(dataset.nrow(), 4);
  EXPECT_EQ(imputed.nrow(), 3);

  EXPECT_EQ(imputed.column(0)->nrows(), 3);
  EXPECT_EQ(imputed.column(1)->nrows(), 3);
  EXPECT_EQ(imputed.column(2)->nrows(), 3);
  EXPECT_EQ(imputed.column(3)->nrows(), 3);
  EXPECT_EQ(imputed.column(4)->nrows(), 0);

  EXPECT_TRUE(
      (imputed
           .MutableColumnWithCast<dataset::VerticalDataset::NumericalColumn>(0)
           ->values() == std::vector<float>{1, 1, 2}) ||
      (imputed
           .MutableColumnWithCast<dataset::VerticalDataset::NumericalColumn>(0)
           ->values() == std::vector<float>{1, 2, 2}));

  EXPECT_TRUE(
      (imputed
           .MutableColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
               2)
           ->values() == std::vector<int>{1, 1, 2}) ||
      (imputed
           .MutableColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
               2)
           ->values() == std::vector<int>{1, 2, 2}));
}

TEST(DecisionTree,
     FindSplitLabelClassificationFeatureCategoricalSetGreedyForward) {
  std::vector<row_t> selected = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> weights = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> labels = {0, 0, 0, 0, 1, 1, 1, 1};

  // A good attribute that perfectly separate the labels.
  dataset::VerticalDataset::CategoricalSetColumn attributes_perfect;
  // Will end up in the positive set.
  attributes_perfect.AddVector({0});
  attributes_perfect.AddVector({1, 3});
  attributes_perfect.AddVector({2, 4});
  attributes_perfect.AddVector({0, 1, 4});
  // Will end up in the negative set.
  attributes_perfect.AddVector({3});
  attributes_perfect.AddVector({4});
  attributes_perfect.AddVector({3, 5});
  attributes_perfect.AddVector({4, 5});

  // A bad attribute that does not separate the labels at all. All the item
  // values are present twice for each label value.
  dataset::VerticalDataset::CategoricalSetColumn attributes_bad;
  attributes_bad.AddVector({0, 1});
  attributes_bad.AddVector({0, 1, 3});
  attributes_bad.AddVector({2, 3, 4});
  attributes_bad.AddVector({2, 4});
  attributes_bad.AddVector({0, 2, 3});
  attributes_bad.AddVector({0, 2, 4});
  attributes_bad.AddVector({1, 4});
  attributes_bad.AddVector({1, 3});

  // A pure/non-valid attribute.
  dataset::VerticalDataset::CategoricalSetColumn attributes_non_valid;
  attributes_non_valid.AddVector({1, 2});
  attributes_non_valid.AddVector({1, 2});
  attributes_non_valid.AddVector({1, 2});
  attributes_non_valid.AddVector({1, 2});
  attributes_non_valid.AddVector({1, 2});
  attributes_non_valid.AddVector({1, 2});
  attributes_non_valid.AddVector({1, 2});
  attributes_non_valid.AddVector({1, 2});

  int num_attribute_classes = 6;
  int num_label_classes = 3;
  int min_num_obs = 1;

  // Compute the label distribution.
  utils::IntegerDistributionDouble label_distribution;
  label_distribution.SetNumClasses(num_label_classes);
  for (const auto example_idx : selected) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }

  utils::RandomEngine rnd(1234);
  proto::NodeCondition best_condition;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_categorical_set_greedy_forward()->set_sampling(1.f);

  EXPECT_EQ(FindSplitLabelClassificationFeatureCategoricalSetGreedyForward(
                selected, weights, attributes_bad, labels,
                num_attribute_classes, num_label_classes, min_num_obs,
                dt_config, label_distribution, -1, &best_condition, &rnd),
            SplitSearchResult::kNoBetterSplitFound);

  EXPECT_EQ(FindSplitLabelClassificationFeatureCategoricalSetGreedyForward(
                selected, weights, attributes_non_valid, labels,
                num_attribute_classes, num_label_classes, min_num_obs,
                dt_config, label_distribution, -1, &best_condition, &rnd),
            SplitSearchResult::kInvalidAttribute);

  EXPECT_EQ(FindSplitLabelClassificationFeatureCategoricalSetGreedyForward(
                selected, weights, attributes_perfect, labels,
                num_attribute_classes, num_label_classes, min_num_obs,
                dt_config, label_distribution, -1, &best_condition, &rnd),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().type_case(),
            proto::Condition::kContainsBitmapConditionFieldNumber);
  // The expected element map is "000111".
  EXPECT_EQ(
      best_condition.condition().contains_bitmap_condition().elements_bitmap(),
      "\07");
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 8);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 8);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 4);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 4);
  EXPECT_EQ(best_condition.na_value(), false);
  // R> entropy(c(4,4)) = 0.6931472
  EXPECT_NEAR(best_condition.split_score(), 0.6931472, 0.0001);
}

TEST(DecisionTree, FindBestCategoricalSetSplitCartWithNA) {
  std::vector<row_t> selected = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> weights = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> labels = {0, 0, 0, 0, 1, 1, 1, 1};

  // A attribute that does not perfectly separate the labels.
  dataset::VerticalDataset::CategoricalSetColumn attributes;
  // Label = 0 + put in negative split.
  attributes.AddVector({0});
  attributes.AddVector({1});
  attributes.AddVector({0, 1});
  attributes.AddNA();
  // Label = 1 + put in negative split.
  attributes.AddVector({0});
  attributes.AddNA();
  // Label = 1 + put in positive split.
  attributes.AddVector({2});
  attributes.AddVector({0, 2});

  int num_attribute_classes = 4;
  int num_label_classes = 3;
  int min_num_obs = 1;

  // Compute the label distribution.
  utils::IntegerDistributionDouble label_distribution;
  label_distribution.SetNumClasses(num_label_classes);
  for (const auto example_idx : selected) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }

  utils::RandomEngine rnd(1234);
  proto::NodeCondition best_condition;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_categorical_set_greedy_forward()->set_sampling(1.f);

  EXPECT_EQ(FindSplitLabelClassificationFeatureCategoricalSetGreedyForward(
                selected, weights, attributes, labels, num_attribute_classes,
                num_label_classes, min_num_obs, dt_config, label_distribution,
                -1, &best_condition, &rnd),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().type_case(),
            proto::Condition::kContainsBitmapConditionFieldNumber);
  // The expected element map is "0010" + padding.
  EXPECT_EQ(
      best_condition.condition().contains_bitmap_condition().elements_bitmap(),
      "\02");
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 8);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 8);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 2);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 2);
  EXPECT_EQ(best_condition.na_value(), false);
  // R> entropy(c(4,4)) - 6/8 * entropy(c(4,2)) = 0.2157616
  EXPECT_NEAR(best_condition.split_score(), 0.2157616, 0.0001);
}

TEST(DecisionTree, FindBestCategoricalSetSplitCartForRegression) {
  // List of selected examples.
  std::vector<row_t> selected = {0, 1, 2, 3, 4, 5, 6, 7};
  // Uniform weights.
  std::vector<float> weights = {1, 1, 1, 1, 1, 1, 1, 1};
  // Example labels. Ultimately, we want the first 4 examples to be in one
  // child, and the 4 other in the other.
  std::vector<float> labels_v1 = {1, 2, 3, 4, 11, 12, 13, 14};
  std::vector<float> labels_v2 = {1, 3, 4, 2, 13, 12, 11, 14};

  // A good attribute that perfectly separate the labels.
  // We would want the split to be "attribute \intersect {0,1,2} is empty".
  //
  // In the case of label_v1 because of the greedy selection, item 5 will be
  // selected first. Therefore the condition will be "attribute \intersect {5}
  // is empty".
  //
  // In the case of label_v2, the item 0 will be selected first, leading to the
  // best possible outcome.
  dataset::VerticalDataset::CategoricalSetColumn attributes_perfect;

  // Will end up in the positive set.
  attributes_perfect.AddVector({0});
  attributes_perfect.AddVector({1, 3});
  attributes_perfect.AddVector({2, 4});
  attributes_perfect.AddVector({0, 1, 4});
  // Will end up in the negative set.
  attributes_perfect.AddVector({3});
  attributes_perfect.AddVector({4});
  attributes_perfect.AddVector({3, 5});
  attributes_perfect.AddVector({4, 5});

  // A pure/non-valid attribute.
  dataset::VerticalDataset::CategoricalSetColumn attributes_non_valid;
  attributes_non_valid.AddVector({1, 2});
  attributes_non_valid.AddVector({1, 2});
  attributes_non_valid.AddVector({1, 2});
  attributes_non_valid.AddVector({1, 2});
  attributes_non_valid.AddVector({1, 2});
  attributes_non_valid.AddVector({1, 2});
  attributes_non_valid.AddVector({1, 2});
  attributes_non_valid.AddVector({1, 2});

  int num_attribute_classes = 6;
  int min_num_obs = 1;

  // Compute the label distribution.
  utils::NormalDistributionDouble label_distribution_v1;
  for (const auto example_idx : selected) {
    label_distribution_v1.Add(labels_v1[example_idx], weights[example_idx]);
  }

  utils::NormalDistributionDouble label_distribution_v2;
  for (const auto example_idx : selected) {
    label_distribution_v2.Add(labels_v2[example_idx], weights[example_idx]);
  }

  utils::RandomEngine rnd(1234);
  proto::NodeCondition best_condition;
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_categorical_set_greedy_forward()->set_sampling(1.f);

  EXPECT_EQ(FindSplitLabelRegressionFeatureCategoricalSetGreedyForward(
                selected, weights, attributes_non_valid, labels_v1,
                num_attribute_classes, min_num_obs, dt_config,
                label_distribution_v1, -1, &best_condition, &rnd),
            SplitSearchResult::kInvalidAttribute);

  EXPECT_EQ(FindSplitLabelRegressionFeatureCategoricalSetGreedyForward(
                selected, weights, attributes_perfect, labels_v1,
                num_attribute_classes, min_num_obs, dt_config,
                label_distribution_v1, -1, &best_condition, &rnd),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().type_case(),
            proto::Condition::kContainsBitmapConditionFieldNumber);
  // The expected element map is "100000" (32).
  EXPECT_EQ(
      best_condition.condition().contains_bitmap_condition().elements_bitmap(),
      " ");
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 8);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 8);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 2);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 2);
  EXPECT_EQ(best_condition.na_value(), false);
  // R>  var(c(1,2,3,4,11,12,13,14)) - (var(c(1,2,3,4,11,12)) * (6/8) +
  // var(c(13,14)) * (2/8)) = 12
  // With "var" the variance.
  EXPECT_NEAR(best_condition.split_score(), 12., 0.0001);

  proto::NodeCondition best_condition_v2;
  EXPECT_EQ(FindSplitLabelRegressionFeatureCategoricalSetGreedyForward(
                selected, weights, attributes_perfect, labels_v2,
                num_attribute_classes, min_num_obs, dt_config,
                label_distribution_v2, -1, &best_condition_v2, &rnd),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition_v2.condition().type_case(),
            proto::Condition::kContainsBitmapConditionFieldNumber);
  // The expected element map is "000111".
  EXPECT_EQ(best_condition_v2.condition()
                .contains_bitmap_condition()
                .elements_bitmap(),
            "\x07");
  EXPECT_EQ(best_condition_v2.num_training_examples_without_weight(), 8);
  EXPECT_EQ(best_condition_v2.num_training_examples_with_weight(), 8);
  EXPECT_EQ(best_condition_v2.num_pos_training_examples_without_weight(), 4);
  EXPECT_EQ(best_condition_v2.num_pos_training_examples_with_weight(), 4);
  EXPECT_EQ(best_condition_v2.na_value(), false);
  // R>   var(c(1,2,3,4,11,12,13,14)) - (var(c(1,2,3,4)) * (4/8) +
  // var(c(11,12,13,14)) * (4/8)) = 25
  // With "var" the variance (not the sampling variance).
  EXPECT_NEAR(best_condition_v2.split_score(), 25., 0.0001);
}

TEST(DecisionTree, MaskItemsForCategoricalForSetGreedySelection) {
  utils::RandomEngine random;

  proto::DecisionTreeTrainingConfig dt_config;
  int32_t num_attribute_classes = 5;
  std::vector<dataset::VerticalDataset::row_t> selected_examples = {0, 1, 2, 3,
                                                                    4};
  std::vector<int64_t> count_examples_without_weights_by_attribute_class = {
      0,  // Pure
      5,  // Pure
      1,  // Candidate
      2,  // Candidate
      3   // Candidate
  };

  dt_config.mutable_categorical_set_greedy_forward()->set_sampling(1.f);
  {
    std::vector<bool> candidate_attributes_bitmap(num_attribute_classes, true);
    internal::MaskPureSampledOrPrunedItemsForCategoricalSetGreedySelection(
        dt_config, num_attribute_classes, selected_examples,
        count_examples_without_weights_by_attribute_class,
        &candidate_attributes_bitmap, &random);
    // All the candidate items are selected.
    EXPECT_EQ(candidate_attributes_bitmap,
              std::vector<bool>({false, false, true, true, true}));
  }

  dt_config.mutable_categorical_set_greedy_forward()->set_sampling(0.f);
  {
    std::vector<bool> candidate_attributes_bitmap(num_attribute_classes, true);
    internal::MaskPureSampledOrPrunedItemsForCategoricalSetGreedySelection(
        dt_config, num_attribute_classes, selected_examples,
        count_examples_without_weights_by_attribute_class,
        &candidate_attributes_bitmap, &random);
    // None of the items are selected.
    EXPECT_EQ(candidate_attributes_bitmap,
              std::vector<bool>({false, false, false, false, false}));
  }

  dt_config.mutable_categorical_set_greedy_forward()->set_sampling(1.f);
  dt_config.mutable_categorical_set_greedy_forward()->set_max_num_items(
      4);  // The first 4 items.
  {
    std::vector<bool> candidate_attributes_bitmap(num_attribute_classes, true);
    internal::MaskPureSampledOrPrunedItemsForCategoricalSetGreedySelection(
        dt_config, num_attribute_classes, selected_examples,
        count_examples_without_weights_by_attribute_class,
        &candidate_attributes_bitmap, &random);
    // The last candidate item is not selected.
    EXPECT_EQ(candidate_attributes_bitmap,
              std::vector<bool>({false, false, true, true, false}));
  }

  dt_config.mutable_categorical_set_greedy_forward()->set_max_num_items(-1);
  dt_config.mutable_categorical_set_greedy_forward()->set_min_item_frequency(
      2);  // Item present in at least 2 examples and not-present in at least 2
  // examples.
  {
    std::vector<bool> candidate_attributes_bitmap(num_attribute_classes, true);
    internal::MaskPureSampledOrPrunedItemsForCategoricalSetGreedySelection(
        dt_config, num_attribute_classes, selected_examples,
        count_examples_without_weights_by_attribute_class,
        &candidate_attributes_bitmap, &random);
    EXPECT_EQ(candidate_attributes_bitmap,
              std::vector<bool>({false, false, false, true, true}));
  }
}

TEST(DecisionTree, GenHistogramBins) {
  const std::vector<float> attributes = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 10};
  const float min_value = 0;
  const float max_value = 10;
  const int num_bins = 1000;

  utils::RandomEngine random;
  const auto bins_random = internal::GenHistogramBins(
      proto::NumericalSplit::HISTOGRAM_RANDOM, num_bins, attributes, min_value,
      max_value, &random);
  const auto bins_equal_width = internal::GenHistogramBins(
      proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH, num_bins, attributes,
      min_value, max_value, &random);

  for (const auto& bins : {
           bins_random,
           bins_equal_width
       }) {
    EXPECT_EQ(bins.size(), num_bins);
    EXPECT_GE(*std::min_element(bins.begin(), bins.end()), min_value);
    EXPECT_LE(*std::max_element(bins.begin(), bins.end()), max_value);
  }

  // Near the middle.
  EXPECT_NEAR(bins_random[num_bins / 2], 5.f, 0.5f);
  // Near the middle.
  EXPECT_NEAR(bins_equal_width[num_bins / 2], 5.f, 0.5f);
}

TEST(DecisionTree, FindBestConditionConcurrentManager_NoFeatures) {
  dataset::VerticalDataset dataset;
  utils::RandomEngine random(1234);
  std::vector<dataset::VerticalDataset::row_t> selected_examples;
  std::vector<float> weights;
  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  proto::DecisionTreeTrainingConfig dt_config;
  proto::Node parent;
  InternalTrainConfig internal_config;
  internal_config.num_threads = 2;
  FakeLabelStats label_stats;
  PerThreadCache cache;

  proto::NodeCondition best_condition;

  SplitterConcurrencySetup setup{
      .num_threads = internal_config.num_threads,
      .split_finder_processor =
          absl::make_unique<SplitterFinderStreamProcessor>(
              "SplitFinder", internal_config.num_threads,
              FakeFindBestConditionConcurrentConsumerMultiplicative)};
  setup.split_finder_processor->StartWorkers();

  // Test case: Features are valid and scores are based on feature id, but
  // current best node has a high score.
  random.seed(1234);
  best_condition.set_split_score(1000.f);

  bool result = FindBestConditionConcurrentManager(
                    dataset, selected_examples, weights, config, config_link,
                    dt_config, setup, parent, internal_config, label_stats,
                    &best_condition, &random, &cache)
                    .value();

  EXPECT_FALSE(result);
}

TEST(DecisionTree, FindBestConditionConcurrentManager_AlwaysInvalid) {
  dataset::VerticalDataset dataset;
  utils::RandomEngine random(1234);
  std::vector<dataset::VerticalDataset::row_t> selected_examples;
  std::vector<float> weights;
  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  for (int i = 0; i < 20; i++) {
    config_link.add_features(i);
  }
  proto::DecisionTreeTrainingConfig dt_config;
  proto::Node parent;
  InternalTrainConfig internal_config;
  internal_config.num_threads = 2;
  FakeLabelStats label_stats;
  PerThreadCache cache;

  proto::NodeCondition best_condition;

  SplitterConcurrencySetup setup{
      .num_threads = internal_config.num_threads,
      .split_finder_processor =
          absl::make_unique<SplitterFinderStreamProcessor>(
              "SplitFinder", internal_config.num_threads,
              FakeFindBestConditionConcurrentConsumerAlwaysInvalid)};
  setup.split_finder_processor->StartWorkers();

  // Test case: All features are invalid.
  best_condition.set_split_score(0.f);
  bool result = FindBestConditionConcurrentManager(
                    dataset, selected_examples, weights, config, config_link,
                    dt_config, setup, parent, internal_config, label_stats,
                    &best_condition, &random, &cache)
                    .value();

  EXPECT_EQ(cache.splitter_cache_list.size(), 2);
  EXPECT_EQ(cache.work_status_list.size(), 20);
  EXPECT_EQ(cache.condition_list.size(), 4);
  EXPECT_FALSE(result);
}

TEST(DecisionTree, FindBestConditionConcurrentManager_Multiplicative) {
  dataset::VerticalDataset dataset;
  utils::RandomEngine random(1234);
  std::vector<dataset::VerticalDataset::row_t> selected_examples;
  std::vector<float> weights;
  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  for (int i = 0; i < 20; i++) {
    config_link.add_features(i);
  }
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.set_num_candidate_attributes(-1);
  proto::Node parent;
  InternalTrainConfig internal_config;
  internal_config.num_threads = 2;
  FakeLabelStats label_stats;
  PerThreadCache cache;

  proto::NodeCondition best_condition;

  SplitterConcurrencySetup setup{
      .num_threads = internal_config.num_threads,
      .split_finder_processor =
          absl::make_unique<SplitterFinderStreamProcessor>(
              "SplitFinder", internal_config.num_threads,
              FakeFindBestConditionConcurrentConsumerMultiplicative)};
  setup.split_finder_processor->StartWorkers();

  // Test case: Features are valid and scores are based on feature id, but
  // current best node has a high score.
  random.seed(1234);
  best_condition.set_split_score(1000.f);
  bool result = FindBestConditionConcurrentManager(
                    dataset, selected_examples, weights, config, config_link,
                    dt_config, setup, parent, internal_config, label_stats,
                    &best_condition, &random, &cache)
                    .value();

  EXPECT_FALSE(result);

  // Test case: Features are valid and scores are based on feature id.
  random.seed(1234);
  best_condition.set_split_score(0.f);
  result = FindBestConditionConcurrentManager(
               dataset, selected_examples, weights, config, config_link,
               dt_config, setup, parent, internal_config, label_stats,
               &best_condition, &random, &cache)
               .value();

  EXPECT_TRUE(result);

  EXPECT_NEAR(best_condition.split_score(), 190.f, 0.001);
}

TEST(DecisionTree, FindBestConditionConcurrentManager_Alternate) {
  dataset::VerticalDataset dataset;
  utils::RandomEngine random(1234);
  std::vector<dataset::VerticalDataset::row_t> selected_examples;
  std::vector<float> weights;
  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  for (int i = 0; i < 20; i++) {
    config_link.add_features(i);
  }
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.set_num_candidate_attributes(-1);
  proto::Node parent;
  InternalTrainConfig internal_config;
  internal_config.num_threads = 2;
  FakeLabelStats label_stats;
  PerThreadCache cache;

  proto::NodeCondition best_condition;

  SplitterConcurrencySetup setup{
      .num_threads = internal_config.num_threads,
      .split_finder_processor =
          absl::make_unique<SplitterFinderStreamProcessor>(
              "SplitFinder", internal_config.num_threads,
              FakeFindBestConditionConcurrentConsumerAlternate)};
  setup.split_finder_processor->StartWorkers();

  // Test case: Features alternate between valid and invalid.
  random.seed(1234);
  best_condition.set_split_score(0.f);
  bool result = FindBestConditionConcurrentManager(
                    dataset, selected_examples, weights, config, config_link,
                    dt_config, setup, parent, internal_config, label_stats,
                    &best_condition, &random, &cache)
                    .value();

  EXPECT_TRUE(result);
  EXPECT_NEAR(best_condition.split_score(), 190.f, 0.001);
}

TEST(DecisionTree, FindBestConditionConcurrentManagerScaled) {
  dataset::VerticalDataset dataset;
  utils::RandomEngine random(4321);
  std::vector<dataset::VerticalDataset::row_t> selected_examples;
  std::vector<float> weights;
  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  for (int i = 0; i < 100; i++) {
    config_link.add_features(i);
  }
  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.set_num_candidate_attributes(-1);
  proto::Node parent;
  InternalTrainConfig internal_config;
  internal_config.num_threads = 10;
  FakeLabelStats label_stats{};
  PerThreadCache cache;

  proto::NodeCondition best_condition;

  SplitterConcurrencySetup setup{
      .concurrent_execution = true,
      .num_threads = internal_config.num_threads,
      .split_finder_processor =
          absl::make_unique<SplitterFinderStreamProcessor>(
              "SplitFinder", internal_config.num_threads,
              FakeFindBestConditionConcurrentConsumerAlternate)};
  setup.split_finder_processor->StartWorkers();

  // Test case: Features alternate between valid and invalid, and processed in
  // reverse order.
  best_condition.set_split_score(1000.f);
  bool result = FindBestConditionConcurrentManager(
                    dataset, selected_examples, weights, config, config_link,
                    dt_config, setup, parent, internal_config, label_stats,
                    &best_condition, &random, &cache)
                    .value();

  EXPECT_EQ(cache.splitter_cache_list.size(), 10);
  EXPECT_EQ(cache.work_status_list.size(), 100);
  EXPECT_EQ(cache.condition_list.size(), 20);
  EXPECT_FALSE(result);

  random.seed(4321);
  best_condition.set_split_score(0.f);
  result = FindBestConditionConcurrentManager(
               dataset, selected_examples, weights, config, config_link,
               dt_config, setup, parent, internal_config, label_stats,
               &best_condition, &random, &cache)
               .value();
  EXPECT_TRUE(result);
  EXPECT_NEAR(best_condition.split_score(), 990.f, 0.001);
}

TEST(DecisionTree, GenericHyperParameterCategorical) {
  // Ensure the parameter is defined.
  model::proto::GenericHyperParameterSpecification hparam_def;
  EXPECT_OK(GetGenericHyperParameterSpecification({}, &hparam_def));
  EXPECT_FALSE(std::find_if(hparam_def.fields().begin(),
                            hparam_def.fields().end(), [](const auto& field) {
                              return field.first == kHParamCategoricalAlgorithm;
                            }) == hparam_def.fields().end());

  model::proto::GenericHyperParameters hparams;
  auto* field = hparams.add_fields();
  field->set_name(kHParamCategoricalAlgorithm);
  field->mutable_value()->set_categorical(kCategoricalAlgorithmRandom);

  proto::DecisionTreeTrainingConfig dt_config;
  EXPECT_FALSE(dt_config.categorical().has_random());

  absl::flat_hash_set<std::string> consumed_hparams;
  utils::GenericHyperParameterConsumer generic_hyper_params(hparams);
  EXPECT_OK(
      SetHyperParameters(&consumed_hparams, &dt_config, &generic_hyper_params));
  EXPECT_TRUE(dt_config.categorical().has_random());
}

TEST(DecisionTree, MidThreshold) {
  const auto test = [](float a, float b) {
    CHECK_GT(b, a);
    const float r = MidThreshold(a, b);
    CHECK(std::isfinite(r));
    CHECK_GT(r, a);
    CHECK_LE(r, b);
  };
  using nl = std::numeric_limits<float>;

  test(1.f, 2.f);
  test(-10.f, -5.f);

  test(0.f, nl::max());
  test(nl::min(), nl::max());

  test(1.f, std::nextafter(1.f, 2.f));
  test(std::nextafter(1.f, 0.f), 1.f);
  test(std::nextafter(1.f, 0.f), std::nextafter(1.f, 2.f));

  test(std::nextafter(nl::max(), 0.f), nl::max());
  test(nl::lowest(), std::nextafter(nl::lowest(), 0.f));
}

TEST(DecisionTree, FindBestNumericalDiscretizedSplitCartBase) {
  const std::vector<row_t> selected_examples = {0, 1, 2, 3};
  const std::vector<float> weights(selected_examples.size(), 1.f);

  const int num_binds = 6;
  // The best split is on the right size of the bin[1].
  std::vector<dataset::DiscretizedNumericalIndex> attributes = {0, 1, 4, 5};
  const std::vector<int32_t> labels = {0, 0, 1, 1};
  const int32_t num_label_classes = 2;

  proto::DecisionTreeTrainingConfig dt_config;
  utils::IntegerDistributionDouble label_distribution;
  label_distribution.SetNumClasses(num_label_classes);
  for (const auto example_idx : selected_examples) {
    label_distribution.Add(labels[example_idx], weights[example_idx]);
  }
  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  EXPECT_EQ(FindSplitLabelClassificationFeatureDiscretizedNumericalCart(
                selected_examples, weights, attributes, num_binds, labels,
                num_label_classes, /*na_replacement=*/0, /*min_num_obs=*/1,
                dt_config, label_distribution, -1, &best_condition, &cache),
            SplitSearchResult::kBetterSplitFound);

  // The equivalent threshold values are in [2,4]. We take the center.
  EXPECT_EQ(
      best_condition.condition().discretized_higher_condition().threshold(), 3);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 4);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 4);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 2);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 2);
  EXPECT_EQ(best_condition.na_value(), false);
}

TEST(UpliftCategoricalLabelDistribution, Base) {
  UpliftLabelDistribution dist;

  dist.InitializeAndClearCategoricalOutcome(
      /*num_unique_values_in_treatments_column=*/3,
      /*num_unique_in_outcomes_column=*/3);

  // Empty dist.
  EXPECT_EQ(dist.num_examples(), 0);
  EXPECT_EQ(dist.MinNumExamplesPerTreatment(), 0);
  EXPECT_EQ(dist.MeanOutcomePerTreatment(0), 0);
  EXPECT_EQ(dist.MeanOutcomePerTreatment(1), 0);
  EXPECT_EQ(dist.Uplift(), 0);
  EXPECT_EQ(dist.UpliftSplitScore(
                proto::DecisionTreeTrainingConfig::Uplift::EUCLIDEAN_DISTANCE),
            0);
  EXPECT_EQ(dist.UpliftSplitScore(
                proto::DecisionTreeTrainingConfig::Uplift::KULLBACK_LEIBLER),
            0);
  EXPECT_EQ(dist.UpliftSplitScore(
                proto::DecisionTreeTrainingConfig::Uplift::CHI_SQUARED),
            0);

  dist.AddCategoricalOutcome(/*outcome_value=*/2, /*treatment_value=*/2,
                             /*weight=*/1.f);
  dist.AddCategoricalOutcome(/*outcome_value=*/2, /*treatment_value=*/2,
                             /*weight=*/1.f);
  dist.AddCategoricalOutcome(/*outcome_value=*/2, /*treatment_value=*/1,
                             /*weight=*/1.f);
  dist.AddCategoricalOutcome(/*outcome_value=*/1, /*treatment_value=*/1,
                             /*weight=*/1.f);

  EXPECT_EQ(dist.num_examples(), 4);
  EXPECT_EQ(dist.MinNumExamplesPerTreatment(), 2);
  EXPECT_EQ(dist.MeanOutcomePerTreatment(0), 0.5);
  EXPECT_EQ(dist.MeanOutcomePerTreatment(1), 1.0);
  EXPECT_EQ(dist.Uplift(), 0.5);
  EXPECT_EQ(dist.UpliftSplitScore(
                proto::DecisionTreeTrainingConfig::Uplift::EUCLIDEAN_DISTANCE),
            0.5 * 0.5);
  EXPECT_EQ(dist.UpliftSplitScore(
                proto::DecisionTreeTrainingConfig::Uplift::KULLBACK_LEIBLER),
            std::log(2));
  EXPECT_EQ(dist.UpliftSplitScore(
                proto::DecisionTreeTrainingConfig::Uplift::CHI_SQUARED),
            (1.0 - 0.5) * (1.0 - 0.5) / 0.5);

  UpliftLabelDistribution dist2;
  dist2.InitializeAndClearLike(dist);

  // Empty dist.
  EXPECT_EQ(dist2.num_examples(), 0);

  dist2.Add(dist);
  EXPECT_EQ(dist2.num_examples(), 4);
  EXPECT_EQ(dist2.MinNumExamplesPerTreatment(), 2);
  EXPECT_EQ(dist2.MeanOutcomePerTreatment(0), 0.5);
  EXPECT_EQ(dist2.MeanOutcomePerTreatment(1), 1.0);
  EXPECT_EQ(dist2.Uplift(), 0.5);
  EXPECT_EQ(dist2.UpliftSplitScore(
                proto::DecisionTreeTrainingConfig::Uplift::EUCLIDEAN_DISTANCE),
            0.5 * 0.5);
  EXPECT_EQ(dist.UpliftSplitScore(
                proto::DecisionTreeTrainingConfig::Uplift::KULLBACK_LEIBLER),
            std::log(2));
  EXPECT_EQ(dist.UpliftSplitScore(
                proto::DecisionTreeTrainingConfig::Uplift::CHI_SQUARED),
            (1.0 - 0.5) * (1.0 - 0.5) / 0.5);

  // Empty dist.
  dist2.Sub(dist);
  EXPECT_EQ(dist2.num_examples(), 0);
}

TEST(UpliftCategoricalLabelDistribution, FromToLeafProto) {
  decision_tree::proto::NodeUpliftOutput source_leaf_proto =
      PARSE_TEST_PROTO(R"pb(
        sum_weights: 4
        sum_weights_per_treatment: 2
        sum_weights_per_treatment: 2
        sum_weights_per_treatment_and_outcome: 1
        sum_weights_per_treatment_and_outcome: 2
        num_examples_per_treatment: 2
        num_examples_per_treatment: 2
      )pb");
  UpliftLabelDistribution dist;
  dist.ImportSetFromLeafProto(source_leaf_proto);

  EXPECT_EQ(dist.num_examples(), 4);
  EXPECT_EQ(dist.MinNumExamplesPerTreatment(), 2);
  EXPECT_EQ(dist.MeanOutcomePerTreatment(0), 0.5);
  EXPECT_EQ(dist.MeanOutcomePerTreatment(1), 1.0);
  EXPECT_EQ(dist.Uplift(), 0.5);
  EXPECT_EQ(dist.UpliftSplitScore(
                proto::DecisionTreeTrainingConfig::Uplift::EUCLIDEAN_DISTANCE),
            0.5 * 0.5);

  decision_tree::proto::NodeUpliftOutput extracted_leaf_proto;
  dist.ExportToLeafProto(&extracted_leaf_proto);
  decision_tree::proto::NodeUpliftOutput expected_extracted_leaf_proto =
      PARSE_TEST_PROTO(R"pb(
        # Same as before.
        sum_weights: 4
        sum_weights_per_treatment: 2
        sum_weights_per_treatment: 2
        sum_weights_per_treatment_and_outcome: 1
        sum_weights_per_treatment_and_outcome: 2
        num_examples_per_treatment: 2
        num_examples_per_treatment: 2

        # Extra field.
        treatment_effect: 0.5
      )pb");
  EXPECT_THAT(extracted_leaf_proto, EqualsProto(expected_extracted_leaf_proto));
}

TEST(DecisionTree, FindBestSplitNumericalFeatureTaskCategoricalUplift) {
  const int num_examples = 8;
  std::vector<row_t> selected_examples(num_examples);
  std::iota(selected_examples.begin(), selected_examples.end(), 0);
  const std::vector<float> weights(num_examples, 1.f);

  std::vector<float> attributes = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<int32_t> outcomes = {1, 1, 1, 1, 1, 2, 1, 2};
  const std::vector<int32_t> treatments = {1, 2, 1, 2, 1, 2, 1, 2};

  const int32_t num_outcome_classes = 2 + 1;
  const int32_t num_treatment_classes = 2 + 1;

  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_uplift()->set_min_examples_in_treatment(1);
  // EUCLIDEAN_DISTANCE is the only split score that handle natively the
  // distance between pure distributions.
  dt_config.mutable_uplift()->set_split_score(
      proto::DecisionTreeTrainingConfig::Uplift::EUCLIDEAN_DISTANCE);
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::IN_NODE);

  CategoricalUpliftLabelStats label_stats(outcomes, num_outcome_classes,
                                          treatments, num_treatment_classes);

  auto& label_dist = label_stats.label_distribution;
  label_dist.InitializeAndClearCategoricalOutcome(num_outcome_classes,
                                                  num_treatment_classes);

  for (const auto example_idx : selected_examples) {
    label_dist.AddCategoricalOutcome(
        outcomes[example_idx], treatments[example_idx], weights[example_idx]);
  }

  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  EXPECT_EQ(FindSplitLabelUpliftCategoricalFeatureNumericalCart(
                selected_examples, weights, attributes, label_stats,
                /*na_replacement=*/2.5,
                /*min_num_obs=*/1, dt_config, -1, {}, &best_condition, &cache),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.condition().higher_condition().threshold(), 4.5f);
  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 8);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 8);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 4);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 4);
  EXPECT_EQ(best_condition.na_value(), false);
  EXPECT_NEAR(best_condition.split_score(), 1.f * 0.5f + 0.f * 0.5f - 0.25f,
              0.0001);
}

TEST(DecisionTree, FindBestSplitCategoricalFeatureTaskCategoricalUplift) {
  const int num_examples = 8;
  std::vector<row_t> selected_examples(num_examples);
  std::iota(selected_examples.begin(), selected_examples.end(), 0);
  const std::vector<float> weights(num_examples, 1.f);

  // The values  {1,2} or {3,4} are discriminative.
  std::vector<int32_t> attributes = {1, 2, 1, 2, 3, 4, 3, 4};
  const std::vector<int32_t> outcomes = {1, 1, 1, 1, 1, 2, 1, 2};
  const std::vector<int32_t> treatments = {1, 2, 1, 2, 1, 2, 1, 2};

  const int32_t num_attribute_classes = 4 + 1;
  const int32_t num_outcome_classes = 2 + 1;
  const int32_t num_treatment_classes = 2 + 1;

  proto::DecisionTreeTrainingConfig dt_config;
  dt_config.mutable_uplift()->set_min_examples_in_treatment(1);
  // EUCLIDEAN_DISTANCE is the only split score that handle natively the
  // distance between pure distributions.
  dt_config.mutable_uplift()->set_split_score(
      proto::DecisionTreeTrainingConfig::Uplift::EUCLIDEAN_DISTANCE);
  dt_config.mutable_internal()->set_sorting_strategy(
      proto::DecisionTreeTrainingConfig::Internal::IN_NODE);

  CategoricalUpliftLabelStats label_stats(outcomes, num_outcome_classes,
                                          treatments, num_treatment_classes);

  auto& label_dist = label_stats.label_distribution;
  label_dist.InitializeAndClearCategoricalOutcome(num_outcome_classes,
                                                  num_treatment_classes);

  for (const auto example_idx : selected_examples) {
    label_dist.AddCategoricalOutcome(
        outcomes[example_idx], treatments[example_idx], weights[example_idx]);
  }

  proto::NodeCondition best_condition;
  SplitterPerThreadCache cache;
  utils::RandomEngine random;
  EXPECT_EQ(FindSplitLabelUpliftCategoricalFeatureCategorical(
                selected_examples, weights, attributes, label_stats,
                num_attribute_classes,
                /*na_replacement=*/0,
                /*min_num_obs=*/1, dt_config, -1, {}, &best_condition, &cache,
                &random),
            SplitSearchResult::kBetterSplitFound);

  EXPECT_EQ(best_condition.num_training_examples_without_weight(), 8);
  EXPECT_EQ(best_condition.num_training_examples_with_weight(), 8);
  EXPECT_EQ(best_condition.num_pos_training_examples_without_weight(), 4);
  EXPECT_EQ(best_condition.num_pos_training_examples_with_weight(), 4);
  // Both the splits of attribute values [neg={1,2}, pos={3,4}] and [pos={1,2},
  // neg={3,4}] are equivalent and acceptable.
  EXPECT_NEAR(best_condition.split_score(), 1.f * 0.5f + 0.f * 0.5f - 0.25f,
              0.0001);
}

}  // namespace
}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
