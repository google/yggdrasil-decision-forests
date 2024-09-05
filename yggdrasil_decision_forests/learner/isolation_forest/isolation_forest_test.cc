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

#include "yggdrasil_decision_forests/learner/isolation_forest/isolation_forest.h"

#include <algorithm>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/isolation_forest/isolation_forest.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/isolation_forest/isolation_forest.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/test_utils.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::model::isolation_forest {
namespace {

using test::StatusIs;
using ::testing::ElementsAre;

class IsolationForestOnGaussians : public utils::TrainAndTestTester {
  proto::IsolationForestTrainingConfig* if_config() {
    return train_config_.MutableExtension(
        isolation_forest::proto::isolation_forest_config);
  }

  void SetUp() override {
    train_config_.set_learner(IsolationForestLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::ANOMALY_DETECTION);
    train_config_.add_features("f.*");
    train_config_.set_label("label");
    dataset_filename_ = "gaussians_train.csv";
    dataset_test_filename_ = "gaussians_test.csv";
    eval_options_.set_task(model::proto::Task::CLASSIFICATION);
    evaluation_override_type_ = model::proto::CLASSIFICATION;
  }
};

TEST_F(IsolationForestOnGaussians, DefaultHyperParameters) {
  TrainAndEvaluateModel();
  LOG(INFO) << "Model:\n" << model_->DescriptionAndStatistics(true);
  // Confirmed with Scikit-learn.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.980, 0.01);
  EXPECT_NEAR(evaluation_.classification().rocs(1).auc(), 0.995, 0.004);
}

TEST_F(IsolationForestOnGaussians, ModelStructure) {
  auto* if_config = train_config_.MutableExtension(
      isolation_forest::proto::isolation_forest_config);
  if_config->set_num_trees(5);
  TrainAndEvaluateModel();

  EXPECT_EQ(model_->task(), model::proto::Task::ANOMALY_DETECTION);
  EXPECT_EQ(model_->label_col_idx(), 2);
  EXPECT_THAT(model_->input_features(), ElementsAre(0, 1));

  auto if_model = dynamic_cast<const IsolationForestModel*>(model_.get());
  EXPECT_EQ(if_model->num_trees(), 5);
  EXPECT_GT(if_model->NumNodes(), if_model->num_trees() * 32);
}

TEST_F(IsolationForestOnGaussians, MaxDepth) {
  auto* if_config = train_config_.MutableExtension(
      isolation_forest::proto::isolation_forest_config);
  if_config->set_num_trees(5);
  if_config->set_subsample_count(128);
  TrainAndEvaluateModel();
  auto* df_model = dynamic_cast<model::DecisionForestInterface*>(model_.get());
  ASSERT_NE(df_model, nullptr);
  int max_depth = -1;
  for (const auto& tree : df_model->decision_trees()) {
    max_depth = std::max(max_depth, tree->MaximumDepth());
  }
  EXPECT_EQ(max_depth, 7);
}

TEST_F(IsolationForestOnGaussians, SmallModel) {
  auto* if_config = train_config_.MutableExtension(
      isolation_forest::proto::isolation_forest_config);
  if_config->set_num_trees(32);
  if_config->set_subsample_count(32);
  TrainAndEvaluateModel();
  LOG(INFO) << "Model:\n" << model_->DescriptionAndStatistics(true);
  // Confirmed with Scikit-learn.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.805, 0.1);
  EXPECT_NEAR(evaluation_.classification().rocs(1).auc(), 0.975, 0.02);
}

class IsolationForestOnAdult : public utils::TrainAndTestTester {
  proto::IsolationForestTrainingConfig* if_config() {
    return train_config_.MutableExtension(
        isolation_forest::proto::isolation_forest_config);
  }

  void SetUp() override {
    train_config_.set_learner(IsolationForestLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::ANOMALY_DETECTION);
    train_config_.set_label("income");
    dataset_filename_ = "adult_train.csv";
    dataset_test_filename_ = "adult_test.csv";
    eval_options_.set_task(model::proto::Task::CLASSIFICATION);
    evaluation_override_type_ = model::proto::CLASSIFICATION;
  }
};

TEST_F(IsolationForestOnAdult, DefaultHyperParameters) {
  TrainAndEvaluateModel();
  LOG(INFO) << "Model:\n" << model_->DescriptionAndStatistics(true);
  // Confirmed with Scikit-learn.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.759, 0.02);
  EXPECT_NEAR(evaluation_.classification().rocs(1).auc(), 0.615, 0.03);
}

TEST_F(IsolationForestOnAdult, SmallModel) {
  auto* if_config = train_config_.MutableExtension(
      isolation_forest::proto::isolation_forest_config);
  if_config->set_num_trees(32);
  if_config->set_subsample_count(32);
  TrainAndEvaluateModel();
  LOG(INFO) << "Model:\n" << model_->DescriptionAndStatistics(true);
  // Confirmed with Scikit-learn.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.715, 0.06);
  EXPECT_NEAR(evaluation_.classification().rocs(1).auc(), 0.624, 0.06);
}

TEST_F(IsolationForestOnAdult, Oblique) {
  auto* if_config = train_config_.MutableExtension(
      isolation_forest::proto::isolation_forest_config);
  if_config->mutable_decision_tree()->mutable_sparse_oblique_split();
  TrainAndEvaluateModel();
  LOG(INFO) << "Model:\n" << model_->DescriptionAndStatistics(true);
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.77, 0.01);
  EXPECT_NEAR(evaluation_.classification().rocs(1).auc(), 0.615, 0.03);
}

class IsolationForestOnMammographicMasses : public utils::TrainAndTestTester {
  proto::IsolationForestTrainingConfig* if_config() {
    return train_config_.MutableExtension(
        isolation_forest::proto::isolation_forest_config);
  }

  void SetUp() override {
    train_config_.set_learner(IsolationForestLearner::kRegisteredName);
    train_config_.set_task(model::proto::Task::ANOMALY_DETECTION);
    train_config_.set_label("Severity");
    dataset_filename_ = "mammographic_masses.csv";
    // Warning: For this dataset, evaluate on the training data in the test.
    dataset_test_filename_ = "mammographic_masses.csv";
    auto* label_guide = guide_.add_column_guides();
    label_guide->set_column_name_pattern("Severity");
    label_guide->set_type(dataset::proto::CATEGORICAL);
    eval_options_.set_task(model::proto::Task::CLASSIFICATION);
    evaluation_override_type_ = model::proto::CLASSIFICATION;
  }
};

TEST_F(IsolationForestOnMammographicMasses, DefaultHyperParameters) {
  TrainAndEvaluateModel();
  LOG(INFO) << "Model:\n" << model_->DescriptionAndStatistics(true);
  // Confirmed with Scikit-learn.
  EXPECT_NEAR(metric::Accuracy(evaluation_), 0.507, 0.027);
  EXPECT_NEAR(evaluation_.classification().rocs(1).auc(), 0.479, 0.058);
}

TEST_F(IsolationForestOnMammographicMasses, Oblique) {
  auto* if_config = train_config_.MutableExtension(
      isolation_forest::proto::isolation_forest_config);
  if_config->mutable_decision_tree()->mutable_sparse_oblique_split();
  TrainAndEvaluateModel();

  auto* df_model = dynamic_cast<model::DecisionForestInterface*>(model_.get());
  ASSERT_NE(df_model, nullptr);
  ASSERT_EQ(df_model->decision_trees().size(), if_config->num_trees());
  const auto& first_tree = df_model->decision_trees()[0];
  EXPECT_TRUE(first_tree->root()
                  .node()
                  .condition()
                  .condition()
                  .has_oblique_condition());

  EXPECT_GE(metric::Accuracy(evaluation_), 0.507);
  EXPECT_GE(evaluation_.classification().rocs(1).auc(), 0.45);
}

TEST(IsolationForest, BadTask) {
  std::string dataset_path = absl::StrCat(
      "csv:", file::JoinPath(test::DataRootDirectory(),
                             "yggdrasil_decision_forests/"
                             "test_data/dataset/gaussians_train.csv"));

  ASSERT_OK_AND_ASSIGN(auto dataspec, dataset::CreateDataSpec(dataset_path));

  model::proto::TrainingConfig train_config;
  train_config.set_learner(IsolationForestLearner::kRegisteredName);
  train_config.set_task(model::proto::Task::CLASSIFICATION);
  train_config.add_features("f.*");
  train_config.set_label("label");
  ASSERT_OK_AND_ASSIGN(auto learner, model::GetLearner(train_config));

  EXPECT_THAT(learner->TrainWithStatus(dataset_path, dataspec).status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(DefaultMaximumDepth, Base) {
  EXPECT_EQ(internal::DefaultMaximumDepth(254), 8);
  EXPECT_EQ(internal::DefaultMaximumDepth(255), 8);
  EXPECT_EQ(internal::DefaultMaximumDepth(256), 8);
  EXPECT_EQ(internal::DefaultMaximumDepth(257), 9);
}

TEST(SampleExamples, Base) {
  utils::RandomEngine rnd;
  const auto samples = internal::SampleExamples(100, 10, &rnd);
  EXPECT_EQ(samples.size(), 10);
  EXPECT_TRUE(std::is_sorted(samples.begin(), samples.end()));

  // Look for duplicates
  for (int i = 1; i < samples.size(); i++) {
    EXPECT_LT(samples[i - 1], samples[i]);
  }
}

TEST(GetNumExamplesPerTrees, Default) {
  proto::IsolationForestTrainingConfig if_config;
  EXPECT_EQ(internal::GetNumExamplesPerTrees(if_config, 1000), 256);
}

TEST(GetNumExamplesPerTrees, Count) {
  proto::IsolationForestTrainingConfig if_config;
  if_config.set_subsample_count(5);
  EXPECT_EQ(internal::GetNumExamplesPerTrees(if_config, 100), 5);
}

TEST(GetNumExamplesPerTrees, Rate) {
  proto::IsolationForestTrainingConfig if_config;
  if_config.set_subsample_ratio(0.5f);
  EXPECT_EQ(internal::GetNumExamplesPerTrees(if_config, 100), 50);
}

TEST(FindSplit, Numerical) {
  for (int seed = 0; seed < 10; seed++) {
    // This is a stochastic test.
    utils::RandomEngine rnd(seed);

    internal::Configuration config;
    proto::IsolationForestTrainingConfig if_config;
    config.if_config = &if_config;
    config.config_link.add_features(0);  // Only select "f1".

    decision_tree::NodeWithChildren node;

    dataset::VerticalDataset dataset;
    dataset::AddNumericalColumn("f1", dataset.mutable_data_spec());
    dataset::AddNumericalColumn("f2", dataset.mutable_data_spec());
    ASSERT_OK(dataset.CreateColumnsFromDataspec());
    ASSERT_OK_AND_ASSIGN(auto* column,
                         dataset.MutableColumnWithCastWithStatus<
                             dataset::VerticalDataset::NumericalColumn>(0));
    *column->mutable_values() = {1, 2, 4, 100};

    ASSERT_OK_AND_ASSIGN(
        const bool found_condition,
        FindSplit(config, dataset,
                  {0, 1, 2},  // Don't select the example with value "100".
                  &node, &rnd));
    EXPECT_TRUE(found_condition);
    EXPECT_EQ(node.node().condition().attribute(), 0);  // Always "f1".
    EXPECT_TRUE(node.node().condition().condition().has_higher_condition());
    const float threshold =
        node.node().condition().condition().higher_condition().threshold();
    EXPECT_GE(threshold, 1.0f);
    EXPECT_LE(threshold, 4.0f);  // The value 100 is clearly ignored.
  }
}

TEST(GetGenericHyperParameterSpecification, Base) {
  model::proto::TrainingConfig train_config;
  train_config.set_learner(IsolationForestLearner::kRegisteredName);
  train_config.set_task(model::proto::Task::ANOMALY_DETECTION);
  ASSERT_OK_AND_ASSIGN(auto learner, model::GetLearner(train_config));

  ASSERT_OK_AND_ASSIGN(const auto hp_specs,
                       learner->GetGenericHyperParameterSpecification());

  for (absl::string_view field : {
           IsolationForestLearner::kHParamNumTrees,
           IsolationForestLearner::kHParamSubsampleRatio,
           IsolationForestLearner::kHParamSubsampleCount,
       }) {
    EXPECT_TRUE(hp_specs.fields().contains(field));
  }
}

TEST(GetGenericHyperParameterSpecification,
     GenericHyperParameterMutualExclusive) {
  model::proto::TrainingConfig train_config;
  train_config.set_learner(IsolationForestLearner::kRegisteredName);
  train_config.set_task(model::proto::Task::ANOMALY_DETECTION);
  ASSERT_OK_AND_ASSIGN(auto learner, model::GetLearner(train_config));

  ASSERT_OK_AND_ASSIGN(const auto hparam_def,
                       learner->GetGenericHyperParameterSpecification());

  for (const auto& field : hparam_def.fields()) {
    if (field.second.has_mutual_exclusive()) {
      bool is_default = field.second.mutual_exclusive().is_default();
      const auto& other_parameters =
          field.second.mutual_exclusive().other_parameters();
      for (const auto& other_parameter : other_parameters) {
        auto other_param_it =
            std::find_if(hparam_def.fields().begin(), hparam_def.fields().end(),
                         [other_parameter](const auto& field) {
                           return other_parameter == field.first;
                         });
        EXPECT_FALSE(other_param_it == hparam_def.fields().end());
        EXPECT_THAT(
            other_param_it->second.mutual_exclusive().other_parameters(),
            testing::Contains(field.first));
        if (is_default) {
          EXPECT_FALSE(other_param_it->second.mutual_exclusive().is_default());
        }
      }
    }
  }
}

}  // namespace
}  // namespace yggdrasil_decision_forests::model::isolation_forest
