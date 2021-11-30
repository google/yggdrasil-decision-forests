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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/training.h"

#include "gmock/gmock.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/label_accessor.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/multi_thread/multi_thread.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace {

// Generic training loop. Displays all the intermediate results.
template <typename LabelAccessor, typename Tester>
void GenericTrainingLoop(LabelAccessor* label_accessor, Tester* tester,
                         const int num_threads = 2) {
  utils::concurrency::ThreadPool thread_pool("", num_threads);
  thread_pool.StartWorkers();

  auto example_to_node =
      CreateExampleToNodeMap(tester->dataset_->num_examples());

  // Initializer the decision tree.
  auto tree_builder = TreeBuilder::Create(tester->config_, tester->config_link_,
                                          tester->dt_config_)
                          .value();

  // Computes the entire training dataset label statistics.
  LabelStatsPerNode label_stats_per_node(1);
  CHECK_OK(tree_builder->AggregateLabelStatistics(
      *label_accessor, &label_stats_per_node.front(), &thread_pool));

  LOG(INFO) << "Input features: "
            << absl::StrJoin(tester->config_link_.features().begin(),
                             tester->config_link_.features().end(), ", ");

  // Set the root tree value using the global label statistics.
  CHECK_OK(tree_builder->SetRootValue(label_stats_per_node.front()));

  for (int iter_idx = 0; iter_idx < tester->dt_config_.max_depth();
       iter_idx++) {
    LOG(INFO) << "Iteration # " << iter_idx;

    LOG(INFO) << "Label statistics (" << label_stats_per_node.size() << "):";
    for (const auto& label_stats : label_stats_per_node) {
      LOG(INFO) << label_stats.DebugString();
    }

    // Collect the input features. All the nodes are seeing all the same
    // features.
    std::vector<std::vector<int>> input_features_per_node;
    for (int node_idx = 0; node_idx < label_stats_per_node.size(); node_idx++) {
      input_features_per_node.push_back(
          {tester->config_link_.features().begin(),
           tester->config_link_.features().end()});
    }

    // Look for the best splits.
    SplitPerOpenNode splits;
    CHECK_OK(tree_builder->FindBestSplits(
        {input_features_per_node, example_to_node, tester->data_spec_,
         *label_accessor, label_stats_per_node, iter_idx > 0,
         tester->dataset_.get(), &splits}));
    const auto num_valid_splits = NumValidSplits(splits);
    LOG(INFO) << "Found " << num_valid_splits << " / " << splits.size()
              << " split(s)";

    if (num_valid_splits == 0) {
      break;
    }

    // Merge the best split with empty splits. This is a No-op operation.
    SplitPerOpenNode no_splits(splits.size());
    CHECK_OK(MergeBestSplits(no_splits, &splits));

    LOG(INFO) << "Splits (" << splits.size() << "):";
    for (const auto& split : splits) {
      LOG(INFO) << "\tCondition:\n" << split.condition.DebugString();
      LOG(INFO) << "\tNegative label stats:\n"
                << split.label_statistics[0].DebugString();
      LOG(INFO) << "\tPositive label stats:\n"
                << split.label_statistics[1].DebugString();
      LOG(INFO);
    }

    // Add the found splits to the tree structure.
    const auto node_remapping = tree_builder->ApplySplitToTree(splits).value();
    LOG(INFO) << "Remapping:";
    for (int i = 0; i < node_remapping.size(); i++) {
      LOG(INFO) << "\t" << i << " -> " << node_remapping[i].indices[0] << " + "
                << node_remapping[i].indices[1];
    }

    std::string description;
    tree_builder->tree().AppendModelStructure(
        tester->data_spec_, tester->config_link_.label(), &description);
    LOG(INFO) << "Tree:\n" << description;

    // Evaluate the split for all the active training examples.
    SplitEvaluationPerOpenNode split_evaluation;
    CHECK_OK(EvaluateSplits(example_to_node, splits, &split_evaluation,
                            tester->dataset_.get(), &thread_pool));

    LOG(INFO) << "Split evaluation (" << split_evaluation.size() << "):";
    for (const auto& evaluation : split_evaluation) {
      LOG(INFO) << "\t" << evaluation.size() << " byte(s)";
      LOG(INFO) << "\t (first 10 values) "
                << utils::bitmap::ToStringBit(
                       evaluation, std::min<int>(10, evaluation.size() * 8));
    }

    // Update the example->node map.
    CHECK_OK(UpdateExampleNodeMap(splits, split_evaluation, node_remapping,
                                  &example_to_node, &thread_pool));

    LOG(INFO) << "Example to node map (first 10 values):";
    ExampleIndex example_idx = 0;
    for (const auto& node_idx : example_to_node) {
      LOG(INFO) << "\t" << node_idx;
      if (example_idx > 10) {
        break;
      }
      example_idx++;
    }

    // Update the label statistics.
    const auto previous_label_stats_per_node_size = label_stats_per_node.size();
    CHECK_OK(
        UpdateLabelStatistics(splits, node_remapping, &label_stats_per_node));
    LOG(INFO) << "Update the number of open nodes "
              << previous_label_stats_per_node_size << " -> "
              << label_stats_per_node.size();
  }
}

// Prepare the dataset for training: Create the dataspec, split the dataset into
// shards, and generate the dataset cache.
void PrepareDataset(const model::proto::TrainingConfig& train_config,
                    const absl::string_view dataset_filename,
                    const absl::string_view dataset_name,
                    dataset::proto::DataSpecification* data_spec,
                    std::string* cache_path) {
  dataset::VerticalDataset vertical_dataset;

  // Infer the dataspec.
  const auto dataset_path = absl::StrCat(
      "csv:", file::JoinPath(test::DataRootDirectory(),
                             "yggdrasil_decision_forests/test_data/"
                             "dataset",
                             dataset_filename));

  dataset::CreateDataSpec(dataset_path, false, {}, data_spec);

  // Shard the dataset.
  CHECK_OK(LoadVerticalDataset(dataset_path, *data_spec, &vertical_dataset));
  const auto sharded_dataset_path = absl::StrCat(
      "csv:", file::JoinPath(test::TmpDirectory(), "train.csv@20"));
  CHECK_OK(SaveVerticalDataset(
      vertical_dataset, sharded_dataset_path,
      /*num_records_by_shard=*/vertical_dataset.nrow() / 20));

  // Create the dataset cache.

  // Multi-threads distribution strategy.
  distribute::proto::Config distribute_config;
  distribute_config.set_implementation_key("MULTI_THREAD");
  distribute_config.MutableExtension(distribute::proto::multi_thread)
      ->set_num_workers(5);

  dataset_cache::proto::CreateDatasetCacheConfig create_dataset_cache_config;
  create_dataset_cache_config.set_max_unique_values_for_discretized_numerical(
      32);
  int32_t label_idx;
  CHECK_OK(dataset::GetSingleColumnIdxFromName(train_config.label(), *data_spec,
                                               &label_idx));
  create_dataset_cache_config.set_label_column_idx(label_idx);

  *cache_path = file::JoinPath(test::TmpDirectory(), dataset_name, "cache");

  EXPECT_OK(dataset_cache::CreateDatasetCacheFromShardedFiles(
      sharded_dataset_path, *data_spec, nullptr, *cache_path,
      create_dataset_cache_config, distribute_config));
}

class AdultClassificationDataset : public ::testing::Test {
 public:
  void SetUp() override {
    config_.set_task(model::proto::Task::CLASSIFICATION);
    config_.set_label("income");

    PrepareDataset(config_, "adult_train.csv", "adult", &data_spec_,
                   &cache_path_);

    // Load the dataset cache.
    dataset_cache::proto::DatasetCacheReaderOptions cache_reader_options;
    dataset_ = dataset_cache::DatasetCacheReader::Create(cache_path_,
                                                         cache_reader_options)
                   .value();
    LOG(INFO) << "Cache meta-data:\n" << dataset_->meta_data().DebugString();

    CHECK_OK(AbstractLearner::LinkTrainingConfig(config_, data_spec_,
                                                 &config_link_));
    dt_config_.set_max_depth(10);
    decision_tree::SetDefaultHyperParameters(&dt_config_);
  }

  dataset::proto::DataSpecification data_spec_;
  std::string cache_path_;
  std::unique_ptr<dataset_cache::DatasetCacheReader> dataset_;

  // Training configuration.
  model::proto::TrainingConfig config_;
  model::proto::TrainingConfigLinking config_link_;
  decision_tree::proto::DecisionTreeTrainingConfig dt_config_;
};

TEST_F(AdultClassificationDataset, AggregateLabelStatisticsClassification) {
  utils::concurrency::ThreadPool thread_pool("", 2);
  thread_pool.StartWorkers();

  auto example_to_node = CreateExampleToNodeMap(dataset_->num_examples());

  // Accessor to the label data.
  ClassificationLabelAccessor label_accessor(
      dataset_->categorical_labels(), dataset_->weights(), /*num_classes=*/3);

  // Initializer the decision tree.
  auto tree_builder =
      TreeBuilder::Create(config_, config_link_, dt_config_).value();

  // Computes the entire training dataset label statistics.
  decision_tree::proto::LabelStatistics label_stats;
  CHECK_OK(tree_builder->AggregateLabelStatistics(label_accessor, &label_stats,
                                                  &thread_pool));

  const decision_tree::proto::LabelStatistics expected_label_stats =
      PARSE_TEST_PROTO(
          R"pb(
            num_examples: 22792
            classification {
              labels { counts: 0 counts: 17308 counts: 5484 sum: 22792 }
            }
          )pb");
  EXPECT_THAT(label_stats, test::EqualsProto(expected_label_stats));
}

TEST_F(AdultClassificationDataset, ManualCheck) {
  utils::concurrency::ThreadPool thread_pool("", 2);
  thread_pool.StartWorkers();

  auto example_to_node = CreateExampleToNodeMap(dataset_->num_examples());

  // Accessor to the label data.
  ClassificationLabelAccessor label_accessor(
      dataset_->categorical_labels(), dataset_->weights(), /*num_classes=*/3);

  // Initializer the decision tree.
  auto tree_builder =
      TreeBuilder::Create(config_, config_link_, dt_config_).value();

  // Computes the entire training dataset label statistics.
  decision_tree::proto::LabelStatistics label_stats;
  CHECK_OK(tree_builder->AggregateLabelStatistics(label_accessor, &label_stats,
                                                  &thread_pool));

  const decision_tree::proto::LabelStatistics expected_label_stats =
      PARSE_TEST_PROTO(
          R"pb(
            num_examples: 22792
            classification {
              labels { counts: 0 counts: 17308 counts: 5484 sum: 22792 }
            }
          )pb");
  EXPECT_THAT(label_stats, test::EqualsProto(expected_label_stats));

  CHECK_OK(tree_builder->SetRootValue(label_stats));
  EXPECT_EQ(
      tree_builder->tree().root().node().classifier().distribution().sum(),
      22792.0);

  {
    // Best split on "workclass" feature.
    SplitPerOpenNode splits;
    CHECK_OK(tree_builder->FindBestSplits({{{1 /*workclass*/}},
                                           example_to_node,
                                           data_spec_,
                                           label_accessor,
                                           {label_stats},
                                           false,
                                           dataset_.get(),
                                           &splits}));
    EXPECT_EQ(NumValidSplits(splits), 1);
    EXPECT_EQ(splits.size(), 1);

    decision_tree::proto::NodeCondition expected_condition = PARSE_TEST_PROTO(
        R"pb(
          na_value: false
          attribute: 1
          condition { contains_bitmap_condition { elements_bitmap: "`" } }
          num_training_examples_without_weight: 22792
          num_training_examples_with_weight: 22792
          split_score: 0.008267824
          num_pos_training_examples_without_weight: 1479
          num_pos_training_examples_with_weight: 1479
        )pb");
    EXPECT_THAT(splits.front().condition,
                test::EqualsProto(expected_condition));

    decision_tree::proto::LabelStatistics expected_neg_statistics =
        PARSE_TEST_PROTO(
            R"pb(
              num_examples: 21313
              classification {
                labels { counts: 0 counts: 16515 counts: 4798 sum: 21313 }
              }
            )pb");
    EXPECT_THAT(splits.front().label_statistics[0],
                test::EqualsProto(expected_neg_statistics));

    decision_tree::proto::LabelStatistics expected_pos_statistics =
        PARSE_TEST_PROTO(
            R"pb(
              num_examples: 1479
              classification {
                labels { counts: 0 counts: 793 counts: 686 sum: 1479 }
              }
            )pb");
    EXPECT_THAT(splits.front().label_statistics[1],
                test::EqualsProto(expected_pos_statistics));
  }

  {
    // Best split on "education_num" feature.
    SplitPerOpenNode splits;
    CHECK_OK(tree_builder->FindBestSplits({{{4 /*education_num*/}},
                                           example_to_node,
                                           data_spec_,
                                           label_accessor,
                                           {label_stats},
                                           false,
                                           dataset_.get(),
                                           &splits}));
    EXPECT_EQ(NumValidSplits(splits), 1);
    EXPECT_EQ(splits.size(), 1);

    decision_tree::proto::NodeCondition expected_condition = PARSE_TEST_PROTO(
        R"pb(
          na_value: false
          attribute: 4
          condition { higher_condition { threshold: 12.5 } }
          num_training_examples_without_weight: 22792
          num_training_examples_with_weight: 22792
          split_score: 0.05001844
          num_pos_training_examples_without_weight: 5672
          num_pos_training_examples_with_weight: 5672
        )pb");
    EXPECT_THAT(splits.front().condition,
                test::EqualsProto(expected_condition));

    decision_tree::proto::LabelStatistics expected_neg_statistics =
        PARSE_TEST_PROTO(
            R"pb(
              num_examples: 17120
              classification {
                labels { counts: 0 counts: 14393 counts: 2727 sum: 17120 }
              }
            )pb");
    EXPECT_THAT(splits.front().label_statistics[0],
                test::EqualsProto(expected_neg_statistics));

    decision_tree::proto::LabelStatistics expected_pos_statistics =
        PARSE_TEST_PROTO(
            R"pb(
              num_examples: 5672
              classification {
                labels { counts: 0 counts: 2915 counts: 2757 sum: 5672 }
              }
            )pb");
    EXPECT_THAT(splits.front().label_statistics[1],
                test::EqualsProto(expected_pos_statistics));
  }

  // Best split on "age" feature.
  SplitPerOpenNode splits;
  CHECK_OK(tree_builder->FindBestSplits({{{0 /*age*/}},
                                         example_to_node,
                                         data_spec_,
                                         label_accessor,
                                         {label_stats},
                                         false,
                                         dataset_.get(),
                                         &splits}));
  EXPECT_EQ(NumValidSplits(splits), 1);
  EXPECT_EQ(splits.size(), 1);

  decision_tree::proto::NodeCondition expected_condition = PARSE_TEST_PROTO(
      R"pb(
        attribute: 0
        na_value: true
        condition { higher_condition { threshold: 27.5 } }
        num_training_examples_without_weight: 22792
        num_training_examples_with_weight: 22792
        split_score: 0.05020765
        num_pos_training_examples_without_weight: 17223
        num_pos_training_examples_with_weight: 17223
      )pb");
  EXPECT_THAT(splits.front().condition, test::EqualsProto(expected_condition));

  decision_tree::proto::LabelStatistics expected_neg_statistics =
      PARSE_TEST_PROTO(
          R"pb(
            num_examples: 5569
            classification {
              labels { counts: 0 counts: 5388 counts: 181 sum: 5569 }
            }
          )pb");
  EXPECT_THAT(splits.front().label_statistics[0],
              test::EqualsProto(expected_neg_statistics));

  decision_tree::proto::LabelStatistics expected_pos_statistics =
      PARSE_TEST_PROTO(
          R"pb(
            num_examples: 17223
            classification {
              labels { counts: 0 counts: 11920 counts: 5303 sum: 17223 }
            }
          )pb");
  EXPECT_THAT(splits.front().label_statistics[1],
              test::EqualsProto(expected_pos_statistics));

  const auto node_remapping = tree_builder->ApplySplitToTree(splits).value();
  EXPECT_EQ(node_remapping.size(), 1);
  EXPECT_EQ(node_remapping.front().indices[0], 0);
  EXPECT_EQ(node_remapping.front().indices[1], 1);
  EXPECT_EQ(tree_builder->tree().NumNodes(), 3);
  EXPECT_THAT(tree_builder->tree().root().node().condition(),
              test::EqualsProto(expected_condition));

  SplitEvaluationPerOpenNode split_evaluation;
  CHECK_OK(EvaluateSplits(example_to_node, splits, &split_evaluation,
                          dataset_.get(), &thread_pool));

  EXPECT_EQ(split_evaluation.size(), 1);
  EXPECT_EQ(split_evaluation.front().size(), (22792 + 7) / 8);
  EXPECT_EQ(utils::bitmap::ToStringBit(split_evaluation[0], 10), "1011101111");

  CHECK_OK(UpdateExampleNodeMap(splits, split_evaluation, node_remapping,
                                &example_to_node, &thread_pool));
  EXPECT_EQ(example_to_node.size(), dataset_->num_examples());
  EXPECT_EQ(example_to_node[0], 1);
  EXPECT_EQ(example_to_node[1], 0);
  EXPECT_EQ(example_to_node[2], 1);
  EXPECT_EQ(example_to_node[3], 1);

  LabelStatsPerNode new_label_stats;
  CHECK_OK(UpdateLabelStatistics(splits, node_remapping, &new_label_stats));

  const decision_tree::proto::LabelStatistics expected_new_label_stats_0 =
      PARSE_TEST_PROTO(
          R"pb(
            num_examples: 5569
            classification {
              labels { counts: 0 counts: 5388 counts: 181 sum: 5569 }
            }
          )pb");
  const decision_tree::proto::LabelStatistics expected_new_label_stats_1 =
      PARSE_TEST_PROTO(
          R"pb(
            num_examples: 17223
            classification {
              labels { counts: 0 counts: 11920 counts: 5303 sum: 17223 }
            }
          )pb");
  EXPECT_THAT(new_label_stats[0],
              test::EqualsProto(expected_new_label_stats_0));
  EXPECT_THAT(new_label_stats[1],
              test::EqualsProto(expected_new_label_stats_1));
}

// Trains a tree with multiple-node layers.
TEST_F(AdultClassificationDataset, End2EndTrainClassification_2Threads) {
  // Accessor to the label data.
  ClassificationLabelAccessor label_accessor(
      dataset_->categorical_labels(), dataset_->weights(), /*num_classes=*/3);

  GenericTrainingLoop(&label_accessor, this, /*num_threads=*/2);
}

// Trains a tree with multiple-node layers.
TEST_F(AdultClassificationDataset, End2EndTrainClassification_5Threads) {
  // Accessor to the label data.
  ClassificationLabelAccessor label_accessor(
      dataset_->categorical_labels(), dataset_->weights(), /*num_classes=*/3);

  GenericTrainingLoop(&label_accessor, this, /*num_threads=*/5);
}

class AbaloneRegressionDataset : public ::testing::Test {
 public:
  void SetUp() override {
    config_.set_task(model::proto::Task::REGRESSION);
    config_.set_label("Rings");

    PrepareDataset(config_, "abalone.csv", "abalone", &data_spec_,
                   &cache_path_);

    // Load the dataset cache.
    dataset_cache::proto::DatasetCacheReaderOptions cache_reader_options;
    dataset_ = dataset_cache::DatasetCacheReader::Create(cache_path_,
                                                         cache_reader_options)
                   .value();
    LOG(INFO) << "Cache meta-data:\n" << dataset_->meta_data().DebugString();

    CHECK_OK(AbstractLearner::LinkTrainingConfig(config_, data_spec_,
                                                 &config_link_));
    dt_config_.set_max_depth(10);
    decision_tree::SetDefaultHyperParameters(&dt_config_);
  }

  dataset::proto::DataSpecification data_spec_;
  std::string cache_path_;
  std::unique_ptr<dataset_cache::DatasetCacheReader> dataset_;

  // Training configuration.
  model::proto::TrainingConfig config_;
  model::proto::TrainingConfigLinking config_link_;
  decision_tree::proto::DecisionTreeTrainingConfig dt_config_;
};

// Trains a tree with multiple-node layers.
TEST_F(AbaloneRegressionDataset, End2EndTrainRegression) {
  // Accessor to the label data.
  RegressionLabelAccessor label_accessor(dataset_->regression_labels(),
                                         dataset_->weights());
  GenericTrainingLoop(&label_accessor, this);
}

}  // namespace
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
