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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/multi_thread/multi_thread.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/test_utils.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace dataset_cache {
namespace {

class End2End : public ::testing::Test {
 public:
  void SetUp() override {
    // Prepare the dataspec.
    const auto dataset_path = absl::StrCat(
        "csv:",
        file::JoinPath(test::DataRootDirectory(),
                       "yggdrasil_decision_forests/test_data/"
                       "dataset/adult_train.csv"));
    dataset::CreateDataSpec(dataset_path, false, {}, &data_spec_);

    // Shard the dataset.
    CHECK_OK(LoadVerticalDataset(dataset_path, data_spec_, &dataset_));
    const auto sharded_dataset_path = absl::StrCat(
        "csv:", file::JoinPath(test::TmpDirectory(), "sharded_dataset.csv@20"));
    CHECK_OK(
        SaveVerticalDataset(dataset_, sharded_dataset_path,
                            /*num_records_by_shard=*/dataset_.nrow() / 20));

    // Multi-threads distribution.
    distribute::proto::Config distribute_config;
    distribute_config.set_implementation_key("MULTI_THREAD");

    proto::CreateDatasetCacheConfig config;

    // Tha "age" column will not be discretized (72 unique values), while the
    // "education_num" will be (15 unique values).
    config.set_max_unique_values_for_discretized_numerical(32);
    int32_t label_column_idx;
    CHECK_OK(dataset::GetSingleColumnIdxFromName("income", data_spec_,
                                                 &label_column_idx));
    config.set_label_column_idx(label_column_idx);

    cache_path_ = file::JoinPath(test::TmpDirectory(), "cache");

    EXPECT_OK(CreateDatasetCacheFromShardedFiles(sharded_dataset_path,
                                                 data_spec_, {}, cache_path_,
                                                 config, distribute_config));

    // Try to generate the cache again. Will be instantaneous as the cache is
    // already there.
    EXPECT_OK(CreateDatasetCacheFromShardedFiles(sharded_dataset_path,
                                                 data_spec_, {}, cache_path_,
                                                 config, distribute_config));

    meta_data_ = LoadCacheMetadata(cache_path_).value();
  }

  dataset::proto::DataSpecification data_spec_;
  std::string cache_path_;
  proto::CacheMetadata meta_data_;
  dataset::VerticalDataset dataset_;
};

// Check an in order numerical column.
TEST_F(End2End, NumericalInOrder) {
  ShardedFloatColumnReader reader;
  CHECK_OK(reader.Open(
      file::JoinPath(cache_path_, kFilenameRaw,
                     absl::StrCat(kFilenameColumn, 0),  // "age" column.
                     kFilenameShardNoUnderscore),
      /*max_num_values=*/1000,
      /*begin_shard_idx=*/0,
      /*end_shard_idx=*/meta_data_.num_shards_in_feature_cache()));

  const auto& ground_truth_values =
      dataset_.ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(0)
          ->values();

  size_t num_examples = 0;
  double sum = 0;
  while (true) {
    CHECK_OK(reader.Next());
    const auto values = reader.Values();
    if (values.empty()) {
      break;
    }
    for (const float value : values) {
      EXPECT_EQ(value, ground_truth_values[num_examples]);
      sum += value;
      num_examples++;
    }
  }
  CHECK_OK(reader.Close());
  EXPECT_EQ(num_examples, 22792);
  EXPECT_NEAR(sum / num_examples, data_spec_.columns(0).numerical().mean(),
              0.001);
}

// Check an in order categorical column.
TEST_F(End2End, CategoricalInOrder) {
  EXPECT_EQ(data_spec_.columns(1).categorical().number_of_unique_values(),
            meta_data_.columns(1).categorical().num_values());
  const auto column_spec = data_spec_.columns(1);
  ShardedIntegerColumnReader<int32_t> reader;
  CHECK_OK(reader.Open(
      file::JoinPath(cache_path_, kFilenameRaw,
                     absl::StrCat(kFilenameColumn, 1),  // "workclass" column.
                     kFilenameShardNoUnderscore),
      /*max_value=*/
      meta_data_.columns(1).categorical().num_values(),
      /*max_num_values=*/1000,
      /*begin_shard_idx=*/0,
      /*end_shard_idx=*/meta_data_.num_shards_in_feature_cache()));

  const auto& ground_truth_values =
      dataset_.ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(1)
          ->values();

  size_t num_examples = 0;
  std::unordered_map<int, int> histogram;
  while (true) {
    CHECK_OK(reader.Next());
    const auto values = reader.Values();
    if (values.empty()) {
      break;
    }
    for (const int32_t value : values) {
      // Note: Missing values are replaced in the cache creation.
      if (ground_truth_values[num_examples] != -1) {
        EXPECT_EQ(value, ground_truth_values[num_examples]);
      }
      histogram[value] += 1;
      num_examples++;
    }
  }

  LOG(INFO) << "Histogram:";
  for (auto x : histogram) {
    LOG(INFO) << "\t" << x.first << " : " << x.second;
  }

  CHECK_OK(reader.Close());
  EXPECT_EQ(num_examples, 22792);
  EXPECT_EQ(histogram.size(), meta_data_.columns(1).categorical().num_values());
  EXPECT_EQ(histogram[0], 3);
  EXPECT_EQ(histogram[1], 15879 + 1257 /*missing*/);
  EXPECT_EQ(histogram[2], 1777);
}

}  // namespace
}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
