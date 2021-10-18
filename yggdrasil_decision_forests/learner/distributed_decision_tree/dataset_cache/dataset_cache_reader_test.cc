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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_reader.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache.h"
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
  }

  dataset::proto::DataSpecification data_spec_;
  std::string cache_path_;
  dataset::VerticalDataset dataset_;
};

TEST_F(End2End, Base) {
  proto::DatasetCacheReaderOptions options;
  auto reader = DatasetCacheReader::Create(cache_path_, options).value();
  EXPECT_EQ(reader->num_examples(), 22792);
  EXPECT_EQ(reader->categorical_labels().size(), reader->num_examples());

  // Sorted numerical
  {
    int count = 0;
    auto col_reader =
        reader->PresortedNumericalFeatureExampleIterator(0).value();
    while (true) {
      CHECK_OK(col_reader->Next());
      const auto values = col_reader->Values();
      if (values.empty()) {
        break;
      }
      count += values.size();
    }
    EXPECT_EQ(count, reader->num_examples());
  }

  // In order numerical
  {
    int count = 0;
    auto col_reader = reader->InOrderNumericalFeatureValueIterator(0).value();
    while (true) {
      CHECK_OK(col_reader->Next());
      const auto values = col_reader->Values();
      if (values.empty()) {
        break;
      }
      count += values.size();
    }
    EXPECT_EQ(count, reader->num_examples());
  }

  // In order categorical
  {
    int count = 0;
    auto col_reader = reader->InOrderCategoricalFeatureValueIterator(1).value();
    while (true) {
      CHECK_OK(col_reader->Next());
      const auto values = col_reader->Values();
      if (values.empty()) {
        break;
      }
      count += values.size();
    }
    EXPECT_EQ(count, reader->num_examples());
  }

  // Discretized, in order, numerical
  {
    int count = 0;
    auto col_reader =
        reader->InOrderDiscretizedNumericalFeatureValueIterator(4).value();
    while (true) {
      CHECK_OK(col_reader->Next());
      const auto values = col_reader->Values();
      if (values.empty()) {
        break;
      }
      count += values.size();
    }
    EXPECT_EQ(count, reader->num_examples());
    EXPECT_EQ(reader->DiscretizedNumericalFeatureBoundaries(4).size(), 15);
  }

  LOG(INFO) << reader->MetadataInformation();
}

}  // namespace
}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
