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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_reader.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
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

using test::StatusIs;

class End2End : public ::testing::Test {
 public:
  void Prepare(absl::string_view dataset_name, absl::string_view label_key,
               absl::optional<absl::string_view> group_column) {
    // Prepare the dataspec.
    const auto dataset_path = absl::StrCat(
        "csv:",
        file::JoinPath(
            test::DataRootDirectory(),
            absl::StrCat("yggdrasil_decision_forests/test_data/"
                         "dataset/",
                         dataset_name, "_train.csv")));
    dataset::CreateDataSpec(dataset_path, false, guide_, &data_spec_);
    LOG(INFO) << "Dataspec:" << dataset::PrintHumanReadable(data_spec_);

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

    // The "age" column will not be discretized (72 unique values), while the
    // "education_num" will be (15 unique values).
    config.set_max_unique_values_for_discretized_numerical(32);

    int32_t label_column_idx;
    CHECK_OK(dataset::GetSingleColumnIdxFromName(label_key, data_spec_,
                                                 &label_column_idx));
    config.set_label_column_idx(label_column_idx);

    if (group_column.has_value()) {
      int32_t group_column_idx;
      CHECK_OK(dataset::GetSingleColumnIdxFromName(
          group_column.value(), data_spec_, &group_column_idx));
      config.set_group_column_idx(group_column_idx);
    }

    cache_path_ = file::JoinPath(test::TmpDirectory(),
                                 absl::StrCat(dataset_name, "_cache"));

    EXPECT_OK(CreateDatasetCacheFromShardedFiles(sharded_dataset_path,
                                                 data_spec_, {}, cache_path_,
                                                 config, distribute_config));
  }

  dataset::proto::DataSpecification data_spec_;
  std::string cache_path_;
  dataset::VerticalDataset dataset_;
  dataset::proto::DataSpecificationGuide guide_;
};

TEST_F(End2End, Adult) {
  Prepare("adult", "income", {});

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
    EXPECT_THAT(reader->InOrderNumericalFeatureValueIterator(4).status(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         "Column 4 is not available"));

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

TEST_F(End2End, SyntheticRanking) {
  auto* hash_column = guide_.add_column_guides();
  hash_column->set_column_name_pattern("^GROUP$");
  hash_column->set_type(dataset::proto::ColumnType::HASH);

  auto* cat_column = guide_.add_column_guides();
  cat_column->set_column_name_pattern("^cat_.*");
  cat_column->set_type(dataset::proto::ColumnType::CATEGORICAL);

  Prepare("synthetic_ranking", "LABEL", "GROUP");

  proto::DatasetCacheReaderOptions options;
  options.add_features(6);
  options.add_features(2);
  options.set_load_all_features(false);
  auto reader = DatasetCacheReader::Create(cache_path_, options).value();
  EXPECT_EQ(reader->num_examples(), 3990);
  EXPECT_EQ(reader->ranking_labels().size(), reader->num_examples());
  EXPECT_EQ(reader->ranking_groups().size(), reader->num_examples());

  // The label and hash columns are not loaded in memory.
  EXPECT_THAT(reader->InOrderHashFeatureValueIterator(0).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Column 0 is not available"));
  EXPECT_THAT(reader->InOrderNumericalFeatureValueIterator(1).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Column 1 is not available"));

  LOG(INFO) << reader->MetadataInformation();
}

}  // namespace
}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
