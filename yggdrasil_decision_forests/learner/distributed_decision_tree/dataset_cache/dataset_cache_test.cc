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
#include "absl/container/flat_hash_map.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_reader.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/multi_thread/multi_thread.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/test_utils.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace dataset_cache {
namespace {

using test::EqualsProto;
using testing::ElementsAre;

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
  absl::flat_hash_map<int, int> histogram;
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

// Check an sorted numerical categorical column.
TEST_F(End2End, SortedNumericalColumn) {
  const auto column_spec = data_spec_.columns(0);

  // List the delta values.
  ShardedFloatColumnReader delta_reader;
  CHECK_OK(delta_reader.Open(file::JoinPath(cache_path_, kFilenameIndexed,
                                            absl::StrCat(kFilenameColumn, 0),
                                            kFilenameDeltaValueNoUnderscore),
                             /*max_num_values=*/1000, 0, 1));

  std::vector<float> delta_values;
  float last_value = column_spec.numerical().min_value() - 1;
  while (true) {
    CHECK_OK(delta_reader.Next());
    const auto values = delta_reader.Values();
    if (values.empty()) {
      break;
    }
    for (const float value : values) {
      // The value are stored in strict increasing order.
      EXPECT_LT(last_value, value);
      delta_values.push_back(value);
      last_value = value;
    }
  }

  // List the example indices.
  ShardedIntegerColumnReader<int64_t> example_idx_reader;
  CHECK_OK(example_idx_reader.Open(
      file::JoinPath(cache_path_, kFilenameIndexed,
                     absl::StrCat(kFilenameColumn, 0),  // "age" column.
                     kFilenameExampleIdxNoUnderscore),
      /*max_value=*/MaxValueWithDeltaBit(meta_data_.num_examples()),
      /*max_num_values=*/1000,
      /*begin_shard_idx=*/0,
      /*end_shard_idx=*/meta_data_.num_shards_in_index_cache()));

  // Grab and sort the values of the feature.
  auto ground_truth_values =
      dataset_.ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(0)
          ->values();
  for (auto& value : ground_truth_values) {
    if (std::isnan(value)) {
      value = column_spec.numerical().mean();
    }
  }
  auto sorted_ground_truth_values = ground_truth_values;
  std::sort(sorted_ground_truth_values.begin(),
            sorted_ground_truth_values.end());

  const auto mask_delta_bit = MaskDeltaBit(meta_data_.num_examples());
  const auto mask_example_idx = MaskExampleIdx(meta_data_.num_examples());

  int delta_bit_idx = 0;
  size_t num_examples = 0;
  while (true) {
    CHECK_OK(example_idx_reader.Next());
    const auto values = example_idx_reader.Values();
    if (values.empty()) {
      break;
    }
    for (const int64_t value : values) {
      const auto example_idx = value & mask_example_idx;
      EXPECT_EQ(ground_truth_values[example_idx],
                sorted_ground_truth_values[num_examples]);
      if (value & mask_delta_bit) {
        delta_bit_idx++;
      }
      EXPECT_EQ(delta_values[delta_bit_idx], ground_truth_values[example_idx]);
      num_examples++;
    }
  }

  CHECK_OK(example_idx_reader.Close());
  CHECK_OK(delta_reader.Close());

  EXPECT_EQ(num_examples, 22792);
  EXPECT_EQ(delta_values.size(), 73);
}

// Check an in order numerical column.
TEST_F(End2End, NumericalInOrderDiscretized) {
  ShardedFloatColumnReader reader;
  CHECK_OK(reader.Open(
      file::JoinPath(
          cache_path_, kFilenameRaw,
          absl::StrCat(kFilenameColumn, 4),  // "education_num" column.
          kFilenameShardNoUnderscore),
      /*max_num_values=*/1000,
      /*begin_shard_idx=*/0,
      /*end_shard_idx=*/meta_data_.num_shards_in_feature_cache()));

  const auto& ground_truth_values =
      dataset_.ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(4)
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
}

class TestCreateDatasetCacheFromPartialDatasetCache : public ::testing::Test {
 public:
  void SetUp() override {
    const auto partial_dataset_cache = CreatePartialCache();

    // Create the dataspec from the partial dataspec.
    dataset::CreateDataSpec(
        absl::StrCat("partial_dataset_cache:", partial_dataset_cache), false,
        {}, &data_spec_);

    // Convert the partial cache into a (final) cache.

    // Multi-threads distribution.
    distribute::proto::Config distribute_config;
    distribute_config.set_implementation_key("MULTI_THREAD");

    proto::CreateDatasetCacheConfig config;

    cache_path_ = file::JoinPath(test::TmpDirectory(),
                                 "cache_from_partial_dataset_cache");
    EXPECT_OK(CreateDatasetCacheFromPartialDatasetCache(
        data_spec_, partial_dataset_cache, cache_path_, config,
        distribute_config, /*delete_source_file=*/true));

    // Try to generate the cache again. Will be instantaneous as the cache is
    // already there.
    EXPECT_OK(CreateDatasetCacheFromPartialDatasetCache(
        data_spec_, partial_dataset_cache, cache_path_, config,
        distribute_config, /*delete_source_file=*/true));

    meta_data_ = LoadCacheMetadata(cache_path_).value();
  }

  void WriteFloatValues(absl::string_view path,
                        absl::Span<const float> values) {
    FloatColumnWriter writer;
    CHECK_OK(writer.Open(path));
    CHECK_OK(writer.WriteValues(values));
    CHECK_OK(writer.Close());
  }

  void WriteInt32Values(absl::string_view path,
                        absl::Span<const int32_t> values) {
    IntegerColumnWriter writer;
    CHECK_OK(writer.Open(path, std::numeric_limits<int32_t>::max()));
    CHECK_OK(writer.WriteValues(values));
    CHECK_OK(writer.Close());
  }

  std::string CreatePartialCache() {
    // Create a partial cache.
    const auto partial_dataset_cache =
        file::JoinPath(test::TmpDirectory(), "partial_dataset_cache");

    CreatePartialCacheFeature0(partial_dataset_cache);
    CreatePartialCacheFeature1(partial_dataset_cache);
    CreatePartialCacheFeature2(partial_dataset_cache);

    proto::PartialDatasetMetadata meta_data;
    meta_data.add_column_names("f0");
    meta_data.add_column_names("f1");
    meta_data.add_column_names("f2");
    meta_data.set_num_shards(2);
    CHECK_OK(file::SetBinaryProto(
        file::JoinPath(partial_dataset_cache, kFilenamePartialMetaData),
        meta_data, file::Defaults()));

    return partial_dataset_cache;
  }

  void CreatePartialCacheFeature0(absl::string_view partial_dataset_cache) {
    const int feature_idx = 0;

    CHECK_OK(file::RecursivelyCreateDir(
        PartialRawColumnFileDirectory(partial_dataset_cache, feature_idx),
        file::Defaults()));

    {
      WriteFloatValues(PartialRawColumnFilePath(partial_dataset_cache,
                                                feature_idx, /*shard_idx=*/0),
                       {1, 5, 2, 4, 3});

      proto::PartialColumnShardMetadata feature_shard_meta_data;
      feature_shard_meta_data.set_num_examples(5);
      feature_shard_meta_data.set_num_missing_examples(0);
      feature_shard_meta_data.mutable_numerical()->set_mean(
          (1. + 5. + 2. + 4. + 3.) / 5);

      CHECK_OK(file::SetBinaryProto(
          absl::StrCat(PartialRawColumnFilePath(partial_dataset_cache,
                                                feature_idx, /*shard_idx=*/0),
                       kFilenameMetaDataPostfix),
          feature_shard_meta_data, file::Defaults()));
    }

    {
      WriteFloatValues(PartialRawColumnFilePath(partial_dataset_cache,
                                                feature_idx, /*shard_idx=*/1),
                       {1, 2, 3, std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN()});

      proto::PartialColumnShardMetadata feature_shard_meta_data;
      feature_shard_meta_data.set_num_examples(5);
      feature_shard_meta_data.set_num_missing_examples(2);
      feature_shard_meta_data.mutable_numerical()->set_mean((1. + 2. + 3.) / 3);

      CHECK_OK(file::SetBinaryProto(
          absl::StrCat(PartialRawColumnFilePath(partial_dataset_cache,
                                                feature_idx, /*shard_idx=*/1),
                       kFilenameMetaDataPostfix),
          feature_shard_meta_data, file::Defaults()));
    }
  }

  void CreatePartialCacheFeature1(absl::string_view partial_dataset_cache) {
    const int feature_idx = 1;

    CHECK_OK(file::RecursivelyCreateDir(
        PartialRawColumnFileDirectory(partial_dataset_cache, feature_idx),
        file::Defaults()));

    {
      WriteInt32Values(
          PartialRawColumnFilePath(partial_dataset_cache, feature_idx,
                                   /*shard_idx=*/0),
          {1, 5, 2, 4, 3});

      proto::PartialColumnShardMetadata feature_shard_meta_data;
      feature_shard_meta_data.set_num_examples(5);
      feature_shard_meta_data.set_num_missing_examples(0);
      feature_shard_meta_data.mutable_categorical()
          ->set_number_of_unique_values(6);

      CHECK_OK(file::SetBinaryProto(
          absl::StrCat(PartialRawColumnFilePath(partial_dataset_cache,
                                                feature_idx, /*shard_idx=*/0),
                       kFilenameMetaDataPostfix),
          feature_shard_meta_data, file::Defaults()));
    }

    {
      WriteInt32Values(
          PartialRawColumnFilePath(partial_dataset_cache, feature_idx,
                                   /*shard_idx=*/1),
          {1, 2, 3, -1, -1});

      proto::PartialColumnShardMetadata feature_shard_meta_data;
      feature_shard_meta_data.set_num_examples(5);
      feature_shard_meta_data.set_num_missing_examples(2);
      feature_shard_meta_data.mutable_categorical()
          ->set_number_of_unique_values(4);

      CHECK_OK(file::SetBinaryProto(
          absl::StrCat(PartialRawColumnFilePath(partial_dataset_cache,
                                                feature_idx, /*shard_idx=*/1),
                       kFilenameMetaDataPostfix),
          feature_shard_meta_data, file::Defaults()));
    }
  }

  void CreatePartialCacheFeature2(absl::string_view partial_dataset_cache) {
    const int feature_idx = 2;

    CHECK_OK(file::RecursivelyCreateDir(
        PartialRawColumnFileDirectory(partial_dataset_cache, feature_idx),
        file::Defaults()));

    {
      WriteInt32Values(
          PartialRawColumnFilePath(partial_dataset_cache, feature_idx,
                                   /*shard_idx=*/0),
          {0, 1, 2, 3, 1});

      proto::PartialColumnShardMetadata feature_shard_meta_data;
      feature_shard_meta_data.set_num_examples(5);
      feature_shard_meta_data.set_num_missing_examples(0);
      auto& items =
          *feature_shard_meta_data.mutable_categorical()->mutable_items();

      {
        auto& item = items["a"];
        item.set_index(0);
        item.set_count(1);  // "a" will be pruned because too infrequent.
      }

      {
        auto& item = items["b"];
        item.set_index(1);
        item.set_count(20);
      }

      {
        auto& item = items["c"];
        item.set_index(2);
        item.set_count(30);
      }

      {
        auto& item = items["d"];
        item.set_index(3);
        item.set_count(40);
      }

      CHECK_OK(file::SetBinaryProto(
          absl::StrCat(PartialRawColumnFilePath(partial_dataset_cache,
                                                feature_idx, /*shard_idx=*/0),
                       kFilenameMetaDataPostfix),
          feature_shard_meta_data, file::Defaults()));
    }

    {
      WriteInt32Values(
          PartialRawColumnFilePath(partial_dataset_cache, feature_idx,
                                   /*shard_idx=*/1),
          {0, 1, 0, -1, -1});

      proto::PartialColumnShardMetadata feature_shard_meta_data;
      feature_shard_meta_data.set_num_examples(5);
      feature_shard_meta_data.set_num_missing_examples(2);

      auto& items =
          *feature_shard_meta_data.mutable_categorical()->mutable_items();

      {
        auto& item = items["c"];
        item.set_index(0);
        item.set_count(50);
      }

      {
        auto& item = items["d"];
        item.set_index(1);
        item.set_count(60);
      }

      CHECK_OK(file::SetBinaryProto(
          absl::StrCat(PartialRawColumnFilePath(partial_dataset_cache,
                                                feature_idx, /*shard_idx=*/1),
                       kFilenameMetaDataPostfix),
          feature_shard_meta_data, file::Defaults()));
    }
  }

  dataset::proto::DataSpecification data_spec_;
  std::string cache_path_;
  proto::CacheMetadata meta_data_;
  dataset::VerticalDataset dataset_;
};

TEST_F(TestCreateDatasetCacheFromPartialDatasetCache, Base) {
  proto::DatasetCacheReaderOptions options;
  auto reader = DatasetCacheReader::Create(cache_path_, options).value();
  EXPECT_EQ(reader->num_examples(), 10);

  const dataset::proto::DataSpecification expected_dataspec = PARSE_TEST_PROTO(
      R"pb(
        columns {
          type: NUMERICAL
          name: "f0"
          numerical {
            mean: 2.625
            min_value: 0
            max_value: 0
            standard_deviation: 0
          }
          count_nas: 2
        }
        columns {
          type: CATEGORICAL
          name: "f1"
          categorical {
            number_of_unique_values: 6
            is_already_integerized: true
          }
          count_nas: 2
        }
        columns {
          type: CATEGORICAL
          name: "f2"
          categorical {
            most_frequent_value: 1
            number_of_unique_values: 4
            is_already_integerized: false
            items {
              key: "<OOD>"
              value { index: 0 count: 1 }
            }
            items {
              key: "d"
              value { index: 1 count: 100 }
            }
            items {
              key: "c"
              value { index: 2 count: 80 }
            }
            items {
              key: "b"
              value { index: 3 count: 20 }
            }
          }
          count_nas: 2
        }
        created_num_rows: 10
      )pb");
  EXPECT_THAT(data_spec_, EqualsProto(expected_dataspec));

  const proto::CacheMetadata expected_meta_data = PARSE_TEST_PROTO(
      R"pb(
        num_examples: 10
        num_shards_in_feature_cache: 2
        num_shards_in_index_cache: 1
        columns {
          available: true
          numerical {
            replacement_missing_value: 2.625  # mean(c(1,2,3,4,5,1,2,3))
            num_unique_values: 5
            discretized: true
            discretized_replacement_missing_value: 2
            num_discretized_shards: 1
            num_discretized_values: 6
          }
        }
        columns {
          available: true
          categorical { num_values: 6 replacement_missing_value: 0 }
        }
        columns {
          available: true
          categorical { num_values: 4 replacement_missing_value: 1 }
        }
      )pb");
  EXPECT_THAT(reader->meta_data(), EqualsProto(expected_meta_data));

  {
    auto iter = reader->InOrderNumericalFeatureValueIterator(0).value();
    CHECK_OK(iter->Next());
    EXPECT_THAT(iter->Values(),
                ElementsAre(1, 5, 2, 4, 3, 1, 2, 3, 2.625, 2.625));
    CHECK_OK(iter->Next());
    EXPECT_TRUE(iter->Values().empty());
    CHECK_OK(iter->Close());
  }

  {
    auto iter = reader->InOrderCategoricalFeatureValueIterator(1).value();
    CHECK_OK(iter->Next());
    EXPECT_THAT(iter->Values(), ElementsAre(1, 5, 2, 4, 3, 1, 2, 3, 0, 0));
    CHECK_OK(iter->Next());
    EXPECT_TRUE(iter->Values().empty());
    CHECK_OK(iter->Close());
  }

  {
    auto iter = reader->InOrderCategoricalFeatureValueIterator(2).value();
    CHECK_OK(iter->Next());
    EXPECT_THAT(iter->Values(), ElementsAre(0, 3, 2, 1, 3, 2, 1, 2, 1, 1));
    CHECK_OK(iter->Next());
    EXPECT_TRUE(iter->Values().empty());
    CHECK_OK(iter->Close());
  }
}

}  // namespace
}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
