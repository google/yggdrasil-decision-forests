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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"

#include "gmock/gmock.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace dataset_cache {
namespace {

// Create a sharded record with the pattern: value_i = value_idx * 2.
// Returns the base path.
std::string CreatedShardedInteger(const int num_shards,
                                  const int num_values_per_shards,
                                  const size_t max_value) {
  const auto base_path = file::JoinPath(test::TmpDirectory(), "record");
  int64_t value_idx = 0;
  for (int shard_idx = 0; shard_idx < num_shards; shard_idx++) {
    IntegerColumnWriter writer;
    CHECK_OK(writer.Open(ShardFilename(base_path, shard_idx, num_shards),
                         max_value));
    for (int value_in_shard_idx = 0; value_in_shard_idx < num_values_per_shards;
         value_in_shard_idx++) {
      CHECK_OK(writer.WriteValues<int64_t>(
          absl::Span<const int64_t>{2 * value_idx}));
      value_idx++;
    }
    CHECK_OK(writer.Close());
  }
  return base_path;
}

// Create a sharded record with the pattern: value_i = value_idx * 2 + 0.5.
// Returns the base path.
std::string CreatedShardedFloat(const int num_shards,
                                const int num_values_per_shards) {
  const auto base_path = file::JoinPath(test::TmpDirectory(), "record");
  int64_t value_idx = 0;
  for (int shard_idx = 0; shard_idx < num_shards; shard_idx++) {
    FloatColumnWriter writer;
    CHECK_OK(writer.Open(ShardFilename(base_path, shard_idx, num_shards)));
    for (int value_in_shard_idx = 0; value_in_shard_idx < num_values_per_shards;
         value_in_shard_idx++) {
      CHECK_OK(writer.WriteValues({2.f * value_idx + 0.5f}));
      value_idx++;
    }
    CHECK_OK(writer.Close());
  }
  return base_path;
}

TEST(NumBytes, Base) {
  EXPECT_EQ(NumBytes(0), 1);
  EXPECT_EQ(NumBytes(1), 1);
  EXPECT_EQ(NumBytes(0x7F), 1);
  EXPECT_EQ(NumBytes(0x80), 2);
  EXPECT_EQ(NumBytes(0x7FFF), 2);
  EXPECT_EQ(NumBytes(0x8000), 4);
  EXPECT_EQ(NumBytes(0x7FFFFF), 4);
  EXPECT_EQ(NumBytes(0x800000), 4);
  EXPECT_EQ(NumBytes(0x7FFFFFFF), 4);
  EXPECT_EQ(NumBytes(0x80000000), 8);
}

template <typename WriteValue, typename ReadValue, int64_t max_value>
void TestIntegerColumn() {
  // Create 11 values.
  std::vector<WriteValue> raw_values{
      1, 8, max_value - 1, max_value, -1, -8, -max_value, -max_value - 1, 1,
      2, 3};
  std::vector<WriteValue> write_values(raw_values.begin(), raw_values.end());
  std::vector<ReadValue> read_values(raw_values.begin(), raw_values.end());
  auto write_span = absl::MakeConstSpan(write_values);
  auto read_span = absl::MakeConstSpan(read_values);

  // Write the 11 values.
  const auto path = file::JoinPath(test::TmpDirectory(), "record");
  IntegerColumnWriter writer;
  CHECK_OK(writer.Open(path, max_value));
  CHECK_OK(writer.WriteValues<WriteValue>(write_span.subspan(0, 8)));
  CHECK_OK(writer.WriteValues<WriteValue>(write_span.subspan(8, 3)));
  CHECK_OK(writer.Close());

  // Read the 11 values.
  IntegerColumnReader<ReadValue> reader;
  CHECK_OK(reader.Open(path, max_value, 3));
  CHECK_OK(reader.Next());
  EXPECT_EQ(reader.Values(), read_span.subspan(0, 3));
  CHECK_OK(reader.Next());
  EXPECT_EQ(reader.Values(), read_span.subspan(3, 3));
  CHECK_OK(reader.Next());
  EXPECT_EQ(reader.Values(), read_span.subspan(6, 3));
  CHECK_OK(reader.Next());
  EXPECT_EQ(reader.Values(), read_span.subspan(9, 2));
  CHECK_OK(reader.Next());
  EXPECT_EQ(reader.Values(), read_span.subspan(11, 0));

  CHECK_OK(reader.Close());
}

TEST(IntegerColumn, WriteAndRead) {
  TestIntegerColumn<int8_t, int32_t, 0x7F>();
  TestIntegerColumn<int8_t, int64_t, 0x7F>();

  TestIntegerColumn<int16_t, int32_t, 0x7FFF>();
  TestIntegerColumn<int16_t, int64_t, 0x7FFF>();

  TestIntegerColumn<int32_t, int32_t, 0x7FFFFFFF>();
  TestIntegerColumn<int32_t, int64_t, 0x7FFFFFFF>();

  TestIntegerColumn<int64_t, int64_t, 0x7FFFFFFFFFFFFFFF>();
}

TEST(ShardedIntegerColumnReader, Base) {
  const auto base_path = CreatedShardedInteger(
      /*num_shards=*/5, /*num_values_per_shards=*/10, /*max_value=*/1000);

  ShardedIntegerColumnReader<int32_t> reader;
  CHECK_OK(reader.Open(base_path, /*max_value=*/1000, /*max_num_values=*/2,
                       /*begin_shard_idx=*/0, /*end_shard_idx=*/5));
  for (int i = 0; i < 5 * 10; i += 2) {
    CHECK_OK(reader.Next());
    EXPECT_EQ(reader.Values(), (absl::Span<const int32_t>{2 * i, 2 * i + 2}));
  }
  CHECK_OK(reader.Next());
  EXPECT_TRUE(reader.Values().empty());

  CHECK_OK(reader.Close());
}

TEST(InMemoryIntegerColumnReaderFactory, SameFormat) {
  const size_t max_value = 0x10000000;  // Required 4 bytes per integer.

  const auto base_path = CreatedShardedInteger(
      /*num_shards=*/5, /*num_values_per_shards=*/10, max_value);

  InMemoryIntegerColumnReaderFactory<int32_t> reader_factory;
  CHECK_OK(reader_factory.Load(base_path, max_value, /*max_num_values=*/2,
                               /*begin_shard_idx=*/0, /*end_shard_idx=*/5));

  EXPECT_EQ(reader_factory.MemoryUsage(), 5 * 10 * 4);

  auto reader = reader_factory.CreateIterator();
  for (int i = 0; i < 5 * 10; i += 2) {
    CHECK_OK(reader->Next());
    EXPECT_EQ(reader->Values(), (absl::Span<const int32_t>{2 * i, 2 * i + 2}));
  }
  CHECK_OK(reader->Next());
  EXPECT_TRUE(reader->Values().empty());
}

TEST(InMemoryIntegerColumnReaderFactory, Differentformat) {
  const int max_value = 0x7F;  // Required 1 byte per integer.

  const auto base_path = CreatedShardedInteger(
      /*num_shards=*/5, /*num_values_per_shards=*/10, max_value);

  InMemoryIntegerColumnReaderFactory<int32_t> reader_factory;
  CHECK_OK(reader_factory.Load(base_path, max_value, /*max_num_values=*/2,
                               /*begin_shard_idx=*/0, /*end_shard_idx=*/5));
  EXPECT_EQ(reader_factory.MemoryUsage(), 5 * 10 * 1);

  auto reader = reader_factory.CreateIterator();
  for (int i = 0; i < 5 * 10; i += 2) {
    CHECK_OK(reader->Next());
    EXPECT_EQ(reader->Values(), (absl::Span<const int32_t>{2 * i, 2 * i + 2}));
  }
  CHECK_OK(reader->Next());
  EXPECT_TRUE(reader->Values().empty());
}

TEST(FloatColumn, WriteAndRead) {
  // Write the 5 values.
  const auto path = file::JoinPath(test::TmpDirectory(), "record");
  FloatColumnWriter writer;
  CHECK_OK(writer.Open(path));
  CHECK_OK(writer.WriteValues({1.5f, 2.5f, 3.5f}));
  CHECK_OK(writer.WriteValues({4.5f, 5.5f}));
  CHECK_OK(writer.Close());

  // Read the 5 values.
  FloatColumnReader reader;
  CHECK_OK(reader.Open(path, 3));
  CHECK_OK(reader.Next());
  EXPECT_EQ(reader.Values(), (absl::Span<const float>{1.5f, 2.5f, 3.5f}));
  CHECK_OK(reader.Next());
  EXPECT_EQ(reader.Values(), (absl::Span<const float>{4.5f, 5.5f}));
  CHECK_OK(reader.Next());
  EXPECT_TRUE(reader.Values().empty());

  CHECK_OK(reader.Close());
}

TEST(ShardedFloatColumnReader, Base) {
  const auto base_path = CreatedShardedFloat(
      /*num_shards=*/5, /*num_values_per_shards=*/10);

  ShardedFloatColumnReader reader;
  CHECK_OK(reader.Open(base_path, /*max_num_values=*/2,
                       /*begin_shard_idx=*/0, /*end_shard_idx=*/5));
  for (int i = 0; i < 5 * 10; i += 2) {
    CHECK_OK(reader.Next());
    EXPECT_EQ(reader.Values(),
              (absl::Span<const float>{2 * i + 0.5f, 2.f * i + 2.5f}));
  }
  CHECK_OK(reader.Next());
  EXPECT_TRUE(reader.Values().empty());

  CHECK_OK(reader.Close());
}

TEST(InMemoryFloatColumnReaderFactory, Base) {
  const auto base_path = CreatedShardedFloat(
      /*num_shards=*/5, /*num_values_per_shards=*/10);

  InMemoryFloatColumnReaderFactory reader_factory;
  CHECK_OK(reader_factory.Load(base_path, /*max_num_values=*/2,
                               /*begin_shard_idx=*/0, /*end_shard_idx=*/5));

  EXPECT_EQ(reader_factory.MemoryUsage(), 5 * 10 * 4);

  auto reader = reader_factory.CreateIterator();
  for (int i = 0; i < 5 * 10; i += 2) {
    CHECK_OK(reader->Next());
    EXPECT_EQ(reader->Values(),
              (absl::Span<const float>{2.f * i + 0.5f, 2.f * i + 2.5f}));
  }
  CHECK_OK(reader->Next());
  EXPECT_TRUE(reader->Values().empty());
}

}  // namespace
}  // namespace dataset_cache
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
