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

#include "yggdrasil_decision_forests/utils/bitmap.h"

#include <stddef.h>

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace bitmap {
namespace {

TEST(Bitmap, GetValueBit) {
  std::string map;
  AllocateAndZeroBitMap(1, &map);
  EXPECT_EQ(GetValueBit(map, 0), false);
  SetValueBit(0, &map);
  EXPECT_EQ(GetValueBit(map, 0), true);
}

TEST(Bitmap, ToStringBit) {
  std::string map;
  const int size = 6;
  AllocateAndZeroBitMap(size, &map);
  EXPECT_EQ(ToStringBit(map, size), "000000");
  SetValueBit(1, &map);
  EXPECT_EQ(ToStringBit(map, size), "010000");
  SetValueBit(5, &map);
  EXPECT_EQ(ToStringBit(map, size), "010001");
  SetValueBit(2, &map);
  EXPECT_EQ(ToStringBit(map, size), "011001");
}

TEST(Bitmap, BitmapToVectorBool) {
  std::string map;
  std::vector<bool> bools;
  const int size = 4;
  AllocateAndZeroBitMap(size, &map);
  BitmapToVectorBool(map, size, &bools);
  EXPECT_EQ(bools, std::vector<bool>({false, false, false, false}));
  SetValueBit(1, &map);
  BitmapToVectorBool(map, size, &bools);
  EXPECT_EQ(bools, std::vector<bool>({false, true, false, false}));
  SetValueBit(2, &map);
  BitmapToVectorBool(map, size, &bools);
  EXPECT_EQ(bools, std::vector<bool>({false, true, true, false}));
}

TEST(Bitmap, SetValueBit) {
  std::string map;
  std::vector<bool> check_map;
  const int size = 40;
  AllocateAndZeroBitMap(size, &map);
  check_map.assign(size, false);
  for (int index : {0, 5, 6, 7, 16, 34}) {
    check_map[index] = true;
    SetValueBit(index, &map);
    for (int check_index = 0; check_index < size; check_index++) {
      EXPECT_EQ(check_map[check_index], GetValueBit(map, check_index));
    }
  }
}

TEST(Bitmap, BitReader) {
  std::string map;
  std::vector<bool> check_map;
  const int size = 40;
  AllocateAndZeroBitMap(size, &map);
  check_map.assign(size, false);
  BitReader reader;
  for (int index : {0, 5, 6, 7, 16, 34}) {
    check_map[index] = true;
    SetValueBit(index, &map);

    reader.Open(map.data(), size);
    for (int check_index = 0; check_index < size; check_index++) {
      EXPECT_EQ(check_map[check_index], reader.Read());
    }
  }
}

TEST(Bitmap, SetGetValueMultiBit) {
  std::string map;
  int bits_by_elements = 3;
  int num_elements = 6;
  AllocateMultibitmap(bits_by_elements, num_elements, &map);
  EXPECT_EQ(GetValueMultibit(map, bits_by_elements, 3), 0);
  SetValueMultibit(3, 6, bits_by_elements, &map);
  EXPECT_EQ(GetValueMultibit(map, bits_by_elements, 3), 6);
}

TEST(Bitmap, SetGetValueMultiBitMax) {
  std::string map;
  int bits_by_elements = MaxNumBits();
  uint64_t max_value = (static_cast<uint64_t>(1) << bits_by_elements) - 1;
  int num_elements = 6;
  AllocateMultibitmap(bits_by_elements, num_elements, &map);
  EXPECT_EQ(GetValueMultibit(map, bits_by_elements, 3), 0);
  SetValueMultibit(3, max_value, bits_by_elements, &map);
  EXPECT_EQ(GetValueMultibit(map, bits_by_elements, 3), max_value);
}

TEST(Bitmap, SetGetValueMultiBitMaxLarge) {
  std::default_random_engine rnd;
  int num_elements = 100;
  for (int bits_by_elements = 0; bits_by_elements < MaxNumBits();
       bits_by_elements++) {
    uint64_t max_value = (static_cast<uint64_t>(1) << bits_by_elements) - 1;
    std::uniform_int_distribution<int64_t> dist(0, max_value);
    std::vector<int64_t> ground_truth(num_elements, 0);
    std::string map;
    AllocateMultibitmap(bits_by_elements, num_elements, &map);
    for (uint64_t idx = 0; idx < num_elements; idx++) {
      ground_truth[idx] = dist(rnd);
      SetValueMultibit(idx, ground_truth[idx], bits_by_elements, &map);
    }
    for (uint64_t idx = 0; idx < num_elements; idx++) {
      EXPECT_EQ(ground_truth[idx],
                GetValueMultibit(map, bits_by_elements, idx));
    }
  }
}

TEST(Bitmap, ToStringMultibit) {
  int bits_by_elements = MaxNumBits();
  int num_elements = 5;

  std::string map;
  AllocateMultibitmap(bits_by_elements, num_elements, &map);
  CHECK_EQ(ToStringMultibit(map, bits_by_elements, num_elements), "0 0 0 0 0");

  uint64_t max_value = (static_cast<uint64_t>(1) << bits_by_elements) - 1;
  SetValueMultibit(3, max_value, bits_by_elements, &map);
  CHECK_EQ(ToStringMultibit(map, bits_by_elements, num_elements),
           absl::StrCat("0 0 0 ", max_value, " 0"));
}

TEST(Bitmap, NextAlignedIndex) {
  EXPECT_EQ(NextAlignedIndex(5, 0), 0);

  EXPECT_EQ(NextAlignedIndex(5, 1), 4);
  EXPECT_EQ(NextAlignedIndex(5, 2), 4);
  EXPECT_EQ(NextAlignedIndex(5, 3), 4);

  EXPECT_EQ(NextAlignedIndex(5, 4), 7);
  EXPECT_EQ(NextAlignedIndex(5, 5), 7);
  EXPECT_EQ(NextAlignedIndex(5, 6), 7);

  EXPECT_EQ(NextAlignedIndex(5, 7), 10);
  EXPECT_EQ(NextAlignedIndex(5, 8), 10);
  EXPECT_EQ(NextAlignedIndex(5, 9), 10);

  EXPECT_EQ(NextAlignedIndex(5, 10), 13);

  EXPECT_EQ(NextAlignedIndex(2, 0), 0);
  EXPECT_EQ(NextAlignedIndex(2, 1), 8);
  EXPECT_EQ(NextAlignedIndex(2, 2), 8);
  EXPECT_EQ(NextAlignedIndex(2, 3), 8);
  EXPECT_EQ(NextAlignedIndex(2, 4), 8);
}

TEST(Bitmap, BitWriter) {
  auto test = [&](int size) {
    std::default_random_engine rnd(size);
    std::string bitmap;
    BitWriter writer(size, &bitmap);
    writer.AllocateAndZeroBitMap();
    std::vector<bool> ground_truth(size, false);
    for (int j = 0; j < size; j++) {
      bool value = (rnd() % 2) == 0;
      ground_truth[j] = value;
      writer.Write(value);
    }
    writer.Finish();
    for (int j = 0; j < size; j++) {
      CHECK_EQ(GetValueBit(bitmap, j), ground_truth[j]);
    }
  };
  for (int size = 0; size < 1024; size++) test(size);
}

TEST(Bitmap, MultibitWriter) {
  auto test = [&](int size, int bits) {
    std::default_random_engine rnd(size + bits * 300);
    std::string bitmap;
    MultibitWriter writer(bits, size, &bitmap);
    writer.AllocateAndZeroBitMap();
    const int64_t max_value = (static_cast<int64_t>(1) << bits) - 1;
    std::uniform_int_distribution<int64_t> dist(0, max_value);
    std::vector<int64_t> ground_truth;
    for (int k = 0; k < size; k++) {
      const int64_t value = dist(rnd);
      writer.Write(value);
      ground_truth.push_back(value);
    }
    writer.Finish();
    for (int j = 0; j < size; j++) {
      const int64_t value = GetValueMultibit(bitmap, bits, j);
      EXPECT_EQ(value, ground_truth[j]);
    }
  };
  for (int size = 0; size < 32; size++) {
    for (int nbits = 0; nbits <= MaxNumBits(); nbits++) {
      test(size, nbits);
    }
  }
}

TEST(Bitmap, PartialMultibitWriter) {
  const int bits_by_elements = 5;
  const int num_elements = 24;
  const int begin_index = 8;
  const int aligned_begin_index =
      NextAlignedIndex(bits_by_elements, begin_index);
  EXPECT_EQ(aligned_begin_index, 10);
  std::string bitmap;
  {
    MultibitWriter writer(bits_by_elements, num_elements, aligned_begin_index,
                          &bitmap);
    writer.AllocateAndZeroBitMap();
    for (int idx = aligned_begin_index; idx < num_elements; idx++) {
      writer.Write(3);
    }
    writer.Finish();
  }

  CHECK_EQ(ToStringMultibit(bitmap, bits_by_elements, num_elements),
           "0 0 0 0 0 0 0 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3");

  SetValueMultibit(2, 6, bits_by_elements, &bitmap);
  CHECK_EQ(ToStringMultibit(bitmap, bits_by_elements, num_elements),
           "0 0 6 0 0 0 0 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3");

  SetValueMultibit(3, 8, bits_by_elements, &bitmap);
  CHECK_EQ(ToStringMultibit(bitmap, bits_by_elements, num_elements),
           "0 0 6 8 0 0 0 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3");

  SetValueMultibit(2, 2, bits_by_elements, &bitmap);
  CHECK_EQ(ToStringMultibit(bitmap, bits_by_elements, num_elements),
           "0 0 2 8 0 0 0 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3");

  for (int idx = 0; idx < aligned_begin_index; idx++) {
    SetValueMultibit(idx, 6, bits_by_elements, &bitmap);
  }
  CHECK_EQ(ToStringMultibit(bitmap, bits_by_elements, num_elements),
           "6 6 6 6 6 6 6 6 6 6 3 3 3 3 3 3 3 3 3 3 3 3 3 3");

  {
    MultibitWriter writer(bits_by_elements, num_elements, num_elements - 2,
                          &bitmap);
    writer.Write(2);
    writer.Write(2);
    writer.Finish();
  }

  CHECK_EQ(ToStringMultibit(bitmap, bits_by_elements, num_elements),
           "6 6 6 6 6 6 6 6 6 6 3 3 3 3 3 3 3 3 3 3 3 3 2 2");

  {
    MultibitWriter writer(bits_by_elements, num_elements, 1, &bitmap);
    writer.Write(3);
    writer.Write(3);
    writer.Write(3);
    writer.Finish();
  }

  CHECK_EQ(ToStringMultibit(bitmap, bits_by_elements, num_elements),
           "6 3 3 3 6 6 6 6 6 6 3 3 3 3 3 3 3 3 3 3 3 3 2 2");
}

void SubPartialWriterMultibitLarge(const int bits_by_elements,
                                   const int64_t num_elements,
                                   const int64_t split_idx,
                                   const int64_t num_write,
                                   const int64_t num_rnd_noise) {
  CHECK_LE(num_write, num_elements);
  int64_t aligned_split_idx = NextAlignedIndex(bits_by_elements, split_idx);
  if (aligned_split_idx >= num_elements) {
    aligned_split_idx = num_elements;
  }

  std::string bitmap;
  AllocateMultibitmap(bits_by_elements, num_elements, &bitmap);
  // Value written after the multi-bitmap buffer to make sure the writing
  // does not goes over the allocated multi-bitmap buffer.
  const char magic_value = 47;
  const auto init_map_size = bitmap.size();
  bitmap.resize(init_map_size + 1);
  bitmap[init_map_size] = magic_value;
  // Contains the same information as the map. Used to check the value of the
  // bitmap.
  std::vector<uint64_t> check_map(num_elements);
  // Check that the "bitmap" and "check_map" are equal, and that the magic
  // number as the end of the allocated buffer is always right.
  auto check = [&](const std::string& step_name) {
    CHECK_EQ(bitmap[init_map_size], magic_value)
        << "step_name:" << step_name << " init_map_size:" << init_map_size
        << " bits_by_elements:" << bits_by_elements
        << " num_elements:" << num_elements << " split_idx:" << split_idx
        << " aligned_split_idx:" << aligned_split_idx
        << " num_write:" << num_write << " num_rnd_noise:" << num_rnd_noise;
    std::string check_string;
    for (size_t idx = 0; idx < check_map.size(); idx++) {
      if (idx > 0) {
        absl::StrAppend(&check_string, " ");
      }
      absl::StrAppend(&check_string, check_map[idx]);
    }
    CHECK_EQ(ToStringMultibit(bitmap, bits_by_elements, num_elements),
             check_string)
        << "step_name:" << step_name << " init_map_size:" << init_map_size
        << " bits_by_elements:" << bits_by_elements
        << " num_elements:" << num_elements << " split_idx:" << split_idx
        << " aligned_split_idx:" << aligned_split_idx
        << " num_write:" << num_write << " num_rnd_noise:" << num_rnd_noise;
  };

  std::default_random_engine rnd;
  std::uniform_int_distribution<int64_t> dist_value(
      0, (static_cast<uint64_t>(1) << bits_by_elements) - 1);
  std::uniform_int_distribution<int64_t> dist_index(0, num_elements - 1);
  check("initial");
  {
    MultibitWriter writer(bits_by_elements, num_elements, aligned_split_idx,
                          &bitmap);
    for (int idx = aligned_split_idx; idx < num_write; idx++) {
      const uint64_t value = dist_value(rnd);
      writer.Write(value);
      check_map[idx] = value;
    }
    writer.Finish();
    check("part 1");
  }
  {
    MultibitWriter writer(bits_by_elements, num_elements, 0, &bitmap);
    for (int idx = 0; idx < split_idx; idx++) {
      uint64_t value = dist_value(rnd);
      writer.Write(value);
      check_map[idx] = value;
    }
    writer.Finish();
    check("part 2");
  }

  {
    for (int idx = split_idx; idx < std::min(aligned_split_idx, num_elements);
         idx++) {
      const uint64_t value = dist_value(rnd);
      SetValueMultibit(idx, value, bits_by_elements, &bitmap);
      check_map[idx] = value;
      check(absl::StrCat("part 3 with idx=", idx));
    }
  }

  {
    MultibitWriter writer(bits_by_elements, num_elements, &bitmap);
    for (int idx = 0; idx < num_elements; idx++) {
      const uint64_t value = dist_value(rnd);
      writer.Write(value);
      check_map[idx] = value;
    }
    writer.Finish();
    check("part 4");
  }
  {
    for (int try_idx = 0; try_idx < num_rnd_noise; try_idx++) {
      const int64_t idx = dist_index(rnd);
      const uint64_t value = dist_value(rnd);
      SetValueMultibit(idx, value, bits_by_elements, &bitmap);
      CHECK_EQ(GetValueMultibit(bitmap, bits_by_elements, idx), value);
      check_map[idx] = value;
      check(absl::StrCat("part 5 with try_idx=", try_idx));
    }
  }
}

TEST(Bitmap, PartialWriterMultibitLarge) {
  std::default_random_engine rnd;
  for (const int bits_by_elements :
       {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 14,
        15, 16, 17, 18, 30, 31, 32, 33, 34, 40, 48}) {
    for (const int64_t num_elements :
         {1, 2, 5, 8, 10, 16, 20, 30, 60, 100, 300}) {
      std::uniform_int_distribution<int64_t> dist_index(0, num_elements - 1);
      for (int split_idx_try = 0; split_idx_try < 100; split_idx_try++) {
        const int64_t split_idx = dist_index(rnd);
        int64_t num_write = dist_index(rnd);
        if (split_idx + num_write > num_elements) {
          num_write = num_elements - split_idx;
        }
        for (const double num_rnd_noise_ratio : {0., 0.2, 0.5, 0.9, 1.}) {
          const int64_t num_rnd_noise =
              static_cast<int64_t>(num_rnd_noise_ratio * num_elements);
          SubPartialWriterMultibitLarge(bits_by_elements, num_elements,
                                        split_idx, num_write, num_rnd_noise);
        }
      }
    }
  }
}

void SubShardedMultiBitmap(const int32_t bits_by_elements,
                           const uint64_t num_elements,
                           const uint64_t num_elements_in_shard) {
  std::default_random_engine rnd;
  uint64_t max_value = (static_cast<uint64_t>(1) << bits_by_elements) - 1;
  std::uniform_int_distribution<uint64_t> dist_value(0, max_value);

  ShardedMultiBitmap sharded_multibitmap;
  sharded_multibitmap.AllocateAndZero(bits_by_elements, num_elements,
                                      num_elements_in_shard, true);
  std::vector<uint64_t> ground_truth(num_elements);
  // Check that "sharded_multibitmap" matches the ground truth.
  auto check = [&](const std::string& step_name) {
    std::string check_string;
    for (uint64_t idx = 0; idx < ground_truth.size(); idx++) {
      if (idx > 0) absl::StrAppend(&check_string, " ");
      absl::StrAppend(&check_string, ground_truth[idx]);
    }
    CHECK_EQ(sharded_multibitmap.ToString(), check_string)
        << "step:" << step_name << " bits_by_elements:" << bits_by_elements
        << " num_elements:" << num_elements;
  };

  check("begin");
  uint64_t num_written = 0;
  for (uint64_t shard_idx = 0; shard_idx < sharded_multibitmap.NumShards();
       shard_idx++) {
    auto writer = sharded_multibitmap.CreateWriter(shard_idx);
    uint64_t begin_idx = sharded_multibitmap.BeginShardIndex(shard_idx);
    uint64_t end_idx = sharded_multibitmap.EndShardIndex(shard_idx);
    for (uint64_t sample_idx = begin_idx; sample_idx < end_idx; sample_idx++) {
      uint64_t value = dist_value(rnd);
      writer->Write(value);
      ground_truth[sample_idx] = value;
      num_written++;
    }
    writer->Finish();
    check("after chunk");
  }
  CHECK_EQ(num_written, num_elements);
  check("end");
}

TEST(Bitmap, ShardedMultiBitmap) {
  for (int32_t bits_by_elements :
       {1, 2, 3, 7, 8, 9, 10, 15, 16, 17, 18, 30, 31, 32, 33}) {
    for (uint64_t num_elements : {0, 1, 2, 3, 4, 10, 20, 50, 100}) {
      for (uint64_t num_elements_in_shard : {2, 3, 4, 20, 40}) {
        SubShardedMultiBitmap(bits_by_elements, num_elements,
                              num_elements_in_shard);
      }
    }
  }
}

TEST(Bitmap, ShardedMultiBitmapSaveAndLoad) {
  const std::string save_base_path =
      file::JoinPath(test::TmpDirectory(), "exported_sharded_multibit");
  ShardedMultiBitmap a;
  a.AllocateAndZero(10, 10, 2, true);
  for (int shard_idx = 0; shard_idx < a.NumShards(); shard_idx++) {
    auto writer = a.CreateWriter(shard_idx);
    uint64_t begin_idx = a.BeginShardIndex(shard_idx);
    uint64_t end_idx = a.EndShardIndex(shard_idx);
    for (uint64_t sample_idx = begin_idx; sample_idx < end_idx; sample_idx++) {
      writer->Write(sample_idx);
    }
    writer->Finish();
  }
  EXPECT_EQ(a.ToString(), "0 1 2 3 4 5 6 7 8 9");

  CHECK_OK(a.SaveToFile(save_base_path, 2));
  ShardedMultiBitmap b;
  CHECK_OK(b.LoadFromFile(save_base_path));
  EXPECT_EQ(a.ToString(), b.ToString());
}

}  // namespace
}  // namespace bitmap
}  // namespace utils
}  // namespace yggdrasil_decision_forests
