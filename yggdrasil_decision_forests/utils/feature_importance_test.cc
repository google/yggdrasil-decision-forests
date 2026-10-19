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

#include "yggdrasil_decision_forests/utils/feature_importance.h"

#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests::utils {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Not;
using ::testing::UnorderedElementsAreArray;

constexpr int kNumCategoricalValues = 100;

// Builds a dataset with "num_columns" columns of type "type" (NUMERICAL or
// CATEGORICAL) and "num_rows" rows filled with random values.
dataset::VerticalDataset BuildSyntheticDataset(
    const int64_t num_rows, const int num_columns,
    const dataset::proto::ColumnType type) {
  dataset::VerticalDataset ds;
  for (int col_idx = 0; col_idx < num_columns; col_idx++) {
    auto* col_spec = dataset::AddColumn(absl::StrCat("f", col_idx), type,
                                        ds.mutable_data_spec());
    if (type == dataset::proto::ColumnType::CATEGORICAL) {
      col_spec->mutable_categorical()->set_number_of_unique_values(
          kNumCategoricalValues);
      col_spec->mutable_categorical()->set_is_already_integerized(true);
    }
  }
  CHECK_OK(ds.CreateColumnsFromDataspec());

  utils::RandomEngine rnd(1234);
  std::uniform_real_distribution<float> num_dist(0.f, 1.f);
  std::uniform_int_distribution<int32_t> cat_dist(1, kNumCategoricalValues - 1);
  for (int col_idx = 0; col_idx < num_columns; col_idx++) {
    switch (type) {
      case dataset::proto::ColumnType::NUMERICAL: {
        auto* col = ds.mutable_numerical_column(col_idx).value();
        col->mutable_values()->reserve(num_rows);
        for (int64_t row = 0; row < num_rows; row++) {
          col->Add(num_dist(rnd));
        }
      } break;
      case dataset::proto::ColumnType::CATEGORICAL: {
        auto* col = ds.mutable_categorical_column(col_idx).value();
        col->mutable_values()->reserve(num_rows);
        for (int64_t row = 0; row < num_rows; row++) {
          col->Add(cat_dist(rnd));
        }
      } break;
      default:
        LOG(FATAL) << "Unsupported column type";
    }
  }
  ds.set_nrow(num_rows);
  return ds;
}

std::vector<int> FirstNColumns(const int n) {
  std::vector<int> idxs(n);
  std::iota(idxs.begin(), idxs.end(), 0);
  return idxs;
}

TEST(ShuffleDatasetColumns, ShuffledColumnIsAPermutation) {
  const auto ds = BuildSyntheticDataset(/*num_rows=*/1000, /*num_columns=*/2,
                                        dataset::proto::ColumnType::NUMERICAL);
  utils::RandomEngine rnd(1);
  ASSERT_OK_AND_ASSIGN(const auto shuffled,
                       ShuffleDatasetColumns(ds, {0}, &rnd));
  ASSERT_EQ(shuffled.nrow(), ds.nrow());

  const auto& src_0 = ds.numerical_column(0).value()->values();
  const auto& dst_0 = shuffled.numerical_column(0).value()->values();
  EXPECT_THAT(dst_0, UnorderedElementsAreArray(src_0));
  EXPECT_THAT(dst_0, Not(ElementsAreArray(src_0)));

  // The non-shuffled column is untouched.
  EXPECT_THAT(shuffled.numerical_column(1).value()->values(),
              ElementsAreArray(ds.numerical_column(1).value()->values()));
}

TEST(ShuffleDatasetColumns, ShuffledCategoricalColumnIsAPermutation) {
  const auto ds =
      BuildSyntheticDataset(/*num_rows=*/1000, /*num_columns=*/1,
                            dataset::proto::ColumnType::CATEGORICAL);
  utils::RandomEngine rnd(1);
  ASSERT_OK_AND_ASSIGN(const auto shuffled,
                       ShuffleDatasetColumns(ds, {0}, &rnd));
  EXPECT_THAT(
      shuffled.categorical_column(0).value()->values(),
      UnorderedElementsAreArray(ds.categorical_column(0).value()->values()));
}

TEST(ShuffleDatasetColumns, OtherColumnTypesWithMissingValues) {
  using VerticalDataset = dataset::VerticalDataset;
  constexpr int kNumRows = 200;
  VerticalDataset ds;
  dataset::AddColumn("b", dataset::proto::BOOLEAN, ds.mutable_data_spec());
  dataset::AddColumn("d", dataset::proto::DISCRETIZED_NUMERICAL,
                     ds.mutable_data_spec());
  dataset::AddColumn("h", dataset::proto::HASH, ds.mutable_data_spec());
  // String columns use the generic (slow) code path.
  dataset::AddColumn("s", dataset::proto::STRING, ds.mutable_data_spec());
  ASSERT_OK(ds.CreateColumnsFromDataspec());
  for (int row = 0; row < kNumRows; row++) {
    if (row % 10 == 0) {
      for (int col = 0; col < 4; col++) {
        ds.mutable_column(col)->AddNA();
      }
      continue;
    }
    ds.MutableColumnWithCast<VerticalDataset::BooleanColumn>(0)->Add(row % 2);
    ds.MutableColumnWithCast<VerticalDataset::DiscretizedNumericalColumn>(1)
        ->Add(row % 7);
    ds.MutableColumnWithCast<VerticalDataset::HashColumn>(2)->Add(row + 100);
    ds.MutableColumnWithCast<VerticalDataset::StringColumn>(3)->Add(
        absl::StrCat("s", row));
  }
  ds.set_nrow(kNumRows);

  utils::RandomEngine rnd(5);
  ASSERT_OK_AND_ASSIGN(const auto shuffled,
                       ShuffleDatasetColumns(ds, {0, 1, 2, 3}, &rnd));

  EXPECT_THAT(
      shuffled.ColumnWithCast<VerticalDataset::BooleanColumn>(0)->values(),
      UnorderedElementsAreArray(
          ds.ColumnWithCast<VerticalDataset::BooleanColumn>(0)->values()));
  EXPECT_THAT(
      shuffled.ColumnWithCast<VerticalDataset::DiscretizedNumericalColumn>(1)
          ->values(),
      UnorderedElementsAreArray(
          ds.ColumnWithCast<VerticalDataset::DiscretizedNumericalColumn>(1)
              ->values()));
  const auto& src_hashes =
      ds.ColumnWithCast<VerticalDataset::HashColumn>(2)->values();
  const auto& dst_hashes =
      shuffled.ColumnWithCast<VerticalDataset::HashColumn>(2)->values();
  EXPECT_THAT(dst_hashes, UnorderedElementsAreArray(src_hashes));
  EXPECT_THAT(dst_hashes, Not(ElementsAreArray(src_hashes)));

  // Strings, with "<NA>" for missing values.
  const auto string_values = [](const VerticalDataset& d) {
    std::vector<std::string> values;
    const auto* column = d.ColumnWithCast<VerticalDataset::StringColumn>(3);
    for (int row = 0; row < d.nrow(); row++) {
      values.push_back(column->IsNa(row) ? "<NA>" : column->values()[row]);
    }
    return values;
  };
  const auto src_strings = string_values(ds);
  const auto dst_strings = string_values(shuffled);
  EXPECT_THAT(dst_strings, UnorderedElementsAreArray(src_strings));
  EXPECT_THAT(dst_strings, Not(ElementsAreArray(src_strings)));
}

TEST(ShuffleDatasetColumns, TinyDatasets) {
  for (const int num_rows : {0, 1}) {
    const auto ds = BuildSyntheticDataset(
        num_rows, /*num_columns=*/1, dataset::proto::ColumnType::NUMERICAL);
    utils::RandomEngine rnd(3);
    ASSERT_OK_AND_ASSIGN(const auto shuffled,
                         ShuffleDatasetColumns(ds, {0}, &rnd));
    EXPECT_THAT(shuffled.numerical_column(0).value()->values(),
                ElementsAreArray(ds.numerical_column(0).value()->values()));
  }
}

TEST(ShuffleDatasetColumns, AllPermutationsAreEquallyLikely) {
  // A dataset with 4 rows containing the values 0, 1, 2, 3.
  constexpr int kNumRows = 4;
  constexpr int kNumPermutations = 24;  // 4!
  constexpr int kNumTrials = 1000 * kNumPermutations;
  dataset::VerticalDataset ds;
  dataset::AddColumn("f", dataset::proto::ColumnType::NUMERICAL,
                     ds.mutable_data_spec());
  ASSERT_OK(ds.CreateColumnsFromDataspec());
  for (int row = 0; row < kNumRows; row++) {
    ds.mutable_numerical_column(0).value()->Add(row);
  }
  ds.set_nrow(kNumRows);

  utils::RandomEngine rnd(4);
  absl::flat_hash_map<int, int> permutation_counts;
  for (int trial = 0; trial < kNumTrials; trial++) {
    ASSERT_OK_AND_ASSIGN(const auto shuffled,
                         ShuffleDatasetColumns(ds, {0}, &rnd));
    int key = 0;
    for (const float value : shuffled.numerical_column(0).value()->values()) {
      key = key * kNumRows + static_cast<int>(value);
    }
    permutation_counts[key]++;
  }

  // Each permutation is expected 1000 times (std ~31).
  EXPECT_EQ(permutation_counts.size(), kNumPermutations);
  for (const auto& [key, count] : permutation_counts) {
    EXPECT_GT(count, 850) << "key=" << key;
    EXPECT_LT(count, 1150) << "key=" << key;
  }
}

}  // namespace
}  // namespace yggdrasil_decision_forests::utils
