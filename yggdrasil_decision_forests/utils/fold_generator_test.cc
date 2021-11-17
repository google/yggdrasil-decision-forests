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

#include <algorithm>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/test.h"

#include "absl/container/btree_set.h"
#include "yggdrasil_decision_forests/utils/fold_generator.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

using test::StatusIs;

std::string LargeDatasetDir() {
  return file::JoinPath(
      test::DataRootDirectory(),
      "yggdrasil_decision_forests/test_data/dataset");
}

class FoldGenerator : public ::testing::Test {
 protected:
  void SetUp() override { LoadAdultDataset(&dataset_); }

  void LoadAdultDataset(dataset::VerticalDataset* dataset) {
    const std::string ds_typed_path =
        absl::StrCat("csv:", file::JoinPath(LargeDatasetDir(), "adult.csv"));
    dataset::proto::DataSpecification data_spec;
    dataset::proto::DataSpecificationGuide guide;
    dataset::CreateDataSpec(ds_typed_path, false, guide, &data_spec);
    CHECK_OK(dataset::LoadVerticalDataset(ds_typed_path, data_spec, dataset));
  }

  void CheckFoldValidity() {
    std::vector<bool> row_index_is_used(dataset_.nrow(), false);
    for (const auto& fold : folds_) {
      // Test if sorted.
      EXPECT_TRUE(std::is_sorted(fold.begin(), fold.end()));
      for (auto row_idx : fold) {
        // Test row idx validity.
        EXPECT_GE(row_idx, 0);
        EXPECT_LT(row_idx, dataset_.nrow());
        // Test uniqueness.
        EXPECT_FALSE(row_index_is_used[row_idx]);
        row_index_is_used[row_idx] = true;
      }
    }
    // Test coverage.
    for (const auto should_be_true : row_index_is_used) {
      EXPECT_TRUE(should_be_true);
    }

    const auto all_tests = ExtractTestExampleIndices(generator_, folds_, true);
    EXPECT_TRUE(std::is_sorted(all_tests.begin(), all_tests.end()));
    EXPECT_TRUE(std::adjacent_find(all_tests.begin(), all_tests.end()) ==
                all_tests.end());

    // Test of grouping
    if (generator_.has_cross_validation() &&
        generator_.cross_validation().has_fold_group()) {
      // Ensure that each group only appear in only one fold.
      const int group_column_idx = dataset_.ColumnNameToColumnIdx(
          generator_.cross_validation().fold_group().group_attribute());
      const auto& group_attribute = dataset_.column(group_column_idx);
      // All the groups.
      absl::btree_set<std::string> groups;
      // Groups present in each fold.
      std::vector<absl::btree_set<std::string>> groups_per_folds(folds_.size());
      for (int fold_idx = 0; fold_idx < folds_.size(); fold_idx++) {
        for (const auto example_idx : folds_[fold_idx]) {
          const std::string group = group_attribute->ToString(
              example_idx, dataset_.data_spec().columns(group_column_idx));
          groups.insert(group);
          groups_per_folds[fold_idx].insert(group);
        }
      }

      for (const std::string& group : groups) {
        int number_of_folds_with_group = 0;
        for (const absl::btree_set<std::string>& groups_in_fold :
             groups_per_folds) {
          if (groups_in_fold.find(group) != groups_in_fold.end()) {
            number_of_folds_with_group++;
          }
        }
        EXPECT_EQ(number_of_folds_with_group, 1);
      }
    }

    // Test the exported csv
    std::vector<std::string> flat_folds = absl::StrSplit(csv_export_, '\n');
    EXPECT_EQ(flat_folds.front(), "fold_idx");
    EXPECT_EQ(flat_folds.back(), "");
    flat_folds.erase(flat_folds.begin());
    flat_folds.erase(flat_folds.end() - 1);
    EXPECT_EQ(flat_folds.size(), dataset_.nrow());
    for (int example_idx = 0; example_idx < flat_folds.size(); example_idx++) {
      int fold_idx;
      EXPECT_TRUE(absl::SimpleAtoi(flat_folds[example_idx], &fold_idx));
      const auto fold = folds_[fold_idx];
      EXPECT_TRUE(std::binary_search(fold.begin(), fold.end(), example_idx));
    }
  }

  utils::StatusOr<std::string> GenerateFolds() {
    RETURN_IF_ERROR(utils::GenerateFolds(generator_, &dataset_, &folds_));
    const std::string folds_path =
        file::JoinPath(test::TmpDirectory(), "folds.csv");
    RETURN_IF_ERROR(ExportFoldsToCsv(folds_, folds_path));
    ASSIGN_OR_RETURN(csv_export_, file::GetContent(folds_path));
    return folds_path;
  }

  dataset::VerticalDataset dataset_;
  proto::FoldGenerator generator_;
  FoldList folds_;
  std::string csv_export_;
};

TEST_F(FoldGenerator, TrainTest) {
  generator_.mutable_train_test()->set_test_ratio(0.3);
  EXPECT_EQ(NumberOfFolds(generator_, folds_), 1);
  EXPECT_OK(GenerateFolds());
  EXPECT_EQ(folds_.size(), 2);
  CheckFoldValidity();
}

TEST_F(FoldGenerator, CrossValidation) {
  generator_.mutable_cross_validation()->set_num_folds(10);
  EXPECT_EQ(NumberOfFolds(generator_, folds_), 10);
  EXPECT_OK(GenerateFolds());
  EXPECT_EQ(folds_.size(), 10);
  CheckFoldValidity();
}

// Fold grouping on the "education" attribute.
TEST_F(FoldGenerator, CrossValidationCategoricalGroup) {
  generator_.mutable_cross_validation()->set_num_folds(10);
  generator_.mutable_cross_validation()
      ->mutable_fold_group()
      ->set_group_attribute("education");
  EXPECT_EQ(NumberOfFolds(generator_, folds_), 10);
  EXPECT_OK(GenerateFolds());
  EXPECT_EQ(folds_.size(), 10);
  CheckFoldValidity();
}

// Incorrect fold grouping on a non existing attribute.
TEST_F(FoldGenerator, CrossValidationCategoricalGroupFail1) {
  generator_.mutable_cross_validation()->set_num_folds(10);
  generator_.mutable_cross_validation()
      ->mutable_fold_group()
      ->set_group_attribute("non_existing");
  EXPECT_THAT(GenerateFolds().status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// Incorrect fold grouping on a non-categorical attribute.
TEST_F(FoldGenerator, CrossValidationCategoricalGroupFail2) {
  generator_.mutable_cross_validation()->set_num_folds(10);
  generator_.mutable_cross_validation()
      ->mutable_fold_group()
      ->set_group_attribute("age");
  EXPECT_THAT(GenerateFolds().status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// Incorrect fold grouping on a categorical attribute without enough unique
// values.
TEST_F(FoldGenerator, CrossValidationCategoricalGroupFail3) {
  generator_.mutable_cross_validation()->set_num_folds(10);
  generator_.mutable_cross_validation()
      ->mutable_fold_group()
      ->set_group_attribute("sex");
  EXPECT_THAT(GenerateFolds().status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(FoldGenerator, TestOnOtherDataset) {
  generator_.mutable_test_on_other_dataset()->set_dataset_path(
      absl::StrCat("csv:", file::JoinPath(LargeDatasetDir(), "adult.csv")));
  EXPECT_EQ(NumberOfFolds(generator_, folds_), 1);
  EXPECT_OK(GenerateFolds());
  EXPECT_EQ(folds_.size(), 2);
  EXPECT_EQ(folds_[0].size(), folds_[1].size());
  CheckFoldValidity();
}

TEST_F(FoldGenerator, NoTraining) {
  generator_.mutable_no_training();
  EXPECT_EQ(NumberOfFolds(generator_, folds_), 1);
  EXPECT_OK(GenerateFolds());
  EXPECT_EQ(folds_.size(), 1);
  EXPECT_EQ(folds_[0].size(), dataset_.nrow());
  CheckFoldValidity();
}

TEST_F(FoldGenerator, PrecomputedCrossValidation) {
  generator_.mutable_cross_validation()->set_num_folds(10);
  EXPECT_EQ(NumberOfFolds(generator_, folds_), 10);
  const auto fold_path = GenerateFolds().value();
  EXPECT_EQ(folds_.size(), 10);
  CheckFoldValidity();
  const auto save_folds = folds_;

  generator_.mutable_precomputed_cross_validation()->set_fold_path(
      absl::StrCat("csv:", fold_path));
  EXPECT_EQ(NumberOfFolds(generator_, folds_), 10);
  EXPECT_OK(GenerateFolds());
  EXPECT_EQ(folds_.size(), 10);
  EXPECT_EQ(folds_, save_folds);
  CheckFoldValidity();
}

TEST(MergeIndicesExceptOneFold, SimpleMerge) {
  const FoldList folds = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
  utils::Fold selected_indices = utils::MergeIndicesExceptOneFold(folds, 1);
  const Fold expected_indices = {0, 1, 2, 6, 7, 8};
  EXPECT_EQ(selected_indices, expected_indices);
}

TEST(MergeItemsInFoldToItems, SimpleMerge) {
  const Fold fold = {3, 2};
  const std::vector<std::string> fold_elements = {"a", "b"};
  std::vector<std::string> elements(5);
  MergeItemsInFoldToItems(fold_elements, fold, &elements);
  const std::vector<std::string> expected_elements = {"", "", "b", "a", ""};
  EXPECT_EQ(elements, expected_elements);
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
