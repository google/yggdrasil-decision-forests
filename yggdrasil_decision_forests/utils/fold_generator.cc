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

#include "yggdrasil_decision_forests/utils/fold_generator.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/utils/csv.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/fold_generator.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

absl::Status GenerateFoldsTrainTest(const proto::FoldGenerator& generator,
                                    const dataset::VerticalDataset& dataset,
                                    FoldList* folds) {
  utils::RandomEngine rnd(generator.seed());
  std::uniform_real_distribution<float> dist_01;
  folds->assign(2, Fold());
  auto& testing_fold = (*folds)[0];
  auto& training_fold = (*folds)[1];

  // Assign the examples to either the training or the testing fold.
  for (dataset::VerticalDataset::row_t row_idx = 0; row_idx < dataset.nrow();
       row_idx++) {
    const bool is_test = dist_01(rnd) < generator.train_test().test_ratio();
    if (is_test) {
      testing_fold.push_back(row_idx);
    } else {
      training_fold.push_back(row_idx);
    }
  }
  for (const auto& fold : *folds) {
    if (fold.empty()) {
      return absl::InvalidArgumentError(
          "The dataset does not have enough examples.");
    }
  }
  return absl::OkStatus();
}

absl::Status GenerateFoldsCrossValidationWithoutGroups(
    const proto::FoldGenerator& generator,
    const dataset::VerticalDataset& dataset, FoldList* folds) {
  utils::RandomEngine rnd(generator.seed());
  std::vector<dataset::VerticalDataset::row_t> shuffled_row_indices(
      dataset.nrow());
  std::iota(shuffled_row_indices.begin(), shuffled_row_indices.end(), 0);
  std::shuffle(shuffled_row_indices.begin(), shuffled_row_indices.end(), rnd);
  folds->assign(generator.cross_validation().num_folds(), Fold());
  for (dataset::VerticalDataset::row_t row_idx = 0; row_idx < dataset.nrow();
       row_idx++) {
    const auto fold_idx = shuffled_row_indices[row_idx] %
                          generator.cross_validation().num_folds();
    (*folds)[fold_idx].push_back(row_idx);
  }
  for (const auto& fold : *folds) {
    if (fold.empty()) {
      return absl::InvalidArgumentError(
          "The dataset does not have enough examples.");
    }
  }
  return absl::OkStatus();
}

absl::Status GenerateFoldsCrossValidationWithGroups(
    const proto::FoldGenerator& generator,
    const dataset::VerticalDataset& dataset, FoldList* folds) {
  using Group = std::vector<dataset::VerticalDataset::row_t>;
  // Group the rows per group.
  const int group_column_idx = dataset.ColumnNameToColumnIdx(
      generator.cross_validation().fold_group().group_attribute());

  if (group_column_idx == -1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The fold group \"",
        generator.cross_validation().fold_group().group_attribute(),
        "\" no found."));
  }

  if (dataset.column(group_column_idx)->type() !=
      dataset::proto::ColumnType::CATEGORICAL) {
    return absl::InvalidArgumentError(
        "The fold group attribute is not categorical.");
  }

  const auto* group_attribute =
      dataset.ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
          group_column_idx);
  absl::flat_hash_map<int, Group> rows_per_groups;
  for (dataset::VerticalDataset::row_t row_idx = 0; row_idx < dataset.nrow();
       row_idx++) {
    const int group = group_attribute->values()[row_idx];
    rows_per_groups[group].push_back(row_idx);
  }

  const int num_folds = generator.cross_validation().num_folds();
  if (rows_per_groups.size() < num_folds) {
    return absl::InvalidArgumentError(
        "The number of groups is smaller than the number of folds.");
  }

  // Shuffle the groups.
  // Note: The shuffle is deterministic i.e. running it twice on the same data
  // should produce the same result. std and absl maps have no deterministic
  // iteration (in practice std map are often implemented deterministically).

  std::vector<std::pair<int, Group>> groups;
  groups.reserve(rows_per_groups.size());
  for (auto& rows_in_group : rows_per_groups) {
    groups.push_back(std::move(rows_in_group));
  }
  std::sort(groups.begin(), groups.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  utils::RandomEngine rnd(generator.seed());
  std::shuffle(groups.begin(), groups.end(), rnd);

  // Randomly assign the examples in each group to a fold.
  folds->assign(generator.cross_validation().num_folds(), Fold());
  for (int group_idx = 0; group_idx < groups.size(); group_idx++) {
    const int fold_idx = group_idx % num_folds;
    auto& fold = (*folds)[fold_idx];
    // The fold indices should remain sorted.
    const auto size_before_insertion = fold.size();
    fold.insert(fold.end(), groups[group_idx].second.begin(),
                groups[group_idx].second.end());
    std::inplace_merge(fold.begin(), fold.begin() + size_before_insertion,
                       fold.end());
  }
  return absl::OkStatus();
}

absl::Status GenerateFoldsCrossValidation(
    const proto::FoldGenerator& generator,
    const dataset::VerticalDataset& dataset, FoldList* folds) {
  if (generator.cross_validation().has_fold_group()) {
    return GenerateFoldsCrossValidationWithGroups(generator, dataset, folds);
  } else {
    return GenerateFoldsCrossValidationWithoutGroups(generator, dataset, folds);
  }
}

// Merge the training dataset (already in "dataset") and the test dataset
// specified by the proto.
//
// The test dataset is set to fold 0. The training dataset is set to fold 1.
// The final dataset is composed of the testing dataset examples followed by the
// training dataset examples.
absl::Status GenerateFoldsTestOnOtherDataset(
    const proto::FoldGenerator& generator, dataset::VerticalDataset* dataset,
    FoldList* folds) {
  if (!dataset) {
    return absl::InvalidArgumentError(
        "Non supported fold generation policy TestOnOtherDataset policy "
        "without a "
        "mutable dataset.");
  }

  dataset::VerticalDataset test_dataset;
  RETURN_IF_ERROR(
      LoadVerticalDataset(generator.test_on_other_dataset().dataset_path(),
                          dataset->data_spec(), &test_dataset));

  folds->assign(2, Fold());
  auto& testing_fold = (*folds)[0];
  auto& training_fold = (*folds)[1];

  training_fold.resize(dataset->nrow());
  testing_fold.resize(test_dataset.nrow());

  std::iota(testing_fold.begin(), testing_fold.end(), 0);
  std::iota(training_fold.begin(), training_fold.end(), testing_fold.size());

  RETURN_IF_ERROR(test_dataset.Append(*dataset));
  *dataset = std::move(test_dataset);

  return absl::OkStatus();
}

// Generates folds from a .csv file containing the fold index of each example.
absl::Status GenerateFoldsPrecomputedCrossValidation(
    const proto::FoldGenerator& generator,
    const dataset::VerticalDataset& dataset, FoldList* folds) {
  dataset::proto::DataSpecification fold_dataspec;
  auto* col_spec = dataset::AddColumn(
      "fold_idx", dataset::proto::ColumnType::CATEGORICAL, &fold_dataspec);
  col_spec->mutable_categorical()->set_is_already_integerized(true);
  col_spec->mutable_categorical()->set_number_of_unique_values(
      std::numeric_limits<int>::max());
  dataset::VerticalDataset folds_dataset;
  RETURN_IF_ERROR(
      LoadVerticalDataset(generator.precomputed_cross_validation().fold_path(),
                          fold_dataspec, &folds_dataset));
  const auto fold_values =
      folds_dataset
          .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(0)
          ->values();
  if (fold_values.empty()) {
    return absl::InvalidArgumentError("The set of precomputed folds is empty.");
  }
  int num_folds = *std::max_element(fold_values.begin(), fold_values.end()) + 1;
  if (num_folds < 2) {
    return absl::InvalidArgumentError(
        "At least two folds should be provided for a cross-validation.");
  }
  if (fold_values.size() != dataset.nrow()) {
    return absl::InvalidArgumentError(
        "The number of provided fold indices is different from the number of "
        "examples in the dataset.");
  }
  folds->assign(num_folds, Fold());
  for (dataset::VerticalDataset::row_t example_idx = 0;
       example_idx < fold_values.size(); example_idx++) {
    (*folds)[fold_values[example_idx]].push_back(example_idx);
  }
  return absl::OkStatus();
}

absl::Status GenerateFoldsNoTraining(const proto::FoldGenerator& generator,
                                     const dataset::VerticalDataset& dataset,
                                     FoldList* folds) {
  folds->push_back(Fold());
  auto& testing_fold = (*folds)[0];
  testing_fold.resize(dataset.nrow());
  std::iota(testing_fold.begin(), testing_fold.end(), 0);
  return absl::OkStatus();
}

}  // namespace

int NumberOfFolds(const proto::FoldGenerator& generator,
                  const FoldList& folds) {
  switch (generator.generator_case()) {
    case proto::FoldGenerator::GeneratorCase::kTrainTest:
    case proto::FoldGenerator::GeneratorCase::kTestOnOtherDataset:
    case proto::FoldGenerator::GeneratorCase::kNoTraining:
      return 1;
    case proto::FoldGenerator::GeneratorCase::kCrossValidation:
    // By default, we performs a cross-validation.
    case proto::FoldGenerator::GeneratorCase::GENERATOR_NOT_SET:
      return generator.cross_validation().num_folds();
    case proto::FoldGenerator::GeneratorCase::kPrecomputedCrossValidation:
      return folds.size();
    default:
      LOG(FATAL) << "Unknown fold generator.";
  }
  return -1;
}

int64_t NumberOfTestExamples(const proto::FoldGenerator& generator,
                             const FoldList& folds) {
  const int number_of_folds = NumberOfFolds(generator, folds);
  int64_t number_of_test_examples = 0;
  for (int fold_idx = 0; fold_idx < number_of_folds; fold_idx++) {
    number_of_test_examples += folds[fold_idx].size();
  }
  return number_of_test_examples;
}

absl::Status GenerateFolds(const proto::FoldGenerator& generator,
                           dataset::VerticalDataset* dataset, FoldList* folds) {
  switch (generator.generator_case()) {
    case proto::FoldGenerator::GeneratorCase::kTrainTest:
      return GenerateFoldsTrainTest(generator, *dataset, folds);
    case proto::FoldGenerator::GeneratorCase::kCrossValidation:
      // By default, we performs a cross-validation.
    case proto::FoldGenerator::GeneratorCase::GENERATOR_NOT_SET:
      return GenerateFoldsCrossValidation(generator, *dataset, folds);
    case proto::FoldGenerator::GeneratorCase::kTestOnOtherDataset:
      return GenerateFoldsTestOnOtherDataset(generator, dataset, folds);
    case proto::FoldGenerator::GeneratorCase::kNoTraining:
      return GenerateFoldsNoTraining(generator, *dataset, folds);
    case proto::FoldGenerator::GeneratorCase::kPrecomputedCrossValidation:
      return GenerateFoldsPrecomputedCrossValidation(generator, *dataset,
                                                     folds);
    default:
      LOG(FATAL) << "Not supported fold generator.";
  }
}

absl::Status GenerateFoldsConstDataset(const proto::FoldGenerator& generator,
                                       const dataset::VerticalDataset& dataset,
                                       FoldList* folds) {
  switch (generator.generator_case()) {
    case proto::FoldGenerator::GeneratorCase::kTrainTest:
      return GenerateFoldsTrainTest(generator, dataset, folds);
    case proto::FoldGenerator::GeneratorCase::kCrossValidation:
      // By default, we performs a cross-validation.
    case proto::FoldGenerator::GeneratorCase::GENERATOR_NOT_SET:
      return GenerateFoldsCrossValidation(generator, dataset, folds);
    case proto::FoldGenerator::GeneratorCase::kNoTraining:
      return GenerateFoldsNoTraining(generator, dataset, folds);
    case proto::FoldGenerator::GeneratorCase::kPrecomputedCrossValidation:
      return GenerateFoldsPrecomputedCrossValidation(generator, dataset, folds);
    default:
      LOG(FATAL) << "Not supported fold generator.";
  }
}

Fold MergeIndicesExceptOneFold(const utils::FoldList& folds,
                               const int excluded_fold_idx) {
  CHECK_GE(excluded_fold_idx, 0);
  CHECK_LT(excluded_fold_idx, folds.size());
  Fold dst;
  for (size_t fold_idx = 0; fold_idx < folds.size(); fold_idx++) {
    if (excluded_fold_idx != fold_idx) {
      dst.insert(dst.end(), folds[fold_idx].begin(), folds[fold_idx].end());
    }
  }
  return dst;
}

Fold ExtractTestExampleIndices(const proto::FoldGenerator& generator,
                               const utils::FoldList& folds,
                               const bool ensure_sorted) {
  Fold testing_indices;
  const auto num_folds = utils::NumberOfFolds(generator, folds);
  for (int fold_idx = 0; fold_idx < num_folds; fold_idx++) {
    const auto size_before_insertion = testing_indices.size();
    testing_indices.insert(testing_indices.end(), folds[fold_idx].begin(),
                           folds[fold_idx].end());
    if (ensure_sorted) {
      std::inplace_merge(testing_indices.begin(),
                         testing_indices.begin() + size_before_insertion,
                         testing_indices.end());
    }
  }
  return testing_indices;
}

absl::Status ExportFoldsToCsv(const FoldList& folds, absl::string_view path) {
  ASSIGN_OR_RETURN(auto file_handle, file::OpenOutputFile(path));
  file::OutputFileCloser closer(std::move(file_handle));
  yggdrasil_decision_forests::utils::csv::Writer writer(closer.stream());

  RETURN_IF_ERROR(writer.WriteRow({"fold_idx"}));

  // For each fold, contains the index to the next example index to iterate
  // i.e. folds[j][next_examples[j]] is the next examples to iterate in the
  // fold "j".
  std::vector<size_t> next_examples(folds.size(), 0);

  size_t example_idx = 0;
  while (true) {
    int next_fold_idx = -1;
    for (size_t fold_idx = 0; fold_idx < folds.size(); fold_idx++) {
      if (next_examples[fold_idx] < folds[fold_idx].size() &&
          folds[fold_idx][next_examples[fold_idx]] == example_idx) {
        next_fold_idx = fold_idx;
        break;
      }
    }
    if (next_fold_idx == -1) {
      break;
    }
    RETURN_IF_ERROR(writer.WriteRow({absl::StrCat(next_fold_idx)}));
    next_examples[next_fold_idx]++;
    example_idx++;
  }

  // Ensure that all the examples have been scanned.
  for (size_t fold_idx = 0; fold_idx < folds.size(); fold_idx++) {
    CHECK_EQ(next_examples[fold_idx], folds[fold_idx].size())
        << "The fold where not sorted.";
  }
  return absl::OkStatus();
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
