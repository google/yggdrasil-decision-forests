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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_FOLD_GENERATOR_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_FOLD_GENERATOR_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/fold_generator.pb.h"

namespace yggdrasil_decision_forests {
namespace utils {

// List of row indices defining a fold. Might or might not be sorted.
typedef std::vector<dataset::VerticalDataset::row_t> Fold;

// Definition of a set of folds
using FoldList = std::vector<Fold>;

// Number of evaluation folds.
int NumberOfFolds(const proto::FoldGenerator& generator, const FoldList& folds);

// Number of test examples across all the folds.
int64_t NumberOfTestExamples(const proto::FoldGenerator& generator,
                             const FoldList& folds);

// Returns the indices testing examples for all the folds. The result are
// sorted. This function is deterministic. The dataset can be edited/augmented.
// The "dataset" is both an input and output: Depending on the fold generator,
// the dataset can be edited/augmented.
absl::Status GenerateFolds(const proto::FoldGenerator& generator,
                           dataset::VerticalDataset* dataset, FoldList* folds);

// Similar to "GenerateFolds", but will fail if the fold generator requires for
// the dataset to be augmented.
absl::Status GenerateFoldsConstDataset(const proto::FoldGenerator& generator,
                                       const dataset::VerticalDataset& dataset,
                                       FoldList* folds);

// Merge all the indices except the ones from the fold "fold_idx". The output
// fold is sorted if the input folds are sorted.
utils::Fold MergeIndicesExceptOneFold(const utils::FoldList& folds,
                                      const int excluded_fold_idx);

// Extract all the test example indices from a set of folds.
//
// If "ensure_sorted=false", the output fold is computed from aggregating the
// input folds by increasing fold index. In this case, the output is not
// guaranteed to be sorted.
//
// If "ensure_sorted=true" and if the input folds are sorted, the output fold is
// sorted.
Fold ExtractTestExampleIndices(const proto::FoldGenerator& generator,
                               const utils::FoldList& folds,
                               bool ensure_sorted);

// Exports the folds in a csv file.
//
// The created csv file has a single column, a header ("fold_idx"). The fold
// indices are 0-based indexed (e.g. 10 fold cross-validation will generate fold
// indices in [0, 9]). The i-th fold_idx row correspond to the i-th example.
//
// The folds need to be sorted.
absl::Status ExportFoldsToCsv(const FoldList& folds, absl::string_view path);

// Suppose a fold of "p" examples computed from a list of "n" (n>=p) examples.
// Suppose "items_in_folds" a list of "p" items corresponding to the "p"
// examples of the fold (in the same order), and suppose "items" a list of "n"
// items corresponding to the "n" original examples (also in the same order).
//
// Given a "items_in_folds", "MergeFoldPredictionsIntoGlobalPredictions" assigns
// the subset of values of "items" from the values of "items_in_folds".
//
template <typename Item>
void MergeItemsInFoldToItems(const std::vector<Item>& items_in_folds,
                             const Fold& fold, std::vector<Item>* items) {
  CHECK_EQ(items_in_folds.size(), fold.size());
  for (dataset::VerticalDataset::row_t idx_in_fold = 0;
       idx_in_fold < fold.size(); idx_in_fold++) {
    (*items)[fold[idx_in_fold]] = items_in_folds[idx_in_fold];
  }
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_FOLD_GENERATOR_H_
