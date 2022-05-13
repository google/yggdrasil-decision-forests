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

#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <regex>  // NOLINT
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/formats.h"
#include "yggdrasil_decision_forests/dataset/formats.pb.h"
#include "yggdrasil_decision_forests/utils/accurate_sum.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace dataset {

using proto::ColumnType;
using proto::DatasetFormat;
using proto::Tokenizer;
using yggdrasil_decision_forests::utils::AccurateSum;

void FillContentNumericalFeature(
    const float num_value,
    proto::DataSpecificationAccumulator::Column* col_acc) {
  // Mean
  AccurateSum kahanAcc(col_acc->kahan_sum(), col_acc->kahan_sum_error());
  kahanAcc.Add(num_value);
  col_acc->set_kahan_sum(kahanAcc.Sum());
  col_acc->set_kahan_sum_error(kahanAcc.ErrorSum());

  // Sd
  AccurateSum kahanSquareAcc(col_acc->kahan_sum_of_square(),
                             col_acc->kahan_sum_of_square_error());
  kahanSquareAcc.Add(num_value * num_value);
  col_acc->set_kahan_sum_of_square(kahanSquareAcc.Sum());
  col_acc->set_kahan_sum_of_square_error(kahanSquareAcc.ErrorSum());

  if (!col_acc->has_min_value() || num_value < col_acc->min_value()) {
    col_acc->set_min_value(num_value);
  }
  if (!col_acc->has_max_value() || num_value > col_acc->max_value()) {
    col_acc->set_max_value(num_value);
  }
}

void InitializeDataSpecFromColumnNames(
    const proto::DataSpecificationGuide& guide,
    const std::vector<std::string>& header, proto::DataSpecification* data_spec,
    std::vector<std::pair<int, proto::ColumnGuide>>*
        spec_col_idx_2_csv_col_idx) {
  spec_col_idx_2_csv_col_idx->clear();
  data_spec->clear_columns();
  for (int head_col_idx = 0; head_col_idx < header.size(); head_col_idx++) {
    const auto& col_name = header[head_col_idx];
    proto::ColumnGuide col_guide;
    const bool has_user_col_guide =
        BuildColumnGuide(col_name, guide, &col_guide);
    if (!has_user_col_guide && guide.ignore_columns_without_guides()) {
      continue;
    }
    if (col_guide.ignore_column()) {
      continue;
    }
    proto::Column* column = data_spec->add_columns();
    column->set_name(col_name);
    spec_col_idx_2_csv_col_idx->push_back(
        std::make_pair(head_col_idx, col_guide));
    if (has_user_col_guide && col_guide.has_type()) {
      column->set_is_manual_type(true);
      column->set_type(col_guide.type());
    } else {
      column->set_is_manual_type(false);
    }
  }
}

bool LooksMultiDimensional(const absl::string_view value,
                           const Tokenizer& tokenizer) {
  std::vector<std::string> tokens;
  Tokenize(value, tokenizer, &tokens);
  return tokens.size() >= 2;
}

absl::Status UpdateColSpecsWithGuideInfo(
    const std::vector<std::pair<int, proto::ColumnGuide>>&
        spec_col_idx_2_csv_col_idx,
    proto::DataSpecification* data_spec) {
  for (int col_idx = 0; col_idx < data_spec->columns_size(); col_idx++) {
    proto::Column* col = data_spec->mutable_columns(col_idx);
    const auto& col_guide = spec_col_idx_2_csv_col_idx[col_idx].second;
    RETURN_IF_ERROR(UpdateSingleColSpecWithGuideInfo(col_guide, col));
  }
  return absl::OkStatus();
}

void AddTokensToCategoricalColumnSpec(const std::vector<std::string>& tokens,
                                      proto::Column* col) {
  if (col->categorical().is_already_integerized()) {
    // The tokens are already numbers (stored as strings).
    for (const std::string& token : tokens) {
      int32_t int_value;
      CHECK(absl::SimpleAtoi(token, &int_value));
      CHECK_GE(int_value, 0)
          << "Already integerized categories should be positive (non strict).";
      if (int_value >= col->categorical().number_of_unique_values()) {
        col->mutable_categorical()->set_number_of_unique_values(int_value + 1);
      }
    }
  } else {
    // Update the dictionary mapping.
    for (const std::string& token : tokens) {
      auto* items = col->mutable_categorical()->mutable_items();
      auto& item = (*items)[token];
      item.set_count(item.count() + 1);
    }
  }
}

void UpdateComputeSpecDiscretizedNumerical(
    const float value, proto::Column* column,
    proto::DataSpecificationAccumulator::Column* accumulator) {
  if (!std::isnan(value)) {
    // TODO(gbm): Use absl::bit_cast.
    const uint32_t int_value = *reinterpret_cast<const uint32_t*>(&value);
    (*accumulator->mutable_discretized_numerical())[int_value]++;
  }
}

void UpdateComputeSpecBooleanFeature(float value, proto::Column* column) {
  if (value >= 0.5f) {
    column->mutable_boolean()->set_count_true(column->boolean().count_true() +
                                              1);
  } else {
    column->mutable_boolean()->set_count_false(column->boolean().count_false() +
                                               1);
  }
}

void FinalizeComputeSpecDiscretizedNumerical(
    const proto::DataSpecificationAccumulator::Column& accumulator,
    proto::Column* column) {
  std::vector<std::pair<float, int>> unique_values_and_counts;
  unique_values_and_counts.reserve(accumulator.discretized_numerical_size());
  for (const auto& item : accumulator.discretized_numerical()) {
    unique_values_and_counts.emplace_back(
        *reinterpret_cast<const float*>(&item.first), item.second);
  }
  std::sort(unique_values_and_counts.begin(), unique_values_and_counts.end());

  const auto bounds = GenDiscretizedBoundaries(
      unique_values_and_counts,
      column->discretized_numerical().maximum_num_bins(),
      column->discretized_numerical().min_obs_in_bins(),
      {0.f, static_cast<float>(column->numerical().mean())});

  column->mutable_discretized_numerical()->set_original_num_unique_values(
      unique_values_and_counts.size());

  *column->mutable_discretized_numerical()->mutable_boundaries() = {
      bounds.begin(), bounds.end()};
}

// Finalize the information of a numerical column.
void FinalizeComputeSpecColumnNumerical(
    const uint64_t count_valid_records,
    const proto::DataSpecificationAccumulator::Column& col_acc,
    proto::Column* col) {
  if (count_valid_records >= 0) {
    AccurateSum kahanAcc(col_acc.kahan_sum(), col_acc.kahan_sum_error());
    const double mean = kahanAcc.Sum() / count_valid_records;
    col->mutable_numerical()->set_mean(mean);

    AccurateSum kahanSquareAcc(col_acc.kahan_sum_of_square(),
                               col_acc.kahan_sum_of_square_error());
    double var = kahanSquareAcc.Sum() / count_valid_records - mean * mean;
    if (var < 0) {
      // Possible with rounding error.
      var = 0;
    }
    col->mutable_numerical()->set_standard_deviation(std::sqrt(var));

    col->mutable_numerical()->set_min_value(col_acc.min_value());
    col->mutable_numerical()->set_max_value(col_acc.max_value());
  }
}

// Converted the dictionary (stored as a map) into a vector ordered in
// decreasing order of item frequency. The "OutOfDictionary" item is skipped.
void DictionaryMapToSortedDictionaryVector(
    const proto::Column& col,
    std::vector<std::pair<uint64_t, std::string>>* item_frequency_vector,
    uint64_t* count_ood_items) {
  (*count_ood_items) = 0;
  const auto& items = col.categorical().items();
  item_frequency_vector->reserve(items.size());
  for (auto& item : items) {
    if (item.first == kOutOfDictionaryItemKey) {
      (*count_ood_items) = item.second.count();
    } else {
      item_frequency_vector->emplace_back(item.second.count(), item.first);
    }
  }
  std::sort(item_frequency_vector->begin(), item_frequency_vector->end(),
            std::greater<std::pair<uint64_t, std::string>>());
}

// Converted the dictionary (stored as a vector ordered in decreasing order of
// item frequency) into a map. This is the reverse of
// "DictionaryMapToSortedDictionaryVector".
void SortedDictionaryVectorToDictionaryMap(
    const std::vector<std::pair<uint64_t, std::string>>& item_frequency_vector,
    proto::Column* col) {
  auto* items = col->mutable_categorical()->mutable_items();
  items->clear();
  for (int item_idx = 0; item_idx < item_frequency_vector.size(); item_idx++) {
    auto& item = (*items)[item_frequency_vector[item_idx].second];
    item.set_count(item_frequency_vector[item_idx].first);
    item.set_index(item_idx + 1);
  }
}

// Finalize the information about a categorical column i.e. finalize the
// token dictionary.
void FinalizeComputeSpecColumnCategorical(
    const uint64_t count_valid_records,
    const proto::DataSpecificationAccumulator::Column& col_acc,
    proto::Column* col) {
  if (col->categorical().is_already_integerized()) {
    // Nothing to finalize for already integerized categorical columns.
    return;
  }
  // Compute the dictionary item indices so the item frequency decrease with
  // the index value.
  std::vector<std::pair<uint64_t, std::string>> item_frequency_vector;
  uint64_t count_ood_items;
  DictionaryMapToSortedDictionaryVector(*col, &item_frequency_vector,
                                        &count_ood_items);
  const uint64_t non_pruned_number_of_unique_values =
      item_frequency_vector.size();

  // Remove items with frequency (i.e. count) below the user defined
  // threshold.
  if (col->categorical().min_value_count() > 0) {
    while (!item_frequency_vector.empty() &&
           item_frequency_vector.back().first <
               col->categorical().min_value_count()) {
      count_ood_items++;
      item_frequency_vector.pop_back();
    }
  }

  // If the number of unique items is higher than the maximum threshold
  // "max_vocab_count", we remove the less frequent items.
  if (col->categorical().max_number_of_unique_values() > 0 &&
      item_frequency_vector.size() >
          col->categorical().max_number_of_unique_values()) {
    item_frequency_vector.resize(
        col->categorical().max_number_of_unique_values());
  }

  // Information message if items have been pruned.
  const uint64_t count_pruned_items =
      non_pruned_number_of_unique_values - item_frequency_vector.size();
  if (count_pruned_items > 0) {
    LOG(INFO) << count_pruned_items
              << " item(s) have been pruned (i.e. they are considered "
                 "out of dictionary) for the column "
              << col->name() << " (" << item_frequency_vector.size()
              << " item(s) left) because min_value_count="
              << col->categorical().min_value_count()
              << " and max_number_of_unique_values="
              << col->categorical().max_number_of_unique_values();
  }

  // Update the dictionary map.
  SortedDictionaryVectorToDictionaryMap(item_frequency_vector, col);

  // Special OutOfDictionary item.
  auto* items = col->mutable_categorical()->mutable_items();
  auto& ood_item = (*items)[kOutOfDictionaryItemKey];
  ood_item.set_count(count_ood_items);
  ood_item.set_index(kOutOfDictionaryItemIndex);

  // Most frequent item.
  if (item_frequency_vector.empty() ||
      ood_item.count() > item_frequency_vector.front().first) {
    col->mutable_categorical()->set_most_frequent_value(
        kOutOfDictionaryItemIndex);
  } else {
    // Note: "1" because indices are attributes by decreasing frequency
    // (starting from 1).
    col->mutable_categorical()->set_most_frequent_value(1);
  }

  // Number of unique item.
  col->mutable_categorical()->set_number_of_unique_values(items->size());
}

void InitializeDataspecAccumulator(
    const proto::DataSpecification& data_spec,
    proto::DataSpecificationAccumulator* accumulator) {
  accumulator->mutable_columns()->Reserve(data_spec.columns_size());
  for (int col_idx = 0; col_idx < data_spec.columns_size(); col_idx++) {
    accumulator->mutable_columns()->Add();
  }
}

void MergeColumnGuide(const proto::ColumnGuide& src, proto::ColumnGuide* dst) {
  dst->MergeFrom(src);
}

void CreateDataSpec(const absl::string_view typed_path, const bool use_flume,
                    const proto::DataSpecificationGuide& guide,
                    proto::DataSpecification* data_spec) {
  if (use_flume) {
    LOG(FATAL) << "Dataspec inference with flume is not implemented";
  }

  // Format of the dataset.
  std::string sharded_path;
  proto::DatasetFormat format;
  std::tie(sharded_path, format) = GetDatasetPathAndType(typed_path);

  // Files in the dataset.
  std::vector<std::string> paths;
  CHECK_OK(utils::ExpandInputShards(sharded_path, &paths));

  // Create the dataspec creator.
  const auto& format_name = proto::DatasetFormat_Name(format);
  auto creator = AbstractDataSpecCreatorRegisterer::Create(format_name).value();

  // Detect the column names and semantics.
  creator->InferColumnsAndTypes(paths, guide, data_spec);
  FinalizeInferTypes(guide, data_spec);
  LOG(INFO) << data_spec->columns_size() << " column(s) found";

  // Computes the statistics (e.g. dictionaries, ratio of missing values) for
  // each column.
  proto::DataSpecificationAccumulator accumulator;
  InitializeDataspecAccumulator(*data_spec, &accumulator);
  // TODO(gbm): Call ComputeColumnStatistics in parallel other the different
  // paths.
  creator->ComputeColumnStatistics(paths, guide, data_spec, &accumulator);
  FinalizeComputeSpec(guide, accumulator, data_spec);

  LOG(INFO) << "Finalizing [" << data_spec->created_num_rows()
            << " row(s) found]";
}

void FinalizeInferTypes(const proto::DataSpecificationGuide& guide,
                        proto::DataSpecification* data_spec) {
  // Columns to remove at the end of the finalization.
  absl::flat_hash_set<std::string> columns_to_remove;

  // Name of the first unstacked column for each "unstacked" items in
  // "data_spec".
  std::vector<std::string> unstack_begin_column_name;

  if (guide.unstack_numerical_set_as_numericals()) {
    // Unstack the numerical set features.
    for (int column_idx = 0; column_idx < data_spec->columns_size();
         column_idx++) {
      auto& column = *data_spec->mutable_columns(column_idx);

      // Logic:
      //   - A NUMERICAL_SET column with a fix number of values is converted
      //     into an unstackeds (i.e. multi-dimensional) column.
      //   - A NUMERICAL_SET column with a variable number of values is
      //     converted into a CATEGORICAL_SET column.

      if (column.type() != proto::NUMERICAL_SET) {
        continue;
      }

      if (column.multi_values().max_observed_size() !=
          column.multi_values().min_observed_size()) {
        column.set_type(proto::CATEGORICAL_SET);
        column.mutable_categorical()->set_is_already_integerized(true);
        continue;
      }

      // The original feature will be removed.
      columns_to_remove.insert(column.name());

      // Unstacking information.
      const auto type = guide.detect_numerical_as_discretized_numerical()
                            ? proto::DISCRETIZED_NUMERICAL
                            : proto::NUMERICAL;
      CHECK_GT(column.multi_values().max_observed_size(), 0);
      unstack_begin_column_name.push_back(
          UnstackedColumnName(column.name(), 0));
      auto* unstacked = data_spec->add_unstackeds();
      unstacked->set_original_name(column.name());
      unstacked->set_size(column.multi_values().max_observed_size());
      unstacked->set_type(type);

      // Create the unstacked features.
      for (int dim_idx = 0; dim_idx < column.multi_values().max_observed_size();
           dim_idx++) {
        auto* sub_col = data_spec->add_columns();
        sub_col->set_name(UnstackedColumnName(column.name(), dim_idx));
        sub_col->set_type(type);
        sub_col->set_is_unstacked(true);
      }
    }
  }

  // Remove the unstacked and unknown columns.
  auto saved_columns = std::move(*data_spec->mutable_columns());
  data_spec->mutable_columns()->Clear();
  for (auto& column : saved_columns) {
    if (columns_to_remove.find(column.name()) != columns_to_remove.end()) {
      continue;
    }
    if (guide.ignore_unknown_type_columns() &&
        column.type() == proto::ColumnType::UNKNOWN) {
      continue;
    }
    data_spec->mutable_columns()->Add(std::move(column));
  }

  // Clear multi-values meta-data.
  for (auto& column : *data_spec->mutable_columns()) {
    column.clear_multi_values();
  }

  // After this point, the feature indices cannot be changed anymore.

  // Resolve the "begin_column_idx" in the unstacked metadata.
  for (int unstack_idx = 0; unstack_idx < unstack_begin_column_name.size();
       unstack_idx++) {
    const int column_idx = GetColumnIdxFromName(
        unstack_begin_column_name[unstack_idx], *data_spec);
    data_spec->mutable_unstackeds(unstack_idx)
        ->set_begin_column_idx(column_idx);
  }
}

void FinalizeComputeSpec(const proto::DataSpecificationGuide& guide,
                         const proto::DataSpecificationAccumulator& accumulator,
                         proto::DataSpecification* data_spec) {
  for (int col_idx = 0; col_idx < data_spec->columns_size(); col_idx++) {
    auto* col = data_spec->mutable_columns(col_idx);
    const auto& col_acc = accumulator.columns(col_idx);
    // Valid records i.e. non NA records.
    const uint64_t count_valid_records =
        data_spec->created_num_rows() - col->count_nas();
    // Numerical type.
    if (IsNumerical(col->type())) {
      FinalizeComputeSpecColumnNumerical(count_valid_records, col_acc, col);
    }
    // Categorical type.
    if (IsCategorical(col->type())) {
      FinalizeComputeSpecColumnCategorical(count_valid_records, col_acc, col);
    }
    if (col->type() == ColumnType::DISCRETIZED_NUMERICAL) {
      FinalizeComputeSpecDiscretizedNumerical(col_acc, col);
    }
  }
}

absl::Status UpdateNumericalColumnSpec(
    const float num_value, proto::Column* col,
    proto::DataSpecificationAccumulator::Column* col_acc) {
  if (std::isnan(num_value)) {
    col->set_count_nas(col->count_nas() + 1);
  } else {
    if (std::isinf(num_value)) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Found infinite value for numerical feature $0", col->name()));
    }

    // Mean
    AccurateSum kahanAcc(col_acc->kahan_sum(), col_acc->kahan_sum_error());
    kahanAcc.Add(num_value);
    col_acc->set_kahan_sum(kahanAcc.Sum());
    col_acc->set_kahan_sum_error(kahanAcc.ErrorSum());

    // SD
    AccurateSum kahanSquareAcc(col_acc->kahan_sum_of_square(),
                               col_acc->kahan_sum_of_square_error());
    kahanSquareAcc.Add(num_value * num_value);
    col_acc->set_kahan_sum_of_square(kahanSquareAcc.Sum());
    col_acc->set_kahan_sum_of_square_error(kahanSquareAcc.ErrorSum());

    if (!col_acc->has_min_value() || num_value < col_acc->min_value()) {
      col_acc->set_min_value(num_value);
    }
    if (!col_acc->has_max_value() || num_value > col_acc->max_value()) {
      col_acc->set_max_value(num_value);
    }
  }
  return absl::OkStatus();
}

absl::Status UpdateCategoricalStringColumnSpec(
    const std::string& str_value, proto::Column* col,
    proto::DataSpecificationAccumulator::Column* col_acc) {
  if (str_value.empty()) {
    col->set_count_nas(col->count_nas() + 1);
  } else {
    auto* items = col->mutable_categorical()->mutable_items();
    auto& item = (*items)[str_value];
    item.set_count(item.count() + 1);
  }
  return absl::OkStatus();
}

absl::Status UpdateCategoricalIntColumnSpec(
    int int_value, proto::Column* col,
    proto::DataSpecificationAccumulator::Column* col_acc) {
  if (int_value <= -1) {
    col->set_count_nas(col->count_nas() + 1);
  } else {
    if (int_value < 0) {
      return absl::InvalidArgumentError(
          absl::Substitute("Pre-integerized categorical features should be "
                           "greater or equal (special Out-of-vocabulary value) "
                           "to zero. Value $0 found for feature $1.",
                           int_value, col->name()));
    }
    if (int_value >= col->categorical().number_of_unique_values()) {
      col->mutable_categorical()->set_number_of_unique_values(int_value + 1);
    }
  }
  return absl::OkStatus();
}

utils::StatusOr<int64_t> CountNumberOfExamples(absl::string_view typed_path) {
  std::string sharded_path;
  proto::DatasetFormat format;
  std::tie(sharded_path, format) = GetDatasetPathAndType(typed_path);
  std::vector<std::string> paths;
  CHECK_OK(utils::ExpandInputShards(sharded_path, &paths));
  LOG(INFO) << "Counting the number of examples on " << paths.size()
            << " shard(s)";
  std::atomic<int64_t> number_of_examples{0};

  const auto& format_name = proto::DatasetFormat_Name(format);
  ASSIGN_OR_RETURN(
      auto creator, AbstractDataSpecCreatorRegisterer::Create(format_name),
      _ << "When creating a dataspec creator to read " << sharded_path
        << ". Make sure the format dependency is linked");

  {
    yggdrasil_decision_forests::utils::concurrency::ThreadPool pool(
        "CountNumberOfExamples", 50);
    pool.StartWorkers();
    for (const auto& path : paths) {
      pool.Schedule([&, path]() {
        number_of_examples += creator->CountExamples(path).value();
      });
    }
  }
  return number_of_examples;
}

bool BuildColumnGuide(const absl::string_view col_name,
                      const proto::DataSpecificationGuide& guide,
                      proto::ColumnGuide* col_guide) {
  bool found_user_guide = false;
  std::string matched_column_guide_pattern;

  // Set the default guide.
  *col_guide = guide.default_column_guide();

  // Search for a matching guide.
  for (auto& candidate_guide : guide.column_guides()) {
    if (!std::regex_match(std::string(col_name),
                          std::regex(candidate_guide.column_name_pattern()))) {
      continue;
    }
    // The spec guide contains a column guide matching this column name.

    if (found_user_guide && !candidate_guide.allow_multi_match()) {
      LOG(FATAL)
          << "At least two different column guides are matching the same "
             "column \""
          << col_name << "\".\nColumn guide 1: " << matched_column_guide_pattern
          << "\nColumn guide 2: " << candidate_guide.column_name_pattern()
          << "\n. If this is expected, set allow_multi_match=true in"
             " the column guide. Alterntively, ensure that each column is "
             "matched by only one column guide.";
    }
    MergeColumnGuide(candidate_guide, col_guide);
    found_user_guide = true;
    matched_column_guide_pattern = candidate_guide.column_name_pattern();
  }

  return found_user_guide;
}

absl::Status UpdateSingleColSpecWithGuideInfo(
    const proto::ColumnGuide& col_guide, proto::Column* col) {
  if (IsCategorical(col->type()) && col_guide.has_categorial()) {
    col->mutable_categorical()->set_max_number_of_unique_values(
        col_guide.categorial().max_vocab_count());

    col->mutable_categorical()->set_min_value_count(
        col_guide.categorial().min_vocab_frequency());

    col->mutable_categorical()->set_is_already_integerized(
        col_guide.categorial().is_already_integerized());

    if (col_guide.categorial().has_number_of_already_integerized_values()) {
      if (!col_guide.categorial().is_already_integerized()) {
        return absl::InvalidArgumentError(
            "\"number_of_already_integerized_values\" is set for a categorical "
            "column that is not already integerized.");
      }
      col->mutable_categorical()->set_number_of_unique_values(
          col_guide.categorial().number_of_already_integerized_values());
    }
  }
  if (IsMultiDimensional(col->type()) && col_guide.has_tokenizer()) {
    *col->mutable_tokenizer() = col_guide.tokenizer().tokenizer();
  }

  if (col->type() == ColumnType::DISCRETIZED_NUMERICAL) {
    col->mutable_discretized_numerical()->set_maximum_num_bins(
        col_guide.discretized_numerical().maximum_num_bins());
    col->mutable_discretized_numerical()->set_min_obs_in_bins(
        col_guide.discretized_numerical().min_obs_in_bins());
  }
  return absl::OkStatus();
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
