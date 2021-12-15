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

#include "yggdrasil_decision_forests/dataset/csv_example_reader.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace dataset {

using proto::ColumnType;

CsvExampleReader::Implementation::Implementation(
    const proto::DataSpecification& data_spec,
    const absl::optional<std::vector<int>> ensure_non_missing)
    : data_spec_(data_spec), ensure_non_missing_(ensure_non_missing) {}

absl::Status CsvExampleReader::Implementation::OpenShard(
    const absl::string_view path) {
  ASSIGN_OR_RETURN(auto file_handle, file::OpenInputFile(path));
  csv_reader_ = absl::make_unique<utils::csv::Reader>(file_handle.get());
  RETURN_IF_ERROR(file_closer_.reset(std::move(file_handle)));

  std::vector<absl::string_view>* new_header;
  ASSIGN_OR_RETURN(const bool has_header, csv_reader_->NextRow(&new_header));
  if (!has_header) {
    return absl::InvalidArgumentError("CSV file without header");
  }

  if (csv_header_.empty()) {
    csv_header_ = {new_header->begin(), new_header->end()};
    RETURN_IF_ERROR(BuildColIdxToFeatureLabelIdx(data_spec_, csv_header_,
                                                 &col_idx_to_field_idx_));
  } else {
    if (!std::equal(csv_header_.begin(), csv_header_.end(), new_header->begin(),
                    new_header->end())) {
      return absl::InvalidArgumentError(
          absl::StrCat("The header of ", path,
                       " does not match the header of the other files"));
    }
  }
  return absl::OkStatus();
}

utils::StatusOr<bool> CsvExampleReader::Implementation::NextInShard(
    proto::Example* example) {
  std::vector<absl::string_view>* row;
  ASSIGN_OR_RETURN(const bool has_row, csv_reader_->NextRow(&row));
  if (!has_row) {
    return false;
  }
  RETURN_IF_ERROR(CsvRowToExample({row->begin(), row->end()}, data_spec_,
                                  col_idx_to_field_idx_, example));
  return true;
}

CsvExampleReader::CsvExampleReader(
    const proto::DataSpecification& data_spec,
    const absl::optional<std::vector<int>> ensure_non_missing)
    : sharded_csv_reader_(data_spec, ensure_non_missing) {}

// Does this value looks like to be a numerical value?
bool LooksLikeANumber(const absl::string_view value) {
  float tmp;
  return absl::SimpleAtof(value, &tmp);
}

// Update the dataspec with a new example. This operation is applied once the
// column type is decided. Example of update includes adding new dictionary
// entry for a categorical attribute, or updating the mean for a numerical
// column.
void UpdateDataSpecWithCsvExample(
    const std::vector<std::string>& fields,
    const std::vector<int>& col_idx_to_field_idx,
    proto::DataSpecification* data_spec,
    proto::DataSpecificationAccumulator* accumulator) {
  for (int col_idx = 0; col_idx < data_spec->columns_size(); col_idx++) {
    proto::Column* col = data_spec->mutable_columns(col_idx);
    if (col->is_unstacked()) {
      LOG(FATAL) << "Unstacked numerical features not supported for csv files";
    }
    auto* col_acc = accumulator->mutable_columns(col_idx);
    // Skip NAs
    const auto& value = fields[col_idx_to_field_idx[col_idx]];
    const auto lower_case = absl::AsciiStrToLower(value);
    if (value.empty() || lower_case == CSV_NA || lower_case == CSV_NA_V2) {
      col->set_count_nas(col->count_nas() + 1);
      continue;
    }
    // Mean, min and max of single dimension numerical columns.
    if (IsNumerical(col->type()) && !IsMultiDimensional(col->type())) {
      float num_value;
      CHECK(absl::SimpleAtof(value, &num_value))
          << "The value \"" << value << "\" of attribute \"" << col->name()
          << "\" cannot be parsed as a float.  Possible reasons => solution: "
             "1) You forced the type NUMERICAL => Set the type to something "
             "else. 2) You specified a regression task (simpleML Playground) "
             "for a classification => Set the task to classification.";
      FillContentNumericalFeature(num_value, col_acc);
    }
    if (IsCategorical(col->type())) {
      // Retrieve the items.
      std::vector<std::string> tokens;
      if (IsMultiDimensional(col->type())) {
        Tokenize(value, col->tokenizer(), &tokens);
      } else {
        tokens.push_back(value);
      }
      AddTokensToCategoricalColumnSpec(tokens, col);
    }
    if (col->type() == proto::ColumnType::DISCRETIZED_NUMERICAL) {
      float num_value;
      CHECK(absl::SimpleAtof(value, &num_value))
          << "The value \"" << value << "\" of attribute \"" << col->name()
          << "\" cannot be parsed as a float.  Possible reasons => solution: "
             "1) You forced the type DISCRETIZED_NUMERICAL => Set the type to "
             "something "
             "else.";
      UpdateComputeSpecDiscretizedNumerical(num_value, col, col_acc);
    }
    if (col->type() == proto::ColumnType::BOOLEAN) {
      float num_value;
      CHECK(absl::SimpleAtof(value, &num_value))
          << "The value \"" << value << "\" of attribute \"" << col->name()
          << "\" cannot be parsed as a float.  Possible reasons => solution: "
             "1) You forced the type BOOLEAN => Set the type to something "
             "else.";
      UpdateComputeSpecBooleanFeature(num_value, col);
    }
  }
}

void CsvDataSpecCreator::InferColumnsAndTypes(
    const std::vector<std::string>& paths,
    const proto::DataSpecificationGuide& guide,
    proto::DataSpecification* data_spec) {
  // For each dataspec column index, gives the csv column index and the
  // dataspec guide.
  std::vector<std::pair<int, proto::ColumnGuide>> spec_col_idx_2_csv_col_idx;
  std::vector<std::string> csv_header;

  int nrow = 0;
  for (const auto& path : paths) {
    // Open the csv file.
    auto csv_file = file::OpenInputFile(path).value();
    yggdrasil_decision_forests::utils::csv::Reader reader(csv_file.get());
    file::InputFileCloser closer(std::move(csv_file));

    // Read the header.
    std::vector<absl::string_view>* row;
    const bool has_header = reader.NextRow(&row).value();
    if (!has_header) {
      LOG(FATAL) << path << " is empty.";
    }

    if (csv_header.empty()) {
      // Create the dataspec columns.
      csv_header = {row->begin(), row->end()};
      InitializeDataSpecFromColumnNames(guide, csv_header, data_spec,
                                        &spec_col_idx_2_csv_col_idx);
    } else {
      if (!std::equal(csv_header.begin(), csv_header.end(), row->begin(),
                      row->end())) {
        LOG(FATAL) << "The header of " << path
                   << " does not match the header of " << paths.front();
      }
    }
    while (reader.NextRow(&row).value()) {
      LOG_INFO_EVERY_N_SEC(30, _ << nrow << " row(s) processed");
      // Check if we have seen enought records to determine all the types.
      if (guide.max_num_scanned_rows_to_guess_type() > 0 &&
          nrow > guide.max_num_scanned_rows_to_guess_type()) {
        LOG(INFO)
            << "Stop scanning the csv file to infer the type. Some records "
               "were not considered.";
        break;
      }
      // Check the number of fields.
      if (row->size() != csv_header.size()) {
        LOG(QFATAL) << "Inconsistent number of columns at line " << nrow
                    << " of file " << path << ". The header has "
                    << csv_header.size() << " field(s) while this line has "
                    << row->size();
      }

      for (int col_idx = 0; col_idx < spec_col_idx_2_csv_col_idx.size();
           col_idx++) {
        auto& csv_col_idx_and_guide = spec_col_idx_2_csv_col_idx[col_idx];
        proto::Column* col = data_spec->mutable_columns(col_idx);
        if (col->is_manual_type()) {
          // The user has already specified the type of this column.
          continue;
        }
        const absl::string_view value = (*row)[csv_col_idx_and_guide.first];
        const auto lower_case = absl::AsciiStrToLower(value);
        if (value.empty() || lower_case == CSV_NA || lower_case == CSV_NA_V2) {
          // We cannot do anything with Na values.
          continue;
        }
        // Update the type of the column.
        auto new_type = InferType(
            guide, value, guide.default_column_guide().tokenizer().tokenizer(),
            col->type());
        col->set_type(new_type);
      }
      nrow++;
    }
    if (guide.max_num_scanned_rows_to_guess_type() > 0 &&
        nrow > guide.max_num_scanned_rows_to_guess_type())
      break;
  }
  CHECK_OK(UpdateColSpecsWithGuideInfo(spec_col_idx_2_csv_col_idx, data_spec));
}

void CsvDataSpecCreator::ComputeColumnStatistics(
    const std::vector<std::string>& paths,
    const proto::DataSpecificationGuide& guide,
    proto::DataSpecification* data_spec,
    proto::DataSpecificationAccumulator* accumulator) {
  std::vector<int> col_idx_to_field_idx;
  std::vector<std::string> csv_header;
  uint64_t nrow = 0;
  for (const auto& path : paths) {
    if (guide.max_num_scanned_rows_to_accumulate_statistics() > 0 &&
        nrow > guide.max_num_scanned_rows_to_accumulate_statistics()) {
      break;
    }

    // Open the csv file.
    auto csv_file = file::OpenInputFile(path).value();
    yggdrasil_decision_forests::utils::csv::Reader reader(csv_file.get());
    file::InputFileCloser closer(std::move(csv_file));

    // Read the header.
    std::vector<absl::string_view>* row;
    const bool has_header = reader.NextRow(&row).value();
    if (!has_header) {
      LOG(FATAL) << path << " is empty.";
    }

    if (csv_header.empty()) {
      // Create the dataspec columns.
      csv_header = {row->begin(), row->end()};
      CHECK_OK(BuildColIdxToFeatureLabelIdx(*data_spec, csv_header,
                                            &col_idx_to_field_idx));
    } else {
      if (!std::equal(csv_header.begin(), csv_header.end(), row->begin(),
                      row->end())) {
        LOG(FATAL) << "The header of " << path
                   << " does not match the header of " << paths.front();
      }
    }
    while (reader.NextRow(&row).value()) {
      LOG_INFO_EVERY_N_SEC(30, _ << nrow << " row(s) processed");
      // Check the number of fields.
      if (row->size() != csv_header.size()) {
        LOG(QFATAL) << "Inconsistent number of columns at line " << nrow
                    << " of file " << path << ". The header has "
                    << csv_header.size() << " field(s) while this line has "
                    << row->size();
      }
      UpdateDataSpecWithCsvExample({row->begin(), row->end()},
                                   col_idx_to_field_idx, data_spec,
                                   accumulator);
      nrow++;
    }
  }
  data_spec->set_created_num_rows(nrow);
}

utils::StatusOr<int64_t> CsvDataSpecCreator::CountExamples(
    absl::string_view path) {
  int64_t count = 0;

  ASSIGN_OR_RETURN(auto csv_file, file::OpenInputFile(path));
  yggdrasil_decision_forests::utils::csv::Reader reader(csv_file.get());
  file::InputFileCloser closer(std::move(csv_file));

  std::vector<absl::string_view>* row;
  while (true) {
    ASSIGN_OR_RETURN(const bool has_data, reader.NextRow(&row));
    if (!has_data) {
      break;
    }
    count++;
  }

  // Note: We remove the header.
  return count - 1;
}

ColumnType InferType(const proto::DataSpecificationGuide& guide,
                     const absl::string_view value,
                     const proto::Tokenizer& tokenizer,
                     const ColumnType previous_type) {
  auto type = previous_type;
  // Boolean is the weakest type.
  if (type == ColumnType::UNKNOWN) {
    if (guide.detect_boolean_as_numerical()) {
      if (guide.detect_numerical_as_discretized_numerical()) {
        type = ColumnType::DISCRETIZED_NUMERICAL;
      } else {
        type = ColumnType::NUMERICAL;
      }
    } else {
      type = ColumnType::BOOLEAN;
    }
  }
  // Nothing is more complex than CATEGORICAL_SET.
  if (type == ColumnType::CATEGORICAL_SET) {
    return type;
  }
  // Boolean -> Numerical
  if (type == ColumnType::BOOLEAN) {
    if (value != "0" && value != "1") {
      type = guide.detect_numerical_as_discretized_numerical()
                 ? ColumnType::DISCRETIZED_NUMERICAL
                 : ColumnType::NUMERICAL;
    }
  }
  // One dimension -> Multi dimensions.
  if (!IsMultiDimensional(type)) {
    if (LooksMultiDimensional(value, tokenizer)) {
      if (type == ColumnType::NUMERICAL || type == ColumnType::BOOLEAN ||
          type == ColumnType::DISCRETIZED_NUMERICAL) {
        type = ColumnType::NUMERICAL_SET;
      }
      if (type == ColumnType::CATEGORICAL) {
        type = ColumnType::CATEGORICAL_SET;
      }
    }
  }
  // Numerical -> Categorical.
  if (IsNumerical(type)) {
    bool remain_numerical = true;
    if (IsMultiDimensional(type)) {
      std::vector<std::string> tokens;
      Tokenize(value, tokenizer, &tokens);
      for (const auto& token : tokens) {
        if (!LooksLikeANumber(token)) {
          remain_numerical = false;
          break;
        }
      }
    } else {
      remain_numerical = LooksLikeANumber(value);
    }
    if (!remain_numerical) {
      // Make the type categorical.
      switch (type) {
        case ColumnType::NUMERICAL:
        case ColumnType::DISCRETIZED_NUMERICAL:
          type = ColumnType::CATEGORICAL;
          break;
        case ColumnType::NUMERICAL_SET:
          type = ColumnType::CATEGORICAL_SET;
          break;
        case ColumnType::NUMERICAL_LIST:
          type = ColumnType::CATEGORICAL_LIST;
          break;
        default:
          LOG(FATAL) << "Non supported type for categorization.";
      }
    }
  }
  return type;
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
