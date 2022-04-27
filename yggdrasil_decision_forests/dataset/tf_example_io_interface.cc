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

#include "yggdrasil_decision_forests/dataset/tf_example_io_interface.h"

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/formats.h"
#include "yggdrasil_decision_forests/dataset/formats.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace dataset {

using yggdrasil_decision_forests::dataset::GetDatasetPathAndType;

using proto::ColumnType;

// Information attached to each feature a tf.example when parsing a tf.example
// container, in order to determine the most likely type of each feature.
struct InferTypeInfo {
  // Index of the column in the data spec. -1 indicates that the info is not
  // matched to a column (yet) in the data spec.
  int col_idx = -1;

  // User input guide for this column.
  proto::ColumnGuide col_guide;

  // If true, the feature is to be ignored.
  bool ignore_feature = false;
};

utils::StatusOr<std::unique_ptr<AbstractTFExampleReader>> CreateTFExampleReader(
    const absl::string_view typed_path) {
  std::string sharded_path;
  proto::DatasetFormat format;
  std::tie(sharded_path, format) = GetDatasetPathAndType(typed_path);
  const std::string& format_name = proto::DatasetFormat_Name(format);

  ASSIGN_OR_RETURN(
      auto reader, AbstractTFExampleReaderRegisterer::Create(format_name),
      _ << "When creating an tf example reader to read " << sharded_path
        << ". Make sure the format dependency is linked");
  RETURN_IF_ERROR(reader->Open(sharded_path));
  return std::move(reader);
}

utils::StatusOr<std::unique_ptr<AbstractTFExampleWriter>> CreateTFExampleWriter(
    const absl::string_view typed_path, const int64_t num_records_by_shard) {
  std::string sharded_path;
  proto::DatasetFormat format;
  std::tie(sharded_path, format) = GetDatasetPathAndType(typed_path);

  const std::string& format_name = proto::DatasetFormat_Name(format);
  ASSIGN_OR_RETURN(
      auto writer, AbstractTFExampleWriterRegisterer::Create(format_name),
      _ << "When creating an tf example writer to read " << sharded_path
        << ". Make sure the format dependency is linked");
  RETURN_IF_ERROR(writer->Open(sharded_path, num_records_by_shard));
  return std::move(writer);
}

// Update all the columns in a data spec with the appropriate guide information.
// Used when inferring the column type from a set of tf.examples.
absl::Status UpdateColSpecsWithGuideInfo(
    const absl::node_hash_map<std::string, InferTypeInfo>&
        tfe_feature_to_infer_type_info,
    proto::DataSpecification* data_spec) {
  for (const auto& type_info : tfe_feature_to_infer_type_info) {
    if (type_info.second.ignore_feature) continue;
    proto::Column* col = data_spec->mutable_columns(type_info.second.col_idx);
    const auto& col_guide = type_info.second.col_guide;
    RETURN_IF_ERROR(UpdateSingleColSpecWithGuideInfo(col_guide, col));
  }
  return absl::OkStatus();
}

// Specialization of "UpdateDataSpecWithTFExample" for unstacked features.
void UpdateDataSpecWithTFExampleUnstacked(
    const tensorflow::Example& example, proto::DataSpecification* data_spec,
    proto::DataSpecificationAccumulator* accumulator) {
  for (const auto& unstacked : data_spec->unstackeds()) {
    const auto it_feature =
        example.features().feature().find(unstacked.original_name());

    // Missing value.
    if (it_feature == example.features().feature().end() ||
        it_feature->second.kind_case() ==
            tensorflow::Feature::KindCase::KIND_NOT_SET ||
        (it_feature->second.kind_case() ==
             tensorflow::Feature::KindCase::kInt64List &&
         it_feature->second.int64_list().value_size() == 0) ||
        (it_feature->second.kind_case() ==
             tensorflow::Feature::KindCase::kFloatList &&
         it_feature->second.float_list().value_size() == 0)) {
      // Skip NAs
      for (int dim_idx = 0; dim_idx < unstacked.size(); dim_idx++) {
        proto::Column* col =
            data_spec->mutable_columns(unstacked.begin_column_idx() + dim_idx);
        col->set_count_nas(col->count_nas() + 1);
      }
      continue;
    }

    switch (it_feature->second.kind_case()) {
      case tensorflow::Feature::KindCase::kInt64List: {
        CHECK_EQ(it_feature->second.int64_list().value_size(), unstacked.size())
            << "Wrong number of value for multi dimension feature "
            << unstacked.original_name();
        for (int dim_idx = 0; dim_idx < unstacked.size(); dim_idx++) {
          const int col_idx = unstacked.begin_column_idx() + dim_idx;
          proto::Column* col = data_spec->mutable_columns(col_idx);
          auto* col_acc = accumulator->mutable_columns(col_idx);
          CHECK_OK(UpdateNumericalColumnSpec(
              it_feature->second.int64_list().value(dim_idx), col, col_acc));
        }
      } break;

      case tensorflow::Feature::KindCase::kFloatList: {
        CHECK_EQ(it_feature->second.float_list().value_size(), unstacked.size())
            << "Wrong number of value for multi dimension feature "
            << unstacked.original_name();
        for (int dim_idx = 0; dim_idx < unstacked.size(); dim_idx++) {
          const int col_idx = unstacked.begin_column_idx() + dim_idx;
          proto::Column* col = data_spec->mutable_columns(col_idx);
          auto* col_acc = accumulator->mutable_columns(col_idx);
          CHECK_OK(UpdateNumericalColumnSpec(
              it_feature->second.float_list().value(dim_idx), col, col_acc));
        }
      } break;

      case tensorflow::Feature::KindCase::kBytesList:
        LOG(FATAL) << "Byte value for numerical feature "
                   << unstacked.original_name();
        break;

      default:
        LOG(FATAL) << "Internal error";  // Should not happen;
    }
  }
}

// Specialization of "UpdateDataSpecWithTFExample" for non-unstacked features.
void UpdateDataSpecWithTFExampleBase(
    const tensorflow::Example& example, proto::DataSpecification* data_spec,
    proto::DataSpecificationAccumulator* accumulator) {
  for (int col_idx = 0; col_idx < data_spec->columns_size(); col_idx++) {
    proto::Column* col = data_spec->mutable_columns(col_idx);
    if (col->is_unstacked()) {
      continue;
    }
    auto* col_acc = accumulator->mutable_columns(col_idx);
    const auto it_feature = example.features().feature().find(col->name());
    if (it_feature == example.features().feature().end() ||
        it_feature->second.kind_case() ==
            tensorflow::Feature::KindCase::KIND_NOT_SET) {
      // Skip NAs
      col->set_count_nas(col->count_nas() + 1);
      continue;
    }
    // Mean of single dimension numerical columns.
    if (IsNumerical(col->type()) && !IsMultiDimensional(col->type())) {
      const float num_value =
          GetSingleFloatFromTFFeature(it_feature->second, *col);
      CHECK_OK(UpdateNumericalColumnSpec(num_value, col, col_acc));
    }

    if (IsCategorical(col->type())) {
      std::vector<std::string> tokens;
      GetCategoricalTokensFromTFFeature(it_feature->second, *col, &tokens);
      if (!IsMultiDimensional(col->type()) && tokens.empty()) {
        col->set_count_nas(col->count_nas() + 1);
        continue;
      }
      AddTokensToCategoricalColumnSpec(tokens, col);
    }

    if (col->type() == ColumnType::DISCRETIZED_NUMERICAL) {
      const float num_value =
          GetSingleFloatFromTFFeature(it_feature->second, *col);
      UpdateComputeSpecDiscretizedNumerical(num_value, col, col_acc);
    }

    if (col->type() == ColumnType::BOOLEAN) {
      const float num_value =
          GetSingleFloatFromTFFeature(it_feature->second, *col);
      UpdateComputeSpecBooleanFeature(num_value, col);
    }
  }
}

// Update the dataspec with a new example. This operation is applied once the
// column type is decided. Example of update includes adding new dictionary
// entry for a categorical attribute, or updating the mean for a numerical
// column.
void UpdateDataSpecWithTFExample(
    const tensorflow::Example& example, proto::DataSpecification* data_spec,
    proto::DataSpecificationAccumulator* accumulator) {
  UpdateDataSpecWithTFExampleBase(example, data_spec, accumulator);
  UpdateDataSpecWithTFExampleUnstacked(example, data_spec, accumulator);
}

TFExampleReaderToExampleReader::TFExampleReaderToExampleReader(
    const proto::DataSpecification& data_spec,
    const absl::optional<std::vector<int>> ensure_non_missing)
    : data_spec_(data_spec), ensure_non_missing_(ensure_non_missing) {}

absl::Status TFExampleReaderToExampleReader::Open(
    absl::string_view sharded_path) {
  tf_reader_ = CreateReader();
  RETURN_IF_ERROR(tf_reader_->Open(sharded_path));
  return absl::OkStatus();
}

utils::StatusOr<bool> TFExampleReaderToExampleReader::Next(
    proto::Example* example) {
  ASSIGN_OR_RETURN(bool did_read, tf_reader_->Next(&tfexample_buffer_));
  if (!did_read) {
    return false;
  }
  RETURN_IF_ERROR(TfExampleToExample(tfexample_buffer_, data_spec_, example));
  return true;
}

void TFExampleReaderToDataSpecCreator::InferColumnsAndTypes(
    const std::vector<std::string>& paths,
    const proto::DataSpecificationGuide& guide,
    proto::DataSpecification* data_spec) {
  auto reader = CreateReader();
  CHECK_OK(reader->Open(paths));
  data_spec->clear_columns();
  // Maps each tf.Example features to a column idx and a column guide.
  absl::node_hash_map<std::string, InferTypeInfo>
      tfe_feature_to_infer_type_info;
  // Number of rows scanned so far.
  uint64_t nrow = 0;
  tensorflow::Example example;
  while (reader->Next(&example).value()) {
    LOG_INFO_EVERY_N_SEC(30, _ << nrow << " row(s) processed");

    // Check if we have seen enough records to determine all the types.
    if (guide.max_num_scanned_rows_to_guess_type() > 0 &&
        nrow > guide.max_num_scanned_rows_to_guess_type()) {
      LOG(INFO) << "Stop scanning the dataset to infer the type. Some records "
                   "were not considered.";
      break;
    }

    for (const auto& feature : example.features().feature()) {
      const auto& col_name = feature.first;
      auto& col_type_info = tfe_feature_to_infer_type_info[col_name];
      if (col_type_info.ignore_feature) {
        continue;
      }
      proto::Column* column;

      // Create a new feature column in the data spec.
      if (col_type_info.col_idx == -1) {
        // Check for a user defined guide.
        const bool has_specific_guide =
            BuildColumnGuide(col_name, guide, &col_type_info.col_guide);
        if (!has_specific_guide && guide.ignore_columns_without_guides()) {
          // Set this feature to be ignored.
          col_type_info.ignore_feature = true;
          continue;
        }
        if (col_type_info.col_guide.ignore_column()) {
          col_type_info.ignore_feature = true;
          continue;
        }
        col_type_info.col_idx = data_spec->columns_size();
        column = data_spec->add_columns();
        column->set_name(col_name);
        if (has_specific_guide && col_type_info.col_guide.has_type()) {
          column->set_type(col_type_info.col_guide.type());
          column->set_is_manual_type(true);
        } else {
          column->set_is_manual_type(false);
        }
      } else {
        column = data_spec->mutable_columns(col_type_info.col_idx);
      }
      if (column->is_manual_type()) {
        // The user has already specified the type of this column.
        continue;
      }
      // Update the type of the column.
      int num_sub_values;
      auto new_type =
          InferType(guide, feature.second,
                    guide.default_column_guide().tokenizer().tokenizer(),
                    column->type(), &num_sub_values);
      column->set_type(new_type);
      // Minimum and maximum number of values (for multi dimensional columns).
      if (column->multi_values().has_max_observed_size()) {
        column->mutable_multi_values()->set_max_observed_size(std::max(
            column->multi_values().max_observed_size(), num_sub_values));
      } else {
        column->mutable_multi_values()->set_max_observed_size(num_sub_values);
      }
      if (column->multi_values().has_min_observed_size()) {
        column->mutable_multi_values()->set_min_observed_size(std::min(
            column->multi_values().min_observed_size(), num_sub_values));
      } else {
        column->mutable_multi_values()->set_min_observed_size(num_sub_values);
      }
    }
    nrow++;
  }
  CHECK_OK(
      UpdateColSpecsWithGuideInfo(tfe_feature_to_infer_type_info, data_spec));

  // Sort the column by name.
  std::sort(data_spec->mutable_columns()->begin(),
            data_spec->mutable_columns()->end(),
            [](const proto::Column& a, const proto::Column& b) {
              return a.name() < b.name();
            });
}

void TFExampleReaderToDataSpecCreator::ComputeColumnStatistics(
    const std::vector<std::string>& paths,
    const proto::DataSpecificationGuide& guide,
    proto::DataSpecification* data_spec,
    proto::DataSpecificationAccumulator* accumulator) {
  auto reader = CreateReader();
  CHECK_OK(reader->Open(paths));
  uint64_t nrow = 0;
  tensorflow::Example example;
  while (reader->Next(&example).value()) {
    if (guide.max_num_scanned_rows_to_accumulate_statistics() > 0 &&
        nrow > guide.max_num_scanned_rows_to_accumulate_statistics()) {
      break;
    }
    LOG_INFO_EVERY_N_SEC(30, _ << nrow << " row(s) processed");
    UpdateDataSpecWithTFExample(example, data_spec, accumulator);
    nrow++;
  }
  data_spec->set_created_num_rows(nrow);
}

utils::StatusOr<int64_t> TFExampleReaderToDataSpecCreator::CountExamples(
    absl::string_view path) {
  auto reader = CreateReader();
  CHECK_OK(reader->Open(path));
  int64_t count = 0;
  tensorflow::Example value;
  while (true) {
    ASSIGN_OR_RETURN(const bool has_more, reader->Next(&value));
    if (!has_more) {
      break;
    }
    count++;
  }
  return count;
}

// Returns the most
ColumnType InferType(const proto::DataSpecificationGuide& guide,
                     const tensorflow::Feature& feature,
                     const proto::Tokenizer& tokenizer,
                     const ColumnType previous_type, int* num_sub_values) {
  CHECK(num_sub_values != nullptr);
  *num_sub_values = 0;
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

  switch (feature.kind_case()) {
    case tensorflow::Feature::KindCase::KIND_NOT_SET:
      // We skip NA values.
      break;
    case tensorflow::Feature::KindCase::kFloatList: {
      *num_sub_values = feature.float_list().value_size();
      // Note: num_values == 0 can be a numerical set or a NA.
      if (feature.float_list().value_size() > 1) {
        type = ColumnType::NUMERICAL_SET;
      } else if (feature.float_list().value_size() == 1) {
        const float value = feature.float_list().value(0);
        if (!IsNumerical(type) && value != 0 && value != 1) {
          type = guide.detect_numerical_as_discretized_numerical()
                     ? ColumnType::DISCRETIZED_NUMERICAL
                     : ColumnType::NUMERICAL;
        }
      }
    } break;
    case tensorflow::Feature::KindCase::kInt64List: {
      *num_sub_values = feature.int64_list().value_size();
      // Note: num_values == 0 can be a numerical set or a NA.
      if (feature.int64_list().value_size() > 1) {
        type = ColumnType::NUMERICAL_SET;
      } else if (feature.int64_list().value_size() == 1) {
        const float value = feature.int64_list().value(0);
        if (!IsNumerical(type) && value != 0 && value != 1) {
          type = guide.detect_numerical_as_discretized_numerical()
                     ? ColumnType::DISCRETIZED_NUMERICAL
                     : ColumnType::NUMERICAL;
        }
      }
    } break;
    case tensorflow::Feature::KindCase::kBytesList: {
      *num_sub_values = feature.bytes_list().value_size();
      if (!IsCategorical(type)) {
        type = ColumnType::CATEGORICAL;
      }
      if (feature.bytes_list().value_size() > 1) {
        type = ColumnType::CATEGORICAL_SET;
      }
    } break;
  }
  return type;
}

TFExampleWriterToExampleWriter::TFExampleWriterToExampleWriter(
    const proto::DataSpecification& data_spec)
    : data_spec_(data_spec) {}

absl::Status TFExampleWriterToExampleWriter::Open(
    absl::string_view sharded_path, int64_t num_records_by_shard) {
  tf_writer_ = CreateWriter();
  RETURN_IF_ERROR(tf_writer_->Open(sharded_path, num_records_by_shard));
  return absl::OkStatus();
}

absl::Status TFExampleWriterToExampleWriter::Write(
    const proto::Example& example) {
  RETURN_IF_ERROR(
      ExampleToTfExampleWithStatus(example, data_spec_, &tfexample_buffer_));
  return tf_writer_->Write(tfexample_buffer_);
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
