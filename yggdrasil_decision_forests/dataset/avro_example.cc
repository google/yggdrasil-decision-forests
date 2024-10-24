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

#include "yggdrasil_decision_forests/dataset/avro_example.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/avro.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::dataset::avro {
namespace {

// Computes the mapping from Avro field index to dataspec column index and
// dataspec unroll index.
//
// Args:
//   fields: Avro fields.
//   dataspec: Dataspec.
//   univariate_field_idx_to_column_idx: Mapping between the Avro field index
//    and the column index for the univariate features. -1's are used for
//    ignored fields.
//   multivariate_field_idx_to_unroll_idx: Mapping from Avro field index
//    to dataspec unroll index for multivariate fields. -1's are used for
//    ignored fields.
absl::Status ComputeReadingMaps(
    const std::vector<AvroField>& fields,
    const dataset::proto::DataSpecification& dataspec,
    std::vector<int>* univariate_field_idx_to_column_idx,
    std::vector<int>* multivariate_field_idx_to_unroll_idx) {
  univariate_field_idx_to_column_idx->assign(fields.size(), -1);
  multivariate_field_idx_to_unroll_idx->assign(fields.size(), -1);
  int field_idx = 0;
  for (const auto& field : fields) {
    if (field.type == AvroType::kArray) {
      const auto col_idx = GetOptionalColumnIdxFromName(field.name, dataspec);
      if (col_idx.has_value()) {
        // A multidimensional feature e.g. catset.
        (*univariate_field_idx_to_column_idx)[field_idx] = col_idx.value();
      } else {
        // Unroll the feature into multiple single dimensional features.
        int unstacked_idx = 0;
        for (const auto& unstacked : dataspec.unstackeds()) {
          if (unstacked.original_name() == field.name) {
            (*multivariate_field_idx_to_unroll_idx)[field_idx] = unstacked_idx;
            break;
          }
          unstacked_idx++;
        }
      }
    } else if (field.type != AvroType::kNull) {
      // A single dimensional feature.
      const auto col_idx = GetOptionalColumnIdxFromName(field.name, dataspec);
      if (col_idx.has_value()) {
        (*univariate_field_idx_to_column_idx)[field_idx] = col_idx.value();
      }
    }
    field_idx++;
  }
  return absl::OkStatus();
}

// Populates the dataspec with new columns the first time a multivariate field
// is seen (it is not possible to populate it before because the dimension is
// not known).
template <typename T>
absl::Status InitializeUnstackedColumn(
    const AvroField& field, const bool has_value, const std::vector<T>& values,
    const size_t record_idx, const size_t field_idx,
    const std::vector<int>& univariate_field_idx_to_column_idx,
    const std::vector<int>& multivariate_field_idx_to_unroll_idx,
    const std::vector<proto::ColumnGuide>& unstacked_guides,
    proto::DataSpecification* dataspec,
    proto::DataSpecificationAccumulator* accumulator) {
  // Check if field used.
  const auto unstacked_idx = multivariate_field_idx_to_unroll_idx[field_idx];
  auto* unstacked = dataspec->mutable_unstackeds(unstacked_idx);

  if (has_value) {
    if (!unstacked->has_size()) {
      // First time the field is seen with values. Let's create the
      // dataspec.

      // Populate unstack.
      unstacked->set_begin_column_idx(dataspec->columns_size());
      unstacked->set_size(values.size());
      // Create columns.
      const auto sub_col_names =
          UnstackedColumnNamesV2(field.name, values.size());
      for (const auto& sub_col_name : sub_col_names) {
        auto* col_spec = dataspec->add_columns();
        col_spec->set_name(sub_col_name);
        col_spec->set_type(unstacked->type());
        col_spec->set_count_nas(record_idx);
        RETURN_IF_ERROR(UpdateSingleColSpecWithGuideInfo(
            unstacked_guides[unstacked_idx], col_spec));
        accumulator->add_columns();
      }
    } else {
      // Check number of values
      if (values.size() != unstacked->size()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Inconsistent number of values for field ", field.name,
                         ". All the non-missing values should have the same "
                         "length. ",
                         unstacked->size(), " vs ", values.size()));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status AvroExampleReader::Implementation::OpenShard(
    absl::string_view path) {
  const bool is_first_file = reader_ == nullptr;
  auto save_previous_reader = std::move(reader_);
  ASSIGN_OR_RETURN(reader_, AvroReader::Create(path));
  if (is_first_file) {
    RETURN_IF_ERROR(ComputeReadingMaps(reader_->fields(), dataspec_,
                                       &univariate_field_idx_to_column_idx_,
                                       &multivariate_field_idx_to_unroll_idx_));
  } else {
    if (save_previous_reader->fields() != reader_->fields()) {
      return absl::InvalidArgumentError(
          "All the files in the same shard should have the same schema.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> AvroExampleReader::Implementation::NextInShard(
    proto::Example* example) {
  example->clear_attributes();
  while (example->attributes_size() < dataspec_.columns_size()) {
    example->add_attributes();
  }

  ASSIGN_OR_RETURN(const bool has_record, reader_->ReadNextRecord());
  if (!has_record) {
    return false;
  }

  int field_idx = 0;
  for (const auto& field : reader_->fields()) {
    switch (field.type) {
      case AvroType::kUnknown:
      case AvroType::kNull:
        break;

      case AvroType::kBoolean: {
        ASSIGN_OR_RETURN(const auto value,
                         reader_->ReadNextFieldBoolean(field));
        const int col_idx = univariate_field_idx_to_column_idx_[field_idx];
        if (col_idx == -1) {
          // Ignore field.
          break;
        }
        if (!value.has_value()) {
          break;
        }
        example->mutable_attributes(col_idx)->set_boolean(*value);
      } break;

      case AvroType::kLong:
      case AvroType::kInt: {
        ASSIGN_OR_RETURN(const auto value,
                         reader_->ReadNextFieldInteger(field));
        const int col_idx = univariate_field_idx_to_column_idx_[field_idx];
        if (col_idx == -1) {
          // Ignore field.
          break;
        }
        if (!value.has_value()) {
          break;
        }
        example->mutable_attributes(col_idx)->set_numerical(*value);
      } break;

      case AvroType::kDouble:
      case AvroType::kFloat: {
        std::optional<double> value;
        if (field.type == AvroType::kFloat) {
          ASSIGN_OR_RETURN(value, reader_->ReadNextFieldFloat(field));
        } else {
          ASSIGN_OR_RETURN(value, reader_->ReadNextFieldDouble(field));
        }
        if (value.has_value() && std::isnan(*value)) {
          value = absl::nullopt;
        }
        const int col_idx = univariate_field_idx_to_column_idx_[field_idx];
        if (col_idx == -1) {
          // Ignore field.
          break;
        }
        if (!value.has_value()) {
          break;
        }
        example->mutable_attributes(col_idx)->set_numerical(*value);
      } break;

      case AvroType::kBytes:
      case AvroType::kString: {
        std::string value;
        ASSIGN_OR_RETURN(const auto has_value,
                         reader_->ReadNextFieldString(field, &value));
        const int col_idx = univariate_field_idx_to_column_idx_[field_idx];
        if (col_idx == -1) {
          // Ignore field.
          break;
        }
        if (!has_value) {
          break;
        }
        ASSIGN_OR_RETURN(auto int_value,
                         CategoricalStringToValueWithStatus(
                             value, dataspec_.columns(col_idx)));
        example->mutable_attributes(col_idx)->set_categorical(int_value);
      } break;

      case AvroType::kArray:
        switch (field.sub_type) {
          case AvroType::kDouble:
          case AvroType::kFloat: {
            bool has_value;
            std::vector<float> values;
            if (field.sub_type == AvroType::kFloat) {
              ASSIGN_OR_RETURN(
                  has_value, reader_->ReadNextFieldArrayFloat(field, &values));
            } else {
              ASSIGN_OR_RETURN(
                  has_value,
                  reader_->ReadNextFieldArrayDoubleIntoFloat(field, &values));
            }
            if (!has_value) {
              break;
            }

            // Check if field used.
            const auto unstacked_idx =
                multivariate_field_idx_to_unroll_idx_[field_idx];
            if (unstacked_idx == -1) {
              break;
            }

            auto& unstacked = dataspec_.unstackeds(unstacked_idx);
            for (int dim_idx = 0; dim_idx < values.size(); dim_idx++) {
              const int col_idx = unstacked.begin_column_idx() + dim_idx;
              const float value = values[dim_idx];
              if (!std::isnan(value)) {
                example->mutable_attributes(col_idx)->set_numerical(value);
              }
            }
          } break;

          case AvroType::kString:
          case AvroType::kBytes: {
            std::vector<std::string> values;
            ASSIGN_OR_RETURN(const auto has_value,
                             reader_->ReadNextFieldArrayString(field, &values));
            if (!has_value) {
              break;
            }

            const auto univariate_col_idx =
                univariate_field_idx_to_column_idx_[field_idx];
            if (univariate_col_idx != -1) {
              const auto& col_spec = dataspec_.columns(univariate_col_idx);
              auto* dst = example->mutable_attributes(univariate_col_idx)
                              ->mutable_categorical_set()
                              ->mutable_values();
              dst->Reserve(values.size());
              for (const auto& value : values) {
                ASSIGN_OR_RETURN(
                    auto int_value,
                    CategoricalStringToValueWithStatus(value, col_spec));
                dst->Add(int_value);
              }
              // Sets are expected to be sorted.
              std::sort(dst->begin(), dst->end());
              dst->erase(std::unique(dst->begin(), dst->end()), dst->end());
              break;
            }

            // Check if field used.
            const auto unstacked_idx =
                multivariate_field_idx_to_unroll_idx_[field_idx];
            if (unstacked_idx == -1) {
              break;
            }

            auto& unstacked = dataspec_.unstackeds(unstacked_idx);
            for (int dim_idx = 0; dim_idx < values.size(); dim_idx++) {
              const int col_idx = unstacked.begin_column_idx() + dim_idx;
              const auto& col_spec = dataspec_.columns(col_idx);
              ASSIGN_OR_RETURN(auto int_value,
                               CategoricalStringToValueWithStatus(
                                   values[dim_idx], col_spec));
              example->mutable_attributes(col_idx)->set_categorical(int_value);
            }
          } break;

          case AvroType::kArray:
            switch (field.sub_sub_type) {
              case AvroType::kFloat:
              case AvroType::kDouble: {
                std::vector<std::vector<float>> values;
                bool has_value;

                if (field.sub_sub_type == AvroType::kFloat) {
                  ASSIGN_OR_RETURN(
                      has_value,
                      reader_->ReadNextFieldArrayArrayFloat(field, &values));
                } else {
                  ASSIGN_OR_RETURN(
                      has_value,
                      reader_->ReadNextFieldArrayArrayDoubleIntoFloat(field,
                                                                      &values));
                }

                if (!has_value) {
                  break;
                }
                const int col_idx =
                    univariate_field_idx_to_column_idx_[field_idx];
                if (col_idx == -1) {
                  // Ignore field.
                  break;
                }
                auto* dst_vectors = example->mutable_attributes(col_idx)
                                        ->mutable_numerical_vector_sequence()
                                        ->mutable_vectors();
                dst_vectors->Reserve(values.size());
                for (const auto& sub_values : values) {
                  *dst_vectors->Add()->mutable_values() = {sub_values.begin(),
                                                           sub_values.end()};
                }
              } break;
              default:
                return absl::UnimplementedError(
                    absl::StrCat("Unsupported avro sub_sub_type ",
                                 TypeToString(field.sub_type)));
            }
            break;

          default:
            return absl::UnimplementedError(absl::StrCat(
                "Unsupported sub type ", TypeToString(field.sub_type),
                " for kArray field ", field.name));
        }
        break;
    }
    field_idx++;
  }
  DCHECK_EQ(field_idx, reader_->fields().size());
  return true;
}

absl::StatusOr<dataset::proto::DataSpecification> CreateDataspec(
    absl::string_view path,
    const dataset::proto::DataSpecificationGuide& guide) {
  ASSIGN_OR_RETURN(auto reader, AvroReader::Create(path));
  const auto schema_string = reader->schema_string();
  ASSIGN_OR_RETURN(auto spec, CreateDataspecImpl(std::move(reader), guide),
                   _ << "While creating dataspec for " << path
                     << " with schema " << schema_string);
  return spec;
}

absl::StatusOr<dataset::proto::DataSpecification> CreateDataspecImpl(
    std::unique_ptr<AvroReader> reader,
    const dataset::proto::DataSpecificationGuide& guide) {
  // TODO: Reading of multiple paths.

  // Infer the column spec for the single-dimensional features and the unstacked
  // column (without the size information) for the multi-dimensional features.
  std::vector<proto::ColumnGuide> unstacked_guides;
  ASSIGN_OR_RETURN(
      auto dataspec,
      internal::InferDataspec(reader->fields(), guide, &unstacked_guides));

  // Mapping between the Avro field index and the column index for the
  // univariate features. -1's are used for ignored fields.
  std::vector<int> univariate_field_idx_to_column_idx;
  // Mapping between the Avro field index and the unstacked index for the
  // multivariate features. -1's are used for ignored fields.
  std::vector<int> multivariate_field_idx_to_unroll_idx;
  RETURN_IF_ERROR(ComputeReadingMaps(reader->fields(), dataspec,
                                     &univariate_field_idx_to_column_idx,
                                     &multivariate_field_idx_to_unroll_idx));

  // Create the accumulator for the univariate features.
  proto::DataSpecificationAccumulator accumulator;
  while (accumulator.columns_size() < dataspec.columns_size()) {
    accumulator.add_columns();
  }

  size_t record_idx;
  for (record_idx = 0; true; record_idx++) {
    if (guide.max_num_scanned_rows_to_accumulate_statistics() > 0 &&
        record_idx >= guide.max_num_scanned_rows_to_accumulate_statistics()) {
      // Enough records scanned.
      break;
    }
    LOG_EVERY_N_SEC(INFO, 30) << record_idx << " row(s) processed";

    ASSIGN_OR_RETURN(const bool has_record, reader->ReadNextRecord());
    if (!has_record) {
      break;
    }
    int field_idx = 0;
    for (const auto& field : reader->fields()) {
      switch (field.type) {
        case AvroType::kUnknown:
        case AvroType::kNull:
          break;

        case AvroType::kBoolean: {
          ASSIGN_OR_RETURN(const auto value,
                           reader->ReadNextFieldBoolean(field));
          const int col_idx = univariate_field_idx_to_column_idx[field_idx];
          if (col_idx == -1) {
            // Ignore field.
            break;
          }
          auto* col_spec_ = dataspec.mutable_columns(col_idx);
          if (!value.has_value()) {
            col_spec_->set_count_nas(col_spec_->count_nas() + 1);
          } else {
            switch (col_spec_->type()) {
              case proto::ColumnType::BOOLEAN:
                UpdateComputeSpecBooleanFeatureWithBool(value.value(),
                                                        col_spec_);
                break;
              default:
                return absl::InvalidArgumentError(
                    absl::StrCat("Unsupported type ",
                                 proto::ColumnType_Name(col_spec_->type()),
                                 " for kBoolean field ", field.name));
            }
          }
        } break;

        case AvroType::kLong:
        case AvroType::kInt: {
          ASSIGN_OR_RETURN(const auto value,
                           reader->ReadNextFieldInteger(field));
          const int col_idx = univariate_field_idx_to_column_idx[field_idx];
          if (col_idx == -1) {
            // Ignore field.
            break;
          }
          auto* col_spec_ = dataspec.mutable_columns(col_idx);
          if (!value.has_value()) {
            col_spec_->set_count_nas(col_spec_->count_nas() + 1);
          } else {
            switch (col_spec_->type()) {
              case proto::ColumnType::NUMERICAL:
                FillContentNumericalFeature(
                    value.value(), accumulator.mutable_columns(col_idx));
                break;
              default:
                return absl::InvalidArgumentError(absl::StrCat(
                    "Unsupported type ",
                    proto::ColumnType_Name(col_spec_->type()), " for ",
                    TypeToString(field.type), " field ", field.name));
            }
          }
        } break;

        case AvroType::kDouble:
        case AvroType::kFloat: {
          std::optional<double> value;
          if (field.type == AvroType::kFloat) {
            ASSIGN_OR_RETURN(value, reader->ReadNextFieldFloat(field));
          } else {
            ASSIGN_OR_RETURN(value, reader->ReadNextFieldDouble(field));
          }
          if (value.has_value() && std::isnan(*value)) {
            value = absl::nullopt;
          }

          const int col_idx = univariate_field_idx_to_column_idx[field_idx];
          if (col_idx == -1) {
            // Ignore field.
            break;
          }
          auto* col_spec_ = dataspec.mutable_columns(col_idx);
          if (!value.has_value()) {
            col_spec_->set_count_nas(col_spec_->count_nas() + 1);
          } else {
            switch (col_spec_->type()) {
              case proto::ColumnType::NUMERICAL:
                FillContentNumericalFeature(
                    value.value(), accumulator.mutable_columns(col_idx));
                break;
              default:
                return absl::InvalidArgumentError(absl::StrCat(
                    "Unsupported type ",
                    proto::ColumnType_Name(col_spec_->type()), " for ",
                    TypeToString(field.type), " field ", field.name));
            }
          }
        } break;

        case AvroType::kBytes:
        case AvroType::kString: {
          std::string value;
          ASSIGN_OR_RETURN(const auto has_value,
                           reader->ReadNextFieldString(field, &value));

          const int col_idx = univariate_field_idx_to_column_idx[field_idx];
          if (col_idx == -1) {
            // Ignore field.
            break;
          }
          auto* col_spec_ = dataspec.mutable_columns(col_idx);
          if (!has_value) {
            col_spec_->set_count_nas(col_spec_->count_nas() + 1);
          } else {
            switch (col_spec_->type()) {
              case proto::ColumnType::CATEGORICAL:
                RETURN_IF_ERROR(AddTokensToCategoricalColumnSpec(
                    std::vector<std::string>{value}, col_spec_));
                break;
              default:
                return absl::InvalidArgumentError(absl::StrCat(
                    "Unsupported type ",
                    proto::ColumnType_Name(col_spec_->type()), " for ",
                    TypeToString(field.type), " field ", field.name));
            }
          }
        } break;

        case AvroType::kArray:
          switch (field.sub_type) {
            case AvroType::kDouble:
            case AvroType::kFloat: {
              bool has_value;
              std::vector<float> values;
              if (field.sub_type == AvroType::kFloat) {
                ASSIGN_OR_RETURN(
                    has_value, reader->ReadNextFieldArrayFloat(field, &values));
              } else {
                ASSIGN_OR_RETURN(
                    has_value,
                    reader->ReadNextFieldArrayDoubleIntoFloat(field, &values));
              }

              // Check if field used.
              const auto unstacked_idx =
                  multivariate_field_idx_to_unroll_idx[field_idx];
              if (unstacked_idx == -1) {
                break;
              }
              RETURN_IF_ERROR(InitializeUnstackedColumn(
                  field, has_value, values, record_idx, field_idx,
                  univariate_field_idx_to_column_idx,
                  multivariate_field_idx_to_unroll_idx, unstacked_guides,
                  &dataspec, &accumulator));

              // Populate column statistics.
              auto& unstacked = dataspec.unstackeds(unstacked_idx);
              if (unstacked.has_size()) {
                for (int dim_idx = 0; dim_idx < values.size(); dim_idx++) {
                  const int col_idx = unstacked.begin_column_idx() + dim_idx;
                  const auto value = values[dim_idx];
                  auto* col_spec = dataspec.mutable_columns(col_idx);
                  if (!has_value || std::isnan(value)) {
                    col_spec->set_count_nas(col_spec->count_nas() + 1);
                  } else {
                    switch (dataspec.columns(col_idx).type()) {
                      case proto::ColumnType::NUMERICAL:
                        FillContentNumericalFeature(
                            value, accumulator.mutable_columns(col_idx));
                        break;
                      default:
                        return absl::InvalidArgumentError(
                            absl::StrCat("Unsupported type ",
                                         proto::ColumnType_Name(
                                             dataspec.columns(col_idx).type()),
                                         " for ", TypeToString(field.type),
                                         " field ", field.name));
                    }
                  }
                }
              }
            } break;

            case AvroType::kString:
            case AvroType::kBytes: {
              std::vector<std::string> values;
              ASSIGN_OR_RETURN(
                  const auto has_value,
                  reader->ReadNextFieldArrayString(field, &values));

              const auto univariate_col_idx =
                  univariate_field_idx_to_column_idx[field_idx];
              if (univariate_col_idx != -1) {
                auto* col_spec = dataspec.mutable_columns(univariate_col_idx);
                if (!has_value) {
                  col_spec->set_count_nas(col_spec->count_nas() + 1);
                } else {
                  switch (dataspec.columns(univariate_col_idx).type()) {
                    case proto::ColumnType::CATEGORICAL_SET:
                      RETURN_IF_ERROR(
                          AddTokensToCategoricalColumnSpec(values, col_spec));
                      break;
                    default:
                      return absl::InvalidArgumentError(absl::StrCat(
                          "Unsupported type ",
                          proto::ColumnType_Name(
                              dataspec.columns(univariate_col_idx).type()),
                          " for ", TypeToString(field.type), " field ",
                          field.name));
                  }
                }
                break;
              }

              // Check if field used.
              const auto unstacked_idx =
                  multivariate_field_idx_to_unroll_idx[field_idx];
              if (unstacked_idx == -1) {
                break;
              }
              RETURN_IF_ERROR(InitializeUnstackedColumn(
                  field, has_value, values, record_idx, field_idx,
                  univariate_field_idx_to_column_idx,
                  multivariate_field_idx_to_unroll_idx, unstacked_guides,
                  &dataspec, &accumulator));

              // Populate column statistics.
              auto& unstacked = dataspec.unstackeds(unstacked_idx);
              if (unstacked.has_size()) {
                for (int dim_idx = 0; dim_idx < values.size(); dim_idx++) {
                  const int col_idx = unstacked.begin_column_idx() + dim_idx;
                  auto* col_spec = dataspec.mutable_columns(col_idx);
                  if (!has_value) {
                    col_spec->set_count_nas(col_spec->count_nas() + 1);
                  } else {
                    switch (dataspec.columns(col_idx).type()) {
                      case proto::ColumnType::CATEGORICAL:
                        RETURN_IF_ERROR(AddTokensToCategoricalColumnSpec(
                            std::vector<std::string>{values[dim_idx]},
                            col_spec));
                        break;
                      default:
                        return absl::InvalidArgumentError(
                            absl::StrCat("Unsupported type ",
                                         proto::ColumnType_Name(
                                             dataspec.columns(col_idx).type()),
                                         "for ", TypeToString(field.type),
                                         " field ", field.name));
                    }
                  }
                }
              }
            } break;

            case AvroType::kArray:
              switch (field.sub_sub_type) {
                case AvroType::kFloat:
                case AvroType::kDouble: {
                  std::vector<std::vector<float>> values;

                  bool has_value;
                  if (field.sub_sub_type == AvroType::kFloat) {
                    ASSIGN_OR_RETURN(
                        has_value,
                        reader->ReadNextFieldArrayArrayFloat(field, &values));
                  } else {
                    ASSIGN_OR_RETURN(
                        has_value,
                        reader->ReadNextFieldArrayArrayDoubleIntoFloat(
                            field, &values));
                  }

                  const int col_idx =
                      univariate_field_idx_to_column_idx[field_idx];
                  if (col_idx == -1) {
                    // Ignore field.
                    break;
                  }
                  auto* col_spec_ = dataspec.mutable_columns(col_idx);
                  if (!has_value) {
                    col_spec_->set_count_nas(col_spec_->count_nas() + 1);
                  } else {
                    switch (col_spec_->type()) {
                      case proto::ColumnType::NUMERICAL_VECTOR_SEQUENCE:
                        RETURN_IF_ERROR(
                            UpdateComputeSpecNumericalVectorSequenceWithArrayArrayNumerical(
                                values, col_spec_,
                                accumulator.mutable_columns(col_idx)));
                        break;
                      default:
                        return absl::InvalidArgumentError(absl::StrCat(
                            "Unsupported semantic ",
                            proto::ColumnType_Name(col_spec_->type()),
                            " for an array of array of floats field ",
                            field.name));
                    }
                  }
                } break;
                default:
                  return absl::UnimplementedError(
                      absl::StrCat("Unsupported avro sub_sub_type ",
                                   TypeToString(field.sub_sub_type)));
              }
              break;

            default:
              return absl::UnimplementedError(absl::StrCat(
                  "Unsupported avro type ", TypeToString(field.type)));
          }
          break;
      }
      field_idx++;
    }
    DCHECK_EQ(field_idx, reader->fields().size());
  }

  if (record_idx == 0) {
    return absl::InvalidArgumentError("No record found");
  }
  dataspec.set_created_num_rows(record_idx);
  RETURN_IF_ERROR(FinalizeComputeSpec(guide, accumulator, &dataspec));

  RETURN_IF_ERROR(reader->Close());
  return dataspec;
}

absl::Status AvroDataSpecCreator::CreateDataspec(
    const std::vector<std::string>& paths,
    const proto::DataSpecificationGuide& guide,
    proto::DataSpecification* data_spec) {
  if (paths.empty()) {
    return absl::InvalidArgumentError("No path provided");
  }
  if (paths.size() > 1) {
    LOG(INFO) << "Only using first Avro file (of " << paths.size()
              << " files) to determine dataset schema";
  }
  ASSIGN_OR_RETURN(*data_spec, avro::CreateDataspec(paths.front(), guide),
                   _ << "While creating dataspec for " << paths.front());
  return absl::OkStatus();
}

namespace internal {

absl::StatusOr<dataset::proto::DataSpecification> InferDataspec(
    const std::vector<AvroField>& fields,
    const dataset::proto::DataSpecificationGuide& guide,
    std::vector<proto::ColumnGuide>* unstacked_guides) {
  dataset::proto::DataSpecification dataspec;

  const auto create_column =
      [&dataspec](const absl::string_view key,
                  const proto::ColumnType representation_type,
                  const proto::ColumnGuide& col_guide,
                  const bool manual_column_guide = false) -> absl::Status {
    proto::Column* column = dataspec.add_columns();
    column->set_name(std::string(key));
    if (manual_column_guide && col_guide.has_type()) {
      column->set_is_manual_type(true);
      column->set_type(col_guide.type());
    } else {
      column->set_is_manual_type(false);
      column->set_type(representation_type);
    }
    return UpdateSingleColSpecWithGuideInfo(col_guide, column);
  };

  for (const auto& field : fields) {
    proto::ColumnGuide col_guide;
    ASSIGN_OR_RETURN(bool has_column_guide,
                     BuildColumnGuide(field.name, guide, &col_guide));
    if (!has_column_guide && guide.ignore_columns_without_guides()) {
      continue;
    }
    if (col_guide.ignore_column()) {
      continue;
    }
    if (field.type == AvroType::kUnknown || field.type == AvroType::kNull) {
      continue;
    }

    switch (field.type) {
      case AvroType::kUnknown:
      case AvroType::kNull:
        return absl::InternalError("Unknown field");
      case AvroType::kBoolean:
        RETURN_IF_ERROR(create_column(field.name, proto::ColumnType::BOOLEAN,
                                      col_guide, has_column_guide));
        break;
      case AvroType::kLong:
      case AvroType::kInt:
      case AvroType::kFloat:
      case AvroType::kDouble:
        RETURN_IF_ERROR(create_column(field.name, proto::ColumnType::NUMERICAL,
                                      col_guide, has_column_guide));
        break;
      case AvroType::kBytes:
      case AvroType::kString:
        RETURN_IF_ERROR(create_column(field.name,
                                      proto::ColumnType::CATEGORICAL, col_guide,
                                      has_column_guide));
        break;
      case AvroType::kArray:
        switch (field.sub_type) {
          case AvroType::kFloat:
          case AvroType::kDouble: {
            (*unstacked_guides).push_back(col_guide);
            auto* unstacked = dataspec.add_unstackeds();
            unstacked->set_original_name(field.name);
            unstacked->set_type(proto::ColumnType::NUMERICAL);
          } break;
          case AvroType::kString:
          case AvroType::kBytes: {
            if (has_column_guide && col_guide.has_type()) {
              if (col_guide.type() == proto::ColumnType::CATEGORICAL_SET) {
                RETURN_IF_ERROR(create_column(
                    field.name, proto::ColumnType::CATEGORICAL_SET, col_guide,
                    has_column_guide));
                break;
              } else {
                return absl::InvalidArgumentError(
                    absl::StrCat("Unsupported type ",
                                 proto::ColumnType_Name(col_guide.type()),
                                 " for kString or kBytes field ", field.name));
              }
            }

            (*unstacked_guides).push_back(col_guide);
            auto* unstacked = dataspec.add_unstackeds();
            unstacked->set_original_name(field.name);
            unstacked->set_type(proto::ColumnType::CATEGORICAL);
          } break;

            //============ sub array =====================

          case AvroType::kArray:
            switch (field.sub_sub_type) {
              case AvroType::kFloat:
              case AvroType::kDouble: {
                RETURN_IF_ERROR(create_column(
                    field.name, proto::ColumnType::NUMERICAL_VECTOR_SEQUENCE,
                    col_guide, has_column_guide));
              } break;

              default:
                return absl::UnimplementedError(absl::StrCat(
                    "Unsupported type ", TypeToString(field.sub_type),
                    " in array or array"));
            }
            break;

            // =========== end of sub array ==============

          default:
            return absl::UnimplementedError(
                absl::StrCat("Unsupported type ", TypeToString(field.sub_type),
                             " in array"));
        }
        break;
    }
  }
  return dataspec;
}

}  // namespace internal

}  // namespace yggdrasil_decision_forests::dataset::avro
