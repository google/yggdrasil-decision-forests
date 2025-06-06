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

#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/formats.h"
#include "yggdrasil_decision_forests/dataset/formats.pb.h"
#include "yggdrasil_decision_forests/dataset/synthetic_dataset.h"
#include "yggdrasil_decision_forests/dataset/synthetic_dataset.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace dataset {
namespace {

using test::ApproximatelyEqualsProto;
using test::EqualsProto;

// Detects the names and types of the columns. This is the first stage of the
// full dataspec creation.
void InferDataSpecType(const absl::string_view typed_path,
                       const proto::DataSpecificationGuide& guide,
                       proto::DataSpecification* data_spec) {
  std::string sharded_path;
  proto::DatasetFormat format;
  std::tie(sharded_path, format) = GetDatasetPathAndType(typed_path);
  std::vector<std::string> paths;
  CHECK_OK(utils::ExpandInputShards(sharded_path, &paths));
  const auto& format_name = proto::DatasetFormat_Name(format);
  auto creator = AbstractDataSpecCreatorRegisterer::Create(format_name).value();
  CHECK_OK(creator->InferColumnsAndTypes(paths, guide, data_spec));
  FinalizeInferTypes(guide, data_spec);
}

std::string DatasetDir() {
  return file::JoinPath(
      test::DataRootDirectory(),
      "yggdrasil_decision_forests/test_data/dataset");
}

std::string ToyDatasetTypedPathCsv() {
  return absl::StrCat("csv:", file::JoinPath(DatasetDir(), "toy.csv"));
}

std::string ToyDatasetTypedPathTFExampleTFRecord() {
  return absl::StrCat("tfrecord:",
                      file::JoinPath(DatasetDir(), "toy.tfe-tfrecord@2"));
}

// Sort the column in lexicographic order.
void SortColumnByName(proto::DataSpecification* data_spec) {
  std::sort(data_spec->mutable_columns()->begin(),
            data_spec->mutable_columns()->end(),
            [](const proto::Column& a, const proto::Column& b) {
              return a.name() < b.name();
            });
}

void RemoveDtypes(proto::DataSpecification* data_spec) {
  for (proto::Column& column : *data_spec->mutable_columns()) {
    ASSERT_TRUE(column.has_dtype());
    column.clear_dtype();
  }
}

proto::DataSpecificationGuide ToyDatasetGuide1() {
  proto::DataSpecificationGuide guide = PARSE_TEST_PROTO(
      R"pb(
        default_column_guide { categorial { min_vocab_frequency: 2 } }
        column_guides {
          type: CATEGORICAL
          column_name_pattern: "Cat_3"
          categorial {
            is_already_integerized: true
            number_of_already_integerized_values: 10
          }
        }
      )pb");
  return guide;
}

proto::DataSpecificationGuide ToyDatasetGuide2() {
  proto::DataSpecificationGuide guide = PARSE_TEST_PROTO(
      R"pb(
        column_guides { column_name_pattern: "Num.*" type: STRING }
      )pb");
  return guide;
}

proto::DataSpecificationGuide ToyDatasetGuide3() {
  proto::DataSpecificationGuide guide = PARSE_TEST_PROTO(
      R"pb(
        column_guides {
          column_name_pattern: "Num.*"
          type: DISCRETIZED_NUMERICAL
        }
        ignore_columns_without_guides: true
      )pb");
  return guide;
}

proto::DataSpecificationGuide ToyDatasetGuide4() {
  proto::DataSpecificationGuide guide = PARSE_TEST_PROTO(
      R"pb(
        default_column_guide { categorial { min_vocab_frequency: 1 } }
        column_guides { column_name_pattern: "Num_2" }
        column_guides { column_name_pattern: "Cat_1" }
        ignore_columns_without_guides: true
      )pb");
  return guide;
}

proto::DataSpecificationGuide ToyDatasetGuideIgnoreColumn() {
  proto::DataSpecificationGuide guide = PARSE_TEST_PROTO(
      R"pb(
        column_guides { column_name_pattern: "Num.*" ignore_column: true }
      )pb");
  return guide;
}

proto::DataSpecification ToyDatasetExpectedDataSpecTypeOnlyNoGuide(
    bool with_dtype = false) {
  proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"pb(
        columns {
          type: NUMERICAL
          name: "Num_1"
          is_manual_type: false
          dtype: DTYPE_FLOAT32
        }
        columns {
          type: NUMERICAL
          name: "Num_2"
          is_manual_type: false
          dtype: DTYPE_FLOAT32
        }
        columns {
          type: CATEGORICAL
          name: "Cat_1"
          is_manual_type: false
          dtype: DTYPE_BYTES
        }
        columns {
          type: CATEGORICAL
          name: "Cat_2"
          is_manual_type: false
          dtype: DTYPE_BYTES
        }
        columns {
          type: CATEGORICAL_SET
          name: "Cat_set_1"
          is_manual_type: false
          dtype: DTYPE_BYTES
        }
        columns {
          type: CATEGORICAL_SET
          name: "Cat_set_2"
          is_manual_type: false
          dtype: DTYPE_BYTES
        }
        columns {
          type: BOOLEAN
          name: "Bool_1"
          is_manual_type: false
          dtype: DTYPE_INT64
        }
        columns {
          type: BOOLEAN
          name: "Bool_2"
          is_manual_type: false
          dtype: DTYPE_INT64
        }
        columns {
          type: NUMERICAL
          name: "Cat_3"
          is_manual_type: false
          dtype: DTYPE_FLOAT32
        }
      )pb");

  if (!with_dtype) {
    RemoveDtypes(&data_spec);
  }
  return data_spec;
}

proto::DataSpecification ToyDatasetExpectedDataSpecTypeOnlyGuide2(
    bool with_dtype = false) {
  proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"pb(
        columns {
          type: STRING
          name: "Num_1"
          is_manual_type: true
          dtype: DTYPE_FLOAT32
        }
        columns {
          type: STRING
          name: "Num_2"
          is_manual_type: true
          dtype: DTYPE_FLOAT32
        }
        columns {
          type: CATEGORICAL
          name: "Cat_1"
          is_manual_type: false
          dtype: DTYPE_BYTES
        }
        columns {
          type: CATEGORICAL
          name: "Cat_2"
          is_manual_type: false
          dtype: DTYPE_BYTES
        }
        columns {
          type: CATEGORICAL_SET
          name: "Cat_set_1"
          is_manual_type: false
          dtype: DTYPE_BYTES
        }
        columns {
          type: CATEGORICAL_SET
          name: "Cat_set_2"
          is_manual_type: false
          dtype: DTYPE_BYTES
        }
        columns {
          type: BOOLEAN
          name: "Bool_1"
          is_manual_type: false
          dtype: DTYPE_INT64
        }
        columns {
          type: BOOLEAN
          name: "Bool_2"
          is_manual_type: false
          dtype: DTYPE_INT64
        }
        columns {
          type: NUMERICAL
          name: "Cat_3"
          is_manual_type: false
          dtype: DTYPE_FLOAT32
        }
      )pb");
  if (!with_dtype) {
    RemoveDtypes(&data_spec);
  }
  return data_spec;
}

proto::DataSpecification ToyDatasetExpectedDataSpecGuide1(
    bool with_dtype = false) {
  proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"pb(
        created_num_rows: 4
        columns {
          type: NUMERICAL
          name: "Num_1"
          is_manual_type: false
          numerical {
            mean: 2.5
            min_value: 1
            max_value: 4
            standard_deviation: 1.1180339887498949
          }
          dtype: DTYPE_FLOAT32
        }
        columns {
          type: NUMERICAL
          name: "Num_2"
          is_manual_type: false
          numerical { mean: 3 min_value: 2 max_value: 4 standard_deviation: 1 }
          count_nas: 2
          dtype: DTYPE_FLOAT32
        }
        columns {
          type: CATEGORICAL
          name: "Cat_1"
          is_manual_type: false
          categorical {
            most_frequent_value: 1
            number_of_unique_values: 2
            min_value_count: 2
            max_number_of_unique_values: 2000
            is_already_integerized: false
            items {
              key: "<OOD>"
              value { index: 0 count: 2 }
            }
            items {
              key: "A"
              value { index: 1 count: 2 }
            }
          }
          dtype: DTYPE_BYTES
        }
        columns {
          type: CATEGORICAL
          name: "Cat_2"
          is_manual_type: false
          categorical {
            most_frequent_value: 0
            number_of_unique_values: 1
            min_value_count: 2
            max_number_of_unique_values: 2000
            is_already_integerized: false
            items {
              key: "<OOD>"
              value { index: 0 count: 2 }
            }
          }
          count_nas: 2
          dtype: DTYPE_BYTES
        }
        columns {
          type: CATEGORICAL_SET
          name: "Cat_set_1"
          is_manual_type: false
          categorical {
            most_frequent_value: 1
            number_of_unique_values: 4
            min_value_count: 2
            max_number_of_unique_values: 2000
            is_already_integerized: false
            items {
              key: "<OOD>"
              value { index: 0 count: 0 }
            }
            items {
              key: "x"
              value { index: 1 count: 4 }
            }
            items {
              key: "y"
              value { index: 2 count: 3 }
            }
            items {
              key: "z"
              value { index: 3 count: 2 }
            }
          }
          dtype: DTYPE_BYTES
        }
        columns {
          type: CATEGORICAL_SET
          name: "Cat_set_2"
          is_manual_type: false
          categorical {
            most_frequent_value: 1
            number_of_unique_values: 3
            min_value_count: 2
            max_number_of_unique_values: 2000
            is_already_integerized: false
            items {
              key: "<OOD>"
              value { index: 0 count: 1 }
            }
            items {
              key: "x"
              value { index: 1 count: 3 }
            }
            items {
              key: "y"
              value { index: 2 count: 2 }
            }
          }
          count_nas: 1
          dtype: DTYPE_BYTES
        }
        columns {
          type: BOOLEAN
          name: "Bool_1"
          is_manual_type: false
          boolean { count_true: 2 count_false: 2 }
          dtype: DTYPE_INT64
        }
        columns {
          type: BOOLEAN
          name: "Bool_2"
          is_manual_type: false
          count_nas: 2
          boolean { count_true: 1 count_false: 1 }
          dtype: DTYPE_INT64
        }
        columns {
          type: CATEGORICAL
          name: "Cat_3"
          is_manual_type: true
          categorical {
            number_of_unique_values: 10
            min_value_count: 2
            max_number_of_unique_values: 2000
            is_already_integerized: true
          }
          dtype: DTYPE_FLOAT32
        }
      )pb");
  if (!with_dtype) {
    RemoveDtypes(&data_spec);
  }
  return data_spec;
}

proto::DataSpecification ToyDatasetExpectedDataSpecGuide3(
    bool with_dtype = false) {
  proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"pb(
        created_num_rows: 4
        columns {
          type: DISCRETIZED_NUMERICAL
          name: "Num_1"
          is_manual_type: true
          numerical {
            mean: 2.5
            min_value: 1
            max_value: 4
            standard_deviation: 1.1180339887498949
          }
          discretized_numerical {
            boundaries: 1e-45
            boundaries: 2.4999998
            boundaries: 2.5000002
            boundaries: 3.5
            original_num_unique_values: 4
            maximum_num_bins: 255
            min_obs_in_bins: 3
          }
          dtype: DTYPE_FLOAT32
        }
        columns {
          type: DISCRETIZED_NUMERICAL
          name: "Num_2"
          is_manual_type: true
          numerical { mean: 3 min_value: 2 max_value: 4 standard_deviation: 1 }
          count_nas: 2
          discretized_numerical {
            boundaries: -1e-45
            boundaries: 1e-45
            boundaries: 2.9999998
            original_num_unique_values: 2
            maximum_num_bins: 255
            min_obs_in_bins: 3
          }
          dtype: DTYPE_FLOAT32
        }
      )pb");
  if (!with_dtype) {
    RemoveDtypes(&data_spec);
  }
  return data_spec;
}

proto::DataSpecification ToyDatasetExpectedDataSpecGuide4() {
  proto::DataSpecification data_spec = PARSE_TEST_PROTO(

      R"pb(
        created_num_rows: 2
        columns {
          type: NUMERICAL
          name: "Num_2"
          is_manual_type: false
          numerical { mean: 2 min_value: 2 max_value: 2 standard_deviation: 0 }
          count_nas: 1
        }
        columns {
          type: CATEGORICAL
          name: "Cat_1"
          is_manual_type: false
          categorical {
            most_frequent_value: 1
            number_of_unique_values: 3
            min_value_count: 1
            max_number_of_unique_values: 2000
            is_already_integerized: false
            items {
              key: "<OOD>"
              value { index: 0 count: 0 }
            }
            items {
              key: "A"
              value { index: 2 count: 1 }
            }
            items {
              key: "B"
              value { index: 1 count: 1 }
            }
          }
        }
      )pb");
  return data_spec;
}

proto::DataSpecification ToyDatasetExpectedDataSpecGuide4FirstRowType() {
  proto::DataSpecification data_spec = PARSE_TEST_PROTO(

      R"pb(
        created_num_rows: 4
        columns { name: "Num_2" is_manual_type: false count_nas: 2 }
        columns {
          type: CATEGORICAL
          name: "Cat_1"
          is_manual_type: false
          categorical {
            most_frequent_value: 1
            number_of_unique_values: 4
            min_value_count: 1
            max_number_of_unique_values: 2000
            is_already_integerized: false
            items {
              key: "<OOD>"
              value { index: 0 count: 0 }
            }
            items {
              key: "A"
              value { index: 1 count: 2 }
            }
            items {
              key: "C"
              value { index: 2 count: 1 }
            }
            items {
              key: "B"
              value { index: 3 count: 1 }
            }
          }
        }
      )pb");
  return data_spec;
}

proto::DataSpecification ToyDatasetExpectedDataSpecGuideIgnoreColumn(
    bool with_dtype = false) {
  proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"pb(
        columns {
          type: CATEGORICAL
          name: "Cat_1"
          is_manual_type: false
          categorical {
            most_frequent_value: 0
            number_of_unique_values: 1
            items {
              key: "<OOD>"
              value { index: 0 count: 4 }
            }
          }
          dtype: DTYPE_BYTES
        }
        columns {
          type: CATEGORICAL
          name: "Cat_2"
          is_manual_type: false
          categorical {
            most_frequent_value: 0
            number_of_unique_values: 1
            items {
              key: "<OOD>"
              value { index: 0 count: 2 }
            }
          }
          count_nas: 2
          dtype: DTYPE_BYTES
        }
        columns {
          type: CATEGORICAL_SET
          name: "Cat_set_1"
          is_manual_type: false
          categorical {
            most_frequent_value: 0
            number_of_unique_values: 1
            items {
              key: "<OOD>"
              value { index: 0 count: 9 }
            }
          }
          dtype: DTYPE_BYTES
        }
        columns {
          type: CATEGORICAL_SET
          name: "Cat_set_2"
          is_manual_type: false
          categorical {
            most_frequent_value: 0
            number_of_unique_values: 1
            items {
              key: "<OOD>"
              value { index: 0 count: 6 }
            }
          }
          dtype: DTYPE_BYTES
          count_nas: 1
        }
        columns {
          type: BOOLEAN
          name: "Bool_1"
          is_manual_type: false
          boolean { count_true: 2 count_false: 2 }
          dtype: DTYPE_INT64
        }
        columns {
          type: BOOLEAN
          name: "Bool_2"
          is_manual_type: false
          count_nas: 2
          boolean { count_true: 1 count_false: 1 }
          dtype: DTYPE_INT64
        }
        columns {
          type: NUMERICAL
          name: "Cat_3"
          is_manual_type: false
          numerical {
            mean: 1.75
            min_value: 1
            max_value: 3
            standard_deviation: 0.82915619758885
          }
          dtype: DTYPE_FLOAT32
        }
        created_num_rows: 4
      )pb");
  if (!with_dtype) {
    RemoveDtypes(&data_spec);
  }
  return data_spec;
}

TEST(Dataset, MergeColumnGuideTest1) {
  proto::ColumnGuide src = PARSE_TEST_PROTO(
      R"pb(
      )pb");
  proto::ColumnGuide dst = PARSE_TEST_PROTO(
      R"pb(
      )pb");
  proto::ColumnGuide expected_result = PARSE_TEST_PROTO(
      R"pb(
      )pb");
  proto::ColumnGuide result;
  MergeColumnGuide(src, &dst);
  EXPECT_THAT(dst, EqualsProto(expected_result));
}

TEST(Dataset, MergeColumnGuideTest2) {
  proto::ColumnGuide src = PARSE_TEST_PROTO(
      R"pb(
        type: NUMERICAL
      )pb");
  proto::ColumnGuide dst = PARSE_TEST_PROTO(
      R"pb(
      )pb");
  proto::ColumnGuide expected_result = PARSE_TEST_PROTO(
      R"pb(
        type: NUMERICAL
      )pb");
  proto::ColumnGuide result;
  MergeColumnGuide(src, &dst);
  EXPECT_THAT(dst, EqualsProto(expected_result));
}

TEST(Dataset, MergeColumnGuideTest3) {
  proto::ColumnGuide src = PARSE_TEST_PROTO(
      R"pb(
      )pb");
  proto::ColumnGuide dst = PARSE_TEST_PROTO(
      R"pb(
        type: NUMERICAL
      )pb");
  proto::ColumnGuide expected_result = PARSE_TEST_PROTO(
      R"pb(
        type: NUMERICAL
      )pb");
  proto::ColumnGuide result;
  MergeColumnGuide(src, &dst);
  EXPECT_THAT(dst, EqualsProto(expected_result));
}

TEST(Dataset, MergeColumnGuideTest4) {
  proto::ColumnGuide src = PARSE_TEST_PROTO(
      R"pb(
        type: STRING
      )pb");
  proto::ColumnGuide dst = PARSE_TEST_PROTO(
      R"pb(
        type: NUMERICAL
      )pb");
  proto::ColumnGuide expected_result = PARSE_TEST_PROTO(
      R"pb(
        type: STRING
      )pb");
  proto::ColumnGuide result;
  MergeColumnGuide(src, &dst);
  EXPECT_THAT(dst, EqualsProto(expected_result));
}

TEST(Dataset, MergeColumnGuideTest5) {
  proto::ColumnGuide src = PARSE_TEST_PROTO(
      R"pb(
        categorial { max_vocab_count: 8 }
      )pb");
  proto::ColumnGuide dst = PARSE_TEST_PROTO(
      R"pb(
        categorial { max_vocab_count: 10 }
      )pb");
  proto::ColumnGuide expected_result = PARSE_TEST_PROTO(
      R"pb(
        categorial { max_vocab_count: 8 }
      )pb");
  proto::ColumnGuide result;
  MergeColumnGuide(src, &dst);
  EXPECT_THAT(dst, EqualsProto(expected_result));
}

TEST(Dataset, InferDataSpecTypeCsv) {
  proto::DataSpecificationGuide guide;
  proto::DataSpecification data_spec;
  InferDataSpecType(ToyDatasetTypedPathCsv(), guide, &data_spec);
  auto target = ToyDatasetExpectedDataSpecTypeOnlyNoGuide();
  EXPECT_THAT(data_spec, EqualsProto(target));
}

TEST(Dataset, InferDataSpecTypeCsvBoolAsNumerical) {
  proto::DataSpecificationGuide guide;
  guide.set_detect_boolean_as_numerical(true);
  proto::DataSpecification data_spec;
  InferDataSpecType(ToyDatasetTypedPathCsv(), guide, &data_spec);
  auto target = ToyDatasetExpectedDataSpecTypeOnlyNoGuide();
  // The new boolean columns are seen as numerical.
  EXPECT_EQ(data_spec.columns(7).type(), proto::ColumnType::NUMERICAL);
  EXPECT_EQ(data_spec.columns(8).type(), proto::ColumnType::NUMERICAL);
}

TEST(Dataset, InferDataSpecTypeCsvIntegerizedCat) {
  proto::DataSpecificationGuide guide;
  proto::DataSpecification data_spec;
  auto* num_1 = guide.add_column_guides();
  num_1->set_column_name_pattern("Num_1");
  num_1->set_type(proto::ColumnType::CATEGORICAL);
  num_1->mutable_categorial()->set_is_already_integerized(true);
  InferDataSpecType(ToyDatasetTypedPathCsv(), guide, &data_spec);
  auto target = ToyDatasetExpectedDataSpecTypeOnlyNoGuide();
  EXPECT_EQ(data_spec.columns(0).type(), proto::ColumnType::CATEGORICAL);
  EXPECT_TRUE(data_spec.columns(0).categorical().is_already_integerized());
  EXPECT_TRUE(data_spec.columns(0).categorical().items().empty());
}

TEST(Dataset, InferDataSpecTypeTFExampleTFRecord) {
  proto::DataSpecificationGuide guide;
  proto::DataSpecification data_spec;
  InferDataSpecType(ToyDatasetTypedPathTFExampleTFRecord(), guide, &data_spec);
  auto target = ToyDatasetExpectedDataSpecTypeOnlyNoGuide(/*with_dtype=*/true);
  // Since tf.Example use dictionary, the columns can be in any random order.
  SortColumnByName(&data_spec);
  SortColumnByName(&target);
  EXPECT_THAT(data_spec, EqualsProto(target));
}

TEST(Dataset, InferDataSpecTypeCsvGuide2) {
  auto guide = ToyDatasetGuide2();
  proto::DataSpecification data_spec;
  InferDataSpecType(ToyDatasetTypedPathCsv(), guide, &data_spec);
  auto target = ToyDatasetExpectedDataSpecTypeOnlyGuide2();
  EXPECT_THAT(data_spec, EqualsProto(target));
}

TEST(Dataset, InferDataSpecTypeTFExampleTFRecordGuide2) {
  auto guide = ToyDatasetGuide2();
  proto::DataSpecification data_spec;
  InferDataSpecType(ToyDatasetTypedPathTFExampleTFRecord(), guide, &data_spec);
  auto target = ToyDatasetExpectedDataSpecTypeOnlyGuide2(/*with_dtype=*/true);
  // Since tf.Example use dictionary, the columns can be in any random order.
  SortColumnByName(&data_spec);
  SortColumnByName(&target);
  EXPECT_THAT(data_spec, ApproximatelyEqualsProto(target));
}

TEST(Dataset, CreateLocalDataSpecFromCsvGuide1) {
  auto guide = ToyDatasetGuide1();
  proto::DataSpecification data_spec;
  CreateDataSpec(ToyDatasetTypedPathCsv(), false, guide, &data_spec);
  auto target = ToyDatasetExpectedDataSpecGuide1();
  EXPECT_THAT(data_spec, ApproximatelyEqualsProto(target));
}

TEST(Dataset, CreateLocalDataSpecFromCsvGuideWithMaxNumStatistics) {
  auto guide = ToyDatasetGuide4();
  guide.set_max_num_scanned_rows_to_accumulate_statistics(2);
  proto::DataSpecification data_spec;
  CreateDataSpec(ToyDatasetTypedPathCsv(), false, guide, &data_spec);
  auto target = ToyDatasetExpectedDataSpecGuide4();
  EXPECT_THAT(data_spec, ApproximatelyEqualsProto(target));
}

TEST(Dataset, CreateLocalDataSpecFromCsvGuideWithMaxNumType) {
  auto guide = ToyDatasetGuide4();
  guide.set_max_num_scanned_rows_to_guess_type(1);
  proto::DataSpecification data_spec;
  CreateDataSpec(ToyDatasetTypedPathCsv(), false, guide, &data_spec);
  auto target = ToyDatasetExpectedDataSpecGuide4FirstRowType();
  EXPECT_THAT(data_spec, ApproximatelyEqualsProto(target));
}

TEST(Dataset, CreateLocalDataSpecFromCsvAllHash) {
  proto::DataSpecificationGuide guide;
  auto* col_guide = guide.add_column_guides();
  col_guide->set_column_name_pattern(".*");
  col_guide->set_type(proto::ColumnType::HASH);
  proto::DataSpecification data_spec;
  CreateDataSpec(ToyDatasetTypedPathCsv(), false, guide, &data_spec);
  EXPECT_THAT(data_spec,
              EqualsProto(PARSE_TEST_PROTO_WITH_TYPE(proto::DataSpecification,
                                                     R"(
            columns { type: HASH name: "Num_1" is_manual_type: true }
            columns {
              type: HASH
              name: "Num_2"
              is_manual_type: true
              count_nas: 2
            }
            columns { type: HASH name: "Cat_1" is_manual_type: true }
            columns {
              type: HASH
              name: "Cat_2"
              is_manual_type: true
              count_nas: 2
            }
            columns { type: HASH name: "Cat_set_1" is_manual_type: true }
            columns {
              type: HASH
              name: "Cat_set_2"
              is_manual_type: true
              count_nas: 1
            }
            columns { type: HASH name: "Bool_1" is_manual_type: true }
            columns {
              type: HASH
              name: "Bool_2"
              is_manual_type: true
              count_nas: 2
            }
            columns { type: HASH name: "Cat_3" is_manual_type: true }
            created_num_rows: 4
          )")));
}

TEST(Dataset, CreateLocalDataSpecFromTFExampleTFRecordGuide1) {
  auto guide = ToyDatasetGuide1();
  proto::DataSpecification data_spec;
  CreateDataSpec(ToyDatasetTypedPathTFExampleTFRecord(), false, guide,
                 &data_spec);
  auto target = ToyDatasetExpectedDataSpecGuide1(/*with_dtype=*/true);
  SortColumnByName(&data_spec);
  SortColumnByName(&target);
  EXPECT_THAT(data_spec, EqualsProto(target));
}

TEST(Dataset, CreateLocalDataSpecFromTFExampleTFRecordAllHash) {
  proto::DataSpecificationGuide guide;
  auto* col_guide = guide.add_column_guides();
  col_guide->set_column_name_pattern(".*");
  col_guide->set_type(proto::ColumnType::HASH);
  proto::DataSpecification data_spec;
  CreateDataSpec(ToyDatasetTypedPathTFExampleTFRecord(), false, guide,
                 &data_spec);
  auto target = ToyDatasetExpectedDataSpecGuide1();
  SortColumnByName(&data_spec);
  SortColumnByName(&target);
  EXPECT_THAT(data_spec,
              EqualsProto(PARSE_TEST_PROTO_WITH_TYPE(proto::DataSpecification,
                                                     R"(
            columns {
              type: HASH
              name: "Bool_1"
              is_manual_type: true
              dtype: DTYPE_INT64
            }
            columns {
              type: HASH
              name: "Bool_2"
              is_manual_type: true
              count_nas: 2
              dtype: DTYPE_INT64
            }
            columns {
              type: HASH
              name: "Cat_1"
              is_manual_type: true
              dtype: DTYPE_BYTES
            }
            columns {
              type: HASH
              name: "Cat_2"
              is_manual_type: true
              count_nas: 2
              dtype: DTYPE_BYTES
            }
            columns {
              type: HASH
              name: "Cat_3"
              is_manual_type: true
              dtype: DTYPE_FLOAT32
            }
            columns {
              type: HASH
              name: "Cat_set_1"
              is_manual_type: true
              dtype: DTYPE_BYTES
            }
            columns {
              type: HASH
              name: "Cat_set_2"
              is_manual_type: true
              count_nas: 1
              dtype: DTYPE_BYTES
            }
            columns {
              type: HASH
              name: "Num_1"
              is_manual_type: true
              dtype: DTYPE_FLOAT32
            }
            columns {
              type: HASH
              name: "Num_2"
              is_manual_type: true
              count_nas: 2
              dtype: DTYPE_FLOAT32
            }
            created_num_rows: 4
          )")));
}

TEST(Dataset, CreateLocalDataSpecFromCsvGuide3) {
  auto guide = ToyDatasetGuide3();
  proto::DataSpecification data_spec;
  CreateDataSpec(ToyDatasetTypedPathCsv(), false, guide, &data_spec);
  auto target = ToyDatasetExpectedDataSpecGuide3();
  EXPECT_THAT(data_spec, ApproximatelyEqualsProto(target));
}

TEST(Dataset, CreateLocalDataSpecFromTFExampleTFRecordGuide3) {
  auto guide = ToyDatasetGuide3();
  proto::DataSpecification data_spec;
  CreateDataSpec(ToyDatasetTypedPathTFExampleTFRecord(), false, guide,
                 &data_spec);
  auto target = ToyDatasetExpectedDataSpecGuide3(/*with_dtype=*/true);
  SortColumnByName(&data_spec);
  SortColumnByName(&target);
  EXPECT_THAT(data_spec, ApproximatelyEqualsProto(target));
}

TEST(Dataset, CreateLocalDataSpecFromCsvIgnoreColumn) {
  auto guide = ToyDatasetGuideIgnoreColumn();
  proto::DataSpecification data_spec;
  CreateDataSpec(ToyDatasetTypedPathCsv(), false, guide, &data_spec);
  auto target = ToyDatasetExpectedDataSpecGuideIgnoreColumn();
  EXPECT_THAT(data_spec, ApproximatelyEqualsProto(target));
}

TEST(Dataset, CreateLocalDataSpecFromTFExampleTFRecordIgnoreColumn) {
  auto guide = ToyDatasetGuideIgnoreColumn();
  proto::DataSpecification data_spec;
  CreateDataSpec(ToyDatasetTypedPathTFExampleTFRecord(), false, guide,
                 &data_spec);
  auto target =
      ToyDatasetExpectedDataSpecGuideIgnoreColumn(/*with_dtype=*/true);
  SortColumnByName(&data_spec);
  SortColumnByName(&target);
  EXPECT_THAT(data_spec, EqualsProto(target));
}

int64_t GroundTruthCounterNumberOfExamples(absl::string_view typed_path) {
  proto::DataSpecification data_spec;
  CreateDataSpec(typed_path, false, {}, &data_spec);
  return data_spec.created_num_rows();
}

TEST(CountNumberOfExamples, Base) {
  for (const auto& path :
       {ToyDatasetTypedPathCsv(), ToyDatasetTypedPathTFExampleTFRecord()}) {
    EXPECT_EQ(CountNumberOfExamples(path).value(),
              GroundTruthCounterNumberOfExamples(path))
        << "path:" << path;
  }
}

TEST(Dataset, CreateLocalDataSpecFromSyntheticTFExample) {
  const std::string path = absl::StrCat(
      "tfrecord:", file::JoinPath(test::TmpDirectory(), "dataset.tfr"));
  proto::SyntheticDatasetOptions options;
  options.set_num_multidimensional_numerical(2);
  CHECK_OK(GenerateSyntheticDataset(options, path));
  proto::DataSpecification data_spec;
  // Make sure a valid default dataspec is possible.
  CreateDataSpec(path, false, {}, &data_spec);
}

TEST(Dataset, UnknownType) {
  const auto dataset_path =
      file::JoinPath(test::TmpDirectory(), "dataset_with_unknown_type.csv");
  CHECK_OK(file::SetContent(dataset_path, "a,b\n1,\n2,\n"));
  proto::DataSpecificationGuide guide;
  guide.set_ignore_unknown_type_columns(true);
  proto::DataSpecification data_spec;
  InferDataSpecType(absl::StrCat("csv:", dataset_path), guide, &data_spec);
  // The column "b" is ignored.
  proto::DataSpecification target = PARSE_TEST_PROTO(
      R"pb(
        columns { type: NUMERICAL name: "a" is_manual_type: false }
      )pb");
  EXPECT_THAT(data_spec, EqualsProto(target));
}

TEST(Dataset, OverrideMostFrequentItem) {
  proto::DataSpecificationGuide guide;
  auto* col_guide = guide.add_column_guides();
  col_guide->set_column_name_pattern("^Cat_1$");
  col_guide->set_type(proto::CATEGORICAL);
  col_guide->mutable_categorial()
      ->mutable_override_most_frequent_item()
      ->set_str_value("B");
  col_guide->mutable_categorial()->set_min_vocab_frequency(1);

  proto::DataSpecification data_spec;
  CHECK_OK(CreateDataSpecWithStatus(ToyDatasetTypedPathCsv(), false, guide,
                                    &data_spec));
  LOG(INFO) << PrintHumanReadable(data_spec, false);
  auto& col = data_spec.columns(GetColumnIdxFromName("Cat_1", data_spec));
  ASSERT_OK_AND_ASSIGN(auto b_value,
                       CategoricalStringToValueWithStatus("B", col));
  EXPECT_EQ(col.categorical().most_frequent_value(), b_value);
}

TEST(Dataset, OverrideMostFrequentItemFail1) {
  proto::DataSpecificationGuide guide;
  auto* col_guide = guide.add_column_guides();
  col_guide->set_column_name_pattern("^Cat_2$");
  col_guide->set_type(proto::CATEGORICAL);
  col_guide->mutable_categorial()
      ->mutable_override_most_frequent_item()
      ->set_str_value("B");
  col_guide->mutable_categorial()->set_min_vocab_frequency(1);

  proto::DataSpecification data_spec;
  EXPECT_FALSE(CreateDataSpecWithStatus(ToyDatasetTypedPathCsv(), false, guide,
                                        &data_spec)
                   .ok());
}

TEST(Dataset, OverrideMostFrequentItemFail2) {
  proto::DataSpecificationGuide guide;
  auto* col_guide = guide.add_column_guides();
  col_guide->set_column_name_pattern("^Cat_1$");
  col_guide->set_type(proto::CATEGORICAL);
  col_guide->mutable_categorial()
      ->mutable_override_most_frequent_item()
      ->set_str_value("non-existing-item");
  col_guide->mutable_categorial()->set_min_vocab_frequency(1);

  proto::DataSpecification data_spec;
  EXPECT_FALSE(CreateDataSpecWithStatus(ToyDatasetTypedPathCsv(), false, guide,
                                        &data_spec)
                   .ok());
}

TEST(Dataset, InferCatSetCSVWithExplicitType) {
  proto::DataSpecificationGuide guide;
  guide.set_allow_tokenization_for_inference_as_categorical_set(false);
  auto* col_guide = guide.add_column_guides();
  col_guide->set_column_name_pattern("^Cat_set_1$");
  col_guide->set_type(proto::CATEGORICAL_SET);
  col_guide->mutable_categorial()->set_min_vocab_frequency(1);
  guide.set_ignore_columns_without_guides(true);
  proto::DataSpecification data_spec;
  CHECK_OK(CreateDataSpecWithStatus(ToyDatasetTypedPathCsv(), false, guide,
                                    &data_spec));

  EXPECT_THAT(data_spec,
              EqualsProto(PARSE_TEST_PROTO_WITH_TYPE(proto::DataSpecification,
                                                     R"(
            columns {
              type: CATEGORICAL_SET
              name: "Cat_set_1"
              is_manual_type: true
              categorical {
                most_frequent_value: 1
                number_of_unique_values: 4
                min_value_count: 1
                max_number_of_unique_values: 2000
                is_already_integerized: false
                items {
                  key: "<OOD>"
                  value { index: 0 count: 0 }
                }
                items {
                  key: "x"
                  value { index: 1 count: 4 }
                }
                items {
                  key: "y"
                  value { index: 2 count: 3 }
                }
                items {
                  key: "z"
                  value { index: 3 count: 2 }
                }
              }
            }
            created_num_rows: 4
          )")));
}

TEST(Dataset, InferCatSetCSVWithoutTokenization) {
  proto::DataSpecificationGuide guide;
  guide.set_allow_tokenization_for_inference_as_categorical_set(false);
  auto* col_guide = guide.add_column_guides();
  col_guide->set_column_name_pattern("^Cat_set_1$");
  col_guide->mutable_categorial()->set_min_vocab_frequency(1);
  guide.set_ignore_columns_without_guides(true);
  proto::DataSpecification data_spec;
  CHECK_OK(CreateDataSpecWithStatus(ToyDatasetTypedPathCsv(), false, guide,
                                    &data_spec));

  EXPECT_THAT(data_spec,
              EqualsProto(PARSE_TEST_PROTO_WITH_TYPE(proto::DataSpecification,
                                                     R"(
            columns {
              type: CATEGORICAL
              name: "Cat_set_1"
              is_manual_type: false
              categorical {
                most_frequent_value: 1
                number_of_unique_values: 5
                min_value_count: 1
                max_number_of_unique_values: 2000
                is_already_integerized: false
                items {
                  key: "<OOD>"
                  value { index: 0 count: 0 }
                }
                items {
                  key: "X"
                  value { index: 4 count: 1 }
                }
                items {
                  key: "X Y Z"
                  value { index: 2 count: 1 }
                }
                items {
                  key: "Y X Z"
                  value { index: 1 count: 1 }
                }
                items {
                  key: "X Y"
                  value { index: 3 count: 1}
                }
              }
            }
            created_num_rows: 4
          )")));
}

}  // namespace
}  // namespace dataset
}  // namespace yggdrasil_decision_forests
