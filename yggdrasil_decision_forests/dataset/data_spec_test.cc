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

#include "yggdrasil_decision_forests/dataset/data_spec.h"

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/example/example.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace dataset {
namespace {

using test::EqualsProto;
using test::StatusIs;
using testing::ElementsAre;

std::string DatasetDir() {
  return file::JoinPath(
      test::DataRootDirectory(),
      "yggdrasil_decision_forests/test_data/dataset");
}

TEST(DataSpecUtil, PrintHumanReadable) {
  const std::string ds_typed_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "adult.csv"));
  dataset::proto::DataSpecification data_spec;
  dataset::proto::DataSpecificationGuide guide;
  dataset::CreateDataSpec(ds_typed_path, false, guide, &data_spec);
  std::string readable_representation = PrintHumanReadable(data_spec, true);
  std::cout << readable_representation;
  // The ground truth has been validated using R:
  // > summary(read.csv("adult.csv"))
  std::string expected_result = R"(Number of records: 32561
Number of columns: 15

Number of columns by type:
	CATEGORICAL: 9 (60%)
	NUMERICAL: 6 (40%)

Columns:

CATEGORICAL: 9 (60%)
	3: "education" CATEGORICAL has-dict vocab-size:17 zero-ood-items most-frequent:"HS-grad" 10501 (32.2502%)
	14: "income" CATEGORICAL has-dict vocab-size:3 zero-ood-items most-frequent:"<=50K" 24720 (75.919%)
	5: "marital_status" CATEGORICAL has-dict vocab-size:8 zero-ood-items most-frequent:"Married-civ-spouse" 14976 (45.9937%)
	13: "native_country" CATEGORICAL num-nas:583 (1.79049%) has-dict vocab-size:41 num-oods:1 (0.00312715%) most-frequent:"United-States" 29170 (91.219%)
	6: "occupation" CATEGORICAL num-nas:1843 (5.66015%) has-dict vocab-size:15 zero-ood-items most-frequent:"Prof-specialty" 4140 (13.4774%)
	8: "race" CATEGORICAL has-dict vocab-size:6 zero-ood-items most-frequent:"White" 27816 (85.4274%)
	7: "relationship" CATEGORICAL has-dict vocab-size:7 zero-ood-items most-frequent:"Husband" 13193 (40.5178%)
	9: "sex" CATEGORICAL has-dict vocab-size:3 zero-ood-items most-frequent:"Male" 21790 (66.9205%)
	1: "workclass" CATEGORICAL num-nas:1836 (5.63865%) has-dict vocab-size:9 zero-ood-items most-frequent:"Private" 22696 (73.8682%)

NUMERICAL: 6 (40%)
	0: "age" NUMERICAL mean:38.5816 min:17 max:90 sd:13.6402
	10: "capital_gain" NUMERICAL mean:1077.65 min:0 max:99999 sd:7385.18
	11: "capital_loss" NUMERICAL mean:87.3038 min:0 max:4356 sd:402.954
	4: "education_num" NUMERICAL mean:10.0807 min:1 max:16 sd:2.57268
	2: "fnlwgt" NUMERICAL mean:189778 min:12285 max:1.4847e+06 sd:105548
	12: "hours_per_week" NUMERICAL mean:40.4375 min:1 max:99 sd:12.3472

Terminology:
	nas: Number of non-available (i.e. missing) values.
	ood: Out of dictionary.
	manually-defined: Attribute which type is manually defined by the user i.e. the type was not automatically inferred.
	tokenized: The attribute value is obtained through tokenization.
	has-dict: The attribute is attached to a string dictionary e.g. a categorical attribute stored as a string.
	vocab-size: Number of unique values.
)";
  EXPECT_EQ(readable_representation, expected_result);
}

// Loading two numerical columns of the adult dataset as DISCRETIZED_NUMERICAL.
TEST(Dataset, DiscretizeAdult) {
  const std::string ds_typed_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "adult.csv"));
  dataset::proto::DataSpecification data_spec;
  dataset::proto::DataSpecificationGuide guide = PARSE_TEST_PROTO(
      R"(
        column_guides {
          type: DISCRETIZED_NUMERICAL
          column_name_pattern: "age"
          discretized_numerical { maximum_num_bins: 10 }
        }
        column_guides {
          type: DISCRETIZED_NUMERICAL
          column_name_pattern: "capital_gain"
        }
        ignore_columns_without_guides: true
      )");
  dataset::CreateDataSpec(ds_typed_path, false, guide, &data_spec);

  std::string readable_representation = PrintHumanReadable(data_spec, true);
  LOG(INFO) << readable_representation;
  std::string expected_result = R"(Number of records: 32561
Number of columns: 2

Number of columns by type:
	DISCRETIZED_NUMERICAL: 2 (100%)

Columns:

DISCRETIZED_NUMERICAL: 2 (100%)
	0: "age" DISCRETIZED_NUMERICAL manually-defined mean:38.5816 min:17 max:90 sd:13.6402 discretized bins:10 orig-bins:73
	1: "capital_gain" DISCRETIZED_NUMERICAL manually-defined mean:1077.65 min:0 max:99999 sd:7385.18 discretized bins:104 orig-bins:119

Terminology:
	nas: Number of non-available (i.e. missing) values.
	ood: Out of dictionary.
	manually-defined: Attribute which type is manually defined by the user i.e. the type was not automatically inferred.
	tokenized: The attribute value is obtained through tokenization.
	has-dict: The attribute is attached to a string dictionary e.g. a categorical attribute stored as a string.
	vocab-size: Number of unique values.
)";
  EXPECT_EQ(readable_representation, expected_result);
}

// Loading of the adult dataset with all the numerical columns interpreted as
// DISCRETIZED_NUMERICAL.
TEST(Dataset, AdultAllDiscretized) {
  const std::string ds_typed_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "adult.csv"));
  dataset::proto::DataSpecification data_spec;
  dataset::proto::DataSpecificationGuide guide = PARSE_TEST_PROTO(
      R"(
        detect_numerical_as_discretized_numerical: true
      )");
  dataset::CreateDataSpec(ds_typed_path, false, guide, &data_spec);
  std::string readable_representation = PrintHumanReadable(data_spec, true);
  LOG(INFO) << readable_representation;
  std::string expected_result = R"(Number of records: 32561
Number of columns: 15

Number of columns by type:
	CATEGORICAL: 9 (60%)
	DISCRETIZED_NUMERICAL: 6 (40%)

Columns:

CATEGORICAL: 9 (60%)
	3: "education" CATEGORICAL has-dict vocab-size:17 zero-ood-items most-frequent:"HS-grad" 10501 (32.2502%)
	14: "income" CATEGORICAL has-dict vocab-size:3 zero-ood-items most-frequent:"<=50K" 24720 (75.919%)
	5: "marital_status" CATEGORICAL has-dict vocab-size:8 zero-ood-items most-frequent:"Married-civ-spouse" 14976 (45.9937%)
	13: "native_country" CATEGORICAL num-nas:583 (1.79049%) has-dict vocab-size:41 num-oods:1 (0.00312715%) most-frequent:"United-States" 29170 (91.219%)
	6: "occupation" CATEGORICAL num-nas:1843 (5.66015%) has-dict vocab-size:15 zero-ood-items most-frequent:"Prof-specialty" 4140 (13.4774%)
	8: "race" CATEGORICAL has-dict vocab-size:6 zero-ood-items most-frequent:"White" 27816 (85.4274%)
	7: "relationship" CATEGORICAL has-dict vocab-size:7 zero-ood-items most-frequent:"Husband" 13193 (40.5178%)
	9: "sex" CATEGORICAL has-dict vocab-size:3 zero-ood-items most-frequent:"Male" 21790 (66.9205%)
	1: "workclass" CATEGORICAL num-nas:1836 (5.63865%) has-dict vocab-size:9 zero-ood-items most-frequent:"Private" 22696 (73.8682%)

DISCRETIZED_NUMERICAL: 6 (40%)
	0: "age" DISCRETIZED_NUMERICAL mean:38.5816 min:17 max:90 sd:13.6402 discretized bins:74 orig-bins:73
	10: "capital_gain" DISCRETIZED_NUMERICAL mean:1077.65 min:0 max:99999 sd:7385.18 discretized bins:104 orig-bins:119
	11: "capital_loss" DISCRETIZED_NUMERICAL mean:87.3038 min:0 max:4356 sd:402.954 discretized bins:72 orig-bins:92
	4: "education_num" DISCRETIZED_NUMERICAL mean:10.0807 min:1 max:16 sd:2.57268 discretized bins:19 orig-bins:16
	2: "fnlwgt" DISCRETIZED_NUMERICAL mean:189778 min:12285 max:1.4847e+06 sd:105548 discretized bins:255 orig-bins:21648
	12: "hours_per_week" DISCRETIZED_NUMERICAL mean:40.4375 min:1 max:99 sd:12.3472 discretized bins:89 orig-bins:94

Terminology:
	nas: Number of non-available (i.e. missing) values.
	ood: Out of dictionary.
	manually-defined: Attribute which type is manually defined by the user i.e. the type was not automatically inferred.
	tokenized: The attribute value is obtained through tokenization.
	has-dict: The attribute is attached to a string dictionary e.g. a categorical attribute stored as a string.
	vocab-size: Number of unique values.
)";
  EXPECT_EQ(readable_representation, expected_result);
}

TEST(Dataset, ExampleToCsvRow) {
  const proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"(
        columns { type: NUMERICAL name: "a" }
        columns { type: NUMERICAL_SET name: "b" }
        columns { type: NUMERICAL_LIST name: "c" }
        columns {
          type: CATEGORICAL
          name: "d"
          categorical { is_already_integerized: true }
        }
        columns {
          type: CATEGORICAL_SET
          name: "e"
          categorical { is_already_integerized: true }
        }
        columns {
          type: CATEGORICAL_LIST
          name: "f"
          categorical { is_already_integerized: true }
        }
        columns { type: BOOLEAN name: "g" }
        columns { type: STRING name: "h" }
        columns {
          type: DISCRETIZED_NUMERICAL
          name: "i"
          discretized_numerical { boundaries: 0 boundaries: 1 boundaries: 2 }
        }
      )");
  const proto::Example example = PARSE_TEST_PROTO(
      R"(
        attributes { numerical: 0.5 }
        attributes { numerical_set: { values: 0 values: 1 } }
        attributes { numerical_list: { values: 0 values: 1 } }
        attributes { categorical: 1 }
        attributes { categorical_set: { values: 0 values: 1 } }
        attributes { categorical_list: { values: 0 values: 1 } }
        attributes { boolean: 1 }
        attributes { text: "hello" }
        attributes { discretized_numerical: 2 }
      )");
  std::vector<std::string> csv_fields;
  ExampleToCsvRow(example, data_spec, &csv_fields);
  EXPECT_THAT(csv_fields, ElementsAre("0.5", "0 1", "0 1", "1", "0 1", "0 1",
                                      "1", "hello", "1.5"));
}

TEST(DataSpec, MatchColumnNameToColumnIdxList) {
  std::vector<std::string> column_name_regexs{"A", "[CD]"};
  proto::DataSpecification data_spec;
  data_spec.add_columns()->set_name("A");
  data_spec.add_columns()->set_name("B");
  data_spec.add_columns()->set_name("C");
  data_spec.add_columns()->set_name("D");
  std::vector<int32_t> column_idxs;
  GetMultipleColumnIdxFromName(column_name_regexs, data_spec, &column_idxs);
  EXPECT_THAT(column_idxs, ElementsAre(0, 2, 3));
}

TEST(DataSpec, GetSingleColumnIdxFromName) {
  proto::DataSpecification data_spec;
  data_spec.add_columns()->set_name("A");
  data_spec.add_columns()->set_name("B");
  data_spec.add_columns()->set_name("C");
  data_spec.add_columns()->set_name("D");
  int32_t column_idx;
  EXPECT_FALSE(GetSingleColumnIdxFromName("[CD]", data_spec, &column_idx).ok());
  EXPECT_FALSE(GetSingleColumnIdxFromName("E", data_spec, &column_idx).ok());
  EXPECT_OK(GetSingleColumnIdxFromName("B", data_spec, &column_idx));
  EXPECT_EQ(column_idx, 1);
}

TEST(Dataset, TfExampleToExample) {
  const proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"(
        # Id: 0
        columns { type: NUMERICAL name: "a" }
        # Id: 1
        columns { type: NUMERICAL_SET name: "b" }
        # Id: 2
        columns { type: NUMERICAL_LIST name: "c" }
        # Id: 3
        columns {
          type: CATEGORICAL
          name: "d"
          categorical {
            is_already_integerized: true
            number_of_unique_values: 20
          }
        }
        # Id: 4
        columns {
          type: CATEGORICAL_SET
          name: "e"
          categorical {
            is_already_integerized: true
            number_of_unique_values: 20
          }
        }
        # Id: 5
        columns {
          type: CATEGORICAL_LIST
          name: "f"
          categorical {
            is_already_integerized: true
            number_of_unique_values: 20
          }
        }
        # Id: 6
        columns { type: BOOLEAN name: "g" }
        # Id: 7
        columns { type: STRING name: "h" }
        # Id: 8
        columns {
          type: DISCRETIZED_NUMERICAL
          name: "i"
          discretized_numerical { boundaries: 0 boundaries: 1 boundaries: 2 }
        }
        # Id: 9
        columns {
          type: NUMERICAL
          name: "j_0"
          numerical { mean: 0 }
          is_unstacked: true
        }
        # Id: 10
        columns {
          type: NUMERICAL
          name: "j_1"
          numerical { mean: 1 }
          is_unstacked: true
        }
        # Id: 11
        columns {
          type: NUMERICAL
          name: "j_2"
          numerical { mean: 2 }
          is_unstacked: true
        }
        # Id: 12
        columns {
          type: NUMERICAL
          name: "k_0"
          numerical { mean: 0 }
          is_unstacked: true
        }
        # Id: 13
        columns {
          type: NUMERICAL
          name: "k_1"
          numerical { mean: 1 }
          is_unstacked: true
        }
        # Id: 14
        columns {
          type: NUMERICAL
          name: "k_2"
          numerical { mean: 2 }
          is_unstacked: true
        }
        # Id: 15
        columns {
          type: DISCRETIZED_NUMERICAL
          name: "l_0"
          discretized_numerical { boundaries: 0 boundaries: 1 boundaries: 2 }
          is_unstacked: true
        }
        # Id: 16
        columns {
          type: DISCRETIZED_NUMERICAL
          name: "l_1"
          discretized_numerical { boundaries: 0 boundaries: 1 boundaries: 2 }
          is_unstacked: true
        }
        # Id: 17
        columns {
          type: DISCRETIZED_NUMERICAL
          name: "l_2"
          discretized_numerical { boundaries: 0 boundaries: 1 boundaries: 2 }
          is_unstacked: true
        }
        unstackeds {
          original_name: "j"
          begin_column_idx: 9
          size: 3
          type: NUMERICAL
        }
        unstackeds {
          original_name: "k"
          begin_column_idx: 12
          size: 3
          type: NUMERICAL
        }
        unstackeds {
          original_name: "l"
          begin_column_idx: 15
          size: 3
          type: DISCRETIZED_NUMERICAL
        }
      )");
  tensorflow::Example tf_example = PARSE_TEST_PROTO(
      R"(
        features {
          feature {
            key: "a"
            value { float_list { value: 1.0 } }
          }
          feature {
            key: "b"
            value { float_list { value: 2.0 value: 3.0 } }
          }
          feature {
            key: "c"
            value { float_list { value: 4.0 value: 5.0 } }
          }
          feature {
            key: "d"
            value { int64_list { value: 6 } }
          }
          feature {
            key: "e"
            value { int64_list { value: 7 value: 8 } }
          }
          feature {
            key: "f"
            value { int64_list { value: 9 value: 10 } }
          }
          feature {
            key: "g"
            value { float_list { value: 1.0 } }
          }
          feature {
            key: "h"
            value { bytes_list { value: "toto" } }
          }
          feature {
            key: "i"
            value { float_list { value: 1.5 } }
          }
          feature {
            key: "j"
            value { float_list { value: 10.0 value: 11.0 value: 12.0 } }
          }
          feature {
            key: "k"
            value { int64_list { value: 20 value: 21 value: 22 } }
          }
          feature {
            key: "l"
            value { float_list { value: 0.5 value: 1.5 value: 0.5 } }
          }
        }
      )");
  proto::Example example;
  EXPECT_OK(TfExampleToExample(tf_example, data_spec, &example));
  const proto::Example expected_example = PARSE_TEST_PROTO(
      R"(
        attributes { numerical: 1 }
        attributes { numerical_set { values: 2 values: 3 } }
        attributes { numerical_list { values: 4 values: 5 } }
        attributes { categorical: 6 }
        attributes { categorical_set { values: 7 values: 8 } }
        attributes { categorical_list { values: 9 values: 10 } }
        attributes { boolean: 1 }
        attributes { text: "toto" }
        attributes { discretized_numerical: 2 }
        attributes { numerical: 10 }
        attributes { numerical: 11 }
        attributes { numerical: 12 }
        attributes { numerical: 20 }
        attributes { numerical: 21 }
        attributes { numerical: 22 }
        attributes { discretized_numerical: 1 }
        attributes { discretized_numerical: 2 }
        attributes { discretized_numerical: 1 }
      )");
  EXPECT_THAT(example, EqualsProto(expected_example));

  tensorflow::Example convert_back_tf_example;
  EXPECT_OK(ExampleToTfExampleWithStatus(example, data_spec,
                                         &convert_back_tf_example));
  // The original int64_t is stored in a float.
  auto* values = (*tf_example.mutable_features()->mutable_feature())["k"]
      .mutable_float_list()
      ->mutable_value();
  values->Add(20.f);
  values->Add(21.f);
  values->Add(22.f);
  EXPECT_THAT(convert_back_tf_example, EqualsProto(tf_example));
}

TEST(Dataset, TfExampleToExampleErrors) {
  const proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"(
        columns {
          type: CATEGORICAL
          name: "a_0"
          numerical { mean: 0 }
          is_unstacked: true
        }
        columns {
          type: CATEGORICAL
          name: "a_1"
          numerical { mean: 1 }
          is_unstacked: true
        }
        columns {
          type: NUMERICAL
          name: "b_0"
          numerical { mean: 0 }
          is_unstacked: true
        }
        columns {
          type: NUMERICAL
          name: "b_1"
          numerical { mean: 1 }
          is_unstacked: true
        }
        unstackeds { original_name: "a" begin_column_idx: 0 size: 2 }
        unstackeds { original_name: "b" begin_column_idx: 2 size: 2 }
      )");

  tensorflow::Example example_1 = PARSE_TEST_PROTO(
      R"(
        features {
          feature {
            key: "a"
            value { float_list { value: 1.0 value: 2.0 } }
          }
        }
      )");
  proto::Example example;
  EXPECT_THAT(TfExampleToExample(example_1, data_spec, &example),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "a's type is not supported for stacked feature."));

  tensorflow::Example example_2 = PARSE_TEST_PROTO(
      R"(
        features {
          feature {
            key: "b"
            value { float_list { value: 1.0 value: 2.0 value: 3.0 } }
          }
        }
      )");
  EXPECT_THAT(TfExampleToExample(example_2, data_spec, &example),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Wrong number of elements for feature b"));

  tensorflow::Example example_3 = PARSE_TEST_PROTO(
      R"(
        features {
          feature {
            key: "b"
            value { bytes_list { value: "x" } }
          }
        }
      )");
  EXPECT_THAT(TfExampleToExample(example_3, data_spec, &example),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Feature b is not stored as float or int64."));
}

TEST(Dataset, Tokenizer) {
  std::vector<std::string> tokens;

  Tokenize("", PARSE_TEST_PROTO(R"()"), &tokens);
  EXPECT_THAT(tokens, ElementsAre());

  Tokenize("Hello world", PARSE_TEST_PROTO(R"(splitter: REGEX_MATCH)"),
           &tokens);
  EXPECT_THAT(tokens, ElementsAre("hello", "world"));

  Tokenize("12345Little Monsters zzz",
           PARSE_TEST_PROTO(R"(splitter: REGEX_MATCH)"), &tokens);
  EXPECT_THAT(tokens, ElementsAre("12345little", "monsters", "zzz"));

  Tokenize("Hello;the,world ", PARSE_TEST_PROTO(R"(splitter: SEPARATOR)"),
           &tokens);
  EXPECT_THAT(tokens, ElementsAre("hello", "the", "world"));

  Tokenize("Hello;the,world ",
           PARSE_TEST_PROTO(R"(splitter: SEPARATOR
                               grouping { bigrams: true })"),
           &tokens);
  EXPECT_THAT(tokens,
              ElementsAre("hello", "the", "world", "hello_the", "the_world"));

  Tokenize("Hello world", PARSE_TEST_PROTO(R"(splitter: CHARACTER
                                              grouping { unigrams: true })"),
           &tokens);
  EXPECT_EQ(tokens, std::vector<std::string>({"h", "e", "l", "l", "o", " ", "w",
                                              "o", "r", "l", "d"}));

  Tokenize("Hello world",
           PARSE_TEST_PROTO(
               R"(
                 splitter: CHARACTER
                 grouping { unigrams: false bigrams: true }
               )"),
           &tokens);
  EXPECT_THAT(tokens, ElementsAre("he", "el", "ll", "lo", "o ", " w", "wo",
                                  "or", "rl", "ld"));

  Tokenize("Hello world",
           PARSE_TEST_PROTO(
               R"(
                 splitter: CHARACTER
                 grouping { unigrams: false trigrams: true }
               )"),
           &tokens);
  EXPECT_THAT(tokens, ElementsAre("hel", "ell", "llo", "lo ", "o w", " wo",
                                  "wor", "orl", "rld"));

  Tokenize("Hello world",
           PARSE_TEST_PROTO(
               R"(
                 splitter: CHARACTER
                 grouping { unigrams: false bigrams: true trigrams: true }
               )"),
           &tokens);
  EXPECT_EQ(tokens, std::vector<std::string>({"he", "el", "ll", "lo", "o ",
                                              " w", "wo", "or", "rl", "ld",
                                              "hel", "ell", "llo", "lo ", "o w",
                                              " wo", "wor", "orl", "rld"}));
}

TEST(DataSpecUtil, CategoricalIdxsToRepresentation) {
  const std::vector<int> elements{1, 2, 3};
  proto::Column col_spec = PARSE_TEST_PROTO(
      R"(
        type: CATEGORICAL
        categorical {
          most_frequent_value: 0
          number_of_unique_values: 4
          is_already_integerized: false
          items {
            key: "a"
            value { index: 0 count: 10 }
          }
          items {
            key: "b"
            value { index: 1 count: 9 }
          }
          items {
            key: "c"
            value { index: 2 count: 8 }
          }
          items {
            key: "d"
            value { index: 3 count: 7 }
          }
        }
      )");
  const auto representation =
      CategoricalIdxsToRepresentation(col_spec, elements, 2);
  CHECK_EQ(representation, "b, c, ...[1 left]");
}

TEST(DataSpecUtil, AddColumn) {
  proto::DataSpecification data_spec;
  AddColumn("a", proto::ColumnType::NUMERICAL, &data_spec);
  AddColumn("b", proto::ColumnType::CATEGORICAL, &data_spec);
  const proto::DataSpecification expected = PARSE_TEST_PROTO(
      R"(
        columns { type: NUMERICAL name: "a" }
        columns { type: CATEGORICAL name: "b" }
      )");
  EXPECT_THAT(data_spec, EqualsProto(expected));
}

// Conversion discretized numerical -> numerical.
TEST(DataSpec, DiscretizedNumericalToNumerical) {
  const float eps = 0.0001f;
  proto::Column col = PARSE_TEST_PROTO(
      R"(
        type: DISCRETIZED_NUMERICAL
        discretized_numerical { boundaries: 10 boundaries: 20 boundaries: 30 }
      )");

  EXPECT_TRUE(std::isnan(
      DiscretizedNumericalToNumerical(col, kDiscretizedNumericalMissingValue)));
  EXPECT_NEAR(DiscretizedNumericalToNumerical(col, 0), 9.f, eps);
  EXPECT_NEAR(DiscretizedNumericalToNumerical(col, 1), 15.f, eps);
  EXPECT_NEAR(DiscretizedNumericalToNumerical(col, 2), 25.f, eps);
  EXPECT_NEAR(DiscretizedNumericalToNumerical(col, 3), 31.f, eps);
}

// Conversion numerical -> discretized numerical.
TEST(DataSpec, NumericalToDiscretizedNumerical) {
  proto::Column col = PARSE_TEST_PROTO(
      R"(
        type: DISCRETIZED_NUMERICAL
        discretized_numerical { boundaries: 10 boundaries: 20 boundaries: 30 }
      )");

  EXPECT_EQ(NumericalToDiscretizedNumerical(col, -2.f), 0);
  EXPECT_EQ(NumericalToDiscretizedNumerical(col, -1.f), 0);

  EXPECT_EQ(NumericalToDiscretizedNumerical(col, 9.f), 0);
  EXPECT_EQ(NumericalToDiscretizedNumerical(col, 10.f), 1);
  EXPECT_EQ(NumericalToDiscretizedNumerical(col, 11.f), 1);

  EXPECT_EQ(NumericalToDiscretizedNumerical(col, 19.f), 1);
  EXPECT_EQ(NumericalToDiscretizedNumerical(col, 20.f), 2);
  EXPECT_EQ(NumericalToDiscretizedNumerical(col, 21.f), 2);

  EXPECT_EQ(NumericalToDiscretizedNumerical(col, 29.f), 2);
  EXPECT_EQ(NumericalToDiscretizedNumerical(col, 30.f), 3);
  EXPECT_EQ(NumericalToDiscretizedNumerical(col, 31.f), 3);
}

TEST(Dataset, GenDiscretizedBoundaries) {
  std::vector<std::pair<float, int>> interval_1_5;
  for (int i = 1; i <= 5; i++) {
    interval_1_5.emplace_back(i, 1);
  }

  std::vector<std::pair<float, int>> interval_1_100;
  for (int i = 1; i <= 100; i++) {
    interval_1_100.emplace_back(i, 1);
  }

  EXPECT_THAT(GenDiscretizedBoundaries(interval_1_5,
                                       /*maximum_num_bins=*/4,
                                       /*min_obs_in_bins=*/1, {}),
              ElementsAre(1.5, 2.5, 3.5));

  EXPECT_THAT(GenDiscretizedBoundaries(interval_1_5,
                                       /*maximum_num_bins=*/4,
                                       /*min_obs_in_bins=*/1, {}),
              ElementsAre(1.5, 2.5, 3.5));

  EXPECT_THAT(GenDiscretizedBoundaries(interval_1_100,
                                       /*maximum_num_bins=*/4,
                                       /*min_obs_in_bins=*/1, {}),
              ElementsAre(25.5, 50.5, 75.5));

  EXPECT_THAT(GenDiscretizedBoundaries(interval_1_5,
                                       /*maximum_num_bins=*/10,
                                       /*min_obs_in_bins=*/1, {}),
              ElementsAre(1.5, 2.5, 3.5, 4.5));

  EXPECT_THAT(
      GenDiscretizedBoundaries(interval_1_100,
                               /*maximum_num_bins=*/10, /*min_obs_in_bins=*/1,
                               {}),
      ElementsAre(10.5, 20.5, 30.5, 40.5, 50.5, 60.5, 70.5, 80.5, 90.5));

  EXPECT_THAT(GenDiscretizedBoundaries(interval_1_5,
                                       /*maximum_num_bins=*/1000,
                                       /*min_obs_in_bins=*/3, {}),
              ElementsAre(3.5));

  EXPECT_THAT(GenDiscretizedBoundaries(interval_1_100,
                                       /*maximum_num_bins=*/1000,
                                       /*min_obs_in_bins=*/15, {}),
              ElementsAre(15.5, 30.5, 45.5, 60.5, 75.5, 90.5));
}

TEST(Dataset, GenDiscretizedBoundariesCornerCases) {
  std::vector<std::pair<float, int>> interval_1_5;
  for (int i = 1; i <= 5; i++) {
    interval_1_5.emplace_back(i, 1);
  }

  EXPECT_THAT(GenDiscretizedBoundaries({},
                                       /*maximum_num_bins=*/10,
                                       /*min_obs_in_bins=*/1, {}),
              ElementsAre());

  EXPECT_THAT(GenDiscretizedBoundaries(interval_1_5,
                                       /*maximum_num_bins=*/1,
                                       /*min_obs_in_bins=*/1, {}),
              ElementsAre());

  EXPECT_THAT(GenDiscretizedBoundaries(interval_1_5,
                                       /*maximum_num_bins=*/1,
                                       /*min_obs_in_bins=*/1, {2.f}),
              ElementsAre(1.5, std::nextafter(2.f, 1.f),
                          std::nextafter(2.f, 3.f), 2.5, 3.5, 4.5));
}

TEST(Dataset, GenDiscretizedBoundariesWithSpecialValues) {
  std::vector<std::pair<float, int>> interval_1_5;
  for (int i = 1; i <= 5; i++) {
    interval_1_5.emplace_back(i, 1);
  }

  std::vector<std::pair<float, int>> interval_1_100;
  for (int i = 1; i <= 100; i++) {
    interval_1_100.emplace_back(i, 1);
  }

  EXPECT_THAT(GenDiscretizedBoundaries(interval_1_5,
                                       /*maximum_num_bins=*/4 + 1,
                                       /*min_obs_in_bins=*/1, {0.f}),
              ElementsAre(std::nextafter(0.f, 1.f), 1.5, 2.5, 3.5));

  EXPECT_THAT(GenDiscretizedBoundaries(interval_1_5,
                                       /*maximum_num_bins=*/4 + 2,
                                       /*min_obs_in_bins=*/1, {2.f}),
              ElementsAre(1.5, std::nextafter(2.f, 1.f),
                          std::nextafter(2.f, 3.f), 2.5, 3.5));

  EXPECT_THAT(GenDiscretizedBoundaries(interval_1_5,
                                       /*maximum_num_bins=*/4 + 2,
                                       /*min_obs_in_bins=*/1, {2.5f}),
              ElementsAre(1.5, std::nextafter(2.5f, 2.f),
                          std::nextafter(2.5f, 3.f), 3.5));

  EXPECT_THAT(GenDiscretizedBoundaries(interval_1_5,
                                       /*maximum_num_bins=*/4 + 1,
                                       /*min_obs_in_bins=*/1, {5.f}),
              ElementsAre(1.5, 2.5, 3.5, std::nextafter(5.f, 4.f)));
}

}  // namespace
}  // namespace dataset
}  // namespace yggdrasil_decision_forests
