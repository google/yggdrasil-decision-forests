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

#include "yggdrasil_decision_forests/dataset/tensorflow/tf_example.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/tensorflow_no_dep/tf_example.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace dataset {
namespace {

using test::EqualsProto;
using test::StatusIs;

TEST(Dataset, TfExampleToYdfExample) {
  const proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"pb(
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
      )pb");
  tensorflow::Example tf_example = PARSE_TEST_PROTO(
      R"pb(
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
      )pb");
  proto::Example example;
  EXPECT_OK(TfExampleToYdfExample(tf_example, data_spec, &example));
  const proto::Example expected_example = PARSE_TEST_PROTO(
      R"pb(
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
      )pb");
  EXPECT_THAT(example, EqualsProto(expected_example));

  tensorflow::Example convert_back_tf_example;
  EXPECT_OK(
      YdfExampleToTfExample(example, data_spec, &convert_back_tf_example));
  // The original int64_t is stored in a float.
  auto* values = (*tf_example.mutable_features()->mutable_feature())["k"]
                     .mutable_float_list()
                     ->mutable_value();
  values->Add(20.f);
  values->Add(21.f);
  values->Add(22.f);
  EXPECT_THAT(convert_back_tf_example, EqualsProto(tf_example));
}

TEST(Dataset, TfExampleToYdfExampleErrors) {
  const proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"pb(
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
      )pb");

  tensorflow::Example example_1 = PARSE_TEST_PROTO(
      R"pb(
        features {
          feature {
            key: "a"
            value { float_list { value: 1.0 value: 2.0 } }
          }
        }
      )pb");
  proto::Example example;
  EXPECT_THAT(TfExampleToYdfExample(example_1, data_spec, &example),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "a's type is not supported for stacked feature."));

  tensorflow::Example example_2 = PARSE_TEST_PROTO(
      R"pb(
        features {
          feature {
            key: "b"
            value { float_list { value: 1.0 value: 2.0 value: 3.0 } }
          }
        }
      )pb");
  EXPECT_THAT(TfExampleToYdfExample(example_2, data_spec, &example),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Wrong number of elements for feature b"));

  tensorflow::Example example_3 = PARSE_TEST_PROTO(
      R"pb(
        features {
          feature {
            key: "b"
            value { bytes_list { value: "x" } }
          }
        }
      )pb");
  EXPECT_THAT(TfExampleToYdfExample(example_3, data_spec, &example),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Feature b is not stored as float or int64."));
}

}  // namespace
}  // namespace dataset
}  // namespace yggdrasil_decision_forests
