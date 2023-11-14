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

#include "yggdrasil_decision_forests/serving/tf_example.h"

#include <stdint.h>

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/tensorflow_no_dep/tf_example.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace {

using test::EqualsProto;
using test::StatusIs;

dataset::proto::DataSpecification ToyDataSpec() {
  return PARSE_TEST_PROTO(R"pb(
    # Id:0
    columns {
      type: NUMERICAL
      name: "a"
      numerical { mean: -1 }
    }

    # Id:1
    columns {
      type: CATEGORICAL
      name: "b"
      categorical {
        is_already_integerized: true
        number_of_unique_values: 3
        most_frequent_value: 0
      }
    }

    # Id:2
    columns {
      type: CATEGORICAL
      name: "c"
      categorical {
        is_already_integerized: false
        number_of_unique_values: 3
        items {
          key: "x_c"
          value { index: 0 }
        }
        items {
          key: "y_c"
          value { index: 1 }
        }
        items {
          key: "z_c"
          value { index: 2 }
        }
      }
    }

    # Id:3
    columns {
      type: CATEGORICAL_SET
      name: "d"
      categorical { is_already_integerized: true number_of_unique_values: 5 }
    }

    # Id:4
    columns {
      type: CATEGORICAL_SET
      name: "e"
      categorical {
        is_already_integerized: false
        number_of_unique_values: 3
        items {
          key: "x_d"
          value { index: 0 }
        }
        items {
          key: "y_d"
          value { index: 1 }
        }
        items {
          key: "z_d"
          value { index: 2 }
        }
      }
    }

    # Id:5
    columns { type: NUMERICAL name: "UNUSED" }

    # Id:6
    columns {
      type: DISCRETIZED_NUMERICAL
      name: "f"
      numerical { mean: -1 }
      discretized_numerical {
        boundaries: 0
        boundaries: 1
        boundaries: 2
        boundaries: 3
      }
    }

    # Id:7
    columns {
      type: NUMERICAL
      name: "g_0"
      numerical { mean: 0 }
      is_unstacked: true
    }

    # Id:8
    columns {
      type: NUMERICAL
      name: "g_1"
      numerical { mean: 1 }
      is_unstacked: true
    }

    # Id:9
    columns {
      type: NUMERICAL
      name: "g_2"
      numerical { mean: 2 }
      is_unstacked: true
    }

    # Id:10
    columns {
      type: NUMERICAL
      name: "h_0"
      numerical { mean: 0 }
      is_unstacked: true
    }

    # Id:11
    columns {
      type: NUMERICAL
      name: "h_1"
      numerical { mean: 1 }
      is_unstacked: true
    }

    # Id: 12
    columns {
      type: DISCRETIZED_NUMERICAL
      name: "i_0"
      discretized_numerical { boundaries: 0 boundaries: 1 boundaries: 2 }
      is_unstacked: true
    }

    # Id: 13
    columns {
      type: DISCRETIZED_NUMERICAL
      name: "i_1"
      discretized_numerical { boundaries: 0 boundaries: 1 boundaries: 2 }
      is_unstacked: true
    }

    # Id:14
    columns {
      type: BOOLEAN
      name: "j"
      boolean { count_true: 5 count_false: 10 }
    }

    unstackeds {
      original_name: "g"
      begin_column_idx: 7
      size: 3
      type: NUMERICAL
    }
    unstackeds {
      original_name: "h"
      begin_column_idx: 10
      size: 2
      type: NUMERICAL
    }
    unstackeds {
      original_name: "i"
      begin_column_idx: 12
      size: 2
      type: DISCRETIZED_NUMERICAL
    }
  )pb");
}

struct ToyModel : serving::EmptyModel {
  // Skipping columns 7, 10 and 11 on purpose.
  ToyModel() {
    CHECK_OK(Initialize({0, 1, 2, 3, 4, 6, 8, 9, 12, 13, 14}, ToyDataSpec()));
  }
};

TEST(ExampleSet, FromTensorflowExample) {
  ToyModel model;
  ToyModel::ExampleSet example_set(5, model);

  tensorflow::Example example;
  tensorflow::SetFeatureValues({3.0f}, "a", &example);
  tensorflow::SetFeatureValues({1}, "b", &example);
  tensorflow::SetFeatureValues({"y_c"}, "c", &example);
  tensorflow::SetFeatureValues({2, 3}, "d", &example);
  tensorflow::SetFeatureValues({"y_d", "z_d"}, "e", &example);
  tensorflow::SetFeatureValues({1.9}, "f", &example);
  tensorflow::SetFeatureValues({10.f, 11.f, 12.f}, "g", &example);
  tensorflow::SetFeatureValues({1.5f, 1.5f}, "i", &example);
  tensorflow::SetFeatureValues({1.0f}, "j", &example);
  EXPECT_OK(TfExampleToExampleSet(example, 0, model.features(), &example_set));
  const dataset::proto::Example expected_example = PARSE_TEST_PROTO(
      R"pb(
        attributes { numerical: 3.0 }
        attributes { categorical: 1 }
        attributes { categorical: 1 }
        attributes { categorical_set { values: 2 values: 3 } }
        attributes { categorical_set { values: 1 values: 2 } }
        attributes {}
        attributes { discretized_numerical: 2 }
        attributes { numerical: 10 }
        attributes { numerical: 11 }
        attributes { numerical: 12 }
        attributes {}
        attributes {}
        attributes { discretized_numerical: 2 }
        attributes { discretized_numerical: 2 }
        attributes { boolean: true }
      )pb");
  EXPECT_THAT(example_set.ExtractProtoExample(0, model).value(),
              EqualsProto(expected_example));

  tensorflow::Example example_1 = example;
  tensorflow::SetFeatureValues({1.0f}, "a", &example_1);
  EXPECT_OK(
      TfExampleToExampleSet(example_1, 1, model.features(), &example_set));
  const dataset::proto::Example expected_example_2 = PARSE_TEST_PROTO(
      R"pb(
        attributes { numerical: 1.0 }
        attributes { categorical: 1 }
        attributes { categorical: 1 }
        attributes { categorical_set { values: 2 values: 3 } }
        attributes { categorical_set { values: 1 values: 2 } }
        attributes {}
        attributes { discretized_numerical: 2 }
        attributes { numerical: 10 }
        attributes { numerical: 11 }
        attributes { numerical: 12 }
        attributes {}
        attributes {}
        attributes { discretized_numerical: 2 }
        attributes { discretized_numerical: 2 }
        attributes { boolean: true }
      )pb");
  EXPECT_THAT(example_set.ExtractProtoExample(1, model).value(),
              EqualsProto(expected_example_2));

  tensorflow::Example example_2 = example;
  tensorflow::SetFeatureValues({1.0f, 2.0f}, "a", &example_2);
  EXPECT_THAT(
      TfExampleToExampleSet(example_2, 1, model.features(), &example_set),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Too many values for feature: a"));

  tensorflow::Example example_3 = example;
  tensorflow::SetFeatureValues({1, 2}, "b", &example_3);
  EXPECT_THAT(
      TfExampleToExampleSet(example_3, 1, model.features(), &example_set),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Too many values for feature: b"));

  tensorflow::Example example_4 = example;
  tensorflow::SetFeatureValues({"1.0f"}, "a", &example_4);
  EXPECT_THAT(
      TfExampleToExampleSet(example_4, 1, model.features(), &example_set),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Feature a is not numerical."));

  tensorflow::Example example_5 = example;
  tensorflow::SetFeatureValues({10.f, 11.f, 12.f, 13.f}, "g", &example_5);
  EXPECT_THAT(
      TfExampleToExampleSet(example_5, 1, model.features(), &example_set),
      StatusIs(absl::StatusCode::kInvalidArgument, "Wrong number of values."));
}

}  // namespace
}  // namespace serving
}  // namespace yggdrasil_decision_forests
