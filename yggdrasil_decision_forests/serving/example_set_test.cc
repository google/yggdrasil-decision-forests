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

#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow/core/example/feature_util.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"

#include "yggdrasil_decision_forests/serving/example_set.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace {

using test::EqualsProto;
using test::StatusIs;

dataset::proto::DataSpecification ToyDataSpec() {
  return PARSE_TEST_PROTO(R"(
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
  )");
}

struct ToyModel : EmptyModel {
  // Skipping columns 7, 10 and 11 on purpose.
  ToyModel() {
    CHECK_OK(Initialize({0, 1, 2, 3, 4, 6, 8, 9, 12, 13, 14}, ToyDataSpec()));
  }
};

void SetToyValues(const ToyModel& model, ToyModel::ExampleSet* example_set,
                  const bool apply_fill_missing = true,
                  const bool apply_set_missing = true,
                  const bool apply_set_values = true) {
  const auto feature_a =
      ToyModel::ExampleSet::GetNumericalFeatureId("a", model).value();
  const auto feature_b =
      ToyModel::ExampleSet::GetCategoricalFeatureId("b", model).value();
  const auto feature_c =
      ToyModel::ExampleSet::GetCategoricalFeatureId("c", model).value();
  const auto feature_d =
      ToyModel::ExampleSet::GetCategoricalSetFeatureId("d", model).value();
  const auto feature_e =
      ToyModel::ExampleSet::GetCategoricalSetFeatureId("e", model).value();
  const auto feature_f =
      ToyModel::ExampleSet::GetNumericalFeatureId("f", model).value();
  const auto feature_g =
      ToyModel::ExampleSet::GetMultiDimNumericalFeatureId("g", model).value();
  const auto feature_i =
      ToyModel::ExampleSet::GetMultiDimNumericalFeatureId("i", model).value();
  const auto feature_j =
      ToyModel::ExampleSet::GetBooleanFeatureId("j", model).value();

  // Applies some "random" values to make sure the test does not depend on
  // undefined behavior (i.e. reading without setting first). For the tests to
  // work, these values should be different from the ones set in the
  // "apply_set_values" block.
  example_set->SetNumerical(0, feature_a, 1000.0f, model);
  example_set->SetCategorical(0, feature_b, 2, model);
  example_set->SetCategorical(0, feature_c, "x_c", model);
  example_set->SetCategoricalSet(0, feature_d, std::vector<int>{1}, model);
  example_set->SetCategoricalSet(0, feature_e, std::vector<std::string>{"x_d"},
                                 model);
  example_set->SetNumerical(1, feature_f, 100.f, model);
  CHECK_OK(
      example_set->SetMultiDimNumerical(0, feature_g, {-4, -5, -6}, model));
  CHECK_OK(example_set->SetMultiDimNumerical(0, feature_i, {0.5, 0.5}, model));
  example_set->SetBoolean(0, feature_j, false, model);

  if (apply_fill_missing) {
    example_set->FillMissing(model);
  }

  if (apply_set_missing) {
    example_set->SetMissingNumerical(0, feature_a, model);
    example_set->SetMissingCategorical(0, feature_b, model);
    example_set->SetMissingCategorical(0, feature_c, model);
    example_set->SetMissingCategoricalSet(0, feature_d, model);
    example_set->SetMissingCategoricalSet(0, feature_e, model);
    example_set->SetMissingNumerical(0, feature_f, model);
    example_set->SetMissingMultiDimNumerical(0, feature_g, model);
    example_set->SetMissingMultiDimNumerical(0, feature_i, model);
    example_set->SetMissingBoolean(0, feature_j, model);
  }

  if (apply_set_values) {
    example_set->SetNumerical(1, feature_a, 1.0f, model);
    example_set->SetCategorical(1, feature_b, 1, model);
    example_set->SetCategorical(1, feature_c, "y_c", model);
    example_set->SetCategoricalSet(1, feature_d, {2, 3}, model);
    example_set->SetCategoricalSet(
        1, feature_e, std::vector<std::string>{"y_d", "z_d"}, model);
    example_set->SetNumerical(1, feature_f, 1.5f, model);
    CHECK_OK(example_set->SetMultiDimNumerical(1, feature_g, {4, 5, 6}, model));
    CHECK_OK(
        example_set->SetMultiDimNumerical(1, feature_i, {1.5, 1.5}, model));
    example_set->SetBoolean(1, feature_j, true, model);
  }
}

TEST(ExampleSet, GetValue) {
  ToyModel model;
  ToyModel::ExampleSet example_set(5, model);
  SetToyValues(model, &example_set);

  const auto feature_a =
      ToyModel::ExampleSet::GetNumericalFeatureId("a", model).value();
  const auto feature_b =
      ToyModel::ExampleSet::GetCategoricalFeatureId("b", model).value();
  const auto feature_c =
      ToyModel::ExampleSet::GetCategoricalFeatureId("c", model).value();
  const auto feature_j =
      ToyModel::ExampleSet::GetBooleanFeatureId("j", model).value();

  const float kEpsilon = 0.0001;
  EXPECT_NEAR(example_set.GetNumerical(1, feature_a, model), 1.0f, kEpsilon);
  EXPECT_EQ(example_set.GetCategoricalInt(1, feature_b, model), 1);
  EXPECT_EQ(example_set.GetCategoricalString(1, feature_c, model), "y_c");
  EXPECT_EQ(example_set.GetBoolean(1, feature_j, model), true);
}

TEST(ExampleSet, HasFeature) {
  ToyModel model;
  ToyModel::ExampleSet example_set(5, model);

  EXPECT_TRUE(ToyModel::ExampleSet::HasInputFeature("a", model));
  EXPECT_TRUE(ToyModel::ExampleSet::HasInputFeature("b", model));
  EXPECT_TRUE(ToyModel::ExampleSet::HasInputFeature("c", model));
  EXPECT_TRUE(ToyModel::ExampleSet::HasInputFeature("d", model));
  EXPECT_FALSE(ToyModel::ExampleSet::HasInputFeature("toto", model));
  EXPECT_TRUE(ToyModel::ExampleSet::HasInputFeature("g", model));
  EXPECT_TRUE(ToyModel::ExampleSet::HasInputFeature("g_0", model));
  EXPECT_TRUE(ToyModel::ExampleSet::HasInputFeature("j", model));
}

TEST(ExampleSet, GetValueMissing) {
  ToyModel model;
  ToyModel::ExampleSet example_set(5, model);
  SetToyValues(model, &example_set);

  EXPECT_THAT(
      ToyModel::ExampleSet::GetNumericalFeatureId("UNUSED", model).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "but it is not used by the"));
  EXPECT_THAT(
      ToyModel::ExampleSet::GetNumericalFeatureId("Non existing feature", model)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument, "Unknown input feature"));
}

TEST(ExampleSet, ExtractProtoExampleMissing) {
  ToyModel model;
  ToyModel::ExampleSet example_set(5, model);
  example_set.FillMissing(model);
  const dataset::proto::Example expected_example = PARSE_TEST_PROTO(
      R"(
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
      )");
  EXPECT_THAT(example_set.ExtractProtoExample(0, model).value(),
              EqualsProto(expected_example));
  EXPECT_THAT(example_set.ExtractProtoExample(1, model).value(),
              EqualsProto(expected_example));
  EXPECT_THAT(example_set.ExtractProtoExample(2, model).value(),
              EqualsProto(expected_example));
}

TEST(ExampleSet, ExtractProtoExampleMissingManually) {
  ToyModel model;
  ToyModel::ExampleSet example_set(5, model);
  SetToyValues(model, &example_set, /*apply_fill_missing=*/false,
               /*apply_set_missing=*/true, /*apply_set_values=*/false);
  const dataset::proto::Example expected_example = PARSE_TEST_PROTO(
      R"(
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
      )");
  EXPECT_THAT(example_set.ExtractProtoExample(0, model).value(),
              EqualsProto(expected_example));
}

TEST(ExampleSet, ExtractProtoManualExample) {
  ToyModel model;
  ToyModel::ExampleSet example_set(5, model);
  SetToyValues(model, &example_set);

  const dataset::proto::Example expected_example_0 = PARSE_TEST_PROTO(
      R"(
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
      )");
  EXPECT_THAT(example_set.ExtractProtoExample(0, model).value(),
              EqualsProto(expected_example_0));

  const dataset::proto::Example expected_example_1 = PARSE_TEST_PROTO(
      R"(
        attributes { numerical: 1.0 }
        attributes { categorical: 1 }
        attributes { categorical: 1 }
        attributes { categorical_set { values: 2 values: 3 } }
        attributes { categorical_set { values: 1 values: 2 } }
        attributes {}
        attributes { discretized_numerical: 2 }
        attributes { numerical: 4 }
        attributes { numerical: 5 }
        attributes { numerical: 6 }
        attributes {}
        attributes {}
        attributes { discretized_numerical: 2 }
        attributes { discretized_numerical: 2 }
        attributes { boolean: true }
      )");
  EXPECT_THAT(example_set.ExtractProtoExample(1, model).value(),
              EqualsProto(expected_example_1));
}

TEST(ExampleSet, FromProtoExample) {
  ToyModel model;
  ToyModel::ExampleSet example_set(5, model);

  const dataset::proto::Example example_0 = PARSE_TEST_PROTO(
      R"(
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
      )");
  EXPECT_OK(example_set.FromProtoExample(example_0, 0, model));
  EXPECT_THAT(example_set.ExtractProtoExample(0, model).value(),
              EqualsProto(example_0));

  const dataset::proto::Example example_1 = PARSE_TEST_PROTO(
      R"(
        attributes { numerical: 1.0 }
        attributes { categorical: 1 }
        attributes { categorical: 1 }
        attributes { categorical_set { values: 2 values: 3 } }
        attributes { categorical_set { values: 1 values: 2 } }
        attributes {}
        attributes { discretized_numerical: 2 }
        attributes { numerical: 4 }
        attributes { numerical: 5 }
        attributes { numerical: 6 }
        attributes {}
        attributes {}
        attributes { discretized_numerical: 2 }
        attributes { discretized_numerical: 2 }
        attributes { boolean: true }
      )");

  EXPECT_OK(example_set.FromProtoExample(example_1, 1, model));
  EXPECT_THAT(example_set.ExtractProtoExample(1, model).value(),
              EqualsProto(example_1));
}

TEST(ExampleSet, FromTensorflowExample) {
  ToyModel model;
  ToyModel::ExampleSet example_set(5, model);

  tensorflow::Example example;
  tensorflow::SetFeatureValues({3.0f}, "a", &example);
  tensorflow::SetFeatureValues<int64_t>({1}, "b", &example);
  tensorflow::SetFeatureValues({"y_c"}, "c", &example);
  tensorflow::SetFeatureValues<int64_t>({2, 3}, "d", &example);
  tensorflow::SetFeatureValues({"y_d", "z_d"}, "e", &example);
  tensorflow::SetFeatureValues({1.9}, "f", &example);
  tensorflow::SetFeatureValues({10.f, 11.f, 12.f}, "g", &example);
  tensorflow::SetFeatureValues({1.5f, 1.5f}, "i", &example);
  tensorflow::SetFeatureValues({1.0f}, "j", &example);
  EXPECT_OK(example_set.FromTensorflowExample(example, 0, model));
  const dataset::proto::Example expected_example = PARSE_TEST_PROTO(
      R"(
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
      )");
  EXPECT_THAT(example_set.ExtractProtoExample(0, model).value(),
              EqualsProto(expected_example));

  tensorflow::Example example_1 = example;
  tensorflow::SetFeatureValues({1.0f}, "a", &example_1);
  EXPECT_OK(example_set.FromTensorflowExample(example_1, 1, model));
  const dataset::proto::Example expected_example_2 = PARSE_TEST_PROTO(
      R"(
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
      )");
  EXPECT_THAT(example_set.ExtractProtoExample(1, model).value(),
              EqualsProto(expected_example_2));

  tensorflow::Example example_2 = example;
  tensorflow::SetFeatureValues({1.0f, 2.0f}, "a", &example_2);
  EXPECT_THAT(example_set.FromTensorflowExample(example_2, 1, model),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Too many values for feature: a"));

  tensorflow::Example example_3 = example;
  tensorflow::SetFeatureValues<int64_t>({1, 2}, "b", &example_3);
  EXPECT_THAT(example_set.FromTensorflowExample(example_3, 1, model),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Too many values for feature: b"));

  tensorflow::Example example_4 = example;
  tensorflow::SetFeatureValues({"1.0f"}, "a", &example_4);
  EXPECT_THAT(example_set.FromTensorflowExample(example_4, 1, model),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Feature a is not numerical."));

  tensorflow::Example example_5 = example;
  tensorflow::SetFeatureValues({10.f, 11.f, 12.f, 13.f}, "g", &example_5);
  EXPECT_THAT(
      example_set.FromTensorflowExample(example_5, 1, model),
      StatusIs(absl::StatusCode::kInvalidArgument, "Wrong number of values."));
}

}  // namespace
}  // namespace serving
}  // namespace yggdrasil_decision_forests
