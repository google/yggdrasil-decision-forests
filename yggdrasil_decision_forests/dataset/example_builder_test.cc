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

#include "yggdrasil_decision_forests/dataset/example_builder.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests::dataset {
namespace {
using test::EqualsProto;

TEST(ExampleProtoBuilder, Base) {
  proto::DataSpecification data_spec;
  AddNumericalColumn("a", &data_spec);
  AddColumn("b", proto::ColumnType::DISCRETIZED_NUMERICAL, &data_spec);
  AddColumn("c", proto::ColumnType::CATEGORICAL, &data_spec);
  AddCategoricalColumn("d", {"X", "Y", "Z"}, &data_spec);
  AddBooleanColumn("e", &data_spec);
  AddColumn("f", proto::ColumnType::CATEGORICAL_SET, &data_spec);
  AddCategoricalSetColumn("g", {"X", "Y", "Z"}, &data_spec);

  ExampleProtoBuilder builder(&data_spec);
  builder.SetNumericalValue(0, 1);
  builder.SetDiscretizedNumericalValue(1, 1);
  builder.SetCategoricalValue(2, 2);
  builder.SetCategoricalValue(3, "X");
  builder.SetBooleanValue(4, true);
  builder.SetCategoricalSetValue(5, {1, 2});
  builder.SetCategoricalSetValue(6, {"X", "Y"});

  const proto::Example expected = PARSE_TEST_PROTO(
      R"pb(
        attributes { numerical: 1 }
        attributes { discretized_numerical: 1 }
        attributes { categorical: 2 }
        attributes { categorical: 1 }
        attributes { boolean: true }
        attributes { categorical_set { values: 1 values: 2 } }
        attributes { categorical_set { values: 1 values: 2 } }
      )pb");
  EXPECT_THAT(builder.example(), EqualsProto(expected));
}

}  // namespace
}  // namespace yggdrasil_decision_forests::dataset
