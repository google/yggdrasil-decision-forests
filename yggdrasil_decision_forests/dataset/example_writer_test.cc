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

#include "yggdrasil_decision_forests/dataset/example_writer.h"

#include <memory>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/example_writer_interface.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace dataset {

proto::DataSpecification CreateDataspec() {
  return PARSE_TEST_PROTO(
      R"pb(
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
      )pb");
}

proto::Example CreateExample() {
  return PARSE_TEST_PROTO(
      R"pb(
        attributes { numerical: 0.5 }
        attributes { numerical_set: { values: 0 values: 1 } }
        attributes { numerical_list: { values: 0 values: 1 } }
        attributes { categorical: 1 }
        attributes { categorical_set: { values: 0 values: 1 } }
        attributes { categorical_list: { values: 0 values: 1 } }
        attributes { boolean: 1 }
        attributes { text: "hello" }
      )pb");
}

TEST(CreateExampleWriter, Csv) {
  const proto::DataSpecification data_spec = CreateDataspec();
  const proto::Example example = CreateExample();

  const std::string output_dataset_path_csv =
      file::JoinPath(test::TmpDirectory(), "test.csv");

  {
    auto writer_or_status = CreateExampleWriter(
        absl::StrCat("csv:", output_dataset_path_csv), data_spec, -1);
    auto writer = std::move(writer_or_status.value());
    EXPECT_OK(writer->Write(example));
  }

  std::string content = file::GetContent(output_dataset_path_csv).value();
  EXPECT_EQ(content, "a,b,c,d,e,f,g,h\n0.5,0 1,0 1,1,0 1,0 1,1,hello\n");
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
