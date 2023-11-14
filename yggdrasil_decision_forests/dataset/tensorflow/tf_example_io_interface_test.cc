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

#include "yggdrasil_decision_forests/dataset/tensorflow/tf_example_io_interface.h"

#include <memory>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/tensorflow_no_dep/tf_example.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/example_writer.h"
#include "yggdrasil_decision_forests/dataset/example_writer_interface.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace dataset {

using test::EqualsProto;

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

std::string DatasetDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/"
                        "test_data/dataset");
}

std::string ToyDatasetTypedPathTFExampleTFRecord() {
  return absl::StrCat("tfrecord+tfe:",
                      file::JoinPath(DatasetDir(), "toy.tfe-tfrecord@2"));
}

TEST(DataSpecUtil, TFExampleReader) {
  for (const auto& dataset_path : {ToyDatasetTypedPathTFExampleTFRecord()}) {
    auto reader = CreateTFExampleReader(dataset_path).value();
    tensorflow::Example example;
    int num_rows = 0;
    while (reader->Next(&example).value()) {
      num_rows++;
    }
    EXPECT_EQ(num_rows, 4);
  }
}

TEST(CreateExampleWriter, TFRecord) {
  const proto::DataSpecification data_spec = CreateDataspec();
  const proto::Example example = CreateExample();

  const std::string typed_output_dataset_path_recordio_tfe = absl::StrCat(
      "tfrecord+tfe:", file::JoinPath(test::TmpDirectory(), "test.tfrecord"));
  {
    auto writer_or_status = CreateExampleWriter(
        typed_output_dataset_path_recordio_tfe, data_spec, -1);
    auto writer = std::move(writer_or_status.value());
    EXPECT_OK(writer->Write(example));
  }

  auto tfrecord_tfe_reader =
      CreateTFExampleReader(typed_output_dataset_path_recordio_tfe);
  tensorflow ::Example read_example;
  EXPECT_TRUE(tfrecord_tfe_reader.value()->Next(&read_example).value());
  const tensorflow::Example expected_read_example = PARSE_TEST_PROTO(
      R"pb(
        features {
          feature {
            key: "a"
            value { float_list { value: 0.5 } }
          }
          feature {
            key: "b"
            value { float_list { value: 0 value: 1 } }
          }
          feature {
            key: "c"
            value { float_list { value: 0 value: 1 } }
          }
          feature {
            key: "d"
            value { int64_list { value: 1 } }
          }
          feature {
            key: "e"
            value { int64_list { value: 0 value: 1 } }
          }
          feature {
            key: "f"
            value { int64_list { value: 0 value: 1 } }
          }
          feature {
            key: "g"
            value { float_list { value: 1 } }
          }
          feature {
            key: "h"
            value { bytes_list { value: "hello" } }
          }
        }
      )pb");
  EXPECT_THAT(read_example, EqualsProto(expected_read_example));
  EXPECT_FALSE(tfrecord_tfe_reader.value()->Next(&read_example).value());
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
