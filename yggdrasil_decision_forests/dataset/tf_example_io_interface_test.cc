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

#include <sys/types.h>

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/example/example.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace dataset {

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

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
