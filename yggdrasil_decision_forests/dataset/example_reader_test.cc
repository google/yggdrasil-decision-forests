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

#include "yggdrasil_decision_forests/dataset/example_reader.h"

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/example_reader_interface.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace dataset {

std::string DatasetDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/"
                        "test_data/dataset");
}

TEST(ExampleReader, CreateExampleReader) {
  for (const auto& dataset_path :
       {absl::StrCat("tfrecord+tfe:",
                     file::JoinPath(DatasetDir(), "toy.tfe-tfrecord@2")),
        absl::StrCat("csv:", file::JoinPath(DatasetDir(), "toy.csv"))}) {
    LOG(INFO) << "Create dataspec for " << dataset_path;
    proto::DataSpecificationGuide guide;
    proto::DataSpecification data_spec;
    CreateDataSpec(dataset_path, false, guide, &data_spec);
    LOG(INFO) << "Scan " << dataset_path;
    auto reader = CreateExampleReader(dataset_path, data_spec).value();
    proto::Example example;
    int num_rows = 0;
    int sum_column_num_1 = 0;
    int sum_column_num_2 = 0;
    int num_na_column_num_2 = 0;
    int num_na_column_bool_2 = 0;
    while (reader->Next(&example).value()) {
      if (absl::StartsWith(dataset_path, "csv:")) {
        sum_column_num_1 +=
            example.attributes(GetColumnIdxFromName("Num_1", data_spec))
                .numerical();
        sum_column_num_2 +=
            example.attributes(GetColumnIdxFromName("Num_2", data_spec))
                .numerical();
        num_na_column_num_2 +=
            IsNa(example.attributes(GetColumnIdxFromName("Num_2", data_spec)));
        num_na_column_bool_2 +=
            IsNa(example.attributes(GetColumnIdxFromName("Bool_2", data_spec)));
      }
      num_rows++;
    }

    if (absl::StartsWith(dataset_path, "csv:")) {
      EXPECT_EQ(sum_column_num_1, 1 + 2 + 3 + 4);
      EXPECT_EQ(sum_column_num_2, 2 + 4);
      EXPECT_EQ(num_na_column_num_2, 2);
      EXPECT_EQ(num_na_column_bool_2, 2);
    }

    EXPECT_EQ(num_rows, 4);
  }
}

TEST(ExampleReader, IsFormatSupported) {
  EXPECT_TRUE(IsFormatSupported("csv:/path/to/ds").value());
  EXPECT_FALSE(IsFormatSupported("capacitor:/path/to/ds").value());
  EXPECT_FALSE(IsFormatSupported("non-existing-format:/path/to/ds").value());
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
