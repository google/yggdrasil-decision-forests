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

#include "yggdrasil_decision_forests/dataset/formats.h"

#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/formats.pb.h"

namespace yggdrasil_decision_forests {
namespace dataset {
namespace {

TEST(Dataset, GetDatasetPathAndType) {
  EXPECT_EQ(GetDatasetPathAndType("csv:dataset.csv"),
            (std::pair<std::string, proto::DatasetFormat>(
                "dataset.csv", proto::DatasetFormat::FORMAT_CSV)));
  EXPECT_EQ(GetDatasetPathAndType("csv:/dataset.csv"),
            (std::pair<std::string, proto::DatasetFormat>(
                "/dataset.csv", proto::DatasetFormat::FORMAT_CSV)));
  EXPECT_EQ(GetDatasetPathAndType("tfrecord+tfe:dataset.rio_tfe"),
            (std::pair<std::string, proto::DatasetFormat>(
                "dataset.rio_tfe", proto::DatasetFormat::FORMAT_TFE_TFRECORD)));

  EXPECT_EQ(GetDatasetPathAndType("csv:/tmp/dataset.csv"),
            (std::pair<std::string, proto::DatasetFormat>(
                "/tmp/dataset.csv", proto::DatasetFormat::FORMAT_CSV)));
}

}  // namespace
}  // namespace dataset
}  // namespace yggdrasil_decision_forests
