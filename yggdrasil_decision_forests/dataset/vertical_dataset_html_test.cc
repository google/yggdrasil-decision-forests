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

#include "yggdrasil_decision_forests/dataset/vertical_dataset_html.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_replace.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/html.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace dataset {
namespace {

std::string DatasetDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/"
                        "test_data/dataset");
}

TEST(VerticalDataset, AppendHtml) {
  const auto dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "toy.csv"));
  proto::DataSpecificationGuide guide;
  guide.mutable_default_column_guide()
      ->mutable_categorial()
      ->set_min_vocab_frequency(0);
  proto::DataSpecification data_spec;
  CreateDataSpec(dataset_path, false, guide, &data_spec);
  VerticalDataset dataset;
  EXPECT_OK(LoadVerticalDataset(dataset_path, data_spec, &dataset));

  std::string html;
  AppendVerticalDatasetToHtml(dataset, {}, &html);
  const std::string expected =
      R"(
<div class="table-container">
<table>
<tr>
<th align="right">Num_1</th>
<th align="right">Num_2</th>
<th align="left">Cat_1</th>
<th align="left">Cat_2</th>
<th align="left">Cat_set_1</th>
<th align="left">Cat_set_2</th>
<th align="left">Bool_1</th>
<th align="left">Bool_2</th>
<th align="right">Cat_3</th>
</tr><tr>
<td align="right">1</td>
<td align="right"></td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">x</td>
<td align="left">EMPTY</td>
<td align="left">0</td>
<td align="left">0</td>
<td align="right">1</td>
</tr><tr>
<td align="right">2</td>
<td align="right">2</td>
<td align="left">B</td>
<td align="left"></td>
<td align="left">x, y</td>
<td align="left">x</td>
<td align="left">1</td>
<td align="left"></td>
<td align="right">2</td>
</tr><tr>
<td align="right">3</td>
<td align="right"></td>
<td align="left">A</td>
<td align="left">B</td>
<td align="left">x, y, z</td>
<td align="left">x, y</td>
<td align="left">0</td>
<td align="left">1</td>
<td align="right">1</td>
</tr><tr>
<td align="right">4</td>
<td align="right">4</td>
<td align="left">C</td>
<td align="left"></td>
<td align="left">x, y, z</td>
<td align="left">x, y, z</td>
<td align="left">1</td>
<td align="left"></td>
<td align="right">3</td>
</tr>
</table>
</div>)";
  EXPECT_EQ(html, absl::StrReplaceAll(expected, {{"\n", ""}}));
}

}  // namespace
}  // namespace dataset
}  // namespace yggdrasil_decision_forests
