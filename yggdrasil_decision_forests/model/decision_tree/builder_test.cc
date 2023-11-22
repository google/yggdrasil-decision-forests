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

#include "yggdrasil_decision_forests/model/decision_tree/builder.h"

#include <string>

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {
namespace {

TEST(TreeBuilder, Base) {
  DecisionTree tree;
  TreeBuilder builder(&tree);

  dataset::proto::DataSpecification dataspec;
  dataset::AddColumn("l", dataset::proto::ColumnType::NUMERICAL, &dataspec);
  dataset::AddColumn("f1", dataset::proto::ColumnType::NUMERICAL, &dataspec);
  auto* f2 = dataset::AddColumn("f2", dataset::proto::ColumnType::CATEGORICAL,
                                &dataspec);

  f2->mutable_categorical()->set_number_of_unique_values(5);
  f2->mutable_categorical()->set_is_already_integerized(true);

  auto [pos, l1] = builder.ConditionIsGreater(1, 1);
  auto [l2, l3] = pos.ConditionContains(2, {1, 2, 3});
  l1.LeafRegression(1);
  l2.LeafRegression(2);
  l3.LeafRegression(3);

  std::string description;
  tree.AppendModelStructure(dataspec, 0, &description);
  test::ExpectEqualGolden(description,
                          "yggdrasil_decision_forests/test_data/"
                          "golden/build_decision_tree.txt.expected");
}

}  // namespace
}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
