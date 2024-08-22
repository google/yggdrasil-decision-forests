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

#include "yggdrasil_decision_forests/learner/decision_tree/preprocessing.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {
namespace {

using ::yggdrasil_decision_forests::dataset::proto::ColumnType;

TEST(Preprocessing, Base) {
  dataset::VerticalDataset dataset;
  ASSERT_OK(dataset.AddColumn("l", ColumnType::NUMERICAL).status());
  ASSERT_OK(dataset.AddColumn("f1", ColumnType::NUMERICAL).status());
  ASSERT_OK(dataset.AddColumn("f2", ColumnType::NUMERICAL).status());
  ASSERT_OK(dataset.CreateColumnsFromDataspec());
  dataset.AppendExample({{"l", "0"}, {"f1", "1"}, {"f2", "1"}});
  dataset.AppendExample({{"l", "0"}, {"f1", "3"}, {"f2", "2"}});
  dataset.AppendExample({{"l", "1"}, {"f1", "3"}, {"f2", "1"}});
  dataset.AppendExample({{"l", "1"}, {"f1", "2"}, {"f2", "1"}});
  dataset.AppendExample({{"l", "1.5"}, {"f1", "1"}, {"f2", "2"}});

  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  proto::DecisionTreeTrainingConfig dt_config;

  config_link.set_label(0);
  config_link.add_features(1);
  config_link.add_features(2);

  ASSERT_OK_AND_ASSIGN(const auto preprocessing,
                       decision_tree::PreprocessTrainingDataset(
                           dataset, config, config_link, dt_config, 1));

  EXPECT_EQ(preprocessing.num_examples(), 5);
  ASSERT_EQ(preprocessing.presorted_numerical_features().size(), 3);

  EXPECT_EQ(preprocessing.presorted_numerical_features()[0].items.size(), 0);
  EXPECT_EQ(preprocessing.presorted_numerical_features()[1].items.size(), 5);
  EXPECT_EQ(preprocessing.presorted_numerical_features()[2].items.size(), 5);
}

}  // namespace
}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
