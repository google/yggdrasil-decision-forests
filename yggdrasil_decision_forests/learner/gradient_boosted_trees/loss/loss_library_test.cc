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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_library.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

TEST(LossLibrary, CanonicalLosses) {
  proto::GradientBoostedTreesTrainingConfig config;
  dataset::proto::Column binary_categorical_label_column =
      PARSE_TEST_PROTO(R"pb(
        type: CATEGORICAL
        categorical { number_of_unique_values: 3 is_already_integerized: true }
      )pb");

  dataset::proto::Column numerical_label_column = PARSE_TEST_PROTO(R"pb(
    type: NUMERICAL
  )pb");
  dataset::proto::Column multiclass_categorical_label_column =
      PARSE_TEST_PROTO(R"pb(
        type: CATEGORICAL
        categorical { number_of_unique_values: 20 is_already_integerized: true }
      )pb");

  EXPECT_OK(CreateLoss(proto::Loss::BINOMIAL_LOG_LIKELIHOOD,
                       model::proto::Task::CLASSIFICATION,
                       binary_categorical_label_column, config));

  EXPECT_OK(CreateLoss(proto::Loss::SQUARED_ERROR,
                       model::proto::Task::REGRESSION, numerical_label_column,
                       config));

  EXPECT_OK(CreateLoss(proto::Loss::MULTINOMIAL_LOG_LIKELIHOOD,
                       model::proto::Task::CLASSIFICATION,
                       multiclass_categorical_label_column, config));

  EXPECT_OK(CreateLoss(proto::Loss::LAMBDA_MART_NDCG5,
                       model::proto::Task::RANKING, numerical_label_column,
                       config));

  EXPECT_OK(CreateLoss(proto::Loss::XE_NDCG_MART, model::proto::Task::RANKING,
                       numerical_label_column, config));

  EXPECT_OK(CreateLoss(proto::Loss::BINARY_FOCAL_LOSS,
                       model::proto::Task::CLASSIFICATION,
                       binary_categorical_label_column, config));
}

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
