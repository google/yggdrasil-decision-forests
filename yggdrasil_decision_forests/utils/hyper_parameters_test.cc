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

#include "yggdrasil_decision_forests/utils/hyper_parameters.h"


#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

using test::StatusIs;

TEST(HyperParameterConsumer, Base) {
  const model::proto::GenericHyperParameters hparams = PARSE_TEST_PROTO(
      R"(
        fields {
          name: "B"
          value { categorical: "TOTO" }
        }
      )");
  GenericHyperParameterConsumer consumer(hparams);
  EXPECT_FALSE(consumer.Get("A").has_value());
  EXPECT_FALSE(consumer.CheckThatAllHyperparametersAreConsumed().ok());
  EXPECT_TRUE(consumer.Get("B").has_value());
  EXPECT_TRUE(consumer.CheckThatAllHyperparametersAreConsumed().ok());
}

TEST(HyperParameterConsumer, SatisfyDefaultCondition) {
  EXPECT_THAT(
      SatisfyDefaultCondition({}, {}).status(),
      StatusIs(absl::StatusCode::kInvalidArgument, "Invalid condition"));

  EXPECT_TRUE(
      SatisfyDefaultCondition(
          PARSE_TEST_PROTO("categorical { default_value : \"A\"}"),
          PARSE_TEST_PROTO("categorical { values: \"A\" values:\"B\" }"))
          .value());

  EXPECT_FALSE(
      SatisfyDefaultCondition(
          PARSE_TEST_PROTO("categorical { default_value : \"C\"}"),
          PARSE_TEST_PROTO("categorical { values: \"A\" values:\"B\" }"))
          .value());

  EXPECT_THAT(
      SatisfyDefaultCondition(PARSE_TEST_PROTO("real { default_value : 1.0 }"),
                              PARSE_TEST_PROTO("categorical { values: \"A\"}"))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "The value is not categorical."));
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
