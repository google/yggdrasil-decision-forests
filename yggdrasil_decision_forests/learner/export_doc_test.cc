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

#include "yggdrasil_decision_forests/learner/export_doc.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace {

class FakeLearner1 : public AbstractLearner {
 public:
  explicit FakeLearner1(const proto::TrainingConfig& training_config)
      : AbstractLearner(training_config) {}

  utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset = {}) const override {
    return std::unique_ptr<AbstractModel>();
  }

  utils::StatusOr<model::proto::GenericHyperParameterSpecification>
  GetGenericHyperParameterSpecification() const override {
    model::proto::GenericHyperParameterSpecification spec;
    auto& a = (*spec.mutable_fields())["a"];
    a.mutable_real()->set_minimum(1);
    a.mutable_real()->set_minimum(2);
    a.mutable_real()->set_default_value(1);
    a.mutable_documentation()->set_description("b");
    a.mutable_documentation()->set_proto_field("c");
    return spec;
  }
};

REGISTER_AbstractLearner(FakeLearner1, "FakeLearner1");

class FakeLearner2 : public AbstractLearner {
 public:
  explicit FakeLearner2(const proto::TrainingConfig& training_config)
      : AbstractLearner(training_config) {}

  utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset = {}) const override {
    return std::unique_ptr<AbstractModel>();
  }
};

REGISTER_AbstractLearner(FakeLearner2, "FakeLearner2");

class FakeLearner3 : public AbstractLearner {
 public:
  explicit FakeLearner3(const proto::TrainingConfig& training_config)
      : AbstractLearner(training_config) {}

  utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset = {}) const override {
    return std::unique_ptr<AbstractModel>();
  }
};

REGISTER_AbstractLearner(FakeLearner3, "FakeLearner3");

TEST(ExportDoc, Base) {
  auto content =
      ExportSeveralLearnersToMarkdown(
          {"FakeLearner1", "FakeLearner2", "FakeLearner3"},
          [](absl::string_view a, absl::string_view b) -> std::string {
            return absl::StrCat("_", a, b);
          },
          {"FakeLearner2", "FakeLearner3"})
          .value();

  LOG(INFO) << content;

  EXPECT_EQ(content, R"(## FakeLearner2

No generic hyper-parameters. Use the <a href="_learner/fakelearner2.proto">FakeLearner2</a> TrainingConfig proto instead.

## FakeLearner3

No generic hyper-parameters. Use the <a href="_learner/fakelearner3.proto">FakeLearner3</a> TrainingConfig proto instead.

## FakeLearner1

<font size="2">

### Generic Hyper-parameters (compatible with TensorFlow Decision Forests)

#### a

 - **Type:** Real **Default:** 1 **Possible values:** min:2

 - b

</font>


)");
}

}  // namespace
}  // namespace model
}  // namespace yggdrasil_decision_forests
