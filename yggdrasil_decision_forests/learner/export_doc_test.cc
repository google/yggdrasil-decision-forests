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

#include "yggdrasil_decision_forests/learner/export_doc.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace {

class FakeLearner1 : public AbstractLearner {
 public:
  explicit FakeLearner1(const proto::TrainingConfig& training_config)
      : AbstractLearner(training_config) {}

  absl::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset = {}) const override {
    return std::unique_ptr<AbstractModel>();
  }

  absl::StatusOr<model::proto::GenericHyperParameterSpecification>
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

  absl::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
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

  absl::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
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

  const std::string content_path =
      file::JoinPath(test::TmpDirectory(), "content.csv");
  EXPECT_OK(file::SetContent(content_path, content));

  LOG(INFO) << "Exporting content to: " << content_path;

  EXPECT_EQ(content, R"(## FakeLearner2

<font size="2">

### Training configuration

Following are the protobuffer definitions used in TrainingConfiguration to set learner hyper-parameters.

- <a href="_learner/abstract_learner.proto">learner/abstract_learner.proto</a>

### Generic Hyper-parameters

#### [maximum_model_size_in_memory_in_bytes](_learner/abstract_learner.protomaximum_model_size_in_memory_in_bytes)

 - **Type:** Real **Default:** -1

 - Limit the size of the model when stored in ram. Different algorithms can enforce this limit differently. Note that when models are compiled into an inference, the size of the inference engine is generally much smaller than the original model.

#### [maximum_training_duration_seconds](_learner/abstract_learner.protomaximum_training_duration_seconds)

 - **Type:** Real **Default:** -1

 - Maximum training duration of the model expressed in seconds. Each learning algorithm is free to use this parameter at it sees fit. Enabling maximum training duration makes the model training non-deterministic.

#### [pure_serving_model](_learner/abstract_learner.protopure_serving_model)

 - **Type:** Categorical **Default:** false **Possible values:** true, false

 - Clear the model from any information that is not required for model serving. This includes debugging, model interpretation and other meta-data. The size of the serialized model can be reduced significatively (50% model size reduction is common). This parameter has no impact on the quality, serving speed or RAM usage of model serving.

#### [random_seed](_learner/abstract_learner.protorandom_seed)

 - **Type:** Integer **Default:** 123456

 - Random seed for the training of the model. Learners are expected to be deterministic by the random seed.

</font>


## FakeLearner3

<font size="2">

### Training configuration

Following are the protobuffer definitions used in TrainingConfiguration to set learner hyper-parameters.

- <a href="_learner/abstract_learner.proto">learner/abstract_learner.proto</a>

### Generic Hyper-parameters

#### [maximum_model_size_in_memory_in_bytes](_learner/abstract_learner.protomaximum_model_size_in_memory_in_bytes)

 - **Type:** Real **Default:** -1

 - Limit the size of the model when stored in ram. Different algorithms can enforce this limit differently. Note that when models are compiled into an inference, the size of the inference engine is generally much smaller than the original model.

#### [maximum_training_duration_seconds](_learner/abstract_learner.protomaximum_training_duration_seconds)

 - **Type:** Real **Default:** -1

 - Maximum training duration of the model expressed in seconds. Each learning algorithm is free to use this parameter at it sees fit. Enabling maximum training duration makes the model training non-deterministic.

#### [pure_serving_model](_learner/abstract_learner.protopure_serving_model)

 - **Type:** Categorical **Default:** false **Possible values:** true, false

 - Clear the model from any information that is not required for model serving. This includes debugging, model interpretation and other meta-data. The size of the serialized model can be reduced significatively (50% model size reduction is common). This parameter has no impact on the quality, serving speed or RAM usage of model serving.

#### [random_seed](_learner/abstract_learner.protorandom_seed)

 - **Type:** Integer **Default:** 123456

 - Random seed for the training of the model. Learners are expected to be deterministic by the random seed.

</font>


## FakeLearner1

<font size="2">

### Generic Hyper-parameters

#### a

 - **Type:** Real **Default:** 1 **Possible values:** min:2

 - b

</font>


)");
}

}  // namespace
}  // namespace model
}  // namespace yggdrasil_decision_forests
