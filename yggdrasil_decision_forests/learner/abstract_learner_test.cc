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

#include <limits>
#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/weight.pb.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_testing.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

#include "yggdrasil_decision_forests/learner/abstract_learner.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace {

using test::StatusIs;
using ::testing::ElementsAre;

TEST(AbstractModel, LinkTrainingConfig) {
  proto::TrainingConfig training_config;
  training_config.set_label("A");
  training_config.add_features(".*");
  training_config.mutable_weight_definition()->set_attribute("C");
  training_config.mutable_weight_definition()->mutable_categorical();

  dataset::proto::DataSpecification data_spec;
  data_spec.add_columns()->set_name("A");
  data_spec.add_columns()->set_name("B");
  data_spec.add_columns()->set_name("C");
  data_spec.add_columns()->set_name("D");

  proto::TrainingConfigLinking config_link;
  CHECK_OK(AbstractLearner::LinkTrainingConfig(training_config, data_spec,
                                               &config_link));

  EXPECT_EQ(config_link.label(), 0);
  EXPECT_THAT(config_link.features(), ElementsAre(1, 3));
  EXPECT_EQ(config_link.weight_definition().attribute_idx(), 2);
}

TEST(AbstractModel, LinkTrainingConfigNoInputFeatures) {
  proto::TrainingConfig training_config;
  training_config.set_label("A");

  dataset::proto::DataSpecification data_spec;
  data_spec.add_columns()->set_name("A");
  data_spec.add_columns()->set_name("B");
  data_spec.add_columns()->set_name("C");
  data_spec.add_columns()->set_name("D");

  proto::TrainingConfigLinking config_link;
  CHECK_OK(AbstractLearner::LinkTrainingConfig(training_config, data_spec,
                                               &config_link));

  EXPECT_EQ(config_link.label(), 0);
  EXPECT_THAT(config_link.features(), ElementsAre(1, 2, 3));
}

TEST(AbstractModel, LinkTrainingConfigFullyMissingFeatures) {
  proto::TrainingConfig training_config;
  training_config.set_label("A");

  dataset::proto::DataSpecification data_spec;
  data_spec.add_columns()->set_name("A");
  data_spec.add_columns()->set_name("B");
  data_spec.add_columns()->set_name("C");
  data_spec.add_columns()->set_name("D");

  data_spec.set_created_num_rows(10);
  // "B" and "C" only have missing values.
  data_spec.mutable_columns(1 /*B*/)->set_count_nas(10);
  data_spec.mutable_columns(2 /*C*/)->mutable_numerical()->set_mean(
      std::numeric_limits<float>::quiet_NaN());

  proto::TrainingConfigLinking config_link;
  CHECK_OK(AbstractLearner::LinkTrainingConfig(training_config, data_spec,
                                               &config_link));

  EXPECT_EQ(config_link.label(), 0);
  EXPECT_THAT(config_link.features(), ElementsAre(3));
}

TEST(AbstractLearner, GenericHyperParameters) {
  const model::proto::GenericHyperParameterSpecification hparam_spec =
      PARSE_TEST_PROTO(R"pb(
        fields {
          key: "num_param"
          value { integer { minimum: 1 } }
        }
        fields {
          key: "cat_param"
          value { categorical { possible_values: "cat_1" } }
        }
      )pb");

  EXPECT_OK(CheckGenericHyperParameterSpecification(
      model::proto::GenericHyperParameters(), hparam_spec));

  EXPECT_OK(CheckGenericHyperParameterSpecification(
      PARSE_TEST_PROTO("fields { name: \"num_param\" value { integer: 10 } }"),
      hparam_spec));

  EXPECT_THAT(CheckGenericHyperParameterSpecification(
                  PARSE_TEST_PROTO(
                      "fields { name: \"num_param\" value {  integer: -1 } }"),
                  hparam_spec),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(CheckGenericHyperParameterSpecification(
                  PARSE_TEST_PROTO(
                      "fields { name: \"num_param\" value {  integer: 10 } }"
                      " fields { name: \"num_param\" value {  integer: 10 } }"),
                  hparam_spec),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(CheckGenericHyperParameterSpecification(
                  PARSE_TEST_PROTO("fields { name: \"num_param\" value {  "
                                   "categorical: \"wrong_type\" } }"),
                  hparam_spec),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(CheckGenericHyperParameterSpecification(
                  PARSE_TEST_PROTO("fields { name: \"non_existing_h_param\" "
                                   "value {  integer: 10 } }"),
                  hparam_spec),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(CheckGenericHyperParameterSpecification(
                  PARSE_TEST_PROTO(
                      "fields { name: \"cat_param\" value {  integer: 10  } }"),
                  hparam_spec),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_OK(CheckGenericHyperParameterSpecification(
      PARSE_TEST_PROTO(
          "fields { name: \"cat_param\" value {  categorical: \"cat_1\" } }"),
      hparam_spec));

  EXPECT_THAT(
      CheckGenericHyperParameterSpecification(
          PARSE_TEST_PROTO("fields { name: \"cat_param\" value { categorical: "
                           "\"non_existing_cat\" } }"),
          hparam_spec),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

// Creates a classification model always returning class "1". Create a dataset
// with 4 x 1000 observations: 2 labels of class "2" and two labels of class "1"
// x 1000. Run a 10 fold cross-validation.
TEST(AbstractLearner, EvaluateLearner) {
  class FakeClassificationModel : public FakeModel {
   public:
    FakeClassificationModel() : FakeModel() {}

    void Predict(const dataset::VerticalDataset& dataset,
                 dataset::VerticalDataset::row_t row_idx,
                 model::proto::Prediction* prediction) const override {
      *prediction = PARSE_TEST_PROTO(R"pb(
        classification {
          value: 1
          distribution { counts: 0 counts: 1 counts: 0 sum: 1 }
        }
      )pb");
    }
  };

  class FakeLearner : public AbstractLearner {
   public:
    explicit FakeLearner(const proto::TrainingConfig& training_config)
        : AbstractLearner(training_config) {}

    utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
        const dataset::VerticalDataset& train_dataset,
        absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
            valid_dataset = {}) const override {
      auto model = absl::make_unique<FakeClassificationModel>();
      model::proto::TrainingConfigLinking config_link;
      CHECK_OK(AbstractLearner::LinkTrainingConfig(
          training_config(), train_dataset.data_spec(), &config_link));
      InitializeModelWithAbstractTrainingConfig(training_config(), config_link,
                                                model.get());
      return model;
    }
  };

  const proto::TrainingConfig train_config = PARSE_TEST_PROTO(R"pb(
    label: "a"
    task: CLASSIFICATION
  )pb");
  FakeLearner learner(train_config);

  const dataset::proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"pb(
        columns {
          type: CATEGORICAL
          name: "a"
          categorical {
            is_already_integerized: true
            number_of_unique_values: 3
          }
        }
      )pb");

  dataset::VerticalDataset dataset;
  dataset.set_data_spec(data_spec);
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  for (int i = 0; i < 1000; i++) {
    dataset.AppendExample({{"a", "1"}});
    dataset.AppendExample({{"a", "2"}});
    dataset.AppendExample({{"a", "1"}});
    dataset.AppendExample({{"a", "2"}});
  }

  const metric::proto::EvaluationOptions evaluation_options =
      PARSE_TEST_PROTO(R"pb(
        task: CLASSIFICATION
      )pb");
  const utils::proto::FoldGenerator fold_generator = PARSE_TEST_PROTO(R"pb(
    cross_validation { num_folds: 10 }
  )pb");
  const auto eval =
      EvaluateLearner(learner, dataset, fold_generator, evaluation_options);

  EXPECT_NEAR(metric::Accuracy(eval), 0.5f, 0.001f);
  EXPECT_NEAR(eval.count_predictions(), 4000., 0.001);
}

TEST(AbstractLearner, MaximumModelSizeInMemoryInBytes) {
  class FakeLearner : public AbstractLearner {
   public:
    explicit FakeLearner(const proto::TrainingConfig& training_config)
        : AbstractLearner(training_config) {}

    utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
        const dataset::VerticalDataset& train_dataset,
        absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
            valid_dataset = {}) const override {
      return absl::UnimplementedError("");
    }

    void set_tested_capability(bool value) {
      capabilities_.set_support_max_model_size_in_memory(value);
    }

    model::proto::LearnerCapabilities Capabilities() const override {
      return capabilities_;
    }

   private:
    model::proto::LearnerCapabilities capabilities_;
  };

  const proto::TrainingConfig train_config =
      PARSE_TEST_PROTO(R"pb(
        label: "a" task: CLASSIFICATION learner: "fake"
      )pb");
  FakeLearner learner(train_config);

  // Set the hparam without actual support for it.
  EXPECT_OK(learner.SetHyperParameters(PARSE_TEST_PROTO(
      "fields { name: \"maximum_model_size_in_memory_in_bytes\" "
      "value { real: "
      "500 } }")));
  EXPECT_THAT(learner.CheckCapabilities(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "does not support the "
                       "\"maximum_model_size_in_memory_in_bytes\" flag"));

  // Add support for the hparam.
  learner.set_tested_capability(true);
  EXPECT_OK(learner.CheckCapabilities());

  // Check the hparam value.
  EXPECT_EQ(learner.training_config().maximum_model_size_in_memory_in_bytes(),
            500);

  // Remove the hparam value i.e. return to the default logic.
  EXPECT_OK(learner.SetHyperParameters(PARSE_TEST_PROTO(
      "fields { name: \"maximum_model_size_in_memory_in_bytes\" "
      "value { real: "
      "-1 } }")));
  EXPECT_FALSE(
      learner.training_config().has_maximum_model_size_in_memory_in_bytes());
}

}  // namespace
}  // namespace model
}  // namespace yggdrasil_decision_forests
