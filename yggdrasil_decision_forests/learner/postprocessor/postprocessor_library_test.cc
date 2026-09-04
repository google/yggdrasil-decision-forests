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

#include "yggdrasil_decision_forests/learner/postprocessor/postprocessor_library.h"

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/postprocessor/abstract_postprocessor.pb.h"
#include "yggdrasil_decision_forests/model/model_testing.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests::model::postprocessor {
namespace {

class FakeClassificationModel : public model::FakeModel {
 public:
  FakeClassificationModel(const std::vector<float>& positive_probabilities)
      : model::FakeModel(), positive_probabilities_(positive_probabilities) {}

 protected:
  void PredictImpl(const dataset::VerticalDataset& dataset,
                   dataset::VerticalDataset::row_t row_idx,
                   model::proto::Prediction* prediction) const override {
    CHECK_LT(row_idx, positive_probabilities_.size());
    float p = positive_probabilities_[row_idx];
    prediction->Clear();
    auto* classification = prediction->mutable_classification();
    auto* distribution = classification->mutable_distribution();
    distribution->set_sum(1.0f);
    distribution->add_counts(0.0f);      // OOV
    distribution->add_counts(1.0f - p);  // Negative (index 1)
    distribution->add_counts(p);         // Positive (index 2)
    classification->set_value(p >= 0.5f ? 2 : 1);
  }

 private:
  std::vector<float> positive_probabilities_;
};

dataset::VerticalDataset CreateDataset(const std::vector<int>& labels) {
  dataset::proto::DataSpecification data_spec;
  auto* col = data_spec.add_columns();
  col->set_name("label");
  col->set_type(dataset::proto::ColumnType::CATEGORICAL);
  auto* cat = col->mutable_categorical();
  cat->set_is_already_integerized(true);
  cat->set_number_of_unique_values(3);  // 0 (OOV), 1 (neg), 2 (pos)

  dataset::VerticalDataset dataset;
  dataset.set_data_spec(data_spec);
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  for (int label : labels) {
    CHECK_OK(
        dataset.AppendExampleWithStatus({{"label", std::to_string(label)}}));
  }
  return dataset;
}

TEST(PostprocessorLibraryTest, CreateSmoothedPavCalibrator) {
  std::vector<int> labels = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
  std::vector<float> predictions = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
                                    0.8f, 0.8f, 0.8f, 0.8f, 0.8f};

  auto dataset = CreateDataset(labels);
  FakeClassificationModel model(predictions);

  model::proto::DeploymentConfig deployment;
  deployment.set_num_threads(1);

  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(0);

  proto::AbstractPostprocessorTrainingConfig postprocessor_config;
  // Initialize the oneof
  postprocessor_config.mutable_smoothed_pav_calibrator_training_config();

  auto postprocessor_or = CreatePostprocessor(
      deployment, config_link, postprocessor_config, model, dataset);

  ASSERT_OK(postprocessor_or.status());
  auto postprocessor = postprocessor_or.value();
  ASSERT_NE(postprocessor, nullptr);
}

TEST(PostprocessorLibraryTest, InvalidConfig) {
  std::vector<int> labels = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
  std::vector<float> predictions = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
                                    0.8f, 0.8f, 0.8f, 0.8f, 0.8f};

  auto dataset = CreateDataset(labels);
  FakeClassificationModel model(predictions);

  model::proto::DeploymentConfig deployment;
  deployment.set_num_threads(1);

  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(0);

  // Empty config, no postprocessor set in oneof.
  proto::AbstractPostprocessorTrainingConfig postprocessor_config;

  auto postprocessor_or = CreatePostprocessor(
      deployment, config_link, postprocessor_config, model, dataset);

  EXPECT_THAT(postprocessor_or.status(),
              test::StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace yggdrasil_decision_forests::model::postprocessor
