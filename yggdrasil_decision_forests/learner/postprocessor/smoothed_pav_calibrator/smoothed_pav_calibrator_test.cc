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

#include "yggdrasil_decision_forests/learner/postprocessor/smoothed_pav_calibrator/smoothed_pav_calibrator.h"

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/model/model_testing.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"

namespace yggdrasil_decision_forests::model::postprocessor {
namespace smoothed_pav_calibrator {

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

TEST(SmoothedPavCalibratorTest, Basic) {
  // 10 examples.
  // 5 negative (label 1) with prediction 0.2.
  // 5 positive (label 2) with prediction 0.8.
  std::vector<int> labels = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
  std::vector<float> predictions = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
                                    0.8f, 0.8f, 0.8f, 0.8f, 0.8f};

  auto dataset = CreateDataset(labels);
  FakeClassificationModel model(predictions);

  model::proto::DeploymentConfig deployment;
  deployment.set_num_threads(1);

  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(0);

  proto::SmoothedPavCalibratorTrainingConfig postprocessor_config;
  // Use default config values.

  auto calibrator_or = CreateSmoothedPavCalibrator(
      deployment, config_link, postprocessor_config, model, dataset);

  ASSERT_OK(calibrator_or.status());
  auto calibrator = calibrator_or.value();
  ASSERT_NE(calibrator, nullptr);
}

TEST(SmoothedPavCalibratorTest, Quantitative) {
  // 10 examples.
  // 5 negative (label 1) with prediction 0.2.
  // 5 positive (label 2) with prediction 0.8.
  std::vector<int> labels = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
  std::vector<float> predictions = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
                                    0.8f, 0.8f, 0.8f, 0.8f, 0.8f};

  auto dataset = CreateDataset(labels);
  FakeClassificationModel model(predictions);

  model::proto::DeploymentConfig deployment;
  deployment.set_num_threads(1);

  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(0);

  proto::SmoothedPavCalibratorTrainingConfig postprocessor_config;

  auto calibrator_or = CreateSmoothedPavCalibrator(
      deployment, config_link, postprocessor_config, model, dataset);

  ASSERT_OK(calibrator_or.status());
  auto calibrator = calibrator_or.value();
  ASSERT_NE(calibrator, nullptr);

  auto make_prediction = [](float p) {
    model::proto::Prediction prediction;
    auto* dist = prediction.mutable_classification()->mutable_distribution();
    dist->set_sum(1.0f);
    dist->add_counts(0.0f);      // OOV
    dist->add_counts(1.0f - p);  // Neg
    dist->add_counts(p);         // Pos
    return prediction;
  };

  dataset::VerticalDataset dummy_dataset;

  // Test prediction 0.2 -> should be calibrated to ~0.0
  {
    auto pred = make_prediction(0.2f);
    calibrator->Process(dummy_dataset, 0, &pred);
    EXPECT_NEAR(pred.classification().distribution().counts(2), 0.0f, 1e-4f);
  }

  // Test prediction 0.8 -> should be calibrated to ~1.0
  {
    auto pred = make_prediction(0.8f);
    calibrator->Process(dummy_dataset, 0, &pred);
    EXPECT_NEAR(pred.classification().distribution().counts(2), 1.0f, 1e-4f);
  }

  // Test prediction 0.5 -> should be calibrated to ~0.5 (linear interpolation)
  {
    auto pred = make_prediction(0.5f);
    calibrator->Process(dummy_dataset, 0, &pred);
    EXPECT_NEAR(pred.classification().distribution().counts(2), 0.5f, 1e-3f);
  }

  // Test prediction 0.1 (outside range) -> should be clamped to 0.2's value
  // (0.0)
  {
    auto pred = make_prediction(0.1f);
    calibrator->Process(dummy_dataset, 0, &pred);
    EXPECT_NEAR(pred.classification().distribution().counts(2), 0.0f, 1e-5f);
  }

  // Test prediction 0.9 (outside range) -> should be clamped to 0.8's value
  // (1.0)
  {
    auto pred = make_prediction(0.9f);
    calibrator->Process(dummy_dataset, 0, &pred);
    EXPECT_NEAR(pred.classification().distribution().counts(2), 1.0f, 1e-5f);
  }
}

TEST(SmoothedPavCalibratorTest, ConfigVariation) {
  std::vector<int> labels = {
      1, 1, 1, 2, 2,  // pred 0.2 (2/5 pos = 0.4)
      1, 1, 2, 2, 2,  // pred 0.5 (3/5 pos = 0.6)
      2, 2, 2, 2, 2   // pred 0.8 (5/5 pos = 1.0)
  };
  std::vector<float> predictions = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
                                    0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                                    0.8f, 0.8f, 0.8f, 0.8f, 0.8f};

  auto dataset = CreateDataset(labels);
  FakeClassificationModel model(predictions);

  model::proto::DeploymentConfig deployment;
  deployment.set_num_threads(1);

  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(0);

  dataset::VerticalDataset dummy_dataset;
  auto make_prediction = [](float p) {
    model::proto::Prediction prediction;
    auto* dist = prediction.mutable_classification()->mutable_distribution();
    dist->set_sum(1.0f);
    dist->add_counts(0.0f);
    dist->add_counts(1.0f - p);
    dist->add_counts(p);
    return prediction;
  };

  // Test with z_threshold = 0.0 (no merging of these bins)
  {
    proto::SmoothedPavCalibratorTrainingConfig postprocessor_config;
    postprocessor_config.set_z_threshold(0.0f);

    auto calibrator_or = CreateSmoothedPavCalibrator(
        deployment, config_link, postprocessor_config, model, dataset);
    ASSERT_OK(calibrator_or.status());
    auto calibrator = calibrator_or.value();

    // h(0.2) should be ~0.4
    auto pred = make_prediction(0.2f);
    calibrator->Process(dummy_dataset, 0, &pred);
    EXPECT_NEAR(pred.classification().distribution().counts(2), 0.4f, 1e-4f);
  }

  // Test with z_threshold = 1.0 (should merge 0.2 and 0.5 bins)
  {
    proto::SmoothedPavCalibratorTrainingConfig postprocessor_config;
    postprocessor_config.set_z_threshold(1.0f);

    auto calibrator_or = CreateSmoothedPavCalibrator(
        deployment, config_link, postprocessor_config, model, dataset);
    ASSERT_OK(calibrator_or.status());
    auto calibrator = calibrator_or.value();

    // h(0.2) should be clamped to h(0.35) = 0.5
    auto pred = make_prediction(0.2f);
    calibrator->Process(dummy_dataset, 0, &pred);
    EXPECT_NEAR(pred.classification().distribution().counts(2), 0.5f, 1e-4f);
  }
}

}  // namespace smoothed_pav_calibrator
}  // namespace yggdrasil_decision_forests::model::postprocessor
