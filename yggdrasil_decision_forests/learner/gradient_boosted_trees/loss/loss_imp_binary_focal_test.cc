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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_binary_focal.h"

#include "gmock/gmock.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::IsEmpty;
using ::testing::IsNan;
using ::testing::Not;
using ::testing::NotNull;
using ::testing::SizeIs;

// Margin of error for numerical tests.
constexpr float kTestPrecision = 0.000001f;

absl::StatusOr<dataset::VerticalDataset> CreateToyDataset() {
  dataset::VerticalDataset dataset;
  // TODO Replace by a modern function when possible.
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "a" }
    columns {
      type: CATEGORICAL
      name: "b"
      categorical { number_of_unique_values: 3 is_already_integerized: true }
    }
  )pb");
  RETURN_IF_ERROR(dataset.CreateColumnsFromDataspec());
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "1"}, {"b", "1"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "2"}, {"b", "2"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "3"}, {"b", "1"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "4"}, {"b", "2"}}));
  return dataset;
}

class BinaryFocalLossTest : public testing::TestWithParam<bool> {};

TEST_P(BinaryFocalLossTest, InitialPredictions) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const bool weighted = GetParam();
  std::vector<float> weights;
  if (weighted) {
    weights = {2.f, 4.f, 6.f, 8.f};
  }

  const BinaryFocalLoss loss_imp({}, model::proto::Task::CLASSIFICATION,
                                 dataset.data_spec().columns(1));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> init_pred,
      loss_imp.InitialPredictions(dataset, /*label_col_idx=*/1, weights));
  // Initial predictions for binary focal loss are the same as for binomial log
  // loss.
  if (weighted) {
    EXPECT_THAT(init_pred,
                ElementsAre(FloatNear(std::log(3. / 2.), kTestPrecision)));
  } else {
    EXPECT_THAT(init_pred, ElementsAre(0.f));
  }
}

TEST(BinaryFocalLossTest, UpdateGradients) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  std::vector<float> weights = {1.f, 1.f, 1.f, 1.f};

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions = {0.f, 0.f, 0.f, 0.f};
  proto::GradientBoostedTreesTrainingConfig gbt_config;
  gbt_config.set_use_hessian_gain(true);
  const BinaryFocalLoss loss_imp(gbt_config, model::proto::Task::CLASSIFICATION,
                                 dataset.data_spec().columns(1));
  ASSERT_OK(internal::CreateGradientDataset(dataset,
                                            /*label_col_idx=*/1,
                                            /*hessian_splits=*/false, loss_imp,
                                            &gradient_dataset, &gradients,
                                            &predictions));

  utils::RandomEngine random(1234);
  ASSERT_OK(loss_imp.UpdateGradients(
      gradient_dataset, /*label_col_idx=*/1, predictions,
      /*ranking_index=*/nullptr, &gradients, &random));

  ASSERT_THAT(gradients, Not(IsEmpty()));
  const std::vector<float>& gradient = gradients.front().gradient;
  // Values validated with tensorflow focal loss implementation
  // (tfa.losses.sigmoid_focal_crossentropy).
  EXPECT_THAT(gradient, ElementsAre(FloatNear(-0.149143f, kTestPrecision),
                                    FloatNear(0.149143f, kTestPrecision),
                                    FloatNear(-0.149143f, kTestPrecision),
                                    FloatNear(0.149143f, kTestPrecision)));

  const std::vector<float>& hessian = gradients.front().hessian;
  EXPECT_THAT(hessian, ElementsAre(FloatNear(0.199572f, kTestPrecision),
                                   FloatNear(0.199572f, kTestPrecision),
                                   FloatNear(0.199572f, kTestPrecision),
                                   FloatNear(0.199572f, kTestPrecision)));
}

TEST(BinaryFocalLossTest, UpdateGradientsCustomPredictions) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  std::vector<float> weights = {1.f, 1.f, 1.f, 1.f};

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  proto::GradientBoostedTreesTrainingConfig gbt_config;
  gbt_config.set_use_hessian_gain(true);
  const BinaryFocalLoss loss_imp(gbt_config, model::proto::Task::CLASSIFICATION,
                                 dataset.data_spec().columns(1));
  ASSERT_OK(internal::CreateGradientDataset(dataset,
                                            /*label_col_idx=*/1,
                                            /*hessian_splits=*/false, loss_imp,
                                            &gradient_dataset, &gradients,
                                            &predictions));

  std::vector<float> custom_predictions = {2.0, 2.0, -1.0, -1.0};

  utils::RandomEngine random(1234);
  ASSERT_OK(loss_imp.UpdateGradients(
      gradient_dataset, /*label_col_idx=*/1, custom_predictions,
      /*ranking_index=*/nullptr, &gradients, &random));

  ASSERT_THAT(gradients, Not(IsEmpty()));
  const std::vector<float>& gradient = gradients.front().gradient;
  EXPECT_THAT(gradient, ElementsAre(FloatNear(-0.538357f, kTestPrecision),
                                    FloatNear(0.00243547f, kTestPrecision),
                                    FloatNear(-0.0262906f, kTestPrecision),
                                    FloatNear(0.384117f, kTestPrecision)));

  const std::vector<float>& hessian = gradients.front().hessian;
  EXPECT_THAT(hessian, ElementsAre(FloatNear(0.0772814f, kTestPrecision),
                                   FloatNear(0.00633879f, kTestPrecision),
                                   FloatNear(0.0553163f, kTestPrecision),
                                   FloatNear(0.226232f, kTestPrecision)));
}

TEST(BinaryFocalLossTest, ComputeLossWithoutWeights) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  std::vector<float> weights;
  std::vector<float> predictions = {0.f, 0.f, 0.f, 0.f};
  const BinaryFocalLoss loss_imp({}, model::proto::Task::CLASSIFICATION,
                                 dataset.data_spec().columns(1));
  ASSERT_OK_AND_ASSIGN(
      LossResults loss_results,
      loss_imp.Loss(dataset,
                    /*label_col_idx=*/1, predictions, weights, nullptr));

  // The loss for every example is log(2)/8, with gamma=2, alpha=.5 and all
  // predictions 0.
  EXPECT_NEAR(loss_results.loss, std::log(2) / 8.f, kTestPrecision);
  EXPECT_THAT(loss_results.secondary_metrics,
              ElementsAre(FloatNear(0.5f, kTestPrecision)));
}

TEST(BinaryFocalLossTest, ComputeLossWithWeights) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  std::vector<float> weights = {2.f, 4.f, 6.f, 8.f};
  std::vector<float> predictions = {0.f, 0.f, 0.f, 0.f};
  const BinaryFocalLoss loss_imp({}, model::proto::Task::CLASSIFICATION,
                                 dataset.data_spec().columns(1));
  ASSERT_OK_AND_ASSIGN(
      LossResults loss_results,
      loss_imp.Loss(dataset,
                    /*label_col_idx=*/1, predictions, weights, nullptr));

  // The loss for every example is log(2)/8, with gamma=2, alpha=.5 and all
  // predictions 0.
  EXPECT_NEAR(loss_results.loss, std::log(2) / 8.f, kTestPrecision);
  EXPECT_THAT(loss_results.secondary_metrics,
              ElementsAre(FloatNear(12.f / 20.f, kTestPrecision)));
}

TEST(BinaryFocalLossTest, ComputeLossWithNullWeights) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  std::vector<float> weights(dataset.nrow(), 0.f);
  std::vector<float> predictions(dataset.nrow(), 0.f);
  const BinaryFocalLoss loss_imp({}, model::proto::Task::CLASSIFICATION,
                                 dataset.data_spec().columns(1));
  ASSERT_OK_AND_ASSIGN(
      LossResults loss_results,
      loss_imp.Loss(dataset,
                    /*label_col_idx=*/1, predictions, weights, nullptr));

  EXPECT_THAT(loss_results.loss, IsNan());
  EXPECT_THAT(loss_results.secondary_metrics, ElementsAre(IsNan()));
}

TEST(BinaryFocalLossTest, SecondaryMetricName) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const BinaryFocalLoss loss_imp({}, model::proto::Task::CLASSIFICATION,
                                 dataset.data_spec().columns(1));
  EXPECT_THAT(loss_imp.SecondaryMetricNames(), ElementsAre("accuracy"));
}

INSTANTIATE_TEST_SUITE_P(BinaryFocalLossTestWithWeights, BinaryFocalLossTest,
                         testing::Bool());

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
