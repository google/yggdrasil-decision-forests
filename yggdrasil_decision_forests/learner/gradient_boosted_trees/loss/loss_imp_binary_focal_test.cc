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

TEST(BinaryFocalLossTest, InitialPredictions) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  std::vector<float> weights = {1.f, 1.f, 1.f, 1.f};
  const BinaryFocalLoss loss_imp({}, model::proto::Task::CLASSIFICATION,
                                 dataset.data_spec().columns(1));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> init_pred,
      loss_imp.InitialPredictions(dataset, /*label_col_idx=*/1, weights));
  // Initial predictions for binary focal loss are the same as for binomial log
  // loss.
  EXPECT_THAT(init_pred, ElementsAre(0.f));
}

TEST(BinaryFocalLossTest, UpdateGradients) {
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
  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> loss_initial_predictions,
      loss_imp.InitialPredictions(dataset,
                                  /*label_col_idx=*/1, weights));
  internal::SetInitialPredictions(loss_initial_predictions, dataset.nrow(),
                                  &predictions);

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

  ASSERT_THAT(gradients.front().hessian, NotNull());
  const std::vector<float>& hessian = *gradients.front().hessian;
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

  ASSERT_THAT(gradients.front().hessian, NotNull());
  const std::vector<float>& hessian = *gradients.front().hessian;
  EXPECT_THAT(hessian, ElementsAre(FloatNear(0.0772814f, kTestPrecision),
                                   FloatNear(0.00633879f, kTestPrecision),
                                   FloatNear(0.0553163f, kTestPrecision),
                                   FloatNear(0.226232f, kTestPrecision)));
}

TEST(BinaryFocalLossTest, SetLabelDistribution) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  std::vector<float> weights = {1.f, 1.f, 1.f, 1.f};

  std::vector<GradientData> gradients;
  proto::GradientBoostedTreesTrainingConfig gbt_config;
  gbt_config.set_shrinkage(1.f);

  const BinaryFocalLoss loss_imp(gbt_config, model::proto::Task::CLASSIFICATION,
                                 dataset.data_spec().columns(1));
  dataset::VerticalDataset gradient_dataset;
  ASSERT_OK(internal::CreateGradientDataset(dataset,
                                            /*label_col_idx=*/1,
                                            /*hessian_splits=*/false, loss_imp,
                                            &gradient_dataset, &gradients,
                                            nullptr));
  EXPECT_THAT(gradients, SizeIs(1));

  std::vector<UnsignedExampleIdx> selected_examples{0, 1, 2, 3};
  std::vector<float> predictions = {0.f, 0.f, 0.f, 0.f};

  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(2);  // Gradient column.

  decision_tree::NodeWithChildren node;
  ASSERT_OK(loss_imp.SetLeaf(gradient_dataset, selected_examples, weights,
                             config, config_link, predictions,
                             /*label_col_idx=*/1, &node));

  // Node output: Half positive, half negative.
  // (2*(1-0.5)+2*(0-0.5))/( 4*0.5*(1-0.5) ) => 0
  EXPECT_EQ(node.node().regressor().top_value(), 0);
  // Distribution of the gradients:
  EXPECT_EQ(node.node().regressor().distribution().sum(), 0);
  EXPECT_EQ(node.node().regressor().distribution().sum_squares(), 0);
  // Same as the number of examples in the dataset.
  EXPECT_EQ(node.node().regressor().distribution().count(), 4.);
}

TEST(BinaryFocalLossTest, ComputeLoss) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  std::vector<float> weights = {1.f, 1.f, 1.f, 1.f};
  std::vector<float> predictions = {0.f, 0.f, 0.f, 0.f};
  const BinaryFocalLoss loss_imp({}, model::proto::Task::CLASSIFICATION,
                                 dataset.data_spec().columns(1));
  float loss_value;
  std::vector<float> secondary_metric;
  ASSERT_OK(loss_imp.Loss(dataset,
                          /*label_col_idx=*/1, predictions, weights, nullptr,
                          &loss_value, &secondary_metric));

  // The loss for every example is log(2)/8, with gamma=2, alpha=.5 and all
  // predictions 0.
  EXPECT_NEAR(loss_value, std::log(2) / 8.f, kTestPrecision);
  EXPECT_THAT(secondary_metric, ElementsAre(FloatNear(0.5f, kTestPrecision)));
}

TEST(BinaryFocalLossTest, SecondaryMetricName) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const BinaryFocalLoss loss_imp({}, model::proto::Task::CLASSIFICATION,
                                 dataset.data_spec().columns(1));
  EXPECT_THAT(loss_imp.SecondaryMetricNames(), ElementsAre("accuracy"));
}

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
