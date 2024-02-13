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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_custom_regression.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Not;

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

// Creates a custom loss.
// The initial prediction is the sum of weight*labels.
// The gradient is the (prediction + label).
// The hessian is ((prediction + label)**2)
// The loss is the sum of weight*(prediction + label)
CustomRegressionLossFunctions CreateToyLoss() {
  CustomRegressionLossFunctions toy_loss_data;
  toy_loss_data.initial_predictions =
      [](const absl::Span<const float> labels,
         const absl::Span<const float> weights) -> absl::StatusOr<float> {
    return std::inner_product(labels.begin(), labels.end(), weights.begin(),
                              0.f);
  };
  toy_loss_data.gradient_and_hessian =
      [](const absl::Span<const float> labels,
         const absl::Span<const float> predictions, absl::Span<float> gradient,
         absl::Span<float> hessian) -> absl::Status {
    STATUS_CHECK_EQ(labels.size(), predictions.size());
    STATUS_CHECK_EQ(labels.size(), gradient.size());
    STATUS_CHECK_EQ(labels.size(), hessian.size());

    std::transform(labels.begin(), labels.end(), predictions.begin(),
                   gradient.begin(), std::plus<float>());
    std::transform(gradient.begin(), gradient.end(), gradient.begin(),
                   hessian.begin(), std::multiplies<float>());
    return absl::OkStatus();
  };
  toy_loss_data.loss = [](const absl::Span<const float> labels,
                          const absl::Span<const float> predictions,
                          const absl::Span<const float> weights) -> float {
    float loss =
        std::inner_product(labels.begin(), labels.end(), weights.begin(), 0.0);
    loss += std::inner_product(predictions.begin(), predictions.end(),
                               weights.begin(), 0.0);
    return loss;
  };
  return toy_loss_data;
}

TEST(CustomRegressionLossTest, InitialPredictions) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  std::vector<float> weights = {2.f, 4.f, 6.f, 8.f};

  CustomRegressionLoss loss_imp({}, model::proto::Task::REGRESSION,
                                dataset.data_spec().columns(0),
                                CreateToyLoss());
  ASSERT_OK(loss_imp.Status());
  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> init_pred,
      loss_imp.InitialPredictions(dataset, /* label_col_idx= */ 0, weights));

  EXPECT_THAT(init_pred,
              ElementsAre(1.f * 2.f + 2.f * 4.f + 3.f * 6.f + 4.f * 8.f));
}

TEST(CustomRegressionLossTest, UpdateGradients) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  CustomRegressionLoss loss_imp({}, model::proto::Task::REGRESSION,
                                dataset.data_spec().columns(0),
                                CreateToyLoss());
  ASSERT_OK(loss_imp.Status());
  ASSERT_OK(internal::CreateGradientDataset(dataset,
                                            /* label_col_idx= */ 0,
                                            /*hessian_splits=*/false, loss_imp,
                                            &gradient_dataset, &gradients,
                                            &predictions));
  const std::vector<float> loss_initial_predictions = {1};
  internal::SetInitialPredictions(loss_initial_predictions, dataset.nrow(),
                                  &predictions);

  utils::RandomEngine random(1234);
  ASSERT_OK(loss_imp.UpdateGradients(gradient_dataset,
                                     /* label_col_idx= */ 0, predictions,
                                     /*ranking_index=*/nullptr, &gradients,
                                     &random));

  ASSERT_THAT(gradients, Not(IsEmpty()));
  EXPECT_THAT(gradients.front().gradient, ElementsAre(2.f, 3.f, 4.f, 5.f));
}

TEST(CustomRegressionLossTest, ComputeLoss) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  std::vector<float> weights = {2.f, 4.f, 6.f, 8.f};

  std::vector<float> predictions = {2.f, 2.f, 2.f, 2.f};
  CustomRegressionLoss loss_imp({}, model::proto::Task::REGRESSION,
                                dataset.data_spec().columns(0),
                                CreateToyLoss());
  ASSERT_OK(loss_imp.Status());
  LossResults loss_results;
  ASSERT_OK_AND_ASSIGN(
      loss_results,
      loss_imp.Loss(dataset,
                    /* label_col_idx= */ 0, predictions, weights, nullptr));
  EXPECT_EQ(loss_results.loss, 2.f * 3.f + 4.f * 4.f + 6.f * 5.f + 8.f * 6.f);
  // There are no secondary metrics.
  EXPECT_THAT(loss_results.secondary_metrics, IsEmpty());
}

TEST(CustomRegressionLossTest, SecondaryMetricNames) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const CustomRegressionLoss loss_imp({}, model::proto::Task::REGRESSION,
                                      dataset.data_spec().columns(1),
                                      CreateToyLoss());
  ASSERT_OK(loss_imp.Status());
  EXPECT_THAT(loss_imp.SecondaryMetricNames(), IsEmpty());
}

TEST(CustomRegressionLossTest, ValidForRegression) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const CustomRegressionLoss loss_imp({}, model::proto::Task::REGRESSION,
                                      dataset.data_spec().columns(1),
                                      CreateToyLoss());
  EXPECT_OK(loss_imp.Status());
}

TEST(CustomRegressionLossTest, InvalidForClassification) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const CustomRegressionLoss loss_imp({}, model::proto::Task::CLASSIFICATION,
                                      dataset.data_spec().columns(1),
                                      CreateToyLoss());
  EXPECT_FALSE(loss_imp.Status().ok());
}

TEST(CustomRegressionLossTest, InvalidForRanking) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  const CustomRegressionLoss loss_imp({}, model::proto::Task::RANKING,
                                      dataset.data_spec().columns(1),
                                      CreateToyLoss());
  EXPECT_FALSE(loss_imp.Status().ok());
}

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
