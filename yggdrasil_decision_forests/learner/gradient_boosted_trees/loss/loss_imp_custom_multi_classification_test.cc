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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_custom_multi_classification.h"

#include <algorithm>
#include <cstdint>
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
using ::testing::SizeIs;

absl::StatusOr<dataset::VerticalDataset> CreateToyDataset() {
  dataset::VerticalDataset dataset;
  // TODO Replace by a modern function when possible.
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns {
      type: CATEGORICAL
      name: "a"
      categorical { number_of_unique_values: 4 is_already_integerized: true }
    }
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
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "1"}, {"b", "2"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "2"}, {"b", "1"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "3"}, {"b", "2"}}));
  return dataset;
}

// Creates a very simple loss for testing 3-class classification.
// The initial prediction is the sum of weight*labels * class_idx.
// The gradient is the (prediction + label) * class_idx.
// The hessian is (gradient**2)
// The loss is the sum of weight*(prediction + label)
CustomMultiClassificationLossFunctions Create3DimToyLoss() {
  // Number of classes in the dataset.
  constexpr int dimension = 3;

  CustomMultiClassificationLossFunctions toy_loss;
  toy_loss.initial_predictions =
      [](const absl::Span<const int32_t> labels,
         const absl::Span<const float> weights,
         absl::Span<float> initial_predictions_out) -> absl::Status {
    STATUS_CHECK_EQ(initial_predictions_out.size(), 3);
    auto weight_times_labels =
        std::inner_product(labels.begin(), labels.end(), weights.begin(), 0.f);
    for (int dim_idx = 0; dim_idx < dimension; ++dim_idx) {
      initial_predictions_out[dim_idx] = (dim_idx + 1) * weight_times_labels;
    }
    return absl::OkStatus();
  };
  toy_loss.loss =
      [](const absl::Span<const int32_t> labels,
         const absl::Span<const float> predictions,
         const absl::Span<const float> weights) -> absl::StatusOr<float> {
    STATUS_CHECK_EQ(labels.size() * dimension, predictions.size());
    STATUS_CHECK_EQ(labels.size(), weights.size());
    float loss = 0.0;
    for (int i = 0; i < dimension; i++) {
      loss = std::inner_product(labels.begin(), labels.end(), weights.begin(),
                                loss);
      loss = std::inner_product(predictions.begin() + i * labels.size(),
                                predictions.begin() + (i + 1) * labels.size(),
                                weights.begin(), loss);
    }
    return loss;
  };
  toy_loss.gradient_and_hessian =
      [](const absl::Span<const int32_t> labels,
         const absl::Span<const float> predictions,
         absl::Span<const absl::Span<float>> gradient,
         absl::Span<const absl::Span<float>> hessian) -> absl::Status {
    STATUS_CHECK_EQ(labels.size() * dimension, predictions.size());
    STATUS_CHECK_EQ(hessian.size(), gradient.size());
    STATUS_CHECK_EQ(gradient.size(), dimension);

    for (int dim_idx = 0; dim_idx < dimension; ++dim_idx) {
      STATUS_CHECK_EQ(labels.size(), gradient[dim_idx].size());
      STATUS_CHECK_EQ(labels.size(), hessian[dim_idx].size());
      std::transform(labels.begin(), labels.end(), predictions.begin(),
                     gradient[dim_idx].begin(), std::plus<float>());
      std::transform(gradient[dim_idx].begin(), gradient[dim_idx].end(),
                     gradient[dim_idx].begin(),
                     [&dim_idx](auto& c) { return c * (dim_idx + 1); });
      std::transform(gradient[dim_idx].begin(), gradient[dim_idx].end(),
                     gradient[dim_idx].begin(), hessian[dim_idx].begin(),
                     std::multiplies<float>());
    }
    return absl::OkStatus();
  };
  return toy_loss;
}

TEST(CustomMultiClassificationLossTest, LossShape) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());

  ASSERT_OK_AND_ASSIGN(const auto loss_imp,
                       CustomMultiClassificationLoss::RegistrationCreate(
                           {{},
                            {},
                            model::proto::Task::CLASSIFICATION,
                            dataset.data_spec().columns(0)},
                           Create3DimToyLoss()));

  auto loss_shape = loss_imp->Shape();
  EXPECT_EQ(loss_shape.gradient_dim, 3);
  EXPECT_EQ(loss_shape.prediction_dim, 3);
}

TEST(CustomMultiClassificationLossTest, LossShapeBinaryFails) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  EXPECT_FALSE(CustomMultiClassificationLoss::RegistrationCreate(
                   {{},
                    {},
                    model::proto::Task::CLASSIFICATION,
                    dataset.data_spec().columns(1)},
                   Create3DimToyLoss())
                   .ok());
}

TEST(CustomMultiClassificationLossTest, InitialPredictions) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  std::vector<float> weights = {2.f, 4.f, 6.f, 8.f, 10.f, 12.f};

  ASSERT_OK_AND_ASSIGN(const auto loss_imp,
                       CustomMultiClassificationLoss::RegistrationCreate(
                           {{},
                            {},
                            model::proto::Task::CLASSIFICATION,
                            dataset.data_spec().columns(0)},
                           Create3DimToyLoss()));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> init_pred,
      loss_imp->InitialPredictions(dataset, /* label_col_idx= */ 0, weights));

  EXPECT_THAT(init_pred, ElementsAre(92, 184, 276));
}

TEST(CustomMultiClassificationLossTest, UpdateGradients) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  ASSERT_OK_AND_ASSIGN(const auto loss_imp,
                       CustomMultiClassificationLoss::RegistrationCreate(
                           {{},
                            {},
                            model::proto::Task::CLASSIFICATION,
                            dataset.data_spec().columns(0)},
                           Create3DimToyLoss()));
  ASSERT_OK(internal::CreateGradientDataset(dataset,
                                            /* label_col_idx= */ 0, *loss_imp,
                                            &gradient_dataset, &gradients,
                                            &predictions));
  utils::RandomEngine random(1234);
  ASSERT_OK(loss_imp->UpdateGradients(gradient_dataset,
                                      /* label_col_idx= */ 0, predictions,
                                      /*ranking_index=*/nullptr, &gradients,
                                      &random));

  ASSERT_THAT(gradients, SizeIs(3));
  EXPECT_THAT(gradients[0].gradient, ElementsAre(1.f, 2.f, 3.f, 1.f, 2.f, 3.));
  EXPECT_THAT(gradients[0].hessian, ElementsAre(1.f, 4.f, 9.f, 1.f, 4.f, 9.f));
  EXPECT_THAT(gradients[1].gradient, ElementsAre(2.f, 4.f, 6., 2.f, 4.f, 6.f));
  EXPECT_THAT(gradients[1].hessian,
              ElementsAre(4.f, 16.f, 36.f, 4.f, 16.f, 36.f));
  EXPECT_THAT(gradients[2].gradient, ElementsAre(3.f, 6.f, 9.f, 3.f, 6.f, 9.f));
  EXPECT_THAT(gradients[2].hessian,
              ElementsAre(9.f, 36.f, 81.f, 9.f, 36.f, 81.f));
}

TEST(CustomMultiClassificationLossTest, ComputeLoss) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  std::vector<float> weights = {2.f, 4.f, 6.f, 8.f, 10.f, 12.f};

  std::vector<float> predictions(dataset.nrow() * 3, 2.f);
  ASSERT_OK_AND_ASSIGN(const auto loss_imp,
                       CustomMultiClassificationLoss::RegistrationCreate(
                           {{},
                            {},
                            model::proto::Task::CLASSIFICATION,
                            dataset.data_spec().columns(0)},
                           Create3DimToyLoss()));
  LossResults loss_results;
  ASSERT_OK_AND_ASSIGN(
      loss_results,
      loss_imp->Loss(dataset,
                     /* label_col_idx= */ 0, predictions, weights, nullptr));
  EXPECT_EQ(loss_results.loss, 528);
  // There are no secondary metrics.
  EXPECT_THAT(loss_results.secondary_metrics, IsEmpty());
}

TEST(CustomMultiClassificationLossTest, ValidForClassification) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());

  ASSERT_OK_AND_ASSIGN(const auto loss_imp,
                       CustomMultiClassificationLoss::RegistrationCreate(
                           {{},
                            {},
                            model::proto::Task::CLASSIFICATION,
                            dataset.data_spec().columns(0)},
                           Create3DimToyLoss()));
}

TEST(CustomMultiClassificationLossTest, InvalidForRegression) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  EXPECT_FALSE(CustomMultiClassificationLoss::RegistrationCreate(
                   {{},
                    {},
                    model::proto::Task::REGRESSION,
                    dataset.data_spec().columns(1)},
                   Create3DimToyLoss())
                   .ok());
}

TEST(CustomMultiClassificationLossTest, InvalidForRanking) {
  ASSERT_OK_AND_ASSIGN(const dataset::VerticalDataset dataset,
                       CreateToyDataset());
  EXPECT_FALSE(
      CustomMultiClassificationLoss::RegistrationCreate(
          {{}, {}, model::proto::Task::RANKING, dataset.data_spec().columns(1)},
          Create3DimToyLoss())
          .ok());
}

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
