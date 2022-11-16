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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_multinomial.h"

#include "gmock/gmock.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
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
using ::testing::SizeIs;

// Margin of error for numerical tests.
constexpr float kTestPrecision = 0.000001f;

absl::StatusOr<dataset::VerticalDataset> CreateToyDataset() {
  dataset::VerticalDataset dataset;
  // TODO Replace by a modern macro when possible.
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    columns { type: NUMERICAL name: "a" }
    columns {
      type: CATEGORICAL
      name: "b"
      categorical { number_of_unique_values: 4 is_already_integerized: true }
    }
  )pb");
  RETURN_IF_ERROR(dataset.CreateColumnsFromDataspec());
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "1"}, {"b", "1"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "2"}, {"b", "2"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "3"}, {"b", "3"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "4"}, {"b", "1"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "5"}, {"b", "2"}}));
  RETURN_IF_ERROR(dataset.AppendExampleWithStatus({{"a", "6"}, {"b", "3"}}));
  return dataset;
}

TEST(MultinomialLogLikelihoodLossTest, InitialPredictions) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  std::vector<float> weights = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  const MultinomialLogLikelihoodLoss loss_imp(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<float> init_pred,
      loss_imp.InitialPredictions(dataset, /* label_col_idx= */ 1, weights));
  EXPECT_THAT(init_pred, ElementsAre(0.f, 0.f, 0.f));
}

TEST(MultinomialLogLikelihoodLossTest, UpdateGradients) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  std::vector<float> weights(dataset.nrow(), 1.f);

  dataset::VerticalDataset gradient_dataset;
  std::vector<GradientData> gradients;
  std::vector<float> predictions;
  const MultinomialLogLikelihoodLoss loss_imp(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  ASSERT_OK(internal::CreateGradientDataset(dataset,
                                            /* label_col_idx= */ 1,
                                            /*hessian_splits=*/false, loss_imp,
                                            &gradient_dataset, &gradients,
                                            &predictions));
  ASSERT_OK_AND_ASSIGN(
      const auto initial_predictions,
      loss_imp.InitialPredictions(dataset,
                                  /* label_col_idx =*/1, weights));
  internal::SetInitialPredictions(initial_predictions, dataset.nrow(),
                                  &predictions);

  utils::RandomEngine random(1234);
  ASSERT_OK(loss_imp.UpdateGradients(
      gradient_dataset, /* label_col_idx= */ 1, predictions,
      /*ranking_index=*/nullptr, &gradients, &random));

  ASSERT_THAT(gradients, Not(IsEmpty()));
  ASSERT_THAT(gradients.front().gradient, SizeIs(6));

  // TODO: Implement and use "AllElementsNear" matcher.
  EXPECT_NEAR(gradients.front().gradient[0], 2.f / 3.f, kTestPrecision);
  EXPECT_NEAR(gradients.front().gradient[1], -1.f / 3.f, kTestPrecision);
  EXPECT_NEAR(gradients.front().gradient[2], -1.f / 3.f, kTestPrecision);
  EXPECT_NEAR(gradients.front().gradient[3], 2.f / 3.f, kTestPrecision);
  EXPECT_NEAR(gradients.front().gradient[4], -1.f / 3.f, kTestPrecision);
  EXPECT_NEAR(gradients.front().gradient[5], -1.f / 3.f, kTestPrecision);
}

TEST(MultinomialLogLikelihoodLossTest, SetLabelDistribution) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  std::vector<float> weights(dataset.nrow(), 1.f);

  std::vector<GradientData> gradients;

  proto::GradientBoostedTreesTrainingConfig gbt_config;
  gbt_config.set_shrinkage(1.f);
  const MultinomialLogLikelihoodLoss loss_imp(
      gbt_config, model::proto::Task::CLASSIFICATION,
      dataset.data_spec().columns(1));
  dataset::VerticalDataset gradient_dataset;
  ASSERT_OK(internal::CreateGradientDataset(dataset,
                                            /* label_col_idx= */ 1,
                                            /*hessian_splits=*/false, loss_imp,
                                            &gradient_dataset, &gradients,
                                            nullptr));
  ASSERT_THAT(gradients, SizeIs(3));

  std::vector<UnsignedExampleIdx> selected_examples{0, 1, 2, 3, 4, 5};
  std::vector<float> predictions(dataset.nrow(), 0.f);

  model::proto::TrainingConfig config;
  model::proto::TrainingConfigLinking config_link;
  config_link.set_label(2);  // Gradient column.

  decision_tree::NodeWithChildren node;
  ASSERT_OK(loss_imp.SetLeaf(gradient_dataset, selected_examples, weights,
                             config, config_link, predictions,
                             /* label_col_idx= */ 1, &node));

  // Node output: Half positive, half negative.
  // (2*(1-0.5)+2*(0-0.5))/( 4*0.5*(1-0.5) ) => 0
  EXPECT_EQ(node.node().regressor().top_value(), 0);
  // Distribution of the gradients:
  EXPECT_EQ(node.node().regressor().distribution().sum(), 0);
  EXPECT_EQ(node.node().regressor().distribution().sum_squares(), 0);
  // Same as the number of examples in the dataset.
  EXPECT_EQ(node.node().regressor().distribution().count(), 6.);
}

TEST(MultinomialLogLikelihoodLossTest, SecondaryMetricName) {
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  const MultinomialLogLikelihoodLoss loss_imp(
      {}, model::proto::Task::CLASSIFICATION, dataset.data_spec().columns(1));
  EXPECT_THAT(loss_imp.SecondaryMetricNames(), ElementsAre("accuracy"));
}

using ThreadedMultinomialLogLikelihoodLossTest = ::testing::TestWithParam<bool>;

TEST_P(ThreadedMultinomialLogLikelihoodLossTest, ComputeLoss) {
  const bool threaded = GetParam();
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  std::vector<float> weights;
  auto label_column = dataset.data_spec().columns(1);
  std::vector<float> predictions(
      dataset.nrow() *
          (label_column.categorical().number_of_unique_values() - 1),
      0.f);
  const MultinomialLogLikelihoodLoss loss_imp(
      {}, model::proto::Task::CLASSIFICATION, label_column);
  LossResults loss_results;
  if (threaded) {
    utils::concurrency::ThreadPool thread_pool("", 4);
    thread_pool.StartWorkers();
    ASSERT_OK_AND_ASSIGN(
        loss_results,
        loss_imp.Loss(dataset,
                      /* label_col_idx= */ 1, predictions, weights, nullptr));
  } else {
    ASSERT_OK_AND_ASSIGN(
        loss_results,
        loss_imp.Loss(dataset,
                      /* label_col_idx= */ 1, predictions, weights, nullptr));
  }

  EXPECT_NEAR(loss_results.loss, std::log(3), kTestPrecision);
  EXPECT_THAT(loss_results.secondary_metrics,
              ElementsAre(FloatNear(0.333333f, kTestPrecision)));
  ASSERT_TRUE(loss_results.confusion_table.has_value());
  EXPECT_EQ(loss_results.confusion_table->at(1, 1), 2);
  EXPECT_EQ(loss_results.confusion_table->at(2, 1), 2);
  EXPECT_EQ(loss_results.confusion_table->at(3, 1), 2);
  EXPECT_EQ(loss_results.confusion_table->sum(), 6);
}

TEST_P(ThreadedMultinomialLogLikelihoodLossTest, ComputeLossWithWeights) {
  const bool threaded = GetParam();
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  std::vector<float> weights = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  auto label_column = dataset.data_spec().columns(1);
  std::vector<float> predictions(
      dataset.nrow() *
          (label_column.categorical().number_of_unique_values() - 1),
      0.f);
  const MultinomialLogLikelihoodLoss loss_imp(
      {}, model::proto::Task::CLASSIFICATION, label_column);
  LossResults loss_results;
  if (threaded) {
    utils::concurrency::ThreadPool thread_pool("", 4);
    thread_pool.StartWorkers();
    ASSERT_OK_AND_ASSIGN(
        loss_results,
        loss_imp.Loss(dataset,
                      /* label_col_idx= */ 1, predictions, weights, nullptr));
  } else {
    ASSERT_OK_AND_ASSIGN(
        loss_results,
        loss_imp.Loss(dataset,
                      /* label_col_idx= */ 1, predictions, weights, nullptr));
  }

  EXPECT_NEAR(loss_results.loss, std::log(3), kTestPrecision);
  EXPECT_THAT(loss_results.secondary_metrics,
              ElementsAre(FloatNear(5. / 21., kTestPrecision)));
  ASSERT_TRUE(loss_results.confusion_table.has_value());
  EXPECT_EQ(loss_results.confusion_table->at(1, 1), 5.);
  EXPECT_EQ(loss_results.confusion_table->at(2, 1), 7.);
  EXPECT_EQ(loss_results.confusion_table->at(3, 1), 9.);
  EXPECT_EQ(loss_results.confusion_table->sum(), 21.);
}

TEST_P(ThreadedMultinomialLogLikelihoodLossTest, ComputeLossWithNullWeights) {
  const bool threaded = GetParam();
  ASSERT_OK_AND_ASSIGN(const auto dataset, CreateToyDataset());
  std::vector<float> weights(dataset.nrow(), 0.f);
  auto label_column = dataset.data_spec().columns(1);
  std::vector<float> predictions(
      dataset.nrow() *
          (label_column.categorical().number_of_unique_values() - 1),
      0.f);
  const MultinomialLogLikelihoodLoss loss_imp(
      {}, model::proto::Task::CLASSIFICATION, label_column);
  LossResults loss_results;
  if (threaded) {
    utils::concurrency::ThreadPool thread_pool("", 4);
    thread_pool.StartWorkers();
    ASSERT_OK_AND_ASSIGN(
        loss_results,
        loss_imp.Loss(dataset,
                      /* label_col_idx= */ 1, predictions, weights, nullptr));
  } else {
    ASSERT_OK_AND_ASSIGN(
        loss_results,
        loss_imp.Loss(dataset,
                      /* label_col_idx= */ 1, predictions, weights, nullptr));
  }

  EXPECT_THAT(loss_results.loss, IsNan());
  EXPECT_THAT(loss_results.secondary_metrics, ElementsAre(IsNan()));
  EXPECT_EQ(loss_results.confusion_table, absl::nullopt);
}

INSTANTIATE_TEST_SUITE_P(ThreadedMultinomialLogLikelihoodLossTests,
                         ThreadedMultinomialLogLikelihoodLossTest,
                         testing::ValuesIn<bool>({true, false}));

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
