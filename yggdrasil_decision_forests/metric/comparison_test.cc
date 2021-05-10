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

#include "yggdrasil_decision_forests/metric/comparison.h"

#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "boost/math/distributions/normal.hpp"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace metric {
namespace {

class ModelComparisonPValueTest : public ::testing::Test {
 public:
  void SetUp() override {
    label_column_.set_type(dataset::proto::ColumnType::CATEGORICAL);
    label_column_.set_name("label");
    label_column_.mutable_categorical()->set_number_of_unique_values(2);
    label_column_.mutable_categorical()->set_most_frequent_value(1);
    label_column_.mutable_categorical()->set_is_already_integerized(true);
    option_.set_task(model::proto::Task::CLASSIFICATION);
    option_.mutable_classification()->set_roc_enable(true);
    option_.mutable_classification()->add_precision_at_recall(0.5);
    option_.mutable_classification()->add_recall_at_precision(1.0);
    option_.mutable_classification()->add_precision_at_volume(0.0);
    option_.mutable_classification()->add_recall_at_false_positive_rate(0.5);

    InitializeEvaluation(option_, label_column_, &eval_results1_);
    InitializeEvaluation(option_, label_column_, &eval_results2_);
  }

  void AddPredictionToEvaluationResult(const float proba,
                                       const int ground_truth, const int value,
                                       proto::EvaluationResults* eval_result) {
    model::proto::Prediction pred;
    auto* pred_proba = pred.mutable_classification()->mutable_distribution();
    pred_proba->mutable_counts()->Resize(2, 0);
    pred_proba->set_sum(1);
    pred.mutable_classification()->set_value(value);
    pred_proba->set_counts(0, proba);
    pred_proba->set_counts(1, 1.0f - proba);
    pred.mutable_classification()->set_ground_truth(ground_truth);
    AddPrediction(option_, pred, &rnd_, eval_result);
  }

  proto::EvaluationResults eval_results1_;
  proto::EvaluationResults eval_results2_;
  dataset::proto::Column label_column_;
  proto::EvaluationOptions option_;
  utils::RandomEngine rnd_;
};

TEST_F(ModelComparisonPValueTest, BasicTest) {
  for (int i = 0; i < 30; ++i) {
    const int ground_truth = i < 15 ? 0 : 1;

    float proba = ((i + 18) % 30) / 30.0;
    int value = proba > 0.5 ? 1 : 0;
    AddPredictionToEvaluationResult(proba, ground_truth, value,
                                    &eval_results1_);

    proba = ((i + 6) % 30) / 30.0;
    value = proba > 0.5 ? 1 : 0;
    AddPredictionToEvaluationResult(proba, ground_truth, value,
                                    &eval_results2_);
  }

  FinalizeEvaluation(option_, label_column_, &eval_results1_);
  FinalizeEvaluation(option_, label_column_, &eval_results2_);

  const std::vector<std::pair<std::string, float>> result =
      OneSidedMcNemarTest(eval_results1_, eval_results2_);
  const std::vector<std::pair<std::string, float>> expected_results = {
      // For model 1, the max accuracy is 0.8.
      // For model 2, the max accuracy is 0.4.
      // n12 = 20, n21 = 4.
      std::make_pair("0_vs_the_others@MaxAccuracy", 0.019287109),
      // n12 = 18, n21 = 0.
      std::make_pair("0_vs_the_others@Recall=0.5", 0.00369262),
      // n12 = 20, n21 = 4.
      std::make_pair("0_vs_the_others@Precision=1", 0.000138581),
      // n12 = 0, n21 = 0.
      std::make_pair("0_vs_the_others@Volume=0", 1),
      // n12 = 11, n21 = 9.
      std::make_pair("0_vs_the_others@False Positive Rate=0.5", 0.0033053),
      // n12 = 20, n21 = 2.
      std::make_pair("1_vs_the_others@MaxAccuracy", 0.01928710),
      // n12 = 15, n21 = 9.
      std::make_pair("1_vs_the_others@Recall=0.5", 0.003305),
      // n12 = 0, n21 = 0.
      std::make_pair("1_vs_the_others@Precision=1", 0.9375),
      // n12 = 0, n21 = 0.
      std::make_pair("1_vs_the_others@Volume=0", 1),
      // n12 = 14, n21 = 0.
      std::make_pair("1_vs_the_others@False Positive Rate=0.5", 0.00369),
  };

  EXPECT_EQ(result.size(), expected_results.size());
  for (int i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i].first, expected_results[i].first);
    EXPECT_NEAR(result[i].second, expected_results[i].second, 0.0001f)
        << "metric:" << result[i].first;
  }
}

TEST(PairwiseRegressiveResidualTest, Base) {
  proto::EvaluationResults eval_1;
  proto::EvaluationResults eval_2;

  utils::RandomEngine rnd;
  std::uniform_real_distribution<float> dist_01;

  const auto clear_examples = [&]() {
    eval_1.clear_sampled_predictions();
    eval_2.clear_sampled_predictions();
  };

  const auto add_example = [&](const float m1, float m2, float l) {
    auto& p1 = *eval_1.mutable_sampled_predictions()->Add();
    auto& p2 = *eval_2.mutable_sampled_predictions()->Add();
    p1.mutable_regression()->set_ground_truth(l);
    p2.mutable_regression()->set_ground_truth(l);
    p1.mutable_regression()->set_value(m1);
    p2.mutable_regression()->set_value(m2);
  };

  // Always the same residual.
  clear_examples();
  for (int l = 0; l < 50; l++) {
    add_example(l, l + 1.0f, l + 0.5f);
  }
  EXPECT_NEAR(PairwiseRegressiveResidualTest(eval_1, eval_2), 1.f, 0.0001f);

  // M2 is always better.
  clear_examples();
  for (int l = 0; l < 50; l++) {
    add_example(l + 1.0f + 0.1 * dist_01(rnd), l - 0.5f - 0.1 * dist_01(rnd),
                l);
  }
  EXPECT_NEAR(PairwiseRegressiveResidualTest(eval_1, eval_2), 0.f, 0.0001f);

  // M2 is always better. Constant residual difference.
  clear_examples();
  for (int l = 0; l < 50; l++) {
    add_example(l + 1.0f, l - 0.5f, l);
  }
  EXPECT_NEAR(PairwiseRegressiveResidualTest(eval_1, eval_2), 0.f, 0.0001f);

  // M2 is better 50% of the time, by twice the margin.
  clear_examples();
  for (int l = 0; l < 50; l++) {
    add_example(l + 0.4f, l - 0.1f, l);
  }
  for (int l = 0; l < 50; l++) {
    add_example(l + 0.1f, l - 0.2f, l);
  }
  EXPECT_NEAR(PairwiseRegressiveResidualTest(eval_1, eval_2), 0.f, 0.0001f);

  // absResidual(M1)-absResidual(M2) ~ Norma with CDF(0)=0.2 and sd=sqrt(num
  // examples).
  clear_examples();
  const auto rs_dist_mean = -boost::math::quantile(boost::math::normal(), 0.20);
  const int n = 100;
  std::normal_distribution<double> rs_dist(rs_dist_mean, std::sqrt(n));
  for (int l = 0; l < n; l++) {
    const double rs = rs_dist(rnd);
    add_example(l + 1000.f + rs, l + 1000.f, l);
  }
  EXPECT_NEAR(PairwiseRegressiveResidualTest(eval_1, eval_2), 0.365, 0.1f);
}

TEST(PairwiseRankingNDCG5Test, Base) {
  proto::EvaluationResults eval_1;
  proto::EvaluationResults eval_2;

  utils::RandomEngine rnd;
  std::uniform_real_distribution<float> dist_01;

  const auto clear_examples = [&]() {
    eval_1.clear_sampled_predictions();
    eval_2.clear_sampled_predictions();
  };

  const auto add_example = [&](const float m1, float m2, float l, int g) {
    auto& p1 = *eval_1.mutable_sampled_predictions()->Add();
    auto& p2 = *eval_2.mutable_sampled_predictions()->Add();
    p1.mutable_ranking()->set_ground_truth_relevance(l);
    p2.mutable_ranking()->set_ground_truth_relevance(l);
    p1.mutable_ranking()->set_group_id(g);
    p2.mutable_ranking()->set_group_id(g);
    p1.mutable_ranking()->set_relevance(m1);
    p2.mutable_ranking()->set_relevance(m2);
  };

  // Two perfect models.
  clear_examples();
  for (int g = 0; g < 100; g++) {
    add_example(0, 10, 1, g);
    add_example(1, 11, 2, g);
    add_example(2, 12, 3, g);
    add_example(3, 13, 4, g);
  }
  EXPECT_NEAR(PairwiseRankingNDCG5Test(eval_1, eval_2), 1.f, 0.0001f);

  // Two equally terrible models.
  clear_examples();
  for (int g = 0; g < 100; g++) {
    add_example(4, 40, 1, g);
    add_example(3, 30, 2, g);
    add_example(2, 20, 3, g);
    add_example(1, 10, 4, g);
  }
  EXPECT_NEAR(PairwiseRankingNDCG5Test(eval_1, eval_2), 1.f, 0.0001f);

  // M2 is better.
  clear_examples();
  for (int g = 0; g < 100; g++) {
    add_example(1, 1, 1, g);
    add_example(3, 2, 2, g);
    add_example(2, 3, 3, g);
    add_example(4, 4, 4, g);
  }
  EXPECT_NEAR(PairwiseRankingNDCG5Test(eval_1, eval_2), 0.f, 0.0001f);
}

}  // namespace
}  // namespace metric
}  // namespace yggdrasil_decision_forests
