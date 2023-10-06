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

// Implementation of the "mean average error" a.k.a. laplace error.
//
// loss = \sum abs(x_i - y_i); where x_i are the predictions and y_i the labels.
// gradient = (x_i>=y_i) ? +1 ; -1
// hessian = 1
// initial_predictions = median[y_i]
//
// Note: The hessian should be zero, but this is not allowed with a Newton's
// method (division by zero). Different libraries have different tricks e.g.
// R.gbm set the leaves to be the median of the gradient value or set the
// hessian to 1. We use this last option.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_MEAN_AVERAGE_ERROR_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_MEAN_AVERAGE_ERROR_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"  // IWYU pragma: keep
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests::model::gradient_boosted_trees {

class MeanAverageErrorLoss : public AbstractLoss {
 public:
  // For unit testing.
  using AbstractLoss::Loss;
  using AbstractLoss::UpdateGradients;

  MeanAverageErrorLoss(
      const proto::GradientBoostedTreesTrainingConfig& gbt_config,
      model::proto::Task task, const dataset::proto::Column& label_column)
      : AbstractLoss(gbt_config, task, label_column) {}

  absl::Status Status() const override;

  LossShape Shape() const override {
    return {.gradient_dim = 1, .prediction_dim = 1};
  };

  absl::StatusOr<std::vector<float>> InitialPredictions(
      const dataset::VerticalDataset& dataset, int label_col_idx,
      const std::vector<float>& weights) const override;

  absl::StatusOr<std::vector<float>> InitialPredictions(
      const decision_tree::proto::LabelStatistics& label_statistics)
      const override;

  absl::Status UpdateGradients(
      const std::vector<float>& labels, const std::vector<float>& predictions,
      const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
      utils::RandomEngine* random,
      utils::concurrency::ThreadPool* thread_pool) const override;

  std::vector<std::string> SecondaryMetricNames() const override;

  absl::StatusOr<LossResults> Loss(
      const std::vector<float>& labels, const std::vector<float>& predictions,
      const std::vector<float>& weights,
      const RankingGroupsIndices* ranking_index,
      utils::concurrency::ThreadPool* thread_pool) const override;
};

REGISTER_AbstractGradientBoostedTreeLoss(MeanAverageErrorLoss,
                                         "MEAN_AVERAGE_ERROR");

}  // namespace yggdrasil_decision_forests::model::gradient_boosted_trees

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_MEAN_AVERAGE_ERROR_H_
