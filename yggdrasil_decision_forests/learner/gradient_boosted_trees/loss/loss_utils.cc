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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_utils.h"

#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

absl::Status SetLeafValueWithNewtonRaphsonStep(
    const proto::GradientBoostedTreesTrainingConfig& gbt_config_,
    const decision_tree::proto::LabelStatistics& label_statistics,
    decision_tree::proto::Node* node) {
  node->set_num_pos_training_examples_without_weight(
      label_statistics.num_examples());

  double sum_gradients = 0;
  double sum_hessians = 0;
  double sum_weights = 0;

  switch (label_statistics.type_case()) {
    case decision_tree::proto::LabelStatistics::kRegressionWithHessian:
      sum_weights = label_statistics.regression_with_hessian().labels().count();
      sum_gradients = label_statistics.regression_with_hessian().labels().sum();
      sum_hessians = label_statistics.regression_with_hessian().sum_hessian();
      break;

    default:
      return absl::InternalError("No hessian data available");
  }

  if (sum_hessians <= kMinHessianForNewtonStep) {
    sum_hessians = kMinHessianForNewtonStep;
  }

  const auto leaf_value =
      gbt_config_.shrinkage() *
      static_cast<float>(decision_tree::l1_threshold(
                             sum_gradients, gbt_config_.l1_regularization()) /
                         (sum_hessians + gbt_config_.l2_regularization()));

  node->mutable_regressor()->set_top_value(
      utils::clamp(leaf_value, -gbt_config_.clamp_leaf_logit(),
                   gbt_config_.clamp_leaf_logit()));

  return absl::OkStatus();
}

void UpdatePredictionWithSingleUnivariateTree(
    const dataset::VerticalDataset& dataset,
    const decision_tree::DecisionTree& tree, std::vector<float>* predictions,
    double* mean_abs_prediction) {
  double sum_abs_predictions = 0;
  for (row_t example_idx = 0; example_idx < dataset.nrow(); example_idx++) {
    const auto& leaf = tree.GetLeaf(dataset, example_idx);
    (*predictions)[example_idx] += leaf.regressor().top_value();
    sum_abs_predictions += std::abs(leaf.regressor().top_value());
  }
  if (mean_abs_prediction) {
    *mean_abs_prediction = sum_abs_predictions / dataset.nrow();
  }
}

void UpdatePredictionWithMultipleUnivariateTrees(
    const dataset::VerticalDataset& dataset,
    const std::vector<const decision_tree::DecisionTree*>& trees,
    std::vector<float>* predictions, double* mean_abs_prediction) {
  double sum_abs_predictions = 0;
  const int num_trees = trees.size();
  for (row_t example_idx = 0; example_idx < dataset.nrow(); example_idx++) {
    for (int grad_idx = 0; grad_idx < num_trees; grad_idx++) {
      const auto& leaf = trees[grad_idx]->GetLeaf(dataset, example_idx);
      (*predictions)[grad_idx + example_idx * num_trees] +=
          leaf.regressor().top_value();
      sum_abs_predictions += std::abs(leaf.regressor().top_value());
    }
  }
  if (mean_abs_prediction) {
    *mean_abs_prediction = sum_abs_predictions / dataset.nrow();
  }
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
