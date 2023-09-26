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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_utils.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

// Set the value of a leaf node.
template <bool weighted>
absl::Status SetLeafValueWithNewtonRaphsonStep(
    const proto::GradientBoostedTreesTrainingConfig& gbt_config,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const GradientData& gradients,
    decision_tree::NodeWithChildren* node) {
  if constexpr (weighted) {
    DCHECK_LE(selected_examples.size(), weights.size());
  }
  if constexpr (!weighted) {
    DCHECK(weights.empty());
  }

  const auto& gradient = gradients.gradient;
  const auto& hessian = gradients.hessian;

  const bool use_hessian_gain = gbt_config.use_hessian_gain();

  double sum_weighted_gradient = 0;
  double sum_weighted_square_gradient = 0;
  double sum_weighted_hessian = 0;
  double sum_weights = 0;
  if constexpr (!weighted) {
    sum_weights = selected_examples.size();
  }

  for (const auto example_idx : selected_examples) {
    const float unit_gradient = gradient[example_idx];
    const float unit_hessian = hessian[example_idx];

    DCheckIsFinite(unit_gradient);
    DCheckIsFinite(unit_hessian);

    if constexpr (weighted) {
      const float weight = weights[example_idx];
      sum_weighted_gradient += weight * unit_gradient;
      sum_weighted_hessian += weight * unit_hessian;
      if (!use_hessian_gain) {
        sum_weighted_square_gradient += weight * unit_gradient * unit_gradient;
      }
      sum_weights += weight;
    } else {
      sum_weighted_gradient += unit_gradient;
      sum_weighted_hessian += unit_hessian;
      if (!use_hessian_gain) {
        sum_weighted_square_gradient += unit_gradient * unit_gradient;
      }
    }
  }

  DCheckIsFinite(sum_weighted_gradient);
  DCheckIsFinite(sum_weighted_hessian);

  if (sum_weighted_hessian <= kMinHessianForNewtonStep) {
    sum_weighted_hessian = kMinHessianForNewtonStep;
  }

  auto* reg = node->mutable_node()->mutable_regressor();

  if (use_hessian_gain) {
    reg->set_sum_gradients(sum_weighted_gradient);
    reg->set_sum_hessians(sum_weighted_hessian);
    reg->set_sum_weights(sum_weights);
  } else {
    reg->mutable_distribution()->set_sum(sum_weighted_gradient);
    reg->mutable_distribution()->set_sum_squares(sum_weighted_square_gradient);
    reg->mutable_distribution()->set_count(sum_weights);
  }

  const double numerator = decision_tree::l1_threshold(
      sum_weighted_gradient, gbt_config.l1_regularization());
  const double denominator =
      sum_weighted_hessian + gbt_config.l2_regularization();
  float value = gbt_config.shrinkage() * numerator / denominator;
  value = utils::clamp(value, -gbt_config.clamp_leaf_logit(),
                       gbt_config.clamp_leaf_logit());
  reg->set_top_value(value);
  return absl::OkStatus();
}

decision_tree::CreateSetLeafValueFunctor
SetLeafValueWithNewtonRaphsonStepFunctor(
    const proto::GradientBoostedTreesTrainingConfig& gbt_config,
    const GradientData& gradients) {
  return [&gradients, &gbt_config](
             const dataset::VerticalDataset& train_dataset,
             const std::vector<UnsignedExampleIdx>& selected_examples,
             const std::vector<float>& weights,
             const model::proto::TrainingConfig& config,
             const model::proto::TrainingConfigLinking& config_link,
             decision_tree::NodeWithChildren* node) -> absl::Status {
    if (weights.empty()) {
      return SetLeafValueWithNewtonRaphsonStep</*weighted=*/false>(
          gbt_config, selected_examples, weights, gradients, node);
    } else {
      return SetLeafValueWithNewtonRaphsonStep</*weighted=*/true>(
          gbt_config, selected_examples, weights, gradients, node);
    }
  };
}

absl::Status SetLeafValueWithNewtonRaphsonStep(
    const proto::GradientBoostedTreesTrainingConfig& gbt_config_,
    const decision_tree::proto::LabelStatistics& label_statistics,
    decision_tree::proto::Node* node) {
  node->set_num_pos_training_examples_without_weight(
      label_statistics.num_examples());

  double sum_gradients = 0;
  double sum_hessians = 0;

  switch (label_statistics.type_case()) {
    case decision_tree::proto::LabelStatistics::kRegressionWithHessian:
      sum_gradients = label_statistics.regression_with_hessian().labels().sum();
      sum_hessians = label_statistics.regression_with_hessian().sum_hessian();
      break;

    default:
      return absl::InternalError("No hessian data available");
  }

  if (sum_hessians <= kMinHessianForNewtonStep) {
    sum_hessians = kMinHessianForNewtonStep;
  }

  const double numerator = decision_tree::l1_threshold(
      sum_gradients, gbt_config_.l1_regularization());
  const double denominator = sum_hessians + gbt_config_.l2_regularization();
  float value = gbt_config_.shrinkage() * numerator / denominator;
  value = utils::clamp(value, -gbt_config_.clamp_leaf_logit(),
                       gbt_config_.clamp_leaf_logit());
  node->mutable_regressor()->set_top_value(value);
  return absl::OkStatus();
}

void UpdatePredictionWithSingleUnivariateTree(
    const dataset::VerticalDataset& dataset,
    const decision_tree::DecisionTree& tree, std::vector<float>* predictions,
    double* mean_abs_prediction) {
  double sum_abs_predictions = 0;
  const UnsignedExampleIdx num_examples = dataset.nrow();
  for (UnsignedExampleIdx example_idx = 0; example_idx < num_examples;
       example_idx++) {
    const auto& leaf = tree.GetLeaf(dataset, example_idx);
    (*predictions)[example_idx] += leaf.regressor().top_value();
    sum_abs_predictions += std::abs(leaf.regressor().top_value());
  }
  if (mean_abs_prediction) {
    *mean_abs_prediction = sum_abs_predictions / num_examples;
  }
}

void UpdatePredictionWithMultipleUnivariateTrees(
    const dataset::VerticalDataset& dataset,
    const std::vector<const decision_tree::DecisionTree*>& trees,
    std::vector<float>* predictions, double* mean_abs_prediction) {
  double sum_abs_predictions = 0;
  const int num_trees = trees.size();
  const UnsignedExampleIdx num_examples = dataset.nrow();
  for (UnsignedExampleIdx example_idx = 0; example_idx < num_examples;
       example_idx++) {
    for (int grad_idx = 0; grad_idx < num_trees; grad_idx++) {
      const auto& leaf = trees[grad_idx]->GetLeaf(dataset, example_idx);
      (*predictions)[grad_idx + example_idx * num_trees] +=
          leaf.regressor().top_value();
      sum_abs_predictions += std::abs(leaf.regressor().top_value());
    }
  }
  if (mean_abs_prediction) {
    if (num_examples == 0) {
      *mean_abs_prediction = 0;
    } else {
      *mean_abs_prediction = sum_abs_predictions / num_examples;
    }
  }
}

absl::Status UpdatePredictions(
    const std::vector<const decision_tree::DecisionTree*>& trees,
    const dataset::VerticalDataset& dataset, std::vector<float>* predictions,
    double* mean_abs_prediction) {
  if (trees.size() == 1) {
    UpdatePredictionWithSingleUnivariateTree(dataset, *trees.front(),
                                             predictions, mean_abs_prediction);
  } else {
    UpdatePredictionWithMultipleUnivariateTrees(dataset, trees, predictions,
                                                mean_abs_prediction);
  }
  return absl::OkStatus();
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
