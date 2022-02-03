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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_multinomial.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_utils.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {


absl::Status MultinomialLogLikelihoodLoss::Status() const {
  if (task_ != model::proto::Task::CLASSIFICATION) {
    return absl::InvalidArgumentError(
        "Multinomial log-likelihood loss is only compatible with a "
        "classification task");
  }
  return absl::OkStatus();
}

utils::StatusOr<std::vector<float>>
MultinomialLogLikelihoodLoss::InitialPredictions(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const std::vector<float>& weights) const {
  // In Friedman paper (https://statweb.stanford.edu/~jhf/ftp/trebst.pdf),
  // the initial prediction is 0 for multi-class classification (algorithm
  // 6).
  return std::vector<float>(dimension_, 0);
}

utils::StatusOr<std::vector<float>>
MultinomialLogLikelihoodLoss::InitialPredictions(
    const decision_tree::proto::LabelStatistics& label_statistics) const {
  return std::vector<float>(dimension_, 0);
}

template <typename T>
absl::Status MultinomialLogLikelihoodLoss::TemplatedUpdateGradients(
    const std::vector<T>& labels, const std::vector<float>& predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  static_assert(std::is_integral<T>::value, "Integral required.");
  // TODO(gbm): Implement thread_pool.

  // Set the gradient to:
  //   label_i - pred_i
  // where "label_i" is in {0,1}.
  absl::FixedArray<float> accumulator(gradients->size());
  const auto num_examples = labels.size();
  const auto use_hessian_gain = (*gradients)[0].hessian;
  if (gbt_config_.use_hessian_gain() && !use_hessian_gain) {
    return absl::InternalError("Hessian missing");
  }
  for (size_t example_idx = 0; example_idx < num_examples; example_idx++) {
    // Compute normalization term.
    float sum_exp = 0;
    for (int grad_idx = 0; grad_idx < gradients->size(); grad_idx++) {
      float exp_val =
          std::exp(predictions[grad_idx + example_idx * gradients->size()]);
      accumulator[grad_idx] = exp_val;
      sum_exp += exp_val;
    }
    const float normalization = 1.f / sum_exp;
    // Update gradient.
    const int label_cat = labels[example_idx];
    for (int grad_idx = 0; grad_idx < gradients->size(); grad_idx++) {
      const float label = (label_cat == (grad_idx + 1)) ? 1.f : 0.f;
      DCheckIsFinite(label);
      const float prediction = accumulator[grad_idx] * normalization;
      DCheckIsFinite(prediction);
      const float grad = label - prediction;
      const float abs_grad = std::abs(grad);
      DCheckIsFinite(grad);
      (*(*gradients)[grad_idx].gradient)[example_idx] = grad;
      if (use_hessian_gain) {
        (*(*gradients)[grad_idx].hessian)[example_idx] =
            abs_grad * (1 - abs_grad);
        DCheckIsFinite(abs_grad * (1 - abs_grad));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status MultinomialLogLikelihoodLoss::UpdateGradients(
    const std::vector<int32_t>& labels, const std::vector<float>& predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedUpdateGradients(labels, predictions, ranking_index, gradients,
                                  random, thread_pool);
}

absl::Status MultinomialLogLikelihoodLoss::UpdateGradients(
    const std::vector<int16_t>& labels, const std::vector<float>& predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedUpdateGradients(labels, predictions, ranking_index, gradients,
                                  random, thread_pool);
}

decision_tree::CreateSetLeafValueFunctor
MultinomialLogLikelihoodLoss::SetLeafFunctor(
    const std::vector<float>& predictions,
    const std::vector<GradientData>& gradients, const int label_col_idx) const {
  return
      [this, &predictions, label_col_idx](
          const dataset::VerticalDataset& train_dataset,
          const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
          const std::vector<float>& weights,
          const model::proto::TrainingConfig& config,
          const model::proto::TrainingConfigLinking& config_link,
          decision_tree::NodeWithChildren* node) {
        return SetLeaf(train_dataset, selected_examples, weights, config,
                       config_link, predictions, label_col_idx, node);
      };
}

void MultinomialLogLikelihoodLoss::SetLeaf(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const std::vector<float>& predictions, const int label_col_idx,
    decision_tree::NodeWithChildren* node) const {
  // Initialize the distribution (as the "top_value" is overridden right
  // after.
  if (!gbt_config_.use_hessian_gain()) {
    decision_tree::SetRegressionLabelDistribution(
        train_dataset, selected_examples, weights, config_link,
        node->mutable_node());
  }

  // Set the value of the leaf to:
  //  (dim-1) / dim * ( \sum_i weight[i] grad[i] ) / (\sum_i |grad[i]| *
  //  (1-|grad[i]|))
  //
  // Note: The leaf value does not depend on the label value (directly).
  const auto& grad =
      train_dataset
          .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
              config_link.label())
          ->values();
  double numerator = 0;
  double denominator = 0;
  double sum_weights = 0;
  for (const auto example_idx : selected_examples) {
    const float weight = weights[example_idx];
    numerator += weight * grad[example_idx];
    const float abs_grad = std::abs(grad[example_idx]);
    denominator += weight * abs_grad * (1 - abs_grad);
    sum_weights += weight;
    DCheckIsFinite(numerator);
    DCheckIsFinite(denominator);
  }

  if (denominator <= kMinHessianForNewtonStep) {
    denominator = kMinHessianForNewtonStep;
  }

  if (gbt_config_.use_hessian_gain()) {
    auto* reg = node->mutable_node()->mutable_regressor();
    reg->set_sum_gradients(numerator);
    reg->set_sum_hessians(denominator);
    reg->set_sum_weights(sum_weights);
  }

  numerator *= dimension_ - 1;
  denominator *= dimension_;
  const auto leaf_value =
      gbt_config_.shrinkage() *
      static_cast<float>(decision_tree::l1_threshold(
                             numerator, gbt_config_.l1_regularization()) /
                         (denominator + gbt_config_.l2_regularization()));
  DCheckIsFinite(leaf_value);

  node->mutable_node()->mutable_regressor()->set_top_value(
      utils::clamp(leaf_value, -gbt_config_.clamp_leaf_logit(),
                   gbt_config_.clamp_leaf_logit()));
}

absl::Status MultinomialLogLikelihoodLoss::UpdatePredictions(
    const std::vector<const decision_tree::DecisionTree*>& new_trees,
    const dataset::VerticalDataset& dataset, std::vector<float>* predictions,
    double* mean_abs_prediction) const {
  if (new_trees.size() != dimension_) {
    return absl::InternalError("Wrong number of trees");
  }
  UpdatePredictionWithMultipleUnivariateTrees(dataset, new_trees, predictions,
                                              mean_abs_prediction);
  return absl::OkStatus();
}

std::vector<std::string> MultinomialLogLikelihoodLoss::SecondaryMetricNames()
    const {
  return {"accuracy"};
}

template <typename T>
absl::Status MultinomialLogLikelihoodLoss::TemplatedLoss(
    const std::vector<T>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index, float* loss_value,
    std::vector<float>* secondary_metric,
    utils::concurrency::ThreadPool* thread_pool) const {
  double sum_loss = 0;
  double count_correct_predictions = 0;
  double sum_weights = 0;

  if (weights.empty()) {
    for (size_t example_idx = 0; example_idx < labels.size(); example_idx++) {
      const int label = labels[example_idx];

      int predicted_class = -1;
      float predicted_class_exp_value = 0;
      float sum_exp = 0;
      for (int grad_idx = 0; grad_idx < dimension_; grad_idx++) {
        const float exp_val =
            std::exp(predictions[grad_idx + example_idx * dimension_]);
        sum_exp += exp_val;
        DCheckIsFinite(sum_exp);
        if (exp_val > predicted_class_exp_value) {
          predicted_class_exp_value = exp_val;
          predicted_class = grad_idx + 1;
        }
      }
      if (label == predicted_class) {
        count_correct_predictions += 1;
      }
      // Loss:
      //   - log(predict_proba[true_label])
      const float tree_label_exp_value =
          std::exp(predictions[(label - 1) + example_idx * dimension_]);
      sum_loss -= std::log(tree_label_exp_value / sum_exp);
      DCheckIsFinite(sum_loss);
      DCheckIsFinite(sum_weights);
    }
    sum_weights += labels.size();
  } else {
    for (size_t example_idx = 0; example_idx < labels.size(); example_idx++) {
      const int label = labels[example_idx];
      const float weight = weights[example_idx];
      sum_weights += weight;

      int predicted_class = -1;
      float predicted_class_exp_value = 0;
      float sum_exp = 0;
      for (int grad_idx = 0; grad_idx < dimension_; grad_idx++) {
        const float exp_val =
            std::exp(predictions[grad_idx + example_idx * dimension_]);
        sum_exp += exp_val;
        DCheckIsFinite(sum_exp);
        if (exp_val > predicted_class_exp_value) {
          predicted_class_exp_value = exp_val;
          predicted_class = grad_idx + 1;
        }
      }
      if (label == predicted_class) {
        count_correct_predictions += weight;
      }
      // Loss:
      //   - log(predict_proba[true_label])
      const float tree_label_exp_value =
          std::exp(predictions[(label - 1) + example_idx * dimension_]);
      sum_loss -= weight * std::log(tree_label_exp_value / sum_exp);
      DCheckIsFinite(sum_loss);
      DCheckIsFinite(sum_weights);
    }
  }

  secondary_metric->resize(1);
  if (sum_weights > 0) {
    *loss_value = static_cast<float>(sum_loss / sum_weights);
    DCheckIsFinite(*loss_value);
    (*secondary_metric)[kBinomialLossSecondaryMetricClassificationIdx] =
        static_cast<float>(count_correct_predictions / sum_weights);
  } else {
    *loss_value =
        (*secondary_metric)[kBinomialLossSecondaryMetricClassificationIdx] =
            std::numeric_limits<float>::quiet_NaN();
  }
  return absl::OkStatus();
}

absl::Status MultinomialLogLikelihoodLoss::Loss(
    const std::vector<int32_t>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index, float* loss_value,
    std::vector<float>* secondary_metric,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedLoss(labels, predictions, weights, ranking_index, loss_value,
                       secondary_metric, thread_pool);
}

absl::Status MultinomialLogLikelihoodLoss::Loss(
    const std::vector<int16_t>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index, float* loss_value,
    std::vector<float>* secondary_metric,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedLoss(labels, predictions, weights, ranking_index, loss_value,
                       secondary_metric, thread_pool);
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
