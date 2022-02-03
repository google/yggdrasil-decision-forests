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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_binary_focal.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

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

BinaryFocalLoss::BinaryFocalLoss(
    const proto::GradientBoostedTreesTrainingConfig& gbt_config,
    const model::proto::Task task, const dataset::proto::Column& label_column)
    : BinomialLogLikelihoodLoss(gbt_config, task, label_column),
      gamma_(gbt_config.binary_focal_loss_options().misprediction_exponent()),
      alpha_(gbt_config.binary_focal_loss_options()
                 .positive_sample_coefficient()) {}

absl::Status BinaryFocalLoss::Status() const {
  if (task_ != model::proto::Task::CLASSIFICATION)
    return absl::InvalidArgumentError(
        "Focal loss is only compatible with a classification task");
  if (label_column_.categorical().number_of_unique_values() != 3)
    return absl::InvalidArgumentError(
        "Focal loss is only compatible with a BINARY classification task");
  return absl::OkStatus();
}

// Local helper functions for calculating the (slightly more involved)
// focal loss gradients and hessians at a common place to avoid code duplication
struct FocalLossBasicData {
  float y;      // Label as from {-1, 1} set to follow paper's notation.
  float label;  // Label as from {0, 1} set.
  float pt;     // Probability of "being right".
  // Log of prob of being right. Calculated directly from log odds.
  float log_pt;
  float mispred;  // Probability of miss-prediction.
  float at;       // Sample weight (alpha_t, depends on ground truth label)
};

struct FocalLossGradientData {
  FocalLossBasicData basic;
  float gradient;
  float term1;  // The first derivative term of the first derivative
  float term2;  // The second derivative term of the first derivative
};

// Calculate log(pt) for formula (5) from page 3 and other reusable stuff from
// https://arxiv.org/pdf/1708.02002.pdf
// Note: 'prediction' is in log odds space.
FocalLossBasicData CalculateFocalLossBasic(bool is_positive, float prediction,
                                           float gamma, float alpha) {
  const float label = is_positive ? 1.f : 0.f;
  const float y = label * 2 - 1;
  const float prediction_proba = 1.f / (1.f + std::exp(-prediction));
  DCheckIsFinite(prediction_proba);
  // Pt is probability of predicting the right label (1-p for negative labels).
  const float pt = prediction_proba * y + 1.f - label;
  // Note: It is better to calculate log(pt) this way, to avoid NaNs when
  // pt is very close to zero.
  // Why is calculation of log_pt correct?
  // Let d denote "prediction' in log odds space.
  // If label = 0:
  //   log(pt) = log(1-p) = log(1-1/(1+exp(-d))) =
  //   = log([1 + exp(-d) - 1] / [1 + exp(-d)]) =
  //   = log(exp(-d)) - log(1 + exp(-d)) =
  //   = log(1/exp(d)) - log([exp(d) + 1] / exp(d)) =
  //   = -log(exp(d)) - log(exp(d) + 1) + log(exp(d)) =
  //   = - log(1+ exp(d))
  //   = label * d - log(1 + exp(d))         Q.E.D.
  //
  // If label = 1:
  //   log(pt) = log(p) = log(1/(1+exp(-d))) =
  //   = -log(1+exp(-d)) = -log([exp(d) + 1] / exp(d)) =
  //   = -[log(exp(d) + 1) - log(exp(d))] =
  //   = -log(exp(d) + 1) + d =
  //   = label * d - log(1+ exp(d))          Q.E.D.
  const float log_pt =
      label * prediction - std::log(1.0f + std::exp(prediction));
  const float mispred = 1.0f - pt;
  const float at = is_positive ? alpha : (1.0f - alpha);
  return (FocalLossBasicData){y, label, pt, log_pt, mispred, at};
}

// We have a separate function to only calculate what's necessary for gradient
// (and not for the hessian - to save time).
FocalLossGradientData CalculateFocalLossGradient(bool is_positive,
                                                 float prediction, float gamma,
                                                 float alpha) {
  const FocalLossBasicData& basic =
      CalculateFocalLossBasic(is_positive, prediction, gamma, alpha);
  // We calculate and store the two terms of the first derivative separately
  // to be reused in the hessian (when needed)
  const float term1 = basic.at * basic.y * std::pow(basic.mispred, gamma);
  const float term2 = gamma * basic.pt * basic.log_pt - basic.mispred;
  const float gradient = -term1 * term2;
  return (FocalLossGradientData){basic, gradient, term1, term2};
}

float CalculateFocalLossHessian(FocalLossGradientData gradient_data,
                                float gamma, float alpha) {
  const FocalLossBasicData& basic = gradient_data.basic;
  if (basic.mispred <= std::numeric_limits<float>::epsilon()) {
    return 0.0f;
  }
  // Derivative of term1 (see term1 in CalculateFocalLossGradient)
  const float dterm1 =
      -basic.at * basic.y * gamma * std::pow(basic.mispred, gamma - 1.0f);
  // Derivative of term2 (see term2 in CalculateFocalLossGradient)
  const float dterm2 = gamma * basic.log_pt + gamma + 1.0f;

  const float hessian =
      basic.y * (basic.pt * basic.mispred) *
      (gradient_data.term1 * dterm2 + dterm1 * gradient_data.term2);

  return hessian;
}

// Local help functions end.

template <typename T>
absl::Status BinaryFocalLoss::TemplatedUpdateGradients(
    const std::vector<T>& labels, const std::vector<float>& predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  static_assert(std::is_integral<T>::value, "Integral required.");
  // TODO(gbm): Implement thread_pool.

  if (gradients->size() != 1) {
    return absl::InternalError("Wrong gradient shape");
  }

  std::vector<float>& gradient_data = *(*gradients)[0].gradient;
  std::vector<float>* hessian_data = (*gradients)[0].hessian;
  if (gbt_config_.use_hessian_gain() && hessian_data == nullptr) {
    return absl::InternalError("Hessian missing");
  }
  const size_t num_examples = labels.size();

  if (thread_pool == nullptr) {
    TemplatedUpdateGradientsImp(labels, predictions, 0, num_examples, gamma_,
                                alpha_, &gradient_data, hessian_data);
  } else {
    decision_tree::ConcurrentForLoop(
        thread_pool->num_threads(), thread_pool, num_examples,
        [this, &labels, &predictions, &gradient_data, hessian_data](
            size_t block_idx, size_t begin_idx, size_t end_idx) -> void {
          TemplatedUpdateGradientsImp(labels, predictions, begin_idx, end_idx,
                                      gamma_, alpha_, &gradient_data,
                                      hessian_data);
        });
  }

  return absl::OkStatus();
}

absl::Status BinaryFocalLoss::UpdateGradients(
    const std::vector<int32_t>& labels, const std::vector<float>& predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedUpdateGradients(labels, predictions, ranking_index, gradients,
                                  random, thread_pool);
}

absl::Status BinaryFocalLoss::UpdateGradients(
    const std::vector<int16_t>& labels, const std::vector<float>& predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedUpdateGradients(labels, predictions, ranking_index, gradients,
                                  random, thread_pool);
}

template <typename T>
void BinaryFocalLoss::TemplatedUpdateGradientsImp(
    const std::vector<T>& labels, const std::vector<float>& predictions,
    size_t begin_example_idx, size_t end_example_idx, float gamma, float alpha,
    std::vector<float>* gradient_data, std::vector<float>* hessian_data) {
  for (size_t example_idx = begin_example_idx; example_idx < end_example_idx;
       example_idx++) {
    const bool is_positive = (labels[example_idx] == 2);
    const float prediction = predictions[example_idx];
    DCheckIsFinite(prediction);
    const FocalLossGradientData calculated_gradient_data =
        CalculateFocalLossGradient(is_positive, prediction, gamma, alpha);
    DCHECK(is_positive || calculated_gradient_data.gradient <= 0.0)
        << is_positive << ", " << calculated_gradient_data.gradient;
    DCHECK(!is_positive || calculated_gradient_data.gradient >= 0.0)
        << is_positive << ", " << calculated_gradient_data.gradient;
    DCheckIsFinite(calculated_gradient_data.gradient);
    (*gradient_data)[example_idx] = calculated_gradient_data.gradient;

    if (hessian_data) {
      const float hessian =
          CalculateFocalLossHessian(calculated_gradient_data, gamma, alpha);
      DCheckIsFinite(hessian);
      (*hessian_data)[example_idx] = hessian;
    }
  }
}

decision_tree::CreateSetLeafValueFunctor BinaryFocalLoss::SetLeafFunctor(
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

void BinaryFocalLoss::SetLeaf(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const std::vector<float>& predictions, const int label_col_idx,
    decision_tree::NodeWithChildren* node) const {
  if (!gbt_config_.use_hessian_gain()) {
    decision_tree::SetRegressionLabelDistribution(
        train_dataset, selected_examples, weights, config_link,
        node->mutable_node());
    // Even if "use_hessian_gain" is not enabled for the splits. We use a
    // Newton step in the leaves i.e. if "use_hessian_gain" is false, we need
    // all the information.
  }

  // Set the value of the leaf to the sum(gradients) / sum(hessians)
  const auto* labels =
      train_dataset.ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
          label_col_idx);
  double numerator = 0;
  double denominator = 0;
  double sum_weights = 0;
  for (const auto example_idx : selected_examples) {
    const float weight = weights[example_idx];
    const bool is_positive = labels->values()[example_idx] == 2;
    const float prediction = predictions[example_idx];
    const FocalLossGradientData gradient_data =
        CalculateFocalLossGradient(is_positive, prediction, gamma_, alpha_);
    DCheckIsFinite(gradient_data.gradient);

    const double hessian =
        CalculateFocalLossHessian(gradient_data, gamma_, alpha_);
    DCheckIsFinite(hessian);
    numerator += weight * gradient_data.gradient;
    denominator += weight * hessian;
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

  const auto leaf_value =
      gbt_config_.shrinkage() *
      static_cast<float>(decision_tree::l1_threshold(
                             numerator, gbt_config_.l1_regularization()) /
                         (denominator + gbt_config_.l2_regularization()));

  node->mutable_node()->mutable_regressor()->set_top_value(
      utils::clamp(leaf_value, -gbt_config_.clamp_leaf_logit(),
                   gbt_config_.clamp_leaf_logit()));
}

absl::Status BinaryFocalLoss::UpdatePredictions(
    const std::vector<const decision_tree::DecisionTree*>& new_trees,
    const dataset::VerticalDataset& dataset, std::vector<float>* predictions,
    double* mean_abs_prediction) const {
  if (new_trees.size() != 1) {
    return absl::InternalError("Wrong number of trees");
  }
  UpdatePredictionWithSingleUnivariateTree(dataset, *new_trees.front(),
                                           predictions, mean_abs_prediction);
  return absl::OkStatus();
}

std::vector<std::string> BinaryFocalLoss::SecondaryMetricNames() const {
  return {"accuracy"};
}

template <bool use_weights, typename T>
void BinaryFocalLoss::TemplatedLossImp(
    const std::vector<T>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights, size_t begin_example_idx,
    size_t end_example_idx, float gamma, float alpha,
    double* __restrict sum_loss, double* __restrict count_correct_predictions,
    double* __restrict sum_weights) {
  for (size_t example_idx = begin_example_idx; example_idx < end_example_idx;
       example_idx++) {
    const bool pos_label = labels[example_idx] == 2;
    const float prediction = predictions[example_idx];
    const FocalLossBasicData& basic =
        CalculateFocalLossBasic(pos_label, prediction, gamma, alpha);
    const bool pos_prediction = prediction >= 0;
    if constexpr (use_weights) {
      const float weight = weights[example_idx];
      *sum_weights += weight;
      if (pos_label == pos_prediction) {
        *count_correct_predictions += weight;
      }
      *sum_loss -=
          weight * basic.at * std::pow(basic.mispred, gamma) * basic.log_pt;
    } else {
      if (pos_label == pos_prediction) {
        *count_correct_predictions += 1.;
      }
      *sum_loss -= basic.at * std::pow(basic.mispred, gamma) * basic.log_pt;
      DCheckIsFinite(*sum_loss);
    }
    DCheckIsFinite(*sum_loss);
  }
  if constexpr (!use_weights) {
    *sum_weights += end_example_idx - begin_example_idx;
  }
}

template <typename T>
absl::Status BinaryFocalLoss::TemplatedLoss(
    const std::vector<T>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index, float* loss_value,
    std::vector<float>* secondary_metric,
    utils::concurrency::ThreadPool* thread_pool) const {
  double sum_loss = 0;
  double count_correct_predictions = 0;
  double sum_weights = 0;

  if (thread_pool == nullptr) {
    if (weights.empty()) {
      TemplatedLossImp<false>(labels, predictions, weights, 0, labels.size(),
                              gamma_, alpha_, &sum_loss,
                              &count_correct_predictions, &sum_weights);
    } else {
      TemplatedLossImp<true>(labels, predictions, weights, 0, labels.size(),
                             gamma_, alpha_, &sum_loss,
                             &count_correct_predictions, &sum_weights);
    }
  } else {
    const auto num_threads = thread_pool->num_threads();

    struct PerThread {
      double sum_loss = 0;
      double count_correct_predictions = 0;
      double sum_weights = 0;
    };
    std::vector<PerThread> per_threads(num_threads);

    decision_tree::ConcurrentForLoop(
        num_threads, thread_pool, labels.size(),
        [this, &labels, &predictions, &per_threads, &weights](
            size_t block_idx, size_t begin_idx, size_t end_idx) -> void {
          auto& block = per_threads[block_idx];

          if (weights.empty()) {
            TemplatedLossImp<false>(labels, predictions, weights, begin_idx,
                                    end_idx, gamma_, alpha_, &block.sum_loss,
                                    &block.count_correct_predictions,
                                    &block.sum_weights);
          } else {
            TemplatedLossImp<true>(labels, predictions, weights, begin_idx,
                                   end_idx, gamma_, alpha_, &block.sum_loss,
                                   &block.count_correct_predictions,
                                   &block.sum_weights);
          }
        });

    for (const auto& block : per_threads) {
      sum_loss += block.sum_loss;
      sum_weights += block.sum_weights;
      count_correct_predictions += block.count_correct_predictions;
    }
  }

  secondary_metric->resize(1);
  if (sum_weights > 0) {
    *loss_value = static_cast<float>(sum_loss / sum_weights);
    (*secondary_metric)[kBinomialLossSecondaryMetricClassificationIdx] =
        static_cast<float>(count_correct_predictions / sum_weights);
  } else {
    *loss_value =
        (*secondary_metric)[kBinomialLossSecondaryMetricClassificationIdx] =
            std::numeric_limits<float>::quiet_NaN();
  }
  return absl::OkStatus();
}

absl::Status BinaryFocalLoss::Loss(
    const std::vector<int32_t>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index, float* loss_value,
    std::vector<float>* secondary_metric,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedLoss(labels, predictions, weights, ranking_index, loss_value,
                       secondary_metric, thread_pool);
}

absl::Status BinaryFocalLoss::Loss(
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
