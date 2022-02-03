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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_binomial.h"

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

absl::Status BinomialLogLikelihoodLoss::Status() const {
  if (task_ != model::proto::Task::CLASSIFICATION)
    return absl::InvalidArgumentError(
        "Binomial log likelihood loss is only compatible with a "
        "classification task");
  if (label_column_.categorical().number_of_unique_values() != 3)
    return absl::InvalidArgumentError(
        "Binomial log likelihood loss is only compatible with a BINARY "
        "classification task");
  return absl::OkStatus();
}

utils::StatusOr<std::vector<float>>
BinomialLogLikelihoodLoss::InitialPredictions(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const std::vector<float>& weights) const {
  // Return: log(y/(1-y)) with y the ratio of positive labels.
  double weighted_sum_positive = 0;
  double sum_weights = 0;
  const auto* labels =
      dataset.ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
          label_col_idx);
  for (row_t example_idx = 0; example_idx < dataset.nrow(); example_idx++) {
    sum_weights += weights[example_idx];
    weighted_sum_positive +=
        weights[example_idx] * (labels->values()[example_idx] == 2);
  }
  const auto ratio_positive = weighted_sum_positive / sum_weights;
  if (ratio_positive == 0.0) {
    return std::vector<float>{-std::numeric_limits<float>::max()};
  } else if (ratio_positive == 1.0) {
    return std::vector<float>{std::numeric_limits<float>::max()};
  } else {
    return std::vector<float>{
        static_cast<float>(std::log(ratio_positive / (1. - ratio_positive)))};
  }
}

utils::StatusOr<std::vector<float>>
BinomialLogLikelihoodLoss::InitialPredictions(
    const decision_tree::proto::LabelStatistics& label_statistics) const {
  // Return: log(y/(1-y)) with y the ratio of positive labels.
  if (label_statistics.classification().labels().counts_size() != 3) {
    return absl::InternalError(absl::Substitute(
        "The binary loglikelihood loss expects 2 classes i.e. 3 unique values "
        "(including the OOV item). Got $0 unique values instead.",
        label_statistics.classification().labels().counts_size()));
  }
  const auto ratio_positive =
      label_statistics.classification().labels().counts(2) /
      label_statistics.classification().labels().sum();
  if (ratio_positive == 0.0) {
    return std::vector<float>{-std::numeric_limits<float>::max()};
  } else if (ratio_positive == 1.0) {
    return std::vector<float>{std::numeric_limits<float>::max()};
  } else {
    return std::vector<float>{
        static_cast<float>(std::log(ratio_positive / (1. - ratio_positive)))};
  }
}

template <typename T>
void BinomialLogLikelihoodLoss::TemplatedUpdateGradientsImp(
    const std::vector<T>& labels, const std::vector<float>& predictions,
    size_t begin_example_idx, size_t end_example_idx,
    std::vector<float>* gradient_data, std::vector<float>* hessian_data) {
  // Set the gradient to:
  //   label - 1/(1 + exp(-prediction))
  // where "label" is in {0,1} and prediction is the probability of
  // label=1.
  for (size_t example_idx = begin_example_idx; example_idx < end_example_idx;
       example_idx++) {
    const float label = (labels[example_idx] == 2) ? 1.f : 0.f;
    const float prediction = predictions[example_idx];
    const float prediction_proba = 1.f / (1.f + std::exp(-prediction));
    DCheckIsFinite(prediction);
    DCheckIsFinite(prediction_proba);
    (*gradient_data)[example_idx] = label - prediction_proba;
    if (hessian_data) {
      (*hessian_data)[example_idx] = prediction_proba * (1 - prediction_proba);
    }
  }
}

template <typename T>
absl::Status BinomialLogLikelihoodLoss::TemplatedUpdateGradients(
    const std::vector<T>& labels, const std::vector<float>& predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  static_assert(std::is_integral<T>::value, "Integral required.");

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
    TemplatedUpdateGradientsImp(labels, predictions, 0, num_examples,
                                &gradient_data, hessian_data);
  } else {
    decision_tree::ConcurrentForLoop(
        thread_pool->num_threads(), thread_pool, num_examples,
        [&labels, &predictions, &gradient_data, hessian_data](
            size_t block_idx, size_t begin_idx, size_t end_idx) -> void {
          TemplatedUpdateGradientsImp(labels, predictions, begin_idx, end_idx,
                                      &gradient_data, hessian_data);
        });
  }

  return absl::OkStatus();
}

absl::Status BinomialLogLikelihoodLoss::UpdateGradients(
    const std::vector<int32_t>& labels, const std::vector<float>& predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedUpdateGradients(labels, predictions, ranking_index, gradients,
                                  random, thread_pool);
}

absl::Status BinomialLogLikelihoodLoss::UpdateGradients(
    const std::vector<int16_t>& labels, const std::vector<float>& predictions,
    const RankingGroupsIndices* ranking_index, GradientDataRef* gradients,
    utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedUpdateGradients(labels, predictions, ranking_index, gradients,
                                  random, thread_pool);
}

decision_tree::CreateSetLeafValueFunctor
BinomialLogLikelihoodLoss::SetLeafFunctor(
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

void BinomialLogLikelihoodLoss::SetLeaf(
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

  // Set the value of the leaf to:
  //   (\sum_i weight[i] * (label[i] - p[i]) ) / (\sum_i weight[i] * p[i] *
  //   (1-p[i]))
  // with: p[i] = 1/(1+exp(-prediction)
  const auto* labels =
      train_dataset.ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
          label_col_idx);
  double numerator = 0;
  double denominator = 0;
  double sum_weights = 0;
  static const float bool_to_float[] = {0.f, 1.f};
  for (const auto example_idx : selected_examples) {
    const float weight = weights[example_idx];
    const float label = bool_to_float[labels->values()[example_idx] == 2];
    const float prediction = predictions[example_idx];
    const float p = 1.f / (1.f + std::exp(-prediction));
    numerator += weight * (label - p);
    denominator += weight * p * (1.f - p);
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

absl::Status BinomialLogLikelihoodLoss::UpdatePredictions(
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

std::vector<std::string> BinomialLogLikelihoodLoss::SecondaryMetricNames()
    const {
  return {"accuracy"};
}

template <bool use_weights, typename T>
void BinomialLogLikelihoodLoss::TemplatedLossImp(
    const std::vector<T>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights, size_t begin_example_idx,
    size_t end_example_idx, double* __restrict sum_loss,
    double* __restrict count_correct_predictions,
    double* __restrict sum_weights) {
  for (size_t example_idx = begin_example_idx; example_idx < end_example_idx;
       example_idx++) {
    const bool pos_label = labels[example_idx] == 2;
    const float label = pos_label ? 1.f : 0.f;
    const float prediction = predictions[example_idx];
    const bool pos_prediction = prediction >= 0;
    if constexpr (use_weights) {
      const float weight = weights[example_idx];
      *sum_weights += weight;
      if (pos_label == pos_prediction) {
        *count_correct_predictions += weight;
      }
      *sum_loss -= 2 * weight *
                   (label * prediction - std::log(1 + std::exp(prediction)));
    } else {
      if (pos_label == pos_prediction) {
        *count_correct_predictions += 1.;
      }
      // Loss:
      //   -2 * ( label * prediction - log(1+exp(prediction)))
      *sum_loss -=
          2 * (label * prediction - std::log(1 + std::exp(prediction)));
      DCheckIsFinite(*sum_loss);
    }
    DCheckIsFinite(*sum_loss);
  }
  if constexpr (!use_weights) {
    *sum_weights += end_example_idx - begin_example_idx;
  }
}

template <typename T>
absl::Status BinomialLogLikelihoodLoss::TemplatedLoss(
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
                              &sum_loss, &count_correct_predictions,
                              &sum_weights);
    } else {
      TemplatedLossImp<true>(labels, predictions, weights, 0, labels.size(),
                             &sum_loss, &count_correct_predictions,
                             &sum_weights);
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
        [&labels, &predictions, &per_threads, &weights](
            size_t block_idx, size_t begin_idx, size_t end_idx) -> void {
          auto& block = per_threads[block_idx];

          if (weights.empty()) {
            TemplatedLossImp<false>(labels, predictions, weights, begin_idx,
                                    end_idx, &block.sum_loss,
                                    &block.count_correct_predictions,
                                    &block.sum_weights);
          } else {
            TemplatedLossImp<true>(labels, predictions, weights, begin_idx,
                                   end_idx, &block.sum_loss,
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

absl::Status BinomialLogLikelihoodLoss::Loss(
    const std::vector<int32_t>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index, float* loss_value,
    std::vector<float>* secondary_metric,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedLoss(labels, predictions, weights, ranking_index, loss_value,
                       secondary_metric, thread_pool);
}

absl::Status BinomialLogLikelihoodLoss::Loss(
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
