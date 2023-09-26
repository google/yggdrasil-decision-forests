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

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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

absl::StatusOr<std::vector<float>>
MultinomialLogLikelihoodLoss::InitialPredictions(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const std::vector<float>& weights) const {
  // YDF follows Friedman's paper "Greedy Function Approximation: A Gradient
  // Boosting Machine" (https://statweb.stanford.edu/~jhf/ftp/trebst.pdf),
  // setting the initial prediction to 0 for multi-class classification
  // (Algorithm 6).
  // TODO: Experiment with different initial predictions.
  return std::vector<float>(dimension_, 0);
}

absl::StatusOr<std::vector<float>>
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
  // TODO: Implement thread_pool.

  // Set the gradient to:
  //   label_i - pred_i
  // where "label_i" is in {0,1}.
  absl::FixedArray<float> accumulator(gradients->size());
  const auto num_examples = labels.size();
  const auto use_hessian_gain = (*gradients)[0].hessian;
  if (!use_hessian_gain) {
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

      auto& gradient_data = (*(*gradients)[grad_idx].gradient);
      auto& hessian_data = (*(*gradients)[grad_idx].hessian);
      DCHECK_EQ(gradient_data.size(), hessian_data.size());
      gradient_data[example_idx] = grad;
      hessian_data[example_idx] = abs_grad * (1 - abs_grad);
      DCheckIsFinite(abs_grad * (1 - abs_grad));
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


std::vector<std::string> MultinomialLogLikelihoodLoss::SecondaryMetricNames()
    const {
  return {"accuracy"};
}

template <bool weighted, typename T>
void MultinomialLogLikelihoodLoss::TemplatedLossImp(
    const std::vector<T>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights, size_t begin_example_idx,
    size_t end_example_idx, double* __restrict sum_loss,
    utils::IntegersConfusionMatrixDouble* confusion_matrix) {
  const int dimension = confusion_matrix->ncol() - 1;
  double loss = 0;

  for (size_t example_idx = begin_example_idx; example_idx < end_example_idx;
       example_idx++) {
    const int label = labels[example_idx];
    int predicted_class = -1;
    float predicted_class_exp_value = 0;
    float sum_exp = 0;
    if constexpr (weighted) {
      const float weight = weights[example_idx];
      for (int grad_idx = 0; grad_idx < dimension; grad_idx++) {
        const float exp_val =
            std::exp(predictions[grad_idx + example_idx * dimension]);
        sum_exp += exp_val;
        DCheckIsFinite(sum_exp);
        if (exp_val > predicted_class_exp_value) {
          predicted_class_exp_value = exp_val;
          predicted_class = grad_idx + 1;
        }
      }
      confusion_matrix->Add(label, predicted_class, weight);
      // Loss:
      //   - log(predict_proba[true_label])
      const float tree_label_exp_value =
          std::exp(predictions[(label - 1) + example_idx * dimension]);
      loss -= weight * std::log(tree_label_exp_value / sum_exp);
    } else {
      for (int grad_idx = 0; grad_idx < dimension; grad_idx++) {
        const float exp_val =
            std::exp(predictions[grad_idx + example_idx * dimension]);
        sum_exp += exp_val;
        DCheckIsFinite(sum_exp);
        if (exp_val > predicted_class_exp_value) {
          predicted_class_exp_value = exp_val;
          predicted_class = grad_idx + 1;
        }
      }
      confusion_matrix->Add(label, predicted_class, 1);
      // Loss:
      //   - log(predict_proba[true_label])
      const float tree_label_exp_value =
          std::exp(predictions[(label - 1) + example_idx * dimension]);
      loss -= std::log(tree_label_exp_value / sum_exp);
    }
    DCheckIsFinite(loss);
    DCheckIsFinite(confusion_matrix->sum());
  }
  DCheckIsFinite(loss);
  *sum_loss = loss;
}

template <typename T>
absl::StatusOr<LossResults> MultinomialLogLikelihoodLoss::TemplatedLoss(
    const std::vector<T>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index,
    utils::concurrency::ThreadPool* thread_pool) const {
  double sum_loss = 0;
  utils::IntegersConfusionMatrixDouble confusion_matrix;
  int confusion_matrix_size = dimension_ + 1;
  confusion_matrix.SetSize(confusion_matrix_size, confusion_matrix_size);

  if (thread_pool == nullptr) {
    if (weights.empty()) {
      TemplatedLossImp<false>(labels, predictions, weights, 0, labels.size(),
                              &sum_loss, &confusion_matrix);
    } else {
      TemplatedLossImp<true>(labels, predictions, weights, 0, labels.size(),
                             &sum_loss, &confusion_matrix);
    }
  } else {
    const auto num_threads = thread_pool->num_threads();

    struct PerThread {
      double sum_loss = 0;
      utils::IntegersConfusionMatrixDouble confusion_matrix;
    };
    std::vector<PerThread> per_threads(num_threads);

    utils::concurrency::ConcurrentForLoop(
        num_threads, thread_pool, labels.size(),
        [&labels, &predictions, &per_threads, &weights, &confusion_matrix_size](
            size_t block_idx, size_t begin_idx, size_t end_idx) -> void {
          auto& block = per_threads[block_idx];
          block.confusion_matrix.SetSize(confusion_matrix_size,
                                         confusion_matrix_size);

          if (weights.empty()) {
            TemplatedLossImp<false>(labels, predictions, weights, begin_idx,
                                    end_idx, &block.sum_loss,
                                    &block.confusion_matrix);
          } else {
            TemplatedLossImp<true>(labels, predictions, weights, begin_idx,
                                   end_idx, &block.sum_loss,
                                   &block.confusion_matrix);
          }
        });

    for (const auto& block : per_threads) {
      sum_loss += block.sum_loss;
      confusion_matrix.Add(block.confusion_matrix);
    }
  }

  if (confusion_matrix.sum() > 0) {
    const float loss = sum_loss / confusion_matrix.sum();
    const float secondary_metric =
        static_cast<float>(confusion_matrix.Trace() / confusion_matrix.sum());
    DCheckIsFinite(loss);
    return LossResults{/*.loss =*/loss,
                       /*.secondary_metrics =*/{secondary_metric},
                       /*.confusion_table =*/std::move(confusion_matrix)};
  } else {
    return LossResults{
        /*.loss =*/std::numeric_limits<float>::quiet_NaN(),
        /*.secondary_metrics =*/{std::numeric_limits<float>::quiet_NaN()}};
  }
}

absl::StatusOr<LossResults> MultinomialLogLikelihoodLoss::Loss(
    const std::vector<int32_t>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedLoss(labels, predictions, weights, ranking_index,
                       thread_pool);
}

absl::StatusOr<LossResults> MultinomialLogLikelihoodLoss::Loss(
    const std::vector<int16_t>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedLoss(labels, predictions, weights, ranking_index,
                       thread_pool);
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
