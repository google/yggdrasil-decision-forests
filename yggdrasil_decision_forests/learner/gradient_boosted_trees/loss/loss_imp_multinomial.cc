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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_utils.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

absl::StatusOr<std::unique_ptr<AbstractLoss>>
MultinomialLogLikelihoodLoss::RegistrationCreate(const ConstructorArgs& args) {
  if (args.task != model::proto::Task::CLASSIFICATION) {
    return absl::InvalidArgumentError(
        "Multinomial log-likelihood loss is only compatible with a "
        "classification task");
  }
  return std::make_unique<MultinomialLogLikelihoodLoss>(args);
}

absl::StatusOr<std::vector<float>>
MultinomialLogLikelihoodLoss::InitialPredictions(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const absl::Span<const float> weights) const {
  if (!initialize_with_class_priors_) {
    return std::vector<float>(dimension_, 0);
  }
  // Return: log(pr(i)) with pr(i) the probability of class i.
  std::vector<double> weighted_counts(dimension_, 0.0);
  double sum_weights = 0;

  ASSIGN_OR_RETURN(
      const auto* labels,
      dataset.ColumnWithCastWithStatus<
          dataset::VerticalDataset::CategoricalColumn>(label_col_idx));
  const UnsignedExampleIdx n = dataset.nrow();
  if (weights.empty()) {
    sum_weights = static_cast<double>(n);
    const auto& label_values = labels->values();
    for (UnsignedExampleIdx example_idx = 0; example_idx < n; ++example_idx) {
      int label_val = label_values[example_idx];
      if (label_val >= 1 && label_val <= dimension_) {
        weighted_counts[label_val - 1] += 1.0;
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "Label value at example_idx ", example_idx, " is invalid: ",
            label_val, ". Expected value between 1 and ", dimension_));
      }
    }
  } else {
    const auto& label_values = labels->values();
    for (UnsignedExampleIdx example_idx = 0; example_idx < n; ++example_idx) {
      float w = weights[example_idx];
      sum_weights += w;
      int label_val = label_values[example_idx];
      if (label_val >= 1 && label_val <= dimension_) {
        weighted_counts[label_val - 1] += w;
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "Label value at example_idx ", example_idx, " is invalid: ",
            label_val, ". Expected value between 1 and ", dimension_));
      }
    }
  }
  STATUS_CHECK_GT(sum_weights, 0);

  std::vector<float> initial_predictions(dimension_);
  const double inv_sum_weights = 1. / sum_weights;
  for (int dim = 0; dim < dimension_; ++dim) {
    const double prob = weighted_counts[dim] * inv_sum_weights;
    if (prob <= 0.) {
      initial_predictions[dim] = std::numeric_limits<float>::lowest();
    } else {
      initial_predictions[dim] = static_cast<float>(std::log(prob));
    }
  }
  return initial_predictions;
}

absl::StatusOr<std::vector<float>>
MultinomialLogLikelihoodLoss::InitialPredictions(
    const decision_tree::proto::LabelStatistics& label_statistics) const {
  if (!initialize_with_class_priors_) {
    return std::vector<float>(dimension_, 0);
  }
  // Return: log(pr(i)) with pr(i) the probability of class i.
  if (label_statistics.classification().labels().counts_size() !=
      dimension_ + 1) {
    return absl::InternalError(absl::Substitute(
        "The multinomial loglikelihood loss expects $0 classes i.e. $1 "
        "unique values (including the OOV item). Got $2 unique values "
        "instead.",
        dimension_, dimension_ + 1,
        label_statistics.classification().labels().counts_size()));
  }
  const auto& counts = label_statistics.classification().labels().counts();
  const double inv_sum = 1. / label_statistics.classification().labels().sum();
  std::vector<float> initial_predictions(dimension_);
  for (size_t dim = 0; dim < dimension_; ++dim) {
    double prob = counts[dim + 1] * inv_sum;
    if (prob <= 0.) {
      initial_predictions[dim] = std::numeric_limits<float>::lowest();
    } else {
      initial_predictions[dim] = static_cast<float>(std::log(prob));
    }
  }
  return initial_predictions;
}

template <typename T>
absl::Status MultinomialLogLikelihoodLoss::TemplatedUpdateGradients(
    const absl::Span<T> labels, const absl::Span<const float> predictions,
    const AbstractLossCache* cache, GradientDataRef* gradients,
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
    const absl::Span<const int32_t> labels,
    const absl::Span<const float> predictions, const AbstractLossCache* cache,
    GradientDataRef* gradients, utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedUpdateGradients(labels, predictions, cache, gradients, random,
                                  thread_pool);
}

absl::Status MultinomialLogLikelihoodLoss::UpdateGradients(
    const absl::Span<const int16_t> labels,
    const absl::Span<const float> predictions, const AbstractLossCache* cache,
    GradientDataRef* gradients, utils::RandomEngine* random,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedUpdateGradients(labels, predictions, cache, gradients, random,
                                  thread_pool);
}

std::vector<std::string> MultinomialLogLikelihoodLoss::SecondaryMetricNames()
    const {
  return {"accuracy"};
}

template <bool weighted, typename T>
void MultinomialLogLikelihoodLoss::TemplatedLossImp(
    const absl::Span<T> labels, const absl::Span<const float> predictions,
    const absl::Span<const float> weights, size_t begin_example_idx,
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
    const absl::Span<T> labels, const absl::Span<const float> predictions,
    const absl::Span<const float> weights, const AbstractLossCache* cache,
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
      // Maybe the number of labels is so small that not all threads were used.
      if (block.confusion_matrix.ncol() > 0) {
        confusion_matrix.Add(block.confusion_matrix);
      }
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
    const absl::Span<const int32_t> labels,
    const absl::Span<const float> predictions,
    const absl::Span<const float> weights, const AbstractLossCache* cache,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedLoss(labels, predictions, weights, cache, thread_pool);
}

absl::StatusOr<LossResults> MultinomialLogLikelihoodLoss::Loss(
    const absl::Span<const int16_t> labels,
    const absl::Span<const float> predictions,
    const absl::Span<const float> weights, const AbstractLossCache* cache,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedLoss(labels, predictions, weights, cache, thread_pool);
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
