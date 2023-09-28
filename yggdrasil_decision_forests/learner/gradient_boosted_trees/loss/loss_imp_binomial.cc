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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_binomial.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

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

absl::StatusOr<std::vector<float>>
BinomialLogLikelihoodLoss::InitialPredictions(
    const dataset::VerticalDataset& dataset, int label_col_idx,
    const std::vector<float>& weights) const {
  // Return: log(y/(1-y)) with y the ratio of positive labels.
  double weighted_sum_positive = 0;
  double sum_weights = 0;
  ASSIGN_OR_RETURN(
      const auto* labels,
      dataset.ColumnWithCastWithStatus<
          dataset::VerticalDataset::CategoricalColumn>(label_col_idx));
  const UnsignedExampleIdx n = dataset.nrow();
  if (weights.empty()) {
    sum_weights = static_cast<double>(n);
    weighted_sum_positive = static_cast<double>(
        std::count(labels->values().begin(), labels->values().end(), 2));
  } else {
    for (UnsignedExampleIdx example_idx = 0; example_idx < n; example_idx++) {
      sum_weights += weights[example_idx];
      weighted_sum_positive +=
          weights[example_idx] * (labels->values()[example_idx] == 2);
    }
  }
  STATUS_CHECK_GT(sum_weights, 0);

  const double ratio_positive = weighted_sum_positive / sum_weights;
  if (ratio_positive == 0.0) {
    return std::vector<float>{-std::numeric_limits<float>::max()};
  } else if (ratio_positive == 1.0) {
    return std::vector<float>{std::numeric_limits<float>::max()};
  } else {
    return std::vector<float>{
        static_cast<float>(std::log(ratio_positive / (1. - ratio_positive)))};
  }
}

absl::StatusOr<std::vector<float>>
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
  DCHECK_EQ(gradient_data->size(), hessian_data->size());
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
    (*hessian_data)[example_idx] = prediction_proba * (1 - prediction_proba);
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
  if (hessian_data == nullptr) {
    return absl::InternalError("Hessian missing");
  }
  const size_t num_examples = labels.size();

  if (thread_pool == nullptr) {
    TemplatedUpdateGradientsImp(labels, predictions, 0, num_examples,
                                &gradient_data, hessian_data);
  } else {
    utils::concurrency::ConcurrentForLoop(
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


std::vector<std::string> BinomialLogLikelihoodLoss::SecondaryMetricNames()
    const {
  return {"accuracy"};
}

template <bool use_weights, typename T>
void BinomialLogLikelihoodLoss::TemplatedLossImp(
    const std::vector<T>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights, size_t begin_example_idx,
    size_t end_example_idx, double* __restrict sum_loss,
    utils::IntegersConfusionMatrixDouble* confusion_matrix) {
  double local_sum_loss = 0;
  for (size_t example_idx = begin_example_idx; example_idx < end_example_idx;
       example_idx++) {
    // The loss function expects a 0/1 label.
    const bool pos_label = labels[example_idx] == 2;
    const float label_for_loss = pos_label ? 1.f : 0.f;
    const float prediction = predictions[example_idx];
    const int predicted_label = prediction > 0.f ? 2 : 1;
    if constexpr (use_weights) {
      const float weight = weights[example_idx];
      confusion_matrix->Add(labels[example_idx], predicted_label, weight);
      local_sum_loss -=
          2 * weight *
          (label_for_loss * prediction - std::log(1.f + std::exp(prediction)));
    } else {
      confusion_matrix->Add(labels[example_idx], predicted_label, 1.f);
      // Loss:
      //   -2 * ( label * prediction - log(1+exp(prediction)))
      local_sum_loss -= 2 * (label_for_loss * prediction -
                             std::log(1.f + std::exp(prediction)));
    }
    DCheckIsFinite(local_sum_loss);
  }
  *sum_loss += local_sum_loss;
}

template <typename T>
absl::StatusOr<LossResults> BinomialLogLikelihoodLoss::TemplatedLoss(
    const std::vector<T>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index,
    utils::concurrency::ThreadPool* thread_pool) const {
  double sum_loss = 0;
  utils::IntegersConfusionMatrixDouble confusion_matrix;
  int confusion_matrix_size =
      label_column_.categorical().number_of_unique_values();
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
    double total_example_weight = confusion_matrix.sum();
    float loss = sum_loss / total_example_weight;
    double correct_predictions = confusion_matrix.Trace();
    DCheckIsFinite(loss);
    return LossResults{
        /*.loss =*/loss,
        /*.secondary_metrics =*/
        {static_cast<float>(correct_predictions / total_example_weight)},
        /*.confusion_table =*/std::move(confusion_matrix)};
  } else {
    return LossResults{
        /*.loss =*/std::numeric_limits<float>::quiet_NaN(),
        /*.secondary_metrics =*/{std::numeric_limits<float>::quiet_NaN()}};
  }
}

absl::StatusOr<LossResults> BinomialLogLikelihoodLoss::Loss(
    const std::vector<int32_t>& labels, const std::vector<float>& predictions,
    const std::vector<float>& weights,
    const RankingGroupsIndices* ranking_index,
    utils::concurrency::ThreadPool* thread_pool) const {
  return TemplatedLoss(labels, predictions, weights, ranking_index,
                       thread_pool);
}

absl::StatusOr<LossResults> BinomialLogLikelihoodLoss::Loss(
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
