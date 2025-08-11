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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_COX_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_COX_H_

#include <stddef.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests::model::gradient_boosted_trees {

// Cox Proportional Hazard loss.
//
// Suited for survival analysis. It implements the Cox proportional hazard loss
// described in:
//
//   "The state of boosting. Computing Science and Statistics."
//   G. Ridgeway.
//   https://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/a29acbd26d902d6fc125822a0031c09b/$FILE/boosting.pdf
//
// With implementation details from:
//
//  "Generalized Boosted Models: A guide to the gbm package."
//  G. Ridgeway
//  https://cran.r-project.org/web/packages/gbm/vignettes/gbm.pdf
//
// See "AbstractLoss" for the method documentation.
class CoxProportionalHazardLoss : public AbstractLoss {
 public:
  // For unit testing.
  using AbstractLoss::Loss;
  using AbstractLoss::UpdateGradients;
  using row_t = dataset::VerticalDataset::row_t;

  CoxProportionalHazardLoss(const ConstructorArgs& args) : AbstractLoss(args) {}

  class Cache : public AbstractLossCache {
   public:
    Cache() {}
    virtual ~Cache() = default;

    // These pair of structures are used to store the risk sets. A risk set R_i
    // contains the idxs of the examples that were at risk at the departure age
    // of the i-th example. Risk sets are only created for examples with
    // departure events (non-censored).
    //
    // We store all sets in a flat vector `risk_set_idxs` to make sure all data
    // is contiguous in memory. And since the sets do not have constant sizes,
    // we use an additional vector of pairs `risk_set_sizes` to store the sizes
    // of the sets, and the idx they belong to. Specifically, if
    // `risk_set_sizes[j] = (i, n)`, then the j-th set in `risk_set_idxs`
    // corresponds to R_i, and has n elements.
    //
    // TODO: Today the `risk_set_sizes` are sorted in increasing order
    // of the departure age of the example. But the idxs could be sorted by
    // value to process the dataset sequentially and have a better cache
    // locality.
    std::vector<std::pair<row_t, row_t>> risk_set_sizes;
    std::vector<row_t> risk_set_bank;
  };

  absl::StatusOr<std::unique_ptr<AbstractLossCache>> CreateLossCache(
      const dataset::VerticalDataset& dataset) const override;

  static absl::StatusOr<std::unique_ptr<AbstractLoss>> RegistrationCreate(
      const ConstructorArgs& args);

  LossShape Shape() const override {
    return LossShape{.gradient_dim = 1, .prediction_dim = 1};
  };

  inline absl::StatusOr<std::vector<float>> InitialPredictions(
      const dataset::VerticalDataset& dataset, int label_col_idx,
      absl::Span<const float> weights) const override {
    return std::vector<float>{0.f};
  }

  inline absl::StatusOr<std::vector<float>> InitialPredictions(
      const decision_tree::proto::LabelStatistics& label_statistics)
      const override {
    return std::vector<float>{0.f};
  }

  inline std::vector<std::string> SecondaryMetricNames() const override {
    return {};
  }

  inline absl::Status UpdateGradients(
      absl::Span<const float> labels, absl::Span<const float> predictions,
      const AbstractLossCache* cache, GradientDataRef* gradients,
      utils::RandomEngine* random,
      utils::concurrency::ThreadPool* thread_pool) const override {
    return UpdateGradients(predictions, cache, gradients);
  }

  inline absl::StatusOr<LossResults> Loss(
      absl::Span<const float> labels, absl::Span<const float> predictions,
      absl::Span<const float> weights, const AbstractLossCache* cache,
      utils::concurrency::ThreadPool* thread_pool) const override {
    return Loss(predictions, cache);
  }

 private:
  absl::Status UpdateGradients(absl::Span<const float> log_hazard_predictions,
                               const AbstractLossCache* cache,
                               GradientDataRef* gradient_data) const;

  absl::StatusOr<LossResults> Loss(
      absl::Span<const float> log_hazard_predictions,
      const AbstractLossCache* cache) const;
};

REGISTER_AbstractGradientBoostedTreeLoss(CoxProportionalHazardLoss,
                                         "COX_PROPORTIONAL_HAZARD");

}  // namespace yggdrasil_decision_forests::model::gradient_boosted_trees

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_LOSS_LOSS_IMP_COX_H_
