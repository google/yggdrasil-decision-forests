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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_cox.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::model::gradient_boosted_trees {

using BooleanColumn = dataset::VerticalDataset::BooleanColumn;
using NumericalColumn = dataset::VerticalDataset::NumericalColumn;

absl::StatusOr<std::unique_ptr<AbstractLoss>>
CoxProportionalHazardLoss::RegistrationCreate(const ConstructorArgs& args) {
  if (args.task != model::proto::Task::SURVIVAL_ANALYSIS) {
    return absl::InvalidArgumentError(
        "Cox proportional hazard loss is only compatible with survival a "
        "analysis task.");
  }
  return absl::make_unique<CoxProportionalHazardLoss>(args);
}

absl::StatusOr<std::unique_ptr<AbstractLossCache>>
CoxProportionalHazardLoss::CreateLossCache(
    const dataset::VerticalDataset& dataset) const {
  auto event_index = train_config_link_.label_event_observed();
  if (event_index == -1) {
    return absl::InvalidArgumentError(
        "label_event_observed must be set for Cox proportional hazard loss.");
  }
  ASSIGN_OR_RETURN(const std::vector<int8_t>& events,
                   dataset.ColumnWithCastWithStatus<BooleanColumn>(event_index))
      ->values();

  auto departure_age_index = train_config_link_.label();
  if (departure_age_index == -1) {
    return absl::InvalidArgumentError(
        "label must be set for Cox proportional hazard loss.");
  }
  ASSIGN_OR_RETURN(
      const std::vector<float>& departure_ages,
      dataset.ColumnWithCastWithStatus<NumericalColumn>(departure_age_index))
      ->values();

  const auto entry_age_index = train_config_link_.label_entry_age();
  std::optional<std::vector<float>> entry_ages;
  if (train_config_link_.label_entry_age() != -1) {
    ASSIGN_OR_RETURN(
        entry_ages,
        dataset.ColumnWithCastWithStatus<NumericalColumn>(entry_age_index))
        ->values();
  }

  auto cache = absl::make_unique<CoxProportionalHazardLoss::Cache>();
  LOG(INFO) << "Precomputing Cox Proportional Hazard Cache";

  cache->updates.reserve(dataset.nrow() * 2);
  for (row_t idx = 0; idx < dataset.nrow(); ++idx) {
    // Populate Arrival times.
    cache->updates.push_back(
        {entry_ages.has_value() ? entry_ages.value()[idx] : 0.f,
         CoxProportionalHazardLoss::Update::Type::ARRIVAL, idx});
    // Populate Event or Censoring times.
    cache->updates.push_back(
        {departure_ages[idx],
         events[idx] ? CoxProportionalHazardLoss::Update::Type::EVENT
                     : CoxProportionalHazardLoss::Update::Type::CENSORING,
         idx});
  }
  std::sort(cache->updates.begin(), cache->updates.end());

  LOG(INFO) << "Done precomputing Cox Proportional Hazard Cache";
  return cache;
}

absl::StatusOr<LossResults> CoxProportionalHazardLoss::Loss(
    const absl::Span<const float> log_hazard_predictions,
    const AbstractLossCache* cache) const {
  if (cache == nullptr) {
    return absl::InvalidArgumentError("Cache is null.");
  }
  const auto* cox_cache = dynamic_cast<const Cache*>(cache);
  // TODO: Add support for non-uniform weights.
  const double w = 1.0 / log_hazard_predictions.size();

  std::vector<double> exp_preds(log_hazard_predictions.size());
  for (size_t i = 0; i < log_hazard_predictions.size(); ++i) {
    exp_preds[i] = std::exp(log_hazard_predictions[i]);
  }

  double loss = 0.;
  double hazard = 0.;
  // Updates are sorted by time and every item first arrives before
  // event / censor.
  for (const auto& [time, update_type, example_idx] : cox_cache->updates) {
    switch (update_type) {
      case CoxProportionalHazardLoss::Update::Type::ARRIVAL:
        hazard += exp_preds[example_idx];
        break;
      case CoxProportionalHazardLoss::Update::Type::EVENT:
        if (hazard > 0.0) {
          loss += w * (std::log(hazard) - log_hazard_predictions[example_idx]);
        }
        [[fallthrough]];
      case CoxProportionalHazardLoss::Update::Type::CENSORING:
        hazard -= exp_preds[example_idx];
        if (hazard < 0.0) {
          LOG_EVERY_POW_2(INFO) << "Cox loss has encountered negative hazard "
                                << hazard << " setting hazard to 0.";
          hazard = 0.0;
        }
        break;
    }
  }

  return LossResults{static_cast<float>(loss), /*.secondary_metrics =*/{}};
}

absl::Status CoxProportionalHazardLoss::UpdateGradients(
    const absl::Span<const float> log_hazard_predictions,
    const AbstractLossCache* cache, GradientDataRef* gradient_data) const {
  if (cache == nullptr) {
    return absl::InvalidArgumentError("Cache is null.");
  }
  const auto* cox_cache = dynamic_cast<const Cache*>(cache);
  // TODO: Add support for non-uniform weights.
  float w = 1.f / log_hazard_predictions.size();

  std::vector<float>& hessians = *(*gradient_data)[0].hessian;
  std::vector<float>& gradients = *(*gradient_data)[0].gradient;

  std::vector<double> exp_preds(log_hazard_predictions.size());
  for (size_t i = 0; i < log_hazard_predictions.size(); ++i) {
    exp_preds[i] = std::exp(log_hazard_predictions[i]);
  }

  double hazard = 0.;
  double sum_1_over_hazard = 0.;
  double sum_1_over_hazard_sq = 0.;
  std::vector<double> snapshot_S1(log_hazard_predictions.size(), 0.0);
  std::vector<double> snapshot_S2(log_hazard_predictions.size(), 0.0);

  // Updates are sorted by time and every item first arrives before
  // event / censor.
  for (const auto& [time, update_type, example_idx] : cox_cache->updates) {
    switch (update_type) {
      case CoxProportionalHazardLoss::Update::Type::ARRIVAL:
        snapshot_S1[example_idx] = sum_1_over_hazard;
        snapshot_S2[example_idx] = sum_1_over_hazard_sq;
        hazard += exp_preds[example_idx];
        break;
      case CoxProportionalHazardLoss::Update::Type::EVENT: {
        if (hazard > 0.0) {
          sum_1_over_hazard += 1.0 / hazard;
          sum_1_over_hazard_sq += 1.0 / (hazard * hazard);
        }
        double dS1 = sum_1_over_hazard - snapshot_S1[example_idx];
        double dS2 = sum_1_over_hazard_sq - snapshot_S2[example_idx];
        double exp_pred = exp_preds[example_idx];
        gradients[example_idx] = w * (1.0 - exp_pred * dS1);
        hessians[example_idx] =
            w * (exp_pred * dS1 - exp_pred * exp_pred * dS2);
        hazard -= exp_preds[example_idx];
        if (hazard < 0.0) {
          LOG_EVERY_POW_2(INFO) << "Cox loss has encountered negative hazard "
                                << hazard << " setting hazard to 0.";
          hazard = 0.0;
        }
        break;
      }
      case CoxProportionalHazardLoss::Update::Type::CENSORING: {
        double dS1 = sum_1_over_hazard - snapshot_S1[example_idx];
        double dS2 = sum_1_over_hazard_sq - snapshot_S2[example_idx];
        double exp_pred = exp_preds[example_idx];
        gradients[example_idx] = w * (-exp_pred * dS1);
        hessians[example_idx] =
            w * (exp_pred * dS1 - exp_pred * exp_pred * dS2);
        hazard -= exp_preds[example_idx];
        if (hazard < 0.0) {
          LOG_EVERY_POW_2(INFO) << "Cox loss has encountered negative hazard "
                                << hazard << " setting hazard to 0.";
          hazard = 0.0;
        }
        break;
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::gradient_boosted_trees
