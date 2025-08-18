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
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_set.h"
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
  const double w = 1.f / log_hazard_predictions.size();
  const double log_w = std::log(w);

  double loss = 0.f;
  double hazard = 0.f;
  absl::flat_hash_set<row_t> at_risk;
  for (const auto& [time, update_type, example_idx] : cox_cache->updates) {
    switch (update_type) {
      case CoxProportionalHazardLoss::Update::Type::ARRIVAL:
        at_risk.insert(example_idx);
        break;
      case CoxProportionalHazardLoss::Update::Type::EVENT:
        hazard = 0.f;
        for (auto at_risk_idx : at_risk) {
          hazard += w * std::exp(log_hazard_predictions[at_risk_idx]);
        }
        loss += w * (std::log(hazard) - log_hazard_predictions[example_idx] -
                     log_w);
        [[fallthrough]];
      case CoxProportionalHazardLoss::Update::Type::CENSORING:
        at_risk.erase(example_idx);
    }
  }
  DCHECK(at_risk.empty());

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
  std::fill(hessians.begin(), hessians.end(), 0.f);
  std::fill(gradients.begin(), gradients.end(), 0.f);

  double hazard = 0.f;
  absl::flat_hash_set<row_t> at_risk;
  for (const auto& [time, update_type, example_idx] : cox_cache->updates) {
    switch (update_type) {
      case CoxProportionalHazardLoss::Update::Type::ARRIVAL:
        at_risk.insert(example_idx);
        break;
      case CoxProportionalHazardLoss::Update::Type::EVENT:
        hazard = 0.f;
        for (auto at_risk_idx : at_risk) {
          hazard += std::exp(log_hazard_predictions[at_risk_idx]);
        }
        for (auto at_risk_idx : at_risk) {
          float p = std::exp(log_hazard_predictions[at_risk_idx]) / hazard;
          gradients[at_risk_idx] -= w * p;
          hessians[at_risk_idx] += w * p * (1.f - p);
        }
        gradients[example_idx] += w;
        [[fallthrough]];
      case CoxProportionalHazardLoss::Update::Type::CENSORING:
        at_risk.erase(example_idx);
    }
  }
  DCHECK(at_risk.empty());
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::gradient_boosted_trees
