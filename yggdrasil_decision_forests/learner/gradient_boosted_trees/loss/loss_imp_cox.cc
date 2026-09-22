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
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/accurate_sum.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::model::gradient_boosted_trees {

using BooleanColumn = dataset::VerticalDataset::BooleanColumn;
using NeumaierSum = utils::NeumaierSum;
using NumericalColumn = dataset::VerticalDataset::NumericalColumn;

namespace {

// Smallest shifted log-hazard that is exponentiated.
//
// The value is chosen so that no intermediate quantity can overflow: with
// "h_min = exp(-250) ~= 2.7e-109", the sum of the hazards at risk "S" is in
// [h_min, n], so "1/S <= 3.8e108" and "1/S^2 <= 1.4e217" are finite, and so are
// their sums over the events (bounded by "n/h_min^2 <= 1.4e226" for a dataset
// of at most 1e9 examples). A floor of, say, -700 would keep the hazards
// normal but would let "1/S^2" overflow to infinity and poison the gradients of
// every subsequent example.
//
// Clamping is harmless in practice: an example whose log-hazard is 250 nats
// below the largest one contributes ~1e-109 of the risk set, i.e. nothing.
constexpr double kMinShiftedLogHazard = -250.;

// Computes exp(prediction - max(predictions)) for each example.
//
// The Cox partial likelihood is invariant to the addition of a constant to all
// the log-hazards: log(sum_j exp(p_j - c)) - (p_i - c) does not depend on "c".
// Shifting by the largest prediction keeps all the exponentials in (0, 1] and
// makes overflow impossible, whatever the magnitude of the predictions.
//
// Note: The vector returned by this function only contains strictly positive
// values.
absl::StatusOr<std::vector<double>> ShiftedHazards(
    const absl::Span<const float> log_hazard_predictions) {
  double max_log_hazard = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < log_hazard_predictions.size(); ++i) {
    const double log_hazard = log_hazard_predictions[i];
    if (!std::isfinite(log_hazard)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The Cox proportional hazard loss received the non-finite "
          "log-hazard prediction ",
          log_hazard, " for example ", i,
          ". This generally indicates that the training diverged; consider "
          "reducing the shrinkage or increasing the regularization."));
    }
    max_log_hazard = std::max(max_log_hazard, log_hazard);
  }

  std::vector<double> hazards(log_hazard_predictions.size());
  for (size_t i = 0; i < log_hazard_predictions.size(); ++i) {
    hazards[i] = std::exp(std::max(
        static_cast<double>(log_hazard_predictions[i]) - max_log_hazard,
        kMinShiftedLogHazard));
  }
  return hazards;
}

}  // namespace

absl::StatusOr<std::unique_ptr<AbstractLoss>>
CoxProportionalHazardLoss::RegistrationCreate(const ConstructorArgs& args) {
  if (args.task != model::proto::Task::SURVIVAL_ANALYSIS) {
    return absl::InvalidArgumentError(
        "Cox proportional hazard loss is only compatible with survival a "
        "analysis task.");
  }
  return std::make_unique<CoxProportionalHazardLoss>(args);
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

  auto cache = std::make_unique<CoxProportionalHazardLoss::Cache>();
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

  ASSIGN_OR_RETURN(const std::vector<double> hazards,
                   ShiftedHazards(log_hazard_predictions));

  double loss = 0.;
  NeumaierSum risk_set;
  // Updates are sorted by time and every item first arrives before
  // event / censor.
  for (const auto& [time, update_type, example_idx] : cox_cache->updates) {
    const double hazard = hazards[example_idx];
    switch (update_type) {
      case CoxProportionalHazardLoss::Update::Type::ARRIVAL:
        risk_set.Add(hazard);
        break;
      case CoxProportionalHazardLoss::Update::Type::EVENT: {
        // The example is part of its own risk set, so the sum of the hazards
        // at risk is at least "hazard", i.e. the max() returns
        // risk_set.Value(). Using the max just enforces it in case the
        // arithmetic drifted.
        const double sum_at_risk = std::max(risk_set.Value(), hazard);
        loss += w * std::log(sum_at_risk / hazard);
      }
        [[fallthrough]];
      case CoxProportionalHazardLoss::Update::Type::CENSORING:
        risk_set.Add(-hazard);
        if (risk_set.Value() < 0.0) {
          LOG_EVERY_POW_2(WARNING)
              << "Cox loss has encountered a negative sum of hazards "
              << risk_set.Value() << " at risk. Setting it to 0.";
          risk_set.Reset();
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
  const double w = 1.f / log_hazard_predictions.size();

  std::vector<float>& hessians = *(*gradient_data)[0].hessian;
  std::vector<float>& gradients = *(*gradient_data)[0].gradient;

  ASSIGN_OR_RETURN(const std::vector<double> hazards,
                   ShiftedHazards(log_hazard_predictions));

  NeumaierSum risk_set;
  NeumaierSum sum_1_over_risk_set;
  NeumaierSum sum_1_over_risk_set_sq;
  std::vector<double> snapshot_S1(log_hazard_predictions.size(), 0.0);
  std::vector<double> snapshot_S2(log_hazard_predictions.size(), 0.0);

  // Computes the gradient and the hessian of the example leaving the risk set,
  // and removes it from the risk set.
  const auto depart = [&](const row_t example_idx) {
    const double hazard = hazards[example_idx];
    const double dS1 =
        std::max(sum_1_over_risk_set.Value() - snapshot_S1[example_idx], 0.0);
    const double dS2 = std::max(
        sum_1_over_risk_set_sq.Value() - snapshot_S2[example_idx], 0.0);
    // hazard * dS2 <= dS1 since the example is part of each of the risk sets
    // accumulated in dS1 and dS2. The clamping only removes the residual
    // numerical noise.
    hessians[example_idx] =
        static_cast<float>(w * std::max(hazard * (dS1 - hazard * dS2), 0.0));
    risk_set.Add(-hazard);
    if (risk_set.Value() < 0.0) {
      LOG_EVERY_POW_2(WARNING)
          << "Cox loss has encountered a negative sum of hazards "
          << risk_set.Value() << " at risk. Setting it to 0.";
      risk_set.Reset();
    }
    return std::pair<double, double>{hazard, dS1};
  };

  // Updates are sorted by time and every item first arrives before
  // event / censor.
  for (const auto& [time, update_type, example_idx] : cox_cache->updates) {
    switch (update_type) {
      case CoxProportionalHazardLoss::Update::Type::ARRIVAL:
        snapshot_S1[example_idx] = sum_1_over_risk_set.Value();
        snapshot_S2[example_idx] = sum_1_over_risk_set_sq.Value();
        risk_set.Add(hazards[example_idx]);
        break;
      case CoxProportionalHazardLoss::Update::Type::EVENT: {
        // The example is part of its own risk set, so the sum of the hazards
        // at risk is at least "hazard", i.e. the max() returns
        // risk_set.Value(). Using the max just enforces it in case the
        // arithmetic drifted.
        const double sum_at_risk =
            std::max(risk_set.Value(), hazards[example_idx]);
        // sum_at_risk > 0 since all hazards are > 0.
        sum_1_over_risk_set.Add(1.0 / sum_at_risk);
        sum_1_over_risk_set_sq.Add(1.0 / (sum_at_risk * sum_at_risk));
        const auto [hazard, dS1] = depart(example_idx);
        gradients[example_idx] = static_cast<float>(w * (1.0 - hazard * dS1));
        break;
      }
      case CoxProportionalHazardLoss::Update::Type::CENSORING: {
        const auto [hazard, dS1] = depart(example_idx);
        gradients[example_idx] = static_cast<float>(w * (-hazard * dS1));
        break;
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::gradient_boosted_trees
