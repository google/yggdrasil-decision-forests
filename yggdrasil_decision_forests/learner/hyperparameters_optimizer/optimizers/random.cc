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

#include "yggdrasil_decision_forests/learner/hyperparameters_optimizer/optimizers/random.h"

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/learner/hyperparameters_optimizer/optimizers/random.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace hyperparameters_optimizer_v2 {

constexpr char RandomOptimizer::kRegisteredName[];

RandomOptimizer::RandomOptimizer(const proto::Optimizer& config,
                                 const model::proto::HyperParameterSpace& space)
    : OptimizerInterface(config, space), space_(space) {
  config_ = config.GetExtension(proto::random);
  constructor_status_ = internal::UpdateWeights(&space_);
}

int RandomOptimizer::NumExpectedRounds() { return config_.num_trials(); }

absl::Status RandomOptimizer::BuildRandomSet(
    const model::proto::HyperParameterSpace::Field& field,
    model::proto::GenericHyperParameters* candidate) {
  // Create the field.
  auto* field_value = candidate->add_fields();
  field_value->set_name(field.name());
  auto& value = *field_value->mutable_value();

  // Select a random value.
  switch (field.Type_case()) {
    case model::proto::HyperParameterSpace::Field::TypeCase::
        kDiscreteCandidates: {
      std::vector<float> sampling_weight{
          field.discrete_candidates().weights().begin(),
          field.discrete_candidates().weights().end()};
      ASSIGN_OR_RETURN(const auto selected_value_idx,
                       internal::Sample(sampling_weight, &rnd_));
      value = field.discrete_candidates().possible_values(selected_value_idx);
    } break;
    default:
      return absl::InvalidArgumentError("Type of search space not supported");
  }

  // Call the children matching this parent value.
  for (const auto& child : field.children()) {
    bool match_child = false;
    for (const auto& match_value :
         child.parent_discrete_values().possible_values()) {
      if (match_value.DebugString() == value.DebugString()) {
        match_child = true;
        break;
      }
    }
    if (match_child) {
      RETURN_IF_ERROR(BuildRandomSet(child, candidate));
    }
  }
  return absl::OkStatus();
}

utils::StatusOr<NextCandidateStatus> RandomOptimizer::NextCandidate(
    model::proto::GenericHyperParameters* candidate) {
  RETURN_IF_ERROR(constructor_status_);

  if (observed_evaluations_ >= config_.num_trials()) {
    return NextCandidateStatus::kExplorationIsDone;
  }

  if (observed_evaluations_ + pending_evaluations_ >= config_.num_trials()) {
    return NextCandidateStatus::kWaitForEvaluation;
  }

  int tries_left = num_tries_per_candidates_;

  while (tries_left > 0) {
    candidate->Clear();

    for (const auto& field : space_.fields()) {
      RETURN_IF_ERROR(BuildRandomSet(field, candidate));
    }

    if (already_proposed_candidates_.find(candidate->ShortDebugString()) ==
        already_proposed_candidates_.end()) {
      break;
    }

    tries_left--;
    if (tries_left == 0) {
      if (pending_evaluations_ > 0) {
        return NextCandidateStatus::kWaitForEvaluation;
      } else {
        return NextCandidateStatus::kExplorationIsDone;
      }
    }
  }

  already_proposed_candidates_.insert(candidate->ShortDebugString());
  pending_evaluations_++;
  return NextCandidateStatus::kNewCandidateAvailable;
}

absl::Status RandomOptimizer::ConsumeEvaluation(
    const model::proto::GenericHyperParameters& candidate, const double score) {
  observed_evaluations_++;
  pending_evaluations_--;
  DCHECK_GE(pending_evaluations_, 0);
  DCHECK_LE(observed_evaluations_, config_.num_trials());
  if (std::isnan(score)) {
    // Unfeasible trial.
    return absl::OkStatus();
  }
  if (!std::isfinite(score)) {
    return absl::InvalidArgumentError("Non finite score");
  }
  if (std::isnan(best_score_) || score > best_score_) {
    best_score_ = score;
    best_params_ = candidate;
  }
  return absl::OkStatus();
}

std::pair<model::proto::GenericHyperParameters, double>
RandomOptimizer::BestParameters() {
  return std::make_pair(best_params_, best_score_);
}
namespace internal {

absl::Status UpdateWeights(model::proto::HyperParameterSpace* space) {
  for (auto& field : *space->mutable_fields()) {
    RETURN_IF_ERROR(UpdateWeights(&field).status());
  }
  return absl::OkStatus();
}

utils::StatusOr<double> UpdateWeights(
    model::proto::HyperParameterSpace::Field* field) {
  if (!field->has_discrete_candidates()) {
    return absl::InvalidArgumentError("Discrete candidate missing");
  }

  const bool has_weights = field->discrete_candidates().weights_size() != 0;
  if (has_weights) {
    if (field->discrete_candidates().weights_size() !=
        field->discrete_candidates().possible_values_size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The number of weights of the field ", field->name(),
          " does not match the number of possible discret candidates"));
    }
  } else {
    // Allocate the weights.
    field->mutable_discrete_candidates()->mutable_weights()->Resize(
        field->discrete_candidates().possible_values_size(), 1.0);
  }

  // Recursively update the weight of the children, and collect the number of
  // combinations in each children.
  std::vector<double> children_weights(field->children_size());
  for (int child_idx = 0; child_idx < field->children_size(); child_idx++) {
    auto& child = *field->mutable_children(child_idx);
    ASSIGN_OR_RETURN(children_weights[child_idx], UpdateWeights(&child));
  }

  // Number of combination for "field".
  double field_weight = 0.0;

  for (int value_idx = 0;
       value_idx < field->discrete_candidates().possible_values_size();
       value_idx++) {
    // Number of combinations for this specific field value.
    auto& value_weight =
        *field->mutable_discrete_candidates()->mutable_weights()->Mutable(
            value_idx);

    // Count the number of comination in the children.
    const auto& value = field->discrete_candidates().possible_values(value_idx);
    for (int child_idx = 0; child_idx < field->children_size(); child_idx++) {
      // Check if the child is compatible with this specific value.
      const auto& child = field->children(child_idx);
      bool match_child = false;
      for (const auto& match_value :
           child.parent_discrete_values().possible_values()) {
        if (match_value.DebugString() == value.DebugString()) {
          match_child = true;
          break;
        }
      }
      if (match_child) {
        value_weight *= children_weights[child_idx];
      }
    }

    field_weight += value_weight;
  }

  return field_weight;
}

utils::StatusOr<size_t> Sample(std::vector<float>& weights,
                               utils::RandomEngine* random) {
  const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (sum <= 0) {
    return absl::InvalidArgumentError("Zero weight sum");
  }
  const double sample = std::uniform_real_distribution<double>(0, sum)(*random);
  double acc_sum = 0;
  for (size_t idx = 0; idx < weights.size(); idx++) {
    acc_sum += weights[idx];
    if (sample < acc_sum) {
      return idx;
    }
  }
  return weights.size() - 1;
}

}  // namespace internal
}  // namespace hyperparameters_optimizer_v2
}  // namespace model
}  // namespace yggdrasil_decision_forests
