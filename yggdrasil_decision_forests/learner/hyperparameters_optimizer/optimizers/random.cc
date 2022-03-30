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
}

int RandomOptimizer::NumExpectedRounds() { return config_.num_trials(); }

// TODO(gbm): Weight by the number of children combinations.
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
      std::uniform_int_distribution<int> dist(
          0, field.discrete_candidates().possible_values().size() - 1);
      value = field.discrete_candidates().possible_values(dist(rnd_));
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

}  // namespace hyperparameters_optimizer_v2
}  // namespace model
}  // namespace yggdrasil_decision_forests
