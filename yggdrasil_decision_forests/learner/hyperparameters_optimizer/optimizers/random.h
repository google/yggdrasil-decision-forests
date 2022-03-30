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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_HYPERPARAMETER_OPTIMIZER_RANDOM_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_HYPERPARAMETER_OPTIMIZER_RANDOM_H_

#include "absl/container/node_hash_set.h"
#include "yggdrasil_decision_forests/learner/hyperparameters_optimizer/optimizer_interface.h"
#include "yggdrasil_decision_forests/learner/hyperparameters_optimizer/optimizers/random.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace hyperparameters_optimizer_v2 {

// Randomly test a given number of hyper-parameters.
class RandomOptimizer : public OptimizerInterface {
 public:
  // Unique identifier of the optimizer algorithm.
  static constexpr char kRegisteredName[] = "RANDOM";

  RandomOptimizer(const proto::Optimizer& config,
                  const model::proto::HyperParameterSpace& space);

  utils::StatusOr<NextCandidateStatus> NextCandidate(
      model::proto::GenericHyperParameters* candidate) override;

  absl::Status ConsumeEvaluation(
      const model::proto::GenericHyperParameters& candidate,
      const double score) override;

  std::pair<model::proto::GenericHyperParameters, double> BestParameters()
      override;

  int NumExpectedRounds() override;

 private:
  // Build recursively a set of random hyper-parameter values.
  absl::Status BuildRandomSet(
      const model::proto::HyperParameterSpace::Field& field,
      model::proto::GenericHyperParameters* candidate);

  // Configuration of the optimizer.
  proto::RandomOptimizerConfig config_;

  // Search space.
  model::proto::HyperParameterSpace space_;

  // Best hyper-parameter found so far.
  model::proto::GenericHyperParameters best_params_;

  // Score of the best hyper-parameter found so far.
  double best_score_ = std::numeric_limits<double>::quiet_NaN();

  // Random generator.
  utils::RandomEngine rnd_;

  // Number of generated candidates without yet an evaluation.
  int pending_evaluations_ = 0;

  // Number of generated candidates with an evaluation.
  int observed_evaluations_ = 0;

  // Set of already generated candidates (to avoid duplicates).
  // Note: We use the string representation of the proto as a unique identifier.
  absl::node_hash_set<std::string> already_proposed_candidates_;

  // Number of tries to generate a unique new candidate. If no new candidate
  // could be generated, the optimizer considers that no new candidates are
  // available.
  const int num_tries_per_candidates_ = 512;

  absl::Status constructor_status_;
};

REGISTER_AbstractHyperParametersOptimizer(RandomOptimizer,
                                          RandomOptimizer::kRegisteredName);

namespace internal {

// Computes the "weight" (set in the "weight" field) of each field in the
// hyper-parameter space. If "weight" is not already specified, all the
// hyper-parameter combination have the same probability of sampling. If
// "weight" is specified, it is applied as a coefficient factor over the uniform
// sampling.
absl::Status UpdateWeights(model::proto::HyperParameterSpace* space);
utils::StatusOr<double> UpdateWeights(
    model::proto::HyperParameterSpace::Field* field);

utils::StatusOr<size_t> Sample(std::vector<float>& weights,
                               utils::RandomEngine* random);

}  // namespace internal

}  // namespace hyperparameters_optimizer_v2
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_HYPERPARAMETER_OPTIMIZER_RANDOM_H_
