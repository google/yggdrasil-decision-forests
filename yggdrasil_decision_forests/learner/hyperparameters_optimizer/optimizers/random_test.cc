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

#include <limits>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/learner/hyperparameters_optimizer/hyperparameters_optimizer.pb.h"
#include "yggdrasil_decision_forests/learner/hyperparameters_optimizer/optimizer_interface.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/test_utils.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace hyperparameters_optimizer_v2 {
namespace {

using test::EqualsProto;

TEST(Random, Base) {
  model::proto::HyperParameterSpace search_space = PARSE_TEST_PROTO(R"pb(
    fields {
      name: "a"
      discrete_candidates {
        possible_values { integer: 1 }
        possible_values { integer: 2 }
      }
    }
    fields {
      name: "b"
      discrete_candidates {
        possible_values { integer: 3 }
        possible_values { integer: 4 }
        possible_values { integer: 5 }
      }
      children {
        parent_discrete_values {
          possible_values { integer: 3 }
          possible_values { integer: 4 }
        }
        name: "c"
        discrete_candidates {
          possible_values { integer: 1 }
          possible_values { integer: 2 }
        }
      }
      children {
        parent_discrete_values {
          possible_values { integer: 4 }
          possible_values { integer: 5 }
        }
        name: "d"
        discrete_candidates {
          possible_values { integer: 1 }
          possible_values { integer: 2 }
        }
      }
    }
  )pb");

  proto::Optimizer optimizer_config;
  auto& spe_config = *optimizer_config.MutableExtension(proto::random);
  spe_config.set_num_trials(100);

  RandomOptimizer optimizer(optimizer_config, search_space);
  int trial_idx = 0;
  while (true) {
    model::proto::GenericHyperParameters candidate;
    auto status = optimizer.NextCandidate(&candidate).value();
    if (status == NextCandidateStatus::kExplorationIsDone) {
      // No more parameters to evaluate.
      break;
    } else if (status == NextCandidateStatus::kWaitForEvaluation) {
      LOG(FATAL) << "Should not append. As no evaluation pending.";
    }

    LOG(INFO) << "candidate: " << candidate.ShortDebugString();

    double evaluation = trial_idx % 5;
    if (evaluation < 0.5) {
      evaluation = std::numeric_limits<double>::quiet_NaN();
    }
    CHECK_OK(optimizer.ConsumeEvaluation(candidate, evaluation));
    trial_idx++;
  }

  EXPECT_EQ(trial_idx, 16);

  model::proto::GenericHyperParameters best_params;
  double best_score;
  std::tie(best_params, best_score) = optimizer.BestParameters();

  EXPECT_NEAR(best_score, 4, 0.0001);

  LOG(INFO) << "trial_idx: " << trial_idx << " score: " << best_score
            << " params: " << best_params.DebugString();
}

TEST(Random, UpdateWeights) {
  model::proto::HyperParameterSpace space = PARSE_TEST_PROTO(R"pb(
    fields {
      name: "a"
      discrete_candidates {
        possible_values { integer: 1 }
        possible_values { integer: 2 }
      }
    }
    fields {
      name: "b"
      discrete_candidates {
        possible_values { integer: 3 }
        possible_values { integer: 4 }
        possible_values { integer: 5 }
        possible_values { integer: 6 }
      }
      children {
        parent_discrete_values {
          possible_values { integer: 3 }
          possible_values { integer: 4 }
        }
        name: "c"
        discrete_candidates {
          possible_values { integer: 1 }
          possible_values { integer: 2 }
        }
      }
      children {
        parent_discrete_values {
          possible_values { integer: 4 }
          possible_values { integer: 5 }
        }
        name: "d"
        discrete_candidates {
          possible_values { integer: 1 }
          possible_values { integer: 2 }
        }
      }
    }
  )pb");
  CHECK_OK(internal::UpdateWeights(&space));

  const model::proto::HyperParameterSpace expected_space = PARSE_TEST_PROTO(
      R"pb(
        fields {
          name: "a"
          discrete_candidates {
            possible_values { integer: 1 }
            possible_values { integer: 2 }
            weights: 1.0
            weights: 1.0
          }
        }
        fields {
          name: "b"
          discrete_candidates {
            possible_values { integer: 3 }
            possible_values { integer: 4 }
            possible_values { integer: 5 }
            possible_values { integer: 6 }
            weights: 2.0
            weights: 4.0
            weights: 2.0
            weights: 1.0
          }
          children {
            parent_discrete_values {
              possible_values { integer: 3 }
              possible_values { integer: 4 }
            }
            name: "c"
            discrete_candidates {
              possible_values { integer: 1 }
              possible_values { integer: 2 }
              weights: 1.0
              weights: 1.0
            }
          }
          children {
            parent_discrete_values {
              possible_values { integer: 4 }
              possible_values { integer: 5 }
            }
            name: "d"
            discrete_candidates {
              possible_values { integer: 1 }
              possible_values { integer: 2 }
              weights: 1.0
              weights: 1.0
            }
          }
        }
      )pb");
  EXPECT_THAT(space, EqualsProto(expected_space));
}

TEST(Benchmark, Sample) {
  utils::RandomEngine random;
  std::vector<float> weights{0, 0.5, 0, 1.0};
  for (int i = 0; i < 100; i++) {
    int sample = internal::Sample(weights, &random).value();
    EXPECT_TRUE(sample == 1 || sample == 3);
  }

  std::vector<float> weights_2{1.0, 1.0, 1.0, 1.0};
  for (int i = 0; i < 100; i++) {
    int sample_2 = internal::Sample(weights, &random).value();
    EXPECT_TRUE(sample_2 == 1 || sample_2 == 3);
  }
}

}  // namespace
}  // namespace hyperparameters_optimizer_v2
}  // namespace model
}  // namespace yggdrasil_decision_forests
