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

// Abstract classes for model and model builder (called learner).
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_LEARNER_LIBRARY_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_LEARNER_LIBRARY_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"

namespace yggdrasil_decision_forests {
namespace model {

// Get a learner.
absl::Status GetLearner(const proto::TrainingConfig& train_config,
                        std::unique_ptr<AbstractLearner>* learner,
                        const proto::DeploymentConfig& deployment_config = {});

// Returns the list of all registered learners.
std::vector<std::string> AllRegisteredLearners();

}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_LEARNER_LIBRARY_H_
