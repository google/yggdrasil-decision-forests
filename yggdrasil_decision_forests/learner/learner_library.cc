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

#include "yggdrasil_decision_forests/learner/learner_library.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"

namespace yggdrasil_decision_forests {
namespace model {

std::vector<std::string> AllRegisteredLearners() {
  return AbstractLearnerRegisterer::GetNames();
}

absl::Status GetLearner(const proto::TrainingConfig& train_config,
                        std::unique_ptr<AbstractLearner>* learner,
                        const proto::DeploymentConfig& deployment_config) {
  if (train_config.learner().empty()) {
    return absl::InvalidArgumentError("\"learner\" field required.");
  }

  auto effective_train_config = train_config;

  ASSIGN_OR_RETURN(
      *learner,
      AbstractLearnerRegisterer::Create(effective_train_config.learner(),
                                        effective_train_config),
      _ << "The learner is either non-existing or non registered.");
  *learner->get()->mutable_deployment() = deployment_config;
  return (*learner)->CheckCapabilities();
}

}  // namespace model
}  // namespace yggdrasil_decision_forests
