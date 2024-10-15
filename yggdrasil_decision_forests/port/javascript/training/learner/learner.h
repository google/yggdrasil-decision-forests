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

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_JAVASCRIPT_TRAINING_LEARNER_LEARNER_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_JAVASCRIPT_TRAINING_LEARNER_LEARNER_H_

#include <memory>
#include <string>

#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/port/javascript/training/dataset/dataset.h"
#include "yggdrasil_decision_forests/port/javascript/training/model/model.h"
#include "yggdrasil_decision_forests/port/javascript/training/util/status_casters.h"

namespace yggdrasil_decision_forests::port::javascript {

class Learner {
 public:
  Learner() {}

  void Init(std::string learner, std::string label, std::string task);

  std::shared_ptr<Model> TrainFromPath(std::string path);

  std::shared_ptr<Model> TrainFromDataset(const Dataset& dataset);

 private:
  bool initilized_ = false;
  std::unique_ptr<model::AbstractLearner> learner_ptr_;
};

void init_learner();

}  // namespace yggdrasil_decision_forests::port::javascript
#endif  // YGGDRASIL_DECISION_FORESTS_PORT_JAVASCRIPT_TRAINING_LEARNER_LEARNER_H_
