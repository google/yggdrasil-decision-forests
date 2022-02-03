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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_library.h"

#include <memory>

#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

utils::StatusOr<std::unique_ptr<AbstractLoss>> CreateLoss(
    proto::Loss loss, model::proto::Task task,
    const dataset::proto::Column& label_column,
    const proto::GradientBoostedTreesTrainingConfig& config) {
  const auto loss_key = proto::Loss_Name(loss);
  ASSIGN_OR_RETURN(auto loss_imp, AbstractLossRegisterer::Create(
                                      loss_key, config, task, label_column));
  RETURN_IF_ERROR(loss_imp->Status());
  return loss_imp;
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
