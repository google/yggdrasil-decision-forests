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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_library.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/types/variant.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_custom_binary_classification.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_custom_multi_classification.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_imp_custom_regression.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_interface.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

namespace {

absl::StatusOr<std::unique_ptr<AbstractLoss>> CreateCustomLoss(
    model::proto::Task task, const dataset::proto::Column& label_column,
    const proto::GradientBoostedTreesTrainingConfig& config,
    CustomLossFunctions loss_functions) {
  std::unique_ptr<AbstractLoss> custom_loss;
  if (absl::holds_alternative<CustomBinaryClassificationLossFunctions>(
          loss_functions)) {
    custom_loss = absl::make_unique<CustomBinaryClassificationLoss>(
        config, task, label_column,
        absl::get<CustomBinaryClassificationLossFunctions>(loss_functions));
  } else if (absl::holds_alternative<CustomMultiClassificationLossFunctions>(
                 loss_functions)) {
    custom_loss = absl::make_unique<CustomMultiClassificationLoss>(
        config, task, label_column,
        absl::get<CustomMultiClassificationLossFunctions>(loss_functions));
  } else if (absl::holds_alternative<CustomRegressionLossFunctions>(
                 loss_functions)) {
    custom_loss = absl::make_unique<CustomRegressionLoss>(
        config, task, label_column,
        absl::get<CustomRegressionLossFunctions>(loss_functions));
  } else {
    NOTREACHED();
  }
  RETURN_IF_ERROR(custom_loss->Status());
  return custom_loss;
}

}  // namespace

absl::StatusOr<std::unique_ptr<AbstractLoss>> CreateLoss(
    proto::Loss loss, model::proto::Task task,
    const dataset::proto::Column& label_column,
    const proto::GradientBoostedTreesTrainingConfig& config,
    CustomLossFunctions custom_loss_functions) {
  if (custom_loss_functions.index() != 0) {
    return CreateCustomLoss(task, label_column, config, custom_loss_functions);
  }

  auto loss_key = proto::Loss_Name(loss);
  if (loss == proto::LAMBDA_MART_NDCG5) {
    loss_key = "LAMBDA_MART_NDCG";
  }
  ASSIGN_OR_RETURN(auto loss_imp, AbstractLossRegisterer::Create(
                                      loss_key, config, task, label_column));
  RETURN_IF_ERROR(loss_imp->Status());
  return loss_imp;
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
