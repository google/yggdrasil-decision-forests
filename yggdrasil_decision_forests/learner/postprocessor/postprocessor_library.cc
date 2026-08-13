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

#include "yggdrasil_decision_forests/learner/postprocessor/postprocessor_library.h"

#include <memory>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/postprocessor/abstract_postprocessor.pb.h"
#include "yggdrasil_decision_forests/learner/postprocessor/smoothed_pav_calibrator/smoothed_pav_calibrator.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/postprocessor/abstract_postprocessor.h"

namespace yggdrasil_decision_forests::model::postprocessor {

absl::StatusOr<std::shared_ptr<postprocessor::AbstractPostprocessor>>
CreatePostprocessor(
    const model::proto::DeploymentConfig& deployment,
    const model::proto::TrainingConfigLinking& config_link,
    const postprocessor::proto::AbstractPostprocessorTrainingConfig&
        postprocessor_config,
    yggdrasil_decision_forests::model::AbstractModel& model,
    const dataset::VerticalDataset& dataset) {
  switch (postprocessor_config.postprocessor_case()) {
    case postprocessor::proto::AbstractPostprocessorTrainingConfig::
        kSmoothedPavCalibratorTrainingConfig:
      LOG(INFO) << "Creating a SmoothedPavCalibrator.";
      return smoothed_pav_calibrator::CreateSmoothedPavCalibrator(
          deployment, config_link,
          postprocessor_config.smoothed_pav_calibrator_training_config(), model,
          dataset);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown postprocessor config: ", postprocessor_config));
  }
}

}  // namespace yggdrasil_decision_forests::model::postprocessor
