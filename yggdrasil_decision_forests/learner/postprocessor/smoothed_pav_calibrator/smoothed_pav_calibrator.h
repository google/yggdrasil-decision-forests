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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_POSTPROCESSOR_SMOOTHED_PAV_CALIBRATOR_SMOOTHED_PAV_CALIBRATOR_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_POSTPROCESSOR_SMOOTHED_PAV_CALIBRATOR_SMOOTHED_PAV_CALIBRATOR_H_

#include <memory>

#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/postprocessor/smoothed_pav_calibrator/smoothed_pav_calibrator.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/postprocessor/smoothed_pav_calibrator/smoothed_pav_calibrator.h"

namespace yggdrasil_decision_forests::model::postprocessor {
namespace smoothed_pav_calibrator {

absl::StatusOr<std::shared_ptr<
    yggdrasil_decision_forests::model::postprocessor::SmoothedPavCalibrator>>
CreateSmoothedPavCalibrator(
    const model::proto::DeploymentConfig& deployment,
    const model::proto::TrainingConfigLinking& config_link,
    const smoothed_pav_calibrator::proto::SmoothedPavCalibratorTrainingConfig&
        postprocessor_config,
    yggdrasil_decision_forests::model::AbstractModel& model,
    const dataset::VerticalDataset& dataset);

}  // namespace smoothed_pav_calibrator
}  // namespace yggdrasil_decision_forests::model::postprocessor

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_POSTPROCESSOR_SMOOTHED_PAV_CALIBRATOR_SMOOTHED_PAV_CALIBRATOR_H_
