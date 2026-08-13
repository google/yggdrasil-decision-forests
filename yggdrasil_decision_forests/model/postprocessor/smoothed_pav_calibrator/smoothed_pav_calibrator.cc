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

#include "yggdrasil_decision_forests/model/postprocessor/smoothed_pav_calibrator/smoothed_pav_calibrator.h"

#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/smoothed_pav_calibration_inference.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace postprocessor {

// utils.proto.IntegerDistributionFloat.counts indices:
// 0: out of vocabulary
// 1: class negative class
// 2: class positive class
const int kNegativeClassIndex = 1;
const int kPositiveClassIndex = 2;

void update_prediction(
    const utils::CalibrationLookupTable& calibration_lookup_table,
    yggdrasil_decision_forests::model::proto::Prediction* prediction) {
  auto sum = prediction->classification().distribution().sum();
  auto calibrated_score = calibration_lookup_table.apply(
      prediction->classification().distribution().counts(kPositiveClassIndex) /
      sum);

  auto counts = prediction->mutable_classification()
                    ->mutable_distribution()
                    ->mutable_counts();
  // Update the _counts_ as we don't directly store the probabilities.
  counts->Set(kPositiveClassIndex, calibrated_score * sum);
  counts->Set(kNegativeClassIndex, (1.0 - calibrated_score) * sum);
}

void SmoothedPavCalibrator::ProcessImpl(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t row_idx,
    yggdrasil_decision_forests::model::proto::Prediction* prediction) const {
  update_prediction(calibration_lookup_table_, prediction);
}

void SmoothedPavCalibrator::ProcessImpl(
    const dataset::proto::Example& example,
    yggdrasil_decision_forests::model::proto::Prediction* prediction) const {
  update_prediction(calibration_lookup_table_, prediction);
}

void SmoothedPavCalibrator::ExportProtoImpl(
    proto::AbstractPostprocessor* proto) const {
  *proto->mutable_smoothed_pav_calibrator() = proto_;
}

void SmoothedPavCalibrator::AppendDescriptionImpl(
    std::string* description) const {
  absl::StrAppend(description, "Smoothed PAV calibrator\n");
  absl::StrAppend(description, "Number of raw bins: ", proto_.x_size(), "\n");
  absl::StrAppend(description, "Number of lookup table bins: ",
                  calibration_lookup_table_.grid_size(), "\n");
}

std::unique_ptr<SmoothedPavCalibrator> Create(
    const smoothed_pav_calibrator::proto::SmoothedPavCalibrator& proto) {
  return std::make_unique<SmoothedPavCalibrator>(proto);
}

}  // namespace postprocessor
}  // namespace model
}  // namespace yggdrasil_decision_forests
