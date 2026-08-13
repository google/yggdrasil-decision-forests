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

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_POSTPROCESSOR_SMOOTHED_PAV_CALIBRATOR_SMOOTHED_PAV_CALIBRATOR_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_POSTPROCESSOR_SMOOTHED_PAV_CALIBRATOR_SMOOTHED_PAV_CALIBRATOR_H_

#include <memory>
#include <string>
#include <vector>

#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/postprocessor/abstract_postprocessor.h"
#include "yggdrasil_decision_forests/model/postprocessor/smoothed_pav_calibrator/smoothed_pav_calibrator.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/utils/smoothed_pav_calibration_fit.h"
#include "yggdrasil_decision_forests/utils/smoothed_pav_calibration_inference.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace postprocessor {

class SmoothedPavCalibrator : public AbstractPostprocessor {
 public:
  explicit SmoothedPavCalibrator(
      const smoothed_pav_calibrator::proto::SmoothedPavCalibrator& proto)
      : proto_(proto),
        calibration_lookup_table_(utils::CalibrationLookupTable::Create(
            utils::FittedCalibrationCurve(
                std::vector<utils::BinAccumulator::AccumulatorType>(
                    proto.x().begin(), proto.x().end()),
                std::vector<utils::BinAccumulator::AccumulatorType>(
                    proto.y().begin(), proto.y().end()),
                std::vector<utils::BinAccumulator::AccumulatorType>(
                    proto.slope().begin(), proto.slope().end())),
            proto.n_grid())) {}

 private:
  void ProcessImpl(const dataset::VerticalDataset& dataset,
                   dataset::VerticalDataset::row_t row_idx,
                   yggdrasil_decision_forests::model::proto::Prediction*
                       prediction) const override;

  void ProcessImpl(const dataset::proto::Example& example,
                   yggdrasil_decision_forests::model::proto::Prediction*
                       prediction) const override;

  void ExportProtoImpl(proto::AbstractPostprocessor* proto) const override;

  void AppendDescriptionImpl(std::string* description) const override;

  smoothed_pav_calibrator::proto::SmoothedPavCalibrator proto_;
  utils::CalibrationLookupTable calibration_lookup_table_;
};

std::unique_ptr<SmoothedPavCalibrator> Create(
    const smoothed_pav_calibrator::proto::SmoothedPavCalibrator& proto);

}  // namespace postprocessor
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_POSTPROCESSOR_SMOOTHED_PAV_CALIBRATOR_SMOOTHED_PAV_CALIBRATOR_H_
