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

#include "yggdrasil_decision_forests/learner/postprocessor/smoothed_pav_calibrator/smoothed_pav_calibrator.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/postprocessor/smoothed_pav_calibrator/smoothed_pav_calibrator.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/postprocessor/smoothed_pav_calibrator/smoothed_pav_calibrator.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/smoothed_pav_calibration_fit.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

// utils.proto.IntegerDistributionFloat.counts indices:
// 0: out of vocabulary
// 1: class negative class
// 2: class positive class
const int kPositiveClassIndex = 2;

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
    const dataset::VerticalDataset& dataset) {
  RETURN_IF_ERROR(dataset::CheckNumExamples(dataset.nrow()));

  auto n_bins = postprocessor_config.n_bins();
  std::vector<yggdrasil_decision_forests::utils::BinAccumulator> bins(n_bins);
  {
    std::vector<std::vector<yggdrasil_decision_forests::utils::BinAccumulator>>
        thread_bins(
            deployment.num_threads(),
            std::vector<yggdrasil_decision_forests::utils::BinAccumulator>(
                n_bins));
    yggdrasil_decision_forests::utils::concurrency::ThreadPool pool(
        deployment.num_threads(), {.name_prefix = std::string("Calibrate")});
    const auto* labels =
        dataset.ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
            config_link.label());

    RETURN_IF_ERROR(utils::concurrency::ConcurrentForLoopWithStatus(
        deployment.num_threads(), &pool, dataset.nrow(),
        [&model, &dataset, &labels, &thread_bins](
            size_t block_idx, size_t begin_idx,
            size_t end_idx) -> absl::Status {
          auto& block_bins = thread_bins[block_idx];

          auto prediction = std::make_unique<model::proto::Prediction>();
          for (int row_idx = begin_idx; row_idx < end_idx; ++row_idx) {
            model.Predict(dataset, row_idx, prediction.get());
            auto sum = prediction->classification().distribution().sum();
            auto pred = prediction->classification().distribution().counts(
                            kPositiveClassIndex) /
                        sum;
            auto label = labels->values()[row_idx];
            RETURN_IF_ERROR(yggdrasil_decision_forests::utils::accumulate_bins(
                block_bins, {pred},
                {static_cast<float>(label == kPositiveClassIndex ? 1.0 : 0.0)},
                block_bins.size()));
          }
          return absl::OkStatus();
        }));

    for (auto thread_idx = 0; thread_idx < deployment.num_threads();
         ++thread_idx) {
      for (auto bin_idx = 0; bin_idx < n_bins; ++bin_idx) {
        bins[bin_idx] += thread_bins[thread_idx][bin_idx];
      }
    }
  }

  LOG(INFO) << "Number of calibration bins before merging: " << bins.size();

  // Fit the calibration curve.
  ASSIGN_OR_RETURN(
      yggdrasil_decision_forests::utils::FittedCalibrationCurve curve,
      yggdrasil_decision_forests::utils::fit_calibration(
          bins, postprocessor_config.z_threshold()));

  LOG(INFO) << "Number of calibration bins after merging: " << curve.x.size();

  // Create the postprocessor with the lookup table.
  smoothed_pav_calibrator::proto::SmoothedPavCalibrator postprocessor_proto;
  postprocessor_proto.set_n_grid(postprocessor_config.n_grid());
  for (const auto& x : curve.x) {
    postprocessor_proto.add_x(x);
  }
  for (const auto& y : curve.y) {
    postprocessor_proto.add_y(y);
  }
  for (const auto& d : curve.d) {
    postprocessor_proto.add_slope(d);
  }

  return std::make_shared<
      yggdrasil_decision_forests::model::postprocessor::SmoothedPavCalibrator>(
      postprocessor_proto);
}

}  // namespace smoothed_pav_calibrator
}  // namespace yggdrasil_decision_forests::model::postprocessor
