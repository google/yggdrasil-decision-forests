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

#include "ydf/model/gradient_boosted_trees_model/gradient_boosted_trees_wrapper.h"

#include <pybind11/numpy.h>

#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"

namespace yggdrasil_decision_forests::port::python {

absl::StatusOr<std::unique_ptr<GradientBoostedTreesCCModel>>
GradientBoostedTreesCCModel::Create(
    std::unique_ptr<model::AbstractModel>& model_ptr) {
  auto* gbt_model = dynamic_cast<YDFModel*>(model_ptr.get());
  if (gbt_model == nullptr) {
    return absl::InvalidArgumentError(
        "This model is not a gradient boosted trees model.");
  }
  // Both release and the unique_ptr constructor are noexcept.
  model_ptr.release();
  std::unique_ptr<YDFModel> new_model_ptr(gbt_model);

  return std::make_unique<GradientBoostedTreesCCModel>(std::move(new_model_ptr),
                                                       gbt_model);
}

py::array_t<float> GradientBoostedTreesCCModel::initial_predictions() const {
  py::array_t<float, py::array::c_style | py::array::forcecast>
      initial_predictions;
  const auto& gbt_initial_predictions = gbt_model_->initial_predictions();
  static_assert(initial_predictions.itemsize() == sizeof(float),
                "A C++ float should have the same size as a numpy float");
  initial_predictions.resize({gbt_initial_predictions.size()});
  std::memcpy(initial_predictions.mutable_data(),
              gbt_initial_predictions.data(),
              initial_predictions.size() * sizeof(float));
  return initial_predictions;
}

void GradientBoostedTreesCCModel::set_initial_predictions(
    const py::array_t<float>& values) {
  std::vector<float> std_values(values.size(), 0.0f);
  for (int i = 0; i < values.size(); i++) {
    std_values[i] = values.at(i);
  }
  gbt_model_->set_initial_predictions(std::move(std_values));
}

absl::StatusOr<bool> GradientBoostedTreesCCModel::output_logits() const {
  if (gbt_model_->task() != model::proto::CLASSIFICATION) {
    return absl::InvalidArgumentError(
        "output_logits is only supported for classification tasks.");
  }
  return gbt_model_->output_logits();
}

absl::Status GradientBoostedTreesCCModel::set_output_logits(
    const bool output_logits) {
  if (gbt_model_->task() != model::proto::CLASSIFICATION) {
    return absl::InvalidArgumentError(
        "output_logits is only supported for classification tasks.");
  }
  invalidate_engine_ = true;
  gbt_model_->set_output_logits(output_logits);
  return absl::OkStatus();
}

std::vector<GBTCCTrainingLogEntry> GradientBoostedTreesCCModel::training_logs()
    const {
  std::vector<GBTCCTrainingLogEntry> logs;
  const auto& training_logs = gbt_model_->training_logs();
  const auto& label_col_spec = gbt_model_->label_col_spec();
  logs.reserve(training_logs.entries_size());
  for (const auto& entry : training_logs.entries()) {
    const auto& validation_evaluation =
        model::gradient_boosted_trees::internal::TrainingLogToEvaluationResults(
            entry, training_logs, gbt_model_->task(), label_col_spec,
            gbt_model_->loss_config(), gbt_model_->GetLossName(),
            model::gradient_boosted_trees::internal::TrainingLogEvaluationSet::
                kValidation);
    const auto& training_evaluation =
        model::gradient_boosted_trees::internal::TrainingLogToEvaluationResults(
            entry, training_logs, gbt_model_->task(), label_col_spec,
            gbt_model_->loss_config(), gbt_model_->GetLossName(),
            model::gradient_boosted_trees::internal::TrainingLogEvaluationSet::
                kTraining);
    logs.push_back({.iteration = entry.number_of_trees(),
                    .validation_evaluation = validation_evaluation,
                    .training_evaluation = training_evaluation});
  }
  return logs;
}

}  // namespace yggdrasil_decision_forests::port::python
