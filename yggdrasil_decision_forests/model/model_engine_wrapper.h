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

// Wraps an inference engine (i.e. a model compiled and optimized for fast
// inference) into a model.
//
// Note: This solution is significantly slower than using the fast engine
// directly as it computes expensive example format conversion and does not use
// batched predictions.
//
// The wrapper does not model serialization functions.
#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_MODEL_ENGINE_WRAPPER_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_MODEL_ENGINE_WRAPPER_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"

namespace yggdrasil_decision_forests {
namespace model {

class EngineWrapperModel : public AbstractModel {
 public:
  static constexpr char kRegisteredName[] = "ENGINE_WRAPPER";

  EngineWrapperModel(const model::AbstractModel* const model,
                     std::unique_ptr<serving::FastEngine> engine)
      : AbstractModel(kRegisteredName), engine_(std::move(engine)) {
    set_task(model->task());
    set_data_spec(model->data_spec());
    input_features_ = model->input_features();
    num_prediction_dimensions_ = engine_->NumPredictionDimension();
    set_label_col_idx(model->label_col_idx());
    set_ranking_group_col(model->ranking_group_col_idx());
  }

  absl::Status Save(absl::string_view directory,
                    const ModelIOOptions& io_options) const override {
    return absl::InvalidArgumentError(
        "Engine wrapper doesn't support model serialization");
  }

  absl::Status Load(absl::string_view directory,
                    const ModelIOOptions& io_options) override {
    return absl::InvalidArgumentError(
        "Engine wrapper don't support model serialization");
  }

  absl::Status Validate() const override { return absl::OkStatus(); }

  void Predict(const dataset::VerticalDataset& dataset,
               dataset::VerticalDataset::row_t row_idx,
               model::proto::Prediction* prediction) const override;

  void Predict(const dataset::proto::Example& example,
               model::proto::Prediction* prediction) const override;

 private:
  std::unique_ptr<serving::FastEngine> engine_;
  int num_prediction_dimensions_;
};

}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_MODEL_ENGINE_WRAPPER_H_
