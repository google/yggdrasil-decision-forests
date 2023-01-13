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

// A multitasker model is a model containing multiple sub-models solving
// different tasks. Individual models of a multitasker model are generally
// trained in parallel using the multitasker learner.
//
// Warning: The "Predict" methods of a multitasker model is calling the
// "Predict" method on the first sub-model.
//
#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_MULTITASKER_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_MULTITASKER_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace multitasker {

class MultitaskerLearner;

class MultitaskerModel : public AbstractModel {
 public:
  static constexpr char kRegisteredName[] = "MULTITASKER";

  MultitaskerModel() : AbstractModel(kRegisteredName) {}

  absl::Status Save(absl::string_view directory,
                    const ModelIOOptions& io_options) const override;

  absl::Status Load(absl::string_view directory,
                    const ModelIOOptions& io_options) override;

  absl::Status Validate() const override;

  // Generate a predictions with the first model. To make predictions with the
  // other models, use "models(model_idx)->Predict(...)".
  void Predict(const dataset::VerticalDataset& dataset,
               dataset::VerticalDataset::row_t row_idx,
               model::proto::Prediction* prediction) const override;

  void Predict(const dataset::proto::Example& example,
               model::proto::Prediction* prediction) const override;

  void AppendDescriptionAndStatistics(bool full_definition,
                                      std::string* description) const override;

  const AbstractModel* model(int index) const { return models_[index].get(); }

  const std::vector<std::unique_ptr<AbstractModel>>& models() {
    return models_;
  }
  std::vector<std::unique_ptr<AbstractModel>>* mutable_models() {
    return &models_;
  }

 private:
  std::vector<std::unique_ptr<AbstractModel>> models_;

  friend MultitaskerLearner;
};

REGISTER_AbstractModel(MultitaskerModel, MultitaskerModel::kRegisteredName);

}  // namespace multitasker
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_MULTITASKER_H_
