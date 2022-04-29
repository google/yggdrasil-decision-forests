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

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_TESTING_UTILS_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_TESTING_UTILS_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"

namespace yggdrasil_decision_forests {
namespace model {

class FakeModel : public AbstractModel {
 public:
  FakeModel(const absl::string_view name = "FAKE_MODEL")
      : AbstractModel(name) {}

  absl::Status Save(absl::string_view directory,
                    const ModelIOOptions& io_options) const override {
    return absl::UnimplementedError("Save");
  }

  absl::Status Load(absl::string_view directory,
                    const ModelIOOptions& io_options) override {
    return absl::UnimplementedError("Load");
  }

  void Predict(const dataset::proto::Example& example,
               proto::Prediction* prediction) const override {
    LOG(FATAL) << "Unimplemented: Predict Example";
  }

  void Predict(const dataset::VerticalDataset& dataset,
               dataset::VerticalDataset::row_t row_idx,
               proto::Prediction* prediction) const override {
    LOG(FATAL) << "Unimplemented: Predict Dataset";
  }
};

}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_TESTING_UTILS_H_
