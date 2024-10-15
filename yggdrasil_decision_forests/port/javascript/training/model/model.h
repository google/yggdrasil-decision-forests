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

#ifndef YGGDRASIL_DECISION_FORESTS_PORT_JAVASCRIPT_TRAINING_MODEL_MODEL_H_
#define YGGDRASIL_DECISION_FORESTS_PORT_JAVASCRIPT_TRAINING_MODEL_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/port/javascript/training/util/status_casters.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"

namespace yggdrasil_decision_forests::port::javascript {

// Definition of an input feature.
// This struct is the JS visible version of the serving::FeatureDef class.
struct InputFeature {
  // Identifier of the feature.
  std::string name;
  // String version of the column type (dataset::proto::ColumnType).
  std::string type;
  // Index of the feature in the inference engine.
  int internal_idx;
  // Index of the feature in the column spec.
  int spec_idx;
};

// Wrapper for a model::AbstractModel.
class Model {
 public:
  Model() {}

  Model(std::unique_ptr<serving::FastEngine>&& engine,
        std::unique_ptr<model::AbstractModel> model,
        std::vector<std::string> label_classes);

  // Lists the label classes of the model.
  std::vector<std::string> GetLabelClasses() { return label_classes_; }

  // Lists the input features of the model.
  std::vector<InputFeature> GetInputFeatures();

  // Creates a new batch of examples. Should be called before setting the
  // example values.
  void NewBatchOfExamples(int num_examples);

  // Sets the value of a numerical feature.
  void SetNumerical(int example_idx, int feature_id, float value);

  // Sets the value of a boolean feature.
  void SetBoolean(int example_idx, int feature_id, bool value);

  // Sets the value of a categorical feature.
  void SetCategoricalInt(int example_idx, int feature_id, int value);

  // Sets the value of a categorical feature.
  void SetCategoricalString(int example_idx, int feature_id, std::string value);

  // Sets the value of a categorical set feature.
  void SetCategoricalSetString(int example_idx, int feature_id,
                               std::vector<std::string> value);

  // Sets the value of a categorical set feature.
  void SetCategoricalSetInt(int example_idx, int feature_id,
                            std::vector<int> value);

  // Runs the model on the previously set features.
  std::vector<float> Predict();

  // Runs the model on the previously set features.
  std::vector<float> PredictFromPath(std::string path);

  std::string Describe();

  void Save(const std::string& path);

 private:
  // The underlying YDF model.
  std::unique_ptr<model::AbstractModel> model_;

  // Engine i.e. compiled version of the model.
  std::unique_ptr<serving::FastEngine> engine_;

  // Set of allocated examples.
  std::unique_ptr<serving::AbstractExampleSet> examples_;

  // Number of examples allocated in "examples_".
  int num_examples_ = -1;

  // Label classes of the model. Only used for classification models. Otherwise,
  // is empty.
  std::vector<std::string> label_classes_;
};

// Extracts the classification labels of a model.
absl::StatusOr<std::vector<std::string>> ExtractLabelClasses(
    const model::AbstractModel& model);

void init_model();

}  // namespace yggdrasil_decision_forests::port::javascript
#endif  // YGGDRASIL_DECISION_FORESTS_PORT_JAVASCRIPT_TRAINING_MODEL_MODEL_H_
