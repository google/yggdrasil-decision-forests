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

#include <emscripten/bind.h>
#include <emscripten/emscripten.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace ydf = yggdrasil_decision_forests;

// Definition of an input feature.
// This struct is the JS visible version of the ydf::serving::FeatureDef class.
struct InputFeature {
  // Identifier of the feature.
  std::string name;
  // String version of the column type (dataset::proto::ColumnType).
  std::string type;
  // Index of the feature in the inference engine.
  int internal_idx;
};

// Model is the result of loading an Yggdrasil model and it contains all the
// object necessary to run inference.
//
// This class is not thread compatible.
//
// This class is expected to be used as follow:
//
// // Create a new batch of examples.
// .NewBatchOfExamples(3);
// // Set the example values.
// .SetNumerical(0,1,2);
// .SetNumerical(1,1,2);
// .SetNumerical(2,1,2);
// // Generate the predictions.
// .Predict();
//
class Model {
 public:
  Model() {}
  Model(std::unique_ptr<ydf::serving::FastEngine>&& engine)
      : engine_(std::move(engine)) {
    DCHECK(engine_);
  }

  // Lists the input features of the model.
  std::vector<InputFeature> GetInputFeatures() {
    std::vector<InputFeature> input_features;
    for (const auto& feature : engine_->features().input_features()) {
      input_features.push_back(
          {feature.name, ydf::dataset::proto::ColumnType_Name(feature.type),
           feature.internal_idx});
    }
    return input_features;
  }

  // Creates a new batch of examples. Should be called before setting the
  // example values.
  void NewBatchOfExamples(int num_examples) {
    if (num_examples < 0) {
      LOG(WARNING) << "num_examples should be positive";
      return;
    }
    if (!examples_ || num_examples != num_examples_) {
      num_examples_ = num_examples;
      // The number of examples has change. Re-allocate the buffer.
      examples_ = engine_->AllocateExamples(num_examples);
    }
    examples_->FillMissing(engine_->features());
  }

  // Sets the value of a numerical feature.
  void SetNumerical(int example_idx, int feature_id, float value) {
    if (example_idx >= num_examples_) {
      LOG(WARNING) << "example_idx should be less than the number of examples";
      return;
    }
    examples_->SetNumerical(example_idx, {feature_id}, value,
                            engine_->features());
  }

  // Sets the value of a categorical feature.
  void SetCategoricalInt(int example_idx, int feature_id, int value) {
    if (example_idx >= num_examples_) {
      LOG(WARNING) << "example_idx should be less than the number of examples";
      return;
    }
    examples_->SetCategorical(example_idx, {feature_id}, value,
                              engine_->features());
  }

  // Sets the value of a categorical feature.
  void SetCategoricalString(int example_idx, int feature_id,
                            std::string value) {
    if (example_idx >= num_examples_) {
      LOG(WARNING) << "example_idx should be less than the number of examples";
      return;
    }
    examples_->SetCategorical(example_idx, {feature_id}, value,
                              engine_->features());
  }

  // Sets the value of a categorical set feature.
  void SetCategoricalSetString(int example_idx, int feature_id,
                               std::vector<std::string> value) {
    if (example_idx >= num_examples_) {
      LOG(WARNING) << "example_idx should be less than the number of examples";
      return;
    }
    examples_->SetCategoricalSet(example_idx, {feature_id}, value,
                                 engine_->features());
  }

  // Runs the model on the previously set features.
  std::vector<float> Predict() {
    if (num_examples_ == -1) {
      LOG(WARNING) << "predict called before setting any examples";
      return {};
    }
    std::vector<float> predictions;
    engine_->Predict(*examples_, num_examples_, &predictions);
    return predictions;
  }

 private:
  // Engine i.e. compiled version of the model.
  std::unique_ptr<ydf::serving::FastEngine> engine_;

  // Set of allocated examples.
  std::unique_ptr<ydf::serving::AbstractExampleSet> examples_;

  // Number of examples allocated in "examples_".
  int num_examples_ = -1;
};

// Loads a model from a path.
std::shared_ptr<Model> LoadModel(std::string path) {
  // Load model.
  std::unique_ptr<ydf::model::AbstractModel> ydf_model;
  auto status = ydf::model::LoadModel(path, &ydf_model);
  if (!status.ok()) {
    LOG(WARNING) << status.message();
    return {};
  }

  // Compile model.
  auto engine_or = ydf_model->BuildFastEngine();
  if (!engine_or.ok()) {
    LOG(WARNING) << engine_or.status().message();
    return {};
  }

  return std::make_shared<Model>(std::move(engine_or).value());
}

// Expose some of the class/functions to JS.
//
// Keep this list in sync with the corresponding @typedef in wrapper.js.
EMSCRIPTEN_BINDINGS(my_module) {
  emscripten::class_<Model>("InternalModel")
      .smart_ptr_constructor("InternalModel", &std::make_shared<Model>)
      .function("predict", &Model::Predict)
      .function("newBatchOfExamples", &Model::NewBatchOfExamples)
      .function("setNumerical", &Model::SetNumerical)
      .function("setCategoricalInt", &Model::SetCategoricalInt)
      .function("setCategoricalString", &Model::SetCategoricalString)
      .function("setCategoricalSetString", &Model::SetCategoricalSetString)
      .function("getInputFeatures", &Model::GetInputFeatures);

  emscripten::value_object<InputFeature>("InputFeature")
      .field("name", &InputFeature::name)
      .field("type", &InputFeature::type)
      .field("internalIdx", &InputFeature::internal_idx);

  emscripten::function("InternalLoadModel", &LoadModel);

  emscripten::register_vector<InputFeature>("vector<InputFeature>");
  emscripten::register_vector<float>("vector<float>");
  emscripten::register_vector<std::string>("vector<string>");
}
