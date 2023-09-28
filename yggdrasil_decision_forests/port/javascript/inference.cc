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

// Special prefix value that indicates that the prefix is not specified. Loading
// a model without specifying the prefix enables the automatic prefix search
// logic.
constexpr char kNoPrefix[] = "__NO_PREFIX__";

// Definition of an input feature.
// This struct is the JS visible version of the ydf::serving::FeatureDef class.
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

// Output predictions when using the TF-DF signature.
struct TFDFOutputPrediction {
  std::vector<std::vector<float>> dense_predictions;
  std::vector<std::string> dense_col_representation;
};

// Returns the list of input features as listed in the "input_features" proto
// field of the model. This list of input features is used to generate the
// TensorFlow Decision Forests inference signature. This set of features always
// contain all the features required by the inference engine.
std::vector<InputFeature> BuildProtoInputFeatures(
    const ydf::model::AbstractModel* model) {
  std::vector<InputFeature> input_features;
  for (const int column_idx : model->input_features()) {
    const auto& col_spec = model->data_spec().columns(column_idx);
    input_features.push_back(
        {col_spec.name(), ydf::dataset::proto::ColumnType_Name(col_spec.type()),
         -1, column_idx});
  }
  return input_features;
}

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
  Model(std::unique_ptr<ydf::serving::FastEngine>&& engine,
        const ydf::model::AbstractModel* model,
        const bool created_tfdf_signature)
      : engine_(std::move(engine)) {
    DCHECK(engine_);

    if (created_tfdf_signature) {
      proto_input_features_ = BuildProtoInputFeatures(model);
      ComputeDenseColRepresentation(model);
      decompact_probability_ =
          model->task() == ydf::model::proto::Task::CLASSIFICATION &&
          engine_->NumPredictionDimension() == 1 &&
          model->classification_outputs_probabilities();
    }
  }

  // Similar logic as "ComputeDenseColRepresentation" in TF-DF. See
  // "tensorflow/ops/inference/kernel.cc".
  void ComputeDenseColRepresentation(const ydf::model::AbstractModel* model) {
    dense_col_representation_.clear();
    if (model->task() == ydf::model::proto::Task::CLASSIFICATION) {
      const auto& label_spec =
          model->data_spec().columns(model->label_col_idx());
      // Note: We don't report the "OOV" class value.
      const int num_classes =
          label_spec.categorical().number_of_unique_values() - 1;

      if (num_classes == 2 && !model->classification_outputs_probabilities()) {
        // Output the logit of the positive class.
        dense_col_representation_.assign(1, "logit");
      } else {
        // Output the logit or probabilities.
        dense_col_representation_.resize(num_classes);
        for (int class_idx = 0; class_idx < num_classes; class_idx++) {
          dense_col_representation_[class_idx] =
              ydf::dataset::CategoricalIdxToRepresentation(label_spec,
                                                           class_idx + 1);
        }
      }
    } else {
      dense_col_representation_.resize(1);
    }
  }

  // Lists the input features of the model.
  std::vector<InputFeature> GetInputFeatures() {
    std::vector<InputFeature> input_features;
    for (const auto& feature : engine_->features().input_features()) {
      input_features.push_back(
          {feature.name, ydf::dataset::proto::ColumnType_Name(feature.type),
           feature.internal_idx, feature.spec_idx});
    }
    return input_features;
  }

  // Lists the input features of the model as stored in the model proto.
  std::vector<InputFeature> GetProtoInputFeatures() {
    return proto_input_features_;
  }

  // Creates a new batch of examples. Should be called before setting the
  // example values.
  void NewBatchOfExamples(int num_examples) {
    if (num_examples < 0) {
      YDF_LOG(WARNING) << "num_examples should be positive";
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
      YDF_LOG(WARNING)
          << "example_idx should be less than the number of examples";
      return;
    }

    examples_->SetNumerical(example_idx, {feature_id}, value,
                            engine_->features());
  }

  // Sets the value of a boolean feature.
  void SetBoolean(int example_idx, int feature_id, bool value) {
    if (example_idx >= num_examples_) {
      YDF_LOG(WARNING)
          << "example_idx should be less than the number of examples";
      return;
    }
    examples_->SetBoolean(example_idx, {feature_id}, value,
                          engine_->features());
  }

  // Sets the value of a categorical feature.
  void SetCategoricalInt(int example_idx, int feature_id, int value) {
    if (example_idx >= num_examples_) {
      YDF_LOG(WARNING)
          << "example_idx should be less than the number of examples";
      return;
    }

    examples_->SetCategorical(example_idx, {feature_id}, value,
                              engine_->features());
  }

  // Sets the value of a categorical feature.
  void SetCategoricalString(int example_idx, int feature_id,
                            std::string value) {
    if (example_idx >= num_examples_) {
      YDF_LOG(WARNING)
          << "example_idx should be less than the number of examples";
      return;
    }
    examples_->SetCategorical(example_idx, {feature_id}, value,
                              engine_->features());
  }

  // Sets the value of a categorical set feature.
  void SetCategoricalSetString(int example_idx, int feature_id,
                               std::vector<std::string> value) {
    if (example_idx >= num_examples_) {
      YDF_LOG(WARNING)
          << "example_idx should be less than the number of examples";
      return;
    }
    examples_->SetCategoricalSet(example_idx, {feature_id}, value,
                                 engine_->features());
  }

  // Sets the value of a categorical set feature.
  void SetCategoricalSetInt(int example_idx, int feature_id,
                            std::vector<int> value) {
    if (example_idx >= num_examples_) {
      YDF_LOG(WARNING)
          << "example_idx should be less than the number of examples";
      return;
    }
    examples_->SetCategoricalSet(example_idx, {feature_id}, value,
                                 engine_->features());
  }

  // Runs the model on the previously set features.
  std::vector<float> Predict() {
    if (num_examples_ == -1) {
      YDF_LOG(WARNING) << "predict called before setting any examples";
      return {};
    }
    std::vector<float> predictions;
    engine_->Predict(*examples_, num_examples_, &predictions);
    return predictions;
  }

  // Runs the model on the previously set features.
  TFDFOutputPrediction PredictTFDFSignature(const int dense_output_dim) {
    TFDFOutputPrediction output;
    output.dense_col_representation = dense_col_representation_;

    if (num_examples_ == -1) {
      YDF_LOG(WARNING) << "predict called before setting any examples";
      return output;
    }

    // Generate the predictions.
    std::vector<float> predictions;
    engine_->Predict(*examples_, num_examples_, &predictions);

    // Reformat the predictions in the TF-DF format.
    output.dense_predictions.reserve(num_examples_);
    const auto engine_output_dim = engine_->NumPredictionDimension();

    // Same as "RunInference" in "tensorflow/ops/inference/kernel.cc".
    // Export the predictions.
    if (decompact_probability_) {
      if (engine_output_dim != 1) {
        YDF_LOG(FATAL) << "Wrong NumPredictionDimension";
      }
      for (int example_idx = 0; example_idx < num_examples_; example_idx++) {
        const float proba =
            ydf::utils::clamp(predictions[example_idx], 0.f, 1.f);
        output.dense_predictions.push_back(
            std::vector<float>{1.f - proba, proba});
      }

    } else {
      for (int example_idx = 0; example_idx < num_examples_; example_idx++) {
        std::vector<float> example_prediction(engine_output_dim);
        for (int class_idx = 0; class_idx < engine_output_dim; class_idx++) {
          const float value =
              predictions[example_idx * engine_output_dim + class_idx];
          example_prediction.push_back(value);
        }
        output.dense_predictions.push_back(std::move(example_prediction));
      }
    }

    return output;
  }

 private:
  // Engine i.e. compiled version of the model.
  std::unique_ptr<ydf::serving::FastEngine> engine_;

  // Set of allocated examples.
  std::unique_ptr<ydf::serving::AbstractExampleSet> examples_;

  // List of input features as listed in the "input_features" proto
  // field of the model. Used for TF-DF models.
  std::vector<InputFeature> proto_input_features_;

  // Representation of the output predictions for TF-DF models.
  std::vector<std::string> dense_col_representation_;

  // Should the output of the YDF model be "decompated" in the TF-DF signature.
  bool decompact_probability_;

  // Number of examples allocated in "examples_".
  int num_examples_ = -1;
};

// Loads a model from a path.
std::shared_ptr<Model> LoadModel(std::string path,
                                 const bool created_tfdf_signature,
                                 const std::string file_prefix) {
  // Load model.
  ydf::model::ModelIOOptions options;
  if (file_prefix != kNoPrefix) {
    options.file_prefix = file_prefix;
  }
  std::unique_ptr<ydf::model::AbstractModel> ydf_model;
  auto status = ydf::model::LoadModel(path, &ydf_model, options);
  if (!status.ok()) {
    YDF_LOG(WARNING) << status.message();
    return {};
  }

  // Compile model.
  auto engine_or = ydf_model->BuildFastEngine();
  if (!engine_or.ok()) {
    YDF_LOG(WARNING) << engine_or.status().message();
    return {};
  }

  return std::make_shared<Model>(std::move(engine_or).value(), ydf_model.get(),
                                 created_tfdf_signature);
}

std::vector<std::string> CreateVectorString(size_t reserved) {
  std::vector<std::string> v;
  v.reserve(reserved);
  return v;
}

std::vector<int> CreateVectorInt(size_t reserved) {
  std::vector<int> v;
  v.reserve(reserved);
  return v;
}

// Expose some of the class/functions to JS.
//
// Keep this list in sync with the corresponding @typedef in wrapper.js.
EMSCRIPTEN_BINDINGS(my_module) {
  emscripten::class_<Model>("InternalModel")
      .smart_ptr_constructor("InternalModel", &std::make_shared<Model>)
      .function("predict", &Model::Predict)
      .function("predictTFDFSignature", &Model::PredictTFDFSignature)
      .function("newBatchOfExamples", &Model::NewBatchOfExamples)
      .function("setNumerical", &Model::SetNumerical)
      .function("setBoolean", &Model::SetBoolean)
      .function("setCategoricalInt", &Model::SetCategoricalInt)
      .function("setCategoricalString", &Model::SetCategoricalString)
      .function("setCategoricalSetString", &Model::SetCategoricalSetString)
      .function("setCategoricalSetInt", &Model::SetCategoricalSetInt)
      .function("getInputFeatures", &Model::GetInputFeatures)
      .function("getProtoInputFeatures", &Model::GetProtoInputFeatures);

  emscripten::value_object<InputFeature>("InputFeature")
      .field("name", &InputFeature::name)
      .field("type", &InputFeature::type)
      .field("internalIdx", &InputFeature::internal_idx)
      .field("specIdx", &InputFeature::spec_idx);

  emscripten::value_object<TFDFOutputPrediction>("TFDFOutputPrediction")
      .field("densePredictions", &TFDFOutputPrediction::dense_predictions)
      .field("denseColRepresentation",
             &TFDFOutputPrediction::dense_col_representation);

  emscripten::function("InternalLoadModel", &LoadModel);

  emscripten::register_vector<InputFeature>("vector<InputFeature>");
  emscripten::register_vector<float>("vector<float>");
  emscripten::register_vector<int>("vector<int>");
  emscripten::register_vector<std::vector<float>>("vector<vector<float>>");
  emscripten::register_vector<std::string>("vector<string>");

  emscripten::function("CreateVectorString", &CreateVectorString);
  emscripten::function("CreateVectorInt", &CreateVectorInt);
}
