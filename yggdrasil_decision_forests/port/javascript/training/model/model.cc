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

#include "yggdrasil_decision_forests/port/javascript/training/model/model.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#include <emscripten/emscripten.h>
#endif  // __EMSCRIPTEN__

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/port/javascript/training/util/status_casters.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::port::javascript {

namespace {

// Loads a model from a path.
[[maybe_unused]] std::shared_ptr<Model> LoadModel(std::string path) {
  // Load model.
  std::unique_ptr<model::AbstractModel> ydf_model;
  auto status = model::LoadModel(path, &ydf_model);
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

  // Extract the label classes, if any.
  std::vector<std::string> label_classes;
  if (ydf_model->task() == model::proto::Task::CLASSIFICATION) {
    auto label_classes_or = ExtractLabelClasses(*ydf_model);
    if (!label_classes_or.ok()) {
      LOG(WARNING) << label_classes_or.status().message();
      return {};
    }
    label_classes = std::move(label_classes_or.value());
  }

  return std::make_shared<Model>(std::move(engine_or).value(),
                                 std::move(ydf_model),
                                 std::move(label_classes));
}

}  // namespace

absl::StatusOr<std::vector<std::string>> ExtractLabelClasses(
    const model::AbstractModel& model) {
  const auto& col_spec = model.data_spec().columns(model.label_col_idx());
  STATUS_CHECK_EQ(col_spec.type(), dataset::proto::ColumnType::CATEGORICAL);
  std::vector<std::string> label_classes(
      col_spec.categorical().number_of_unique_values() - 1);
  if (col_spec.categorical().is_already_integerized()) {
    for (int i = 1; i < col_spec.categorical().number_of_unique_values(); i++) {
      label_classes[i - 1] = absl::StrCat(i);
    }
  } else {
    for (const auto& item : col_spec.categorical().items()) {
      if (item.second.index() > 0) {
        label_classes[item.second.index() - 1] = item.first;
      }
    }
  }
  return label_classes;
}

Model::Model(std::unique_ptr<serving::FastEngine>&& engine,
             std::unique_ptr<model::AbstractModel> model,
             std::vector<std::string> label_classes)
    : model_(std::move(model)), engine_(std::move(engine)) {
  DCHECK(engine_);

  label_classes_ = std::move(label_classes);
}

// Lists the input features of the model.
std::vector<InputFeature> Model::GetInputFeatures() {
  std::vector<InputFeature> input_features;
  for (const auto& feature : engine_->features().input_features()) {
    input_features.push_back({feature.name,
                              dataset::proto::ColumnType_Name(feature.type),
                              feature.internal_idx, feature.spec_idx});
  }
  return input_features;
}

// Creates a new batch of examples. Should be called before setting the
// example values.
void Model::NewBatchOfExamples(int num_examples) {
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
void Model::SetNumerical(int example_idx, int feature_id, float value) {
  if (example_idx >= num_examples_) {
    LOG(WARNING) << "example_idx should be less than the number of examples";
    return;
  }

  examples_->SetNumerical(example_idx, {feature_id}, value,
                          engine_->features());
}

// Sets the value of a boolean feature.
void Model::SetBoolean(int example_idx, int feature_id, bool value) {
  if (example_idx >= num_examples_) {
    LOG(WARNING) << "example_idx should be less than the number of examples";
    return;
  }
  examples_->SetBoolean(example_idx, {feature_id}, value, engine_->features());
}

// Sets the value of a categorical feature.
void Model::SetCategoricalInt(int example_idx, int feature_id, int value) {
  if (example_idx >= num_examples_) {
    LOG(WARNING) << "example_idx should be less than the number of examples";
    return;
  }

  examples_->SetCategorical(example_idx, {feature_id}, value,
                            engine_->features());
}

// Sets the value of a categorical feature.
void Model::SetCategoricalString(int example_idx, int feature_id,
                                 std::string value) {
  if (example_idx >= num_examples_) {
    LOG(WARNING) << "example_idx should be less than the number of examples";
    return;
  }
  examples_->SetCategorical(example_idx, {feature_id}, value,
                            engine_->features());
}

// Sets the value of a categorical set feature.
void Model::SetCategoricalSetString(int example_idx, int feature_id,
                                    std::vector<std::string> value) {
  if (example_idx >= num_examples_) {
    LOG(WARNING) << "example_idx should be less than the number of examples";
    return;
  }
  examples_->SetCategoricalSet(example_idx, {feature_id}, value,
                               engine_->features());
}

// Sets the value of a categorical set feature.
void Model::SetCategoricalSetInt(int example_idx, int feature_id,
                                 std::vector<int> value) {
  if (example_idx >= num_examples_) {
    LOG(WARNING) << "example_idx should be less than the number of examples";
    return;
  }
  examples_->SetCategoricalSet(example_idx, {feature_id}, value,
                               engine_->features());
}

// Runs the model on the previously set features.
std::vector<float> Model::Predict() {
  if (num_examples_ == -1) {
    LOG(WARNING) << "predict called before setting any examples";
    return {};
  }
  std::vector<float> predictions;
  engine_->Predict(*examples_, num_examples_, &predictions);
  return predictions;
}

// Runs the model on the previously set features.
std::vector<float> Model::PredictFromPath(std::string path) {
  dataset::VerticalDataset dataset;
  QCHECK_OK(LoadVerticalDataset(path, model_->data_spec(), &dataset));
  auto examples = engine_->AllocateExamples(dataset.nrow());
  std::vector<float> predictions;
  QCHECK_OK(serving::CopyVerticalDatasetToAbstractExampleSet(
      dataset, 0, dataset.nrow(), engine_->features(), examples.get()));
  engine_->Predict(*examples, dataset.nrow(), &predictions);
  return predictions;
}

std::string Model::Describe() {
  return model_->DescriptionAndStatistics(/*full_definition=*/false);
}

void Model::Save(const std::string& path) {
  auto save_status = model::SaveModel(path, model_.get());
  CheckOrThrowError(save_status);
}

void init_model() {
#ifdef __EMSCRIPTEN__
  emscripten::value_object<InputFeature>("InputFeature")
      .field("name", &InputFeature::name)
      .field("type", &InputFeature::type)
      .field("internalIdx", &InputFeature::internal_idx)
      .field("specIdx", &InputFeature::spec_idx);

  emscripten::class_<Model>("InternalModel")
      .smart_ptr_constructor("InternalModel", &std::make_shared<Model>)
      .function("predict", &Model::Predict)
      .function("predictFromPath", &Model::PredictFromPath)
      .function("describe", &Model::Describe)
      .function("save", &Model::Save)
      .function("newBatchOfExamples", &Model::NewBatchOfExamples)
      .function("setNumerical", &Model::SetNumerical)
      .function("setBoolean", &Model::SetBoolean)
      .function("setCategoricalInt", &Model::SetCategoricalInt)
      .function("setCategoricalString", &Model::SetCategoricalString)
      .function("setCategoricalSetString", &Model::SetCategoricalSetString)
      .function("setCategoricalSetInt", &Model::SetCategoricalSetInt)
      .function("getInputFeatures", &Model::GetInputFeatures)
      .function("getLabelClasses", &Model::GetLabelClasses);

  emscripten::function("InternalLoadModel", &LoadModel);

  emscripten::register_vector<InputFeature>("vectorInputFeature");
#endif  // __EMSCRIPTEN__
}
}  // namespace yggdrasil_decision_forests::port::javascript
