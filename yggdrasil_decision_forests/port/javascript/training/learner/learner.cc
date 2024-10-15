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

#include "yggdrasil_decision_forests/port/javascript/training/learner/learner.h"

#include "absl/strings/str_cat.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#include <emscripten/emscripten.h>
#endif  // __EMSCRIPTEN__

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/port/javascript/training/dataset/dataset.h"
#include "yggdrasil_decision_forests/port/javascript/training/model/model.h"
#include "yggdrasil_decision_forests/port/javascript/training/util/status_casters.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests::port::javascript {

void Learner::Init(std::string learner, std::string label, std::string task) {
  if (initilized_) {
    CheckOrThrowError(absl::InternalError("Learner not initialized"));
  }
  model::proto::TrainingConfig train_config;
  train_config.set_learner(learner);
  train_config.set_label(label);
  model::proto::Task enum_task;
  bool parse_success = model::proto::Task_Parse(task, &enum_task);
  if (!parse_success) {
    CheckOrThrowError(
        absl::InvalidArgumentError(absl::StrCat("Invalid task name", task)));
  }
  train_config.set_task(enum_task);
  model::proto::DeploymentConfig deployment_config;
  deployment_config.set_num_threads(0);
  CheckOrThrowError(
      model::GetLearner(train_config, &learner_ptr_, deployment_config));
  initilized_ = true;
}

std::shared_ptr<Model> Learner::TrainFromPath(std::string path) {
  if (!initilized_) {
    CheckOrThrowError(absl::InternalError("Learner not initialized"));
  }
  const auto data_spec = dataset::CreateDataSpec(path);
  if (!data_spec.ok()) {
    CheckOrThrowError(data_spec.status());
  }
  auto ydf_model_or = learner_ptr_->TrainWithStatus(path, data_spec.value());
  if (!ydf_model_or.ok()) {
    CheckOrThrowError(ydf_model_or.status());
  }
  auto ydf_model = std::move(ydf_model_or.value());

  // Compile model.
  auto engine_or = ydf_model->BuildFastEngine();
  if (!engine_or.ok()) {
    CheckOrThrowError(engine_or.status());
  }

  // Extract the label classes, if any.
  std::vector<std::string> label_classes;
  if (ydf_model->task() == model::proto::Task::CLASSIFICATION) {
    auto label_classes_or = ExtractLabelClasses(*ydf_model);
    if (!label_classes_or.ok()) {
      CheckOrThrowError(label_classes_or.status());
    }
    label_classes = std::move(label_classes_or.value());
  }

  return std::make_shared<Model>(std::move(engine_or).value(),
                                 std::move(ydf_model),
                                 std::move(label_classes));
}

std::shared_ptr<Model> Learner::TrainFromDataset(const Dataset& dataset) {
  if (!initilized_) {
    CheckOrThrowError(absl::InternalError("Learner not initialized"));
  }
  auto ydf_model_or = learner_ptr_->TrainWithStatus(dataset.dataset());
  if (!ydf_model_or.ok()) {
    CheckOrThrowError(ydf_model_or.status());
  }
  auto ydf_model = std::move(ydf_model_or.value());

  // Compile model.
  auto engine_or = ydf_model->BuildFastEngine();
  if (!engine_or.ok()) {
    CheckOrThrowError(engine_or.status());
  }

  // Extract the label classes, if any.
  std::vector<std::string> label_classes;
  if (ydf_model->task() == model::proto::Task::CLASSIFICATION) {
    auto label_classes_or = ExtractLabelClasses(*ydf_model);
    if (!label_classes_or.ok()) {
      CheckOrThrowError(label_classes_or.status());
    }
    label_classes = std::move(label_classes_or.value());
  }

  return std::make_shared<Model>(std::move(engine_or).value(),
                                 std::move(ydf_model),
                                 std::move(label_classes));
}

void init_learner() {
#ifdef __EMSCRIPTEN__
  emscripten::class_<Learner>("InternalLearner")
      .smart_ptr_constructor("InternalLearner", &std::make_shared<Learner>)
      .function("init", &Learner::Init)
      .function("trainFromPath", &Learner::TrainFromPath)
      .function("trainFromDataset", &Learner::TrainFromDataset);
#endif  // __EMSCRIPTEN__
}

}  // namespace yggdrasil_decision_forests::port::javascript
