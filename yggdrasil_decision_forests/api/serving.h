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

#ifndef YGGDRASIL_DECISION_FORESTS_API_SERVING_H_
#define YGGDRASIL_DECISION_FORESTS_API_SERVING_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"

namespace yggdrasil_decision_forests::serving_api {

using AbstractModel = ::yggdrasil_decision_forests::model::AbstractModel;
using FastEngine = ::yggdrasil_decision_forests::serving::FastEngine;
using FeaturesDefinition =
    ::yggdrasil_decision_forests::serving::FeaturesDefinition;

using NumericalFeatureId = ::yggdrasil_decision_forests::serving::
    FeaturesDefinitionNumericalOrCategoricalFlat::NumericalFeatureId;
using CategoricalFeatureId = ::yggdrasil_decision_forests::serving::
    FeaturesDefinitionNumericalOrCategoricalFlat::CategoricalFeatureId;
using CategoricalSetFeatureId = ::yggdrasil_decision_forests::serving::
    FeaturesDefinitionNumericalOrCategoricalFlat::CategoricalSetFeatureId;
using BooleanFeatureId = ::yggdrasil_decision_forests::serving::
    FeaturesDefinitionNumericalOrCategoricalFlat::BooleanFeatureId;
using MultiDimNumericalFeatureId = ::yggdrasil_decision_forests::serving::
    FeaturesDefinitionNumericalOrCategoricalFlat::MultiDimNumericalFeatureId;

// Loads a model in memory. This model can then be compiled to be run
// efficiently.
inline absl::StatusOr<std::unique_ptr<AbstractModel>> LoadModel(
    absl::string_view directory) {
  std::unique_ptr<AbstractModel> model;
  RETURN_IF_ERROR(model::LoadModel(directory, &model));
  return model;
}

}  // namespace yggdrasil_decision_forests::serving_api

#endif  // YGGDRASIL_DECISION_FORESTS_API_SERVING_H_
