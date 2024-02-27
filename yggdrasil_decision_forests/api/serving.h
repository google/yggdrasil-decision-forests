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

// Inference of a model in C++.
//
// Documentation: https://ydf.readthedocs.io/en/latest/tutorial/cpp/
//
#ifndef YGGDRASIL_DECISION_FORESTS_API_SERVING_H_
#define YGGDRASIL_DECISION_FORESTS_API_SERVING_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"

namespace yggdrasil_decision_forests::serving_api {

// A machine learning model.
using AbstractModel = ::yggdrasil_decision_forests::model::AbstractModel;

// A machine learning optimized for inference.
using FastEngine = ::yggdrasil_decision_forests::serving::FastEngine;

// A feature in a FastEngine.
using FeaturesDefinition =
    ::yggdrasil_decision_forests::serving::FeaturesDefinition;

// Options to load a model with "LoadModel".
using ModelIOOptions = ::yggdrasil_decision_forests::model::ModelIOOptions;

// Types of features in a FastEngine.
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

// Copies a VerticalDataset (a dataset format suited for training) into an
// Example Set (a dataset format suited for inference). When possible, populate
// directly the ExampleSet.
constexpr auto CopyVerticalDatasetToAbstractExampleSet =
    ::yggdrasil_decision_forests::serving::
        CopyVerticalDatasetToAbstractExampleSet;

// Loads a model from a directory.
inline absl::StatusOr<std::unique_ptr<AbstractModel>> LoadModel(
    const absl::string_view directory, ModelIOOptions io_options = {}) {
  return ::yggdrasil_decision_forests::model::LoadModel(directory, io_options);
}

// Saves a model to a directory.
inline absl::Status SaveModel(absl::string_view directory,
                              const AbstractModel& mdl,
                              ModelIOOptions io_options = {}) {
  return ::yggdrasil_decision_forests::model::SaveModel(directory, mdl,
                                                        io_options);
}

// Saved a model to a string.
constexpr auto SerializeModel =
    ::yggdrasil_decision_forests::model::SerializeModel;

// Loads a model from a string.
constexpr auto DeserializeModel =
    ::yggdrasil_decision_forests::model::DeserializeModel;

}  // namespace yggdrasil_decision_forests::serving_api

#endif  // YGGDRASIL_DECISION_FORESTS_API_SERVING_H_
