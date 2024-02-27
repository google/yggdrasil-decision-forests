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

// Training of a model in C++.
//
// Example:
// https://github.com/google/yggdrasil-decision-forests/blob/main/examples/standalone/beginner.cc
#ifndef YGGDRASIL_DECISION_FORESTS_API_TRAINING_H_
#define YGGDRASIL_DECISION_FORESTS_API_TRAINING_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example_builder.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"

namespace yggdrasil_decision_forests::training_api {

// A dataset for training stored in memory.
using VerticalDataset = ::yggdrasil_decision_forests::dataset::VerticalDataset;

// Configuration of a learning algorithm.
using TrainingConfig =
    ::yggdrasil_decision_forests::model::proto::TrainingConfig;

// Configuration of computing resources for training.
using DeploymentConfig =
    ::yggdrasil_decision_forests::model::proto::DeploymentConfig;

// Task solved by a model e.g. classification.
using Task = ::yggdrasil_decision_forests::model::proto::Task;

// A learning algorithm.
using AbstractLearner = ::yggdrasil_decision_forests::model::AbstractLearner;

// A utility class to build examples that are then added to a VerticalDataset.
using ExampleProtoBuilder =
    ::yggdrasil_decision_forests::dataset::ExampleProtoBuilder;

// Definition of the columns in a dataset.
using DataSpecification =
    ::yggdrasil_decision_forests::dataset::proto::DataSpecification;

// Utility to add columns in a DataSpecification.
constexpr auto AddNumericalColumn =
    ::yggdrasil_decision_forests::dataset::AddNumericalColumn;
constexpr auto AddCategoricalColumn =
    ::yggdrasil_decision_forests::dataset::AddCategoricalColumn;
constexpr auto AddBooleanColumn =
    ::yggdrasil_decision_forests::dataset::AddBooleanColumn;
constexpr auto AddCategoricalSetColumn =
    ::yggdrasil_decision_forests::dataset::AddCategoricalSetColumn;

// Instantiates a learner i.e. a learning algorithm.
constexpr auto GetLearner =
    static_cast<absl::StatusOr<std::unique_ptr<AbstractLearner>> (*)(
        const TrainingConfig&, const DeploymentConfig&, const std::string&)>(
        ::yggdrasil_decision_forests::model::GetLearner);

// Accesses the hyper-parameters of a gradient boosted decision trees.
inline model::gradient_boosted_trees::proto::GradientBoostedTreesTrainingConfig*
GetGradientBoostedTreesTrainingConfig(TrainingConfig* training_config) {
  return training_config->MutableExtension(
      model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
}

// Accesses the hyper-parameters of a random forest.
inline model::random_forest::proto::RandomForestTrainingConfig*
GetRandomForestTrainingConfig(TrainingConfig* training_config) {
  return training_config->MutableExtension(
      model::random_forest::proto::random_forest_config);
}

}  // namespace yggdrasil_decision_forests::training_api

#endif  // YGGDRASIL_DECISION_FORESTS_API_TRAINING_H_
