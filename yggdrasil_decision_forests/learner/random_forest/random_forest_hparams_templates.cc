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

// This files defines the pre-configured hyper-parameter templates.

#include "yggdrasil_decision_forests/learner/decision_tree/generic_parameters.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace random_forest {

std::vector<model::proto::PredefinedHyperParameterTemplate>
RandomForestLearner::PredefinedHyperParameters() const {
  std::vector<model::proto::PredefinedHyperParameterTemplate> param_sets;
  {
    model::proto::PredefinedHyperParameterTemplate config;
    config.set_name("better_default");
    config.set_version(1);
    config.set_description(
        R"(A configuration that is generally better than the default parameters without being more expensive.)");
    {
      auto field = config.mutable_parameters()->add_fields();
      field->set_name(kHParamWinnerTakeAll);
      field->mutable_value()->set_categorical("true");
    }
    param_sets.push_back(std::move(config));
  }
  {
    model::proto::PredefinedHyperParameterTemplate config;
    config.set_name("benchmark_rank1");
    config.set_version(1);
    config.set_description(
        R"(Top ranking hyper-parameters on our benchmark slightly modified to run in reasonable time.)");
    {
      auto field = config.mutable_parameters()->add_fields();
      field->set_name(kHParamWinnerTakeAll);
      field->mutable_value()->set_categorical("true");
    }
    {
      auto field = config.mutable_parameters()->add_fields();
      field->set_name(decision_tree::kHParamCategoricalAlgorithm);
      field->mutable_value()->set_categorical(
          decision_tree::kCategoricalAlgorithmRandom);
    }
    {
      auto field = config.mutable_parameters()->add_fields();
      field->set_name(decision_tree::kHParamSplitAxis);
      field->mutable_value()->set_categorical(
          decision_tree::kHParamSplitAxisSparseOblique);
    }
    {
      auto field = config.mutable_parameters()->add_fields();
      field->set_name(
          decision_tree::kHParamSplitAxisSparseObliqueNormalization);
      field->mutable_value()->set_categorical(
          decision_tree::kHParamSplitAxisSparseObliqueNormalizationMinMax);
    }
    {
      auto field = config.mutable_parameters()->add_fields();
      field->set_name(
          decision_tree::kHParamSplitAxisSparseObliqueNumProjectionsExponent);
      field->mutable_value()->set_real(1.f);
    }
    param_sets.push_back(std::move(config));
  }
  return param_sets;
}

}  // namespace random_forest
}  // namespace model
}  // namespace yggdrasil_decision_forests
