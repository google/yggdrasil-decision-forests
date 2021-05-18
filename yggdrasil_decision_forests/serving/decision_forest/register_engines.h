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

// Inference engine names for decision forest models.
//
#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_REGISTER_ENGINE_DECISION_FOREST_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_REGISTER_ENGINE_DECISION_FOREST_H_

#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"

namespace yggdrasil_decision_forests {
namespace serving {

namespace gradient_boosted_trees {
constexpr char kGeneric[] = "GradientBoostedTreesGeneric";
constexpr char kQuickScorerExtended[] =
    "GradientBoostedTreesQuickScorerExtended";
constexpr char kOptPred[] = "GradientBoostedTreesOptPred";
}  // namespace gradient_boosted_trees

namespace random_forest {
constexpr char kGeneric[] = "RandomForestGeneric";
constexpr char kOptPred[] = "RandomForestOptPred";
}  // namespace random_forest

}  // namespace serving
}  // namespace yggdrasil_decision_forests
#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_REGISTER_ENGINE_DECISION_FOREST_H_
