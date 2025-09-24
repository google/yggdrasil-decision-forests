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

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_JAVA_JAVA_EMBED_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_JAVA_JAVA_EMBED_H_

#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

// Java libraries to import in the generated source code.
struct Imports {
  // java.util.BitSet import.
  bool bitset = false;
};

// Specific options for the generation of the model.
// The internal options contains all the precise internal decision aspect of the
// model compilation e.g. how many bits to use to encode numerical features. The
// internal options are computed using the user provided options (simply called
// "options" in the code) and the model.
struct JavaInternalOptions : BaseInternalOptions {
  // Java imports.
  Imports imports;
};

absl::StatusOr<absl::node_hash_map<Filename, Content>> EmbedModelJava(
    const model::AbstractModel& model, const proto::Options& options);

// Computes the internal options of the model.
absl::StatusOr<JavaInternalOptions> ComputeJavaInternalOptions(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface,
    const internal::ModelStatistics& stats, const proto::Options& options);

// Populates the output parts of the internal option.
absl::Status ComputeJavaInternalOptionsOutput(
    const internal::ModelStatistics& stats, const proto::Options& options,
    JavaInternalOptions* out);

absl::StatusOr<SpecializedConversion> SpecializedConversionRandomForestJava(
    const model::random_forest::RandomForestModel& model,
    const internal::ModelStatistics& stats,
    const JavaInternalOptions& internal_options, const proto::Options& options);

absl::StatusOr<SpecializedConversion>
SpecializedConversionGradientBoostedTreesJava(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& model,
    const internal::ModelStatistics& stats,
    const JavaInternalOptions& internal_options, const proto::Options& options);
}  // namespace yggdrasil_decision_forests::serving::embed::internal
#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_JAVA_JAVA_EMBED_H_
