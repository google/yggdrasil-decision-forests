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

#include "yggdrasil_decision_forests/serving/embed/java/java_embed.h"

#include <string>

#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"
#include "yggdrasil_decision_forests/serving/embed/utils.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

absl::StatusOr<absl::node_hash_map<Filename, Content>> EmbedModelJava(
    const model::AbstractModel& model, const proto::Options& options) {
  // Make sure the model is a decision forest.
  const auto* df_interface =
      dynamic_cast<const model::DecisionForestInterface*>(&model);
  if (!df_interface) {
    return absl::InvalidArgumentError(
        "The model is not a decision forest model.");
  }

  // Check names.
  RETURN_IF_ERROR(CheckModelName(options.name(), proto::Options::kJava));
  for (const auto& column_idx : model.input_features()) {
    RETURN_IF_ERROR(
        CheckFeatureName(model.data_spec().columns(column_idx).name()));
  }

  ASSIGN_OR_RETURN(const internal::ModelStatistics stats,
                   internal::ComputeStatistics(model, *df_interface));

  ASSIGN_OR_RETURN(
      const JavaInternalOptions internal_options,
      ComputeJavaInternalOptions(model, *df_interface, stats, options));

  // Generate the code.
  absl::node_hash_map<Filename, Content> result;
  std::string code = "";
  result[absl::StrCat(options.name(), ".java")] = code;
  return result;
}

absl::StatusOr<JavaInternalOptions> ComputeJavaInternalOptions(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface,
    const internal::ModelStatistics& stats, const proto::Options& options) {
  JavaInternalOptions internal_options;
  RETURN_IF_ERROR(ComputeBaseInternalOptionsFeature(stats, model, options,
                                                    &internal_options));
  RETURN_IF_ERROR(
      ComputeJavaInternalOptionsOutput(stats, options, &internal_options));
  RETURN_IF_ERROR(ComputeBaseInternalOptionsCategoricalDictionaries(
      model, stats, options, &internal_options));
  return internal_options;
}

absl::Status ComputeJavaInternalOptionsOutput(
    const internal::ModelStatistics& stats, const proto::Options& options,
    JavaInternalOptions* out) {
  if (stats.has_conditions
          [model::decision_tree::proto::Condition::kContainsBitmapCondition] ||
      stats.has_conditions
          [model::decision_tree::proto::Condition::kContainsCondition]) {
    out->imports.bitset = true;
  }

  switch (stats.task) {
    case model::proto::Task::CLASSIFICATION: {
      switch (options.classification_output()) {
        case proto::ClassificationOutput::CLASS:
          out->output_type = "Label";
          break;
        case proto::ClassificationOutput::SCORE:
          // Nothing to do.
          break;
        case proto::ClassificationOutput::PROBABILITY:
          if (stats.is_binary_classification()) {
            out->output_type = "float";
          }
          break;
      }
    } break;
    case model::proto::Task::REGRESSION:
      if (options.integerize_output()) {
        out->output_type = SignedInteger(options.accumulator_precision_bytes());
      } else {
        out->output_type = "float";
      }
      break;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported task: ", model::proto::Task_Name(stats.task)));
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::serving::embed::internal
