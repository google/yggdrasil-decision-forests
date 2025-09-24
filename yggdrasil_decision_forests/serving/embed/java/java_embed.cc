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
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
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

  // Implementation specific specialization.
  internal::SpecializedConversion specialized_conversion;
  {
    const auto* model_gbt = dynamic_cast<
        const model::gradient_boosted_trees::GradientBoostedTreesModel*>(
        &model);
    const auto* model_rf =
        dynamic_cast<const model::random_forest::RandomForestModel*>(&model);
    if (model_gbt) {
      ASSIGN_OR_RETURN(specialized_conversion,
                       internal::SpecializedConversionGradientBoostedTreesJava(
                           *model_gbt, stats, internal_options, options));
    } else if (model_rf) {
      ASSIGN_OR_RETURN(specialized_conversion,
                       internal::SpecializedConversionRandomForestJava(
                           *model_rf, stats, internal_options, options));
    } else {
      return absl::InvalidArgumentError("The model type is not supported.");
    }
    RETURN_IF_ERROR(specialized_conversion.Validate());
  }

  // Generate the code.
  absl::node_hash_map<Filename, Content> result;
  std::string code = "";

  // Open define and namespace.
  absl::SubstituteAndAppend(&code, R"(package $0;

)",
                            options.java().package_name());

  if (internal_options.imports.bitset) {
    absl::StrAppend(&code, "import java.util.BitSet;\n");
  }

  absl::SubstituteAndAppend(&code, R"(
public final class $0 {
)",
                            options.name());

  // TODO: Add logic

  // Close define and namespace.
  absl::SubstituteAndAppend(&code, R"(

  private $0() {} // Prevent instantiation
}
)",
                            options.name());

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

absl::StatusOr<SpecializedConversion> SpecializedConversionRandomForestJava(
    const model::random_forest::RandomForestModel& model,
    const internal::ModelStatistics& stats,
    const JavaInternalOptions& internal_options,
    const proto::Options& options) {
  return absl::UnimplementedError(
      "Java export is not implemented for Random Forests");
}

absl::StatusOr<internal::SpecializedConversion>
SpecializedConversionGradientBoostedTreesJava(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& model,
    const internal::ModelStatistics& stats,
    const JavaInternalOptions& internal_options,
    const proto::Options& options) {
  SpecializedConversion spec;
  switch (stats.task) {
    case model::proto::Task::CLASSIFICATION: {
      // Leaf setter
      if (stats.is_binary_classification()) {
        spec.accumulator_type = "float";

        spec.set_node_ifelse_fn =
            [](const model::decision_tree::proto::Node& node, const int depth,
               const int tree_idx,
               absl::string_view prefix) -> absl::StatusOr<std::string> {
          const float node_value = node.regressor().top_value();
          return absl::StrCat(prefix, "accumulator += ", node_value, ";\n");
        };

        spec.leaf_value_fn =
            [&](const model::decision_tree::proto::Node& node) -> LeafValue {
          const float node_value = node.regressor().top_value();
          return std::vector<float>{node_value};
        };

        spec.routing_node = R"(
    accumulator += NODE_VAL[currentNodeIndex] ? 1 : 0;
)";

      } else {
        spec.accumulator_type =
            absl::StrCat("float[", stats.num_classification_classes, "]");

        spec.set_node_ifelse_fn =
            [&](const model::decision_tree::proto::Node& node, const int depth,
                const int tree_idx,
                absl::string_view prefix) -> absl::StatusOr<std::string> {
          const float node_value = node.regressor().top_value();
          const int output_dim_idx =
              tree_idx % stats.num_classification_classes;
          return absl::StrCat(prefix, "accumulator[", output_dim_idx,
                              "] += ", node_value, ";\n");
        };

        spec.leaf_value_fn =
            [&](const model::decision_tree::proto::Node& node) -> LeafValue {
          const float node_value = node.regressor().top_value();
          return std::vector<float>{node_value};
        };

        spec.routing_node = absl::Substitute(R"(
    accumulator[tree_idx % $0] += NODE_VAL[currentNodeIndex] ? 1 : 0;
)",
                                             stats.num_classification_classes);
      }
      // Return accumulator
      switch (options.classification_output()) {
        case proto::ClassificationOutput::CLASS:
          if (stats.is_binary_classification()) {
            spec.return_prediction =
                "  return Label.values()[accumulator >= 0 ? 1 : 0];\n";
          } else {
            spec.return_prediction =
                ("  int maxIndex = 0;\n"
                 "  for (int i = 1; i < accumulator.length; i++) {\n"
                 "    if (accumulator[i] > accumulator[maxIndex]) {\n"
                 "      maxIndex = i;\n"
                 "    }\n"
                 "  }\n"
                 "  return Label.values()[maxIndex];\n");
          }
          break;
        case proto::ClassificationOutput::SCORE:
          spec.return_prediction = "  return accumulator;\n";
          break;
        case proto::ClassificationOutput::PROBABILITY:
          if (stats.is_binary_classification()) {
            spec.return_prediction = R"(  // Sigmoid
  return 1.0f / (1.0f + (float)Math.exp(-accumulator));
)";
          } else {
            spec.return_prediction =
                absl::Substitute(R"(  // Softmax
  float[] probas = new float[$0];
  float maxLogit = Float.NEGATIVE_INFINITY;
  for (float val : accumulator) {
    if (val > maxLogit) {
      maxLogit = val;
    }
  }

  float sumExps = 0.0f;
  for (int i = 0; i < $0; i++) {
    probas[i] = (float) Math.exp(accumulator[i] - maxLogit);
    sumExps += probas[i];
  }

  for (int i = 0; i < $0; i++) {
    probas[i] /= sumExps;
  }
  return probas;
)",
                                 stats.num_classification_classes);
          }
          break;
      }
    } break;

    case model::proto::Task::REGRESSION:
      spec.accumulator_type = "float";
      spec.return_prediction = "  return accumulator;\n";

      spec.set_node_ifelse_fn =
          [](const model::decision_tree::proto::Node& node, const int depth,
             const int tree_idx,
             absl::string_view prefix) -> absl::StatusOr<std::string> {
        const float node_value = node.regressor().top_value();
        return absl::StrCat(prefix, "accumulator += ", node_value, ";\n");
      };

      spec.leaf_value_fn =
          [&](const model::decision_tree::proto::Node& node) -> LeafValue {
        const float node_value = node.regressor().top_value();
        return std::vector<float>{node_value};
      };

      spec.routing_node = R"(
    accumulator += NODE_VAL[currentNodeIndex] ? 1 : 0;
)";
      break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported task: ", model::proto::Task_Name(stats.task)));
  }

  // TODO: Integer optimization of leaf values.
  spec.leaf_value_spec = {.dtype = proto::DType::FLOAT32, .dims = 1};

  spec.accumulator_initial_value =
      absl::StrJoin(model.initial_predictions(), ", ");
  return spec;
}

}  // namespace yggdrasil_decision_forests::serving::embed::internal
