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

#include <cstddef>
#include <string>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
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

namespace {
struct JavaFeatureDef {
  std::string type;  // Type to encode a feature using type / enum class.
};

absl::StatusOr<JavaFeatureDef> GenJavaFeatureDef(
    const dataset::proto::Column& col,
    const JavaInternalOptions& internal_options) {
  // TODO: Add support for default values.
  // TODO: For integer numericals, use the min/max to possibly reduce the
  // required precision.
  switch (col.type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (internal_options.numerical_feature_is_float) {
        DCHECK_EQ(internal_options.feature_value_bytes, 4);
        return JavaFeatureDef{.type = "float"};
      } else {
        return JavaFeatureDef{
            .type = JavaInteger(internal_options.feature_value_bytes)};
      }
    } break;
    case dataset::proto::ColumnType::BOOLEAN:
    case dataset::proto::ColumnType::CATEGORICAL: {
      return JavaFeatureDef{
          .type = absl::StrCat("Feature", StringToCamelCase(col.name()))};
    } break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Non supported feature type: ",
                       dataset::proto::ColumnType_Name(col.type())));
  }
}

// Generates the struct for a single instance (i.e., an example without a
// label).
absl::StatusOr<std::string> GenInstanceStruct(
    const model::AbstractModel& model, const proto::Options& options,
    const JavaInternalOptions& internal_options,
    const internal::ModelStatistics& stats) {
  std::string content;

  std::string numerical_type;
  if (internal_options.numerical_feature_is_float) {
    DCHECK_EQ(internal_options.feature_value_bytes, 4);
    numerical_type = "float";
  } else {
    numerical_type = JavaInteger(internal_options.feature_value_bytes);
  }

  size_t num_input_features = model.input_features().size();

  bool has_numerical = false;
  bool has_categorical_or_boolean = false;
  for (const auto& input_feature : model.input_features()) {
    const auto& col = model.data_spec().columns(input_feature);
    switch (col.type()) {
      case dataset::proto::ColumnType::NUMERICAL:
        has_numerical = true;
        break;
      case dataset::proto::ColumnType::CATEGORICAL:
      case dataset::proto::ColumnType::BOOLEAN:
        has_categorical_or_boolean = true;
        break;
      default:
        break;
    }
  }

  // Start
  absl::SubstituteAndAppend(&content, R"(
private static final int NUM_FEATURES = $0;
private static final int NUM_TREES = $1;

public static class Instance {

)",
                            num_input_features,  // $0
                            stats.num_trees      // $1
  );

  if (has_numerical) {
    absl::SubstituteAndAppend(&content,
                              "  private final $0[] numericalFeatureValues;\n",
                              numerical_type);
  }
  if (has_categorical_or_boolean) {
    absl::StrAppend(
        &content, "  private final int[] categoricalOrBooleanFeatureValues;\n");
  }

  // Constructor
  absl::StrAppend(&content, "\n  public Instance(");
  std::vector<std::string> constructor_args;
  for (const auto& input_feature : model.input_features()) {
    const auto& col = model.data_spec().columns(input_feature);
    ASSIGN_OR_RETURN(const auto feature_def,
                     internal::GenJavaFeatureDef(col, internal_options));
    constructor_args.push_back(absl::StrCat(
        feature_def.type, " ", StringToLowerCamelCase(col.name())));
  }
  absl::StrAppend(&content, absl::StrJoin(constructor_args, ", "));
  absl::StrAppend(&content, ") {\n");

  if (has_numerical) {
    std::vector<std::string> numerical_values_names_or_sentinel;
    std::vector<std::string> numerical_values_nan_replacements;
    for (int feature_idx = 0; feature_idx < model.input_features().size();
         ++feature_idx) {
      const auto& col =
          model.data_spec().columns(model.input_features()[feature_idx]);
      if (col.type() == dataset::proto::ColumnType::NUMERICAL) {
        const auto variable_name = StringToLowerCamelCase(col.name());
        numerical_values_names_or_sentinel.push_back(variable_name);
        if (numerical_type == "float") {
          numerical_values_nan_replacements.push_back(
              absl::Substitute("    $0 = Float.isNaN($0) ? $1f : $0;",
                               variable_name, col.numerical().mean()));
        }
      } else {
        numerical_values_names_or_sentinel.push_back("0");  // Sentinel value
      }
    }
    if (!numerical_values_nan_replacements.empty()) {
      absl::StrAppend(
          &content,
          "    // NaN substitutions are required to handle missing values "
          "correctly.\n    // If no missing numerical values are used, "
          "this block can be removed.\n",
          absl::StrJoin(numerical_values_nan_replacements, "\n"), "\n\n");
    }
    absl::SubstituteAndAppend(
        &content, "    this.numericalFeatureValues = new $0[] {$1};\n",
        numerical_type,
        absl::StrJoin(numerical_values_names_or_sentinel, ", "));
  }

  if (has_categorical_or_boolean) {
    std::vector<std::string> categorical_values;
    for (int feature_idx = 0; feature_idx < model.input_features().size();
         ++feature_idx) {
      const auto& col =
          model.data_spec().columns(model.input_features()[feature_idx]);
      if (col.type() == dataset::proto::ColumnType::CATEGORICAL ||
          col.type() == dataset::proto::ColumnType::BOOLEAN) {
        categorical_values.push_back(
            absl::StrCat(StringToLowerCamelCase(col.name()), ".ordinal()"));
      } else {
        categorical_values.push_back("-1");  // Sentinel value
      }
    }
    absl::SubstituteAndAppend(
        &content,
        "    this.categoricalOrBooleanFeatureValues = new int[] {$0};\n",
        absl::StrJoin(categorical_values, ", "));
  }
  absl::StrAppend(&content, "  }\n");

  // End
  absl::StrAppend(&content, "}\n");
  return content;
}

// Generates the enum constants for the categorical string input features and
// the label.
absl::StatusOr<std::string> GenCategoricalStringDictionaries(
    const model::AbstractModel& model, const proto::Options& options,
    const JavaInternalOptions& internal_options) {
  std::string content;

  for (const auto& dict : internal_options.categorical_dicts) {
    absl::SubstituteAndAppend(&content, R"(
public enum $0$1 {
)",
                              dict.second.is_label ? "" : "Feature",  // $0
                              dict.second.sanitized_name              // $1
    );
    // Create the enum values
    for (int item_idx = 0; item_idx < dict.second.sanitized_items.size();
         item_idx++) {
      absl::SubstituteAndAppend(&content, "  $0,\n",
                                dict.second.sanitized_items[item_idx]);
    }
    absl::StrAppend(&content, R"(};
)");

    if (options.categorical_from_string()) {
      return absl::UnimplementedError(
          "The categorical_from_string option is not yet implemented for "
          "Java.");
    }
  }
  return content;
}

}  // namespace

absl::StatusOr<absl::node_hash_map<Filename, Content>> EmbedModelJava(
    const model::AbstractModel& model, const proto::Options& options) {
  // Make sure the model is a decision forest.
  const auto* df_interface =
      dynamic_cast<const model::DecisionForestInterface*>(&model);
  if (!df_interface) {
    return absl::InvalidArgumentError(
        "The model is not a decision forest model.");
  }
  const bool has_global_imputation =
      df_interface->CheckStructure({.global_imputation_is_higher = true});
  if (!has_global_imputation) {
    return absl::InvalidArgumentError(
        "This model has been trained without global imputation. Only models "
        "trained with global imputation are supported for the Java export. "
        "Contact the YDF team for more options.");
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

  // Categorical dictionary
  ASSIGN_OR_RETURN(
      const auto categorical_dict,
      GenCategoricalStringDictionaries(model, options, internal_options));
  absl::StrAppend(&code, categorical_dict);

  // Instance struct.
  ASSIGN_OR_RETURN(const auto instance_struct,
                   GenInstanceStruct(model, options, internal_options, stats));
  absl::StrAppend(&code, instance_struct);

  // TODO: Add remaining logic.

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
