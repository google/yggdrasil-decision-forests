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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
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
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"
#include "yggdrasil_decision_forests/serving/embed/java/model_data_bank.h"
#include "yggdrasil_decision_forests/serving/embed/utils.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

namespace {

std::string GetResourceName(const proto::Options& options) {
  return absl::StrCat(options.name(), "Data.bin");
}

absl::Status AddTreeToBank(const dataset::proto::DataSpecification& dataspec,
                           const SpecializedConversion& specialized_conversion,
                           const proto::Options& options,
                           const JavaInternalOptions& internal_options,
                           const model::decision_tree::NodeWithChildren& node,
                           ModelDataBank* bank) {
  if (node.IsLeaf()) {
    const auto leaf_value = specialized_conversion.leaf_value_fn(node.node());
    if (specialized_conversion.leaf_value_spec.dims > 1) {
      return absl::UnimplementedError(
          "Multi-output nodes (e.g for multi-class Random Forests) are not yet "
          "supported");
    }
    Int64OrFloat val;
    if (std::holds_alternative<std::vector<bool>>(leaf_value)) {
      const auto& v = std::get<std::vector<bool>>(leaf_value);
      if (v.empty()) return absl::InvalidArgumentError("Empty leaf value");
      val = static_cast<int64_t>(v.front());
    } else if (std::holds_alternative<std::vector<int32_t>>(leaf_value)) {
      const auto& v = std::get<std::vector<int32_t>>(leaf_value);
      if (v.empty()) return absl::InvalidArgumentError("Empty leaf value");
      val = static_cast<int64_t>(v.front());
    } else if (std::holds_alternative<std::vector<float>>(leaf_value)) {
      const auto& v = std::get<std::vector<float>>(leaf_value);
      if (v.empty()) return absl::InvalidArgumentError("Empty leaf value");
      val = v.front();
    } else {
      return absl::InvalidArgumentError("Unsupported leaf type.");
    }
    RETURN_IF_ERROR(bank->AddNode({.pos = 0, .val = val}));
    return absl::OkStatus();
  }

  // This is the node offset between a parent node and the positive node. Note
  // that the offset to the negative node is 1 and does not need to be encoded.
  const auto delta_pos_node = node.neg_child()->NumNodes();

  // Get the dense feature index.
  const int attribute_idx = node.node().condition().attribute();
  const int feature_idx =
      internal_options.column_idx_to_feature_idx.at(attribute_idx);

  // Create a contains condition node.
  const auto categorical_contains_condition =
      [&](const std::vector<bool>& bitmap) -> absl::Status {
    // TODO: For bitmap requiring less bits than a bitmap bank index,
    // store the mask in the node directly (instead of using the bank).
    // TODO: Search if the bitmap bank already contains the current
    // bitmap. If so, use the existing bitmap segment instead.
    RETURN_IF_ERROR(
        bank->AddNode({.pos = delta_pos_node,
                       .feat = feature_idx,
                       .cat = static_cast<int64_t>(bank->categorical.size())}));
    bank->categorical.insert(bank->categorical.end(), bitmap.begin(),
                             bitmap.end());
    bank->num_conditions[static_cast<int>(
        RoutingConditionType::CONTAINS_CONDITION_BUFFER_BITMAP)]++;
    return absl::OkStatus();
  };

  // Encode all the possible condition nodes.
  switch (node.node().condition().condition().type_case()) {
    case model::decision_tree::proto::Condition::TypeCase::kHigherCondition: {
      // Condition of the type "a >= threshold".
      const auto& typed_condition =
          node.node().condition().condition().higher_condition();
      Int64OrFloat threshold = typed_condition.threshold();
      if (!internal_options.numerical_feature_is_float) {
        threshold =
            static_cast<int64_t>(std::ceil(typed_condition.threshold()));
      }
      RETURN_IF_ERROR(bank->AddNode(
          {.pos = delta_pos_node, .feat = feature_idx, .thr = threshold}));
      bank->num_conditions[static_cast<int>(
          RoutingConditionType::HIGHER_CONDITION)]++;
    } break;

    case model::decision_tree::proto::Condition::TypeCase::kContainsCondition: {
      // Condition of the type "a in Mask" where mask is represented by a list
      // of integers.
      const auto& typed_condition =
          node.node().condition().condition().contains_condition();
      std::vector<bool> bitmap(dataspec.columns(attribute_idx)
                                   .categorical()
                                   .number_of_unique_values(),
                               false);
      for (const auto item_idx : typed_condition.elements()) {
        bitmap[item_idx] = true;
      }
      RETURN_IF_ERROR(categorical_contains_condition(bitmap));
    } break;

    case model::decision_tree::proto::Condition::TypeCase::
        kContainsBitmapCondition: {
      // Condition of the type "a in Mask" where mask is represented by a
      // bitmap.
      const auto& typed_condition =
          node.node().condition().condition().contains_bitmap_condition();
      std::vector<bool> bitmap(dataspec.columns(attribute_idx)
                                   .categorical()
                                   .number_of_unique_values(),
                               false);
      for (size_t item_idx = 0; item_idx < bitmap.size(); item_idx++) {
        if (utils::bitmap::GetValueBit(typed_condition.elements_bitmap(),
                                       item_idx)) {
          bitmap[item_idx] = true;
        }
      }
      RETURN_IF_ERROR(categorical_contains_condition(bitmap));
    } break;

    case model::decision_tree::proto::Condition::TypeCase::kObliqueCondition: {
      // Sparse oblique condition.
      const auto& typed_condition =
          node.node().condition().condition().oblique_condition();
      // TODO: Compute oblique projections as integers instead of floats if
      // both the feature values and oblique weights are integers.

      const size_t num_projections = typed_condition.weights_size();
      std::vector<float> oblique_weights;
      oblique_weights.reserve(num_projections + 1);
      std::vector<size_t> oblique_features;
      oblique_features.reserve(num_projections + 1);

      // Magic values
      oblique_features.push_back(num_projections);
      oblique_weights.push_back(typed_condition.threshold());

      // Oblique weights + indexes
      for (size_t proj_idx = 0; proj_idx < num_projections; proj_idx++) {
        oblique_weights.push_back(typed_condition.weights(proj_idx));
        const int sub_feature_idx =
            internal_options.column_idx_to_feature_idx.at(
                typed_condition.attributes(proj_idx));
        oblique_features.push_back(sub_feature_idx);
      }

      ASSIGN_OR_RETURN(const int64_t oblique_features_size,
                       bank->GetObliqueFeaturesSize());

      RETURN_IF_ERROR(
          bank->AddNode({.pos = delta_pos_node,
                         .feat = ObliqueFeatureIndex(options, internal_options),
                         .obl = oblique_features_size,
                         .oblique_weights = oblique_weights,
                         .oblique_features = oblique_features}));
      bank->num_conditions[static_cast<int>(
          RoutingConditionType::OBLIQUE_CONDITION)]++;
    } break;

    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Non supported condition type: ",
                       model::decision_tree::ConditionTypeToString(
                           node.node().condition().condition().type_case())));
  }
  RETURN_IF_ERROR(AddTreeToBank(dataspec, specialized_conversion, options,
                                internal_options, *node.neg_child(), bank));
  RETURN_IF_ERROR(AddTreeToBank(dataspec, specialized_conversion, options,
                                internal_options, *node.pos_child(), bank));

  return absl::OkStatus();
}

// Returns the Java type for a given feature column.
absl::StatusOr<std::string> GetJavaFeatureType(
    const dataset::proto::Column& col,
    const JavaInternalOptions& internal_options) {
  // TODO: Add support for default values.
  // TODO: For integer numericals, use the min/max to possibly reduce the
  // required precision.
  switch (col.type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (internal_options.numerical_feature_is_float) {
        DCHECK_EQ(internal_options.feature_value_bytes, 4);
        return "float";
      } else {
        return JavaInteger(internal_options.feature_value_bytes);
      }
    } break;
    case dataset::proto::ColumnType::BOOLEAN:
    case dataset::proto::ColumnType::CATEGORICAL: {
      return absl::StrCat("Feature", StringToCamelCase(col.name()));
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
    ASSIGN_OR_RETURN(const std::string feature_type,
                     internal::GetJavaFeatureType(col, internal_options));
    constructor_args.push_back(
        absl::StrCat(feature_type, " ", StringToLowerCamelCase(col.name())));
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

  absl::StrAppend(&code,
                  R"(import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
)");
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

  // Model data
  // Data stored in the nodes.
  ModelDataBank bank(internal_options, stats, specialized_conversion);

  if (options.algorithm() == proto::Algorithm::ROUTING) {
    RETURN_IF_ERROR(internal::GenRoutingModelDataJava(
        model, model.data_spec(), *df_interface, stats, specialized_conversion,
        options, internal_options, &code, &bank));
  } else {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported algorithm for tha Java export: ",
                     proto::Algorithm_Enum_Name(options.algorithm())));
  }

  // TODO: Add remaining logic.

  // Close define and namespace.
  absl::SubstituteAndAppend(&code, R"(

  private $0() {} // Prevent instantiation
}
)",
                            options.name());

  result[absl::StrCat(options.name(), ".java")] = code;
  ASSIGN_OR_RETURN(result[GetResourceName(options)],
                   bank.SerializeData(internal_options));
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
    accumulator += nodeVal[currentNodeIndex];
)";

      } else {
        spec.accumulator_type = absl::StrCat("float[]");

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
    accumulator[treeIdx % $0] += nodeVal[currentNodeIndex];
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
    accumulator += nodeVal[currentNodeIndex];
)";
      break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported task: ", model::proto::Task_Name(stats.task)));
  }

  // TODO: Integer optimization of leaf values.
  spec.leaf_value_spec = {.dtype = proto::DType::FLOAT32, .dims = 1};

  std::vector<std::string> initial_predictions_str;
  for (const float val : model.initial_predictions()) {
    initial_predictions_str.push_back(absl::StrCat(val, "f"));
  }
  spec.accumulator_initial_value = absl::StrJoin(initial_predictions_str, ", ");
  if (model.initial_predictions().size() > 1) {
    spec.accumulator_initial_value =
        absl::StrCat("{", spec.accumulator_initial_value, "}");
  }
  return spec;
}

absl::Status GenRoutingModelDataJava(
    const model::AbstractModel& model,
    const dataset::proto::DataSpecification& dataspec,
    const model::DecisionForestInterface& df_interface,
    const ModelStatistics& stats,
    const SpecializedConversion& specialized_conversion,
    const proto::Options& options, const JavaInternalOptions& internal_options,
    std::string* content, ModelDataBank* bank) {
  std::vector<int64_t> root_deltas;
  root_deltas.reserve(df_interface.num_trees());
  for (int tree_idx = 0; tree_idx < df_interface.num_trees(); tree_idx++) {
    const auto& tree = df_interface.decision_trees()[tree_idx];
    RETURN_IF_ERROR(AddTreeToBank(dataspec, specialized_conversion, options,
                                  internal_options, tree->root(), bank));
    RETURN_IF_ERROR(bank->AddRootDelta(tree->root().NumNodes()));
  }

  if (stats.has_multiple_condition_types) {
    ASSIGN_OR_RETURN(const auto condition_types,
                     GenRoutingModelDataConditionType(model, stats));
    RETURN_IF_ERROR(bank->AddConditionTypes(condition_types));
  }

  RETURN_IF_ERROR(bank->FinalizeJavaTypes());

  // Append the node bank Java code.
  ASSIGN_OR_RETURN(const std::string bank_code,
                   bank->GenerateJavaCode(internal_options, options.name(),
                                          GetResourceName(options)));
  absl::StrAppend(content, bank_code);

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::serving::embed::internal
