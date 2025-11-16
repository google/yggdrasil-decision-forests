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

#include "yggdrasil_decision_forests/serving/embed/cc_embed.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/serving/embed/utils.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

namespace {
// Generates the struct for a single instance (i.e., an example without a
// label).
absl::StatusOr<std::string> GenInstanceStruct(
    const model::AbstractModel& model, const proto::Options& options,
    const CCInternalOptions& internal_options,
    const internal::ModelStatistics& stats,
    const std::vector<FeatureDef>& feature_defs) {
  std::string content;

  std::string numerical_type;
  if (internal_options.numerical_feature_is_float) {
    DCHECK_EQ(internal_options.feature_value_bytes, 4);
    numerical_type = "float";
  } else {
    numerical_type = SignedInteger(internal_options.feature_value_bytes);
  }

  // Start
  absl::SubstituteAndAppend(&content, R"(
constexpr const int kNumFeatures = $0;
constexpr const int kNumTrees = $1;

struct Instance {
  typedef $2 Numerical;

)",
                            model.input_features().size(),  // $0
                            stats.num_trees,                // $1
                            numerical_type                  // $2
  );

  for (int i = 0; i < model.input_features().size(); ++i) {
    const auto& feature_def = feature_defs[i];
    absl::StrAppend(&content, "  ", feature_def.type, " ",
                    feature_def.variable_name);

    if (!feature_def.default_value.has_value()) {
      absl::StrAppend(&content, ";\n");
    } else {
      absl::StrAppend(&content, " = ", *feature_def.default_value, ";\n");
    }
  }

  // End
  absl::StrAppend(&content, R"(};
)");

  return content;
}

// Generates the enum constants for the categorical string input features and
// the label.
absl::StatusOr<std::string> GenCategoricalStringDictionaries(
    const model::AbstractModel& model, const proto::Options& options,
    const CCInternalOptions& internal_options) {
  std::string content;

  for (const auto& dict : internal_options.categorical_dicts) {
    absl::SubstituteAndAppend(
        &content, R"(
enum class $0$1 : $2 {
)",
        dict.second.is_label ? "" : "Feature",                 // $0
        dict.second.sanitized_name,                            // $1
        UnsignedInteger(internal_options.feature_value_bytes)  // $2
    );
    // Create the enum values
    for (int item_idx = 0; item_idx < dict.second.sanitized_items.size();
         item_idx++) {
      absl::SubstituteAndAppend(&content, "  k$0 = $1,\n",
                                dict.second.sanitized_items[item_idx],
                                item_idx);
    }
    absl::StrAppend(&content, R"(};
)");

    if (options.categorical_from_string() && !dict.second.is_label) {
      // Create a function to create an enum class value from a string.
      absl::SubstituteAndAppend(&content, R"(
Feature$0 Feature$0FromString(const std::string_view name) {
  using F = Feature$0;
  static const std::unordered_map<std::string_view, Feature$0>
      kFeature$0Map = {
)",
                                dict.second.sanitized_name  // $0
      );
      // Note: We skip the OOV item with index 0.
      for (int item_idx = 1; item_idx < dict.second.sanitized_items.size();
           item_idx++) {
        absl::SubstituteAndAppend(
            &content, "          {$0, F::k$1},\n",
            QuoteString(dict.second.items[item_idx]),  // $0
            dict.second.sanitized_items[item_idx]      // $1
        );
      }
      absl::SubstituteAndAppend(&content, R"(      };
  auto it = kFeature$0Map.find(name);
  if (it == kFeature$0Map.end()) {
    return F::kOutOfVocabulary;
  }
  return it->second;
}
)",
                                dict.second.sanitized_name  // $0
      );
    }
  }
  return content;
}

}  // namespace

absl::StatusOr<absl::node_hash_map<Filename, Content>> EmbedModelCC(
    const model::AbstractModel& model, const proto::Options& options) {
  // Make sure the model is a decision forest.
  const auto* df_interface =
      dynamic_cast<const model::DecisionForestInterface*>(&model);
  if (!df_interface) {
    return absl::InvalidArgumentError(
        "The model is not a decision forest model.");
  }

  // Check names.
  RETURN_IF_ERROR(CheckModelName(options.name(), proto::Options::kCc));
  for (const auto& column_idx : model.input_features()) {
    RETURN_IF_ERROR(
        CheckFeatureName(model.data_spec().columns(column_idx).name()));
  }

  ASSIGN_OR_RETURN(const internal::ModelStatistics stats,
                   internal::ComputeStatistics(model, *df_interface));

  ASSIGN_OR_RETURN(
      const CCInternalOptions internal_options,
      internal::ComputeInternalOptions(model, *df_interface, stats, options));

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
                       internal::SpecializedConversionGradientBoostedTreesCC(
                           *model_gbt, stats, internal_options, options));
    } else if (model_rf) {
      ASSIGN_OR_RETURN(specialized_conversion,
                       internal::SpecializedConversionRandomForestCC(
                           *model_rf, stats, internal_options, options));
    } else {
      return absl::InvalidArgumentError("The model type is not supported.");
    }
    RETURN_IF_ERROR(specialized_conversion.Validate(options));
  }

  // Generate the code.
  absl::node_hash_map<Filename, Content> result;
  std::string header;

  // Open define and namespace.
  absl::SubstituteAndAppend(&header, R"(#ifndef YDF_MODEL_$0_H_
#define YDF_MODEL_$0_H_

#include <stdint.h>
#include <cstring>
)",
                            StringToConstantSymbol(options.name()));

  if (internal_options.includes.array) {
    absl::StrAppend(&header, "#include <array>\n");
  }
  if (internal_options.includes.algorithm) {
    absl::StrAppend(&header, "#include <algorithm>\n");
  }
  if (internal_options.includes.cmath) {
    absl::StrAppend(&header, "#include <cmath>\n");
  }
  if (options.categorical_from_string()) {
    absl::StrAppend(&header, "#include <unordered_map>\n");
  }

  // TODO: Only include if necessary.
  absl::StrAppend(&header, "#include <bitset>\n");
  absl::StrAppend(&header, "#include <cassert>\n");

  absl::SubstituteAndAppend(&header, R"(
namespace $0 {
)",
                            StringToVariableSymbol(options.name()));

  // Categorical dictionary
  ASSIGN_OR_RETURN(
      const auto categorical_dict,
      GenCategoricalStringDictionaries(model, options, internal_options));
  absl::StrAppend(&header, categorical_dict);

  // Generate FeatureDefs for all input features.
  std::vector<FeatureDef> feature_defs;
  feature_defs.reserve(model.input_features().size());
  for (const auto input_feature : model.input_features()) {
    const auto& col = model.data_spec().columns(input_feature);
    ASSIGN_OR_RETURN(const auto feature_def,
                     internal::GenFeatureDef(col, internal_options));
    feature_defs.push_back(feature_def);
  }

  // Instance struct.
  ASSIGN_OR_RETURN(
      const auto instance_struct,
      GenInstanceStruct(model, options, internal_options, stats, feature_defs));
  absl::StrAppend(&header, instance_struct);

  // Model data
  internal::ValueBank routing_bank;
  if (options.algorithm() == proto::Algorithm::ROUTING) {
    RETURN_IF_ERROR(internal::GenRoutingModelData(
        model, model.data_spec(), *df_interface, stats, specialized_conversion,
        options, internal_options, &header, &routing_bank));
  }

  // Predict method
  std::string predict_body_with_nan_replacement;
  std::string predict_body_without_nan_replacement;

  RETURN_IF_ERROR(internal::CorePredict(
      model.data_spec(), *df_interface, specialized_conversion, stats,
      internal_options, options, feature_defs, routing_bank,
      &predict_body_with_nan_replacement,
      &predict_body_without_nan_replacement));
  STATUS_CHECK(!predict_body_without_nan_replacement.empty());
  bool requires_nan_replacement = !predict_body_with_nan_replacement.empty();

  std::string predict_output_type;
  if (options.classification_output() != proto::ClassificationOutput::SCORE) {
    // The prediction type is defined by the task, and independent of the model
    // implementation..
    predict_output_type = internal_options.output_type;
  } else {
    // The prediction type is determined by the specific decision forest model
    // implementation.
    predict_output_type = specialized_conversion.accumulator_type;
  }
  STATUS_CHECK(!predict_output_type.empty());

  std::string missing_numerical_warning = "";

  if (requires_nan_replacement) {
    absl::SubstituteAndAppend(&header, R"(
// Predict on an instance with no NaN values. This function may give incorrect
// predictions if `instance` contains missing values (e.g. NaN).
//
// Prefer using `Predict()` unless `instance` is guaranteed to not contain
// missing values.
//
// This function is called by `Predict()`.
inline $0 PredictWithoutMissingValues(const Instance& instance) {
$1
}

// Prediction function.
//
// Replaces NaN numerical features by the global imputation of the model, and
// then calls `PredictWithoutMissingValues()`.
inline $0 Predict(Instance instance) {
$2
  return PredictWithoutMissingValues(instance);
}
)",
                              predict_output_type,                   // $0
                              predict_body_without_nan_replacement,  // $1
                              predict_body_with_nan_replacement      // $2
    );
  } else {
    absl::SubstituteAndAppend(&header, R"(
inline $0 Predict(const Instance& instance) {
$1
}
)",
                              predict_output_type,                  // $0
                              predict_body_without_nan_replacement  // $1
    );
  }

  // Close define and namespace.
  absl::SubstituteAndAppend(&header, R"(
}  // namespace $0
#endif
)",
                            StringToVariableSymbol(options.name()));

  result[absl::StrCat(options.name(), ".h")] = header;
  return result;
}

absl::Status CorePredict(const dataset::proto::DataSpecification& dataspec,
                         const model::DecisionForestInterface& df_interface,
                         const SpecializedConversion& specialized_conversion,
                         const ModelStatistics& stats,
                         const CCInternalOptions& internal_options,
                         const proto::Options& options,
                         const std::vector<FeatureDef>& feature_defs,
                         const ValueBank& routing_bank,
                         std::string* content_with_nan_replacement,
                         std::string* content_without_nan_replacement) {
  // Add NAN checks.
  bool needs_na_replacement = false;
  for (const auto& feature_def : feature_defs) {
    if (feature_def.na_replacement.has_value()) {
      needs_na_replacement = true;
      break;
    }
  }

  if (needs_na_replacement) {
    const bool has_global_imputation =
        df_interface.CheckStructure({.global_imputation_is_higher = true});
    if (!has_global_imputation) {
      LOG(WARNING)
          << "The model is not trained with global imputation. The generated "
             "code will crash if an instance with NaN values is provided. "
             "Ensure that input instances are NaN-free and call "
             "PredictWithoutMissingValues or train the model with global "
             "imputation.";
      *content_with_nan_replacement = R"(
    // If the model has not been trained with global imputation, export to C++
    // is only supported without support for missing values. 
    //
    // Aborting to avoid incorrect predictions.
    abort();
)";
    } else {
      for (const auto& feature_def : feature_defs) {
        if (feature_def.na_replacement.has_value()) {
          // One could use instance.$0 != instance.$0 to avoid inclusion of
          // cmath, but it hurts readability and is likely not worth it.
          absl::SubstituteAndAppend(
              content_with_nan_replacement,
              "  instance.$0 = std::isnan(instance.$0) ? $1 : instance.$0;\n",
              feature_def.variable_name,   // $0
              *feature_def.na_replacement  // $1
          );
        }
      }
    }
  }

  // Accumulator
  absl::SubstituteAndAppend(content_without_nan_replacement,
                            "  $0 accumulator {$1};\n",
                            specialized_conversion.accumulator_type,
                            specialized_conversion.accumulator_initial_value);

  // Accumulate leaf values
  switch (options.algorithm()) {
    case proto::Algorithm::IF_ELSE:
      RETURN_IF_ERROR(GenerateTreeInferenceIfElse(
          dataspec, df_interface, options, internal_options,
          specialized_conversion.set_node_ifelse_fn,
          content_without_nan_replacement));
      break;
    case proto::Algorithm::ROUTING:
      RETURN_IF_ERROR(GenerateTreeInferenceRouting(
          dataspec, df_interface, options, internal_options,
          specialized_conversion, stats, routing_bank,
          content_without_nan_replacement));
      break;
    default:
      return absl::InvalidArgumentError("Non supported algorithm.");
  }

  // Accumulator to predictions.
  absl::StrAppend(content_without_nan_replacement,
                  specialized_conversion.return_prediction);
  return absl::OkStatus();
}

absl::StatusOr<SpecializedConversion> SpecializedConversionRandomForestCC(
    const model::random_forest::RandomForestModel& model,
    const internal::ModelStatistics& stats,
    const CCInternalOptions& internal_options, const proto::Options& options) {
  SpecializedConversion spec;

  switch (stats.task) {
    case model::proto::Task::CLASSIFICATION: {
      // Leaf setter
      if (stats.is_binary_classification()) {
        if (model.winner_take_all_inference()) {
          // We accumulate the count of votes for the positive class.
          spec.accumulator_type =
              UnsignedInteger(MaxUnsignedValueToNumBytes(stats.num_trees));
          spec.leaf_value_spec = {.dtype = proto::DType::BOOL, .dims = 1};

          spec.set_node_ifelse_fn =
              [](const model::decision_tree::proto::Node& node, const int depth,
                 const int tree_idx,
                 absl::string_view prefix) -> absl::StatusOr<std::string> {
            const int node_value = node.classifier().top_value();
            if (node_value == 2) {
              return absl::StrCat(prefix, "accumulator++;\n");
            } else {
              return "";
            }
          };

          spec.leaf_value_fn =
              [](const model::decision_tree::proto::Node& node) -> LeafValue {
            const int node_value = node.classifier().top_value();
            return std::vector<bool>{node_value == 2};
          };

          spec.routing_node = R"(
    accumulator += node->leaf.val;
)";
        } else {
          // We accumulate the probability vote for the positive class.
          spec.accumulator_type = "float";
          spec.leaf_value_spec = {.dtype = proto::DType::FLOAT32, .dims = 1};

          spec.set_node_ifelse_fn =
              [&](const model::decision_tree::proto::Node& node,
                  const int depth, const int tree_idx,
                  absl::string_view prefix) -> absl::StatusOr<std::string> {
            const float node_value =
                node.classifier().distribution().counts(2) /
                (node.classifier().distribution().sum() * stats.num_trees);
            if (node_value == 0) {
              return "";
            } else {
              return absl::StrCat(prefix, "accumulator += ", node_value, ";\n");
            }
          };

          spec.leaf_value_fn =
              [&](const model::decision_tree::proto::Node& node) -> LeafValue {
            const float node_value =
                node.classifier().distribution().counts(2) /
                (node.classifier().distribution().sum() * stats.num_trees);
            return std::vector<float>{node_value};
          };

          spec.routing_node = R"(
    accumulator += node->leaf.val;
)";
        }
      } else {
        if (model.winner_take_all_inference()) {
          // We accumulate the count of votes for each class.
          spec.accumulator_type = absl::StrCat(
              "std::array<",
              UnsignedInteger(MaxUnsignedValueToNumBytes(stats.num_trees)),
              ", ", stats.num_classification_classes, ">");
          spec.leaf_value_spec = {
              .dtype = UnsignedIntegerToDtype(
                  MaxUnsignedValueToNumBytes(stats.num_classification_classes)),
              .dims = 1};

          spec.set_node_ifelse_fn =
              [&](const model::decision_tree::proto::Node& node,
                  const int depth, const int tree_idx,
                  absl::string_view prefix) -> absl::StatusOr<std::string> {
            const int node_value = node.classifier().top_value() - 1;
            return absl::StrCat(prefix, "accumulator[", node_value, "]++;\n");
          };

          spec.leaf_value_fn =
              [&](const model::decision_tree::proto::Node& node) -> LeafValue {
            const int32_t node_value = node.classifier().top_value() - 1;
            return std::vector<int32_t>{node_value};
          };

          spec.routing_node = R"(
    accumulator[node->leaf.val]++;
)";
        } else {
          // We accumulate the probability for each class.
          spec.accumulator_type = absl::StrCat(
              "std::array<float, ", stats.num_classification_classes, ">");
          spec.leaf_value_spec = {.dtype = proto::DType::FLOAT32,
                                  .dims = stats.num_classification_classes};

          spec.set_node_ifelse_fn =
              [&](const model::decision_tree::proto::Node& node,
                  const int depth, const int tree_idx,
                  absl::string_view prefix) -> absl::StatusOr<std::string> {
            std::string content;
            for (int output_idx = 0;
                 output_idx < stats.num_classification_classes; output_idx++) {
              const float node_value =
                  node.classifier().distribution().counts(output_idx + 1) /
                  (node.classifier().distribution().sum() * stats.num_trees);
              if (node_value != 0) {
                absl::SubstituteAndAppend(&content,
                                          "$0accumulator[$1] += $2;\n", prefix,
                                          output_idx, node_value);
              }
            }
            return content;
          };

          spec.leaf_value_fn =
              [&](const model::decision_tree::proto::Node& node) -> LeafValue {
            std::vector<float> values;
            for (int output_idx = 0;
                 output_idx < stats.num_classification_classes; output_idx++) {
              const float node_value =
                  node.classifier().distribution().counts(output_idx + 1) /
                  (node.classifier().distribution().sum() * stats.num_trees);
              values.push_back(node_value);
            }
            return values;
          };

          spec.routing_node =
              absl::Substitute(R"(
    const size_t offset = node->leaf.val * $0;
    for(int dim=0; dim!=$0; dim++) {
      accumulator[dim] += leaf_value_bank[offset + dim];
    }
)",
                               stats.num_classification_classes);
        }
      }

      // Return accumulator
      switch (options.classification_output()) {
        case proto::ClassificationOutput::CLASS:
          if (stats.is_binary_classification()) {
            spec.return_prediction = absl::Substitute(
                "  return static_cast<Label>(accumulator > $0);",
                stats.num_trees / 2);
          } else {
            absl::StrAppend(
                &spec.return_prediction,
                "  return "
                "static_cast<Label>(std::distance(accumulator.begin(), "
                "std::max_element(accumulator.begin(), "
                "accumulator.end())));");
          }
          break;
        case proto::ClassificationOutput::SCORE:
          spec.return_prediction = "  return accumulator;";
          break;
        case proto::ClassificationOutput::PROBABILITY:
          if (model.winner_take_all_inference()) {
            if (stats.is_binary_classification()) {
              absl::SubstituteAndAppend(
                  &spec.return_prediction,
                  "return static_cast<float>(accumulator) / $0;\n",
                  stats.num_trees);
            } else {
              spec.return_prediction = absl::Substitute(
                  R"(
          std::array<float,$0> probas;
          for(int i=0;i<$0;i++){ probas[i] = static_cast<float>(accumulator[i]) / $1; }
          return probas;
        )",
                  stats.num_classification_classes, stats.num_trees);
            }
          } else {
            spec.return_prediction = "return accumulator;";
          }
          break;
      }
    } break;

    case model::proto::Task::REGRESSION:
      spec.accumulator_type = "float";
      spec.return_prediction = "  return accumulator;";
      spec.leaf_value_spec = {.dtype = proto::DType::FLOAT32, .dims = 1};

      spec.set_node_ifelse_fn =
          [&](const model::decision_tree::proto::Node& node, const int depth,
              const int tree_idx,
              absl::string_view prefix) -> absl::StatusOr<std::string> {
        const float node_value = node.regressor().top_value() / stats.num_trees;
        return absl::StrCat(prefix, "accumulator += ", node_value, ";\n");
      };

      spec.leaf_value_fn =
          [&](const model::decision_tree::proto::Node& node) -> LeafValue {
        const float node_value = node.regressor().top_value() / stats.num_trees;
        return std::vector<float>{node_value};
      };

      spec.routing_node = R"(
    accumulator += node->leaf.val;
)";
      break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Non supported task: ", model::proto::Task_Name(stats.task)));
  }

  spec.accumulator_initial_value = "0";
  return spec;
}

absl::StatusOr<internal::SpecializedConversion>
SpecializedConversionGradientBoostedTreesCC(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& model,
    const internal::ModelStatistics& stats,
    const CCInternalOptions& internal_options, const proto::Options& options) {
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
    accumulator += node->leaf.val;
)";

      } else {
        spec.accumulator_type = absl::StrCat(
            "std::array<float, ", stats.num_classification_classes, ">");

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
    accumulator[tree_idx % $0] += node->leaf.val;
)",
                                             stats.num_classification_classes);
      }
      // Return accumulator
      switch (options.classification_output()) {
        case proto::ClassificationOutput::CLASS:
          if (stats.is_binary_classification()) {
            spec.return_prediction =
                "  return static_cast<Label>(accumulator >= 0);";
          } else {
            spec.return_prediction =
                ("  return "
                 "static_cast<Label>(std::distance(accumulator.begin(), "
                 "std::max_element(accumulator.begin(), "
                 "accumulator.end())));");
          }
          break;
        case proto::ClassificationOutput::SCORE:
          spec.return_prediction = "  return accumulator;";
          break;
        case proto::ClassificationOutput::PROBABILITY:
          if (stats.is_binary_classification()) {
            spec.return_prediction = R"(  // Sigmoid
  return 1.f / (1.f + std::exp(-accumulator));
)";
          } else {
            spec.return_prediction =
                absl::Substitute(R"(  // Softmax
  std::array<float,$0> probas;
  const float max_logit = *std::max_element(accumulator.begin(), accumulator.end());
  float sum_exps = 0.f;
  for(int i=0;i<$0;i++){ probas[i] = std::exp(accumulator[i] - max_logit); sum_exps += probas[i];}
  for(int i=0;i<$0;i++){ probas[i] /= sum_exps; }
  return probas;
)",
                                 stats.num_classification_classes);
          }
          break;
      }
    } break;

    case model::proto::Task::REGRESSION:
      spec.accumulator_type = "float";
      spec.return_prediction = "  return accumulator;";

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
    accumulator += node->leaf.val;
)";
      break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Non supported task: ", model::proto::Task_Name(stats.task)));
  }

  // TODO: Integer optimization of leaf values.
  spec.leaf_value_spec = {.dtype = proto::DType::FLOAT32, .dims = 1};

  spec.accumulator_initial_value =
      absl::StrJoin(model.initial_predictions(), ", ");
  return spec;
}

absl::StatusOr<CCInternalOptions> ComputeInternalOptions(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface,
    const ModelStatistics& stats, const proto::Options& options) {
  CCInternalOptions internal_options;
  RETURN_IF_ERROR(
      ComputeInternalOptionsFeature(stats, model, options, &internal_options));
  RETURN_IF_ERROR(
      ComputeInternalOptionsOutput(stats, options, &internal_options));
  RETURN_IF_ERROR(ComputeBaseInternalOptionsCategoricalDictionaries(
      model, stats, options, &internal_options));
  return internal_options;
}

absl::Status ComputeInternalOptionsFeature(const ModelStatistics& stats,
                                           const model::AbstractModel& model,
                                           const proto::Options& options,
                                           CCInternalOptions* out) {
  RETURN_IF_ERROR(
      ComputeBaseInternalOptionsFeature(stats, model, options, out));

  // Include cmath for numerical features for std::isnan().
  if (out->numerical_feature_is_float) {
    out->includes.cmath = true;
  }
  return absl::OkStatus();
}

absl::Status ComputeInternalOptionsOutput(const ModelStatistics& stats,
                                          const proto::Options& options,
                                          CCInternalOptions* out) {
  if (stats.has_conditions
          [model::decision_tree::proto::Condition::kContainsBitmapCondition] ||
      stats.has_conditions
          [model::decision_tree::proto::Condition::kContainsCondition]) {
    out->includes.algorithm = true;
    out->includes.array = true;
  }

  switch (stats.task) {
    case model::proto::Task::CLASSIFICATION: {
      switch (options.classification_output()) {
        case proto::ClassificationOutput::CLASS:
          out->output_type = "Label";
          if (!stats.is_binary_classification()) {
            out->includes.algorithm = true;
            out->includes.array = true;
            out->includes.algorithm = true;
          }
          break;
        case proto::ClassificationOutput::SCORE:
          // The output type is determined by the model specific conversion
          // code.
          if (!stats.is_binary_classification()) {
            out->includes.array = true;
          }
          break;
        case proto::ClassificationOutput::PROBABILITY:
          if (stats.is_binary_classification()) {
            out->output_type = "float";
          } else {
            out->includes.array = true;
            out->includes.algorithm = true;
            out->output_type = absl::StrCat(
                "std::array<float, ", stats.num_classification_classes, ">");
          }
          out->includes.cmath = true;
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
          "Non supported task: ", model::proto::Task_Name(stats.task)));
  }
  return absl::OkStatus();
}

absl::StatusOr<FeatureDef> GenFeatureDef(
    const dataset::proto::Column& col,
    const CCInternalOptions& internal_options) {
  // TODO: Add support for default values.
  // TODO: For integer numericals, use the min/max to possibly reduce the
  // required precision.
  switch (col.type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (internal_options.numerical_feature_is_float) {
        DCHECK_EQ(internal_options.feature_value_bytes, 4);
        return FeatureDef{
            .variable_name = StringToVariableSymbol(col.name()),
            .type = "Numerical",
            .underlying_type = "float",
            .default_value = {},
            .na_replacement = std::to_string(col.numerical().mean())};
      } else {
        return FeatureDef{.variable_name = StringToVariableSymbol(col.name()),
                          .type = "Numerical",
                          .underlying_type = SignedInteger(
                              internal_options.feature_value_bytes),
                          .default_value = {}};
      }
    } break;
    case dataset::proto::ColumnType::BOOLEAN:
    case dataset::proto::ColumnType::CATEGORICAL: {
      return FeatureDef{
          .variable_name = StringToVariableSymbol(col.name()),
          .type = absl::StrCat("Feature", StringToStructSymbol(col.name())),
          .underlying_type =
              UnsignedInteger(internal_options.feature_value_bytes),
          .default_value = {}};
    } break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Non supported feature type: ",
                       dataset::proto::ColumnType_Name(col.type())));
  }
}

absl::Status GenerateTreeInferenceIfElseNode(
    const dataset::proto::DataSpecification& dataspec,
    const model::decision_tree::NodeWithChildren& node, const int depth,
    const IfElseSetNodeFn& set_node_ifelse_fn, const int tree_idx,
    const CCInternalOptions& internal_options, std::string* content) {
  std::string prefix(depth * 2 + 2, ' ');

  if (node.IsLeaf()) {
    // The leaf value
    ASSIGN_OR_RETURN(const auto leaf,
                     set_node_ifelse_fn(node.node(), depth, tree_idx, prefix));
    absl::StrAppend(content, leaf);
    return absl::OkStatus();
  }

  std::string condition;

  // Create a contains condition.
  const auto categorical_contains_condition =
      [&](const int attribute_idx, absl::string_view variable_name,
          absl::Span<const int32_t> elements) -> absl::Status {
    const auto cat_dict_it =
        internal_options.categorical_dicts.find(attribute_idx);
    if (cat_dict_it == internal_options.categorical_dicts.end()) {
      return absl::InternalError("cannot find dict");
    }

    // if the column has a dictionary. elements is large.
    if (elements.size() < 8) {
      // List the elements are a sequence of ==.
      for (int element_idx = 0; element_idx < elements.size(); element_idx++) {
        if (element_idx > 0) {
          absl::StrAppend(&condition, " ||\n", prefix, "    ");
        }
        const auto element_str = absl::StrCat(
            "Feature", cat_dict_it->second.sanitized_name, "::k",
            cat_dict_it->second.sanitized_items[elements[element_idx]]);
        absl::SubstituteAndAppend(&condition, "instance.$0 == $1",
                                  variable_name, element_str);
      }
    } else {
      // Use binary search.
      std::string mask;
      for (const auto element_idx : elements) {
        const auto element_str =
            cat_dict_it->second.sanitized_items[element_idx];
        if (!mask.empty()) {
          absl::StrAppend(&mask, ",");
        }
        absl::StrAppend(&mask, " Feature", cat_dict_it->second.sanitized_name,
                        "::k", element_str);
      }

      absl::SubstituteAndAppend(
          &condition,
          "std::array<Feature$4,$0> mask = {$1};\n$3    "
          "std::binary_search(mask.begin(), mask.end(),  instance.$2)",
          elements.size(),                    //  $0
          mask,                               // $1
          variable_name,                      // $2
          prefix,                             // $3
          cat_dict_it->second.sanitized_name  // $4
      );
    }
    return absl::OkStatus();
  };

  // Evaluate condition
  switch (node.node().condition().condition().type_case()) {
    case model::decision_tree::proto::Condition::TypeCase::kHigherCondition: {
      const auto& typed_condition =
          node.node().condition().condition().higher_condition();
      const int attribute_idx = node.node().condition().attribute();
      const auto variable_name =
          StringToVariableSymbol(dataspec.columns(attribute_idx).name());
      absl::SubstituteAndAppend(&condition, "instance.$0 >= $1", variable_name,
                                typed_condition.threshold());
    } break;

    case model::decision_tree::proto::Condition::TypeCase::kContainsCondition: {
      const auto& typed_condition =
          node.node().condition().condition().contains_condition();
      const int attribute_idx = node.node().condition().attribute();
      const auto variable_name =
          StringToVariableSymbol(dataspec.columns(attribute_idx).name());

      RETURN_IF_ERROR(categorical_contains_condition(
          attribute_idx, variable_name, typed_condition.elements()));
    } break;

    case model::decision_tree::proto::Condition::TypeCase::
        kContainsBitmapCondition: {
      const auto& typed_condition =
          node.node().condition().condition().contains_bitmap_condition();
      const int attribute_idx = node.node().condition().attribute();
      const auto variable_name =
          StringToVariableSymbol(dataspec.columns(attribute_idx).name());
      std::vector<int32_t> elements;
      for (int item_idx = 0; item_idx < dataspec.columns(attribute_idx)
                                            .categorical()
                                            .number_of_unique_values();
           item_idx++) {
        if (utils::bitmap::GetValueBit(typed_condition.elements_bitmap(),
                                       item_idx)) {
          elements.push_back(item_idx);
        }
      }
      RETURN_IF_ERROR(categorical_contains_condition(attribute_idx,
                                                     variable_name, elements));
    } break;

    case model::decision_tree::proto::Condition::TypeCase::
        kTrueValueCondition: {
      const int attribute_idx = node.node().condition().attribute();
      const auto variable_name =
          StringToVariableSymbol(dataspec.columns(attribute_idx).name());
      absl::SubstituteAndAppend(&condition, "instance.$0", variable_name);
    } break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Non supported condition type:",
          model::decision_tree::ConditionTypeToString(
              node.node().condition().condition().type_case()),
          ". Using the algorithm=ROUTING might solve this issue."));
  }

  // Branching
  absl::SubstituteAndAppend(content, "$0if ($1) {\n", prefix, condition);
  RETURN_IF_ERROR(GenerateTreeInferenceIfElseNode(
      dataspec, *node.pos_child(), depth + 1, set_node_ifelse_fn, tree_idx,
      internal_options, content));
  absl::StrAppend(content, prefix, "} else {\n");
  RETURN_IF_ERROR(GenerateTreeInferenceIfElseNode(
      dataspec, *node.neg_child(), depth + 1, set_node_ifelse_fn, tree_idx,
      internal_options, content));
  absl::StrAppend(content, prefix, "}\n");
  return absl::OkStatus();
};

absl::Status GenerateTreeInferenceIfElse(
    const dataset::proto::DataSpecification& dataspec,
    const model::DecisionForestInterface& df_interface,
    const proto::Options& options, const CCInternalOptions& internal_options,
    const IfElseSetNodeFn& set_node_ifelse_fn, std::string* content) {
  for (int tree_idx = 0; tree_idx < df_interface.num_trees(); tree_idx++) {
    absl::StrAppend(content, "  // Tree #", tree_idx, "\n");
    const auto& tree = df_interface.decision_trees()[tree_idx];
    RETURN_IF_ERROR(GenerateTreeInferenceIfElseNode(
        dataspec, tree->root(), 0, set_node_ifelse_fn, tree_idx,
        internal_options, content));
    absl::StrAppend(content, "\n");
  }
  return absl::OkStatus();
}

absl::Status AddRoutingConditions(std::vector<RoutingConditionCode> conditions,
                                  const ValueBank& bank, std::string* content) {
  // Get the number of used conditions.
  // is_condition_used[i] is true if conditions[i] is used by the model.
  std::vector<bool> is_condition_used(conditions.size(), false);
  int num_used_conditions = 0;
  for (int condition_idx = 0; condition_idx < conditions.size();
       condition_idx++) {
    const auto& condition = conditions[condition_idx];
    if (bank.num_conditions[static_cast<int>(condition.type)] == 0) {
      // The model uses this condition code.
      continue;
    }
    is_condition_used[condition_idx] = true;
    num_used_conditions++;
  }
  const std::string prefix(6, ' ');

  if (num_used_conditions == 0) {
    // The model does not contain any condition.
    absl::StrAppend(content, prefix, "assert(false);\n");
  }
  if (num_used_conditions == 1) {
    absl::StrAppend(content, "\n");
  }

  int num_exported_conditions = 0;
  for (int condition_idx = 0; condition_idx < conditions.size();
       condition_idx++) {
    if (!is_condition_used[condition_idx]) {
      continue;
    }
    const auto& condition = conditions[condition_idx];

    if (num_used_conditions >= 2) {
      // Select which condition to apply.
      std::string used_code = condition.used_code;
      if (used_code.empty()) {
        // Implicit condition to see if the condition should be evaluated.
        used_code = absl::Substitute("condition_types[node->cond.feat] == $0",
                                     static_cast<int>(condition.type));
      }
      if (num_exported_conditions == 0) {
        // First condition.
        absl::StrAppend(content, "\n", prefix);
      } else if (num_exported_conditions > 0) {
        // Not the first condition.
        absl::StrAppend(content, prefix, "} else ");
      }
      absl::StrAppend(content, "if (", used_code, ") {\n");
    }

    // Apply the condition.
    absl::StrAppend(content, condition.eval_code);

    num_exported_conditions++;
  }

  // Debug test to make sure at least one condition was evaluated.
  if (num_used_conditions >= 2) {
    absl::StrAppend(content, prefix, R"(} else {
        assert(false);
      })");
  }

  return absl::OkStatus();
}

absl::Status GenerateTreeInferenceRouting(
    const dataset::proto::DataSpecification& dataspec,
    const model::DecisionForestInterface& df_interface,
    const proto::Options& options, const CCInternalOptions& internal_options,
    const SpecializedConversion& specialized_conversion,
    const ModelStatistics& stats, const ValueBank& routing_bank,
    std::string* content) {
  const std::string node_offset_type =
      UnsignedInteger(internal_options.node_offset_bytes);
  const std::string tree_index_type =
      UnsignedInteger(internal_options.tree_index_bytes);

  std::string numerical_type;
  if (internal_options.numerical_feature_is_float) {
    numerical_type = "float";
    STATUS_CHECK_EQ(internal_options.feature_value_bytes, 4);
  } else {
    numerical_type = SignedInteger(internal_options.feature_value_bytes);
  }

  std::string categorical_type;
  if (internal_options.categorical_idx_bytes > 0) {
    categorical_type = UnsignedInteger(internal_options.feature_value_bytes);
  }

  const std::string feature_index_type =
      UnsignedInteger(internal_options.feature_index_bytes);

  // Top of the loop: For-loop on trees & while-loop on nodes.
  absl::SubstituteAndAppend(content, R"(
  const Node* root = nodes;
  const Node* node;
  const char* raw_instance = (const char*)(&instance);)");

  absl::SubstituteAndAppend(content, R"(
  $0 eval;
  for ($1 tree_idx = 0; tree_idx != kNumTrees; tree_idx++) {
    node = root;
    while(node->pos) {)",
                            node_offset_type,  // $0
                            tree_index_type    // $1

  );

  // Condition
  RETURN_IF_ERROR(AddRoutingConditions(
      {{RoutingConditionType::OBLIQUE_CONDITION,
        absl::StrCat("node->cond.feat == ",
                     ObliqueFeatureIndex(options, internal_options)),
        absl::Substitute(
            R"(        const $0 num_projs = oblique_features[node->cond.obl];
        float obl_acc = -oblique_weights[node->cond.obl];
        for ($0 proj_idx=0; proj_idx<num_projs; proj_idx++){
          const auto off = node->cond.obl + proj_idx + 1;
          $1 numerical_feature;
          std::memcpy(&numerical_feature, raw_instance + oblique_features[off] * sizeof($1), sizeof($1));
          obl_acc += numerical_feature * oblique_weights[off];
        }
        eval = obl_acc >= 0;
)",
            ObliqueFeatureType(routing_bank), numerical_type)},
       {RoutingConditionType::HIGHER_CONDITION,
        {},
        absl::Substitute(R"(        $0 numerical_feature;
        std::memcpy(&numerical_feature, raw_instance + node->cond.feat * sizeof($0), sizeof($0));
        eval = numerical_feature >= node->cond.thr;
)",
                         numerical_type)},
       {RoutingConditionType::CONTAINS_CONDITION_BUFFER_BITMAP,
        {},
        absl::Substitute(R"(        $0 categorical_feature;
        std::memcpy(&categorical_feature, raw_instance + node->cond.feat * sizeof($0), sizeof($0));
        eval = categorical_bank[categorical_feature + node->cond.cat];
)",
                         categorical_type)}},
      routing_bank, content));

  // Middle of the loop: Select the next node.
  absl::SubstituteAndAppend(content, R"(
      node += (node->pos & -eval) + 1;
    })");

  // Add the leaf value to the accumulator.
  absl::StrAppend(content, specialized_conversion.routing_node);

  // Bottom of the loop: Go to the next tree.
  absl::SubstituteAndAppend(content, R"(    root += root_deltas[tree_idx];
  }

)");
  return absl::OkStatus();
}

// Generate the static data of a single Node needed for the routing algorithm.
// This function is called by "GenRoutingModelData" on each of the tree nodes.
absl::Status GenRoutingModelDataNode(
    const model::AbstractModel& model,
    const dataset::proto::DataSpecification& dataspec,
    const ModelStatistics& stats,
    const SpecializedConversion& specialized_conversion,
    const proto::Options& options, const CCInternalOptions& internal_options,
    const model::decision_tree::NodeWithChildren& node, const int depth,
    std::string* serialized_nodes, int* node_idx, ValueBank* bank) {
  if (node.IsLeaf()) {
    absl::StrAppend(serialized_nodes, "{.leaf={.val=");

    // Unroll the leaf values.
    const auto leaf_value = specialized_conversion.leaf_value_fn(node.node());
    if (specialized_conversion.leaf_value_spec.dims > 1) {
      // The leaf value is an index into the "leaf_value_bank" array divided by
      // the output node dimension. Since all the leaves have the same
      // dimension, this last division allows to store smaller integers,
      // possibly requiring a smaller integer representation.
      STATUS_CHECK_EQ(
          bank->leaf_value.size() % specialized_conversion.leaf_value_spec.dims,
          0);
      const auto encoded_leaf_value =
          bank->leaf_value.size() / specialized_conversion.leaf_value_spec.dims;
      absl::StrAppend(serialized_nodes, encoded_leaf_value);

      // Add the leaf values to the bank.
      if (std::holds_alternative<std::vector<float>>(leaf_value)) {
        const auto& typed_values = std::get<std::vector<float>>(leaf_value);
        STATUS_CHECK_EQ(typed_values.size(),
                        specialized_conversion.leaf_value_spec.dims);
        bank->leaf_value.insert(bank->leaf_value.end(), typed_values.begin(),
                                typed_values.end());
      } else {
        // Note: We don't implement the int32 and bool version as they are not
        // used (yet?).
        return absl::InvalidArgumentError("Non supported leaf type");
      }
    } else {
      // TODO: The use of variates is verbose. Make this block simpler &
      // more readable.
      if (std::holds_alternative<std::vector<bool>>(leaf_value)) {
        absl::StrAppend(serialized_nodes,
                        std::get<std::vector<bool>>(leaf_value).front());
      } else if (std::holds_alternative<std::vector<int32_t>>(leaf_value)) {
        absl::StrAppend(serialized_nodes,
                        std::get<std::vector<int32_t>>(leaf_value).front());
      } else if (std::holds_alternative<std::vector<float>>(leaf_value)) {
        absl::StrAppend(serialized_nodes,
                        std::get<std::vector<float>>(leaf_value).front());
      } else {
        return absl::InvalidArgumentError("Non supported leaf type");
      }
    }
    absl::StrAppend(serialized_nodes, "}},\n");
    (*node_idx)++;
    return absl::OkStatus();
  }

  // Reserve the node idx.
  (*node_idx)++;

  // The negative child.
  // Note: We don't print the negative child data yet as we don't know how many
  // nodes it will require, and the number of node in the negative child is
  // needed in the data of the parent node that should be written before.
  const auto save_node_idx = *node_idx;
  std::string serialized_neg_nodes;  // Temporary storage for the negative node.
  RETURN_IF_ERROR(GenRoutingModelDataNode(
      model, dataspec, stats, specialized_conversion, options, internal_options,
      *node.neg_child(), depth + 1, &serialized_neg_nodes, node_idx, bank));

  // This is the node offset between a parent node and the positive node. Note
  // that the offset to the negative node is 1 and does not need to be encoded.
  const auto delta_pos_node = *node_idx - save_node_idx;

  // Get the dense feature index.
  const int attribute_idx = node.node().condition().attribute();
  const int feature_idx =
      internal_options.column_idx_to_feature_idx.at(attribute_idx);

  // Create a contains condition node.
  const auto categorical_contains_condition =
      [&](const std::vector<bool>& bitmap) {
        // TODO: For bitmap requiring less bits than a bitmap bank index,
        // store the mask in the node directly (instead of using the bank).
        // TODO: Search if the bitmap bank already contains the current
        // bitmap. If so, use the existing bitmap segment instead.
        absl::SubstituteAndAppend(serialized_nodes,
                                  "{.pos=$0,.cond={.feat=$1,.cat=$2}},\n",
                                  delta_pos_node,           // $0
                                  feature_idx,              // $1
                                  bank->categorical.size()  // $2
        );
        bank->categorical.insert(bank->categorical.end(), bitmap.begin(),
                                 bitmap.end());
        bank->num_conditions[static_cast<int>(
            RoutingConditionType::CONTAINS_CONDITION_BUFFER_BITMAP)]++;
      };

  // Encode all the possible condition nodes.
  switch (node.node().condition().condition().type_case()) {
    case model::decision_tree::proto::Condition::TypeCase::kHigherCondition: {
      // Condition of the type "a >= threhsold".
      const auto& typed_condition =
          node.node().condition().condition().higher_condition();
      float threshold = typed_condition.threshold();
      if (!internal_options.numerical_feature_is_float) {
        threshold = std::ceil(threshold);
      }
      absl::SubstituteAndAppend(serialized_nodes,
                                "{.pos=$0,.cond={.feat=$1,.thr=$2}},\n",
                                delta_pos_node,  // $0
                                feature_idx,     // $1
                                threshold        // $2
      );
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
      categorical_contains_condition(bitmap);
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
      categorical_contains_condition(bitmap);
    } break;

    case model::decision_tree::proto::Condition::TypeCase::kObliqueCondition: {
      // Sparse oblique condition.
      const auto& typed_condition =
          node.node().condition().condition().oblique_condition();
      // TODO: Compute oblique projections as integers instead of floats if
      // both the feature values and oblique weights are integers.

      const size_t oblique_idx = bank->oblique_features.size();
      const size_t num_projections = typed_condition.weights_size();

      // Magic values
      bank->oblique_features.push_back(num_projections);
      bank->oblique_weights.push_back(typed_condition.threshold());

      // Oblique weights + indexes
      for (size_t proj_idx = 0; proj_idx < num_projections; proj_idx++) {
        bank->oblique_weights.push_back(typed_condition.weights(proj_idx));
        const int sub_feature_idx =
            internal_options.column_idx_to_feature_idx.at(
                typed_condition.attributes(proj_idx));
        bank->oblique_features.push_back(sub_feature_idx);
      }

      absl::SubstituteAndAppend(
          serialized_nodes, "{.pos=$0,.cond={.feat=$1,.obl=$2}},\n",
          delta_pos_node,                                  // $0
          ObliqueFeatureIndex(options, internal_options),  // $1
          oblique_idx                                      // $2
      );
      bank->num_conditions[static_cast<int>(
          RoutingConditionType::OBLIQUE_CONDITION)]++;
    } break;

    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Non supported condition type: ",
                       model::decision_tree::ConditionTypeToString(
                           node.node().condition().condition().type_case())));
  }

  absl::StrAppend(serialized_nodes, serialized_neg_nodes);
  serialized_neg_nodes.clear();

  RETURN_IF_ERROR(GenRoutingModelDataNode(
      model, dataspec, stats, specialized_conversion, options, internal_options,
      *node.pos_child(), depth + 1, serialized_nodes, node_idx, bank));

  return absl::OkStatus();
}

// Outputs the code to encode the bank data and other arrays of the routing
// algorithm.
absl::Status GenRoutingModelDataBank(const CCInternalOptions& internal_options,
                                     const ValueBank& bank,
                                     std::string* content) {
  const std::string root_delta_type =
      UnsignedInteger(internal_options.node_offset_bytes);
  absl::SubstituteAndAppend(content, R"(
static const $0 root_deltas[] = {$1};

)",
                            root_delta_type,
                            absl::StrJoin(bank.root_deltas, ","));

  if (!bank.categorical.empty()) {
    STATUS_CHECK_LE(MaxUnsignedValueToNumBytes(bank.categorical.size()),
                    internal_options.categorical_idx_bytes);
  }

  // Record the categorical mask bank.
  if (internal_options.categorical_idx_bytes > 0) {
    absl::SubstituteAndAppend(
        content, R"(
static const std::bitset<$0> categorical_bank {"$1"};
)",
        bank.categorical.size(),
        absl::StrJoin(bank.categorical.rbegin(), bank.categorical.rend(), ""));
  }

  // Record the leaf value bank.
  if (!bank.leaf_value.empty()) {
    absl::SubstituteAndAppend(content, R"(
static const float leaf_value_bank[] = {$0};
)",
                              absl::StrJoin(bank.leaf_value, ","));
  }

  // Oblique bank data
  if (!bank.oblique_features.empty()) {
    const std::string feature_index_type =
        UnsignedInteger(internal_options.feature_index_bytes);

    absl::SubstituteAndAppend(content, R"(
static const float oblique_weights[] = {$0};
)",
                              absl::StrJoin(bank.oblique_weights, ","));

    absl::SubstituteAndAppend(content, R"(
static const $0 oblique_features[] = {$1};
)",
                              feature_index_type,
                              absl::StrJoin(bank.oblique_features, ","));
  }

  return absl::OkStatus();
}

// Outputs the struct code to encode the nodes in the routing
// algorithm.
absl::Status GenRoutingModelDataStruct(
    const CCInternalOptions& internal_options, const ModelStatistics& stats,
    const SpecializedConversion& specialized_conversion, const ValueBank& bank,
    std::string* content) {
  // TODO: Add boolean condition support.
  // TODO: Use the "bank" to find possibly most compact index
  // representations for "categorical_idx_bytes".

  std::string is_greather_threshold_type;
  if (internal_options.numerical_feature_is_float) {
    is_greather_threshold_type = "float";
    STATUS_CHECK_EQ(internal_options.feature_value_bytes, 4);
  } else {
    is_greather_threshold_type =
        SignedInteger(internal_options.feature_value_bytes);
  }

  const std::string feature_index_type =
      UnsignedInteger(internal_options.feature_index_bytes);

  const std::string node_offset_type =
      UnsignedInteger(internal_options.node_offset_bytes);

  std::string node_value_type;
  if (specialized_conversion.leaf_value_spec.dims == 1) {
    // The leaf value is stored in the node struct.
    node_value_type =
        DTypeToCCType(specialized_conversion.leaf_value_spec.dtype);
  } else {
    // The leaf values are stored in a separate buffer. The node struct contains
    // an index to this buffer.
    node_value_type = UnsignedInteger(MaxUnsignedValueToNumBytes(
        stats.num_leaves / specialized_conversion.leaf_value_spec.dims));
  }

  // TODO: Re-order the item dynamically to optimize the alignment.
  absl::SubstituteAndAppend(content, R"(
struct __attribute__((packed)) Node {
  $2 pos = 0;
  union {
    struct __attribute__((packed)) {
      $1 feat;
      union {
        $0 thr;)",
                            is_greather_threshold_type,  // $0
                            feature_index_type,          // $1
                            node_offset_type             // $2
  );

  if (internal_options.categorical_idx_bytes > 0) {
    const std::string categorical_idx_type =
        UnsignedInteger(internal_options.categorical_idx_bytes);
    absl::SubstituteAndAppend(content, R"(
        $0 cat;)",
                              categorical_idx_type  // $0
    );
  }

  if (!bank.oblique_weights.empty()) {
    absl::SubstituteAndAppend(content, R"(
        $0 obl;)",
                              ObliqueFeatureType(bank)  // $0
    );
  }

  absl::SubstituteAndAppend(content, R"(
      };
    } cond;
    struct __attribute__((packed)) {
      $0 val;
    } leaf;
  };
};
)",
                            node_value_type  // $0
  );
  return absl::OkStatus();
}

absl::Status GenRoutingModelData(
    const model::AbstractModel& model,
    const dataset::proto::DataSpecification& dataspec,
    const model::DecisionForestInterface& df_interface,
    const ModelStatistics& stats,
    const SpecializedConversion& specialized_conversion,
    const proto::Options& options, const CCInternalOptions& internal_options,
    std::string* content, ValueBank* bank) {
  // Compute the node data and populate the banks.
  std::string serialized_nodes;

  bank->root_deltas.reserve(df_interface.num_trees());
  int node_idx = 0;
  for (int tree_idx = 0; tree_idx < df_interface.num_trees(); tree_idx++) {
    const auto begin_node_idx = node_idx;
    const auto& tree = df_interface.decision_trees()[tree_idx];
    RETURN_IF_ERROR(GenRoutingModelDataNode(
        model, dataspec, stats, specialized_conversion, options,
        internal_options, tree->root(), 0, &serialized_nodes, &node_idx, bank));
    bank->root_deltas.push_back(node_idx - begin_node_idx);
  }

  // Generate Node struct
  RETURN_IF_ERROR(GenRoutingModelDataStruct(
      internal_options, stats, specialized_conversion, *bank, content));
  // TODO: Add option to encode the node data by a string of bytes (more
  // compact and faster to compile, but less readable).
  absl::StrAppend(content, "static const Node nodes[] = {\n", serialized_nodes,
                  "};\n");

  if (stats.has_multiple_condition_types) {
    ASSIGN_OR_RETURN(const auto condition_types,
                     GenRoutingModelDataConditionType(model, stats));
    absl::SubstituteAndAppend(content, R"(
static const uint8_t condition_types[] = {$0};

)",
                              absl::StrJoin(condition_types, ","));
  }

  // Print the bank.
  RETURN_IF_ERROR(GenRoutingModelDataBank(internal_options, *bank, content));
  return absl::OkStatus();
}

std::string ObliqueFeatureType(const ValueBank& bank) {
  return UnsignedInteger(
      MaxUnsignedValueToNumBytes(bank.oblique_features.size()));
}
}  // namespace yggdrasil_decision_forests::serving::embed::internal
