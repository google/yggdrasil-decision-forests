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

#include "yggdrasil_decision_forests/serving/embed/embed.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
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
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/serving/embed/utils.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed {

namespace {
constexpr absl::string_view kLabelReservedSymbol = "Label";

// Generates the struct for a single instance (i.e., an example without a
// label).
absl::StatusOr<std::string> GenInstanceStruct(
    const model::AbstractModel& model, const proto::Options& options,
    const internal::InternalOptions& internal_options,
    const internal::ModelStatistics& stats) {
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

  for (const auto input_feature : model.input_features()) {
    const auto& col = model.data_spec().columns(input_feature);
    const std::string variable_name = StringToVariableSymbol(col.name());
    ASSIGN_OR_RETURN(const auto feature_def,
                     internal::GenFeatureDef(col, internal_options));
    absl::StrAppend(&content, "  ", feature_def.type, " ", variable_name);

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
    const internal::InternalOptions& internal_options) {
  std::string content;

  // TODO: Create a hashmap with the string values is the user requests it.
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
  RETURN_IF_ERROR(CheckModelName(options.name()));
  for (const auto& column_idx : model.input_features()) {
    RETURN_IF_ERROR(
        CheckFeatureName(model.data_spec().columns(column_idx).name()));
  }

  ASSIGN_OR_RETURN(const internal::ModelStatistics stats,
                   internal::ComputeStatistics(model, *df_interface));

  ASSIGN_OR_RETURN(
      const internal::InternalOptions internal_options,
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
                       internal::SpecializedConversionGradientBoostedTrees(
                           *model_gbt, stats, internal_options, options));
    } else if (model_rf) {
      ASSIGN_OR_RETURN(specialized_conversion,
                       internal::SpecializedConversionRandomForest(
                           *model_rf, stats, internal_options, options));
    } else {
      return absl::InvalidArgumentError("The model type is not supported.");
    }
    RETURN_IF_ERROR(specialized_conversion.Validate());
  }

  // Generate the code.
  absl::node_hash_map<Filename, Content> result;
  std::string header;

  // Open define and namespace.
  absl::SubstituteAndAppend(&header, R"(#ifndef YDF_MODEL_$0_H_
#define YDF_MODEL_$0_H_

#include <stdint.h>
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

  // Instance struct.
  ASSIGN_OR_RETURN(const auto instance_struct,
                   GenInstanceStruct(model, options, internal_options, stats));
  absl::StrAppend(&header, instance_struct);

  // Model data
  if (options.algorithm() == proto::Algorithm::ROUTING) {
    RETURN_IF_ERROR(internal::GenRoutingModelData(
        model, model.data_spec(), *df_interface, stats, specialized_conversion,
        options, internal_options, &header));
  }

  // Predict method
  std::string predict_body;
  RETURN_IF_ERROR(internal::CorePredict(
      model.data_spec(), *df_interface, specialized_conversion, stats,
      internal_options, options, &predict_body));
  STATUS_CHECK(!predict_body.empty());

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

  absl::SubstituteAndAppend(&header, R"(
inline $0 Predict(const Instance& instance) {
$1}
)",
                            predict_output_type, predict_body);

  // Close define and namespace.
  absl::SubstituteAndAppend(&header, R"(
}  // namespace $0
#endif
)",
                            StringToVariableSymbol(options.name()));

  result[absl::StrCat(options.name(), ".h")] = header;
  return result;
}

namespace internal {

absl::Status CorePredict(const dataset::proto::DataSpecification& dataspec,
                         const model::DecisionForestInterface& df_interface,
                         const SpecializedConversion& specialized_conversion,
                         const ModelStatistics& stats,
                         const InternalOptions& internal_options,
                         const proto::Options& options, std::string* content) {
  // Accumulator
  absl::SubstituteAndAppend(content, "  $0 accumulator {$1};\n",
                            specialized_conversion.accumulator_type,
                            specialized_conversion.accumulator_initial_value);

  // Accumulate leaf values
  switch (options.algorithm()) {
    case proto::Algorithm::IF_ELSE:
      RETURN_IF_ERROR(GenerateTreeInferenceIfElse(
          dataspec, df_interface, options, internal_options,
          specialized_conversion.set_node_ifelse_fn, content));
      break;
    case proto::Algorithm::ROUTING:
      RETURN_IF_ERROR(GenerateTreeInferenceRouting(
          dataspec, df_interface, options, internal_options,
          specialized_conversion, stats, content));
      break;
    default:
      return absl::InvalidArgumentError("Non supported algorithm.");
  }

  // Accumulator to predictions.
  absl::StrAppend(content, specialized_conversion.return_prediction);
  return absl::OkStatus();
}

absl::StatusOr<SpecializedConversion> SpecializedConversionRandomForest(
    const model::random_forest::RandomForestModel& model,
    const internal::ModelStatistics& stats,
    const internal::InternalOptions& internal_options,
    const proto::Options& options) {
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
                "  return static_cast<Label>(accumulator >= $0);\n",
                stats.num_trees / 2);
          } else {
            absl::StrAppend(
                &spec.return_prediction,
                "  return "
                "static_cast<Label>(std::distance(accumulator.begin(), "
                "std::max_element(accumulator.begin(), "
                "accumulator.end())));\n");
          }
          break;
        case proto::ClassificationOutput::SCORE:
          spec.return_prediction = "  return accumulator;\n";
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
            spec.return_prediction = "return accumulator;\n";
          }
          break;
      }
    } break;

    case model::proto::Task::REGRESSION:
      spec.accumulator_type = "float";
      spec.return_prediction = "  return accumulator;\n";
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
SpecializedConversionGradientBoostedTrees(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& model,
    const internal::ModelStatistics& stats,
    const internal::InternalOptions& internal_options,
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
                "  return static_cast<Label>(accumulator >= 0);\n";
          } else {
            spec.return_prediction =
                ("  return "
                 "static_cast<Label>(std::distance(accumulator.begin(), "
                 "std::max_element(accumulator.begin(), "
                 "accumulator.end())));\n");
          }
          break;
        case proto::ClassificationOutput::SCORE:
          spec.return_prediction = "  return accumulator;\n";
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

absl::StatusOr<ModelStatistics> ComputeStatistics(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface) {
  ModelStatistics stats{
      .num_trees = df_interface.num_trees(),
      .num_features = static_cast<int>(model.input_features().size()),
      .task = model.task(),
  };

  if (stats.is_classification()) {
    stats.num_classification_classes = static_cast<int>(
        model.LabelColumnSpec().categorical().number_of_unique_values() - 1);
  }

  // Scan the trees
  for (const auto& tree : df_interface.decision_trees()) {
    int64_t num_leaves_in_tree = 0;
    tree->IterateOnNodes([&](const model::decision_tree::NodeWithChildren& node,
                             int depth) {
      stats.max_depth = std::max(stats.max_depth, static_cast<int64_t>(depth));
      if (node.IsLeaf()) {
        num_leaves_in_tree++;
        stats.num_leaves++;
      } else {
        stats.num_conditions++;
        stats.has_conditions[node.node().condition().condition().type_case()] =
            true;

        if (node.node()
                .condition()
                .condition()
                .has_contains_bitmap_condition() ||
            node.node().condition().condition().has_contains_condition()) {
          const int attribute_idx = node.node().condition().attribute();
          const auto num_unique_values = model.data_spec()
                                             .columns(attribute_idx)
                                             .categorical()
                                             .number_of_unique_values();
          stats.sum_size_categorical_bitmap_masks += num_unique_values;
        }
      }
    });
    stats.max_num_leaves_per_tree =
        std::max(stats.max_num_leaves_per_tree, num_leaves_in_tree);
  }

  stats.has_multiple_condition_types =
      std::count(stats.has_conditions.begin(), stats.has_conditions.end(),
                 true) > 1;
  return stats;
}

absl::StatusOr<InternalOptions> ComputeInternalOptions(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface,
    const ModelStatistics& stats, const proto::Options& options) {
  InternalOptions internal_options;
  RETURN_IF_ERROR(
      ComputeInternalOptionsFeature(stats, model, options, &internal_options));
  RETURN_IF_ERROR(
      ComputeInternalOptionsOutput(stats, options, &internal_options));
  RETURN_IF_ERROR(ComputeInternalOptionsCategoricalDictionaries(
      model, stats, options, &internal_options));
  return internal_options;
}

absl::Status ComputeInternalOptionsFeature(const ModelStatistics& stats,
                                           const model::AbstractModel& model,
                                           const proto::Options& options,
                                           InternalOptions* out) {
  out->feature_value_bytes = 1;
  out->numerical_feature_is_float = false;

  out->feature_index_bytes = MaxUnsignedValueToNumBytes(stats.num_features);
  out->tree_index_bytes = MaxUnsignedValueToNumBytes(stats.num_trees);
  out->node_index_bytes =
      MaxUnsignedValueToNumBytes(stats.num_leaves + stats.num_conditions);

  if (stats.sum_size_categorical_bitmap_masks == 0) {
    out->categorical_idx_bytes = 0;
  } else {
    out->categorical_idx_bytes =
        MaxUnsignedValueToNumBytes(stats.sum_size_categorical_bitmap_masks);
  }

  // This is the number of bytes to encode a node index in a tree. The precision
  // for an offset is in average 50% smaller.
  // TODO: Optimize node_offset_bytes.
  out->node_offset_bytes = MaxUnsignedValueToNumBytes(
      NumLeavesToNumNodes(stats.max_num_leaves_per_tree));

  out->column_idx_to_feature_idx.assign(model.data_spec().columns_size(), -1);

  // Feature encoding.
  for (size_t feature_idx = 0; feature_idx < model.input_features().size();
       feature_idx++) {
    const auto& column_idx = model.input_features()[feature_idx];
    const auto& col_spec = model.data_spec().columns(column_idx);

    // Index the input feature.
    out->column_idx_to_feature_idx[column_idx] = feature_idx;

    switch (col_spec.type()) {
      case dataset::proto::ColumnType::NUMERICAL: {
        switch (col_spec.dtype()) {
          // TODO: Add support for unsigned features.
          case dataset::proto::DTYPE_INVALID:  // Default
          case dataset::proto::DTYPE_FLOAT32:
          // Note: float64 are always converted to float32 during training.
          case dataset::proto::DTYPE_FLOAT64:
            out->numerical_feature_is_float = true;
            out->feature_value_bytes = std::max(out->feature_value_bytes, 4);
            break;
          case dataset::proto::DTYPE_INT16:
            out->feature_value_bytes = std::max(out->feature_value_bytes, 2);
            break;
          case dataset::proto::DTYPE_INT32:
          // Note: int64 are always converted to int32 during training.
          case dataset::proto::DTYPE_INT64:
            out->feature_value_bytes = std::max(out->feature_value_bytes, 4);
            break;
          case dataset::proto::DTYPE_INT8:
          case dataset::proto::DTYPE_BOOL:
            // Nothing to do.
            break;
          default:
            return absl::InvalidArgumentError(
                absl::StrCat("Non supported numerical feature type: ",
                             dataset::proto::DType_Name(col_spec.dtype())));
        }
      } break;
      case dataset::proto::ColumnType::CATEGORICAL: {
        const int feature_bytes = MaxUnsignedValueToNumBytes(
            col_spec.categorical().number_of_unique_values());
        out->feature_value_bytes =
            std::max(out->feature_value_bytes, feature_bytes);
      } break;
      case dataset::proto::ColumnType::BOOLEAN:
        // Nothing to do.
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Non supported feature type: ",
                         dataset::proto::ColumnType_Name(col_spec.type())));
    }
  }
  return absl::OkStatus();
}

absl::Status ComputeInternalOptionsOutput(const ModelStatistics& stats,
                                          const proto::Options& options,
                                          InternalOptions* out) {
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

absl::Status ComputeInternalOptionsCategoricalDictionaries(
    const model::AbstractModel& model, const ModelStatistics& stats,
    const proto::Options& options, InternalOptions* out) {
  const auto add_dict = [&](absl::string_view name, const int column_idx,
                            const dataset::proto::CategoricalSpec& column,
                            const bool is_label) {
    // Get the list of values.
    auto& dictionary = out->categorical_dicts[column_idx];
    dictionary.sanitized_name = name;
    dictionary.is_label = is_label;
    dictionary.sanitized_items.assign(
        column.number_of_unique_values() - is_label, "");
    for (const auto& item : column.items()) {
      int index = item.second.index();

      // Labels don't have the OOB item.
      if (is_label) {
        if (index == dataset::kOutOfDictionaryItemIndex) {
          continue;
        }
        index--;
      }

      std::string item_symbol;
      if (!is_label && index == dataset::kOutOfDictionaryItemIndex) {
        item_symbol =
            "OutOfVocabulary";  // Better than the default <OOV> symbol.
      } else {
        item_symbol = StringToStructSymbol(item.first,
                                           /*.ensure_letter_first=*/false);
      }
      dictionary.sanitized_items[index] = item_symbol;
    }
  };

  if (model.task() == model::proto::Task::CLASSIFICATION &&
      !model.LabelColumnSpec().categorical().is_already_integerized()) {
    // The classification labels
    add_dict(kLabelReservedSymbol, model.label_col_idx(),
             model.LabelColumnSpec().categorical(), true);
  }

  // The categorical features.
  for (const auto input_feature : model.input_features()) {
    const auto& col_spec = model.data_spec().columns(input_feature);
    if (col_spec.type() != dataset::proto::ColumnType::CATEGORICAL) {
      continue;
    }
    add_dict(StringToStructSymbol(col_spec.name()), input_feature,
             col_spec.categorical(), false);
  }

  return absl::OkStatus();
}

absl::StatusOr<FeatureDef> GenFeatureDef(
    const dataset::proto::Column& col,
    const internal::InternalOptions& internal_options) {
  // TODO: Add support for default values.
  // TODO: For integer numericals, use the min/max to possibly reduce the
  // required precision.
  switch (col.type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (internal_options.numerical_feature_is_float) {
        DCHECK_EQ(internal_options.feature_value_bytes, 4);
        return FeatureDef{.type = "Numerical",
                          .underlying_type = "float",
                          .default_value = {}};
      } else {
        return FeatureDef{.type = "Numerical",
                          .underlying_type = SignedInteger(
                              internal_options.feature_value_bytes),
                          .default_value = {}};
      }
    } break;
    case dataset::proto::ColumnType::BOOLEAN:
    case dataset::proto::ColumnType::CATEGORICAL: {
      return FeatureDef{
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
    const internal::InternalOptions& internal_options, std::string* content) {
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
      return absl::InvalidArgumentError("Non supported condition type.");
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
    const proto::Options& options, const InternalOptions& internal_options,
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

absl::Status GenerateTreeInferenceRouting(
    const dataset::proto::DataSpecification& dataspec,
    const model::DecisionForestInterface& df_interface,
    const proto::Options& options, const InternalOptions& internal_options,
    const SpecializedConversion& specialized_conversion,
    const ModelStatistics& stats, std::string* content) {
  const std::string node_offset_type =
      UnsignedInteger(internal_options.node_offset_bytes);
  const std::string tree_index_type =
      UnsignedInteger(internal_options.tree_index_bytes);

  std::string is_greather_threshold_type;
  if (internal_options.numerical_feature_is_float) {
    is_greather_threshold_type = "float";
    STATUS_CHECK_EQ(internal_options.feature_value_bytes, 4);
  } else {
    is_greather_threshold_type =
        SignedInteger(internal_options.feature_value_bytes);
  }

  std::string categorical_idx_type;
  if (internal_options.categorical_idx_bytes > 0) {
    categorical_idx_type =
        UnsignedInteger(internal_options.feature_value_bytes);
  }

  // Top of the loop: For-loop on trees & while-loop on nodes.
  absl::SubstituteAndAppend(content, R"(
  const Node* root = nodes;
  const Node* node;
  const auto* raw_numerical = reinterpret_cast<const $0*>(&instance);
  (void) raw_numerical;)",
                            is_greather_threshold_type  // $0
  );

  if (!categorical_idx_type.empty()) {
    absl::SubstituteAndAppend(content, R"(
  const auto* raw_categorical = reinterpret_cast<const $0*>(&instance);
  (void) raw_categorical;)",
                              categorical_idx_type  // $0
    );
  }

  absl::SubstituteAndAppend(content, R"(
  $0 eval;
  for ($1 tree_idx = 0; tree_idx != kNumTrees; tree_idx++) {
    node = root;
    while(node->pos) {)",
                            node_offset_type,  // $0
                            tree_index_type    // $1

  );

  // Condition

  // Add the code of a condition. If there are multiple types of supported
  // conditions, wraps the condition in an "if" block that checks "type".
  const auto add_condition_code =
      [&](absl::Span<const model::decision_tree::proto::Condition::TypeCase>
              ydf_condition_types,
          const RoutingConditionType routing_cond_type,
          const absl::string_view code) {
        const int int_routing_cond_type = static_cast<int>(routing_cond_type);

        bool model_has_condition = false;
        for (const auto ydf_condition_type : ydf_condition_types) {
          if (stats.has_conditions[ydf_condition_type]) {
            model_has_condition = true;
          }
        }
        if (!model_has_condition) {
          // The model does need this condition code.
          return;
        }

        if (stats.has_multiple_condition_types) {
          if (int_routing_cond_type != 0) {
            absl::StrAppend(content, " else ");
          } else {
            absl::StrAppend(content, "\n      ");
          }
          absl::SubstituteAndAppend(
              content, "if (condition_types[node->cond.feat] == $0) {\n",
              routing_cond_type);
        } else {
          absl::StrAppend(content, "\n");
        }
        absl::StrAppend(content, code);
        if (stats.has_multiple_condition_types) {
          absl::StrAppend(content, "      }");
        }
      };
  add_condition_code({model::decision_tree::proto::Condition::kHigherCondition},
                     RoutingConditionType::HIGHER_CONDITION,
                     "        eval = raw_numerical[node->cond.feat] >= "
                     "node->cond.thr;\n");
  add_condition_code(
      {model::decision_tree::proto::Condition::kContainsCondition,
       model::decision_tree::proto::Condition::kContainsBitmapCondition},
      RoutingConditionType::CONTAINS_CONDITION_BUFFER_BITMAP,
      "        eval = categorical_bank[raw_categorical[node->cond.feat] + "
      "node->cond.cat];\n");

  if (stats.has_multiple_condition_types) {
    absl::StrAppend(content, R"( else {
        assert(false);
      })");
  }

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
    const proto::Options& options, const InternalOptions& internal_options,
    const model::decision_tree::NodeWithChildren& node, const int depth,
    std::string* serialized_nodes, int* node_idx,
    std::vector<bool>* categorical_bank, std::vector<float>* leaf_value_bank) {
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
          leaf_value_bank->size() % specialized_conversion.leaf_value_spec.dims,
          0);
      const auto encoded_leaf_value =
          leaf_value_bank->size() / specialized_conversion.leaf_value_spec.dims;
      absl::StrAppend(serialized_nodes, encoded_leaf_value);

      // Add the leaf values to the bank.
      if (std::holds_alternative<std::vector<float>>(leaf_value)) {
        const auto& typed_values = std::get<std::vector<float>>(leaf_value);
        STATUS_CHECK_EQ(typed_values.size(),
                        specialized_conversion.leaf_value_spec.dims);
        leaf_value_bank->insert(leaf_value_bank->end(), typed_values.begin(),
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
      *node.neg_child(), depth + 1, &serialized_neg_nodes, node_idx,
      categorical_bank, leaf_value_bank));

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
                                  categorical_bank->size()  // $2
        );
        categorical_bank->insert(categorical_bank->end(), bitmap.begin(),
                                 bitmap.end());
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

    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Non supported condition type ",
                       node.node().condition().condition().type_case()));
  }

  absl::StrAppend(serialized_nodes, serialized_neg_nodes);
  serialized_neg_nodes.clear();

  RETURN_IF_ERROR(GenRoutingModelDataNode(
      model, dataspec, stats, specialized_conversion, options, internal_options,
      *node.pos_child(), depth + 1, serialized_nodes, node_idx,
      categorical_bank, leaf_value_bank));

  return absl::OkStatus();
}

absl::Status GenRoutingModelData(
    const model::AbstractModel& model,
    const dataset::proto::DataSpecification& dataspec,
    const model::DecisionForestInterface& df_interface,
    const ModelStatistics& stats,
    const SpecializedConversion& specialized_conversion,
    const proto::Options& options, const InternalOptions& internal_options,
    std::string* content) {
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

  const std::string node_index_type =
      UnsignedInteger(internal_options.node_index_bytes);

  std::string categorical_idx_type;
  if (internal_options.categorical_idx_bytes > 0) {
    categorical_idx_type =
        UnsignedInteger(internal_options.categorical_idx_bytes);
  }

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

  // TODO: Add boolean condition support.

  absl::SubstituteAndAppend(content, R"(
struct __attribute__((packed)) Node {
  $2 pos = 0;
  union {
    struct {
      $1 feat;
      union {
        $0 thr;)",
                            is_greather_threshold_type,  // $0
                            feature_index_type,          // $1
                            node_offset_type             // $2
  );

  if (!categorical_idx_type.empty()) {
    absl::SubstituteAndAppend(content, R"(
        $0 cat;)",
                              categorical_idx_type  // $0
    );
  }

  absl::SubstituteAndAppend(content, R"(
      };
    } cond;
    struct {
      $0 val;
    } leaf;
  };
};
)",
                            node_value_type  // $0
  );

  std::string serialized_nodes;

  std::vector<bool> categorical_bank;

  // Values of leaf nodes that are stored outside of the node.
  std::vector<float> leaf_value_bank;

  // The "root_deltas" contains the number of nodes in each tree. The node index
  // of the root of each tree can be computed by running a cumulative sum.
  std::vector<int> root_deltas;
  root_deltas.reserve(df_interface.num_trees());

  // Encode the node data.
  int node_idx = 0;
  for (int tree_idx = 0; tree_idx < df_interface.num_trees(); tree_idx++) {
    const auto begin_node_idx = node_idx;
    const auto& tree = df_interface.decision_trees()[tree_idx];
    RETURN_IF_ERROR(GenRoutingModelDataNode(
        model, dataspec, stats, specialized_conversion, options,
        internal_options, tree->root(), 0, &serialized_nodes, &node_idx,
        &categorical_bank, &leaf_value_bank));
    root_deltas.push_back(node_idx - begin_node_idx);
  }

  // Record a mapping from feature to condition type. This is possible because
  // this implementation assumes that each feature is only used in one type of
  // condition (which is not generally the case in YDF).

  // TODO: Use a virtual feature index system to allow a same feature to be
  // used with different condition types.

  std::vector<uint8_t> condition_types(stats.num_features, 0);
  for (int feature_idx = 0; feature_idx < model.input_features().size();
       feature_idx++) {
    const auto& column_idx = model.input_features()[feature_idx];
    const auto& col_spec = model.data_spec().columns(column_idx);
    switch (col_spec.type()) {
      case dataset::proto::ColumnType::NUMERICAL:
        condition_types[feature_idx] =
            static_cast<uint8_t>(RoutingConditionType::HIGHER_CONDITION);
        break;
      case dataset::proto::ColumnType::CATEGORICAL:
        condition_types[feature_idx] = static_cast<uint8_t>(
            RoutingConditionType::CONTAINS_CONDITION_BUFFER_BITMAP);
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Non supported feature type: ",
                         dataset::proto::ColumnType_Name(col_spec.type())));
    }
  }

  if (stats.has_multiple_condition_types) {
    absl::SubstituteAndAppend(content, R"(
static const uint8_t condition_types[] = {$0};

)",
                              absl::StrJoin(condition_types, ","));
  }

  absl::SubstituteAndAppend(content, R"(
static const $0 root_deltas[] = {$1};

)",
                            node_offset_type, absl::StrJoin(root_deltas, ","));

  if (!categorical_bank.empty()) {
    STATUS_CHECK_LE(MaxUnsignedValueToNumBytes(categorical_bank.size()),
                    internal_options.categorical_idx_bytes);
  }

  // TODO: Add option to encode the node data by a string of bytes (more
  // compact and faster to compile, but less readable).
  absl::StrAppend(content, "static const Node nodes[] = {\n", serialized_nodes,
                  "};\n");

  // Record the categorical mask bank.
  if (internal_options.categorical_idx_bytes > 0) {
    absl::SubstituteAndAppend(
        content, R"(
  static const std::bitset<$0> categorical_bank {"$1"};
  )",
        categorical_bank.size(),
        absl::StrJoin(categorical_bank.rbegin(), categorical_bank.rend(), ""));
  }

  // Record the leaf value bank.
  if (!leaf_value_bank.empty()) {
    absl::SubstituteAndAppend(content, R"(
static const float leaf_value_bank[] = {$0};
)",
                              absl::StrJoin(leaf_value_bank, ","));
  }

  return absl::OkStatus();
}

absl::Status SpecializedConversion::Validate() const {
  STATUS_CHECK(!accumulator_type.empty());
  STATUS_CHECK(!accumulator_initial_value.empty());
  STATUS_CHECK(!return_prediction.empty());
  STATUS_CHECK(!accumulator_type.empty());
  STATUS_CHECK_GT(leaf_value_spec.dims, 0);
  STATUS_CHECK_NE(leaf_value_spec.dtype, proto::DType::UNDEFINED);

  STATUS_CHECK(set_node_ifelse_fn);
  STATUS_CHECK(leaf_value_fn);
  STATUS_CHECK(!routing_node.empty());
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace yggdrasil_decision_forests::serving::embed
