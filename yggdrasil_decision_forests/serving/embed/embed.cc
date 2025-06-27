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
#include <cstdint>
#include <functional>
#include <memory>
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
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
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

  // Start
  absl::SubstituteAndAppend(&content, R"(
constexpr const int kNumFeatures = $0;
constexpr const int kNumTrees = $1;

struct Instance {
)",
                            model.input_features().size(), stats.num_trees);

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
    absl::SubstituteAndAppend(&content, R"(
struct $0$1 {
  enum {
)",
                              dict.second.is_label ? "" : "Feature",
                              dict.second.sanitized_name);
    // Create the enum values
    for (int item_idx = 0; item_idx < dict.second.sanitized_items.size();
         item_idx++) {
      absl::SubstituteAndAppend(&content, "    k$0 = $1,\n",
                                dict.second.sanitized_items[item_idx],
                                item_idx);
    }
    absl::StrAppend(&content, R"(  };
};
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
    STATUS_CHECK(!specialized_conversion.accumulator_type.empty());
    STATUS_CHECK(!specialized_conversion.accumulator_initial_value.empty());
    STATUS_CHECK(!specialized_conversion.return_prediction.empty());
    STATUS_CHECK(!specialized_conversion.accumulator_type.empty());
    STATUS_CHECK_GT(specialized_conversion.leaf_value_spec.dims, 0);
    STATUS_CHECK_NE(specialized_conversion.leaf_value_spec.dtype,
                    proto::DType::UNDEFINED);
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

  absl::SubstituteAndAppend(&header, R"(
namespace $0 {
)",
                            StringToVariableSymbol(options.name()));

  // Instance struct.
  ASSIGN_OR_RETURN(const auto instance_struct,
                   GenInstanceStruct(model, options, internal_options, stats));
  absl::StrAppend(&header, instance_struct);

  // Categorical dictionary
  ASSIGN_OR_RETURN(
      const auto categorical_dict,
      GenCategoricalStringDictionaries(model, options, internal_options));
  absl::StrAppend(&header, categorical_dict);

  // Model data
  if (options.algorithm() == proto::Algorithm::ROUTING) {
    RETURN_IF_ERROR(internal::GenRoutingModelData(
        stats, specialized_conversion, options, internal_options, &header));
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
          specialized_conversion.set_node_ifelse_fn, content));
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
        }
      }

      // Return accumulator
      switch (options.classification_output()) {
        case proto::ClassificationOutput::CLASS:
          if (stats.is_binary_classification()) {
            spec.return_prediction = absl::Substitute(
                "  return accumulator >= $0;\n", stats.num_trees / 2);
          } else {
            absl::StrAppend(&spec.return_prediction,
                            "  return std::distance(accumulator.begin(), "
                            "std::max_element(accumulator.begin(), "
                            "accumulator.end()));\n");
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
      }
      // Return accumulator
      switch (options.classification_output()) {
        case proto::ClassificationOutput::CLASS:
          if (stats.is_binary_classification()) {
            spec.return_prediction = "  return accumulator >= 0;\n";
          } else {
            spec.return_prediction =
                ("  return std::distance(accumulator.begin(), "
                 "std::max_element(accumulator.begin(), "
                 "accumulator.end()));\n");
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
      }
    });
    stats.max_num_leaves_per_tree =
        std::max(stats.max_num_leaves_per_tree, num_leaves_in_tree);
  }
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

  // This is the number of bytes to encode a node index in a tree. The precision
  // for an offset is in average 50% smaller.
  // TODO: Optimize node_offset_bytes.
  out->node_offset_bytes = MaxUnsignedValueToNumBytes(
      NumLeavesToNumNodes(stats.max_num_leaves_per_tree));

  // Feature encoding.
  for (const auto input_feature : model.input_features()) {
    const auto& col_spec = model.data_spec().columns(input_feature);
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
          if (stats.is_binary_classification()) {
            out->output_type = "bool";
          } else {
            out->output_type = UnsignedInteger(
                MaxUnsignedValueToNumBytes(stats.num_classification_classes));
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
  // TODO: Add support for string categorical features.
  // TODO: For integer numericals, use the min/max to possibly reduce the
  // required precision.
  switch (col.type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (internal_options.numerical_feature_is_float) {
        DCHECK_EQ(internal_options.feature_value_bytes, 4);
        return FeatureDef{.type = "float", .default_value = {}};
      } else {
        return FeatureDef{
            .type = SignedInteger(internal_options.feature_value_bytes),
            .default_value = {}};
      }
    } break;
    case dataset::proto::ColumnType::BOOLEAN:
    case dataset::proto::ColumnType::CATEGORICAL:
      return FeatureDef{
          .type = UnsignedInteger(internal_options.feature_value_bytes),
          .default_value = {}};
      break;
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
    std::string* content) {
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
      [&](absl::string_view variable_name, absl::Span<const int32_t> elements) {
        // TODO: Use constants (e.g. kFeatureABC) instead of raw integers
        // if the column has a dictionary. elements is large.
        if (elements.size() < 8) {
          // List the elements are a sequence of ==.
          for (int element_idx = 0; element_idx < elements.size();
               element_idx++) {
            if (element_idx > 0) {
              absl::StrAppend(&condition, " ||\n", prefix, "    ");
            }
            absl::SubstituteAndAppend(&condition, "instance.$0 == $1",
                                      variable_name, elements[element_idx]);
          }
        } else {
          // Use binary search.
          absl::SubstituteAndAppend(
              &condition,
              "std::array<uint32_t,$0> mask = {$1};\n$3    "
              "std::binary_search(mask.begin(), mask.end(),  instance.$2)",
              elements.size(), absl::StrJoin(elements, ", "), variable_name,
              prefix);
        }
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
      categorical_contains_condition(variable_name, typed_condition.elements());
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
      categorical_contains_condition(variable_name, elements);
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
  RETURN_IF_ERROR(GenerateTreeInferenceIfElseNode(dataspec, *node.pos_child(),
                                                  depth + 1, set_node_ifelse_fn,
                                                  tree_idx, content));
  absl::StrAppend(content, prefix, "} else {\n");
  RETURN_IF_ERROR(GenerateTreeInferenceIfElseNode(dataspec, *node.neg_child(),
                                                  depth + 1, set_node_ifelse_fn,
                                                  tree_idx, content));
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
        dataspec, tree->root(), 0, set_node_ifelse_fn, tree_idx, content));
    absl::StrAppend(content, "\n");
  }
  return absl::OkStatus();
}

absl::Status GenerateTreeInferenceRouting(
    const dataset::proto::DataSpecification& dataspec,
    const model::DecisionForestInterface& df_interface,
    const proto::Options& options, const InternalOptions& internal_options,
    const IfElseSetNodeFn& set_node_ifelse_fn, std::string* content) {
  // TODO: Implement.
  return absl::OkStatus();
}

absl::Status GenRoutingModelData(
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

  std::string node_value_type;
  if (specialized_conversion.leaf_value_spec.dims == 1) {
    // The leaf value is stored in the node struct.
    node_value_type =
        DTypeToCCType(specialized_conversion.leaf_value_spec.dtype);
  } else {
    // The leaf values are stored in a separate buffer. The node struct contains
    // an index to this buffer.
    node_value_type =
        UnsignedInteger(MaxUnsignedValueToNumBytes(stats.num_leaves));
  }

  // TODO: Add categorical and boolean condition support.

  absl::SubstituteAndAppend(content, R"(
struct __attribute__((packed)) Node {
  $1 feature;
  union {
    struct {
      $0 threshold;
      $2 pos_child_offset;
    } cond;
    struct {
      $3 value;
    } leaf;
  };
};
)",
                            is_greather_threshold_type,  // $0
                            feature_index_type,          // $1
                            node_offset_type,            // $2
                            node_value_type              // $3
  );

  // TODO: Implement.
  absl::StrAppend(content, R"(
static const uint8_t nodes[] = {};
)");

  // TODO: Implement.
  absl::SubstituteAndAppend(content, R"(
static const $0 roots[] = {};
)",
                            node_index_type);

  return absl::OkStatus();
}

}  // namespace internal
}  // namespace yggdrasil_decision_forests::serving::embed
