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
#include <cstdint>
#include <functional>
#include <optional>
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

// Generates the struct for a single instance (i.e., an example without a
// label).
absl::StatusOr<std::string> GenInstanceStruct(
    const model::AbstractModel& model, const proto::Options& options,
    const internal::InternalOptions& internal_options) {
  std::string content;

  // Start
  absl::SubstituteAndAppend(&content, R"(
constexpr const int kNumFeatures = $0;

struct Instance {
)",
                            model.input_features().size());

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

}  // namespace

absl::StatusOr<absl::node_hash_map<Filename, Content>> EmbedModelCC(
    const model::AbstractModel& model, const proto::Options& options) {
  const auto* df_interface =
      dynamic_cast<const model::DecisionForestInterface*>(&model);
  if (!df_interface) {
    return absl::InvalidArgumentError(
        "The model is not a decision forest model.");
  }

  RETURN_IF_ERROR(CheckModelName(options.name()));

  ASSIGN_OR_RETURN(const internal::ModelStatistics stats,
                   internal::ComputeStatistics(model, *df_interface));

  ASSIGN_OR_RETURN(
      const internal::InternalOptions internal_options,
      internal::ComputeInternalOptions(model, *df_interface, stats, options));

  absl::node_hash_map<Filename, Content> result;

  std::string header;

  // Open define and namespace.
  absl::SubstituteAndAppend(&header, R"(#ifndef YDF_MODEL_$0_H_
#define YDF_MODEL_$0_H_

#include <stdint.h>
)",
                            StringToConstantSymbol(options.name()));

  if (internal_options.include_array) {
    absl::StrAppend(&header, "#include <array>\n");
  }
  if (internal_options.include_algorithm) {
    absl::StrAppend(&header, "#include <algorithm>\n");
  }

  absl::SubstituteAndAppend(&header, R"(
namespace $0 {
)",
                            StringToVariableSymbol(options.name()));

  // Instance struct.
  ASSIGN_OR_RETURN(const auto instance_struct,
                   GenInstanceStruct(model, options, internal_options));
  absl::StrAppend(&header, instance_struct);

  // Predict method
  absl::SubstituteAndAppend(&header, R"(
inline $0 Predict(const Instance& instance) {
)",
                            internal_options.output_type);

  const auto model_gbt = dynamic_cast<
      const model::gradient_boosted_trees::GradientBoostedTreesModel*>(&model);
  const auto model_rf =
      dynamic_cast<const model::random_forest::RandomForestModel*>(&model);
  if (model_gbt) {
    RETURN_IF_ERROR(internal::GenPredictionGBT(
        *model_gbt, stats, internal_options, options, &header));
  } else if (model_rf) {
    RETURN_IF_ERROR(internal::GenPredictionRF(
        *model_rf, stats, internal_options, options, &header));
  } else {
    return absl::InvalidArgumentError("The model type is not supported.");
  }

  absl::StrAppend(&header, "}\n");

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

AccumulatorDef GenAccumulatorDef(const proto::Options& options,
                                 const ModelStatistics& stats) {
  AccumulatorDef accumulator_def;
  // Base type.
  if (options.integerize_output()) {
    if (stats.leaf_output_is_signed) {
      accumulator_def.base_type =
          SignedInteger(options.accumulator_precision_bytes());
    } else {
      accumulator_def.base_type =
          UnsignedInteger(options.accumulator_precision_bytes());
    }
  } else {
    accumulator_def.base_type = "float";
  }

  DCHECK_GE(stats.internal_output_dim, 1);
  if (stats.internal_output_dim == 1) {
    accumulator_def.type = accumulator_def.base_type;
  } else {
    accumulator_def.use_array = true;
    accumulator_def.type =
        absl::StrCat("std::array<", accumulator_def.base_type, ", ",
                     stats.internal_output_dim, ">");
  }

  // TODO: Compute coefficient.
  return accumulator_def;
}

absl::Status GenPredictionGBT(
    const model::gradient_boosted_trees::GradientBoostedTreesModel& model,
    const ModelStatistics& stats, const InternalOptions& internal_options,
    const proto::Options& options, std::string* content) {
  const auto acc_def = GenAccumulatorDef(options, stats);

  // Accumulator
  const auto& initial_predictions = model.initial_predictions();
  std::string accumulator_initial_value;
  // TODO: Handle coefficient.
  accumulator_initial_value = absl::StrJoin(initial_predictions, ", ");
  absl::SubstituteAndAppend(content, "  $0 accumulator {$1};\n", acc_def.type,
                            accumulator_initial_value);

  // Task / loss specifics.
  std::string return_accumulator;
  IfElseSetNodeFn set_node_fn;

  switch (model.task()) {
    case model::proto::Task::CLASSIFICATION: {
      // Leaf setter
      if (stats.internal_output_dim == 1) {
        set_node_fn =
            [](const model::decision_tree::proto::Node& node, const int depth,
               const int tree_idx,
               absl::string_view prefix) -> absl::StatusOr<std::string> {
          const float node_value = node.regressor().top_value();
          return absl::StrCat(prefix, "accumulator += ", node_value, ";\n");
        };
      } else {
        set_node_fn =
            [&](const model::decision_tree::proto::Node& node, const int depth,
                const int tree_idx,
                absl::string_view prefix) -> absl::StatusOr<std::string> {
          const float node_value = node.regressor().top_value();
          const int output_dim_idx = tree_idx % stats.internal_output_dim;
          return absl::StrCat(prefix, "accumulator[", output_dim_idx,
                              "] += ", node_value, ";\n");
        };
      }
      // Return accumulator
      switch (options.classification_output()) {
        case proto::ClassificationOutput::CLASS:
          if (acc_def.use_array) {
            absl::StrAppend(
                &return_accumulator,
                "  return std::distance(accumulator.begin(), "
                "std::max_element(accumulator.begin(), accumulator.end()));\n");
          } else {
            absl::StrAppend(&return_accumulator,
                            "  return accumulator >= 0;\n");
          }
          break;
        case proto::ClassificationOutput::SCORE:
          absl::StrAppend(&return_accumulator, "  return accumulator;\n");
          break;
        case proto::ClassificationOutput::PROBABILITY:
          if (acc_def.use_array) {
            absl::SubstituteAndAppend(&return_accumulator, R"(
  // Softmax
  std::array<float,$0> probas;
  const float max_logit = *std::max_element(accumulator.begin(), accumulator.end());
  float sum_exps = 0.f;
  for(int i=0;i<$0;i++){ probas[i] = std::exp(x - max_logit); sum_exps+= probas[i]; }
  for(int i=0;i<$0;i++){ probas[i] /= sum_exps; }
  return probas;
)",
                                      stats.internal_output_dim);
          } else {
            absl::StrAppend(&return_accumulator, R"(
  // Sigmoid
  return std::clamp(
      1.f / (1.f + std::exp(-accumulator)), 0.f, 1.f)
)");
          }
          break;
      }
    } break;

    case model::proto::Task::REGRESSION:
      absl::StrAppend(&return_accumulator, "  return accumulator;\n");
      set_node_fn =
          [](const model::decision_tree::proto::Node& node, const int depth,
             const int tree_idx,
             absl::string_view prefix) -> absl::StatusOr<std::string> {
        const float node_value = node.regressor().top_value();
        return absl::StrCat(prefix, "accumulator += ", node_value, ";\n");
      };
      break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Non supported task: ", model::proto::Task_Name(model.task())));
  }

  // Accumulate leaf values
  DCHECK_EQ(options.algorithm(), proto::Algorithm::IF_ELSE);
  RETURN_IF_ERROR(GenerateTreeInferenceIfElse(model.data_spec(), model, options,
                                              internal_options, set_node_fn,
                                              content));

  // Accumulator to predictions.
  absl::StrAppend(content, return_accumulator);

  return absl::OkStatus();
}

absl::Status GenPredictionRF(
    const model::random_forest::RandomForestModel& model,
    const ModelStatistics& stats, const InternalOptions& internal_options,
    const proto::Options& options, std::string* content) {
  absl::StrAppend(content, R"(return {};
)");
  return absl::OkStatus();
}

absl::StatusOr<ModelStatistics> ComputeStatistics(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface) {
  ModelStatistics stats{
      .num_trees = df_interface.num_trees(),
      .num_features = static_cast<int>(model.input_features().size()),
  };

  const bool is_classification =
      model.task() == model::proto::Task::CLASSIFICATION;
  const int num_classification_classes =
      model.LabelColumnSpec().categorical().number_of_unique_values() - 1;
  const bool is_binary_classification =
      is_classification && num_classification_classes == 2;

  // Model specific statistics.
  const auto model_gbt = dynamic_cast<
      const model::gradient_boosted_trees::GradientBoostedTreesModel*>(&model);
  const auto model_rf =
      dynamic_cast<const model::random_forest::RandomForestModel*>(&model);

  // TODO: Handle integerized leaf values.
  std::optional<
      std::function<double(const model::decision_tree::proto::Node& node)>>
      get_max_abs_output;

  if (model_gbt) {
    stats.multi_dim_tree = false;  // GBT trees are single dimensional.
    get_max_abs_output =
        [](const model::decision_tree::proto::Node& node) -> double {
      return std::abs(node.regressor().top_value());
    };
  } else if (model_rf) {
    stats.multi_dim_tree = true;  // RF trees are multidimensional.
    if (model.task() == model::proto::Task::REGRESSION) {
      get_max_abs_output =
          [](const model::decision_tree::proto::Node& node) -> double {
        return std::abs(node.regressor().top_value());
      };
    }
  } else {
    return absl::InvalidArgumentError("The model type is not supported.");
  }

  if (is_classification) {
    DCHECK_GE(num_classification_classes, 2);
    if (is_binary_classification) {
      stats.internal_output_dim = 1;
    } else {
      stats.internal_output_dim = num_classification_classes;
    }
  } else {
    stats.internal_output_dim = 1;
  }

  // Scan the trees
  for (const auto& tree : df_interface.decision_trees()) {
    int64_t num_leaves_in_tree = 0;
    double max_abs_output_in_tree = 0;
    tree->IterateOnNodes([&](const model::decision_tree::NodeWithChildren& node,
                             int depth) {
      stats.max_depth = std::max(stats.max_depth, static_cast<int64_t>(depth));
      if (node.IsLeaf()) {
        num_leaves_in_tree++;
        stats.num_leaves++;
        if (get_max_abs_output.has_value()) {
          const auto node_max_abs_output =
              get_max_abs_output.value()(node.node());
          stats.max_abs_output =
              std::max(stats.max_abs_output, node_max_abs_output);
          max_abs_output_in_tree =
              std::max(max_abs_output_in_tree, node_max_abs_output);
        }
      } else {
        stats.has_conditions[node.node().condition().condition().type_case()] =
            true;
      }
    });
    stats.max_num_leaves_per_tree =
        std::max(stats.max_num_leaves_per_tree, num_leaves_in_tree);
    stats.sum_max_abs_output += max_abs_output_in_tree;
  }
  return stats;
}

absl::StatusOr<InternalOptions> ComputeInternalOptions(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface,
    const ModelStatistics& stats, const proto::Options& options) {
  InternalOptions internal_options;
  RETURN_IF_ERROR(
      ComputeInternalOptionsFeature(model, options, &internal_options));
  RETURN_IF_ERROR(
      ComputeInternalOptionsOutput(model, stats, options, &internal_options));
  return internal_options;
}

absl::Status ComputeInternalOptionsFeature(const model::AbstractModel& model,
                                           const proto::Options& options,
                                           InternalOptions* out) {
  out->feature_value_bytes = 1;
  out->numerical_feature_is_float = false;

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

absl::Status ComputeInternalOptionsOutput(const model::AbstractModel& model,
                                          const ModelStatistics& stats,
                                          const proto::Options& options,
                                          InternalOptions* out) {
  if (stats.has_conditions
          [model::decision_tree::proto::Condition::kContainsBitmapCondition] ||
      stats.has_conditions
          [model::decision_tree::proto::Condition::kContainsCondition]) {
    out->include_algorithm = true;
    out->include_array = true;
  }

  switch (model.task()) {
    case model::proto::Task::CLASSIFICATION: {
      const int num_classes =
          model.LabelColumnSpec().categorical().number_of_unique_values() - 1;
      switch (options.classification_output()) {
        case proto::ClassificationOutput::CLASS:
          if (num_classes == 2) {
            out->output_type = "bool";
          } else {
            out->output_type =
                UnsignedInteger(MaxUnsignedValueToNumBytes(num_classes));
            out->include_algorithm = true;
          }
          break;
        case proto::ClassificationOutput::SCORE: {
          const auto acc_def = GenAccumulatorDef(options, stats);
          out->output_type = acc_def.type;
          if (acc_def.use_array) {
            out->include_array = true;
          }
        } break;
        case proto::ClassificationOutput::PROBABILITY:
          if (num_classes == 2) {
            out->output_type = "float";
          } else {
            out->include_array = true;
            out->output_type =
                absl::StrCat("std::array<float, ", num_classes, ">");
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
          "Non supported task: ", model::proto::Task_Name(model.task())));
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
    const IfElseSetNodeFn& set_node_fn, const int tree_idx,
    std::string* content) {
  std::string prefix(depth * 2 + 2, ' ');

  if (node.IsLeaf()) {
    // The leaf value
    ASSIGN_OR_RETURN(const auto leaf,
                     set_node_fn(node.node(), depth, tree_idx, prefix));
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
  RETURN_IF_ERROR(GenerateTreeInferenceIfElseNode(
      dataspec, *node.pos_child(), depth + 1, set_node_fn, tree_idx, content));
  absl::StrAppend(content, prefix, "} else {\n");
  RETURN_IF_ERROR(GenerateTreeInferenceIfElseNode(
      dataspec, *node.neg_child(), depth + 1, set_node_fn, tree_idx, content));
  absl::StrAppend(content, prefix, "}\n");
  return absl::OkStatus();
};

absl::Status GenerateTreeInferenceIfElse(
    const dataset::proto::DataSpecification& dataspec,
    const model::DecisionForestInterface& df_interface,
    const proto::Options& options, const InternalOptions& internal_options,
    const IfElseSetNodeFn& set_node_fn, std::string* content) {
  for (int tree_idx = 0; tree_idx < df_interface.num_trees(); tree_idx++) {
    absl::StrAppend(content, "  // Tree #", tree_idx, "\n");
    const auto& tree = df_interface.decision_trees()[tree_idx];
    RETURN_IF_ERROR(GenerateTreeInferenceIfElseNode(
        dataspec, tree->root(), 0, set_node_fn, tree_idx, content));
    absl::StrAppend(content, "\n");
  }
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace yggdrasil_decision_forests::serving::embed
