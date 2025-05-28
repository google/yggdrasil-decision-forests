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
#include <cctype>
#include <cstdint>
#include <string>

#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed {

namespace {

// Integer type.
std::string UnsignedInteger(int bytes) {
  return absl::StrCat("uint", bytes * 8, "_t");
}
std::string SignedInteger(int bytes) {
  return absl::StrCat("int", bytes * 8, "_t");
}

// Converts any string into a snake case symbol.
std::string StringToSnakeCaseSymbol(const std::string_view input,
                                    const bool to_upper,
                                    const char prefix_char_if_digit) {
  if (input.empty()) {
    return "";
  }

  std::string result;
  result.reserve(input.size());
  bool last_char_was_separator = true;
  bool first_char = true;

  for (const char ch : input) {
    if (std::isalnum(ch)) {
      if (std::isdigit(ch) && first_char) {
        // Add a prefix if the first character is a number.
        result.push_back(prefix_char_if_digit);
      }
      // Change the case of letters.
      if (to_upper) {
        result.push_back(std::toupper(ch));
      } else {
        result.push_back(std::tolower(ch));
      }
      last_char_was_separator = false;
    } else if (ch == ' ' || ch == '-' || ch == '_') {
      // Characters that are replaced with "_".
      if (!result.empty() && !last_char_was_separator) {
        result.push_back('_');
        last_char_was_separator = true;
      }
    }
    // Other characters are skipped.
    first_char = false;
  }

  // Remove the last character if it is a separator.
  if (!result.empty() && result.back() == '_') {
    result.pop_back();
  }
  return result;
}

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
    const std::string variable_name =
        internal::StringToVariableSymbol(col.name());
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

  RETURN_IF_ERROR(internal::CheckModelName(options.name()));

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
                            internal::StringToConstantSymbol(options.name()));

  if (internal_options.include_array) {
    absl::StrAppend(&header, "#include <array>\n");
  }

  absl::SubstituteAndAppend(&header, R"(
namespace $0 {
)",
                            internal::StringToVariableSymbol(options.name()));

  // Instance struct.
  ASSIGN_OR_RETURN(const auto instance_struct,
                   GenInstanceStruct(model, options, internal_options));
  absl::StrAppend(&header, instance_struct);

  // Predict method
  absl::SubstituteAndAppend(&header, R"(
inline $0 Predict(const Instance& instance) {
  return {};
}
)",
                            internal_options.output_type);

  // Close define and namespace.
  absl::SubstituteAndAppend(&header, R"(
}  // namespace $0
#endif
)",
                            internal::StringToVariableSymbol(options.name()));

  result[absl::StrCat(options.name(), ".h")] = header;
  return result;
}

namespace internal {

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

  if (model_gbt) {
    stats.multi_dim_tree = false;  // GBT trees are single dimensional.
  } else if (model_rf) {
    stats.multi_dim_tree = true;  // RF trees are multidimensional.
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
    tree->IterateOnNodes([&](const model::decision_tree::NodeWithChildren& node,
                             int depth) {
      stats.max_depth = std::max(stats.max_depth, static_cast<int64_t>(depth));
      if (node.IsLeaf()) {
        num_leaves_in_tree++;
        stats.num_leaves++;
      }
    });
    stats.max_num_leaves_per_tree =
        std::max(stats.max_num_leaves_per_tree, num_leaves_in_tree);
  }
  return stats;
}

absl::Status CheckModelName(absl::string_view value) {
  for (const char c : value) {
    if (!std::islower(c) && !std::isdigit(c) && c != '_') {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid model name: ", value,
                       ". The model name can only contain lowercase letters, "
                       "numbers, and _."));
    }
  }
  return absl::OkStatus();
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
          }
          break;
        case proto::ClassificationOutput::SCORE: {
          std::string base_output_type;
          if (options.integerize_output()) {
            if (stats.leaf_output_is_signed) {
              base_output_type =
                  SignedInteger(options.accumulator_precision_bytes());
            } else {
              base_output_type =
                  UnsignedInteger(options.accumulator_precision_bytes());
            }
          } else {
            base_output_type = "float";
          }
          if (num_classes == 2) {
            out->output_type = base_output_type;
          } else {
            out->include_array = true;
            out->output_type = absl::StrCat("std::array<", base_output_type,
                                            ", ", num_classes, ">");
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

std::string StringToConstantSymbol(const absl::string_view input) {
  return StringToSnakeCaseSymbol(input, true, 'V');
}

std::string StringToVariableSymbol(const absl::string_view input) {
  return StringToSnakeCaseSymbol(input, false, 'v');
}

std::string StringToStructSymbol(const absl::string_view input) {
  if (input.empty()) {
    return "";
  }

  std::string result;
  result.reserve(input.size());
  bool capitalize_next_char = true;
  bool current_word_started_with_letter = false;
  bool first_char = true;

  for (const char ch : input) {
    if (std::isalnum(ch)) {
      if (std::isdigit(ch) && first_char) {
        // Add a prefix if the first character is a number.
        result.push_back('V');
      }
      // Change the case of letters.
      if (capitalize_next_char) {
        result.push_back(std::toupper(ch));
        current_word_started_with_letter = std::isalpha(ch);
        capitalize_next_char = false;
      } else {
        if (current_word_started_with_letter && std::isalpha(ch)) {
          result.push_back(std::tolower(ch));
        } else {
          result.push_back(ch);
        }
      }
    } else {
      capitalize_next_char = true;
    }
    // Other characters are skipped.
    first_char = false;
  }

  // Remove the last character if it is a separator.
  if (!result.empty() && result.back() == '_') {
    result.pop_back();
  }
  return result;
}

int MaxUnsignedValueToNumBytes(uint32_t value) {
  if (value <= 0xff) {
    return 1;
  } else if (value <= 0xffff) {
    return 2;
  } else {
    return 4;
  }
}

int MaxSignedValueToNumBytes(int32_t value) {
  if (value <= 0x7f && value >= -0x80) {
    return 1;
  } else if (value <= 0x7fff && value >= -0x8000) {
    return 2;
  } else {
    return 4;
  }
}

std::string DTypeToCCType(const proto::DType::Enum value) {
  switch (value) {
    case proto::DType::INT8:
      return "int8_t";
    case proto::DType::INT16:
      return "int16_t";
    case proto::DType::INT32:
      return "int32_t";
    case proto::DType::FLOAT32:
      return "float";
  }
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

}  // namespace internal
}  // namespace yggdrasil_decision_forests::serving::embed
