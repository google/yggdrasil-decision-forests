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

#include "yggdrasil_decision_forests/serving/embed/c/c_target_lowering.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/serving/embed/c/c_ir.h"
#include "yggdrasil_decision_forests/serving/embed/common.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/serving/embed/ir/model_ir.h"
#include "yggdrasil_decision_forests/serving/embed/utils.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed::internal {
namespace {
// Helper to transparently handle floating point vs fixed-point generation
absl::StatusOr<std::string> FormatValue(DoubleOrInt64 val, bool is_float,
                                        const proto::Options& options) {
  if (options.c().linux_kernel_compatible() && is_float) {
    double double_val = AsDouble(val);

    int64_t fp_val = std::round(
        double_val * (1ULL << options.c().fixed_point_fractional_bits()));
    // Catch the overflow before writing bad C code
    if (fp_val > std::numeric_limits<int32_t>::max() ||
        fp_val < std::numeric_limits<int32_t>::min()) {
      return absl::OutOfRangeError(absl::Substitute(
          "Fixed point value $0 (from float $1) exceeds s32 limits. "
          "Decrease 'fixed_point_fractional_bits' in your export options.",
          fp_val, double_val));
    }
    return absl::StrCat(fp_val);
  }
  return FormatExampleLiteral(val, is_float);
}
}  // namespace

absl::StatusOr<CIR> CTargetLowering::Lower(const ModelIR& model_ir,
                                           const proto::Options& options) {
  CTargetLowering lowerer(model_ir, options);
  ASSIGN_OR_RETURN(CIR c_ir, lowerer.Run());
  return c_ir;
}

absl::StatusOr<CIR> CTargetLowering::Run() {
  if (options_.algorithm() != proto::Algorithm::ROUTING) {
    return absl::InvalidArgumentError(
        "Export to C is only implemented for algorithm ROUTING");
  }
  if (options_.categorical_from_string()) {
    return absl::InvalidArgumentError(
        "Generating categorical features from string is not supported for C "
        "export");
  }

  if (options_.c().has_fixed_point_fractional_bits()) {
    if (!options_.c().linux_kernel_compatible()) {
      return absl::InvalidArgumentError(
          "Fixed-point export is currently only supported for "
          "kernel-compatible "
          "export");
    }
    if (options_.c().fixed_point_fractional_bits() < 0 ||
        options_.c().fixed_point_fractional_bits() > 31) {
      return absl::InvalidArgumentError(
          "fixed_point_fractional_bits must be between 0 and 31");
    }
  }

  if (options_.c().linux_kernel_compatible()) {
    if (std::find(model_ir_.active_condition_types.begin(),
                  model_ir_.active_condition_types.end(),
                  ConditionType::OBLIQUE_CONDITION) !=
        model_ir_.active_condition_types.end())
      LOG(WARNING) << "Kernel-mode export with oblique conditions is fragile, "
                      "consider training without oblique conditions";
  }

  RETURN_IF_ERROR(LowerGlobalFormatting());

  // Step 2: Translate Variables & Types
  RETURN_IF_ERROR(LowerEnumsAndFeatures());

  // Step 3: Accumulator & Activation
  RETURN_IF_ERROR(LowerInferenceEngine());

  // Step 4: Tree Nodes
  RETURN_IF_ERROR(LowerNodes());

  RETURN_IF_ERROR(LowerRoutingData());

  return std::move(c_ir_);
}

absl::Status CTargetLowering::LowerGlobalFormatting() {
  c_ir_.header_filename = absl::StrCat(options_.name(), ".h");
  c_ir_.source_filename = absl::StrCat(options_.name(), ".c");
  c_ir_.pseudo_namespace_name =
      absl::StrCat(StringToStructSymbol(options_.name()), "_");
  c_ir_.header_guard = absl::Substitute(
      "YDF_MODEL_$0_H_", StringToConstantSymbol(options_.name()));
  c_ir_.impl_guard = absl::Substitute("YDF_MODEL_$0_IMPL_",
                                      StringToConstantSymbol(options_.name()));
  c_ir_.num_trees = model_ir_.num_trees;
  if (!options_.c().linux_kernel_compatible()) {
    c_ir_.header_includes.insert("<stdint.h>");
    c_ir_.impl_includes.insert("<stdint.h>");
    c_ir_.impl_includes.insert("<assert.h>");
    c_ir_.impl_includes.insert("<stddef.h>");
  } else {
    // Defines offsetof and BUG()
    c_ir_.header_includes.insert("<linux/types.h>");
    c_ir_.impl_includes.insert("<linux/stddef.h>");
    c_ir_.impl_includes.insert("<linux/types.h>");
    c_ir_.impl_includes.insert("<linux/bug.h>");
  }

  ASSIGN_OR_RETURN(c_ir_.types, BuildTypes(options_, model_ir_,
                                           c_ir_.pseudo_namespace_name));

  return absl::OkStatus();
}

absl::Status CTargetLowering::LowerEnumsAndFeatures() {
  absl::flat_hash_set<std::string> sanitized_feature_names;
  absl::flat_hash_set<std::string> sanitized_categorical_type_names;

  const size_t num_features = model_ir_.features.size();

  std::vector<int> condition_types_bank;
  condition_types_bank.reserve(num_features);

  // Generate the Features.
  c_ir_.features.reserve(num_features);
  for (size_t i = 0; i < num_features; ++i) {
    RETURN_IF_ERROR(LowerFeature(i, sanitized_feature_names,
                                 sanitized_categorical_type_names,
                                 condition_types_bank));
  }

  // Generate the Enums.
  for (size_t i = 0; i < num_features; ++i) {
    RETURN_IF_ERROR(
        LowerEnum(i, model_ir_.features[i], sanitized_categorical_type_names));
  }

  c_ir_.condition_types_content = absl::StrJoin(condition_types_bank, ",");
  if (condition_types_bank.size() > std::numeric_limits<uint8_t>::max()) {
    return absl::InvalidArgumentError(
        "The number of condition types exceeds the maximum value of uint8_t.");
  }

  return absl::OkStatus();
}

absl::Status CTargetLowering::LowerFeature(
    const int feature_idx,
    absl::flat_hash_set<std::string>& sanitized_feature_names,
    absl::flat_hash_set<std::string>& sanitized_categorical_type_names,
    std::vector<int>& condition_types_bank) {
  STATUS_CHECK(!c_ir_.pseudo_namespace_name.empty());
  const auto& feature = model_ir_.features[feature_idx];
  // Labels are not features.
  if (feature.is_label) {
    return absl::OkStatus();
  }

  const std::string var_name = StringToVariableSymbol(feature.original_name);
  RETURN_IF_ERROR(CheckFeatureNameCollision(var_name, sanitized_feature_names,
                                            model_ir_.features));

  CFeature c_feature;
  c_feature.var_name = var_name;

  // Track the mapping from the IR index to the C variable name
  feat_idx_to_var_name_[feature_idx] = var_name;

  switch (feature.type) {
    case FeatureInfo::Type::kNumerical: {
      c_feature.c_type = AddPseudoNamespace("Numerical");
      c_ir_.has_numercial_feature = true;
      if (feature.is_float) {
        if (!options_.c().linux_kernel_compatible()) {
          STATUS_CHECK_EQ(model_ir_.feature_value_bytes, 4);
          if (!feature.na_replacement.has_value()) {
            return absl::InvalidArgumentError(absl::Substitute(
                "Missing NA replacement value for feature $0", var_name));
          }
          ASSIGN_OR_RETURN(const std::string un,
                           FormatExampleLiteral(*feature.na_replacement, true));
          c_ir_.impl_includes.insert("<math.h>");
          c_feature.na_sanitization = {
              absl::Substitute("sanitized_instance.$0 = isnan(instance->$0) ? "
                               "$1 : instance->$0;",
                               var_name, un)};
          c_ir_.needs_predict_unsafe = true;
        }
      }
      condition_types_bank.push_back(
          static_cast<int>(ConditionType::HIGHER_CONDITION));
    } break;
    case FeatureInfo::Type::kCategorical: {
      const std::string categorical_type = AddPseudoNamespace(
          absl::StrCat("Feature", StringToStructSymbol(feature.original_name)));
      if (!sanitized_categorical_type_names.insert(categorical_type).second) {
        return absl::InvalidArgumentError(absl::Substitute(
            "Feature name clash on categorical feature $0, consider "
            "renaming your features.",
            categorical_type));
      }
      c_feature.c_type = categorical_type;
      condition_types_bank.push_back(
          static_cast<int>(ConditionType::CONTAINS_CONDITION_BUFFER_BITMAP));
      if (!options_.c().linux_kernel_compatible()) {
        c_ir_.impl_includes.insert("<stdbool.h>");
      }
    } break;
    case FeatureInfo::Type::kIntegerizedCategorical: {
      return absl::InvalidArgumentError(
          "Integerized Categorical Features are not supported for the C "
          "export.");
    } break;
    case FeatureInfo::Type::kBoolean: {
      if (!options_.c().linux_kernel_compatible()) {
        c_ir_.header_includes.insert("<stdbool.h>");
        c_ir_.impl_includes.insert("<stdbool.h>");
      }
      c_feature.c_type = c_ir_.types.boolean;
      condition_types_bank.push_back(
          static_cast<int>(ConditionType::TRUE_CONDITION));
    } break;
    default:
      return absl::InvalidArgumentError(
          absl::Substitute("Unsupported feature type for C export: $0",
                           static_cast<int>(feature.type)));
  }
  c_ir_.features.push_back(c_feature);
  return absl::OkStatus();
}

absl::Status CTargetLowering::LowerEnum(
    const int feature_idx, const FeatureInfo& feature,
    absl::flat_hash_set<std::string>& sanitized_categorical_type_names) {
  if (feature.type != FeatureInfo::Type::kCategorical) {
    return absl::OkStatus();
  }
  CEnum c_enum;

  // Set of all sanitized_items for this specific enum.
  // Used to detect and resolve duplications exactly like common.cc
  absl::flat_hash_set<std::string> sanitized_items;

  if (feature.is_label) {
    c_enum.type_name = AddPseudoNamespace("Label");
    c_enum.name = absl::StrCat(c_enum.type_name, "Enum");

    std::string enum_prefix = AddPseudoNamespace("LabelEnum_");
    // Omit the OOV item for the label
    for (int i = 1; i < feature.vocabulary.size(); ++i) {
      std::string item_symbol = absl::StrCat(
          enum_prefix, StringToStructSymbol(feature.vocabulary[i],
                                            /*ensure_letter_first=*/false));

      if (sanitized_items.contains(item_symbol)) {
        item_symbol = ResolveNameCollision(item_symbol, sanitized_items);
      }
      sanitized_items.insert(item_symbol);

      c_enum.items.push_back(
          {.name = absl::StrCat(item_symbol), .value = i - 1});
    }
    // Labels are at the top of file.
    c_ir_.enums.insert(c_ir_.enums.begin(), c_enum);
  } else {
    const std::string feature_struct_name =
        StringToStructSymbol(feature.original_name);
    c_enum.type_name =
        AddPseudoNamespace(absl::StrCat("Feature", feature_struct_name));
    c_enum.name = absl::StrCat(c_enum.type_name, "Enum");
    std::string enum_prefix = absl::StrCat(c_enum.name, "_");

    // Track the mapping from the IR index to the Enum class name
    feat_idx_to_enum_name_[feature_idx] = c_enum.name;

    if (sanitized_categorical_type_names.insert(c_enum.name).second) {
      // Collision check already handled in previous loop for unique names,
      // but double check here just to be safe.
    }

    for (int i = 0; i < feature.vocabulary.size(); ++i) {
      if (i == 0) {
        std::string ood_name = absl::StrCat(enum_prefix, "OutOfVocabulary");
        c_enum.items.push_back({.name = ood_name, .value = 0});
        // Protect "OutOfVocabulary" from being overwritten by a weird user
        // category
        sanitized_items.insert(ood_name);
      } else {
        std::string item_symbol = absl::StrCat(
            enum_prefix, StringToStructSymbol(feature.vocabulary[i],
                                              /*ensure_letter_first=*/false));

        // Exact collision resolution logic from common.cc
        if (sanitized_items.contains(item_symbol)) {
          item_symbol = ResolveNameCollision(item_symbol, sanitized_items);
        }
        sanitized_items.insert(item_symbol);

        c_enum.items.push_back({.name = item_symbol, .value = i});
      }
    }
    // Other feature enums are in arbitrary order.
    c_ir_.enums.push_back(c_enum);
  }
  return absl::OkStatus();
}

absl::Status CTargetLowering::LowerInferenceEngine() {
  RETURN_IF_ERROR(LowerAccumulator());
  RETURN_IF_ERROR(LowerActivation());
  return absl::OkStatus();
}

absl::Status CTargetLowering::LowerAccumulator() {
  const bool is_float = !model_ir_.winner_takes_all;

  // Case 1: Single output (Regression or Binary Classification)
  if (model_ir_.num_output_classes == 1) {
    ASSIGN_OR_RETURN(const std::string init_val,
                     FormatValue(model_ir_.accumulator_initialization[0],
                                 is_float, options_));
    c_ir_.accumulator_init_statement = absl::Substitute(
        "$0 accumulator = $1;", c_ir_.types.accumulator, init_val);
    c_ir_.accumulator_sum_statement = "accumulator += node->leaf.val;";
    return absl::OkStatus();
  }

  // Case 2: Multi-class
  const auto classification_output = options_.classification_output();
  // Determine if we use a local 'accumulator' or the 'out' parameter.
  const bool use_out =
      (classification_output == proto::ClassificationOutput::SCORE) ||
      (classification_output == proto::ClassificationOutput::PROBABILITY &&
       model_ir_.leaf_value_dims > 1);
  const std::string target = use_out ? "out" : "accumulator";

  std::vector<std::string> init_vals;
  for (const auto& val : model_ir_.accumulator_initialization) {
    ASSIGN_OR_RETURN(std::string s, FormatValue(val, is_float, options_));
    init_vals.push_back(std::move(s));
  }

  if (use_out) {
    c_ir_.accumulator_init_statement = "";
    for (int i = 0; i < init_vals.size(); ++i) {
      absl::SubstituteAndAppend(&c_ir_.accumulator_init_statement,
                                "  out[$0] = $1;\n", i, init_vals[i]);
    }
  } else {
    c_ir_.accumulator_init_statement = absl::Substitute(
        "$0 accumulator[$1] = {$2};", c_ir_.types.accumulator,
        model_ir_.num_output_classes, absl::StrJoin(init_vals, ", "));
  }

  if (model_ir_.leaf_value_dims == 1) {
    if (model_ir_.model_type == ModelIR::ModelType::kRandomForest &&
        (classification_output == proto::ClassificationOutput::PROBABILITY ||
         classification_output == proto::ClassificationOutput::SCORE)) {
      c_ir_.accumulator_sum_statement =
          absl::Substitute("$0[node->leaf.val]++;", target);
    } else {
      c_ir_.accumulator_sum_statement =
          absl::Substitute("$0[tree_idx % $1] += node->leaf.val;", target,
                           model_ir_.num_output_classes);
    }
  } else {
    c_ir_.accumulator_sum_statement = absl::Substitute(
        R"(const size_t offset = node->leaf.val * $0;
  for(int dim=0; dim!=$0; dim++) {
    $1[dim] += $2leaf_value_bank[offset + dim];
  })",
        model_ir_.num_output_classes, target, c_ir_.pseudo_namespace_name);
  }

  return absl::OkStatus();
}

absl::Status CTargetLowering::LowerActivation() {
  if (model_ir_.task == ModelIR::Task::kRegression) {
    c_ir_.activation_statement = "return accumulator;";
    return absl::OkStatus();
  }

  const auto classification_output = options_.classification_output();
  switch (classification_output) {
    case proto::ClassificationOutput::CLASS:
      if (model_ir_.num_output_classes == 1) {
        if (model_ir_.winner_takes_all) {
          c_ir_.activation_statement =
              absl::Substitute("return ($0)(accumulator > $1);",
                               c_ir_.types.output, model_ir_.num_trees / 2);
        } else {
          std::string threshold;
          if (model_ir_.model_type == ModelIR::ModelType::kRandomForest) {
            ASSIGN_OR_RETURN(threshold, FormatValue(1.f, true, options_));
          } else if (model_ir_.model_type ==
                     ModelIR::ModelType::kGradientBoostedTrees) {
            ASSIGN_OR_RETURN(threshold, FormatValue(0.0f, true, options_));
          } else {
            return absl::InvalidArgumentError("Invalid model type");
          }
          c_ir_.activation_statement = absl::Substitute(
              "return ($0)(accumulator >= $1);", c_ir_.types.output, threshold);
        }
      } else {
        c_ir_.activation_statement = absl::Substitute(
            R"(int best_class = 0;
  $0 max_val = accumulator[0];
  for (int i = 1; i < $1; i++) {
    if (accumulator[i] > max_val) {
      max_val = accumulator[i];
      best_class = i;
    }
  }
  return ($2LabelEnum)best_class;
)",
            c_ir_.types.accumulator, model_ir_.num_output_classes,
            c_ir_.pseudo_namespace_name);
      }
      break;

    case proto::ClassificationOutput::SCORE:
      if (model_ir_.num_output_classes == 1) {
        c_ir_.activation_statement = "return accumulator;";
      } else {
        c_ir_.output_parameter = absl::Substitute(
            "$0 out[$1]", c_ir_.types.output, model_ir_.num_output_classes);
        c_ir_.activation_statement = "";
      }
      break;

    case proto::ClassificationOutput::PROBABILITY:
      if (options_.c().linux_kernel_compatible()) {
        return absl::InvalidArgumentError(
            "Probability activations (Sigmoid/Softmax) are not supported in "
            "Kernel mode. Use ClassificationOutput::SCORE");
      }
      if (model_ir_.num_output_classes == 1) {
        if (model_ir_.winner_takes_all) {
          c_ir_.activation_statement =
              absl::Substitute("return ($0)accumulator / $1;",
                               c_ir_.types.output, model_ir_.num_trees);
        } else if (model_ir_.activation == ModelIR::Activation::kSigmoid) {
          c_ir_.impl_includes.insert("<math.h>");
          c_ir_.activation_statement =
              "return 1.f / (1.f + expf(-accumulator));";
        } else {
          c_ir_.activation_statement = "return accumulator;";
        }
      } else {
        c_ir_.output_parameter = absl::Substitute(
            "$0 out[$1]", c_ir_.types.output, model_ir_.num_output_classes);
        if (model_ir_.winner_takes_all) {
          c_ir_.activation_statement = absl::Substitute(
              R"(  for (int i = 0; i < $0; ++i) { 
    out[i] = ($1)accumulator[i] / $2.0f; 
  })",
              model_ir_.num_output_classes, c_ir_.types.output,
              model_ir_.num_trees);
        } else if (model_ir_.activation == ModelIR::Activation::kSoftmax) {
          c_ir_.impl_includes.insert("<math.h>");
          c_ir_.activation_statement = absl::Substitute(
              R"($0 max_logit = accumulator[0];
  for (int i = 1; i < $1; ++i) {
    if (accumulator[i] > max_logit) max_logit = accumulator[i];
  }
  $0 sum_exps = 0.0f;
  for (int i = 0; i < $1; ++i) { 
    out[i] = expf(accumulator[i] - max_logit); 
    sum_exps += out[i];
  }
  for (int i = 0; i < $1; ++i) out[i] /= sum_exps;)",
              c_ir_.types.accumulator, model_ir_.num_output_classes);
        }
      }
      break;

    default:
      return absl::InvalidArgumentError("Unknown classification output type.");
  }
  return absl::OkStatus();
}

absl::Status CTargetLowering::LowerNodes() {
  c_ir_.nodes.reserve(model_ir_.nodes.size());
  // This value is used as a marker in the feature index field to indicate an
  // oblique split. It is chosen to be the largest representable value for the
  // feature index type.
  const auto oblique_feature_idx =
      GetObliqueFeatureSentinel(model_ir_.num_features);
  c_ir_.oblique_feature_idx = oblique_feature_idx;

  for (NodeIdx i = 0; i < model_ir_.nodes.size(); ++i) {
    const auto& ir_node = model_ir_.nodes[i];
    if (ir_node.type == Node::Type::kLeaf) {
      ASSIGN_OR_RETURN(std::string node_str, LowerLeafNode(ir_node));
      c_ir_.nodes.push_back(std::move(node_str));
    } else {
      ASSIGN_OR_RETURN(std::string node_str, LowerConditionNode(ir_node));
      c_ir_.nodes.push_back(std::move(node_str));
    }
  }

  c_ir_.num_used_conditions = model_ir_.active_condition_types.size();

  return absl::OkStatus();
}

absl::StatusOr<std::string> CTargetLowering::LowerLeafNode(
    const Node& ir_node) {
  if (model_ir_.leaf_value_dims == 1) {
    // Scalar Leaf
    // Note: Using threshold_or_offset because builder.cc populated it here
    // instead of leaf_value
    ASSIGN_OR_RETURN(const std::string val_str,
                     FormatValue(ir_node.threshold_or_offset,
                                 !model_ir_.winner_takes_all, options_));
    return absl::Substitute("{.leaf={.val=$0}}", val_str);
  } else {
    // Vector Leaf (Multi-class Classification)
    const int offset = AsInt(ir_node.threshold_or_offset);
    // Routing optimizes binary size by dividing the offset by the
    // dimension.
    const int encoded_leaf_value =
        GetEncodedLeafValue(offset, model_ir_.num_output_classes);
    return absl::Substitute("{.leaf={.val=$0}}", encoded_leaf_value);
  }
}

absl::StatusOr<std::string> CTargetLowering::LowerConditionNode(
    const Node& ir_node) {
  STATUS_CHECK_GE(ir_node.feature_idx, 0);

  bool is_float = true;
  std::string var_name = "";

  FeatureInfo feat;

  // GUARD: If it's Oblique, feature_idx is a bank offset, NOT a dense index!
  if (ir_node.condition_type != ConditionType::OBLIQUE_CONDITION) {
    feat = model_ir_.features[ir_node.feature_idx];
    is_float = feat.is_float;
  }

  // The offset to the False branch, necessary for the Emitter's recursive
  // tree reconstruction. (The True branch is implicitly offset 1).
  const auto jump_offset_false = ir_node.next_pos_node_idx;

  switch (ir_node.condition_type) {
    case ConditionType::HIGHER_CONDITION: {
      DoubleOrInt64 val = ir_node.threshold_or_offset;
      STATUS_CHECK(IsDouble(val));

      // For routing, convert thresholds to the nearest integer, rounded up.
      if (!is_float) {
        val = static_cast<int64_t>(std::ceil(AsDouble(val)));
      }
      ASSIGN_OR_RETURN(const std::string routing_thresh_str,
                       FormatValue(val, is_float, options_));

      return absl::Substitute("{.pos=$0,.cond={.feat=$1,.thr=$2}}",
                              jump_offset_false, ir_node.feature_idx,
                              routing_thresh_str);
    }

    case ConditionType::CONTAINS_CONDITION_BUFFER_BITMAP: {
      const int offset = AsInt(ir_node.threshold_or_offset);

      // Routing format is straightforward: point to the start index of the
      // boolean slice
      return absl::Substitute("{.pos=$0,.cond={.feat=$1,.cat=$2}}",
                              jump_offset_false, ir_node.feature_idx, offset);
    }

    case ConditionType::OBLIQUE_CONDITION: {
      // Your builder stored the oblique bank index directly in feature_idx
      const int oblique_bank_idx = ir_node.feature_idx;

      return absl::Substitute("{.pos=$0,.cond={.feat=$1,.obl=$2}}",
                              jump_offset_false, c_ir_.oblique_feature_idx,
                              oblique_bank_idx);
    }

    case ConditionType::TRUE_CONDITION: {
      ASSIGN_OR_RETURN(const std::string routing_thresh_str,
                       FormatValue(0.5f, true, options_));
      return absl::Substitute("{.pos=$0,.cond={.feat=$1,.thr=$2}}",
                              jump_offset_false, ir_node.feature_idx,
                              routing_thresh_str);
    }

    default:
      return absl::InternalError(
          "Unknown ConditionType encountered in LowerNodes.");
  }
}

absl::Status CTargetLowering::LowerRoutingData() {
  ASSIGN_OR_RETURN(RoutingDataAssets assets,
                   PrepareRoutingDataAssets(model_ir_));

  c_ir_.root_deltas_content = std::move(assets.root_deltas_content);

  if (!model_ir_.bitset_bank.empty()) {
    const auto bytes = PackBoolVector(model_ir_.bitset_bank);
    std::vector<std::string> hex_bytes;
    hex_bytes.reserve(bytes.size());
    for (const auto byte : bytes) {
      hex_bytes.push_back(absl::StrCat("0x", absl::Hex(byte)));
    }
    c_ir_.categorical_bank_content = absl::StrJoin(hex_bytes, ",");
  }
  if (!model_ir_.oblique_weights.empty()) {
    if (options_.c().linux_kernel_compatible()) {
      // Re-format the oblique weights as fixed-point integers
      std::vector<std::string> weights_str;
      weights_str.reserve(model_ir_.oblique_weights.size());
      for (float w : model_ir_.oblique_weights) {
        ASSIGN_OR_RETURN(std::string w_str,
                         FormatValue(w, /*is_float=*/true, options_));
        weights_str.push_back(w_str);
      }
      c_ir_.oblique_weights_content = absl::StrJoin(weights_str, ",");
    } else {
      c_ir_.oblique_weights_content = std::move(assets.oblique_weights_content);
    }
    c_ir_.oblique_features_content = std::move(assets.oblique_features_content);
  }

  for (const auto& cond_type : model_ir_.active_condition_types) {
    if (cond_type == ConditionType::HIGHER_CONDITION) {
      c_ir_.routing_condition_eval_blocks.push_back(std::make_pair(
          cond_type,
          absl::Substitute(
              R"(        $0 numerical_feature = *($0*)(raw_instance + off);
        eval = numerical_feature >= node->cond.thr;)",
              c_ir_.types.numerical_feature)));
    } else if (cond_type == ConditionType::OBLIQUE_CONDITION) {
      if (options_.c().linux_kernel_compatible()) {
        c_ir_.impl_includes.insert(
            "<linux/string.h>");  // Required for kernel memcpy

        c_ir_.routing_condition_eval_blocks.insert(
            c_ir_.routing_condition_eval_blocks.begin(),
            std::make_pair(
                cond_type,
                absl::Substitute(
                    R"(        const $0 num_projs = $2oblique_features[node->cond.obl];
        // Shift the bias to match the doubled scale of the dot product multiplication.
        s64 obl_acc = -((s64)$2oblique_weights[node->cond.obl] << $5);
        for ($0 proj_idx=0; proj_idx<num_projs; proj_idx++){
          const $4 proj_off = node->cond.obl + proj_idx + 1;
          $1 numerical_feature;
          const $0 feat_idx = $2oblique_features[proj_off];
          const $4 safe_off = $2FeatureOffsets[feat_idx];
          memcpy(&numerical_feature, raw_instance + safe_off, sizeof($1));

          // Cast to 64-bit to prevent overflow during fixed-point multiplication
          obl_acc += (s64)numerical_feature * (s64)$2oblique_weights[proj_off];
        }
        eval = obl_acc >= 0;)",
                    c_ir_.types.oblique_features, c_ir_.types.numerical_feature,
                    c_ir_.pseudo_namespace_name, c_ir_.types.oblique_weights,
                    c_ir_.types.feature_offsets,
                    options_.c().fixed_point_fractional_bits())));
      } else {
        c_ir_.impl_includes.insert("<string.h>");
        c_ir_.routing_condition_eval_blocks.insert(
            c_ir_.routing_condition_eval_blocks.begin(),
            std::make_pair(
                cond_type,
                absl::Substitute(
                    R"(        const $0 num_projs = $2oblique_features[node->cond.obl];
        $3 obl_acc = -$2oblique_weights[node->cond.obl];
        for ($0 proj_idx=0; proj_idx<num_projs; proj_idx++){
          const $4 proj_off = node->cond.obl + proj_idx + 1;
          $1 numerical_feature;
          const $0 feat_idx = $2oblique_features[proj_off];
          const $4 safe_off = $2FeatureOffsets[feat_idx];
          memcpy(&numerical_feature, raw_instance + safe_off, sizeof($1));
          obl_acc += numerical_feature * $2oblique_weights[proj_off];
        }
        eval = obl_acc >= 0;)",
                    c_ir_.types.oblique_features, c_ir_.types.numerical_feature,
                    c_ir_.pseudo_namespace_name, c_ir_.types.oblique_weights,
                    c_ir_.types.feature_offsets)));
      }
    } else if (cond_type == ConditionType::CONTAINS_CONDITION_BUFFER_BITMAP) {
      c_ir_.routing_condition_eval_blocks.push_back(std::make_pair(
          cond_type,
          absl::Substitute(
              R"(        $0 categorical_feature = *($0*)(raw_instance + off);
        eval = $1BitTest($1categorical_bank, categorical_feature + node->cond.cat);)",
              c_ir_.types.categorical_feature, c_ir_.pseudo_namespace_name)));
    } else if (cond_type == ConditionType::TRUE_CONDITION) {
      // Find what type categorical features use
      c_ir_.routing_condition_eval_blocks.push_back(std::make_pair(
          cond_type,
          absl::Substitute(
              R"(        $0 boolean_feature = *($0*)(raw_instance + off);
        eval = boolean_feature;)",
              c_ir_.types.boolean)));
    }
  }

  if (!assets.leaf_value_bank_content.empty()) {
    c_ir_.leaf_value_bank_content = std::move(assets.leaf_value_bank_content);
  }

  return absl::OkStatus();
}

// --- Helpers ---

std::string CTargetLowering::AddPseudoNamespace(absl::string_view name) const {
  return absl::StrCat(c_ir_.pseudo_namespace_name, name);
}

}  // namespace yggdrasil_decision_forests::serving::embed::internal
