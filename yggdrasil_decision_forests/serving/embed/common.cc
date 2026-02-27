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

#include "yggdrasil_decision_forests/serving/embed/common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/serving/embed/ir/model_ir.h"
#include "yggdrasil_decision_forests/serving/embed/utils.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

namespace {
constexpr char kLabelReservedSymbol[] = "Label";
}  // namespace

absl::StatusOr<ModelStatistics> ComputeStatistics(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface) {
  ModelStatistics stats{
      .num_trees = static_cast<int64_t>(df_interface.num_trees()),
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

absl::Status ComputeBaseInternalOptionsFeature(
    const ModelStatistics& stats, const model::AbstractModel& model,
    const proto::Options& options, BaseInternalOptions* out) {
  out->feature_value_bytes = 1;
  out->numerical_feature_is_float = false;

  const auto max_value_to_num_bytes = [&options](const int64_t value) {
    return IsJava(options) ? MaxSignedValueToNumBytes(value)
                           : MaxUnsignedValueToNumBytes(value);
  };

  out->feature_index_bytes =
      max_value_to_num_bytes(stats.num_features + kReservedFeatureIndexes);
  out->tree_index_bytes = max_value_to_num_bytes(stats.num_trees);

  if (stats.sum_size_categorical_bitmap_masks == 0) {
    out->categorical_idx_bytes = 0;
  } else {
    out->categorical_idx_bytes =
        max_value_to_num_bytes(stats.sum_size_categorical_bitmap_masks);
  }

  // This is the number of bytes to encode a node index in a tree. The precision
  // for an offset is in average 50% smaller.
  // TODO: Optimize node_offset_bytes.
  const auto max_num_nodes = NumLeavesToNumNodes(stats.max_num_leaves_per_tree);
  out->node_offset_bytes = max_value_to_num_bytes(max_num_nodes);

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
        const int feature_bytes = max_value_to_num_bytes(
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

absl::Status ComputeBaseInternalOptionsCategoricalDictionaries(
    const model::AbstractModel& model, const ModelStatistics& stats,
    const proto::Options& options, BaseInternalOptions* out) {
  const auto add_dict = [&](absl::string_view name, const int column_idx,
                            const dataset::proto::CategoricalSpec& column,
                            const bool is_label) {
    // Get the list of values.
    auto& dictionary = out->categorical_dicts[column_idx];
    dictionary.sanitized_name = name;
    dictionary.is_label = is_label;
    dictionary.sanitized_items.assign(
        column.number_of_unique_values() - is_label, "");
    dictionary.items.assign(column.number_of_unique_values() - is_label, "");

    // Set of all sanitized_items. Used to detect duplications after the
    // conversion to c++ symbols.
    absl::flat_hash_set<std::string> sanitized_items;

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
        switch (options.language_case()) {
          case proto::Options::kC:
          case proto::Options::kCpp:
            item_symbol =
                "OutOfVocabulary";  // Better than the default <OOV> symbol.
            break;
          case proto::Options::kJava:
            item_symbol = "OUT_OF_VOCABULARY";
            break;
          case proto::Options::kCc:
          case proto::Options::LANGUAGE_NOT_SET:
            NOTREACHED();
            break;
        }
      } else {
        switch (options.language_case()) {
          case proto::Options::kC:
          case proto::Options::kCpp:
            item_symbol = StringToStructSymbol(item.first,
                                               /*.ensure_letter_first=*/false);
            break;
          case proto::Options::kJava:
            item_symbol = StringToJavaEnumConstant(item.first);
            break;
          case proto::Options::kCc:
          case proto::Options::LANGUAGE_NOT_SET:
            NOTREACHED();
            break;
        }
        if (sanitized_items.contains(item_symbol)) {
          // This sanitized symbol already exist. Create a new one by adding a
          // prefix.
          int local_index = 1;
          std::string extended_item_symbol;
          do {
            extended_item_symbol =
                absl::StrCat(item_symbol, "_", local_index++);
          } while (sanitized_items.contains(extended_item_symbol));
          item_symbol = extended_item_symbol;
        }
        sanitized_items.insert(item_symbol);
      }
      dictionary.sanitized_items[index] = item_symbol;
      dictionary.items[index] = item.first;
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
    if (col_spec.categorical().has_is_already_integerized() &&
        col_spec.categorical().is_already_integerized()) {
      out->has_integerized_categorical = true;
    } else {
      add_dict(StringToStructSymbol(col_spec.name()), input_feature,
               col_spec.categorical(), false);
    }
  }

  return absl::OkStatus();
}

absl::Status SpecializedConversion::Validate(
    const proto::Options& options) const {
  STATUS_CHECK(!accumulator_type.empty());
  STATUS_CHECK(!accumulator_initial_value.empty());
  STATUS_CHECK(!return_prediction.empty());
  STATUS_CHECK(!accumulator_type.empty());
  STATUS_CHECK_GT(leaf_value_spec.dims, 0);
  STATUS_CHECK_NE(leaf_value_spec.dtype, proto::DType::UNDEFINED);
  STATUS_CHECK(leaf_value_fn);

  if (options.algorithm() == proto::Algorithm::IF_ELSE) {
    STATUS_CHECK(set_node_ifelse_fn);
  }
  if (options.algorithm() == proto::Algorithm::ROUTING) {
    STATUS_CHECK(!routing_node.empty());
  }
  return absl::OkStatus();
}
// Computes the mapping from feature idx to condition type.
//
// Record a mapping from feature to condition type. This is possible because
// this implementation assumes that each feature is only used in one type of
// condition (which is not generally the case in YDF).
//
// TODO: Use a virtual feature index system to allow a same feature to be
// used with different condition types.
absl::StatusOr<std::vector<uint8_t>> GenRoutingModelDataConditionType(
    const model::AbstractModel& model, const ModelStatistics& stats) {
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
            absl::StrCat("Unsupported feature type: ",
                         dataset::proto::ColumnType_Name(col_spec.type())));
    }
  }
  return condition_types;
}

int ObliqueFeatureIndex(const proto::Options& options,
                        const BaseInternalOptions& internal_options) {
  return IsJava(options)
             ? NumBytesToMaxSignedValue(internal_options.feature_index_bytes)
             : NumBytesToMaxUnsignedValue(internal_options.feature_index_bytes);
}

std::string ResolveNameCollision(
    const std::string& name,
    const absl::flat_hash_set<std::string>& existing_names) {
  if (!existing_names.contains(name)) {
    return name;
  }
  int local_index = 1;
  std::string extended_item_symbol;
  do {
    extended_item_symbol = absl::StrCat(name, "_", local_index++);
  } while (existing_names.contains(extended_item_symbol));
  return extended_item_symbol;
}

uint32_t GetObliqueFeatureSentinel(int64_t num_features) {
  return NumBytesToMaxUnsignedValue(
      MaxUnsignedValueToNumBytes(num_features + kReservedFeatureIndexes));
}

int GetEncodedLeafValue(int64_t offset, int num_output_classes) {
  return offset / num_output_classes;
}

absl::StatusOr<std::string> StorageToType(int bytes, bool is_float,
                                          bool is_signed) {
  if (is_float) {
    if (bytes == 4) return "float";
    if (bytes == 8) return "double";
  } else {
    if (is_signed) {
      if (bytes == 1) return "int8_t";
      if (bytes == 2) return "int16_t";
      if (bytes == 4) return "int32_t";
      if (bytes == 8) return "int64_t";
    } else {
      if (bytes == 1) return "uint8_t";
      if (bytes == 2) return "uint16_t";
      if (bytes == 4) return "uint32_t";
      if (bytes == 8) return "uint64_t";
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported storage type: bytes=", bytes,
                   " float=", is_float, " signed=", is_signed));
}

std::string GetBitsetBankString(const std::vector<bool>& bitset_bank) {
  std::string bitset_str;
  bitset_str.reserve(bitset_bank.size());
  for (auto it = bitset_bank.rbegin(); it != bitset_bank.rend(); ++it) {
    bitset_str.push_back(*it ? '1' : '0');
  }
  return bitset_str;
}

int GetMaxObliqueFeatureValue(const std::vector<int>& oblique_features) {
  int max_val = 0;
  for (int f : oblique_features) {
    if (f > max_val) max_val = f;
  }
  return max_val;
}

absl::Status CheckFeatureNameCollision(
    const std::string& var_name,
    absl::flat_hash_set<std::string>& sanitized_feature_names,
    const std::vector<FeatureInfo>& features) {
  if (!sanitized_feature_names.insert(var_name).second) {
    return absl::InvalidArgumentError(
        absl::Substitute("Feature name clash on feature $0, consider "
                         "renaming your features!",
                         var_name));
  }
  return absl::OkStatus();
}

absl::StatusOr<RoutingDataAssets> PrepareRoutingDataAssets(const ModelIR& ir) {
  RoutingDataAssets assets;
  assets.root_deltas_content = absl::StrJoin(ir.tree_start_offsets, ",");

  if (!ir.bitset_bank.empty()) {
    assets.categorical_bank_size = ir.bitset_bank.size();
  }

  if (!ir.oblique_weights.empty()) {
    assets.oblique_weights_content = absl::StrJoin(ir.oblique_weights, ",");

    assets.oblique_features_content = absl::StrJoin(ir.oblique_features, ",");

    if (ir.oblique_weights.size() > std::numeric_limits<uint32_t>::max()) {
      return absl::InternalError(
          "Oblique weights bank size exceeds 32-bit limit.");
    }
  }

  if (!ir.leaf_value_bank.empty()) {
    assets.leaf_value_bank_content = absl::StrJoin(ir.leaf_value_bank, ",");
  }

  return assets;
}

absl::StatusOr<BaseTypes> BuildTypesStandard(
    const proto::Options& options, const ModelIR& model_ir,
    const std::string pseudo_namespace) {
  BaseTypes types;

  // Global
  ASSIGN_OR_RETURN(
      types.num_trees,
      StorageToPrimitiveType(MaxUnsignedValueToNumBytes(model_ir.num_trees),
                             false, false));
  types.accumulator =
      !model_ir.winner_takes_all
          ? "float"
          : UnsignedInteger(MaxUnsignedValueToNumBytes(model_ir.num_trees));
  types.eval = UnsignedInteger(model_ir.node_offset_bytes);
  types.boolean = "bool";
  if (model_ir.task == ModelIR::Task::kRegression) {
    types.output = "float";
  } else {
    switch (options.classification_output()) {
      case proto::ClassificationOutput::CLASS:
        types.output = absl::Substitute("$0LabelEnum", pseudo_namespace);
        break;
      case proto::ClassificationOutput::SCORE:
        types.output = types.accumulator;
        break;
      case proto::ClassificationOutput::PROBABILITY:
        types.output = "float";
        break;
      default:
        return absl::InvalidArgumentError(
            "Unknown classification output type.");
    }
  }

  // Instance
  types.numerical_feature = "";
  for (const auto& feature : model_ir.features) {
    if (feature.is_label) continue;
    if (feature.type == FeatureInfo::Type::kNumerical) {
      if (feature.is_float) {
        types.numerical_feature = "float";
        break;
      } else if (types.numerical_feature.empty()) {
        types.numerical_feature = SignedInteger(model_ir.feature_value_bytes);
      }
    }
  }
  types.categorical_feature = UnsignedInteger(model_ir.feature_value_bytes);
  types.integerized_categorical_feature =
      SignedInteger(model_ir.feature_value_bytes);

  // Node data structure
  ASSIGN_OR_RETURN(
      types.pos,
      StorageToPrimitiveType(MaxUnsignedValueToNumBytes(model_ir.nodes.size()),
                             false, false));
  ASSIGN_OR_RETURN(
      types.feature_idx,
      StorageToPrimitiveType(
          MaxUnsignedValueToNumBytes(model_ir.features.size()), false, false));
  types.threshold = types.numerical_feature;
  if (!model_ir.bitset_bank.empty()) {
    ASSIGN_OR_RETURN(types.cat_bank_idx,
                     StorageToPrimitiveType(MaxUnsignedValueToNumBytes(
                                                model_ir.bitset_bank.size()),
                                            false, false));
  }
  ASSIGN_OR_RETURN(types.obl_bank_idx,
                   StorageToPrimitiveType(MaxUnsignedValueToNumBytes(
                                              model_ir.oblique_weights.size()),
                                          false, false));
  if (model_ir.leaf_value_dims == 1) {
    // The leaf value is stored in the node struct.
    types.leaf_value = DTypeToCppType(model_ir.leaf_value_dtype);
  } else {
    // The leaf values are stored in a separate buffer. The node struct contains
    // an index to this buffer.
    types.leaf_value = UnsignedInteger(MaxUnsignedValueToNumBytes(
        model_ir.num_leaves / model_ir.leaf_value_dims));
  }

  // Banks
  types.categorical_bank = "uint8_t";
  types.condition_types = "uint8_t";
  const NodeIdx max_root_delta = model_ir.tree_start_offsets.empty()
                                     ? 0
                                     : model_ir.tree_start_offsets.back();
  ASSIGN_OR_RETURN(
      types.root_deltas,
      StorageToPrimitiveType(MaxUnsignedValueToNumBytes(max_root_delta), false,
                             false));
  types.oblique_weights = "float";
  ASSIGN_OR_RETURN(types.oblique_features,
                   StorageToPrimitiveType(
                       MaxUnsignedValueToNumBytes(GetMaxObliqueFeatureValue(
                           model_ir.oblique_features)),
                       false, false));
  types.feature_offsets = "size_t";
  types.leaf_value_bank = "float";

  return types;
}

absl::StatusOr<BaseTypes> BuildTypesKernel(const proto::Options& options,
                                           const ModelIR& model_ir,
                                           const std::string pseudo_namespace) {
  BaseTypes types;

  // Global
  ASSIGN_OR_RETURN(
      types.num_trees,
      KernelStorageToPrimitiveType(
          MaxUnsignedValueToNumBytes(model_ir.num_trees), false, false));

  types.accumulator = !model_ir.winner_takes_all
                          ? "s32"
                          : KernelUnsignedInteger(
                                MaxUnsignedValueToNumBytes(model_ir.num_trees));

  types.eval = KernelUnsignedInteger(model_ir.node_offset_bytes);
  types.boolean = "bool";

  if (model_ir.task == ModelIR::Task::kRegression) {
    types.output = "s32";
  } else {
    switch (options.classification_output()) {
      case proto::ClassificationOutput::CLASS:
        types.output = absl::Substitute("$0LabelEnum", pseudo_namespace);
        break;
      case proto::ClassificationOutput::SCORE:
        types.output = types.accumulator;
        break;
      case proto::ClassificationOutput::PROBABILITY:
        types.output = "s32";
        break;
      default:
        return absl::InvalidArgumentError(
            "Unknown classification output type.");
    }
  }

  // Instance
  types.numerical_feature = "";
  for (const auto& feature : model_ir.features) {
    if (feature.is_label) continue;
    if (feature.type == FeatureInfo::Type::kNumerical) {
      if (feature.is_float) {
        types.numerical_feature = "s32";
        break;
      } else if (types.numerical_feature.empty()) {
        types.numerical_feature =
            KernelSignedInteger(model_ir.feature_value_bytes);
      }
    }
  }
  types.categorical_feature =
      KernelUnsignedInteger(model_ir.feature_value_bytes);
  types.integerized_categorical_feature =
      KernelSignedInteger(model_ir.feature_value_bytes);

  // Node data structure
  ASSIGN_OR_RETURN(
      types.pos,
      KernelStorageToPrimitiveType(
          MaxUnsignedValueToNumBytes(model_ir.nodes.size()), false, false));
  ASSIGN_OR_RETURN(
      types.feature_idx,
      KernelStorageToPrimitiveType(
          MaxUnsignedValueToNumBytes(model_ir.features.size()), false, false));

  types.threshold = types.numerical_feature;

  if (!model_ir.bitset_bank.empty()) {
    ASSIGN_OR_RETURN(
        types.cat_bank_idx,
        KernelStorageToPrimitiveType(
            MaxUnsignedValueToNumBytes(model_ir.bitset_bank.size()), false,
            false));
  }
  ASSIGN_OR_RETURN(
      types.obl_bank_idx,
      KernelStorageToPrimitiveType(
          MaxUnsignedValueToNumBytes(model_ir.oblique_weights.size()), false,
          false));
  if (model_ir.leaf_value_dims == 1) {
    types.leaf_value = KernelDTypeToCppType(model_ir.leaf_value_dtype);
  } else {
    types.leaf_value = KernelUnsignedInteger(MaxUnsignedValueToNumBytes(
        model_ir.num_leaves / model_ir.leaf_value_dims));
  }

  // Banks
  types.categorical_bank = "u8";
  types.condition_types = "u8";

  const NodeIdx max_root_delta = model_ir.tree_start_offsets.empty()
                                     ? 0
                                     : model_ir.tree_start_offsets.back();
  ASSIGN_OR_RETURN(
      types.root_deltas,
      KernelStorageToPrimitiveType(MaxUnsignedValueToNumBytes(max_root_delta),
                                   false, false));

  types.oblique_weights = "s32";

  ASSIGN_OR_RETURN(types.oblique_features,
                   KernelStorageToPrimitiveType(
                       MaxUnsignedValueToNumBytes(GetMaxObliqueFeatureValue(
                           model_ir.oblique_features)),
                       false, false));

  types.feature_offsets = "size_t";
  types.leaf_value_bank = "s32";

  return types;
}

absl::StatusOr<BaseTypes> BuildTypes(const proto::Options& options,
                                     const ModelIR& model_ir,
                                     const std::string pseudo_namespace) {
  if (options.c().linux_kernel_compatible()) {
    return BuildTypesKernel(options, model_ir, pseudo_namespace);
  }

  return BuildTypesStandard(options, model_ir, pseudo_namespace);
}

}  // namespace yggdrasil_decision_forests::serving::embed::internal
