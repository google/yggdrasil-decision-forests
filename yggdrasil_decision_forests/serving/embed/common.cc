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
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/serving/embed/utils.h"
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

  out->feature_index_bytes =
      MaxUnsignedValueToNumBytes(stats.num_features + kReservedFeatureIndexes);
  out->tree_index_bytes = MaxUnsignedValueToNumBytes(stats.num_trees);

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
        item_symbol =
            "OutOfVocabulary";  // Better than the default <OOV> symbol.
      } else {
        item_symbol = StringToStructSymbol(item.first,
                                           /*.ensure_letter_first=*/false);

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
    add_dict(StringToStructSymbol(col_spec.name()), input_feature,
             col_spec.categorical(), false);
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

}  // namespace yggdrasil_decision_forests::serving::embed::internal
