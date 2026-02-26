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

#include "yggdrasil_decision_forests/serving/embed/ir/builder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/serving/embed/ir/model_ir.h"
#include "yggdrasil_decision_forests/serving/embed/utils.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

namespace {

absl::StatusOr<std::vector<DoubleOrInt64>> ExtractLeafValues(
    const model::decision_tree::proto::Node& node, const double scale,
    const bool winner_takes_all, const int leaf_value_dims) {
  if (node.has_regressor()) {
    STATUS_CHECK(!winner_takes_all);
    return std::vector<DoubleOrInt64>{scale * node.regressor().top_value()};
  }

  if (!node.has_classifier()) {
    return absl::InvalidArgumentError("Unsupported leaf type.");
  }

  const auto& classifier = node.classifier();

  if (winner_takes_all) {
    STATUS_CHECK_EQ(leaf_value_dims, 1);
    // For both binary and multiclass, the value is the class index - 1.
    // Class indices are 1-based in the proto (0 is usually OOV/missing, but for
    // leaf values it's the class index). 1=Class0, 2=Class1...
    return std::vector<DoubleOrInt64>{
        static_cast<int64_t>(classifier.top_value() - 1)};
  }

  const auto& distribution = classifier.distribution();
  const double sum = distribution.sum();

  if (leaf_value_dims == 1) {
    // Binary classification: Return probability of positive class (index 2).
    return std::vector<DoubleOrInt64>{scale * (distribution.counts(2) / sum)};
  }

  // Multiclass
  std::vector<DoubleOrInt64> node_values(leaf_value_dims);
  for (int i = 0; i < leaf_value_dims; ++i) {
    node_values[i] = scale * (distribution.counts(i + 1) / sum);
  }
  return node_values;
}

// Given a label column, compute the number of output classes.
absl::StatusOr<int32_t> ComputeNumOutputClasses(
    const dataset::proto::Column& col) {
  STATUS_CHECK(col.type() == dataset::proto::CATEGORICAL);
  const auto& cat = col.categorical();
  const auto num_classes = cat.is_already_integerized()
                               ? cat.number_of_unique_values() - 1
                               : cat.items_size() - 1;
  if (num_classes > std::numeric_limits<int32_t>::max()) {
    return absl::InternalError(
        absl::StrCat("The number of output classes ", num_classes,
                     " exceeds the maximum allowed value ",
                     std::numeric_limits<int32_t>::max()));
  }
  STATUS_CHECK_GE(num_classes, 2);
  return num_classes == 2 ? 1 : num_classes;
}

std::vector<std::string> BuildVocabularyVector(
    const dataset::proto::CategoricalSpec& cat_spec) {
  std::vector<std::string> vocab_vector;
  vocab_vector.resize(cat_spec.number_of_unique_values());
  for (const auto& item : cat_spec.items()) {
    if (item.second.index() >= 0 && item.second.index() < vocab_vector.size()) {
      vocab_vector[item.second.index()] = item.first;
    }
  }
  return vocab_vector;
}

bool DtypeIsFloat(dataset::proto::DType dtype) {
  return dtype == dataset::proto::DTYPE_FLOAT16 ||
         dtype == dataset::proto::DTYPE_FLOAT32 ||
         dtype == dataset::proto::DTYPE_FLOAT64;
}

absl::StatusOr<FeatureInfo> GenFeatureInfo(const dataset::proto::Column& col,
                                           int input_feature_idx,
                                           bool is_label) {
  FeatureInfo feature_info;
  feature_info.original_name = col.name();
  feature_info.is_label = is_label;

  switch (col.type()) {
    case dataset::proto::ColumnType::NUMERICAL:
      feature_info.type = FeatureInfo::Type::kNumerical;
      feature_info.na_replacement = col.numerical().mean();
      feature_info.is_float =
          col.has_dtype() ? DtypeIsFloat(col.dtype()) : true;
      break;

    case dataset::proto::ColumnType::CATEGORICAL:
      if (col.categorical().is_already_integerized()) {
        feature_info.type = FeatureInfo::Type::kIntegerizedCategorical;
        feature_info.na_replacement = col.categorical().most_frequent_value();
        feature_info.maximum_value =
            col.categorical().number_of_unique_values() - 1;
      } else {
        feature_info.type = FeatureInfo::Type::kCategorical;
        feature_info.vocabulary = BuildVocabularyVector(col.categorical());
      }
      break;

    case dataset::proto::ColumnType::BOOLEAN:
      feature_info.type = FeatureInfo::Type::kBoolean;
      break;

    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Non supported feature type: ",
                       dataset::proto::ColumnType_Name(col.type())));
  }
  return feature_info;
}

}  // namespace
absl::StatusOr<ModelIR> ModelIRBuilder::Build(const model::AbstractModel& model,
                                              const proto::Options& options) {
  ModelIRBuilder builder(&model, &options);

  RETURN_IF_ERROR(builder.BuildSpecializedConversions());

  RETURN_IF_ERROR(builder.CompileBasicModelFeatures());

  RETURN_IF_ERROR(builder.AnalyzeFeatures());

  RETURN_IF_ERROR(builder.CompileTrees());

  return std::move(builder.ir_);
}

absl::Status ModelIRBuilder::CompileBasicModelFeatures() {
  if (model_->data_spec().columns_size() >
      std::numeric_limits<FeatureIdx>::max()) {
    return absl::InternalError(absl::StrCat(
        "The number of columns ", model_->data_spec().columns_size(),
        " exceeds the maximum allowed value ",
        std::numeric_limits<FeatureIdx>::max()));
  }
  if (model_->input_features().size() >
      std::numeric_limits<FeatureIdx>::max()) {
    return absl::InternalError(absl::StrCat(
        "The number of input features ", model_->input_features().size(),
        " exceeds the maximum allowed value ",
        std::numeric_limits<FeatureIdx>::max()));
  }
  if (df_interface_->num_trees() > std::numeric_limits<uint32_t>::max()) {
    return absl::InternalError("Number of trees exceeds 32-bit limit.");
  }
  if (df_interface_->num_nodes() > std::numeric_limits<uint32_t>::max()) {
    return absl::InternalError("Number of trees exceeds 32-bit limit.");
  }
  ir_.num_features = model_->input_features().size();
  switch (model_->task()) {
    case model::proto::REGRESSION:
      ir_.task = ModelIR::Task::kRegression;
      ir_.num_output_classes = 1;
      break;
    case model::proto::CLASSIFICATION: {
      const int label_col_idx = model_->label_col_idx();
      STATUS_CHECK_GE(label_col_idx, 0);
      STATUS_CHECK_LT(label_col_idx, model_->data_spec().columns_size());
      ASSIGN_OR_RETURN(ir_.num_output_classes,
                       ComputeNumOutputClasses(model_->data_spec().columns(
                           model_->label_col_idx())));
      if (ir_.num_output_classes == 1) {
        ir_.task = ModelIR::Task::kBinaryClassification;
      } else {
        ir_.task = ModelIR::Task::kMulticlassClassification;
      }
      break;
    }
    default:
      return absl::InvalidArgumentError("Unsupported model task");
  }
  if (gradient_boosted_trees_model_ != nullptr) {
    ir_.accumulator_initialization.insert(
        ir_.accumulator_initialization.begin(),
        gradient_boosted_trees_model_->initial_predictions().begin(),
        gradient_boosted_trees_model_->initial_predictions().end());
    ir_.leaf_value_dtype = proto::DType::FLOAT32;
    ir_.leaf_value_dims = 1;
    switch (ir_.task) {
      case ModelIR::Task::kRegression:
        ir_.activation = ModelIR::Activation::kEquality;
        break;
      case ModelIR::Task::kBinaryClassification:
        if (gradient_boosted_trees_model_->output_logits() ||
            model_->classification_outputs_probabilities()) {
          ir_.activation = ModelIR::Activation::kSigmoid;
        } else {
          ir_.activation = ModelIR::Activation::kEquality;
        }
        break;
      case ModelIR::Task::kMulticlassClassification:
        if (gradient_boosted_trees_model_->output_logits() ||
            model_->classification_outputs_probabilities()) {
          ir_.activation = ModelIR::Activation::kSoftmax;
        } else {
          ir_.activation = ModelIR::Activation::kEquality;
        }
        break;
    }
  } else if (random_forest_model_ != nullptr) {
    if (ir_.task != ModelIR::Task::kRegression &&
        random_forest_model_->winner_take_all_inference()) {
      ir_.accumulator_initialization.assign(ir_.num_output_classes, int64_t{0});
      ir_.winner_takes_all = true;
      ir_.leaf_value_dims = 1;
      if (ir_.task == ModelIR::Task::kBinaryClassification) {
        ir_.leaf_value_dtype = proto::DType::BOOL;
      } else {
        ir_.leaf_value_dtype = UnsignedIntegerToDtype(
            MaxUnsignedValueToNumBytes(ir_.num_output_classes));
      }
    } else {
      ir_.accumulator_initialization.assign(ir_.num_output_classes, 0.0);
      ir_.leaf_value_dtype = proto::DType::FLOAT32;
      if (ir_.task == ModelIR::Task::kMulticlassClassification) {
        ir_.leaf_value_dims = ir_.num_output_classes;
      } else {
        ir_.leaf_value_dims = 1;
      }
    }
  } else {
    return absl::InvalidArgumentError("No specialized model found");
  }
  return absl::OkStatus();
}

absl::Status ModelIRBuilder::AnalyzeFeatures() {
  FeatureIdx model_feature_idx = 0;  // Track the dense index
  for (const auto input_feature_idx : model_->input_features()) {
    const auto& col = model_->data_spec().columns(input_feature_idx);
    ASSIGN_OR_RETURN(
        const auto feature_info,
        GenFeatureInfo(col, input_feature_idx,
                       input_feature_idx == model_->label_col_idx()));
    ir_.features.push_back(feature_info);

    column_idx_to_model_feature_idx_[input_feature_idx] = model_feature_idx++;
  }

  if (model_->label_col_idx() != -1) {
    bool is_input = false;
    for (const auto input_feature_idx : model_->input_features()) {
      if (input_feature_idx == model_->label_col_idx()) {
        is_input = true;
        break;
      }
    }
    if (!is_input) {
      const auto& col = model_->data_spec().columns(model_->label_col_idx());
      ASSIGN_OR_RETURN(const auto feature_info,
                       GenFeatureInfo(col, model_->label_col_idx(), true));
      ir_.features.push_back(feature_info);
    }
  }
  // Compute feature_value_bytes;
  for (const int feat : model_->input_features()) {
    const auto& col_spec = model_->data_spec().columns(feat);
    switch (col_spec.type()) {
      case dataset::proto::ColumnType::NUMERICAL: {
        switch (col_spec.dtype()) {
          case dataset::proto::DTYPE_INVALID:  // Default
          case dataset::proto::DTYPE_FLOAT32:
          // Note: float64 are always converted to float32 during training.
          case dataset::proto::DTYPE_FLOAT64:
            ir_.feature_value_bytes = std::max(ir_.feature_value_bytes, 4);
            break;
          case dataset::proto::DTYPE_INT16:
            ir_.feature_value_bytes = std::max(ir_.feature_value_bytes, 2);
            break;
          case dataset::proto::DTYPE_INT32:
          // Note: int64 are always converted to int32 during training.
          case dataset::proto::DTYPE_INT64:
            ir_.feature_value_bytes = std::max(ir_.feature_value_bytes, 4);
            break;
          case dataset::proto::DTYPE_INT8:
          case dataset::proto::DTYPE_BOOL:
            // Nothing to do.
            break;
          default:
            return absl::InvalidArgumentError(
                absl::StrCat("Unsupported numerical feature type: ",
                             dataset::proto::DType_Name(col_spec.dtype())));
        }
      } break;
      case dataset::proto::ColumnType::CATEGORICAL: {
        // TODO: Might have to change this for the Java export, since Java uses
        // signed types.
        const int feature_bytes = MaxUnsignedValueToNumBytes(
            col_spec.categorical().number_of_unique_values());
        ir_.feature_value_bytes =
            std::max(ir_.feature_value_bytes, feature_bytes);
      } break;
      case dataset::proto::ColumnType::BOOLEAN:
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported feature type: ",
                         dataset::proto::ColumnType_Name(col_spec.type())));
    }
  }
  return absl::OkStatus();
}

absl::Status ModelIRBuilder::CompileTrees() {
  ir_.num_trees = df_interface_->num_trees();
  ir_.tree_start_offsets.reserve(ir_.num_trees);
  absl::flat_hash_set<ConditionType> active_condition_types;
  const auto& trees = df_interface_->decision_trees();
  if (ir_.num_trees > 0) {
    ir_.nodes.reserve(ir_.num_trees * trees[0]->NumNodes());
  } else {
    return absl::InvalidArgumentError("Cannot compile a model with no trees.");
  }

  int64_t max_nodes_per_tree = 0;
  for (NodeIdx tree_idx = 0; tree_idx < ir_.num_trees; ++tree_idx) {
    const auto& tree = trees[tree_idx];
    max_nodes_per_tree = std::max(max_nodes_per_tree, tree->NumNodes());
    ir_.num_leaves += tree->NumLeafs();
    std::optional<int> target_class_idx;
    if (ir_.task == ModelIR::Task::kMulticlassClassification &&
        gradient_boosted_trees_model_) {
      STATUS_CHECK_GE(ir_.num_output_classes, 1);
      target_class_idx = tree_idx % ir_.num_output_classes;
    }
    ASSIGN_OR_RETURN(const auto num_nodes,
                     CompileNode(tree->root(), target_class_idx, tree_idx,
                                 active_condition_types));
    ir_.tree_start_offsets.push_back(num_nodes);
  }
  if (ir_.bitset_bank.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::InternalError("Bitset bank size exceeds 32-bit limit.");
  }

  ir_.active_condition_types.insert(ir_.active_condition_types.begin(),
                                    active_condition_types.begin(),
                                    active_condition_types.end());
  std::sort(ir_.active_condition_types.begin(),
            ir_.active_condition_types.end());

  ir_.node_offset_bytes = MaxUnsignedValueToNumBytes(max_nodes_per_tree);

  return absl::OkStatus();
}
absl::StatusOr<int32_t> ModelIRBuilder::CompileNode(
    const model::decision_tree::NodeWithChildren& node,
    const std::optional<int> target_class_idx, const NodeIdx tree_idx,
    absl::flat_hash_set<ConditionType>& active_condition_types) {
  STATUS_CHECK_GT(ir_.leaf_value_dims, 0);
  bool winner_takes_all = false;
  double scale = 1.0;
  if (random_forest_model_) {
    scale = 1.0 / ir_.num_trees;
    winner_takes_all = random_forest_model_->winner_take_all_inference() &&
                       ir_.task != ModelIR::Task::kRegression;
  }
  Node ir_node;
  ir_node.tree_idx = tree_idx;
  if (node.IsLeaf()) {
    ir_node.type = Node::Type::kLeaf;
    ASSIGN_OR_RETURN(const auto leaf_values,
                     ExtractLeafValues(node.node(), scale, winner_takes_all,
                                       ir_.leaf_value_dims));
    if (leaf_values.size() == 1) {
      ir_node.threshold_or_offset = leaf_values[0];
    } else {
      ASSIGN_OR_RETURN(ir_node.threshold_or_offset, AddToLeafBank(leaf_values));
    }
    ir_.nodes.push_back(ir_node);
    return 1;
  }
  ir_.nodes.push_back(std::move(ir_node));
  const size_t cur_node_pos = ir_.nodes.size() - 1;
  ASSIGN_OR_RETURN(const auto num_nodes_neg_subtree,
                   CompileNode(*node.neg_child(), target_class_idx, tree_idx,
                               active_condition_types));

  auto& cur_node = ir_.nodes[cur_node_pos];
  cur_node.next_pos_node_idx = num_nodes_neg_subtree;
  cur_node.type = Node::Type::kCondition;

  RETURN_IF_ERROR(HandleCondition(node.node().condition(), cur_node,
                                  active_condition_types));

  ASSIGN_OR_RETURN(const auto num_nodes_pos_subtree,
                   CompileNode(*node.pos_child(), target_class_idx, tree_idx,
                               active_condition_types));
  return 1 + num_nodes_neg_subtree + num_nodes_pos_subtree;
}

absl::Status ModelIRBuilder::HandleCondition(
    const model::decision_tree::proto::NodeCondition& node_condition,
    Node& cur_node,
    absl::flat_hash_set<ConditionType>& active_condition_types) {
  const auto& attribute = node_condition.attribute();
  const auto& condition = node_condition.condition();

  if (condition.type_case() !=
      model::decision_tree::proto::Condition::kObliqueCondition) {
    cur_node.feature_idx = column_idx_to_model_feature_idx_.at(attribute);
  }

  // Common logic for ContainsCondition and ContainsBitmapCondition
  auto handle_contains_condition =
      [&](const std::vector<int32_t>& items) -> absl::Status {
    cur_node.condition_type = ConditionType::CONTAINS_CONDITION_BUFFER_BITMAP;
    ASSIGN_OR_RETURN(
        cur_node.threshold_or_offset,
        AddToBitsetBank(items,
                        model_->data_spec()
                            .columns(attribute)
                            .categorical()
                            .number_of_unique_values(),
                        model_->data_spec().columns(attribute).name()));
    active_condition_types.insert(
        ConditionType::CONTAINS_CONDITION_BUFFER_BITMAP);
    return absl::OkStatus();
  };

  switch (condition.type_case()) {
    case model::decision_tree::proto::Condition::kHigherCondition:
      cur_node.condition_type = ConditionType::HIGHER_CONDITION;
      cur_node.threshold_or_offset = condition.higher_condition().threshold();
      active_condition_types.insert(ConditionType::HIGHER_CONDITION);
      break;

    case model::decision_tree::proto::Condition::kTrueValueCondition:
      cur_node.condition_type = ConditionType::TRUE_CONDITION;
      active_condition_types.insert(ConditionType::TRUE_CONDITION);
      break;

    case model::decision_tree::proto::Condition::kContainsBitmapCondition: {
      std::vector<int32_t> items;
      const int num_unique = model_->data_spec()
                                 .columns(attribute)
                                 .categorical()
                                 .number_of_unique_values();
      for (int item_idx = 0; item_idx < num_unique; ++item_idx) {
        if (utils::bitmap::GetValueBit(
                condition.contains_bitmap_condition().elements_bitmap(),
                item_idx)) {
          items.push_back(item_idx);
        }
      }
      return handle_contains_condition(items);
    } break;

    case model::decision_tree::proto::Condition::kContainsCondition: {
      const auto& elements = condition.contains_condition().elements();
      std::vector<int32_t> items(elements.begin(), elements.end());
      return handle_contains_condition(items);
    } break;

    case model::decision_tree::proto::Condition::kObliqueCondition: {
      cur_node.condition_type = ConditionType::OBLIQUE_CONDITION;
      cur_node.threshold_or_offset = condition.oblique_condition().threshold();
      if (ir_.oblique_features.size() >
          std::numeric_limits<FeatureIdx>::max()) {
        return absl::InternalError(
            "Maximum number of oblique features exceeded.");
      }
      cur_node.feature_idx = ir_.oblique_features.size();
      cur_node.num_oblique_features =
          condition.oblique_condition().attributes_size();

      // Store threshold and size in the banks as headers.
      ir_.oblique_weights.push_back(condition.oblique_condition().threshold());
      ir_.oblique_features.push_back(cur_node.num_oblique_features);

      for (int i = 0; i < cur_node.num_oblique_features; ++i) {
        ir_.oblique_features.push_back(column_idx_to_model_feature_idx_.at(
            condition.oblique_condition().attributes(i)));
        ir_.oblique_weights.push_back(condition.oblique_condition().weights(i));
      }
      active_condition_types.insert(ConditionType::OBLIQUE_CONDITION);
    } break;

    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported condition type: ", condition.type_case()));
  }

  return absl::OkStatus();
}

absl::StatusOr<int32_t> ModelIRBuilder::AddToBitsetBank(
    const std::vector<int32_t>& items, const int num_unique_values,
    const std::string& column_name) {
  const int32_t offset = ir_.bitset_bank.size();
  std::vector<bool> bitset(num_unique_values, false);
  for (const auto item : items) {
    if (item >= 0 && item < num_unique_values) {
      bitset[item] = true;
    } else {
      return absl::InvalidArgumentError(absl::Substitute(
          "Invalid item $0 on the bitmap for column $1, should be in range [0, "
          "$2)",
          item, column_name, num_unique_values));
    }
  }
  ir_.bitset_bank.insert(ir_.bitset_bank.end(), bitset.begin(), bitset.end());
  return offset;
}

absl::StatusOr<int64_t> ModelIRBuilder::AddToLeafBank(
    const std::vector<DoubleOrInt64>& values) {
  const int32_t offset = ir_.leaf_value_bank.size();
  for (const auto value : values) {
    STATUS_CHECK(IsDouble(value));
    ir_.leaf_value_bank.push_back(static_cast<float>(AsDouble(value)));
  }
  return static_cast<int64_t>(offset);
}

absl::Status ModelIRBuilder::BuildSpecializedConversions() {
  df_interface_ = dynamic_cast<const model::DecisionForestInterface*>(model_);
  if (!df_interface_) {
    return absl::InvalidArgumentError(
        "The model is not a decision forest model.");
  }
  random_forest_model_ =
      dynamic_cast<const model::random_forest::RandomForestModel*>(model_);
  if (random_forest_model_ != nullptr) {
    ir_.model_type = ModelIR::ModelType::kRandomForest;
  }
  gradient_boosted_trees_model_ = dynamic_cast<
      const model::gradient_boosted_trees::GradientBoostedTreesModel*>(model_);
  if (gradient_boosted_trees_model_ != nullptr) {
    ir_.model_type = ModelIR::ModelType::kGradientBoostedTrees;
  }
  if (random_forest_model_ == nullptr &&
      gradient_boosted_trees_model_ == nullptr) {
    return absl::InvalidArgumentError(
        "Only Random Forest models and Gradient Boosted Trees models are "
        "supported");
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::serving::embed::internal
