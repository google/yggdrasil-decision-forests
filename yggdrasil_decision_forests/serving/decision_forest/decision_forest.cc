/*
 * Copyright 2021 Google LLC.
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

#include "yggdrasil_decision_forests/serving/decision_forest/decision_forest.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/usage.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace decision_forest {

using dataset::VerticalDataset;
using dataset::proto::ColumnType;
using model::decision_tree::NodeWithChildren;
using model::gradient_boosted_trees::GradientBoostedTreesModel;
using model::gradient_boosted_trees::proto::Loss;
using model::random_forest::RandomForestModel;
using ConditionType = model::decision_tree::proto::Condition::TypeCase;

namespace {

// Set the value of a non-leaf node only supporting numerical conditions.
template <typename GenericModel, typename SpecializedModel>
absl::Status SetNonLeafNode(const GenericModel& src_model,
                            const NodeWithChildren& src_node,
                            const int spec_feature_idx,
                            SpecializedModel* dst_model,
                            OneDimensionOutputNumericalFeatureNode* dst_node) {
  static_assert(std::is_same<typename SpecializedModel::NodeType,
                             OneDimensionOutputNumericalFeatureNode>::value,
                "Non supported node type.");

  ASSIGN_OR_RETURN(const auto feature,
                   FindFeatureDef(dst_model->features().fixed_length_features(),
                                  spec_feature_idx));

  dst_node->right_idx = 0;
  dst_node->feature_idx =
      static_cast<typename SpecializedModel::NodeType::FeatureIdx>(
          feature.internal_idx);
  if (src_node.node().condition().condition().has_higher_condition()) {
    dst_node->threshold =
        src_node.node().condition().condition().higher_condition().threshold();
    return absl::OkStatus();
  } else if (src_node.node()
                 .condition()
                 .condition()
                 .has_true_value_condition()) {
    dst_node->threshold = 0.5f;
    return absl::OkStatus();
  } else {
    return absl::InvalidArgumentError(
        "Unexpected non-numerical conditions. This inference engine optimized "
        "for speed only supports numerical conditions. Try another inference "
        "engine in .../decision_forest.h.");
  }
  return absl::OkStatus();
}

// Set the value of a non-leaf node only supporting numerical and categorical
// conditions.
template <typename GenericModel, typename SpecializedModel>
absl::Status SetNonLeafNode(
    const GenericModel& src_model, const NodeWithChildren& src_node,
    const int spec_feature_idx, SpecializedModel* dst_model,
    OneDimensionOutputNumericalAndCategoricalFeatureNode* dst_node) {
  static_assert(
      std::is_same<typename SpecializedModel::NodeType,
                   OneDimensionOutputNumericalAndCategoricalFeatureNode>::value,
      "Non supported node type.");

  ASSIGN_OR_RETURN(const auto feature,
                   FindFeatureDef(dst_model->features().fixed_length_features(),
                                  spec_feature_idx));

  dst_node->right_idx = 0;
  dst_node->feature_idx =
      static_cast<typename SpecializedModel::NodeType::FeatureIdx>(
          feature.internal_idx);

  const auto max_categorical_values =
      sizeof(OneDimensionOutputNumericalAndCategoricalFeatureNode::mask) * 8;

  if (src_node.node().condition().condition().has_higher_condition()) {
    dst_node->threshold =
        src_node.node().condition().condition().higher_condition().threshold();
  } else if (src_node.node()
                 .condition()
                 .condition()
                 .has_true_value_condition()) {
    dst_node->threshold = 0.5f;
  } else if (src_node.node()
                 .condition()
                 .condition()
                 .has_contains_bitmap_condition()) {
    const auto bitmap = src_node.node()
                            .condition()
                            .condition()
                            .contains_bitmap_condition()
                            .elements_bitmap();
    if (bitmap.size() * 8 > max_categorical_values) {
      return absl::InvalidArgumentError(absl::StrCat(
          "This inference engine optimized for speed only supports categorical "
          "attributes with less than ",
          max_categorical_values,
          " possible values. Try another inference engine in "
          ".../decision_forest.h, or limit the "
          "number of possible value of this feature using the dataspec "
          "guide."));
    }
    dst_node->mask = 0;
    std::memcpy(&dst_node->mask, bitmap.data(), bitmap.size());
    dst_node->feature_idx = -dst_node->feature_idx - 1;
  } else if (src_node.node().condition().condition().has_contains_condition()) {
    const auto elements =
        src_node.node().condition().condition().contains_condition().elements();
    dst_node->mask = 0;
    for (int element : elements) {
      if (element > max_categorical_values) {
        return absl::InvalidArgumentError(absl::StrCat(
            "This inference engine optimized for speed only supports "
            "categorical "
            "attributes with less than ",
            max_categorical_values,
            " possible values. Try another inference engine in "
            ".../decision_forest.h, or limit the "
            "number of possible value of this feature using the dataspec "
            "guide."));
      }
      dst_node->mask |= 1 << element;
    }
    dst_node->feature_idx = -dst_node->feature_idx - 1;
  } else {
    return absl::InvalidArgumentError(
        "This inference engine optimized for speed only supports categorical "
        "and numerical conditions.  Try another inference engine in "
        ".../decision_forest.h.");
  }

  return absl::OkStatus();
}

// Creates a "contains" condition in the given non-leaf node.
// Contains conditions are used for categorical and categorical-set features.
template <typename SpecializedModel>
absl::Status SetContainsCondition(
    const dataset::proto::ColumnType type, const std::vector<bool>& mask,
    const bool missing_eval_value, SpecializedModel* dst_model,
    typename SpecializedModel::NodeType* dst_node) {
  using GenericNode = typename SpecializedModel::NodeType;

  const bool is_categorical_set = (type == dataset::proto::CATEGORICAL_SET);
  if ((mask.size() >= sizeof(uint32_t) * 8) || is_categorical_set) {
    dst_node->type =
        is_categorical_set
            ? GenericNode::Type::kCategoricalSetContainsBufferOffset
            : GenericNode::Type::kCategoricalContainsBufferOffset;

    if (dst_model->categorical_mask_buffer.size() >=
        std::numeric_limits<uint32_t>::max()) {
      return absl::InvalidArgumentError("Too much categorical conditions.");
    }
    if (is_categorical_set) {
      // For categorical-set the value "-1" represent the missing value.
      // This value is stored such that
      // "categorical_mask_buffer[categorical_contains_buffer_offset+value]"
      // evaluates correctly. Categorical value uses another mechanism.
      dst_model->categorical_mask_buffer.push_back(missing_eval_value);
    }

    dst_node->categorical_contains_buffer_offset =
        dst_model->categorical_mask_buffer.size();
    dst_model->categorical_mask_buffer.insert(
        dst_model->categorical_mask_buffer.end(), mask.begin(), mask.end());

    const int margin =
        (8 - (dst_model->categorical_mask_buffer.size() % 8)) % 8;
    for (int margin_idx = 0; margin_idx < margin; margin_idx++) {
      dst_model->categorical_mask_buffer.push_back(false);
    }
  } else {
    dst_node->type = GenericNode::Type::kCategoricalContainsMask;
    dst_node->categorical_contains_mask = 0;
    for (int item_idx = 0; item_idx < mask.size(); item_idx++) {
      if (mask[item_idx]) {
        dst_node->categorical_contains_mask |= 1 << item_idx;
      }
    }
  }
  return absl::OkStatus();
}

template <typename SpecializedModel>
absl::Status SetObliqueCondition(
    const model::decision_tree::proto::Condition::Oblique& condition,
    SpecializedModel* dst_model,
    typename SpecializedModel::NodeType* dst_node) {
  using GenericNode = typename SpecializedModel::NodeType;
  using FeatureIdx = typename GenericNode::FeatureIdx;

  // Check validity.
  if (condition.weights_size() != condition.attributes_size()) {
    return absl::InvalidArgumentError("Invalid condition");
  }
  dst_node->type = GenericNode::Type::kNumericalObliqueProjectionIsHigher;
  // The number of features in the oblique projection is stored in a FeatureIdx.
  if (condition.weights_size() >= std::numeric_limits<FeatureIdx>::max()) {
    return absl::InvalidArgumentError("Too many projections");
  }
  if (dst_model->oblique_weights.size() !=
      dst_model->oblique_internal_feature_idxs.size()) {
    return absl::InvalidArgumentError("Inconsistent internal buffers");
  }

  // Weights and attributes
  dst_node->oblique_projection_offset = dst_model->oblique_weights.size();
  for (int projection_idx = 0; projection_idx < condition.weights_size();
       projection_idx++) {
    dst_model->oblique_weights.push_back(condition.weights(projection_idx));
    ASSIGN_OR_RETURN(const auto sub_feature,
                     FindFeatureDef(dst_model->features().input_features(),
                                    condition.attributes(projection_idx)));
    dst_model->oblique_internal_feature_idxs.push_back(
        static_cast<FeatureIdx>(sub_feature.internal_idx));
  }

  // Threshold
  dst_model->oblique_weights.push_back(condition.threshold());
  dst_model->oblique_internal_feature_idxs.push_back(0);

  // Number of projections.
  dst_node->feature_idx = static_cast<FeatureIdx>(condition.weights_size());
  return absl::OkStatus();
}

// Set the value of a non-leaf nodes for "generic" models i.e. models that
// support all (or almost all) types of conditions.
template <
    typename GenericModel, typename SpecializedModel,
    std::enable_if_t<utils::is_same_v<typename SpecializedModel::NodeType,
                                      GenericNode<uint16_t>> ||
                         utils::is_same_v<typename SpecializedModel::NodeType,
                                          GenericNode<uint32_t>>,
                     bool> = true>
absl::Status SetNonLeafNode(const GenericModel& src_model,
                            const NodeWithChildren& src_node,
                            const int spec_feature_idx,
                            SpecializedModel* dst_model,
                            typename SpecializedModel::NodeType* dst_node) {
  using GenericNode = typename SpecializedModel::NodeType;
  using FeatureIdx = typename GenericNode::FeatureIdx;
  ASSIGN_OR_RETURN(
      const auto feature,
      FindFeatureDef(dst_model->features().input_features(), spec_feature_idx));

  dst_node->right_idx = 0;
  dst_node->feature_idx = static_cast<FeatureIdx>(feature.internal_idx);

  const auto& src_condition = src_node.node().condition().condition();

  const auto& attribute_spec =
      src_model.data_spec().columns(src_node.node().condition().attribute());

  switch (src_condition.type_case()) {
    case ConditionType::kHigherCondition:
      if (attribute_spec.type() != dataset::proto::NUMERICAL) {
        return absl::InvalidArgumentError("Non supported condition.");
      }
      dst_node->type = GenericNode::Type::kNumericalIsHigher;
      dst_node->numerical_is_higher_threshold =
          src_condition.higher_condition().threshold();
      break;

    case ConditionType::kTrueValueCondition:
      if (attribute_spec.type() != dataset::proto::BOOLEAN) {
        return absl::InvalidArgumentError("Non supported condition.");
      }
      dst_node->type = GenericNode::Type::kNumericalIsHigher;
      dst_node->numerical_is_higher_threshold = 0.5f;
      break;

    case ConditionType::kDiscretizedHigherCondition: {
      if (attribute_spec.type() != dataset::proto::DISCRETIZED_NUMERICAL) {
        return absl::InvalidArgumentError("Non supported condition.");
      }
      dst_node->type = GenericNode::Type::kNumericalIsHigher;
      const auto discretized_threshold =
          src_condition.discretized_higher_condition().threshold();
      dst_node->numerical_is_higher_threshold =
          attribute_spec.discretized_numerical().boundaries(
              discretized_threshold - 1);
    } break;

    case ConditionType::kContainsCondition: {
      if (attribute_spec.type() != dataset::proto::CATEGORICAL &&
          attribute_spec.type() != dataset::proto::CATEGORICAL_SET) {
        return absl::InvalidArgumentError("Non supported condition.");
      }
      const int num_attribute_classes = dst_model->features()
                                            .data_spec()
                                            .columns(spec_feature_idx)
                                            .categorical()
                                            .number_of_unique_values();
      std::vector<bool> mask(num_attribute_classes);
      for (const auto element : src_condition.contains_condition().elements()) {
        mask[element] = true;
      }
      RETURN_IF_ERROR(SetContainsCondition(
          attribute_spec.type(), mask, src_node.node().condition().na_value(),
          dst_model, dst_node));
    } break;

    case ConditionType::kContainsBitmapCondition: {
      if (attribute_spec.type() != dataset::proto::CATEGORICAL &&
          attribute_spec.type() != dataset::proto::CATEGORICAL_SET) {
        return absl::InvalidArgumentError("Non supported condition.");
      }
      const auto bitmap = src_node.node()
                              .condition()
                              .condition()
                              .contains_bitmap_condition()
                              .elements_bitmap();
      const int num_attribute_classes = dst_model->features()
                                            .data_spec()
                                            .columns(spec_feature_idx)
                                            .categorical()
                                            .number_of_unique_values();
      std::vector<bool> mask;
      utils::bitmap::BitmapToVectorBool(bitmap, num_attribute_classes, &mask);
      RETURN_IF_ERROR(SetContainsCondition(
          attribute_spec.type(), mask, src_node.node().condition().na_value(),
          dst_model, dst_node));
    } break;

    case ConditionType::kObliqueCondition: {
      const auto& condition =
          src_node.node().condition().condition().oblique_condition();
      RETURN_IF_ERROR(SetObliqueCondition(condition, dst_model, dst_node));
    } break;

    default:
      return absl::InvalidArgumentError("Non supported condition.");
  }
  return absl::OkStatus();
}

// Signature of a function able to set the value of a leaf node.
template <typename GenericModel, typename SpecializedModel>
using SetLeafFunctor = std::function<absl::Status(
    const GenericModel& src_model, const NodeWithChildren& src_node,
    SpecializedModel* dst_model,
    typename SpecializedModel::NodeType* dst_node)>;

// Set the leaf of a binary classification Random Forest.
template <typename SpecializedModel>
absl::Status SetLeafNodeRandomForestBinaryClassification(
    const RandomForestModel& src_model, const NodeWithChildren& src_node,
    SpecializedModel* dst_model,
    typename SpecializedModel::NodeType* dst_node) {
  using Node = typename SpecializedModel::NodeType;
  static_assert(
      std::is_same<
          Node, OneDimensionOutputNumericalAndCategoricalFeatureNode>::value ||
          std::is_same<Node, OneDimensionOutputNumericalFeatureNode>::value ||
          std::is_same<Node, GenericNode<uint16_t>>::value ||
          std::is_same<Node, GenericNode<uint32_t>>::value,
      "Non supported node type.");
  if (src_model.winner_take_all_inference()) {
    const int32_t vote = src_node.node().classifier().top_value();
    if (vote == dataset::kOutOfDictionaryItemIndex) {
      return absl::InvalidArgumentError(
          "This inference engine optimized for speed only supports model "
          "outputting "
          "out-of-bag values. This can be caused by two errors: 1) Have rare "
          "label values (by default <10 on the entire training dataset) and "
          "not setting \"min_vocab_frequency\" appropriately. 2) Having "
          "\"is_already_integerized=true\" and providing label with "
          "\"OOB\"(=0) values during training.");
    }
    if (vote > 2) {
      return absl::InvalidArgumentError(
          "The model is not a binary classifier. Try another inference engine "
          "in .../decision_forest.h.");
    }
    *dst_node = Node::Leaf(
        /*.right_idx =*/0,
        /*.feature_idx =*/0,
        /*.label =*/((vote == 2) ? (1.0f / src_model.NumTrees()) : 0.0f));
  } else {
    const auto& distribution = src_node.node().classifier().distribution();
    if (distribution.counts_size() != 3) {
      return absl::InvalidArgumentError(
          "The model is not a binary classifier. You likely used the wrong "
          "optimized model class (see header of "
          "yggdrasil_decision_forests/serving/decision_forest/"
          "decision_forest.h).");
    }
    *dst_node = Node::Leaf(
        /*.right_idx =*/0,
        /*.feature_idx =*/0,
        /*.label =*/
        static_cast<float>(distribution.counts(2) /
                           (distribution.sum() * src_model.NumTrees())));
  }
  return absl::OkStatus();
}

template <typename SpecializedModel>
absl::Status SetLeafNodeRandomForestMulticlassClassification(
    const RandomForestModel& src_model, const NodeWithChildren& src_node,
    SpecializedModel* dst_model,
    typename SpecializedModel::NodeType* dst_node) {
  using Node = typename SpecializedModel::NodeType;
  static_assert(std::is_same<Node, GenericNode<uint16_t>>::value ||
                    std::is_same<Node, GenericNode<uint32_t>>::value,
                "Non supported node type.");
  const auto begin_label_index = dst_model->label_buffer.size();
  dst_model->label_buffer.resize(
      dst_model->label_buffer.size() + dst_model->num_classes, 0.f);
  *dst_node = Node::LeafMulticlassClassification(
      /*.right_idx =*/0,
      /*.feature_idx =*/0,
      /*.type = */ Node::Type::kLeaf,
      /*.label_buffer_offset = */ static_cast<uint32_t>(begin_label_index));

  if (src_model.winner_take_all_inference()) {
    const int32_t vote = src_node.node().classifier().top_value();
    if (vote == dataset::kOutOfDictionaryItemIndex) {
      return absl::InvalidArgumentError(
          "This inference engine optimized for speed only supports model "
          "outputting "
          "out-of-bag values. This can be caused by two errors: 1) Have rare "
          "label values (by default <10 on the entire training dataset) and "
          "not setting \"min_vocab_frequency\" appropriately. 2) Having "
          "\"is_already_integerized=true\" and providing label with "
          "\"OOB\"(=0) values during training.");
    }
    dst_model->label_buffer[begin_label_index + vote - 1] =
        1.f / src_model.NumTrees();
  } else {
    const auto& distribution = src_node.node().classifier().distribution();
    for (int class_idx = 0; class_idx < dst_model->num_classes; class_idx++) {
      dst_model->label_buffer[begin_label_index + class_idx] =
          static_cast<float>(distribution.counts(class_idx + 1) /
                             (distribution.sum() * src_model.NumTrees()));
    }
  }
  return absl::OkStatus();
}

// Set the leaf of a regressive tree.
template <typename GenericModel, typename Node>
absl::Status SetRegressiveLeaf(const GenericModel& src_model,
                               const NodeWithChildren& src_node,
                               const float normalization, Node* dst_node) {
  static_assert(
      std::is_same<Node, OneDimensionOutputNumericalFeatureNode>::value ||
          std::is_same<
              Node,
              OneDimensionOutputNumericalAndCategoricalFeatureNode>::value ||
          std::is_same<Node, GenericNode<uint16_t>>::value ||
          std::is_same<Node, GenericNode<uint32_t>>::value,
      "Non supported node type.");
  *dst_node = Node::Leaf(
      /*.right_idx =*/0,
      /*.feature_idx =*/0,
      /*.label =*/src_node.node().regressor().top_value() / normalization);
  return absl::OkStatus();
}

// Set the leaf of a regression Random Forest.
template <typename SpecializedModel>
absl::Status SetLeafNodeRandomForestRegression(
    const RandomForestModel& src_model, const NodeWithChildren& src_node,
    SpecializedModel* dst_model,
    typename SpecializedModel::NodeType* dst_node) {
  return SetRegressiveLeaf(src_model, src_node, src_model.NumTrees(), dst_node);
}

template <typename SpecializedModel>
absl::Status SetLeafNodeRandomForestCategoricalUplift(
    const RandomForestModel& src_model, const NodeWithChildren& src_node,
    SpecializedModel* dst_model,
    typename SpecializedModel::NodeType* dst_node) {
  using Node = typename SpecializedModel::NodeType;
  static_assert(std::is_same<Node, GenericNode<uint16_t>>::value ||
                    std::is_same<Node, GenericNode<uint32_t>>::value,
                "Non supported node type.");
  const auto begin_label_index = dst_model->label_buffer.size();
  dst_model->label_buffer.resize(
      dst_model->label_buffer.size() + dst_model->num_classes, 0.f);
  *dst_node = Node::LeafCategoricalUplift(
      /*.right_idx =*/0,
      /*.feature_idx =*/0,
      /*.type = */ Node::Type::kLeaf,
      /*.label_buffer_offset = */ static_cast<uint32_t>(begin_label_index));

  for (int output_idx = 0; output_idx < dst_model->num_classes; output_idx++) {
    dst_model->label_buffer[begin_label_index + output_idx] =
        src_node.node().uplift().treatment_effect(output_idx) /
        src_model.NumTrees();
  }

  return absl::OkStatus();
}

template <typename SpecializedModel>
absl::Status SetLeafNodeRandomForestNumericalUplift(
    const RandomForestModel& src_model, const NodeWithChildren& src_node,
    SpecializedModel* dst_model,
    typename SpecializedModel::NodeType* dst_node) {
  using Node = typename SpecializedModel::NodeType;
  static_assert(std::is_same<Node, GenericNode<uint16_t>>::value ||
                    std::is_same<Node, GenericNode<uint32_t>>::value,
                "Non supported node type.");

  if (src_node.node().uplift().treatment_effect_size() != 1) {
    return absl::InvalidArgumentError("Invalid uplift model");
  }

  *dst_node = Node::Leaf(
      /*.right_idx =*/0,
      /*.feature_idx =*/0,
      /*.label =*/src_node.node().uplift().treatment_effect(0) /
          src_model.NumTrees());
  return absl::OkStatus();
}

// Set the leaf of a binary classification Gradient Boosted Trees.
template <typename SpecializedModel>
absl::Status SetLeafGradientBoostedTreesClassification(
    const GradientBoostedTreesModel& src_model,
    const NodeWithChildren& src_node, SpecializedModel* dst_model,
    typename SpecializedModel::NodeType* dst_node) {
  return SetRegressiveLeaf(src_model, src_node, 1.f, dst_node);
}

// Set the leaf of a regression Gradient Boosted Trees.
template <typename SpecializedModel>
absl::Status SetLeafGradientBoostedTreesRegression(
    const GradientBoostedTreesModel& src_model,
    const NodeWithChildren& src_node, SpecializedModel* dst_model,
    typename SpecializedModel::NodeType* dst_node) {
  return SetRegressiveLeaf(src_model, src_node, 1.f, dst_node);
}

// Set the leaf of a ranking Gradient Boosted Trees.
template <typename SpecializedModel>
absl::Status SetLeafGradientBoostedTreesRanking(
    const GradientBoostedTreesModel& src_model,
    const NodeWithChildren& src_node, SpecializedModel* dst_model,
    typename SpecializedModel::NodeType* dst_node) {
  return SetRegressiveLeaf(src_model, src_node, 1.f, dst_node);
}

// Recursively explore the children of a node and output the result in the flat
// node array "specialized_node_array".
//
// Arguments:
//   node: Input node to convert.
//   spec_feature_idx_to_node_feature_idx: Mapping between the index of the
//     data_spec and the exported model.
//   set_leaf and set_non_leaf: Respectively set the content of the leaf and
//      non-leaf nodes.
//   specialized_node_array: Output flat node array.
template <typename GenericModel, typename SpecializedModel>
absl::Status ConvertGenericNodeToFlatNode(
    const GenericModel& src_model, const NodeWithChildren& node,
    const SetLeafFunctor<GenericModel, SpecializedModel> set_node,
    SpecializedModel* dst_model,
    std::vector<typename SpecializedModel::NodeType>* specialized_node_array) {
  if (node.IsLeaf()) {
    // Create a leaf.
    typename SpecializedModel::NodeType dst_node;
    RETURN_IF_ERROR(set_node(src_model, node, dst_model, &dst_node));
    specialized_node_array->push_back(dst_node);
  } else {
    // Create a non-leaf node.
    const int spec_feature_idx = node.node().condition().attribute();
    typename SpecializedModel::NodeType non_leaf_node;
    RETURN_IF_ERROR(SetNonLeafNode(src_model, node, spec_feature_idx, dst_model,
                                   &non_leaf_node));
    const size_t new_node_idx = specialized_node_array->size();
    specialized_node_array->push_back(non_leaf_node);

    // Create its children.
    RETURN_IF_ERROR(ConvertGenericNodeToFlatNode(src_model, *node.neg_child(),
                                                 set_node, dst_model,
                                                 specialized_node_array));
    const int node_offset = specialized_node_array->size() - new_node_idx;
    if (node_offset >=
        std::numeric_limits<
            typename SpecializedModel::NodeType::NodeOffset>::max()) {
      return absl::InvalidArgumentError(
          "Tree with too many nodes for this optimized model format.");
    }
    (*specialized_node_array)[new_node_idx].right_idx = node_offset;
    RETURN_IF_ERROR(ConvertGenericNodeToFlatNode(src_model, *node.pos_child(),
                                                 set_node, dst_model,
                                                 specialized_node_array));
  }
  return absl::OkStatus();
}

// Creates the nodes of a flat model.
template <typename GenericModel, typename SpecializedModel>
absl::Status CreateFlatModelNodes(
    const GenericModel& src_model,
    SetLeafFunctor<GenericModel, SpecializedModel> set_node,
    SpecializedModel* dst_model) {
  dst_model->nodes.clear();
  dst_model->nodes.reserve(src_model.NumNodes());
  dst_model->root_offsets.clear();
  dst_model->root_offsets.reserve(src_model.NumTrees());
  for (const auto& tree : src_model.decision_trees()) {
    dst_model->root_offsets.push_back(dst_model->nodes.size());
    RETURN_IF_ERROR(ConvertGenericNodeToFlatNode(
        src_model, tree->root(), set_node, dst_model, &dst_model->nodes));
  }
  LOG(INFO) << "Model loaded with " << dst_model->root_offsets.size()
            << " root(s), " << dst_model->nodes.size() << " node(s), and "
            << dst_model->features().input_features().size()
            << " input feature(s).";

  return absl::OkStatus();
}

// Final function applied by a Gradient Boosted Trees with
// BINOMIAL_LOG_LIKELIHOOD loss function.
template <typename SpecializedModel>
float ActivationGradientBoostedTreesBinomialLogLikelihood(
    const SpecializedModel& model, const float value) {
  return utils::clamp(
      1.f / (1.f + std::exp(-(value + model.initial_predictions))), 0.f, 1.f);
}

// Final function applied by a Gradient Boosted Trees with
// SQUARED_ERROR loss function.
template <typename SpecializedModel>
float ActivationAddInitialPrediction(const SpecializedModel& model,
                                     const float value) {
  return value + model.initial_predictions;
}

// Final function applied by a Gradient Boosted Trees with
// MULTINOMIAL_LOG_LIKELIHOOD loss function. I.e. this is a softmax function.
template <typename SpecializedModel>
void ActivationGradientBoostedTreesMultinomialLogLikelihood(
    const SpecializedModel& model, float* const values, const int num_values) {
  float* cache = static_cast<float*>(alloca(sizeof(float) * num_values));
  float sum = 0;
  for (int i = 0; i < num_values; i++) {
    const float value = std::exp(values[i]);
    cache[i] = value;
    sum += value;
  }
  const float noramlize = 1.f / sum;
  for (int i = 0; i < num_values; i++) {
    values[i] = cache[i] * noramlize;
  }
}

template <typename SpecializedModel>
void ActivationMultiDimIdentity(const SpecializedModel& model,
                                float* const values, const int num_values) {}

// Templated version of "GenericToSpecializedModel".
template <typename GenericModel, typename SpecializedModel>
absl::Status GenericToSpecializedModelHelper(
    const GenericModel& src_model,
    SetLeafFunctor<GenericModel, SpecializedModel> set_node,
    SpecializedModel* dst_model) {
  if (src_model.task() != SpecializedModel::kTask) {
    return absl::InvalidArgumentError("Wrong model class.");
  }
  src_model.metadata().Export(&dst_model->metadata);

  RETURN_IF_ERROR(InitializeFlatNodeModel(src_model, dst_model));

  return CreateFlatModelNodes(src_model, set_node, dst_model);
}

// A more automatized version of "GenericToSpecializedModelHelper".
template <typename SetLeaf, typename GenericModel, typename SpecializedModel>
absl::Status GenericToSpecializedModelHelper2(SetLeaf set_leaf,
                                              const GenericModel& src,
                                              SpecializedModel* dst) {
  return GenericToSpecializedModelHelper(
      src, SetLeafFunctor<GenericModel, SpecializedModel>(set_leaf), dst);
}

// Checks that a model is a binary classifier.
template <typename GenericModel>
absl::Status CheckBinaryClassification(const GenericModel& src) {
  if (src.label_col_spec().categorical().number_of_unique_values() != 3) {
    return absl::InvalidArgumentError("The model is not a binary classifier.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status GenericToSpecializedModel(
    const RandomForestModel& src,
    RandomForestBinaryClassificationNumericalFeatures* dst) {
  return GenericToSpecializedModelHelper(
      src,
      SetLeafFunctor<RandomForestModel,
                     RandomForestBinaryClassificationNumericalFeatures>(
          SetLeafNodeRandomForestBinaryClassification<
              RandomForestBinaryClassificationNumericalFeatures>),
      dst);
}

absl::Status GenericToSpecializedModel(
    const RandomForestModel& src,
    RandomForestBinaryClassificationNumericalAndCategoricalFeatures* dst) {
  return GenericToSpecializedModelHelper(
      src,
      SetLeafFunctor<
          RandomForestModel,
          RandomForestBinaryClassificationNumericalAndCategoricalFeatures>(
          SetLeafNodeRandomForestBinaryClassification<
              RandomForestBinaryClassificationNumericalAndCategoricalFeatures>),
      dst);
}

absl::Status GenericToSpecializedModel(
    const RandomForestModel& src, RandomForestRegressionNumericalOnly* dst) {
  return GenericToSpecializedModelHelper(
      src,
      SetLeafFunctor<RandomForestModel, RandomForestRegressionNumericalOnly>(
          SetLeafNodeRandomForestRegression<
              RandomForestRegressionNumericalOnly>),
      dst);
}

absl::Status GenericToSpecializedModel(
    const RandomForestModel& src,
    RandomForestRegressionNumericalAndCategorical* dst) {
  return GenericToSpecializedModelHelper(
      src,
      SetLeafFunctor<RandomForestModel,
                     RandomForestRegressionNumericalAndCategorical>(
          SetLeafNodeRandomForestRegression<
              RandomForestRegressionNumericalAndCategorical>),
      dst);
}

absl::Status GenericToSpecializedModel(
    const GradientBoostedTreesModel& src,
    GradientBoostedTreesBinaryClassificationNumericalOnly* dst) {
  if (src.loss() != Loss::BINOMIAL_LOG_LIKELIHOOD ||
      src.initial_predictions().size() != 1) {
    return absl::InvalidArgumentError(
        "The GBT is not trained for binary classification.");
  }

  RETURN_IF_ERROR(GenericToSpecializedModelHelper(
      src,
      SetLeafFunctor<GradientBoostedTreesModel,
                     GradientBoostedTreesBinaryClassificationNumericalOnly>(
          SetLeafGradientBoostedTreesClassification<
              GradientBoostedTreesBinaryClassificationNumericalOnly>),
      dst));

  dst->initial_predictions = src.initial_predictions()[0];
  return absl::OkStatus();
}

absl::Status GenericToSpecializedModel(
    const GradientBoostedTreesModel& src,
    GradientBoostedTreesBinaryClassificationNumericalAndCategorical* dst) {
  if (src.loss() != Loss::BINOMIAL_LOG_LIKELIHOOD ||
      src.initial_predictions().size() != 1) {
    return absl::InvalidArgumentError(
        "The GBT is not trained for binary classification.");
  }

  RETURN_IF_ERROR(GenericToSpecializedModelHelper(
      src,
      SetLeafFunctor<
          GradientBoostedTreesModel,
          GradientBoostedTreesBinaryClassificationNumericalAndCategorical>(
          SetLeafGradientBoostedTreesClassification<
              GradientBoostedTreesBinaryClassificationNumericalAndCategorical>),
      dst));

  dst->initial_predictions = src.initial_predictions()[0];
  return absl::OkStatus();
}

absl::Status GenericToSpecializedModel(
    const GradientBoostedTreesModel& src,
    GradientBoostedTreesRegressionNumericalOnly* dst) {
  if (src.loss() != Loss::SQUARED_ERROR ||
      src.initial_predictions().size() != 1) {
    return absl::InvalidArgumentError("The GBT is not trained for regression.");
  }

  RETURN_IF_ERROR(GenericToSpecializedModelHelper(
      src,
      SetLeafFunctor<GradientBoostedTreesModel,
                     GradientBoostedTreesRegressionNumericalOnly>(
          SetLeafGradientBoostedTreesRegression<
              GradientBoostedTreesRegressionNumericalOnly>),
      dst));

  dst->initial_predictions = src.initial_predictions()[0];
  return absl::OkStatus();
}

absl::Status GenericToSpecializedModel(
    const GradientBoostedTreesModel& src,
    GradientBoostedTreesRegressionNumericalAndCategorical* dst) {
  if (src.loss() != Loss::SQUARED_ERROR ||
      src.initial_predictions().size() != 1) {
    return absl::InvalidArgumentError("The GBT is not trained for regression.");
  }

  RETURN_IF_ERROR(GenericToSpecializedModelHelper(
      src,
      SetLeafFunctor<GradientBoostedTreesModel,
                     GradientBoostedTreesRegressionNumericalAndCategorical>(
          SetLeafGradientBoostedTreesRegression<
              GradientBoostedTreesRegressionNumericalAndCategorical>),
      dst));

  dst->initial_predictions = src.initial_predictions()[0];
  return absl::OkStatus();
}

absl::Status GenericToSpecializedModel(
    const GradientBoostedTreesModel& src,
    GradientBoostedTreesRankingNumericalOnly* dst) {
  if (src.loss() != Loss::LAMBDA_MART_NDCG5 ||
      src.initial_predictions().size() != 1) {
    return absl::InvalidArgumentError("The GBT is not trained for ranking.");
  }

  RETURN_IF_ERROR(GenericToSpecializedModelHelper(
      src,
      SetLeafFunctor<GradientBoostedTreesModel,
                     GradientBoostedTreesRankingNumericalOnly>(
          SetLeafGradientBoostedTreesRanking<
              GradientBoostedTreesRankingNumericalOnly>),
      dst));

  dst->initial_predictions = src.initial_predictions()[0];
  return absl::OkStatus();
}

absl::Status GenericToSpecializedModel(
    const GradientBoostedTreesModel& src,
    GradientBoostedTreesRankingNumericalAndCategorical* dst) {
  if (src.loss() != Loss::LAMBDA_MART_NDCG5 ||
      src.initial_predictions().size() != 1) {
    return absl::InvalidArgumentError("The GBT is not trained for ranking.");
  }

  RETURN_IF_ERROR(GenericToSpecializedModelHelper(
      src,
      SetLeafFunctor<GradientBoostedTreesModel,
                     GradientBoostedTreesRankingNumericalAndCategorical>(
          SetLeafGradientBoostedTreesRanking<
              GradientBoostedTreesRankingNumericalAndCategorical>),
      dst));

  dst->initial_predictions = src.initial_predictions()[0];
  return absl::OkStatus();
}

template <>
absl::Status GenericToSpecializedModel(const RandomForestModel& src,
                                       RandomForestBinaryClassification* dst) {
  RETURN_IF_ERROR(CheckBinaryClassification(src));
  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafNodeRandomForestBinaryClassification<DstType>, src, dst);
}

template <>
absl::Status GenericToSpecializedModel(
    const RandomForestModel& src, RandomForestMulticlassClassification* dst) {
  dst->num_classes =
      src.label_col_spec().categorical().number_of_unique_values() - 1;

  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafNodeRandomForestMulticlassClassification<DstType>, src, dst);
}

template <>
absl::Status GenericToSpecializedModel(const RandomForestModel& src,
                                       RandomForestRegression* dst) {
  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafNodeRandomForestRegression<DstType>, src, dst);
}

template <>
absl::Status GenericToSpecializedModel(const RandomForestModel& src,
                                       RandomForestCategoricalUplift* dst) {
  dst->num_classes =
      src.label_col_spec().categorical().number_of_unique_values() - 2;
  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafNodeRandomForestCategoricalUplift<DstType>, src, dst);
}

template <>
absl::Status GenericToSpecializedModel(const RandomForestModel& src,
                                       RandomForestNumericalUplift* dst) {
  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafNodeRandomForestNumericalUplift<DstType>, src, dst);
}

template <>
absl::Status GenericToSpecializedModel(
    const RandomForestModel& src,
    GenericRandomForestBinaryClassification<uint32_t>* dst) {
  RETURN_IF_ERROR(CheckBinaryClassification(src));

  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafNodeRandomForestBinaryClassification<DstType>, src, dst);
}

template <>
absl::Status GenericToSpecializedModel(
    const RandomForestModel& src,
    GenericRandomForestMulticlassClassification<uint32_t>* dst) {
  dst->num_classes =
      src.label_col_spec().categorical().number_of_unique_values() - 1;

  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafNodeRandomForestMulticlassClassification<DstType>, src, dst);
}

template <>
absl::Status GenericToSpecializedModel(
    const RandomForestModel& src,
    GenericRandomForestRegression<uint32_t>* dst) {
  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafNodeRandomForestRegression<DstType>, src, dst);
}

template <>
absl::Status GenericToSpecializedModel(
    const RandomForestModel& src,
    GenericRandomForestCategoricalUplift<uint32_t>* dst) {
  dst->num_classes =
      src.label_col_spec().categorical().number_of_unique_values() - 2;
  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafNodeRandomForestCategoricalUplift<DstType>, src, dst);
}

template <>
absl::Status GenericToSpecializedModel(
    const RandomForestModel& src,
    GenericRandomForestNumericalUplift<uint32_t>* dst) {
  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafNodeRandomForestNumericalUplift<DstType>, src, dst);
}

template <>
absl::Status GenericToSpecializedModel(
    const GradientBoostedTreesModel& src,
    GradientBoostedTreesBinaryClassification* dst) {
  if (src.loss() != Loss::BINOMIAL_LOG_LIKELIHOOD ||
      src.initial_predictions().size() != 1) {
    return absl::InvalidArgumentError(
        "The Gradient Boosted Tree is not trained for binary classification.");
  }
  dst->initial_predictions = src.initial_predictions()[0];

  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafGradientBoostedTreesClassification<DstType>, src, dst);
}

template <>
absl::Status GenericToSpecializedModel(
    const GradientBoostedTreesModel& src,
    GenericGradientBoostedTreesBinaryClassification<uint32_t>* dst) {
  if (src.loss() != Loss::BINOMIAL_LOG_LIKELIHOOD ||
      src.initial_predictions().size() != 1) {
    return absl::InvalidArgumentError(
        "The Gradient Boosted Tree is not trained for binary classification.");
  }
  dst->initial_predictions = src.initial_predictions()[0];
  dst->output_logits = src.output_logits();

  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafGradientBoostedTreesClassification<DstType>, src, dst);
}

template <>
absl::Status GenericToSpecializedModel(
    const GradientBoostedTreesModel& src,
    GradientBoostedTreesMulticlassClassification* dst) {
  if (src.loss() != Loss::MULTINOMIAL_LOG_LIKELIHOOD) {
    return absl::InvalidArgumentError(
        "The Gradient Boosted Tree is not trained for multi-class "
        "classification.");
  }
  dst->num_classes =
      src.label_col_spec().categorical().number_of_unique_values() - 1;
  dst->initial_predictions = src.initial_predictions();
  dst->output_logits = src.output_logits();

  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafGradientBoostedTreesClassification<DstType>, src, dst);
}

template <>
absl::Status GenericToSpecializedModel(const GradientBoostedTreesModel& src,
                                       GradientBoostedTreesRegression* dst) {
  if (src.loss() != Loss::SQUARED_ERROR ||
      src.initial_predictions().size() != 1) {
    return absl::InvalidArgumentError(
        "The Gradient Boosted Tree is not trained for regression.");
  }

  dst->initial_predictions = src.initial_predictions()[0];

  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafGradientBoostedTreesRegression<DstType>, src, dst);
}

template <>
absl::Status GenericToSpecializedModel(const GradientBoostedTreesModel& src,
                                       GradientBoostedTreesRanking* dst) {
  if (src.loss() != Loss::LAMBDA_MART_NDCG5 ||
      src.initial_predictions().size() != 1) {
    return absl::InvalidArgumentError(
        "The Gradient Boosted Tree is not trained for ranking.");
  }

  dst->initial_predictions = src.initial_predictions()[0];

  using DstType = std::remove_pointer<decltype(dst)>::type;
  return GenericToSpecializedModelHelper2(
      SetLeafGradientBoostedTreesRegression<DstType>, src, dst);
}

template <typename Value>
absl::Status LoadFlatBatchFromDataset(
    const VerticalDataset& dataset, VerticalDataset::row_t begin_example_idx,
    VerticalDataset::row_t end_example_idx,
    const std::vector<std::string>& feature_names,
    std::vector<Value>* flat_examples, const ExampleFormat example_format,
    absl::optional<int64_t> batch_size,
    const std::function<utils::StatusOr<Value>(
        const int feature_idx, const int example_idx,
        const std::vector<int>& node_feature_idx_to_spec_feature_idx)>
        get_value) {
  flat_examples->clear();
  flat_examples->reserve((end_example_idx - begin_example_idx) *
                         feature_names.size());

  // Index the dataset feature indices.
  std::vector<int> node_feature_idx_to_spec_feature_idx;
  node_feature_idx_to_spec_feature_idx.reserve(feature_names.size());
  for (int node_feature_idx = 0; node_feature_idx < feature_names.size();
       ++node_feature_idx) {
    const auto& feature_name = feature_names[node_feature_idx];
    const int spec_feature_idx = dataset.ColumnNameToColumnIdx(feature_name);
    if (spec_feature_idx == -1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Feature \"", feature_name, "\" no found in the dataset."));
    }
    node_feature_idx_to_spec_feature_idx.push_back(spec_feature_idx);
  }

  switch (example_format) {
    case ExampleFormat::FORMAT_EXAMPLE_MAJOR:
      for (VerticalDataset::row_t example_idx = begin_example_idx;
           example_idx < end_example_idx; ++example_idx) {
        for (int node_feature_idx = 0; node_feature_idx < feature_names.size();
             ++node_feature_idx) {
          ASSIGN_OR_RETURN(auto value,
                           get_value(node_feature_idx, example_idx,
                                     node_feature_idx_to_spec_feature_idx));
          flat_examples->push_back(value);
        }
      }
      break;

    case ExampleFormat::FORMAT_FEATURE_MAJOR:
      if (batch_size.has_value()) {
        const auto batch_size_value = batch_size.value();
        const auto num_examples = end_example_idx - begin_example_idx;
        const auto num_batches =
            (num_examples + batch_size_value - 1) / batch_size_value;
        for (int64_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
          const int64_t begin_batch_example_idx =
              begin_example_idx + batch_idx * batch_size_value;
          const int64_t end_batch_example_idx = std::min(
              begin_batch_example_idx + batch_size_value, end_example_idx);
          for (int node_feature_idx = 0;
               node_feature_idx < feature_names.size(); ++node_feature_idx) {
            for (VerticalDataset::row_t example_idx = begin_batch_example_idx;
                 example_idx < end_batch_example_idx; ++example_idx) {
              ASSIGN_OR_RETURN(auto value,
                               get_value(node_feature_idx, example_idx,
                                         node_feature_idx_to_spec_feature_idx));
              flat_examples->push_back(value);
            }
          }
        }

      } else {
        for (int node_feature_idx = 0; node_feature_idx < feature_names.size();
             ++node_feature_idx) {
          for (VerticalDataset::row_t example_idx = begin_example_idx;
               example_idx < end_example_idx; ++example_idx) {
            ASSIGN_OR_RETURN(auto value,
                             get_value(node_feature_idx, example_idx,
                                       node_feature_idx_to_spec_feature_idx));
            flat_examples->push_back(value);
          }
        }
      }
      break;

    default:
      return absl::InvalidArgumentError("Example format not supported.");
  }

  return absl::OkStatus();
}

absl::Status LoadFlatBatchFromDataset(
    const VerticalDataset& dataset, VerticalDataset::row_t begin_example_idx,
    VerticalDataset::row_t end_example_idx,
    const std::vector<std::string>& feature_names,
    const std::vector<NumericalOrCategoricalValue>& na_replacement_values,
    std::vector<float>* flat_examples, const ExampleFormat example_format,
    absl::optional<int64_t> batch_size) {
  // Gather and store the feature values.
  const auto get_value =
      [&](const int node_feature_idx, const int example_idx,
          const std::vector<int>& node_feature_idx_to_spec_feature_idx)
      -> utils::StatusOr<float> {
    const int spec_feature_idx =
        node_feature_idx_to_spec_feature_idx[node_feature_idx];
    if (dataset.column(spec_feature_idx)->type() != ColumnType::NUMERICAL) {
      return absl::InvalidArgumentError(
          absl::StrCat("\"", feature_names[node_feature_idx],
                       "\" feature's type is not supported"));
    }
    const auto* numerical_feature_data =
        dataset.ColumnWithCast<VerticalDataset::NumericalColumn>(
            spec_feature_idx);
    float feature_value = numerical_feature_data->values()[example_idx];
    if (std::isnan(feature_value)) {
      feature_value = na_replacement_values[node_feature_idx].numerical_value;
    }
    return feature_value;
  };

  return LoadFlatBatchFromDataset<float>(
      dataset, begin_example_idx, end_example_idx, feature_names, flat_examples,
      example_format, batch_size, get_value);
}

absl::Status LoadFlatBatchFromDataset(
    const VerticalDataset& dataset, VerticalDataset::row_t begin_example_idx,
    VerticalDataset::row_t end_example_idx,
    const std::vector<std::string>& feature_names,
    const std::vector<NumericalOrCategoricalValue>& na_replacement_values,
    std::vector<NumericalOrCategoricalValue>* flat_examples,
    const ExampleFormat example_format, absl::optional<int64_t> batch_size) {
  // Gather and store the feature values.
  const auto get_value =
      [&](const int node_feature_idx, const int example_idx,
          const std::vector<int>& node_feature_idx_to_spec_feature_idx)
      -> utils::StatusOr<NumericalOrCategoricalValue> {
    const int spec_feature_idx =
        node_feature_idx_to_spec_feature_idx[node_feature_idx];
    NumericalOrCategoricalValue feature_value;
    if (dataset.column(spec_feature_idx)->type() == ColumnType::NUMERICAL) {
      const auto* numerical_feature_data =
          dataset.ColumnWithCast<VerticalDataset::NumericalColumn>(
              spec_feature_idx);
      feature_value.numerical_value =
          numerical_feature_data->values()[example_idx];
      if (std::isnan(feature_value.numerical_value)) {
        feature_value = na_replacement_values[node_feature_idx];
      }
    } else if (dataset.column(spec_feature_idx)->type() ==
               ColumnType::CATEGORICAL) {
      const auto* categorical_feature_data =
          dataset.ColumnWithCast<VerticalDataset::CategoricalColumn>(
              spec_feature_idx);
      feature_value.categorical_value =
          categorical_feature_data->values()[example_idx];
      if (feature_value.categorical_value ==
          VerticalDataset::CategoricalColumn::kNaValue) {
        feature_value = na_replacement_values[node_feature_idx];
      }
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("\"", feature_names[node_feature_idx],
                       "\" feature's type is not supported"));
    }
    return feature_value;
  };

  return LoadFlatBatchFromDataset<NumericalOrCategoricalValue>(
      dataset, begin_example_idx, end_example_idx, feature_names, flat_examples,
      example_format, batch_size, get_value);
}

// Identity transformation for the output of a decision forest model.
// Default value for the "FinalTransform" argument in "PredictHelper".
//
// Note: Lambda default segfaults clang.
template <typename Model>
float Idendity(const Model& model, const float value) {
  return value;
}

template <typename Model>
float Clamp01(const Model& model, const float value) {
  return utils::clamp(value, 0.f, 1.f);
}

// Evaluates a numerical condition.
inline bool EvalCondition(const OneDimensionOutputNumericalFeatureNode* node,
                          const float* example) {
  return example[node->feature_idx] >= node->threshold;
}

// Evaluates a numerical or categorical condition.
inline bool EvalCondition(
    const OneDimensionOutputNumericalAndCategoricalFeatureNode* node,
    const NumericalOrCategoricalValue* example) {
  if (node->feature_idx >= 0) {
    // Numerical condition.
    return example[node->feature_idx].numerical_value >= node->threshold;
  } else {
    // Categorical condition.
    const uint32_t example_mask =
        1 << example[-(node->feature_idx + 1)].categorical_value;
    return (example_mask & node->mask) != 0;
  }
}

template <typename Model>
inline bool EvalCondition(const typename Model::NodeType* node,
                          const typename Model::ExampleSet& examples,
                          const int example_idx, const Model& model) {
  using GenericNode = typename Model::NodeType;
  switch (node->type) {
    case GenericNode::Type::kNumericalIsHigher: {
      const auto attribute_value =
          examples.GetNumerical(example_idx, {node->feature_idx}, model);
      return attribute_value >= node->numerical_is_higher_threshold;
    }

    case GenericNode::Type::kCategoricalContainsMask: {
      const uint32_t attribute_value =
          examples.GetCategoricalInt(example_idx, {node->feature_idx}, model);
      return ((1 << attribute_value) & node->categorical_contains_mask) != 0;
    }

    case GenericNode::Type::kCategoricalContainsBufferOffset: {
      const auto attribute_value =
          examples.GetCategoricalInt(example_idx, {node->feature_idx}, model);
      return model
          .categorical_mask_buffer[node->categorical_contains_buffer_offset +
                                   attribute_value];
    }

    case GenericNode::Type::kCategoricalSetContainsBufferOffset: {
      const auto& range_values =
          examples.InternalCategoricalSetBeginAndEnds()
              [node->feature_idx * examples.NumberOfExamples() + example_idx];
      for (int value_idx = range_values.begin; value_idx < range_values.end;
           value_idx++) {
        const auto attribute_value =
            examples.InternalCategoricalItemBuffer()[value_idx];
        if (model.categorical_mask_buffer
                [node->categorical_contains_buffer_offset + attribute_value]) {
          return true;
        }
      }
      return false;
    }

    case GenericNode::Type::kNumericalObliqueProjectionIsHigher: {
      float sum = 0;
      const auto attributes = model.oblique_internal_feature_idxs.begin() +
                              node->oblique_projection_offset;
      const auto weights =
          model.oblique_weights.begin() + node->oblique_projection_offset;

      const uint32_t num_projection = node->feature_idx;
      for (uint32_t projection_idx = 0; projection_idx < num_projection;
           projection_idx++) {
        const auto attribute_value = examples.GetNumerical(
            example_idx, {attributes[projection_idx]}, model);
        const float weight = weights[projection_idx];
        sum += weight * attribute_value;
      }
      return sum >= model.oblique_weights[node->oblique_projection_offset +
                                          num_projection];
    }

    default:
      DCHECK(false);
      return false;
  }
}

// Basic inference of a decision forest on a set of trees.
template <typename Model,
          float (*FinalTransform)(const Model&, const float) = Idendity<Model>>
inline void PredictHelper(
    const Model& model, const std::vector<typename Model::ValueType>& examples,
    int num_examples, std::vector<float>* predictions) {
  utils::usage::OnInference(num_examples, model.metadata);
  const int num_features = model.features().fixed_length_features().size();
  predictions->resize(num_examples);
  for (int example_idx = 0; example_idx < num_examples; ++example_idx) {
    float output = 0.f;
    if (num_features > 0) {
      const auto* sample = &examples[example_idx * num_features];
      for (const auto root_node_idx : model.root_offsets) {
        const auto* node = &model.nodes[root_node_idx];
        while (node->right_idx) {
          node += EvalCondition(node, sample) ? node->right_idx : 1;
        }
        output += node->label;
      }
    }
    (*predictions)[example_idx] = FinalTransform(model, output);
  }
}

template <typename Model,
          float (*FinalTransform)(const Model&, const float) /*= Idendity*/>
inline void PredictHelper(const Model& model,
                          const typename Model::ExampleSet& examples,
                          int num_examples, std::vector<float>* predictions) {
  utils::usage::OnInference(num_examples, model.metadata);
  predictions->resize(num_examples);
  for (int example_idx = 0; example_idx < num_examples; ++example_idx) {
    float output = 0.f;
    for (const auto root_node_idx : model.root_offsets) {
      const auto* node = &model.nodes[root_node_idx];
      while (node->right_idx) {
        node += EvalCondition(node, examples, example_idx, model)
                    ? node->right_idx
                    : 1;
      }
      output += node->label;
    }
    (*predictions)[example_idx] = FinalTransform(model, output);
  }
}

template <typename Model,
          float (*FinalTransform)(const Model&, const float) /*= Idendity*/>
inline void PredictHelperMultiDimensionTrees(
    const Model& model, const typename Model::ExampleSet& examples,
    int num_examples, std::vector<float>* predictions) {
  utils::usage::OnInference(num_examples, model.metadata);
  predictions->assign(num_examples * model.num_classes, 0.f);
  float* cur_predictions = &(*predictions)[0];
  for (int example_idx = 0; example_idx < num_examples; ++example_idx) {
    for (const auto root_node_idx : model.root_offsets) {
      const auto* node = &model.nodes[root_node_idx];
      while (node->right_idx) {
        node += EvalCondition(node, examples, example_idx, model)
                    ? node->right_idx
                    : 1;
      }
      for (int class_idx = 0; class_idx < model.num_classes; class_idx++) {
        cur_predictions[class_idx] +=
            model.label_buffer[node->label_buffer_offset + class_idx];
      }
    }
    for (int class_idx = 0; class_idx < model.num_classes; class_idx++) {
      cur_predictions[class_idx] =
          FinalTransform(model, cur_predictions[class_idx]);
    }
    cur_predictions += model.num_classes;
  }
}

template <typename Model,
          void (*FinalTransform)(const Model&, float* const, const int)>
inline void PredictHelperMultiDimensionFromSingleDimensionTrees(
    const Model& model, const typename Model::ExampleSet& examples,
    int num_examples, std::vector<float>* predictions) {
  predictions->assign(num_examples * model.num_classes, 0.f);
  float* cur_predictions = &(*predictions)[0];
  for (int example_idx = 0; example_idx < num_examples; ++example_idx) {
    int class_idx = 0;
    for (const auto root_node_idx : model.root_offsets) {
      const auto* node = &model.nodes[root_node_idx];
      while (node->right_idx) {
        node += EvalCondition(node, examples, example_idx, model)
                    ? node->right_idx
                    : 1;
      }
      cur_predictions[class_idx] += node->label;
      class_idx = (class_idx + 1) % model.num_classes;
    }
    FinalTransform(model, cur_predictions, model.num_classes);
    cur_predictions += model.num_classes;
  }
}

// See the documentation of "PredictOptimizedV1".
template <typename Model,
          float (*FinalTransform)(const Model&, const float) = Idendity<Model>,
          int kTreeBatchSize = 5>
inline void PredictHelperOptimizedV1(
    const Model& model, const std::vector<typename Model::ValueType>& examples,
    int num_examples, std::vector<float>* predictions) {
  utils::usage::OnInference(num_examples, model.metadata);
  // A Group of "kTreeBatchSize" trees is called a "tree batch".
  predictions->resize(num_examples);

  if (num_examples == 0) {
    return;
  }

  const int num_tree_batches = model.root_offsets.size() / kTreeBatchSize;
  const int num_remaining_trees =
      model.root_offsets.size() - num_tree_batches * kTreeBatchSize;

  // The active nodes in the current tree batch. If "nodes[i]==nullptr", the
  // tree "i" is disabled i.e. it reached a leaf.
  const typename Model::NodeType* nodes[kTreeBatchSize];

  // The number of actives nodes.
  int num_active;

  // Number of input features for the model.
  const int num_features = model.features().fixed_length_features().size();

  // Select the first example.
  // Note: The examples are stored example-major/feature-minor.
  const typename Model::ValueType* sample = &examples[0];
  for (size_t example_idx = 0; example_idx < num_examples; ++example_idx) {
    // Accumulator of the predictions for the current example.
    float output = 0.f;

    // Select the first tree bath.
    auto current_root_node_offset = &model.root_offsets[0];

    for (int tree_batch_idx = 0; tree_batch_idx < num_tree_batches;
         ++tree_batch_idx) {
      // Initialize "nodes" to the roots of the "kTreeBatchSize" trees in the
      // tree batch.
      //
      // Note: "#pragma clang loop unroll(full)" ensures that the loop is
      // unrolled.
#pragma clang loop unroll(full)
      for (int tree_in_batch_idx = 0; tree_in_batch_idx < kTreeBatchSize;
           ++tree_in_batch_idx) {
        nodes[tree_in_batch_idx] =
            &model.nodes[*(current_root_node_offset + tree_in_batch_idx)];
      }
      current_root_node_offset += kTreeBatchSize;
      num_active = kTreeBatchSize;

      // While not all nodes are disabled.
      while (num_active) {
#pragma clang loop unroll(full)
        for (int tree_in_batch_idx = 0; tree_in_batch_idx < kTreeBatchSize;
             ++tree_in_batch_idx) {
          if (nodes[tree_in_batch_idx]) {
            if (nodes[tree_in_batch_idx]->right_idx) {
              // Evaluates the node and go to the correct child.
              nodes[tree_in_batch_idx] +=
                  EvalCondition(nodes[tree_in_batch_idx], sample)
                      ? nodes[tree_in_batch_idx]->right_idx
                      : 1;
            } else {
              // Add the node value to the prediction accumulator.
              output += nodes[tree_in_batch_idx]->label;
              // Disable the node
              --num_active;
              nodes[tree_in_batch_idx] = nullptr;
            }
          }
        }
      }
    }

    for (int tree_in_batch_idx = 0; tree_in_batch_idx < num_remaining_trees;
         tree_in_batch_idx++) {
      auto node = &model.nodes[*(current_root_node_offset + tree_in_batch_idx)];
      while (node->right_idx) {
        node += EvalCondition(node, sample) ? node->right_idx : 1;
      }
      output += node->label;
    }

    // Move to the next example.
    sample += num_features;
    // Store the prediction accumulator result.
    (*predictions)[example_idx] = FinalTransform(model, output);
  }
}

void Predict(const RandomForestBinaryClassificationNumericalFeatures& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Clamp01>(
      model, examples, num_examples, predictions);
}

void Predict(
    const RandomForestBinaryClassificationNumericalAndCategoricalFeatures&
        model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Clamp01>(
      model, examples, num_examples, predictions);
}

void Predict(const GradientBoostedTreesBinaryClassificationNumericalOnly& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationGradientBoostedTreesBinomialLogLikelihood>(
      model, examples, num_examples, predictions);
}

void Predict(
    const GradientBoostedTreesBinaryClassificationNumericalAndCategorical&
        model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationGradientBoostedTreesBinomialLogLikelihood>(
      model, examples, num_examples, predictions);
}

void Predict(const RandomForestRegressionNumericalOnly& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Idendity>(
      model, examples, num_examples, predictions);
}

void Predict(const RandomForestRegressionNumericalAndCategorical& model,
             const std::vector<NumericalOrCategoricalValue>& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Idendity>(
      model, examples, num_examples, predictions);
}

void Predict(const GradientBoostedTreesRegressionNumericalOnly& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationAddInitialPrediction>(model, examples, num_examples,
                                                predictions);
}

void Predict(const GradientBoostedTreesRegressionNumericalAndCategorical& model,
             const std::vector<NumericalOrCategoricalValue>& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationAddInitialPrediction>(model, examples, num_examples,
                                                predictions);
}

void Predict(const GradientBoostedTreesRankingNumericalOnly& model,
             const std::vector<float>& examples, int num_examples,
             std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationAddInitialPrediction>(model, examples, num_examples,
                                                predictions);
}

void Predict(const GradientBoostedTreesRankingNumericalAndCategorical& model,
             const std::vector<NumericalOrCategoricalValue>& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationAddInitialPrediction>(model, examples, num_examples,
                                                predictions);
}

void PredictOptimizedV1(
    const RandomForestBinaryClassificationNumericalFeatures& model,
    const std::vector<float>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<
      RandomForestBinaryClassificationNumericalFeatures,
      Idendity<RandomForestBinaryClassificationNumericalFeatures>>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(
    const RandomForestBinaryClassificationNumericalAndCategoricalFeatures&
        model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<
      RandomForestBinaryClassificationNumericalAndCategoricalFeatures,
      Idendity<
          RandomForestBinaryClassificationNumericalAndCategoricalFeatures>>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(
    const GradientBoostedTreesBinaryClassificationNumericalOnly& model,
    const std::vector<float>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<
      GradientBoostedTreesBinaryClassificationNumericalOnly,
      ActivationGradientBoostedTreesBinomialLogLikelihood>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(
    const GradientBoostedTreesBinaryClassificationNumericalAndCategorical&
        model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<
      GradientBoostedTreesBinaryClassificationNumericalAndCategorical,
      ActivationGradientBoostedTreesBinomialLogLikelihood>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(const RandomForestRegressionNumericalOnly& model,
                        const std::vector<float>& examples, int num_examples,
                        std::vector<float>* predictions) {
  PredictHelperOptimizedV1<RandomForestRegressionNumericalOnly,
                           Idendity<RandomForestRegressionNumericalOnly>>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(
    const RandomForestRegressionNumericalAndCategorical& model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<
      RandomForestRegressionNumericalAndCategorical,
      Idendity<RandomForestRegressionNumericalAndCategorical>>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(
    const GradientBoostedTreesRegressionNumericalOnly& model,
    const std::vector<float>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<GradientBoostedTreesRegressionNumericalOnly,
                           ActivationAddInitialPrediction>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(
    const GradientBoostedTreesRegressionNumericalAndCategorical& model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<
      GradientBoostedTreesRegressionNumericalAndCategorical,
      ActivationAddInitialPrediction>(model, examples, num_examples,
                                      predictions);
}

void PredictOptimizedV1(const GradientBoostedTreesRankingNumericalOnly& model,
                        const std::vector<float>& examples, int num_examples,
                        std::vector<float>* predictions) {
  PredictHelperOptimizedV1<GradientBoostedTreesRankingNumericalOnly,
                           ActivationAddInitialPrediction>(
      model, examples, num_examples, predictions);
}

void PredictOptimizedV1(
    const GradientBoostedTreesRankingNumericalAndCategorical& model,
    const std::vector<NumericalOrCategoricalValue>& examples, int num_examples,
    std::vector<float>* predictions) {
  PredictHelperOptimizedV1<GradientBoostedTreesRankingNumericalAndCategorical,
                           ActivationAddInitialPrediction>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(
    const RandomForestBinaryClassification& model,
    const typename RandomForestBinaryClassification::ExampleSet& examples,
    int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Clamp01>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(
    const RandomForestMulticlassClassification& model,
    const typename RandomForestMulticlassClassification::ExampleSet& examples,
    int num_examples, std::vector<float>* predictions) {
  PredictHelperMultiDimensionTrees<std::remove_reference<decltype(model)>::type,
                                   Clamp01>(model, examples, num_examples,
                                            predictions);
}

template <>
void Predict(const RandomForestRegression& model,
             const typename RandomForestRegression::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Idendity>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(const RandomForestCategoricalUplift& model,
             const typename RandomForestCategoricalUplift::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelperMultiDimensionTrees<std::remove_reference<decltype(model)>::type,
                                   Idendity>(model, examples, num_examples,
                                             predictions);
}

template <>
void Predict(const RandomForestNumericalUplift& model,
             const typename RandomForestNumericalUplift::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Idendity>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(const GenericRandomForestBinaryClassification<uint32_t>& model,
             const typename GenericRandomForestBinaryClassification<
                 uint32_t>::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Clamp01>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(const GenericRandomForestMulticlassClassification<uint32_t>& model,
             const typename GenericRandomForestMulticlassClassification<
                 uint32_t>::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelperMultiDimensionTrees<std::remove_reference<decltype(model)>::type,
                                   Clamp01>(model, examples, num_examples,
                                            predictions);
}

template <>
void Predict(const GenericRandomForestRegression<uint32_t>& model,
             const typename GenericRandomForestRegression<uint32_t>::ExampleSet&
                 examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Idendity>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(
    const GenericRandomForestCategoricalUplift<uint32_t>& model,
    const typename GenericRandomForestCategoricalUplift<uint32_t>::ExampleSet&
        examples,
    int num_examples, std::vector<float>* predictions) {
  PredictHelperMultiDimensionTrees<std::remove_reference<decltype(model)>::type,
                                   Idendity>(model, examples, num_examples,
                                             predictions);
}

template <>
void Predict(
    const GenericRandomForestNumericalUplift<uint32_t>& model,
    const typename GenericRandomForestNumericalUplift<uint32_t>::ExampleSet&
        examples,
    int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type, Idendity>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(
    const GradientBoostedTreesBinaryClassification& model,
    const typename GradientBoostedTreesBinaryClassification::ExampleSet&
        examples,
    int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationGradientBoostedTreesBinomialLogLikelihood>(
      model, examples, num_examples, predictions);
}

template <>
void Predict(
    const GenericGradientBoostedTreesBinaryClassification<uint32_t>& model,
    const typename GenericGradientBoostedTreesBinaryClassification<
        uint32_t>::ExampleSet& examples,
    int num_examples, std::vector<float>* predictions) {
  if (model.output_logits) {
    PredictHelper<std::remove_reference<decltype(model)>::type, Idendity>(
        model, examples, num_examples, predictions);
  } else {
    PredictHelper<std::remove_reference<decltype(model)>::type,
                  ActivationGradientBoostedTreesBinomialLogLikelihood>(
        model, examples, num_examples, predictions);
  }
}

template <>
void Predict(
    const GradientBoostedTreesMulticlassClassification& model,
    const typename GradientBoostedTreesMulticlassClassification::ExampleSet&
        examples,
    int num_examples, std::vector<float>* predictions) {
  if (model.output_logits) {
    PredictHelperMultiDimensionFromSingleDimensionTrees<
        std::remove_reference<decltype(model)>::type,
        ActivationMultiDimIdentity>(model, examples, num_examples, predictions);
  } else {
    PredictHelperMultiDimensionFromSingleDimensionTrees<
        std::remove_reference<decltype(model)>::type,
        ActivationGradientBoostedTreesMultinomialLogLikelihood>(
        model, examples, num_examples, predictions);
  }
}

template <>
void Predict(
    const GradientBoostedTreesRegression& model,
    const typename GradientBoostedTreesRegression::ExampleSet& examples,
    int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationAddInitialPrediction>(model, examples, num_examples,
                                                predictions);
}

template <>
void Predict(const GradientBoostedTreesRanking& model,
             const typename GradientBoostedTreesRanking::ExampleSet& examples,
             int num_examples, std::vector<float>* predictions) {
  PredictHelper<std::remove_reference<decltype(model)>::type,
                ActivationAddInitialPrediction>(model, examples, num_examples,
                                                predictions);
}

}  // namespace decision_forest
}  // namespace serving
}  // namespace yggdrasil_decision_forests
