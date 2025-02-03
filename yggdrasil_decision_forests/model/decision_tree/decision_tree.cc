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

#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

using row_t = dataset::VerticalDataset::row_t;

namespace {

// Prefix added in front of a node when printing it with AppendModelStructure.
std::string PrettyLocalPrefix(const std::optional<bool>& is_pos) {
  if (!is_pos.has_value()) {
    return "";
  }

  if (is_pos.value()) {
    return "    ├─(pos)─ ";
  } else {
    return "    └─(neg)─ ";
  }
}

// Append "num_tabs x tab_width" spaces to the string.
void AppendMargin(const int num_tabs, std::string* dst,
                  const int tab_width = 2) {
  absl::StrAppend(dst, std::string(num_tabs * tab_width, ' '));
}

// Append a human readable description of a node value (i.e. output).
void AppendValueDescription(const dataset::proto::DataSpecification& data_spec,
                            const int label_col_idx, const proto::Node& node,
                            std::string* description) {
  switch (node.output_case()) {
    case proto::Node::OUTPUT_NOT_SET:
      LOG(FATAL) << "Not supported";
      break;

    case proto::Node::OutputCase::kClassifier: {
      const auto& col_spec = data_spec.columns(label_col_idx);

      // Should the label values be quoted?
      const bool use_quotes = !col_spec.categorical().is_already_integerized();
      const std::string optional_quote = use_quotes ? "\"" : "";

      absl::StrAppend(description, "val:", optional_quote,
                      dataset::CategoricalIdxToRepresentation(
                          col_spec, node.classifier().top_value()),
                      optional_quote);

      if (node.classifier().has_distribution()) {
        absl::StrAppend(description, " prob:[");
        const auto& distrib = node.classifier().distribution();
        for (int count_idx = 1; count_idx < distrib.counts_size();
             count_idx++) {
          if (count_idx > 1) {
            absl::StrAppend(description, ", ");
          }
          absl::StrAppend(description,
                          distrib.counts(count_idx) / distrib.sum());
        }
        absl::StrAppend(description, "]");
      }
    } break;

    case proto::Node::OutputCase::kRegressor:
      absl::StrAppend(description, "pred:", node.regressor().top_value());
      break;

    case proto::Node::OutputCase::kUplift: {
      std::string treatment_effect_str;
      for (const auto& value : node.uplift().treatment_effect()) {
        if (!treatment_effect_str.empty()) {
          absl::StrAppend(&treatment_effect_str, ", ");
        }
        absl::StrAppend(&treatment_effect_str, value);
      }

      std::string sum_weights_per_treatment_str;
      for (const auto& value : node.uplift().sum_weights_per_treatment()) {
        if (!sum_weights_per_treatment_str.empty()) {
          absl::StrAppend(&sum_weights_per_treatment_str, ", ");
        }
        absl::StrAppend(&sum_weights_per_treatment_str, value);
      }

      std::string sum_weights_per_treatment_and_outcome_str;
      for (const auto& value :
           node.uplift().sum_weights_per_treatment_and_outcome()) {
        if (!sum_weights_per_treatment_and_outcome_str.empty()) {
          absl::StrAppend(&sum_weights_per_treatment_and_outcome_str, ", ");
        }
        absl::StrAppend(&sum_weights_per_treatment_and_outcome_str, value);
      }

      absl::StrAppend(description, "uplift:[", treatment_effect_str,
                      "] examples_per_treatment:[",
                      sum_weights_per_treatment_str,
                      "] examples_per_treatment_and_outcome:[",
                      sum_weights_per_treatment_and_outcome_str, "]");
    } break;

    case proto::Node::OutputCase::kAnomalyDetection:
      absl::StrAppend(description, "count:",
                      node.anomaly_detection().num_examples_without_weight());
      break;
  }
}

std::vector<int> GetAttributes(const decision_tree::NodeWithChildren& node) {
  const auto& condition = node.node().condition();
  if (condition.condition().has_oblique_condition()) {
    return std::vector<int>{
        condition.condition().oblique_condition().attributes().begin(),
        condition.condition().oblique_condition().attributes().end()};
  } else {
    return {condition.attribute()};
  }
}

// For each path "p" and for each feature "i", adds to
// "min_depth_per_feature[i]" the minimum depth of feature "i" along the path
// "p".
//
// For a given path, "stack[j] = i" with "j \in [0, depth)" indicates that the
// "j-th" node along the path is the "i-th" feature.
//
// "min_depth_per_feature" should be already initialized (its size should be
// set).
//
// If a feature is effectively seen in the tree, set feature_used[feature_idx]
// to be true.
void AddMinimumDepthPerPath(const NodeWithChildren& node, const int depth,
                            std::vector<std::vector<int>>* stack,
                            std::vector<int>* min_depth_per_feature,
                            std::vector<bool>* feature_used) {
  if (node.IsLeaf()) {
    for (int feature_idx = 0; feature_idx < min_depth_per_feature->size();
         feature_idx++) {
      int min_depth = 0;
      while (min_depth < stack->size()) {
        const std::vector<int>& current_depth_features = (*stack)[min_depth];
        if (std::find(current_depth_features.begin(),
                      current_depth_features.end(),
                      feature_idx) == current_depth_features.end()) {
          min_depth++;
        } else {
          break;
        }
      }

      if (min_depth < stack->size()) {
        (*feature_used)[feature_idx] = true;
      }
      (*min_depth_per_feature)[feature_idx] += min_depth;
    }
  } else {
    stack->push_back(GetAttributes(node));
    AddMinimumDepthPerPath(*node.pos_child(), depth + 1, stack,
                           min_depth_per_feature, feature_used);
    AddMinimumDepthPerPath(*node.neg_child(), depth + 1, stack,
                           min_depth_per_feature, feature_used);
    stack->pop_back();
  }
}

// Returns a human readable representation of the anchor. "max_items" is the
// maximum number of anchor values to print. If more values are available,
// "..." is used.
std::string AnchorToString(
    const proto::Condition::NumericalVectorSequence::Anchor& anchor,
    const int max_items = -1) {
  std::string result;
  absl::StrAppend(&result, "[");
  for (int i = 0; i < anchor.grounded_size(); ++i) {
    if (anchor.grounded_size() > max_items && i == max_items - 1) {
      absl::StrAppend(&result, ", ..., ",
                      anchor.grounded(anchor.grounded_size() - 1));
      break;
    }
    if (i > 0) {
      absl::StrAppend(&result, ", ");
    }
    absl::StrAppendFormat(&result, "%.5g", anchor.grounded(i));
  }

  absl::StrAppend(&result, "]");
  return result;
}

absl::Span<const float> AnchorToVector(
    const proto::Condition::NumericalVectorSequence::Anchor& anchor) {
  return absl::Span<const float>(anchor.grounded());
}

}  // namespace

float SquaredDistance(const absl::Span<const float> a,
                      const absl::Span<const float> b) {
  DCHECK_EQ(a.size(), b.size());
  float acc = 0;
  for (size_t i = 0; i < a.size(); i++) {
    const float v = a[i] - b[i];
    acc += v * v;
  }
  return acc;
}

float DotProduct(const absl::Span<const float> a,
                 const absl::Span<const float> b) {
  DCHECK_EQ(a.size(), b.size());
  float acc = 0;
  for (size_t i = 0; i < a.size(); i++) {
    acc += a[i] * b[i];
  }
  return acc;
}

// Converts a map of variable importance into a vector of variable importance
// sorted in decreasing order.
std::vector<model::proto::VariableImportance>
VariableImportanceMapToSortedVector(
    const absl::flat_hash_map<int, double>& importance_map) {
  std::vector<model::proto::VariableImportance> importance_vec;
  for (const auto& item : importance_map) {
    model::proto::VariableImportance importance;
    importance.set_attribute_idx(item.first);
    importance.set_importance(item.second);
    importance_vec.push_back(importance);
  }

  // Sorts the variable importances in decreasing order.
  const auto var_importance_comparer =
      [](const model::proto::VariableImportance& a,
         const model::proto::VariableImportance& b) {
        if (a.importance() != b.importance()) {
          return a.importance() > b.importance();
        } else {
          return a.attribute_idx() < b.attribute_idx();
        }
      };
  std::sort(importance_vec.begin(), importance_vec.end(),
            var_importance_comparer);
  return importance_vec;
}

std::vector<int32_t> ExactElementsFromContainsCondition(
    int vocab_size, const proto::Condition& condition) {
  switch (condition.type_case()) {
    case proto::Condition::TypeCase::kContainsCondition:
      return {condition.contains_condition().elements().begin(),
              condition.contains_condition().elements().end()};

    case proto::Condition::TypeCase::kContainsBitmapCondition: {
      const std::string& bitmap =
          condition.contains_bitmap_condition().elements_bitmap();
      std::vector<int32_t> elements;
      for (int value = 0; value < vocab_size; value++) {
        if (utils::bitmap::GetValueBit(bitmap, value)) {
          elements.push_back(value);
        }
      }
      return elements;
    }
    default:
      LOG(FATAL) << "Not a \"contains\" type condition";
  }
}

void AppendConditionDescription(
    const dataset::proto::DataSpecification& data_spec,
    const proto::NodeCondition& node, std::string* description) {
  if (node.condition().type_case() !=
      proto::Condition::TypeCase::kObliqueCondition) {
    // Oblique conditions print the attribute name themself.
    absl::StrAppend(description, "\"",
                    data_spec.columns(node.attribute()).name(), "\"");
  }

  switch (node.condition().type_case()) {
    case proto::Condition::TypeCase::kNaCondition:
      absl::StrAppend(description, " is Na");

      break;
    case proto::Condition::TypeCase::kTrueValueCondition:
      absl::StrAppend(description, " is true");
      break;

    case proto::Condition::TypeCase::kHigherCondition:
      absl::StrAppend(description,
                      ">=", node.condition().higher_condition().threshold());
      break;

    case proto::Condition::TypeCase::kContainsCondition: {
      const auto& elements = node.condition().contains_condition().elements();
      absl::StrAppend(
          description, " is in {",
          dataset::CategoricalIdxsToRepresentation(
              data_spec.columns(node.attribute()),
              std::vector<int>(elements.begin(), elements.end()), 10),
          "}");
      break;
    }

    case proto::Condition::TypeCase::kContainsBitmapCondition: {
      const auto num_possible_values = data_spec.columns(node.attribute())
                                           .categorical()
                                           .number_of_unique_values();
      const auto elements = ExactElementsFromContainsCondition(
          num_possible_values, node.condition());
      absl::StrAppend(description, " is in [BITMAP] {",
                      dataset::CategoricalIdxsToRepresentation(
                          data_spec.columns(node.attribute()), elements, 10),
                      "}");
      break;
    }
    case proto::Condition::kDiscretizedHigherCondition:
      absl::StrAppend(
          description, ".index >= ",
          node.condition().discretized_higher_condition().threshold(),
          " i.e. \"", data_spec.columns(node.attribute()).name(), "\" >= ",
          data_spec.columns(node.attribute())
              .discretized_numerical()
              .boundaries(
                  node.condition().discretized_higher_condition().threshold() -
                  1));
      break;
    case proto::Condition::kObliqueCondition: {
      const auto& oblique = node.condition().oblique_condition();
      for (int item_idx = 0; item_idx < oblique.attributes_size(); item_idx++) {
        if (item_idx > 0) {
          absl::StrAppend(description, "+");
        }
        absl::SubstituteAndAppend(
            description, "\"$0\"x$1",
            data_spec.columns(oblique.attributes(item_idx)).name(),
            oblique.weights(item_idx));
      }
      absl::StrAppend(description, ">=", oblique.threshold());
    } break;

    case proto::Condition::kNumericalVectorSequence:
      switch (node.condition().numerical_vector_sequence().type_case()) {
        case proto::Condition::NumericalVectorSequence::kCloserThan: {
          const auto& closer_than =
              node.condition().numerical_vector_sequence().closer_than();
          absl::StrAppend(description, " contains X with | X - ",
                          AnchorToString(closer_than.anchor(), 5),
                          " |² <= ", closer_than.threshold2());
        } break;
        case proto::Condition::NumericalVectorSequence::kProjectedMoreThan: {
          const auto& project_more_than = node.condition()
                                              .numerical_vector_sequence()
                                              .projected_more_than();
          absl::StrAppend(description, " contains X with X @ ",
                          AnchorToString(project_more_than.anchor(), 5),
                          " >= ", project_more_than.threshold());
        } break;
        case proto::Condition::NumericalVectorSequence::TYPE_NOT_SET:
          absl::StrAppend(description, "Invalid vector sequence condition");
          break;
      }
      break;

    case proto::Condition::TYPE_NOT_SET:
      absl::StrAppend(description, "Unknown-type");
      break;
  }
  absl::StrAppendFormat(
      description, " [s:%g n:%i np:%i miss:%i]", node.split_score(),
      node.num_training_examples_without_weight(),
      node.num_pos_training_examples_without_weight(), node.na_value());
}

std::optional<size_t> DecisionTree::EstimateModelSizeInBytes() const {
  if (!utils::ProtoSizeInBytesIsAvailable()) {
    return {};
  }
  if (root_) {
    return root_->EstimateSizeInByte().value_or(0) + sizeof(DecisionTree);
  } else {
    return sizeof(DecisionTree);
  }
}

std::optional<size_t> NodeWithChildren::EstimateSizeInByte() const {
  if (!utils::ProtoSizeInBytesIsAvailable()) {
    return 0;
  }
  size_t size = utils::ProtoSizeInBytes(node_).value_or(0);
  if (!IsLeaf()) {
    size += children_[0]->EstimateSizeInByte().value_or(0);
    size += children_[1]->EstimateSizeInByte().value_or(0);
  }
  return size;
}

int64_t NodeWithChildren::NumNodes() const {
  if (!IsLeaf()) {
    return 1 + children_[0]->NumNodes() + children_[1]->NumNodes();
  } else {
    return 1;
  }
}

bool NodeWithChildren::CheckStructure(
    const CheckStructureOptions& options,
    const dataset::proto::DataSpecification& data_spec) const {
  if (IsLeaf()) {
    return true;
  }

  const auto& condition = node().condition();
  const auto& attribute_spec = data_spec.columns(condition.attribute());
  switch (condition.condition().type_case()) {
    case proto::Condition::kHigherCondition:

      if (!options.global_imputation_is_higher) {
        break;
      }

      if ((static_cast<float>(attribute_spec.numerical().mean()) >=
           condition.condition().higher_condition().threshold()) !=
          condition.na_value()) {
        return false;
      }
      break;

    case proto::Condition::kObliqueCondition:
      // There is currently not logic to train oblique condition that don't
      // follow global imputation.
      return true;

    case proto::Condition::kDiscretizedHigherCondition:

      if (!options.global_imputation_is_higher) {
        break;
      }

      {
        const auto discretized_threshold =
            condition.condition().discretized_higher_condition().threshold();
        const float is_higher_threshold =
            attribute_spec.discretized_numerical().boundaries(
                discretized_threshold - 1);
        if ((attribute_spec.numerical().mean() >= is_higher_threshold) !=
            condition.na_value()) {
          return false;
        }
      }
      break;

    case proto::Condition::kTrueValueCondition:

      if (!options.global_imputation_others) {
        break;
      }

      if ((attribute_spec.boolean().count_true() >=
           attribute_spec.boolean().count_false()) != condition.na_value()) {
        return false;
      }
      break;

    case proto::Condition::kContainsCondition:

      if (!options.global_imputation_others) {
        break;
      }

      if (attribute_spec.type() == dataset::proto::CATEGORICAL) {
        const auto& elements =
            condition.condition().contains_condition().elements();
        if (!std::binary_search(
                elements.begin(), elements.end(),
                attribute_spec.categorical().most_frequent_value()) ==
            condition.na_value()) {
          return false;
        }
      }
      break;

    case proto::Condition::kContainsBitmapCondition:

      if (!options.global_imputation_others) {
        break;
      }

      if (attribute_spec.type() == dataset::proto::CATEGORICAL) {
        if (!utils::bitmap::GetValueBit(
                condition.condition()
                    .contains_bitmap_condition()
                    .elements_bitmap(),
                attribute_spec.categorical().most_frequent_value()) ==
            condition.na_value()) {
          return false;
        }
      }
      break;

    case proto::Condition::kNaCondition:
      if (options.check_no_na_conditions) {
        return false;
      }
      break;

    case proto::Condition::kNumericalVectorSequence:
      break;

    case proto::Condition::TYPE_NOT_SET:
      break;
  }

  return pos_child()->CheckStructure(options, data_spec) &&
         neg_child()->CheckStructure(options, data_spec);
}

int64_t DecisionTree::NumNodes() const {
  CHECK(root_);
  return root_->NumNodes();
}

int64_t DecisionTree::NumLeafs() const { return (NumNodes() + 1) / 2; }

absl::Status DecisionTree::Validate(
    const dataset::proto::DataSpecification& data_spec,
    std::function<absl::Status(const decision_tree::proto::Node& node)>
        check_leaf) const {
  if (!root_) {
    return absl::InvalidArgumentError(
        "DecisionTree is invalid because it's missing a root node.");
  }
  RETURN_IF_ERROR(root_->Validate(data_spec, check_leaf));
  return absl::OkStatus();
}

void DecisionTree::CreateRoot() {
  DCHECK(!root_);
  root_ = std::make_unique<NodeWithChildren>();
}

absl::Status DecisionTree::WriteNodes(
    utils::ProtoWriterInterface<proto::Node>* writer) const {
  CHECK(root_) << "You cannot export an empty tree";
  if (!root_) {
    return absl::InvalidArgumentError("You cannot export an empty tree");
  }
  return root_->WriteNodes(writer);
}

absl::Status DecisionTree::ReadNodes(
    utils::ProtoReaderInterface<proto::Node>* reader) {
  CreateRoot();
  return root_->ReadNodes(reader);
}

absl::Status NodeWithChildren::WriteNodes(
    utils::ProtoWriterInterface<proto::Node>* writer) const {
  RETURN_IF_ERROR(writer->Write(node_));
  if (!IsLeaf()) {
    RETURN_IF_ERROR(children_[0]->WriteNodes(writer));
    RETURN_IF_ERROR(children_[1]->WriteNodes(writer));
  }
  return absl::OkStatus();
}

absl::Status NodeWithChildren::ReadNodes(
    utils::ProtoReaderInterface<proto::Node>* reader) {
  ASSIGN_OR_RETURN(bool did_read, reader->Next(&node_));
  if (!did_read) {
    return absl::InvalidArgumentError("Unexpected EOF");
  }
  if (node_.has_condition()) {
    CreateChildren();
    RETURN_IF_ERROR(children_[0]->ReadNodes(reader));
    RETURN_IF_ERROR(children_[1]->ReadNodes(reader));
  }
  return absl::OkStatus();
}

void NodeWithChildren::CreateChildren() {
  DCHECK(!children_[0]);
  DCHECK(!children_[1]);
  children_[0] = std::make_unique<NodeWithChildren>();
  children_[1] = std::make_unique<NodeWithChildren>();
}

void NodeWithChildren::ClearChildren() {
  DCHECK(children_[0]);
  DCHECK(children_[1]);
  children_[0] = {};
  children_[1] = {};
}

void NodeWithChildren::ClearLabelDistributionDetails() {
  switch (node_.output_case()) {
    case proto::Node::OUTPUT_NOT_SET:
      CHECK(false);
      break;
    case proto::Node::OutputCase::kClassifier:
      node_.mutable_classifier()->clear_distribution();
      break;
    case proto::Node::OutputCase::kRegressor:
      node_.mutable_regressor()->clear_distribution();
      node_.mutable_regressor()->clear_sum_gradients();
      node_.mutable_regressor()->clear_sum_hessians();
      node_.mutable_regressor()->clear_sum_weights();
      break;
    case proto::Node::OutputCase::kUplift:
      break;
    case proto::Node::OutputCase::kAnomalyDetection:
      break;
  }
}

void NodeWithChildren::FinalizeAsLeaf(
    const bool store_detailed_label_distribution) {
  CHECK(IsLeaf());
  if (!store_detailed_label_distribution) {
    ClearLabelDistributionDetails();
  }
  node_.clear_condition();
}

void NodeWithChildren::FinalizeAsNonLeaf(
    const bool keep_non_leaf_label_distribution,
    const bool store_detailed_label_distribution) {
  CHECK(!IsLeaf());
  if (!keep_non_leaf_label_distribution) {
    node_.clear_output();
  } else {
    if (!store_detailed_label_distribution) {
      ClearLabelDistributionDetails();
    }
  }
}

void NodeWithChildren::TurnIntoLeaf() {
  node_.clear_condition();
  children_[0].reset();
  children_[1].reset();
}

struct EvalConditionTrueValue {
  absl::StatusOr<bool> operator()(
      const std::vector<dataset::VerticalDataset::BooleanColumn::Format>& data,
      UnsignedExampleIdx example_idx, const bool na_value) {
    const auto value = data[example_idx];
    if (ABSL_PREDICT_FALSE(value ==
                           dataset::VerticalDataset::BooleanColumn::kNaValue)) {
      return na_value;
    }
    return value == dataset::VerticalDataset::BooleanColumn::kTrueValue;
  }
};

struct EvalConditionDiscretizedHigher {
  EvalConditionDiscretizedHigher(
      const proto::Condition::DiscretizedHigher& condition)
      : threshold(condition.threshold()) {}

  absl::StatusOr<bool> operator()(
      const std::vector<
          dataset::VerticalDataset::DiscretizedNumericalColumn::Format>& data,
      UnsignedExampleIdx example_idx, const bool na_value) {
    const auto value = data[example_idx];
    if (ABSL_PREDICT_FALSE(
            value ==
            dataset::VerticalDataset::DiscretizedNumericalColumn::kNaValue)) {
      return na_value;
    }
    return value >= threshold;
  }

  int32_t threshold;
};

struct EvalConditionHigher {
  EvalConditionHigher(const proto::Condition::Higher condition)
      : threshold(condition.threshold()) {}

  absl::StatusOr<bool> operator()(
      const std::vector<dataset::VerticalDataset::NumericalColumn::Format>&
          data,
      UnsignedExampleIdx example_idx, const bool na_value) {
    const float value = data[example_idx];
    if (ABSL_PREDICT_FALSE(std::isnan(value))) {
      return na_value;
    }
    return value >= threshold;
  }

  float threshold;
};

struct EvalConditionContainsCategorical {
  EvalConditionContainsCategorical(
      const proto::Condition::ContainsVector& condition)
      : mask(condition.elements().begin(), condition.elements().end()) {}

  absl::StatusOr<bool> operator()(
      const std::vector<dataset::VerticalDataset::CategoricalColumn::Format>&
          data,
      UnsignedExampleIdx example_idx, const bool na_value) {
    const auto value = data[example_idx];
    if (ABSL_PREDICT_FALSE(
            value == dataset::VerticalDataset::CategoricalColumn::kNaValue)) {
      return na_value;
    }
    // TODO: For small masks, should we use linear search instead?
    return std::binary_search(mask.begin(), mask.end(), value);
  }
  std::vector<dataset::VerticalDataset::CategoricalColumn::Format> mask;
};

struct EvalConditionContainsCategoricalSet {
  EvalConditionContainsCategoricalSet(
      const proto::Condition::ContainsVector& condition)
      : mask(condition.elements().begin(), condition.elements().end()) {}

  absl::StatusOr<bool> operator()(
      const dataset::VerticalDataset::CategoricalSetColumn& data,
      UnsignedExampleIdx example_idx, const bool na_value) {
    if (ABSL_PREDICT_FALSE(data.IsNa(example_idx))) {
      return na_value;
    }
    return DoSortedRangesIntersect(
        mask.begin(), mask.end(),
        data.bank().begin() + data.values()[example_idx].first,
        data.bank().begin() + data.values()[example_idx].second);
  }
  std::vector<dataset::VerticalDataset::CategoricalColumn::Format> mask;
};

struct EvalConditionContainsBitmapCategorical {
  EvalConditionContainsBitmapCategorical(
      const proto::Condition::ContainsBitmap& condition)
      : mask_bitmap(condition.elements_bitmap()) {}

  absl::StatusOr<bool> operator()(
      const std::vector<dataset::VerticalDataset::CategoricalColumn::Format>&
          data,
      UnsignedExampleIdx example_idx, const bool na_value) {
    const auto value = data[example_idx];
    if (ABSL_PREDICT_FALSE(
            value == dataset::VerticalDataset::CategoricalColumn::kNaValue)) {
      return na_value;
    }
    return utils::bitmap::GetValueBit(mask_bitmap, value);
  }
  std::string mask_bitmap;
};

struct EvalConditionContainsBitmapCategoricalSet {
  EvalConditionContainsBitmapCategoricalSet(
      const proto::Condition::ContainsBitmap& condition)
      : mask_bitmap(condition.elements_bitmap()) {}

  absl::StatusOr<bool> operator()(
      const dataset::VerticalDataset::CategoricalSetColumn& data,
      UnsignedExampleIdx example_idx, const bool na_value) {
    if (ABSL_PREDICT_FALSE(data.IsNa(example_idx))) {
      return na_value;
    }
    for (size_t bank_idx = data.values()[example_idx].first;
         bank_idx < data.values()[example_idx].second; bank_idx++) {
      const int32_t value = data.bank()[bank_idx];
      if (utils::bitmap::GetValueBit(mask_bitmap, value)) {
        return true;
      }
    }
    return false;
  }
  std::string mask_bitmap;
};

struct EvalConditionOblique {
  struct Data {
    static absl::StatusOr<Data> Create(
        const dataset::VerticalDataset& dataset,
        const proto::Condition::Oblique& condition) {
      Data data;
      data.attribute_data.reserve(condition.attributes_size());
      for (const auto attribute : condition.attributes()) {
        ASSIGN_OR_RETURN(
            const auto* column_data,
            dataset.ColumnWithCastWithStatus<
                dataset::VerticalDataset::NumericalColumn>(attribute));
        data.attribute_data.push_back(&column_data->values());
      }
      return data;
    }

    std::vector<
        const std::vector<dataset::VerticalDataset::NumericalColumn::Format>*>
        attribute_data;
  };

  EvalConditionOblique(const proto::Condition::Oblique& condition)
      : threshold(condition.threshold()),
        attributes(condition.attributes().begin(),
                   condition.attributes().end()),
        weights(condition.weights().begin(), condition.weights().end()),
        na_replacements(condition.na_replacements().begin(),
                        condition.na_replacements().end()) {}

  absl::StatusOr<bool> operator()(const Data& data,
                                  UnsignedExampleIdx example_idx,
                                  const bool na_value) {
    float sum = 0.f;
    for (size_t item_idx = 0; item_idx < attributes.size(); item_idx++) {
      float value = (*data.attribute_data[item_idx])[example_idx];
      if (std::isnan(value)) {
        if (na_replacements.empty()) {
          return na_value;
        }
        value = na_replacements[item_idx];
      }
      sum += value * weights[item_idx];
    }
    return sum >= threshold;
  }

  float threshold;
  std::vector<int> attributes;
  std::vector<float> weights;
  std::vector<float> na_replacements;
};

struct EvalConditionVectorSequenceCloserThan {
  EvalConditionVectorSequenceCloserThan(
      const proto::Condition::NumericalVectorSequence::CloserThan& condition)
      : condition_(condition), threshold2_(condition_.threshold2()) {}

  absl::StatusOr<bool> operator()(
      const dataset::VerticalDataset::NumericalVectorSequenceColumn& col,
      UnsignedExampleIdx example_idx, const bool na_value) {
    if (col.IsNa(example_idx)) {
      return na_value;
    }

    const auto anchor_value = AnchorToVector(condition_.anchor());
    const auto sequence_len = col.SequenceLength(example_idx);
    for (int vector_idx = 0; vector_idx < sequence_len; vector_idx++) {
      ASSIGN_OR_RETURN(const auto vector,
                       col.GetVector(example_idx, vector_idx));
      const float distance2 = SquaredDistance(vector, anchor_value);
      if (distance2 <= threshold2_) {
        return true;
      }
    }
    return false;
  }

  const proto::Condition::NumericalVectorSequence::CloserThan& condition_;
  float threshold2_;
};

struct EvalConditionVectorSequenceProjectedMoreThan {
  EvalConditionVectorSequenceProjectedMoreThan(
      const proto::Condition::NumericalVectorSequence::ProjectedMoreThan&
          condition)
      : condition_(condition), threshold_(condition_.threshold()) {}

  absl::StatusOr<bool> operator()(
      const dataset::VerticalDataset::NumericalVectorSequenceColumn& col,
      UnsignedExampleIdx example_idx, const bool na_value) {
    if (ABSL_PREDICT_FALSE(col.IsNa(example_idx))) {
      return na_value;
    }

    const auto anchor_value = AnchorToVector(condition_.anchor());
    const auto sequence_len = col.SequenceLength(example_idx);
    for (int vector_idx = 0; vector_idx < sequence_len; vector_idx++) {
      ASSIGN_OR_RETURN(const auto vector,
                       col.GetVector(example_idx, vector_idx));
      const float p = DotProduct(vector, anchor_value);
      if (p >= threshold_) {
        return true;
      }
    }
    return false;
  }

  const proto::Condition::NumericalVectorSequence::ProjectedMoreThan&
      condition_;
  float threshold_;
};

template <typename EvalFn, typename T>
absl::Status EvalConditionTemplate(EvalFn eval_fn,
                                   const SelectedExamplesRollingBuffer examples,
                                   const T& data, const bool dataset_is_dense,
                                   const bool na_value,
                                   ExampleSplitRollingBuffer* example_split) {
  std::ptrdiff_t next_pos_idx = 0;
  std::ptrdiff_t next_neg_idx =
      static_cast<std::ptrdiff_t>(examples.size()) - 1;
  DCHECK_EQ(examples.active.size(), examples.inactive.size());

  if (!dataset_is_dense) {
    for (const UnsignedExampleIdx example_idx : examples.active) {
      ASSIGN_OR_RETURN(const bool eval, eval_fn(data, example_idx, na_value));
      if (eval) {
        examples.inactive[next_pos_idx] = example_idx;
        next_pos_idx++;
      } else {
        examples.inactive[next_neg_idx] = example_idx;
        next_neg_idx--;
      }
    }
  } else {
    UnsignedExampleIdx dense_example_idx = 0;
    for (const UnsignedExampleIdx example_idx : examples.active) {
      ASSIGN_OR_RETURN(const bool eval,
                       eval_fn(data, dense_example_idx, na_value));
      dense_example_idx++;
      if (eval) {
        examples.inactive[next_pos_idx] = example_idx;
        next_pos_idx++;
      } else {
        examples.inactive[next_neg_idx] = example_idx;
        next_neg_idx--;
      }
    }
  }
  DCHECK_EQ(next_pos_idx, next_neg_idx + 1);

  example_split->positive_examples = {
      .active = examples.inactive.subspan(0, next_pos_idx),
      .inactive = examples.active.subspan(0, next_pos_idx)};
  example_split->negative_examples = {
      .active = examples.inactive.subspan(next_pos_idx),
      .inactive = examples.active.subspan(next_pos_idx)};

  // TODO: Populate directly in the right direction, and only use the
  // "reverse option" is the expected number of positive examples does not match
  // the evaluation. Alternatively, we can remove the reverse. The results will
  // be the same (modulo rounding error variation from summing float values in a
  // different order).

  std::reverse(example_split->negative_examples.active.begin(),
               example_split->negative_examples.active.end());
  return absl::OkStatus();
}

absl::Status EvalConditionIsNaTemplate(
    const proto::NodeCondition& condition,
    const dataset::VerticalDataset& dataset,
    SelectedExamplesRollingBuffer examples, const bool dataset_is_dense,
    ExampleSplitRollingBuffer* example_split) {
  const dataset::VerticalDataset::AbstractColumn& column_data =
      *dataset.column(condition.attribute());

  struct EvalConditionIsNaValue {
    absl::StatusOr<bool> operator()(
        const dataset::VerticalDataset::AbstractColumn& data,
        UnsignedExampleIdx example_idx, const bool na_value) {
      return data.IsNa(example_idx);
    }
  };

  return EvalConditionTemplate(EvalConditionIsNaValue(), examples, column_data,
                               dataset_is_dense, false, example_split);
}

absl::Status EvalConditionOnDataset(const dataset::VerticalDataset& dataset,
                                    SelectedExamplesRollingBuffer examples,
                                    const proto::NodeCondition& condition,
                                    const bool dataset_is_dense,
                                    ExampleSplitRollingBuffer* example_split) {
  switch (condition.condition().type_case()) {
    case proto::Condition::TypeCase::TYPE_NOT_SET:
      return absl::InvalidArgumentError("Non set condition");

    case proto::Condition::TypeCase::kNaCondition:
      RETURN_IF_ERROR(EvalConditionIsNaTemplate(
          condition, dataset, examples, dataset_is_dense, example_split));
      break;

    case proto::Condition::TypeCase::kTrueValueCondition: {
      ASSIGN_OR_RETURN(
          const auto* column_data,
          dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::BooleanColumn>(condition.attribute()));
      const auto& data = column_data->values();
      RETURN_IF_ERROR(EvalConditionTemplate(
          EvalConditionTrueValue(), examples, data, dataset_is_dense,
          condition.na_value(), example_split));
    } break;

    case proto::Condition::TypeCase::kDiscretizedHigherCondition: {
      ASSIGN_OR_RETURN(
          const auto* column_data,
          dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::DiscretizedNumericalColumn>(
              condition.attribute()));
      const auto& data = column_data->values();
      RETURN_IF_ERROR(EvalConditionTemplate(
          EvalConditionDiscretizedHigher(
              condition.condition().discretized_higher_condition()),
          examples, data, dataset_is_dense, condition.na_value(),
          example_split));
    } break;

    case proto::Condition::TypeCase::kHigherCondition: {
      ASSIGN_OR_RETURN(const auto* column_data,
                       dataset.ColumnWithCastWithStatus<
                           dataset::VerticalDataset::NumericalColumn>(
                           condition.attribute()));
      const auto& data = column_data->values();
      RETURN_IF_ERROR(EvalConditionTemplate(
          EvalConditionHigher(condition.condition().higher_condition()),
          examples, data, dataset_is_dense, condition.na_value(),
          example_split));
    } break;

    case proto::Condition::TypeCase::kContainsCondition: {
      const auto column_data = dataset.column(condition.attribute());
      if (column_data->type() == dataset::proto::ColumnType::CATEGORICAL) {
        ASSIGN_OR_RETURN(const auto* column_data,
                         dataset.ColumnWithCastWithStatus<
                             dataset::VerticalDataset::CategoricalColumn>(
                             condition.attribute()));
        const auto& data = column_data->values();
        RETURN_IF_ERROR(EvalConditionTemplate(
            EvalConditionContainsCategorical(
                condition.condition().contains_condition()),
            examples, data, dataset_is_dense, condition.na_value(),
            example_split));
      } else if (column_data->type() ==
                 dataset::proto::ColumnType::CATEGORICAL_SET) {
        ASSIGN_OR_RETURN(const auto* column_data,
                         dataset.ColumnWithCastWithStatus<
                             dataset::VerticalDataset::CategoricalSetColumn>(
                             condition.attribute()));
        RETURN_IF_ERROR(EvalConditionTemplate(
            EvalConditionContainsCategoricalSet(
                condition.condition().contains_condition()),
            examples, *column_data, dataset_is_dense, condition.na_value(),
            example_split));
      } else {
        return absl::InternalError(absl::StrCat(
            "Non supported column type for kContainsCondition condition: ",
            column_data->type()));
      }
    } break;

    case proto::Condition::TypeCase::kContainsBitmapCondition: {
      const auto column_data = dataset.column(condition.attribute());
      if (column_data->type() == dataset::proto::ColumnType::CATEGORICAL) {
        ASSIGN_OR_RETURN(const auto* column_data,
                         dataset.ColumnWithCastWithStatus<
                             dataset::VerticalDataset::CategoricalColumn>(
                             condition.attribute()));
        const auto& data = column_data->values();
        RETURN_IF_ERROR(EvalConditionTemplate(
            EvalConditionContainsBitmapCategorical(
                condition.condition().contains_bitmap_condition()),
            examples, data, dataset_is_dense, condition.na_value(),
            example_split));
      } else if (column_data->type() ==
                 dataset::proto::ColumnType::CATEGORICAL_SET) {
        ASSIGN_OR_RETURN(const auto* column_data,
                         dataset.ColumnWithCastWithStatus<
                             dataset::VerticalDataset::CategoricalSetColumn>(
                             condition.attribute()));
        RETURN_IF_ERROR(EvalConditionTemplate(
            EvalConditionContainsBitmapCategoricalSet(
                condition.condition().contains_bitmap_condition()),
            examples, *column_data, dataset_is_dense, condition.na_value(),
            example_split));
      } else {
        return absl::InternalError(
            absl::StrCat("Non supported column type for "
                         "kContainsBitmapCondition condition: ",
                         column_data->type()));
      }
    } break;

    case proto::Condition::TypeCase::kObliqueCondition: {
      const auto& ob_condition = condition.condition().oblique_condition();
      ASSIGN_OR_RETURN(const auto data, EvalConditionOblique::Data::Create(
                                            dataset, ob_condition));
      RETURN_IF_ERROR(EvalConditionTemplate(
          EvalConditionOblique(ob_condition), examples, data, dataset_is_dense,
          condition.na_value(), example_split));
    } break;

    case proto::Condition::TypeCase::kNumericalVectorSequence: {
      ASSIGN_OR_RETURN(
          const auto* column_data,
          dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::NumericalVectorSequenceColumn>(
              condition.attribute()));
      switch (condition.condition().numerical_vector_sequence().type_case()) {
        case proto::Condition::NumericalVectorSequence::kCloserThan: {
          RETURN_IF_ERROR(
              EvalConditionTemplate(EvalConditionVectorSequenceCloserThan(
                                        condition.condition()
                                            .numerical_vector_sequence()
                                            .closer_than()),
                                    examples, *column_data, dataset_is_dense,
                                    condition.na_value(), example_split));

        } break;
        case proto::Condition::NumericalVectorSequence::kProjectedMoreThan: {
          RETURN_IF_ERROR(EvalConditionTemplate(
              EvalConditionVectorSequenceProjectedMoreThan(
                  condition.condition()
                      .numerical_vector_sequence()
                      .projected_more_than()),
              examples, *column_data, dataset_is_dense, condition.na_value(),
              example_split));

        } break;
        case proto::Condition::NumericalVectorSequence::TYPE_NOT_SET:
          return absl::InvalidArgumentError("No vector sequence condition set");
      }
    } break;
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> EvalConditionFromColumn(
    const proto::NodeCondition& condition,
    const dataset::VerticalDataset::AbstractColumn* column_data,
    const dataset::VerticalDataset& dataset, const row_t example_idx) {
  // Handle NA values.
  if (ABSL_PREDICT_FALSE(condition.condition().type_case() !=
                             proto::Condition::TypeCase::kObliqueCondition &&
                         column_data->IsNa(example_idx))) {
    if (condition.condition().type_case() !=
        proto::Condition::TypeCase::kNaCondition) {
      return condition.na_value();
    } else {
      return true;
    }
  }

  switch (condition.condition().type_case()) {
    case proto::Condition::TypeCase::kNaCondition:
      // The NA value have been filtered already.
      return false;

    case proto::Condition::TypeCase::kTrueValueCondition: {
      const auto* boolean_column =
          static_cast<const dataset::VerticalDataset::BooleanColumn* const>(
              column_data);
      return boolean_column->IsTrue(example_idx);
    }

    case proto::Condition::TypeCase::kDiscretizedHigherCondition: {
      const auto* discretized_numerical_column = static_cast<
          const dataset::VerticalDataset::DiscretizedNumericalColumn* const>(
          column_data);
      return discretized_numerical_column->values()[example_idx] >=
             condition.condition().discretized_higher_condition().threshold();
    }

    case proto::Condition::TypeCase::kHigherCondition: {
      const auto* numerical_column =
          static_cast<const dataset::VerticalDataset::NumericalColumn* const>(
              column_data);
      return numerical_column->values()[example_idx] >=
             condition.condition().higher_condition().threshold();
    }

    case proto::Condition::TypeCase::kContainsCondition: {
      if (column_data->type() == dataset::proto::ColumnType::CATEGORICAL) {
        const auto* categorical_column = static_cast<
            const dataset::VerticalDataset::CategoricalColumn* const>(
            column_data);
        const auto& elements =
            condition.condition().contains_condition().elements();
        return std::binary_search(elements.begin(), elements.end(),
                                  categorical_column->values()[example_idx]);
      } else if (column_data->type() ==
                 dataset::proto::ColumnType::CATEGORICAL_SET) {
        const auto* categorical_column = static_cast<
            const dataset::VerticalDataset::CategoricalSetColumn* const>(
            column_data);
        const auto& elements =
            condition.condition().contains_condition().elements();
        return DoSortedRangesIntersect(
            elements.begin(), elements.end(),
            categorical_column->bank().begin() +
                categorical_column->values()[example_idx].first,
            categorical_column->bank().begin() +
                categorical_column->values()[example_idx].second);

      } else {
        LOG(FATAL) << "Cannot evaluate condition on column "
                   << condition.attribute();
      }
    }

    case proto::Condition::TypeCase::kContainsBitmapCondition: {
      if (column_data->type() == dataset::proto::ColumnType::CATEGORICAL) {
        const auto* categorical_column = static_cast<
            const dataset::VerticalDataset::CategoricalColumn* const>(
            column_data);
        const auto value = categorical_column->values()[example_idx];
        const std::string& bitmap =
            condition.condition().contains_bitmap_condition().elements_bitmap();
        return utils::bitmap::GetValueBit(bitmap, value);
      } else if (column_data->type() ==
                 dataset::proto::ColumnType::CATEGORICAL_SET) {
        const auto* categorical_column = static_cast<
            const dataset::VerticalDataset::CategoricalSetColumn* const>(
            column_data);
        for (size_t bank_idx = categorical_column->values()[example_idx].first;
             bank_idx < categorical_column->values()[example_idx].second;
             bank_idx++) {
          const int32_t value = categorical_column->bank()[bank_idx];
          if (utils::bitmap::GetValueBit(condition.condition()
                                             .contains_bitmap_condition()
                                             .elements_bitmap(),
                                         value)) {
            return true;
          }
        }
        return false;
      } else {
        LOG(FATAL) << "Cannot evaluate condition on column "
                   << condition.attribute();
      }
    }

    case proto::Condition::TypeCase::kObliqueCondition: {
      float sum = 0;
      const auto& oblique = condition.condition().oblique_condition();
      for (int item_idx = 0; item_idx < oblique.attributes_size(); item_idx++) {
        const auto attribute = oblique.attributes(item_idx);
        const auto local_column_data = dataset.column(attribute);
        const auto* numerical_column =
            static_cast<const dataset::VerticalDataset::NumericalColumn* const>(
                local_column_data);
        float value = numerical_column->values()[example_idx];
        if (std::isnan(value)) {
          if (oblique.na_replacements_size() == 0) {
            return condition.na_value();
          }
          DCHECK_EQ(oblique.na_replacements_size(), oblique.attributes_size());
          value = oblique.na_replacements(item_idx);
        }
        sum += value * oblique.weights(item_idx);
      }
      return sum >= oblique.threshold();
    }

    case proto::Condition::TypeCase::kNumericalVectorSequence: {
      const auto* col = static_cast<
          const dataset::VerticalDataset::NumericalVectorSequenceColumn* const>(
          column_data);

      switch (condition.condition().numerical_vector_sequence().type_case()) {
        case proto::Condition::NumericalVectorSequence::kCloserThan: {
          const auto& closer_than =
              condition.condition().numerical_vector_sequence().closer_than();
          const auto anchor_value = AnchorToVector(closer_than.anchor());
          const auto sequence_len = col->SequenceLength(example_idx);
          for (int vector_idx = 0; vector_idx < sequence_len; vector_idx++) {
            ASSIGN_OR_RETURN(const auto vector,
                             col->GetVector(example_idx, vector_idx));
            const float distance2 = SquaredDistance(vector, anchor_value);
            DCHECK_GT(closer_than.threshold2(), 0);
            if (distance2 <= closer_than.threshold2()) {
              return true;
            }
          }
          return false;
        } break;
        case proto::Condition::NumericalVectorSequence::kProjectedMoreThan: {
          const auto& projected_more_than = condition.condition()
                                                .numerical_vector_sequence()
                                                .projected_more_than();
          const auto anchor_value =
              AnchorToVector(projected_more_than.anchor());
          const auto sequence_len = col->SequenceLength(example_idx);
          for (int vector_idx = 0; vector_idx < sequence_len; vector_idx++) {
            ASSIGN_OR_RETURN(const auto vector,
                             col->GetVector(example_idx, vector_idx));
            const float p = DotProduct(vector, anchor_value);
            if (p >= projected_more_than.threshold()) {
              return true;
            }
          }
          return false;
        } break;
        case proto::Condition::NumericalVectorSequence::TYPE_NOT_SET:
          LOG(FATAL) << "Non implemented";
      }
    } break;

    default:
      LOG(FATAL) << "Non implemented";
  }
  return false;
}

absl::StatusOr<bool> EvalCondition(const proto::NodeCondition& condition,
                                   const dataset::VerticalDataset& dataset,
                                   const row_t example_idx) {
  DCHECK_GE(condition.attribute(), 0);
  DCHECK_LT(condition.attribute(), dataset.ncol());

  // Handle NA values.
  if (ABSL_PREDICT_FALSE(example_idx < 0 || example_idx >= dataset.nrow())) {
    if (condition.condition().type_case() !=
        proto::Condition::TypeCase::kNaCondition) {
      return condition.na_value();
    } else {
      return true;
    }
  }

  const auto column_data = dataset.column(condition.attribute());
  return EvalConditionFromColumn(condition, column_data, dataset, example_idx);
}

absl::StatusOr<bool> EvalCondition(const proto::NodeCondition& condition,
                                   const dataset::proto::Example& example) {
  DCHECK_GE(condition.attribute(), 0);
  DCHECK_LT(condition.attribute(), example.attributes_size());
  const auto& attribute = example.attributes(condition.attribute());

  // Handle NA values. Numerical attribute is the only attribute type than has
  // two representation for NA.
  if (attribute.type_case() ==
          dataset::proto::Example::Attribute::TYPE_NOT_SET ||
      (attribute.has_numerical() && std::isnan(attribute.numerical()))) {
    if (condition.condition().type_case() ==
        proto::Condition::TypeCase::kNaCondition) {
      return true;
    } else {
      return condition.na_value();
    }
  }

  switch (condition.condition().type_case()) {
    case proto::Condition::TypeCase::kNaCondition:
      // The NA value have been filtered already.
      return false;

    case proto::Condition::TypeCase::kTrueValueCondition:
      DCHECK(attribute.has_boolean());
      return attribute.boolean();

    case proto::Condition::TypeCase::kHigherCondition:
      DCHECK(attribute.has_numerical());
      return attribute.numerical() >=
             condition.condition().higher_condition().threshold();

    case proto::Condition::TypeCase::kDiscretizedHigherCondition:
      DCHECK(attribute.has_discretized_numerical());
      return attribute.discretized_numerical() >=
             condition.condition().discretized_higher_condition().threshold();

    case proto::Condition::TypeCase::kContainsCondition: {
      if (attribute.has_categorical()) {
        const auto& elements =
            condition.condition().contains_condition().elements();
        return std::binary_search(elements.begin(), elements.end(),
                                  attribute.categorical());
      } else if (attribute.has_categorical_set()) {
        const auto& elements =
            condition.condition().contains_condition().elements();
        return DoSortedRangesIntersect(
            elements.begin(), elements.end(),
            attribute.categorical_set().values().begin(),
            attribute.categorical_set().values().end());

      } else {
        LOG(FATAL) << "Cannot evaluate condition on column "
                   << condition.attribute();
      }
    }

    case proto::Condition::TypeCase::kContainsBitmapCondition: {
      if (attribute.has_categorical()) {
        const std::string& bitmap =
            condition.condition().contains_bitmap_condition().elements_bitmap();
        return utils::bitmap::GetValueBit(bitmap, attribute.categorical());
      } else if (attribute.has_categorical_set()) {
        for (const auto value : attribute.categorical_set().values()) {
          if (utils::bitmap::GetValueBit(condition.condition()
                                             .contains_bitmap_condition()
                                             .elements_bitmap(),
                                         value)) {
            return true;
          }
        }
        return false;
      } else {
        LOG(FATAL) << "Cannot evaluate condition on column "
                   << condition.attribute();
      }
    }

    case proto::Condition::TypeCase::kObliqueCondition: {
      float sum = 0;
      const auto& oblique = condition.condition().oblique_condition();
      for (int item_idx = 0; item_idx < oblique.attributes_size(); item_idx++) {
        const auto attribute_idx = oblique.attributes(item_idx);
        const auto& sub_attribute = example.attributes(attribute_idx);
        auto value = sub_attribute.numerical();
        if (!sub_attribute.has_numerical() || std::isnan(value)) {
          if (oblique.na_replacements_size() == 0) {
            return condition.na_value();
          }
          DCHECK_EQ(oblique.na_replacements_size(), oblique.attributes_size());
          value = oblique.na_replacements(item_idx);
        }
        sum += value * oblique.weights(item_idx);
      }
      return sum >= condition.condition().oblique_condition().threshold();
    }

    case proto::Condition::TypeCase::kNumericalVectorSequence: {
      switch (condition.condition().numerical_vector_sequence().type_case()) {
        case proto::Condition::NumericalVectorSequence::kCloserThan: {
          const auto& closer_than =
              condition.condition().numerical_vector_sequence().closer_than();
          const auto anchor_value = AnchorToVector(closer_than.anchor());
          for (const auto& proto_vector :
               attribute.numerical_vector_sequence().vectors()) {
            const absl::Span<const float> vector(proto_vector.values());
            const float distance2 = SquaredDistance(vector, anchor_value);
            if (distance2 <= closer_than.threshold2()) {
              return true;
            }
          }
          return false;
        }

        case proto::Condition::NumericalVectorSequence::kProjectedMoreThan: {
          const auto& project_more_than = condition.condition()
                                              .numerical_vector_sequence()
                                              .projected_more_than();
          const auto anchor_value = AnchorToVector(project_more_than.anchor());
          for (const auto& proto_vector :
               attribute.numerical_vector_sequence().vectors()) {
            const absl::Span<const float> vector(proto_vector.values());
            const float p = DotProduct(vector, anchor_value);
            if (p >= project_more_than.threshold()) {
              return true;
            }
          }
          return false;
        }

        default:
          DCHECK(false);
          break;
      }
    } break;

    default:
      LOG(FATAL) << "Non implemented";
  }
  return false;
}

void NodeWithChildren::CountFeatureUsage(
    std::unordered_map<int32_t, int64_t>* feature_usage) const {
  if (!IsLeaf()) {
    if (node_.condition().condition().has_oblique_condition()) {
      for (const auto attribute :
           node_.condition().condition().oblique_condition().attributes()) {
        (*feature_usage)[attribute]++;
      }
    } else {
      (*feature_usage)[node_.condition().attribute()]++;
    }

    neg_child()->CountFeatureUsage(feature_usage);
    pos_child()->CountFeatureUsage(feature_usage);
  }
}

void DecisionTree::CountFeatureUsage(
    std::unordered_map<int32_t, int64_t>* feature_usage) const {
  DCHECK(root_);
  root_->CountFeatureUsage(feature_usage);
}

const NodeWithChildren& DecisionTree::GetLeafAlt(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t row_idx) const {
  // Go down the tree according to an observation attribute values.
  DCHECK(root_ != nullptr);
  const NodeWithChildren* current_node = root_.get();
  while (!current_node->IsLeaf()) {
    const bool condition_result =
        EvalCondition(current_node->node().condition(), dataset, row_idx)
            .value();
    current_node = condition_result ? current_node->pos_child()
                                    : current_node->neg_child();
  }
  return *current_node;
}

const proto::Node& DecisionTree::GetLeaf(
    const dataset::VerticalDataset& dataset,
    const dataset::VerticalDataset::row_t row_idx) const {
  return GetLeafAlt(dataset, row_idx).node();
}

const proto::Node& DecisionTree::GetLeafWithSwappedAttribute(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t row_idx, int selected_attribute_idx,
    dataset::VerticalDataset::row_t row_id_for_selected_attribute) const {
  // Go down the tree according to an observation attribute values.
  CHECK(root_ != nullptr);
  const auto* current_node = root_.get();
  while (!current_node->IsLeaf()) {
    const auto node_row_idx =
        (current_node->node().condition().attribute() == selected_attribute_idx)
            ? row_id_for_selected_attribute
            : row_idx;
    const bool condition_result =
        EvalCondition(current_node->node().condition(), dataset, node_row_idx)
            .value();
    current_node = condition_result ? current_node->pos_child()
                                    : current_node->neg_child();
  }
  return current_node->node();
}

const NodeWithChildren& DecisionTree::GetLeafAlt(
    const dataset::proto::Example& example) const {
  // Go down the tree according to an observation attribute values.
  CHECK(root_ != nullptr);
  const auto* current_node = root_.get();
  while (!current_node->IsLeaf()) {
    const bool condition_result =
        EvalCondition(current_node->node().condition(), example).value();
    current_node = condition_result ? current_node->pos_child()
                                    : current_node->neg_child();
  }
  return *current_node;
}

const proto::Node& DecisionTree::GetLeaf(
    const dataset::proto::Example& example) const {
  return GetLeafAlt(example).node();
}

void DecisionTree::GetPath(const dataset::VerticalDataset& dataset,
                           dataset::VerticalDataset::row_t row_idx,
                           std::vector<const NodeWithChildren*>* path) const {
  DCHECK(root_ != nullptr);
  path->clear();
  const NodeWithChildren* current_node = root_.get();
  while (!current_node->IsLeaf()) {
    path->push_back(current_node);
    const bool condition_result =
        EvalCondition(current_node->node().condition(), dataset, row_idx)
            .value();
    current_node = condition_result ? current_node->pos_child()
                                    : current_node->neg_child();
  }
  path->push_back(current_node);
}

std::string ConditionTypeToString(proto::Condition::TypeCase type) {
  switch (type) {
    case proto::Condition::TypeCase::kNaCondition:
      return "NaCondition";
    case proto::Condition::TypeCase::kTrueValueCondition:
      return "TrueValueCondition";
    case proto::Condition::TypeCase::kHigherCondition:
      return "HigherCondition";
    case proto::Condition::TypeCase::kContainsCondition:
      return "ContainsCondition";
    case proto::Condition::TypeCase::kContainsBitmapCondition:
      return "ContainsBitmapCondition";
    case proto::Condition::TypeCase::kDiscretizedHigherCondition:
      return "DiscretizedHigherCondition";
    case proto::Condition::kObliqueCondition:
      return "ObliqueCondition";
    case proto::Condition::kNumericalVectorSequence:
      return "VectorSequence";
    case proto::Condition::TYPE_NOT_SET:
      CHECK(false);
  }
  return "error";
}

void DecisionTree::IterateOnNodes(
    const std::function<void(const NodeWithChildren& node, const int depth)>&
        call_back) const {
  root().IterateOnNodes(call_back);
}

void DecisionTree::IterateOnMutableNodes(
    const std::function<void(NodeWithChildren* node, const int depth)>&
        call_back,
    const bool neg_before_pos_child) {
  mutable_root()->IterateOnMutableNodes(call_back, neg_before_pos_child,
                                        /*depth=*/0);
}

void NodeWithChildren::IterateOnNodes(
    const std::function<void(const NodeWithChildren& node, const int depth)>&
        call_back,
    const int depth) const {
  call_back(*this, depth);
  if (!IsLeaf()) {
    pos_child()->IterateOnNodes(call_back, depth + 1);
    neg_child()->IterateOnNodes(call_back, depth + 1);
  }
}

void NodeWithChildren::IterateOnMutableNodes(
    const std::function<void(NodeWithChildren* node, const int depth)>&
        call_back,
    const bool neg_before_pos_child, const int depth) {
  call_back(this, depth);
  if (!IsLeaf()) {
    if (neg_before_pos_child) {
      mutable_neg_child()->IterateOnMutableNodes(
          call_back, neg_before_pos_child, depth + 1);
      mutable_pos_child()->IterateOnMutableNodes(
          call_back, neg_before_pos_child, depth + 1);
    } else {
      mutable_pos_child()->IterateOnMutableNodes(
          call_back, neg_before_pos_child, depth + 1);
      mutable_neg_child()->IterateOnMutableNodes(
          call_back, neg_before_pos_child, depth + 1);
    }
  }
}

absl::Status NodeWithChildren::Validate(
    const dataset::proto::DataSpecification& data_spec,
    std::function<absl::Status(const decision_tree::proto::Node& node)>
        check_leaf) const {
  if (!IsLeaf()) {
    if (!pos_child() || !neg_child()) {
      return absl::InvalidArgumentError("Non-leaf with missing child");
    }
    if (!node_.has_condition() || !node_.condition().has_condition()) {
      return absl::InvalidArgumentError("Non-leaf with missing condition");
    }
    if (node_.condition().attribute() < 0 ||
        node_.condition().attribute() >= data_spec.columns_size()) {
      return absl::InvalidArgumentError("Invalid attribute index");
    }
    const auto& condition = node_.condition().condition();
    const auto& attribute_spec =
        data_spec.columns(node_.condition().attribute());
    switch (condition.type_case()) {
      case proto::Condition::TypeCase::kNaCondition:
        // Compatible with all the dataspec types.
        break;
      case proto::Condition::TypeCase::kTrueValueCondition:
        if (attribute_spec.type() != dataset::proto::BOOLEAN) {
          return absl::InvalidArgumentError(
              "Invalid condition. Expect boolean feature.");
        }
        break;
      case proto::Condition::TypeCase::kDiscretizedHigherCondition:
        if (attribute_spec.type() != dataset::proto::DISCRETIZED_NUMERICAL) {
          return absl::InvalidArgumentError(
              "Invalid condition. Expect discretized numerical feature.");
        }
        break;
      case proto::Condition::TypeCase::kHigherCondition:
        if (attribute_spec.type() != dataset::proto::NUMERICAL) {
          return absl::InvalidArgumentError(
              "Invalid condition. Expect numerical feature.");
        }
        break;
      case proto::Condition::TypeCase::kContainsCondition:
        if (attribute_spec.type() != dataset::proto::CATEGORICAL &&
            attribute_spec.type() != dataset::proto::CATEGORICAL_SET) {
          return absl::InvalidArgumentError(
              "Invalid condition. Expect categorical or categorical-set "
              "feature.");
        }
        for (const auto value : condition.contains_condition().elements()) {
          if (value < 0 ||
              value >= attribute_spec.categorical().number_of_unique_values()) {
            return absl::InvalidArgumentError("Invalid \"contains\" element.");
          }
        }
        break;
      case proto::Condition::TypeCase::kContainsBitmapCondition:
        if (attribute_spec.type() != dataset::proto::CATEGORICAL &&
            attribute_spec.type() != dataset::proto::CATEGORICAL_SET) {
          return absl::InvalidArgumentError(
              "Invalid condition. Expect categorical or categorical-set "
              "feature.");
        }
        if (condition.contains_bitmap_condition().elements_bitmap().size() * 8 <
            attribute_spec.categorical().number_of_unique_values()) {
          return absl::InvalidArgumentError(
              "Condition bitmap does not contain enough elements");
        }
        break;
      case proto::Condition::TypeCase::kObliqueCondition:
        if (attribute_spec.type() != dataset::proto::NUMERICAL) {
          return absl::InvalidArgumentError(
              "Invalid condition. Expect numerical feature.");
        }
        if (condition.oblique_condition().weights_size() !=
            condition.oblique_condition().attributes_size()) {
          return absl::InvalidArgumentError(
              "Non matching weights and attributes for oblique condition");
        }
        if (condition.oblique_condition().weights_size() == 0) {
          return absl::InvalidArgumentError("Empty oblique condition");
        }
        if (condition.oblique_condition().attributes(0) !=
            node_.condition().attribute()) {
          return absl::InvalidArgumentError(
              "Non matching attribute in oblique condition");
        }
        break;
      case proto::Condition::kNumericalVectorSequence:
        if (attribute_spec.type() !=
            dataset::proto::NUMERICAL_VECTOR_SEQUENCE) {
          return absl::InvalidArgumentError(
              "Invalid condition. Expect numerical vector sequence feature.");
        }
        break;
      case proto::Condition::TYPE_NOT_SET:
        return absl::InvalidArgumentError("Unknown condition");
    }

    RETURN_IF_ERROR(pos_child()->Validate(data_spec, check_leaf));
    RETURN_IF_ERROR(neg_child()->Validate(data_spec, check_leaf));
  } else {
    if (node_.output_case() == proto::Node::OUTPUT_NOT_SET) {
      return absl::InvalidArgumentError("Leaf with missing output");
    }
    if (pos_child() || neg_child()) {
      return absl::InvalidArgumentError("Leaf with child(ren).");
    }
    RETURN_IF_ERROR(check_leaf(node()));
  }

  return absl::OkStatus();
}

void DecisionTree::AppendModelStructure(
    const dataset::proto::DataSpecification& data_spec, const int label_col_idx,
    std::string* description) const {
  if (!root_) {
    absl::StrAppend(description, "*empty tree*");
    return;
  }
  root_->AppendModelStructure(data_spec, label_col_idx, 0, {}, "    ",
                              description);
}

void NodeWithChildren::AppendModelStructure(
    const dataset::proto::DataSpecification& data_spec, const int label_col_idx,
    const int depth, const std::optional<bool> is_pos,
    const std::string& prefix, std::string* description) const {
  auto children_prefix = prefix;
  if (is_pos.has_value()) {
    if (is_pos.value()) {
      absl::StrAppend(&children_prefix, "    |    ");
    } else {
      absl::StrAppend(&children_prefix, "         ");
    }
  }

  absl::StrAppend(description, prefix, PrettyLocalPrefix(is_pos));

  // The condition.
  if (!IsLeaf()) {
    AppendConditionDescription(data_spec, node().condition(), description);
  }

  // The node value.
  if (node().output_case() != proto::Node::OUTPUT_NOT_SET) {
    if (!IsLeaf()) {
      absl::StrAppend(description, " ; ");
    }
    AppendValueDescription(data_spec, label_col_idx, node(), description);
  }

  absl::StrAppend(description, "\n");

  // The children.
  if (!IsLeaf()) {
    pos_child()->AppendModelStructure(data_spec, label_col_idx, depth + 1, true,
                                      children_prefix, description);
    neg_child()->AppendModelStructure(data_spec, label_col_idx, depth + 1,
                                      false, children_prefix, description);
  }
}

void SetLeafIndices(DecisionForest* trees) {
  for (auto& tree : *trees) {
    tree->SetLeafIndices();
  }
}

std::optional<size_t> EstimateSizeInByte(
    const std::vector<std::unique_ptr<DecisionTree>>& trees) {
  if (!utils::ProtoSizeInBytesIsAvailable()) {
    return {};
  }
  size_t size = 0;
  for (const auto& tree : trees) {
    size += tree->EstimateModelSizeInBytes().value_or(0);
  }
  return size;
}

// Number of nodes in a list of decision trees.
int64_t NumberOfNodes(const std::vector<std::unique_ptr<DecisionTree>>& trees) {
  int64_t num_nodes = 0;
  for (const auto& tree : trees) {
    num_nodes += tree->NumNodes();
  }
  return num_nodes;
}

bool CheckStructure(const CheckStructureOptions& options,
                    const dataset::proto::DataSpecification& data_spec,
                    const std::vector<std::unique_ptr<DecisionTree>>& trees) {
  for (const auto& tree : trees) {
    if (!tree->CheckStructure(options, data_spec)) {
      return false;
    }
  }
  return true;
}

void AppendModelStructure(
    const std::vector<std::unique_ptr<DecisionTree>>& trees,
    const dataset::proto::DataSpecification& data_spec, const int label_col_idx,
    std::string* description) {
  AppendModelStructureHeader(trees, data_spec, label_col_idx, description);
  absl::StrAppend(description, "\n");
  for (int tree_idx = 0; tree_idx < trees.size(); tree_idx++) {
    absl::StrAppend(description, "Tree #", tree_idx, ":\n");
    trees[tree_idx]->AppendModelStructure(data_spec, label_col_idx,
                                          description);
    absl::StrAppend(description, "\n");
  }
}

void AppendModelStructureHeader(
    const DecisionForest& trees,
    const dataset::proto::DataSpecification& data_spec, const int label_col_idx,
    std::string* description) {
  if (label_col_idx != -1) {
    const auto& label_col_spec = data_spec.columns(label_col_idx);
    // Print the label values.
    if (label_col_spec.type() == dataset::proto::CATEGORICAL &&
        !label_col_spec.categorical().is_already_integerized()) {
      absl::StrAppend(description, "Label values:\n");
      for (int value = 1;
           value < label_col_spec.categorical().number_of_unique_values();
           value++) {
        absl::StrAppend(description, "\t",
                        dataset::CategoricalIdxToRepresentation(label_col_spec,
                                                                value, true),
                        "\n");
      }
    }
  }

  absl::StrAppend(description, "Legend:\n");
  absl::StrAppend(description, "    s: Split score\n");
  absl::StrAppend(description, "    n: Number of training examples\n");
  absl::StrAppend(description,
                  "    np: Number of positive training examples\n");
  absl::StrAppend(description, "    miss: Number of missing values\n");
  absl::StrAppend(description,
                  "    val: Prediction of the leaf/non-leaf node\n");
  absl::StrAppend(description,
                  "    prob: Predicted probability for the label "
                  "values listed above (only used for classification)\n");

  absl::StrAppend(description, "Number of trees:", trees.size(), "\n");
}

int DecisionTree::MaximumDepth() const {
  int max_depth = -1;
  IterateOnNodes([&max_depth](const NodeWithChildren& node, const int depth) {
    max_depth = std::max(max_depth, depth);
  });
  return max_depth;
}

void DecisionTree::SetLeafIndices() {
  int next_leaf_idx = 0;
  IterateOnMutableNodes(
      [&next_leaf_idx](NodeWithChildren* node, const int depth) {
        if (node->IsLeaf()) {
          node->set_leaf_idx(next_leaf_idx++);
        }
        node->set_depth(depth);
      },
      /*neg_before_pos_child=*/true);
}

bool DecisionTree::CheckStructure(
    const CheckStructureOptions& options,
    const dataset::proto::DataSpecification& data_spec) const {
  return root().CheckStructure(options, data_spec);
}

void DecisionTree::ScaleRegressorOutput(const float scale) {
  IterateOnMutableNodes([&scale](NodeWithChildren* node, const int depth) {
    if (node->IsLeaf()) {
      CHECK(node->node().has_regressor());
      node->mutable_node()->mutable_regressor()->set_top_value(
          node->node().regressor().top_value() * scale);
    }
  });
}

std::vector<model::proto::VariableImportance> StructureNumberOfTimesAsRoot(
    const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
        decision_trees) {
  absl::flat_hash_map<int, double> importance;
  for (const auto& tree : decision_trees) {
    if (!tree->root().IsLeaf()) {
      for (const auto& att : GetAttributes(tree->root())) {
        importance[att]++;
      }
    }
  }
  return VariableImportanceMapToSortedVector(importance);
}

std::vector<model::proto::VariableImportance> StructureNumberOfTimesInNode(
    const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
        decision_trees) {
  absl::flat_hash_map<int, double> importance;
  for (auto& tree : decision_trees) {
    tree->IterateOnNodes(
        [&](const decision_tree::NodeWithChildren& node, const int depth) {
          if (!node.IsLeaf()) {
            for (const auto& att : GetAttributes(node)) {
              importance[att]++;
            }
          }
        });
  }
  return VariableImportanceMapToSortedVector(importance);
}

std::vector<model::proto::VariableImportance> StructureMeanMinDepth(
    const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
        decision_trees,
    const int num_features) {
  struct Importance {
    double mean_min_depth = 0;
    bool used = false;
  };
  std::vector<Importance> importance_per_feature(num_features);

  for (auto& tree : decision_trees) {
    const auto num_nodes = tree->NumLeafs();
    std::vector<std::vector<int>> stack;
    std::vector<int> min_depth_per_feature(num_features, 0);
    std::vector<bool> feature_used(num_features, false);

    AddMinimumDepthPerPath(tree->root(), 0, &stack, &min_depth_per_feature,
                           &feature_used);

    for (int feature_idx = 0; feature_idx < num_features; feature_idx++) {
      importance_per_feature[feature_idx].mean_min_depth +=
          static_cast<double>(min_depth_per_feature[feature_idx]) /
          (num_nodes * decision_trees.size());
      if (feature_used[feature_idx]) {
        importance_per_feature[feature_idx].used = true;
      }
    }
  }

  // Do the three following operations:
  //   - Copy the importance from the vector to the map.
  //   - Compute the **inverse** mean min depth.
  //   - Skip non used features.
  absl::flat_hash_map<int, double> importance_map;
  for (int feature_idx = 0; feature_idx < num_features; feature_idx++) {
    const auto& importance = importance_per_feature[feature_idx];

    if (!importance.used) {
      continue;
    }
    const auto inv_mean_min_depth = 1. / (1. + importance.mean_min_depth);
    importance_map[feature_idx] = inv_mean_min_depth;
  }

  return VariableImportanceMapToSortedVector(importance_map);
}

std::vector<model::proto::VariableImportance> StructureSumScore(
    const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
        decision_trees) {
  absl::flat_hash_map<int, double> importance;
  for (auto& tree : decision_trees) {
    tree->IterateOnNodes(
        [&](const decision_tree::NodeWithChildren& node, const int depth) {
          if (!node.IsLeaf()) {
            const double weighted_score =
                node.node().condition().split_score() *
                node.node().condition().num_training_examples_with_weight();
            for (const auto& att : GetAttributes(node)) {
              importance[att] += weighted_score;
            }
          }
        });
  }
  return VariableImportanceMapToSortedVector(importance);
}

namespace {

// Gets the leaf index for each example and each tree.
//
// The returned values "leaves" is defined as follow: "leaves[i+j *
// trees.size()]" is the leaf index of the j-th example in "dataset" for the
// i-th tree.
absl::StatusOr<std::vector<int32_t>> GetLeavesIdxs(
    const absl::Span<const std::unique_ptr<decision_tree::DecisionTree>> trees,
    const dataset::VerticalDataset& dataset) {
  const size_t num_trees = trees.size();
  const size_t num_rows = dataset.nrow();
  std::vector<int32_t> leaves(num_rows * num_trees);
  for (size_t example_idx = 0; example_idx < num_rows; example_idx++) {
    for (size_t tree_idx = 0; tree_idx < num_trees; tree_idx++) {
      const NodeWithChildren& leaf =
          trees[tree_idx]->GetLeafAlt(dataset, example_idx);
      STATUS_CHECK_GE(leaf.leaf_idx(), 0);
      leaves[tree_idx + example_idx * num_trees] = leaf.leaf_idx();
    }
  }
  return leaves;
}

}  // namespace

absl::Status Distance(
    const absl::Span<const std::unique_ptr<decision_tree::DecisionTree>> trees,
    const dataset::VerticalDataset& dataset1,
    const dataset::VerticalDataset& dataset2, const absl::Span<float> distances,
    const std::optional<std::reference_wrapper<std::vector<float>>>&
        tree_weights) {
  const size_t num_trees = trees.size();
  if (num_trees == 0) {
    return absl::InvalidArgumentError("No tree was provided");
  }
  if (tree_weights.has_value() &&
      tree_weights.value().get().size() != num_trees) {
    return absl::InvalidArgumentError(
        "The number of trees does not match the number of weights");
  }

  const size_t num_example1 = dataset1.nrow();
  const size_t num_example2 = dataset2.nrow();

  STATUS_CHECK_EQ(distances.size(), num_example1 * num_example2);
  ASSIGN_OR_RETURN(const std::vector<int32_t> leaves1,
                   GetLeavesIdxs(trees, dataset1));
  ASSIGN_OR_RETURN(const std::vector<int32_t> leaves2,
                   GetLeavesIdxs(trees, dataset2));

  for (size_t example1_idx = 0; example1_idx < num_example1; example1_idx++) {
    for (size_t example2_idx = 0; example2_idx < num_example2; example2_idx++) {
      double sum_weighted_similarity = 0;
      double sum_weights = 0;
      for (size_t tree_idx = 0; tree_idx < num_trees; tree_idx++) {
        const bool is_similar = leaves1[tree_idx + example1_idx * num_trees] ==
                                leaves2[tree_idx + example2_idx * num_trees];
        const double weight =
            tree_weights.has_value() ? tree_weights.value().get()[tree_idx] : 1;
        sum_weighted_similarity += is_similar * weight;
        sum_weights += weight;
      }
      double distance;
      if (sum_weights > 0) {
        distance = 1.f - sum_weighted_similarity / sum_weights;
      } else {
        distance = 1;
      }
      distances[example1_idx * num_example2 + example2_idx] = distance;
    }
  }

  return absl::OkStatus();
}

std::vector<int> input_features(
    absl::Span<const std::unique_ptr<decision_tree::DecisionTree>> trees) {
  absl::flat_hash_set<int> input_feature_set;
  for (auto& tree : trees) {
    tree->IterateOnNodes(
        [&](const decision_tree::NodeWithChildren& node, const int depth) {
          if (!node.IsLeaf()) {
            if (node.node().condition().condition().has_oblique_condition()) {
              const auto& oblique_condition =
                  node.node().condition().condition().oblique_condition();
              input_feature_set.insert(oblique_condition.attributes().begin(),
                                       oblique_condition.attributes().end());
            } else {
              input_feature_set.insert(node.node().condition().attribute());
            }
          }
        });
  }
  std::vector<int> input_features{input_feature_set.begin(),
                                  input_feature_set.end()};
  std::sort(input_features.begin(), input_features.end());
  return input_features;
}

std::string DebugCompare(
    const dataset::proto::DataSpecification& dataspec, const int label_idx,
    absl::Span<const std::unique_ptr<decision_tree::DecisionTree>> a,
    absl::Span<const std::unique_ptr<decision_tree::DecisionTree>> b) {
  if (a.size() != b.size()) {
    const int min_tree_count = std::min(a.size(), b.size());
    for (int tree_idx = 0; tree_idx < min_tree_count; tree_idx++) {
      const std::string sub_compare =
          a[tree_idx]->DebugCompare(dataspec, label_idx, *b[tree_idx]);
      if (!sub_compare.empty()) {
        return absl::StrCat("The number of trees doesn't match. ", a.size(),
                            " != ", b.size(),
                            ". The first different tree is the tree #",
                            tree_idx, "\n", sub_compare);
      }
    }

    return absl::StrCat(
        "The number of trees doesn't match. ", a.size(), " != ", b.size(),
        ". There is no difference in tree structure in the first ",
        min_tree_count, " trees");
  }

  for (int tree_idx = 0; tree_idx < a.size(); tree_idx++) {
    const std::string sub_compare =
        a[tree_idx]->DebugCompare(dataspec, label_idx, *b[tree_idx]);
    if (!sub_compare.empty()) {
      return absl::StrCat("In the tree #", tree_idx, ":\n", sub_compare);
    }
  }
  return {};
}

std::string DecisionTree::DebugCompare(
    const dataset::proto::DataSpecification& dataspec, const int label_idx,
    const DecisionTree& other) const {
  if ((root_ == nullptr) && (other.root_ == nullptr)) {
    return {};
  }
  if ((root_ != nullptr) != (other.root_ != nullptr)) {
    return "Only one tree has a root";
  }
  const auto result = root_->DebugCompare(dataspec, label_idx, *other.root_);
  if (!result.empty()) {
    // Print the trees.
    std::string tree_description;
    std::string other_tree_description;
    AppendModelStructure(dataspec, label_idx, &tree_description);
    other.AppendModelStructure(dataspec, label_idx, &other_tree_description);
    return absl::StrCat(result, "\n==========\nFull trees (me vs other):\n\n",
                        tree_description, "\nvs\n\n", other_tree_description);
  }
  return {};
}

std::string NodeWithChildren::DebugCompare(
    const dataset::proto::DataSpecification& dataspec, const int label_idx,
    const NodeWithChildren& other) const {
  std::string node_text =
      utils::SerializeTextProto(node_).value_or("cannot serialize first arg");
  std::string other_node_text = utils::SerializeTextProto(other.node_)
                                    .value_or("cannot serialize second arg");
  if (node_text != other_node_text) {
    return absl::StrCat("Nodes don't match.\n\n", node_text, "\nvs\n\n",
                        other_node_text);
  }
  if (!IsLeaf()) {
    for (const int i : {0, 1}) {
      if (const auto r = children_[i]->DebugCompare(dataspec, label_idx,
                                                    *other.children_[i]);
          !r.empty()) {
        return r;
      }
    }
  }
  return {};
}

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
