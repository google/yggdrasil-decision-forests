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

#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

using row_t = dataset::VerticalDataset::row_t;

namespace {

// Append "num_tabs x tab_width" spaces to the string.
void AppendMargin(const int num_tabs, std::string* dst,
                  const int tab_width = 2) {
  absl::StrAppend(dst, std::string(num_tabs * tab_width, ' '));
}

// Append a human readable description of a node value (i.e. output).
void AppendValueDescription(const dataset::proto::DataSpecification& data_spec,
                            const int label_col_idx, const proto::Node& node,
                            std::string* description) {
  absl::StrAppend(description, "Value:: ");
  switch (node.output_case()) {
    case proto::Node::OUTPUT_NOT_SET:
      LOG(FATAL) << "Not supported";
      break;

    case proto::Node::OutputCase::kClassifier: {
      const auto& col_spec = data_spec.columns(label_col_idx);

      // Should the label values be quoted?
      const bool use_quotes = !col_spec.categorical().is_already_integerized();
      const std::string optional_quote = use_quotes ? "\"" : "";

      absl::StrAppend(description, "top:", optional_quote,
                      dataset::CategoricalIdxToRepresentation(
                          col_spec, node.classifier().top_value()),
                      optional_quote);

      if (node.classifier().has_distribution()) {
        absl::StrAppend(description, " proba:[");
        const auto& distrib = node.classifier().distribution();
        for (int count_idx = 0; count_idx < distrib.counts_size();
             count_idx++) {
          if (count_idx > 0) {
            absl::StrAppend(description, ", ");
          }
          absl::StrAppend(description,
                          distrib.counts(count_idx) / distrib.sum());
        }
        absl::StrAppend(description, "]");
        absl::StrAppend(description, " sum:", distrib.sum());
      }
    } break;

    case proto::Node::OutputCase::kRegressor:
      absl::StrAppend(description, "mean:", node.regressor().top_value(),
                      " sum:", node.regressor().sum_weights());
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
  }
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

// For each path "p" and for each feature "i", adds to
// "min_depth_per_feature[i]" the minimum depth of feature "i" along the path
// "p".
//
// For a given path, "stack[j] = i" with "j \in [0, depth)" indicates that the
// "j-th" node along the path is the "i-th" feature.
//
// "min_depth_per_feature" should be already initialized (its size should be
// set).
void AddMininumDepthPerPath(const NodeWithChildren& node, const int depth,
                            std::vector<int>* stack,
                            std::vector<int>* min_depth_per_feature) {
  if (node.IsLeaf()) {
    for (int feature_idx = 0; feature_idx < min_depth_per_feature->size();
         feature_idx++) {
      int min_depth = 0;
      while (min_depth < stack->size() && (*stack)[min_depth] != feature_idx) {
        min_depth++;
      }
      (*min_depth_per_feature)[feature_idx] += min_depth;
    }
  } else {
    stack->push_back(node.node().condition().attribute());
    AddMininumDepthPerPath(*node.pos_child(), depth + 1, stack,
                           min_depth_per_feature);
    AddMininumDepthPerPath(*node.neg_child(), depth + 1, stack,
                           min_depth_per_feature);
    stack->pop_back();
  }
}

}  // namespace

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
  absl::StrAppend(description, "Condition:: ");

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
          description, " index >=",
          node.condition().discretized_higher_condition().threshold());
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
    case proto::Condition::TYPE_NOT_SET:
      absl::StrAppend(description, "Unknown-type");
      break;
  }
  absl::StrAppendFormat(
      description,
      " score:%f training_examples:%i positive_training_examples:%i "
      "missing_value_evaluation:%i",
      node.split_score(), node.num_training_examples_without_weight(),
      node.num_pos_training_examples_without_weight(), node.na_value());
}

size_t DecisionTree::EstimateModelSizeInBytes() const {
  if (root_) {
    return root_->EstimateSizeInByte() + sizeof(DecisionTree);
  } else {
    return sizeof(DecisionTree);
  }
}

size_t NodeWithChildren::EstimateSizeInByte() const {
  size_t size = node_.SpaceUsedLong();
  if (!IsLeaf()) {
    size += children_[0]->EstimateSizeInByte();
    size += children_[1]->EstimateSizeInByte();
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

bool NodeWithChildren::IsMissingValueConditionResultFollowGlobalImputation(
    const dataset::proto::DataSpecification& data_spec) const {
  if (IsLeaf()) {
    return true;
  }

  const auto& condition = node().condition();
  const auto& attribute_spec = data_spec.columns(condition.attribute());
  switch (condition.condition().type_case()) {
    case proto::Condition::kHigherCondition:
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

    case proto::Condition::kDiscretizedHigherCondition: {
      const auto discretized_threshold =
          condition.condition().discretized_higher_condition().threshold();
      const float is_higher_threshold =
          attribute_spec.discretized_numerical().boundaries(
              discretized_threshold - 1);
      if ((attribute_spec.numerical().mean() >= is_higher_threshold) !=
          condition.na_value()) {
        return false;
      }
      break;
    }

    case proto::Condition::kTrueValueCondition:
      if ((attribute_spec.boolean().count_true() >=
           attribute_spec.boolean().count_false()) != condition.na_value()) {
        return false;
      }
      break;

    case proto::Condition::kContainsCondition:
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
      if (attribute_spec.type() == dataset::proto::CATEGORICAL) {
        if (!utils::bitmap::GetValueBit(
                condition.condition()
                    .contains_bitmap_condition()
                    .elements_bitmap(),
                attribute_spec.categorical().most_frequent_value()) ==
            condition.na_value()) {
          return false;
        }
        break;
      }
      break;

    case proto::Condition::TYPE_NOT_SET:
    case proto::Condition::kNaCondition:
      break;
  }

  return pos_child()->IsMissingValueConditionResultFollowGlobalImputation(
             data_spec) &&
         neg_child()->IsMissingValueConditionResultFollowGlobalImputation(
             data_spec);
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
  CHECK(!root_);
  root_ = absl::make_unique<NodeWithChildren>();
}

absl::Status DecisionTree::WriteNodes(
    utils::ShardedWriter<proto::Node>* writer) const {
  CHECK(root_) << "You cannot export an empty tree";
  return root_->WriteNodes(writer);
}

absl::Status DecisionTree::ReadNodes(
    utils::ShardedReader<proto::Node>* reader) {
  CreateRoot();
  return root_->ReadNodes(reader);
}

absl::Status NodeWithChildren::WriteNodes(
    utils::ShardedWriter<proto::Node>* writer) const {
  RETURN_IF_ERROR(writer->Write(node_));
  if (!IsLeaf()) {
    RETURN_IF_ERROR(children_[0]->WriteNodes(writer));
    RETURN_IF_ERROR(children_[1]->WriteNodes(writer));
  }
  return absl::OkStatus();
}

absl::Status NodeWithChildren::ReadNodes(
    utils::ShardedReader<proto::Node>* reader) {
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
  children_[0] = absl::make_unique<NodeWithChildren>();
  children_[1] = absl::make_unique<NodeWithChildren>();
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

bool EvalConditionFromColumn(
    const proto::NodeCondition& condition,
    const dataset::VerticalDataset::AbstractColumn* column_data,
    const dataset::VerticalDataset& dataset, const row_t example_idx) {
  // Handle NA values.
  if (ABSL_PREDICT_FALSE(column_data->IsNa(example_idx))) {
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
        if (numerical_column->IsNa(example_idx)) {
          return condition.na_value();
        }
        const auto value = numerical_column->values()[example_idx];
        sum += value * oblique.weights(item_idx);
      }
      return sum >= oblique.threshold();
    }

    default:
      LOG(FATAL) << "Non implemented";
  }
  return false;
}

bool EvalCondition(const proto::NodeCondition& condition,
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

bool EvalCondition(const proto::NodeCondition& condition,
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
        const auto value = sub_attribute.numerical();
        if (!sub_attribute.has_numerical() || std::isnan(value)) {
          return condition.na_value();
        }
        sum += value * oblique.weights(item_idx);
      }
      return sum >= condition.condition().oblique_condition().threshold();
    }

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
        EvalCondition(current_node->node().condition(), dataset, row_idx);
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
        EvalCondition(current_node->node().condition(), dataset, node_row_idx);
    current_node = condition_result ? current_node->pos_child()
                                    : current_node->neg_child();
  }
  return current_node->node();
}

const proto::Node& DecisionTree::GetLeaf(
    const dataset::proto::Example& example) const {
  // Go down the tree according to an observation attribute values.
  CHECK(root_ != nullptr);
  const auto* current_node = root_.get();
  while (!current_node->IsLeaf()) {
    const bool condition_result =
        EvalCondition(current_node->node().condition(), example);
    current_node = condition_result ? current_node->pos_child()
                                    : current_node->neg_child();
  }
  return current_node->node();
}

const void DecisionTree::GetPath(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t row_idx,
    std::vector<const NodeWithChildren*>* path) const {
  DCHECK(root_ != nullptr);
  path->clear();
  const NodeWithChildren* current_node = root_.get();
  while (!current_node->IsLeaf()) {
    path->push_back(current_node);
    const bool condition_result =
        EvalCondition(current_node->node().condition(), dataset, row_idx);
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
  root_->AppendModelStructure(data_spec, label_col_idx, 0, description);
}

void NodeWithChildren::AppendModelStructure(
    const dataset::proto::DataSpecification& data_spec, const int label_col_idx,
    const int depth, std::string* description) const {
  // The node value.
  if (node().output_case() != proto::Node::OUTPUT_NOT_SET) {
    AppendMargin(depth, description);
    AppendValueDescription(data_spec, label_col_idx, node(), description);
    absl::StrAppend(description, "\n");
  }

  if (!IsLeaf()) {
    // The condition.
    AppendMargin(depth, description);
    AppendConditionDescription(data_spec, node().condition(), description);
    absl::StrAppend(description, "\n");

    // The children.
    AppendMargin(depth, description);
    absl::StrAppend(description, "Positive child\n");
    pos_child()->AppendModelStructure(data_spec, label_col_idx, depth + 1,
                                      description);
    AppendMargin(depth, description);
    absl::StrAppend(description, "Negative child\n");
    neg_child()->AppendModelStructure(data_spec, label_col_idx, depth + 1,
                                      description);
  }
}

void SetLeafIndices(DecisionForest* trees) {
  for (auto& tree : *trees) {
    tree->SetLeafIndices();
  }
}

size_t EstimateSizeInByte(
    const std::vector<std::unique_ptr<DecisionTree>>& trees) {
  size_t size = 0;
  for (const auto& tree : trees) {
    size += tree->EstimateModelSizeInBytes();
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

bool IsMissingValueConditionResultFollowGlobalImputation(
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<std::unique_ptr<DecisionTree>>& trees) {
  for (const auto& tree : trees) {
    if (!tree->IsMissingValueConditionResultFollowGlobalImputation(data_spec)) {
      return false;
    }
  }
  return true;
}

void AppendModelStructure(
    const std::vector<std::unique_ptr<DecisionTree>>& trees,
    const dataset::proto::DataSpecification& data_spec, const int label_col_idx,
    std::string* description) {
  absl::StrAppend(description, "Number of trees:", trees.size(), "\n");
  for (int tree_idx = 0; tree_idx < trees.size(); tree_idx++) {
    absl::StrAppend(description, "Tree #", tree_idx, "\n");
    trees[tree_idx]->AppendModelStructure(data_spec, label_col_idx,
                                          description);
    absl::StrAppend(description, "\n");
  }
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
      },
      /*neg_before_pos_child=*/true);
}

bool DecisionTree::IsMissingValueConditionResultFollowGlobalImputation(
    const dataset::proto::DataSpecification& data_spec) const {
  return root().IsMissingValueConditionResultFollowGlobalImputation(data_spec);
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
      importance[tree->root().node().condition().attribute()]++;
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
            importance[node.node().condition().attribute()]++;
          }
        });
  }
  return VariableImportanceMapToSortedVector(importance);
}

std::vector<model::proto::VariableImportance> StructureMeanMinDepth(
    const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
        decision_trees,
    const int num_features) {
  absl::flat_hash_map<int, double> importance;
  for (auto& tree : decision_trees) {
    const auto num_nodes = tree->NumLeafs();
    std::vector<int> stack;
    std::vector<int> min_depth_per_feature(num_features, 0);
    AddMininumDepthPerPath(tree->root(), 0, &stack, &min_depth_per_feature);
    for (int feature_idx = 0; feature_idx < num_features; feature_idx++) {
      importance[feature_idx] +=
          static_cast<double>(min_depth_per_feature[feature_idx]) /
          (num_nodes * decision_trees.size());
    }
  }
  return VariableImportanceMapToSortedVector(importance);
}

std::vector<model::proto::VariableImportance> StructureSumScore(
    const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
        decision_trees) {
  absl::flat_hash_map<int, double> importance;
  for (auto& tree : decision_trees) {
    tree->IterateOnNodes(
        [&](const decision_tree::NodeWithChildren& node, const int depth) {
          if (!node.IsLeaf()) {
            importance[node.node().condition().attribute()] +=
                node.node().condition().split_score() *
                node.node().condition().num_training_examples_with_weight();
          }
        });
  }
  return VariableImportanceMapToSortedVector(importance);
}

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
