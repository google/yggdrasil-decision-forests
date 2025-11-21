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

#include "yggdrasil_decision_forests/utils/shap.h"

#include <stddef.h>

#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::utils::shap {

namespace {

// Get the number of training examples in a non-leaf node. This function queries
// the node's condition, since the node itself does not store a weighted number
// of examples.
absl::StatusOr<double> GetWeightedNumExamples(
    const model::decision_tree::NodeWithChildren& node) {
  DCHECK(!node.IsLeaf());
  STATUS_CHECK(node.node().has_condition());
  double result = node.node().condition().num_training_examples_with_weight();
  STATUS_CHECK_GT(result, 0);
  return result;
}

struct NodeChildrenWeights {
  double neg;
  double pos;
};

absl::StatusOr<NodeChildrenWeights> GetChildrenWeightedNumExamples(
    const model::decision_tree::NodeWithChildren& node) {
  ASSIGN_OR_RETURN(const double current_node_weight,
                   GetWeightedNumExamples(node));
  double pos_weight =
      node.node().condition().num_pos_training_examples_with_weight();
  double neg_weight = current_node_weight - pos_weight;
  STATUS_CHECK_GT(pos_weight, 0);
  STATUS_CHECK_GT(neg_weight, 0);
  return NodeChildrenWeights{neg_weight, pos_weight};
}

bool HasInnerNodeNumNodeExamples(
    const model::decision_tree::NodeWithChildren& node) {
  return node.node().condition().has_num_training_examples_with_weight();
}

double GetRegressionNodeValue(
    const model::decision_tree::NodeWithChildren& node, const int output_idx) {
  DCHECK_EQ(output_idx, 0);
  DCHECK(node.node().regressor().has_top_value());
  return node.node().regressor().top_value();
}

double GetClassificationNodeWinnerTakeAllValue(
    const model::decision_tree::NodeWithChildren& node, const int output_idx) {
  return node.node().classifier().top_value() == (output_idx + 1);
}

double GetClassificationNodeNonWinnerTakeAllValue(
    const model::decision_tree::NodeWithChildren& node, const int output_idx) {
  DCHECK(node.node().classifier().has_distribution());
  const auto& d = node.node().classifier().distribution();
  return d.counts(output_idx + 1) / d.sum();
}

}  // namespace

namespace {

// Finds the first item in the patch with the given attribute idx. If not found,
// returns -1.
std::optional<int> FindFirst(const internal::Path& path,
                             const int attribute_idx) {
  for (int item_idx = 0; item_idx < path.size(); item_idx++) {
    if (path[item_idx].column_idx == attribute_idx) {
      return item_idx;
    }
  }
  return std::nullopt;
}

}  // namespace

namespace internal {

bool operator==(const PathItem& a, const PathItem& b) {
  return a.column_idx == b.column_idx && a.zero_fraction == b.zero_fraction &&
         a.one_fraction == b.one_fraction && a.weight == b.weight;
}

void extend(const double zero_fraction,  // "p_z" in paper.
            const double one_fraction,   // "p_o" in paper.
            const int attribute_idx,     // "p_i" in paper.
            Path& path                   // "m" in paper.
) {
  DCHECK_GT(zero_fraction, 0);
  const int n = path.size();
  path.push_back({
      .column_idx = attribute_idx,
      .zero_fraction = zero_fraction,
      .one_fraction = one_fraction,
      .weight = path.empty() ? 1.f : 0.f,
  });
  for (int i = n - 1; i >= 0; i--) {
    // The paper is 1-indexed.
    path[i + 1].weight += one_fraction * path[i].weight * (i + 1) / (n + 1);
    DCHECK(!std::isnan(path[i + 1].weight));
    path[i].weight *= zero_fraction * (n - i) / (n + 1);
    DCHECK(!std::isnan(path[i].weight));
  }
}

void unwind(const int path_idx, Path& path) {
  const int n = path.size() - 1;  // Size of the path after the "pop" back.
  const double one_fraction = path[path_idx].one_fraction;
  const double zero_fraction = path[path_idx].zero_fraction;
  DCHECK_GT(zero_fraction, 0);
  double save_weight = path.back().weight;

  for (int j = n - 1; j >= 0; j--) {
    if (one_fraction != 0) {
      const double tmp_weight = path[j].weight;
      path[j].weight = save_weight * (n + 1) / ((j + 1) * one_fraction);
      DCHECK(!std::isnan(path[j].weight));
      save_weight =
          tmp_weight - path[j].weight * zero_fraction * (n - j) / (n + 1);
    } else {
      path[j].weight = path[j].weight * (n + 1) / (zero_fraction * (n - j));
      DCHECK(!std::isnan(path[j].weight));
    }
  }

  for (int j = path_idx; j < n; j++) {
    path[j].column_idx = path[j + 1].column_idx;
    path[j].zero_fraction = path[j + 1].zero_fraction;
    path[j].one_fraction = path[j + 1].one_fraction;
  }

  // Note: The paper contains an error. The pop should be done at the end
  // (instead of at the start). Otherwise, the loop above can go out of bounds.
  // The SHAP python package tracks the size of the array, but does not pop the
  // value. So this bug does not manifest.
  path.pop_back();
}

double unwound_sum(const int path_idx, Path& path) {
  const int unwound_path_length = path.size() - 1;
  const double one_fraction = path[path_idx].one_fraction;
  const double zero_fraction = path[path_idx].zero_fraction;
  DCHECK_GT(zero_fraction, 0);
  double next_one_portion = path.back().weight;
  double total = 0;

  if (one_fraction != 0) {
    for (int i = unwound_path_length - 1; i >= 0; i--) {
      const double tmp = next_one_portion / (one_fraction * (i + 1));
      total += tmp;
      next_one_portion =
          path[i].weight - tmp * zero_fraction * (unwound_path_length - i);
    }
  } else {
    for (int i = unwound_path_length - 1; i >= 0; i--) {
      total += path[i].weight / (zero_fraction * (unwound_path_length - i));
    }
  }
  return total * (unwound_path_length + 1);
}

absl::Status recurse(const model::decision_tree::NodeWithChildren& node,
                     const double zero_fraction, const double one_fraction,
                     const int attribute_idx, const NodeValueFn node_value_fn,
                     const bool multi_output_trees, const int tree_idx,
                     const dataset::proto::Example& example, Path path,
                     ExampleShapValues& shap_values) {
  extend(zero_fraction, one_fraction, attribute_idx, path);

  if (node.IsLeaf()) {
    for (int path_idx = 1; path_idx < path.size(); path_idx++) {
      const double weight = unwound_sum(path_idx, path);
      const double scale =
          weight * (path[path_idx].one_fraction - path[path_idx].zero_fraction);

      if (multi_output_trees) {
        for (int output_idx = 0; output_idx < shap_values.num_outputs();
             output_idx++) {
          const double node_value = node_value_fn(node, output_idx);
          const int acc_index =
              shap_values.Index(path[path_idx].column_idx, output_idx);
          shap_values.values()[acc_index] += node_value * scale;
        }
      } else {
        const int output_idx = tree_idx % shap_values.num_outputs();
        const double node_value = node_value_fn(node, 0);
        const int acc_index =
            shap_values.Index(path[path_idx].column_idx, output_idx);
        shap_values.values()[acc_index] += node_value * scale;
      }
    }
  } else {
    // Evaluate condition
    ASSIGN_OR_RETURN(const bool eval, model::decision_tree::EvalCondition(
                                          node.node().condition(), example));

    bool swapped = false;
    auto hot_node = node.pos_child();
    auto cold_node = node.neg_child();
    if (!eval) {
      swapped = true;
      std::swap(hot_node, cold_node);
    }

    // Called "k" in paper
    const int new_attribute_idx = node.node().condition().attribute();
    const auto path_index = FindFirst(path, new_attribute_idx);
    // TODO: Support oblique splits properly. Currently, only the first
    // attribute of the oblique splits gather all the SHAP values.

    double incoming_zero_fraction = 1.;
    double incoming_one_fraction = 1.;
    if (path_index.has_value()) {
      // The attribute already exists in the branch.
      incoming_zero_fraction = path[*path_index].zero_fraction;
      incoming_one_fraction = path[*path_index].one_fraction;
      unwind(*path_index, path);
    }

    ASSIGN_OR_RETURN(const double num_weighted_examples,
                     GetWeightedNumExamples(node));
    double cold_zero_fraction =
        1. - node.node().condition().num_pos_training_examples_with_weight() /
                 num_weighted_examples;
    double hot_zero_fraction =
        node.node().condition().num_pos_training_examples_with_weight() /
        num_weighted_examples;
    if (swapped) {
      std::swap(cold_zero_fraction, hot_zero_fraction);
    }
    STATUS_CHECK_GT(hot_zero_fraction, 0);
    STATUS_CHECK_GT(cold_zero_fraction, 0);
    STATUS_CHECK_LT(hot_zero_fraction, 1);
    STATUS_CHECK_LT(cold_zero_fraction, 1);

    RETURN_IF_ERROR(
        recurse(*hot_node, incoming_zero_fraction * hot_zero_fraction,
                incoming_one_fraction, new_attribute_idx, node_value_fn,
                multi_output_trees, tree_idx, example, path, shap_values));

    RETURN_IF_ERROR(
        recurse(*cold_node, incoming_zero_fraction * cold_zero_fraction, 0.,
                new_attribute_idx, node_value_fn, multi_output_trees, tree_idx,
                example, path, shap_values));
  }
  return absl::OkStatus();
}

// Struct to get the output of leaf values.
struct ModelAccessor {
  // Access the leaf values.
  internal::NodeValueFn node_value_fn;
  // Scale SHAP values by this value.
  double scale;
  // Number of output of the model.
  int num_outputs;
  // If true, each tree contains values for all the outputs. If false, each tree
  // contributes to a single output.
  bool multi_output_trees;
  // Tree accessor.
  const model::DecisionForestInterface* decision_forest;
  // Possibly null variable. If set, the zero-trees bias of the model.
  const std::vector<float>* maybe_zero_tree_bias;
};

absl::StatusOr<ModelAccessor> GetModelAccessor(
    const model::AbstractModel& model) {
  auto* maybe_gbt = dynamic_cast<
      const model::gradient_boosted_trees::GradientBoostedTreesModel*>(&model);
  if (maybe_gbt) {
    return ModelAccessor{
        .node_value_fn = GetRegressionNodeValue,
        .scale = 1.,
        .num_outputs = maybe_gbt->num_trees_per_iter(),
        .multi_output_trees = false,
        .decision_forest = maybe_gbt,
        .maybe_zero_tree_bias = &maybe_gbt->initial_predictions(),
    };
  }

  auto* maybe_rf =
      dynamic_cast<const model::random_forest::RandomForestModel*>(&model);
  if (maybe_rf) {
    switch (model.task()) {
      case model::proto::Task::CLASSIFICATION: {
        // TODO: Optimize for binary classification.
        const int num_classes = model.data_spec()
                                    .columns(model.label_col_idx())
                                    .categorical()
                                    .number_of_unique_values() -
                                1;
        return ModelAccessor{
            .node_value_fn = maybe_rf->winner_take_all_inference()
                                 ? GetClassificationNodeWinnerTakeAllValue
                                 : GetClassificationNodeNonWinnerTakeAllValue,
            .scale = 1. / maybe_rf->NumTrees(),
            .num_outputs = num_classes,
            .multi_output_trees = true,
            .decision_forest = maybe_rf,
            .maybe_zero_tree_bias = nullptr,
        };
      }
      case model::proto::Task::REGRESSION:
        return ModelAccessor{
            .node_value_fn = GetRegressionNodeValue,
            .scale = 1. / maybe_rf->NumTrees(),
            .num_outputs = 1,
            .multi_output_trees = true,
            .decision_forest = maybe_rf,
            .maybe_zero_tree_bias = nullptr,
        };
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("SHAP is not implemented for task ", model.task(),
                         " Random Forest models"));
    }
  }

  // TODO: Add support for other models here.
  return absl::InvalidArgumentError(
      absl::StrCat("SHAP is not implemented for ", model.name(), " models"));
}

absl::Status GetModelBias(const ModelAccessor& accessor,
                          std::vector<double>& bias) {
  bias.assign(accessor.num_outputs, 0.);

  double num_weighted_examples = 0.;
  const int num_trees = accessor.decision_forest->decision_trees().size();
  // Number of trees that contain more than just the root node.
  int num_nontrivial_trees = 0;
  for (int tree_idx = 0; tree_idx < num_trees; tree_idx++) {
    const auto& tree = accessor.decision_forest->decision_trees()[tree_idx];
    if (tree->root().IsLeaf()) {
      continue;
    }
    num_nontrivial_trees++;
    const int output_idx_per_tree = tree_idx % accessor.num_outputs;

    if (!HasInnerNodeNumNodeExamples(tree->root())) {
      return absl::InvalidArgumentError(
          "The model does not have number of examples per nodes metadata");
    }
    auto process_leaf = [&](const model::decision_tree::NodeWithChildren& leaf,
                            double node_num_weights_examples) {
      DCHECK_GT(node_num_weights_examples, 0);
      num_weighted_examples += node_num_weights_examples;

      if (accessor.multi_output_trees) {
        for (int output_idx = 0; output_idx < accessor.num_outputs;
             output_idx++) {
          bias[output_idx] += accessor.node_value_fn(leaf, output_idx) *
                              node_num_weights_examples;
        }
      } else {
        bias[output_idx_per_tree] +=
            accessor.node_value_fn(leaf, 0) * node_num_weights_examples;
      }
    };

    std::function<absl::Status(const model::decision_tree::NodeWithChildren&)>
        process_subtree =
            [&](const model::decision_tree::NodeWithChildren& node)
        -> absl::Status {
      if (node.IsLeaf()) {
        // This can only happen if the subtree is a leaf, which has been
        // excluded earlier for the root and below during the recursion.
        DCHECK(false);
        return absl::OkStatus();
      }
      ASSIGN_OR_RETURN(const auto child_weights,
                       GetChildrenWeightedNumExamples(node));
      if (node.pos_child()->IsLeaf()) {
        process_leaf(*node.pos_child(), child_weights.pos);
      } else {
        RETURN_IF_ERROR(process_subtree(*node.pos_child()));
      }
      if (node.neg_child()->IsLeaf()) {
        process_leaf(*node.neg_child(), child_weights.neg);
      } else {
        RETURN_IF_ERROR(process_subtree(*node.neg_child()));
      }
      return absl::OkStatus();
    };
    RETURN_IF_ERROR(process_subtree(tree->root()));
  }

  // Normalize the accumulated biases according to the number of examples and
  // extra scaling. If present, add the model zero-tree bias (after the
  // scaling).
  double scale = 1.0;
  if (num_nontrivial_trees > 0) {
    STATUS_CHECK_GT(num_weighted_examples, 0);
    scale = num_nontrivial_trees * (accessor.scale / num_weighted_examples);
  }
  for (int bias_idx = 0; bias_idx < bias.size(); bias_idx++) {
    double& b = bias[bias_idx];
    b *= scale;
    if (accessor.maybe_zero_tree_bias) {
      DCHECK_EQ(bias.size(), accessor.num_outputs);
      b += (*accessor.maybe_zero_tree_bias)[bias_idx];
    }
  }

  return absl::OkStatus();
}

}  // namespace internal

void ExampleShapValues::Initialize(const int num_columns, const int num_outputs,
                                   const bool compute_bias) {
  num_outputs_ = num_outputs;
  num_columns_ = num_columns;
  values_.assign(num_columns * num_outputs, 0.);
  if (compute_bias) {
    bias_.assign(num_outputs, 0.);
  }
}

double ExampleShapValues::SumValues(const int output_idx) const {
  double sum = 0.;
  for (int col_idx = 0; col_idx < num_columns_; col_idx++) {
    sum += values_[Index(col_idx, output_idx)];
  }
  return sum;
}

std::string ExampleShapValues::ToString(
    const dataset::proto::DataSpecification& data_spec) const {
  std::string rep;
  absl::StrAppend(&rep, "Values:\n");
  for (int column_idx = 0; column_idx < num_columns(); column_idx++) {
    const auto& col_name = data_spec.columns(column_idx).name();
    absl::StrAppendFormat(&rep, "\t%s:", col_name);
    for (int output_idx = 0; output_idx < num_outputs(); output_idx++) {
      const auto value = values_[Index(column_idx, output_idx)];
      absl::StrAppendFormat(&rep, " %f", value);
    }
    absl::StrAppendFormat(&rep, "\n");
  }
  absl::StrAppend(&rep, "Bias:\n\t", absl::StrJoin(bias_, ", "), "\n");
  return rep;
}

absl::StatusOr<Shape> GetShape(const model::AbstractModel& model) {
  ASSIGN_OR_RETURN(const auto accessor, internal::GetModelAccessor(model));
  return Shape{
      .num_attributes = static_cast<size_t>(model.data_spec().columns_size()),
      .num_outputs = static_cast<size_t>(accessor.num_outputs),
  };
}

absl::Status tree_shap(const model::AbstractModel& model,
                       const dataset::proto::Example& example,
                       ExampleShapValues* shap_values, bool compute_bias) {
  ASSIGN_OR_RETURN(const auto accessor, internal::GetModelAccessor(model));

  // Initialize the shap value accumulator.
  shap_values->Initialize(model.data_spec().columns_size(),
                          accessor.num_outputs, compute_bias);

  if (compute_bias) {
    RETURN_IF_ERROR(GetModelBias(accessor, shap_values->bias()));
  }

  // Traverse tree and populate the accumulator
  internal::Path path;
  const int num_trees = accessor.decision_forest->decision_trees().size();
  for (int tree_idx = 0; tree_idx < num_trees; tree_idx++) {
    const auto& tree = accessor.decision_forest->decision_trees()[tree_idx];

    if (!tree->root().IsLeaf() && !HasInnerNodeNumNodeExamples(tree->root())) {
      return absl::InvalidArgumentError(
          "The model does not have number of examples per nodes metadata");
    }

    path.clear();
    // Note: Unlike the paper, we use the magic value of -1 instead of 0 for
    // "attribute_idx".
    RETURN_IF_ERROR(internal::recurse(
        tree->root(), 1., 1., -1, accessor.node_value_fn,
        accessor.multi_output_trees, tree_idx, example, path, *shap_values));
  }

  // Normalize the values.
  if (accessor.scale != 1.) {
    for (double& value : shap_values->values()) {
      value *= accessor.scale;
    }
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::utils::shap
