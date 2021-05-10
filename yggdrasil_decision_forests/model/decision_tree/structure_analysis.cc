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

#include "yggdrasil_decision_forests/model/decision_tree/structure_analysis.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/histogram.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

ForestStructureStatistics ComputeForestStructureStatistics(
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
        decision_trees) {
  ForestStructureStatistics statistics;

  // List of max depths that we care about. "-1" means no max depth.
  const auto candidate_max_depth = {-1, 0, 1, 2, 3, 5};

  // Allocates the counters.
  for (const int max_depth : candidate_max_depth) {
    statistics.condition_attribute_sliced_by_max_depth.push_back(std::make_pair(
        max_depth, std::vector<int>(data_spec.columns_size(), 0)));
    statistics.condition_type_sliced_by_max_depth.push_back(std::make_pair(
        max_depth, std::vector<int>(decision_tree::kNumConditionTypes + 1, 0)));
  }

  statistics.num_trees = decision_trees.size();

  // Fills the histograms and counters.
  std::vector<int> number_of_nodes_per_tree_values;
  std::vector<int> leaf_depths_values;
  std::vector<int> num_training_examples_by_leaf_values;

  for (auto& tree : decision_trees) {
    const auto num_nodes = tree->NumNodes();
    statistics.total_num_nodes += num_nodes;
    number_of_nodes_per_tree_values.push_back(num_nodes);

    tree->IterateOnNodes([&](const decision_tree::NodeWithChildren& node,
                             const int depth) {
      if (!node.IsLeaf()) {
        for (auto& attribute_and_max_depth :
             statistics.condition_attribute_sliced_by_max_depth) {
          if (attribute_and_max_depth.first != -1 &&
              depth > attribute_and_max_depth.first) {
            continue;
          }
          const int attribute_idx = node.node().condition().attribute();
          CHECK_GE(attribute_idx, 0);
          CHECK_LT(attribute_idx, attribute_and_max_depth.second.size());
          attribute_and_max_depth.second[node.node().condition().attribute()]++;
        }

        for (auto& condition_type_and_max_depth :
             statistics.condition_type_sliced_by_max_depth) {
          if (condition_type_and_max_depth.first != -1 &&
              depth > condition_type_and_max_depth.first) {
            continue;
          }
          const int condition_type_idx =
              static_cast<int>(node.node().condition().condition().type_case());
          CHECK_GE(condition_type_idx, 0);
          CHECK_LT(condition_type_idx,
                   condition_type_and_max_depth.second.size());
          condition_type_and_max_depth.second[condition_type_idx]++;
        }

      } else {
        leaf_depths_values.push_back(depth);
        num_training_examples_by_leaf_values.push_back(
            node.node().num_pos_training_examples_without_weight());
      }
    });
  }
  statistics.number_of_nodes_per_tree =
      yggdrasil_decision_forests::utils::histogram::Histogram<int>::MakeUniform(
          number_of_nodes_per_tree_values, 20);
  statistics.leaf_depths =
      yggdrasil_decision_forests::utils::histogram::Histogram<int>::MakeUniform(
          leaf_depths_values, 20);
  statistics.num_training_examples_by_leaf =
      yggdrasil_decision_forests::utils::histogram::Histogram<int>::MakeUniform(
          num_training_examples_by_leaf_values, 20);

  return statistics;
}

void StrAppendForestStructureStatistics(
    const ForestStructureStatistics& statistics,
    const dataset::proto::DataSpecification& data_spec,
    std::string* description) {
  // Display statistics.
  absl::StrAppend(description, "Number of trees: ", statistics.num_trees, "\n");
  absl::StrAppend(description,
                  "Total number of nodes: ", statistics.total_num_nodes, "\n");
  absl::StrAppend(description, "\n");

  absl::StrAppend(description, "Number of nodes by tree:\n");
  absl::StrAppend(description, statistics.number_of_nodes_per_tree.ToString());
  absl::StrAppend(description, "\n");

  absl::StrAppend(description, "Depth by leafs:\n");
  absl::StrAppend(description, statistics.leaf_depths.ToString());
  absl::StrAppend(description, "\n");

  absl::StrAppend(description, "Number of training obs by leaf:\n");
  absl::StrAppend(description,
                  statistics.num_training_examples_by_leaf.ToString());
  absl::StrAppend(description, "\n");

  // Given a vector of integer "counts", returns the index and value sorted by
  // decreasing value. Excludes the zeros.
  const auto sort_by_count = [](const std::vector<int>& counts) {
    std::vector<std::pair<int, int>> index_and_value;
    index_and_value.reserve(counts.size());
    for (int attr_idx = 0; attr_idx < counts.size(); attr_idx++) {
      if (counts[attr_idx] == 0) {
        continue;
      }
      index_and_value.push_back(std::make_pair(counts[attr_idx], attr_idx));
    }
    std::sort(index_and_value.begin(), index_and_value.end(),
              std::greater<std::pair<int, int>>());
    return index_and_value;
  };

  for (const auto& attribute_and_max_depth :
       statistics.condition_attribute_sliced_by_max_depth) {
    const auto max_depth = attribute_and_max_depth.first;
    const auto& attributes = attribute_and_max_depth.second;
    const auto index_and_value = sort_by_count(attributes);
    absl::StrAppend(description, "Attribute in nodes");
    if (max_depth >= 0) {
      absl::StrAppend(description, " with depth <= ", max_depth);
    }
    absl::StrAppend(description, ":\n");
    for (auto count_and_attr_idx : index_and_value) {
      const auto count = count_and_attr_idx.first;
      const auto attr_idx = count_and_attr_idx.second;
      const auto& col = data_spec.columns(attr_idx);
      absl::StrAppend(description, "\t", count, " : ", col.name(), " [",
                      ColumnType_Name(col.type()), "]\n");
    }
    absl::StrAppend(description, "\n");
  }

  for (const auto& condition_type_and_max_depth :
       statistics.condition_type_sliced_by_max_depth) {
    const auto max_depth = condition_type_and_max_depth.first;
    const auto& condition_types = condition_type_and_max_depth.second;
    const auto index_and_value = sort_by_count(condition_types);
    absl::StrAppend(description, "Condition type in nodes");
    if (max_depth >= 0) {
      absl::StrAppend(description, " with depth <= ", max_depth);
    }
    absl::StrAppend(description, ":\n");
    for (auto count_and_condition_type : index_and_value) {
      const auto count = count_and_condition_type.first;
      const auto cond_type = count_and_condition_type.second;
      absl::StrAppend(
          description, "\t", count, " : ",
          decision_tree::ConditionTypeToString(
              static_cast<decision_tree::proto::Condition::TypeCase>(
                  cond_type)),
          "\n");
    }
  }
}

void StrAppendForestStructureStatistics(
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
        decision_trees,
    std::string* description) {
  const auto statistics =
      ComputeForestStructureStatistics(data_spec, decision_trees);
  StrAppendForestStructureStatistics(statistics, data_spec, description);
}

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
