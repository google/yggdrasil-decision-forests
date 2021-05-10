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

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_STRUCTURE_ANALYSIS_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_STRUCTURE_ANALYSIS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/histogram.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

// Number of type of conditions.
// Should be 1 + the higher field in proto::Condition.
constexpr int kNumConditionTypes = 8;

// Statistics about a forest.
struct ForestStructureStatistics {
  yggdrasil_decision_forests::utils::histogram::Histogram<int>
      number_of_nodes_per_tree;

  // Distribution of the depth of the leafs.
  yggdrasil_decision_forests::utils::histogram::Histogram<int> leaf_depths;

  // Distribution of the number of training examples that reaches the leafs.
  yggdrasil_decision_forests::utils::histogram::Histogram<int>
      num_training_examples_by_leaf;

  int64_t total_num_nodes = 0;

  int num_trees = 0;

  // "condition_attribute_sliced_by_max_depth[i].second[j]" is the number of
  // nodes, of depth equal of below
  // "condition_attribute_sliced_by_max_depth[i].first" , with attribute index
  // "j".
  std::vector<std::pair<int, std::vector<int>>>
      condition_attribute_sliced_by_max_depth;

  // "condition_type_sliced_by_max_depth[i].second[j]" is the number of
  // nodes, of depth equal of below
  // "condition_type_sliced_by_max_depth[i].first" , with condition type
  // "j" (when casted as "proto::Condition::TypeCase").
  std::vector<std::pair<int, std::vector<int>>>
      condition_type_sliced_by_max_depth;
};

// Extracts statistics.
ForestStructureStatistics ComputeForestStructureStatistics(
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
        decision_trees);

// Append statistics in a text human readable form.
void StrAppendForestStructureStatistics(
    const ForestStructureStatistics& statistics,
    const dataset::proto::DataSpecification& data_spec,
    std::string* description);

// Utility function that combines ComputeForestStructureStatistics and
// StrAppendForestStructureStatistics.
void StrAppendForestStructureStatistics(
    const dataset::proto::DataSpecification& data_spec,
    const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
        decision_trees,
    std::string* description);

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_STRUCTURE_ANALYSIS_H_
