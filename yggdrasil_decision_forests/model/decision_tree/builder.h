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

// Utility to build decision trees manually.

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_BUILDER_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_BUILDER_H_

#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"

namespace yggdrasil_decision_forests::model::decision_tree {

class TreeBuilder {
 public:
  // The builder does not own the tree.
  TreeBuilder(DecisionTree* tree) {
    tree->CreateRoot();
    node_ = tree->mutable_root();
  }
  TreeBuilder(NodeWithChildren* node) : node_(node) {}

  // Creates a condition of the type "attribute >= threshold".
  std::pair<TreeBuilder, TreeBuilder> ConditionIsGreater(
      int attribute, float threshold,
      std::optional<int> num_examples = std::nullopt,
      std::optional<int> num_pos_examples = std::nullopt);

  // Creates a condition of the type "\sum attributes_i * weights_i >=
  // threshold". "attributes" and "weights" should have the same size.
  // "attributes" cannot be empty.
  std::pair<TreeBuilder, TreeBuilder> ConditionOblique(
      absl::Span<const int> attributes, absl::Span<const float> weights,
      float threshold);

  // Creates a condition of the type "attribute in mask".
  std::pair<TreeBuilder, TreeBuilder> ConditionContains(
      int attribute, const std::vector<int>& mask);

  // Creates a regression leaf.
  void LeafRegression(float value);

  // Creates an anomaly detection leaf.
  void LeafAnomalyDetection(int num_examples_without_weight);

 private:
  NodeWithChildren* node_;
};

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_BUILDER_H_
