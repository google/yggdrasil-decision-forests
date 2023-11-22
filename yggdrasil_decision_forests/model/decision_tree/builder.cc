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

#include "yggdrasil_decision_forests/model/decision_tree/builder.h"

#include <utility>
#include <vector>

namespace yggdrasil_decision_forests::model::decision_tree {

std::pair<TreeBuilder, TreeBuilder> TreeBuilder::ConditionIsGreater(
    const int attribute, const float threshold) {
  node_->CreateChildren();
  node_->mutable_node()->mutable_condition()->set_attribute(attribute);
  node_->mutable_node()
      ->mutable_condition()
      ->mutable_condition()
      ->mutable_higher_condition()
      ->set_threshold(threshold);
  return {TreeBuilder(node_->mutable_pos_child()),
          TreeBuilder(node_->mutable_neg_child())};
}

std::pair<TreeBuilder, TreeBuilder> TreeBuilder::ConditionContains(
    const int attribute, const std::vector<int>& mask) {
  node_->CreateChildren();
  node_->mutable_node()->mutable_condition()->set_attribute(attribute);
  *node_->mutable_node()
       ->mutable_condition()
       ->mutable_condition()
       ->mutable_contains_condition()
       ->mutable_elements() = {mask.begin(), mask.end()};
  return {TreeBuilder(node_->mutable_pos_child()),
          TreeBuilder(node_->mutable_neg_child())};
}

void TreeBuilder::LeafRegression(const float value) {
  node_->mutable_node()->mutable_regressor()->set_top_value(value);
}

}  // namespace yggdrasil_decision_forests::model::decision_tree
