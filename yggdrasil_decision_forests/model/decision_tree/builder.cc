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

#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"

namespace yggdrasil_decision_forests::model::decision_tree {

std::pair<TreeBuilder, TreeBuilder> TreeBuilder::ConditionIsGreater(
    const int attribute, const float threshold, std::optional<int> num_examples,
    std::optional<int> num_pos_examples) {
  node_->CreateChildren();
  node_->mutable_node()->mutable_condition()->set_attribute(attribute);
  node_->mutable_node()
      ->mutable_condition()
      ->mutable_condition()
      ->mutable_higher_condition()
      ->set_threshold(threshold);
  if (num_examples.has_value()) {
    node_->mutable_node()
        ->mutable_condition()
        ->set_num_training_examples_without_weight(*num_examples);
  }
  if (num_pos_examples.has_value()) {
    node_->mutable_node()
        ->mutable_condition()
        ->set_num_pos_training_examples_without_weight(*num_pos_examples);
  }
  return {TreeBuilder(node_->mutable_pos_child()),
          TreeBuilder(node_->mutable_neg_child())};
}

std::pair<TreeBuilder, TreeBuilder> TreeBuilder::ConditionOblique(
    absl::Span<const int> attributes, absl::Span<const float> weights,
    float threshold) {
  DCHECK(!attributes.empty());
  DCHECK_EQ(weights.size(), attributes.size());

  node_->CreateChildren();
  node_->mutable_node()->mutable_condition()->set_attribute(attributes.front());
  auto* oblique_condition = node_->mutable_node()
                                ->mutable_condition()
                                ->mutable_condition()
                                ->mutable_oblique_condition();
  oblique_condition->set_threshold(threshold);
  *oblique_condition->mutable_attributes() = {attributes.begin(),
                                              attributes.end()};
  *oblique_condition->mutable_weights() = {weights.begin(), weights.end()};
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

void TreeBuilder::LeafAnomalyDetection(const int num_examples_without_weight) {
  node_->mutable_node()
      ->mutable_anomaly_detection()
      ->set_num_examples_without_weight(num_examples_without_weight);
  node_->mutable_node()->set_num_pos_training_examples_without_weight(
      num_examples_without_weight);
}

}  // namespace yggdrasil_decision_forests::model::decision_tree
