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

#include "yggdrasil_decision_forests/serving/embed/common.h"

#include <algorithm>
#include <cstdint>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"

namespace yggdrasil_decision_forests::serving::embed::internal {

absl::StatusOr<ModelStatistics> ComputeStatistics(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface) {
  ModelStatistics stats{
      .num_trees = static_cast<int64_t>(df_interface.num_trees()),
      .num_features = static_cast<int>(model.input_features().size()),
      .task = model.task(),
  };

  if (stats.is_classification()) {
    stats.num_classification_classes = static_cast<int>(
        model.LabelColumnSpec().categorical().number_of_unique_values() - 1);
  }

  // Scan the trees
  for (const auto& tree : df_interface.decision_trees()) {
    int64_t num_leaves_in_tree = 0;
    tree->IterateOnNodes([&](const model::decision_tree::NodeWithChildren& node,
                             int depth) {
      stats.max_depth = std::max(stats.max_depth, static_cast<int64_t>(depth));
      if (node.IsLeaf()) {
        num_leaves_in_tree++;
        stats.num_leaves++;
      } else {
        stats.num_conditions++;
        stats.has_conditions[node.node().condition().condition().type_case()] =
            true;

        if (node.node()
                .condition()
                .condition()
                .has_contains_bitmap_condition() ||
            node.node().condition().condition().has_contains_condition()) {
          const int attribute_idx = node.node().condition().attribute();
          const auto num_unique_values = model.data_spec()
                                             .columns(attribute_idx)
                                             .categorical()
                                             .number_of_unique_values();
          stats.sum_size_categorical_bitmap_masks += num_unique_values;
        }
      }
    });
    stats.max_num_leaves_per_tree =
        std::max(stats.max_num_leaves_per_tree, num_leaves_in_tree);
  }

  stats.has_multiple_condition_types =
      std::count(stats.has_conditions.begin(), stats.has_conditions.end(),
                 true) > 1;
  return stats;
}

}  // namespace yggdrasil_decision_forests::serving::embed::internal
