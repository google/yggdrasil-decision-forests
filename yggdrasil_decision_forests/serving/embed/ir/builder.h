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

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_IR_BUILDER_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_IR_BUILDER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"
#include "yggdrasil_decision_forests/serving/embed/ir/model_ir.h"

namespace yggdrasil_decision_forests::serving::embed::internal {
class ModelIRBuilder {
 public:
  static absl::StatusOr<ModelIR> Build(const model::AbstractModel& model,
                                       const proto::Options& options);

 private:
  ModelIRBuilder(const model::AbstractModel* model,
                 const proto::Options* options)
      : model_(model), options_(options) {};

  // Checks if the model is valid for conversion to an embedded model.
  absl::Status BuildSpecializedConversions();

  absl::Status CompileBasicModelFeatures();

  absl::Status AnalyzeFeatures();
  absl::Status CompileTrees();

  // Recursive function to flatten the tree.
  // Returns the index of the compiled node in ir_.nodes.
  absl::StatusOr<int32_t> CompileNode(
      const model::decision_tree::NodeWithChildren& node,
      std::optional<int> target_class_idx, NodeIdx tree_idx,
      absl::flat_hash_set<ConditionType>& active_condition_types);

  // Converts a categorical mask (set of integers) into 32-bit chunks,
  // adds them to bitset_bank.
  absl::StatusOr<int32_t> AddToBitsetBank(const std::vector<int32_t>& items,
                                          int num_unique_values,
                                          const std::string& name);

  absl::StatusOr<int64_t> AddToLeafBank(
      const std::vector<DoubleOrInt64>& values);

  absl::Status HandleCondition(
      const model::decision_tree::proto::NodeCondition& condition,
      Node& cur_node,
      absl::flat_hash_set<ConditionType>& active_condition_types);

  const model::DecisionForestInterface* df_interface_ = nullptr;
  const model::random_forest::RandomForestModel* random_forest_model_ = nullptr;
  const model::gradient_boosted_trees::GradientBoostedTreesModel*
      gradient_boosted_trees_model_ = nullptr;
  const model::AbstractModel* model_;
  const proto::Options* options_;
  ModelIR ir_;
  absl::flat_hash_map<int, FeatureIdx> column_idx_to_model_feature_idx_;
};
}  // namespace yggdrasil_decision_forests::serving::embed::internal

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_IR_BUILDER_H_
