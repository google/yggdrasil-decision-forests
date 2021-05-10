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

// Serialization / de-serialization of decision trees.

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_DECISION_TREE_IO_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_DECISION_TREE_IO_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

// Saves or loads a list of decision trees to drive into a sharded container of
// serialized protos.
//
// The trees and their nodes will be written in
// {directory}/{basename}@{num_shards} while 'num_shards' determined on the fly.
//
// The tree are written sequentially. For each trees, nodes are written
// sequentially, in a depth first transversal, with the <node, negative child,
// positive child> order.
absl::Status SaveTreesToDisk(
    absl::string_view directory, absl::string_view basename,
    const std::vector<std::unique_ptr<DecisionTree>>& trees,
    absl::string_view format, int* num_shards);

absl::Status LoadTreesFromDisk(
    absl::string_view directory, absl::string_view basename, int num_shards,
    int num_trees, absl::string_view format,
    std::vector<std::unique_ptr<DecisionTree>>* trees);

// Gets the recommended format to store decision trees with
// SaveTreesToDisk among the registered ones. At least one
// format should be registered.
utils::StatusOr<std::string> RecommendedSerializationFormat();

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_DECISION_TREE_IO_H_
