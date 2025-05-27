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

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_EMBED_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_EMBED_H_

#include <cstdint>
#include <string>

#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/serving/embed/embed.pb.h"

namespace yggdrasil_decision_forests::serving::embed {

// Embed a model into a C++ library without dependencies to the YDF library.
// Returns a list of filenames and matching content.
typedef std::string Filename;
typedef std::string Content;
absl::StatusOr<absl::node_hash_map<Filename, Content>> EmbedModelCC(
    const model::AbstractModel& model, const proto::Options& options = {});

namespace internal {

// Statistics about the model.
struct ModelStatistics {
  // Number of trees.
  int64_t num_trees = 0;
  // Number of leaves.
  int64_t num_leaves = 0;
  // Maximum number of leaves among all the trees.
  int64_t max_num_leaves_per_tree = 0;
  // Maximum depth. max_depth=0 indicates that the tree contains a single node.
  int64_t max_depth = 0;
  // Number of input features.
  int num_features = 0;
  // Number of outputs of the forest i.e., before any possibly reduction by the
  // compiled model. For instance, a 3 classes classification model will have
  // "internal_output_dim=3".
  int internal_output_dim = 0;
  // Are individual trees returning multidimensional outputs.
  bool multi_dim_tree = false;
};

// Computes the statistics of the model.
absl::StatusOr<ModelStatistics> ComputeStatistics(
    const model::AbstractModel& model,
    const model::DecisionForestInterface& df_interface);

// Checks that a model name is valid. A model name can only contain letters,
// numbers, and _.
absl::Status CheckModelName(absl::string_view value);

// Converts any string into a c++ constant (with the "k") e.g. "HELLO_WOLRD_1".
std::string StringToConstantSymbol(absl::string_view input);

// Converts any string into a c++ variable name e.g. "hello_world_1".
std::string StringToVariableSymbol(absl::string_view input);

// Converts any string into a c++ struct name e.g. "HelloWorld1".
std::string StringToStructSymbol(absl::string_view input);

}  // namespace internal
}  // namespace yggdrasil_decision_forests::serving::embed

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EMBED_EMBED_H_
