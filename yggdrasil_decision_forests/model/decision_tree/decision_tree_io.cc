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

#include "yggdrasil_decision_forests/model/decision_tree/decision_tree_io.h"

#include <stddef.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree_io_interface.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

// Maximum size (approximate) of the shards used to store list of trees.
constexpr size_t kMaxShardSizeInByte = static_cast<size_t>(200)
                                       << 20;  // 200 MB.

absl::Status SaveTreesToDisk(
    absl::string_view directory, absl::string_view basename,
    const std::vector<std::unique_ptr<DecisionTree>>& trees,
    absl::string_view format, int* num_shards) {
  ASSIGN_OR_RETURN(const auto format_impl, GetFormatImplementation(format));

  // FutureWork(gbm): The current function is fully sequential. If speed
  // becomes an issue, make it so that the shards are written in parallel.
  *num_shards =
      std::max<int>(1, (EstimateSizeInByte(trees) + kMaxShardSizeInByte - 1) /
                           kMaxShardSizeInByte);
  const int64_t num_nodes = NumberOfNodes(trees);
  const int num_nodes_per_shard =
      std::max<int>(1, (num_nodes + *num_shards - 1) / *num_shards);
  auto node_writer = format_impl->CreateWriter();
  const auto base_path = file::JoinPath(directory, basename);
  RETURN_IF_ERROR(
      node_writer->Open(file::GenerateShardedFileSpec(base_path, *num_shards),
                        num_nodes_per_shard));
  for (const auto& tree : trees) {
    RETURN_IF_ERROR(tree->WriteNodes(node_writer.get()));
  }
  RETURN_IF_ERROR(node_writer->CloseWithStatus());
  return absl::OkStatus();
}

absl::Status LoadTreesFromDisk(
    absl::string_view directory, absl::string_view basename, int num_shards,
    int num_trees, absl::string_view format,
    std::vector<std::unique_ptr<DecisionTree>>* trees) {
  ASSIGN_OR_RETURN(const auto format_impl, GetFormatImplementation(format));
  auto node_reader = format_impl->CreateReader();
  RETURN_IF_ERROR(node_reader->Open(file::GenerateShardedFileSpec(
      file::JoinPath(directory, basename), num_shards)));
  for (int64_t tree_idx = 0; tree_idx < num_trees; tree_idx++) {
    auto decision_tree = absl::make_unique<decision_tree::DecisionTree>();
    RETURN_IF_ERROR(decision_tree->ReadNodes(node_reader.get()));
    decision_tree->SetLeafIndices();
    trees->push_back(std::move(decision_tree));
  }
  return absl::OkStatus();
}

utils::StatusOr<std::string> RecommendedSerializationFormat() {
  for (const auto& candidate : {
           "BLOB_SEQUENCE",
       }) {
    if (AbstractFormatRegisterer::IsName(candidate)) {
      return candidate;
    }
  }
  return absl::InvalidArgumentError(
      "No container/formats registered to export/import decision trees to "
      "disk.");
}

utils::StatusOr<std::unique_ptr<AbstractFormat>> GetFormatImplementation(
    absl::string_view format) {
  ASSIGN_OR_RETURN(auto imp, AbstractFormatRegisterer::Create(format));
  return std::move(imp);
}

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
