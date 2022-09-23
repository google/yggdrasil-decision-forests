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

#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/common.h"

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_gradient_boosted_trees {

std::string SnapshotDirectory(absl::string_view work_directory) {
  return file::JoinPath(work_directory, kFileNameCheckPoint, kFileNameSnapshot);
}

EndIterTreeProtoWriter::EndIterTreeProtoWriter(
    proto::WorkerRequest::EndIter::Tree* dst)
    : dst_(dst) {}

absl::Status EndIterTreeProtoWriter::Write(
    const decision_tree::proto::Node& value) {
  *dst_->add_nodes() = value;
  return absl::OkStatus();
}

EndIterTreeProtoReader::EndIterTreeProtoReader(
    const proto::WorkerRequest::EndIter::Tree& src)
    : src_(src) {}

absl::StatusOr<bool> EndIterTreeProtoReader::Next(
    decision_tree::proto::Node* value) {
  if (next_node_idx_ >= src_.nodes_size()) {
    return false;
  }
  *value = src_.nodes(next_node_idx_++);
  return true;
}

std::string ValidationPredictionCheckpointPath(absl::string_view checkpoint_dir,
                                               int evaluation_worker_idx,
                                               int num_evaluation_workers) {
  return file::JoinPath(checkpoint_dir,
                        distributed_decision_tree::dataset_cache::ShardFilename(
                            "validation_predictions", evaluation_worker_idx,
                            num_evaluation_workers));
}

}  // namespace distributed_gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
