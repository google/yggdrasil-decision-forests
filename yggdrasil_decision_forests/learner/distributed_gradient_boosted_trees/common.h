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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_GRADIENT_BOOSTED_TREES_COMMON_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_GRADIENT_BOOSTED_TREES_COMMON_H_

#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/worker.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_gradient_boosted_trees {

constexpr char kFileNameDatasetCache[] = "dataset_cache";
constexpr char kFileNameCheckPoint[] = "checkpoint";
constexpr char kFileNameSnapshot[] = "snapshot";
constexpr char kFileNameTmp[] = "tmp";

// Gets the path to the snapshot directory from the working directory.
std::string SnapshotDirectory(absl::string_view work_directory);

// Gets the path to the validation predictions file in a checkpoint.
std::string ValidationPredictionCheckpointPath(absl::string_view checkpoint_dir,
                                               int evaluation_worker_idx,
                                               int num_evaluation_workers);

// Encodes a tree into a EndIter::Tree proto.
class EndIterTreeProtoWriter
    : public utils::ProtoWriterInterface<decision_tree::proto::Node> {
 public:
  EndIterTreeProtoWriter(proto::WorkerRequest::EndIter::Tree* dst);
  virtual ~EndIterTreeProtoWriter() = default;

  absl::Status Write(const decision_tree::proto::Node& value) override;

 private:
  proto::WorkerRequest::EndIter::Tree* dst_;
};

// Decode a tree from an EndIter::Tree proto.
class EndIterTreeProtoReader
    : public utils::ProtoReaderInterface<decision_tree::proto::Node> {
 public:
  EndIterTreeProtoReader(const proto::WorkerRequest::EndIter::Tree& src);
  virtual ~EndIterTreeProtoReader() = default;

  absl::StatusOr<bool> Next(decision_tree::proto::Node* value) override;

 private:
  const proto::WorkerRequest::EndIter::Tree& src_;
  size_t next_node_idx_ = 0;
};

}  // namespace distributed_gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_GRADIENT_BOOSTED_TREES_COMMON_H_
