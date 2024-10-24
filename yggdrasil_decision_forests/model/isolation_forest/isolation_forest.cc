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

#include "yggdrasil_decision_forests/model/isolation_forest/isolation_forest.h"

#include <stddef.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree_io.h"
#include "yggdrasil_decision_forests/model/decision_tree/structure_analysis.h"
#include "yggdrasil_decision_forests/model/isolation_forest/isolation_forest.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::model::isolation_forest {

namespace {
// Basename for the shards containing the nodes.
constexpr char kNodeBaseFilename[] = "nodes";
// Filename containing the isolation forest header.
constexpr char kHeaderBaseFilename[] = "isolation_forest_header.pb";

}  // namespace

float PreissAveragePathLength(UnsignedExampleIdx num_examples) {
  DCHECK_GT(num_examples, 0);
  const float num_examples_float = static_cast<float>(num_examples);

  // Harmonic number
  // This is the approximation proposed in "Isolation Forest" by Liu et al.
  const auto H = [](const float x) {
    constexpr float euler_constant = 0.5772156649f;
    return std::log(x) + euler_constant;
  };

  if (num_examples > 2) {
    return 2.f * H(num_examples_float - 1.f) -
           2.f * (num_examples_float - 1.f) / num_examples_float;
  } else if (num_examples == 2) {
    return 1.f;
  } else {
    return 0.f;  // To be safe.
  }
}

float IsolationForestPredictionFromDenominator(const float average_h,
                                               const float denominator) {
  if (denominator == 0.f) {
    return 0.f;
  }
  const float term = -average_h / denominator;
  return std::pow(2.f, term);
}

float IsolationForestPrediction(const float average_h,
                                const UnsignedExampleIdx num_examples) {
  return IsolationForestPredictionFromDenominator(
      average_h, PreissAveragePathLength(num_examples));
}

proto::Header IsolationForestModel::BuildHeaderProto() const {
  proto::Header header;
  header.set_num_trees(decision_trees_.size());
  header.mutable_isolation_forest();
  header.set_num_examples_per_trees(num_examples_per_trees_);
  return header;
}

void IsolationForestModel::ApplyHeaderProto(const proto::Header& header) {
  num_examples_per_trees_ = header.num_examples_per_trees();
}

absl::Status IsolationForestModel::Save(
    absl::string_view directory, const ModelIOOptions& io_options) const {
  RETURN_IF_ERROR(file::RecursivelyCreateDir(directory, file::Defaults()));
  RETURN_IF_ERROR(ValidateModelIOOptions(io_options));

  // Format used to store the nodes.
  std::string format;
  if (node_format_.has_value()) {
    format = node_format_.value();
  } else {
    ASSIGN_OR_RETURN(format, decision_tree::RecommendedSerializationFormat());
  }

  int num_shards;
  const auto node_base_filename =
      absl::StrCat(io_options.file_prefix.value(), kNodeBaseFilename);
  RETURN_IF_ERROR(decision_tree::SaveTreesToDisk(
      directory, node_base_filename, decision_trees_, format, &num_shards));

  auto header = BuildHeaderProto();
  header.set_node_format(format);
  header.set_num_node_shards(num_shards);

  const auto header_filename =
      absl::StrCat(io_options.file_prefix.value(), kHeaderBaseFilename);
  RETURN_IF_ERROR(file::SetBinaryProto(
      file::JoinPath(directory, header_filename), header, file::Defaults()));
  return absl::OkStatus();
}

absl::Status IsolationForestModel::Load(absl::string_view directory,
                                        const ModelIOOptions& io_options) {
  RETURN_IF_ERROR(ValidateModelIOOptions(io_options));

  proto::Header header;
  decision_trees_.clear();
  const auto header_filename =
      absl::StrCat(io_options.file_prefix.value(), kHeaderBaseFilename);
  RETURN_IF_ERROR(file::GetBinaryProto(
      file::JoinPath(directory, header_filename), &header, file::Defaults()));
  const auto node_base_filename =
      absl::StrCat(io_options.file_prefix.value(), kNodeBaseFilename);
  RETURN_IF_ERROR(decision_tree::LoadTreesFromDisk(
      directory, node_base_filename, header.num_node_shards(),
      header.num_trees(), header.node_format(), &decision_trees_));
  node_format_ = header.node_format();
  ApplyHeaderProto(header);
  return absl::OkStatus();
}

absl::Status IsolationForestModel::SerializeModelImpl(
    model::proto::SerializedModel* dst_proto, std::string* dst_raw) const {
  const auto& specialized_proto = dst_proto->MutableExtension(
      isolation_forest::proto::isolation_forest_serialized_model);
  *specialized_proto->mutable_header() = BuildHeaderProto();
  if (node_format_.has_value()) {
    specialized_proto->mutable_header()->set_node_format(node_format_.value());
  }
  ASSIGN_OR_RETURN(*dst_raw, decision_tree::SerializeTrees(decision_trees_));
  return absl::OkStatus();
}

absl::Status IsolationForestModel::DeserializeModelImpl(
    const model::proto::SerializedModel& src_proto, absl::string_view src_raw) {
  const auto& specialized_proto = src_proto.GetExtension(
      isolation_forest::proto::isolation_forest_serialized_model);
  ApplyHeaderProto(specialized_proto.header());
  if (specialized_proto.header().has_node_format()) {
    node_format_ = specialized_proto.header().node_format();
  }
  return decision_tree::DeserializeTrees(
      src_raw, specialized_proto.header().num_trees(), &decision_trees_);
}

absl::Status IsolationForestModel::Validate() const {
  RETURN_IF_ERROR(AbstractModel::Validate());
  if (decision_trees_.empty()) {
    return absl::InvalidArgumentError("Empty isolation forest");
  }
  if (task_ != model::proto::Task::ANOMALY_DETECTION) {
    return absl::InvalidArgumentError("Wrong task");
  }
  return absl::OkStatus();
}

std::optional<size_t> IsolationForestModel::ModelSizeInBytes() const {
  return AbstractAttributesSizeInBytes() +
         decision_tree::EstimateSizeInByte(decision_trees_);
}

void IsolationForestModel::PredictLambda(
    std::function<const decision_tree::NodeWithChildren&(
        const decision_tree::DecisionTree&)>
        get_leaf,
    model::proto::Prediction* prediction) const {
  float sum_h = 0.0;
  for (const auto& tree : decision_trees_) {
    const auto& leaf = get_leaf(*tree);
    const auto num_examples =
        leaf.node().anomaly_detection().num_examples_without_weight();
    sum_h += leaf.depth() + PreissAveragePathLength(num_examples);
  }

  if (!decision_trees_.empty()) {
    sum_h /= decision_trees_.size();
  }
  DCHECK_GT(num_examples_per_trees_, 0);
  const float p = IsolationForestPrediction(
      /*average_h=*/sum_h,
      /*num_examples=*/num_examples_per_trees_);
  prediction->mutable_anomaly_detection()->set_value(p);
}

void IsolationForestModel::Predict(const dataset::VerticalDataset& dataset,
                                   dataset::VerticalDataset::row_t row_idx,
                                   model::proto::Prediction* prediction) const {
  PredictLambda(
      [&](const decision_tree::DecisionTree& tree)
          -> const decision_tree::NodeWithChildren& {
        return tree.GetLeafAlt(dataset, row_idx);
      },
      prediction);
}

void IsolationForestModel::Predict(const dataset::proto::Example& example,
                                   model::proto::Prediction* prediction) const {
  PredictLambda(
      [&](const decision_tree::DecisionTree& tree)
          -> const decision_tree::NodeWithChildren& {
        return tree.GetLeafAlt(example);
      },
      prediction);
}

absl::Status IsolationForestModel::PredictGetLeaves(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t row_idx, absl::Span<int32_t> leaves) const {
  if (leaves.size() != num_trees()) {
    return absl::InvalidArgumentError("Wrong number of trees");
  }
  for (int tree_idx = 0; tree_idx < decision_trees_.size(); tree_idx++) {
    auto& leaf = decision_trees_[tree_idx]->GetLeafAlt(dataset, row_idx);
    if (leaf.leaf_idx() < 0) {
      return absl::InvalidArgumentError("Leaf idx not set");
    }
    leaves[tree_idx] = leaf.leaf_idx();
  }
  return absl::OkStatus();
}

bool IsolationForestModel::CheckStructure(
    const decision_tree::CheckStructureOptions& options) const {
  return decision_tree::CheckStructure(options, data_spec(), decision_trees_);
}

void IsolationForestModel::AppendDescriptionAndStatistics(
    bool full_definition, std::string* description) const {
  AbstractModel::AppendDescriptionAndStatistics(full_definition, description);
  absl::StrAppend(description, "\n");
  StrAppendForestStructureStatistics(data_spec(), decision_trees(),
                                     description);
  absl::StrAppend(description,
                  "Node format: ", node_format_.value_or("NOT_SET"), "\n");

  absl::StrAppend(description,
                  "Number of examples per tree: ", num_examples_per_trees_,
                  "\n");

  if (full_definition) {
    absl::StrAppend(description, "\nModel Structure:\n");
    decision_tree::AppendModelStructure(decision_trees_, data_spec(),
                                        label_col_idx_, description);
  }
}

absl::Status IsolationForestModel::MakePureServing() {
  for (auto& tree : decision_trees_) {
    tree->IterateOnMutableNodes(
        [](decision_tree::NodeWithChildren* node, const int depth) {
          if (!node->IsLeaf()) {
            // Remove the label information from the non-leaf nodes.
            node->mutable_node()->clear_output();
          }
        });
  }
  return AbstractModel::MakePureServing();
}

absl::Status IsolationForestModel::Distance(
    const dataset::VerticalDataset& dataset1,
    const dataset::VerticalDataset& dataset2,
    absl::Span<float> distances) const {
  return decision_tree::Distance(decision_trees(), dataset1, dataset2,
                                 distances);
}

std::string IsolationForestModel::DebugCompare(
    const AbstractModel& other) const {
  if (const auto parent_compare = AbstractModel::DebugCompare(other);
      !parent_compare.empty()) {
    return parent_compare;
  }
  const auto* other_cast = dynamic_cast<const IsolationForestModel*>(&other);
  if (!other_cast) {
    return "Non matching types";
  }
  return decision_tree::DebugCompare(
      data_spec_, label_col_idx_, decision_trees_, other_cast->decision_trees_);
}

REGISTER_AbstractModel(IsolationForestModel,
                       IsolationForestModel::kRegisteredName);

}  // namespace yggdrasil_decision_forests::model::isolation_forest
