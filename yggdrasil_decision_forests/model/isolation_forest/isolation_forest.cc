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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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

absl::StatusOr<std::vector<model::proto::VariableImportance>>
StructureMeanPartitionScore(
    const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
        decision_trees) {
  struct ImportancePerFeature {
    double total_score = 0;
    int num_usage = 0;
  };
  absl::flat_hash_map<int, ImportancePerFeature> scores;
  bool has_training_information = true;

  for (auto& tree : decision_trees) {
    tree->IterateOnNodes(
        [&](const decision_tree::NodeWithChildren& node, const int depth) {
          if (!node.IsLeaf()) {
            if (!has_training_information ||
                !node.node()
                     .condition()
                     .has_num_training_examples_without_weight() ||
                !node.node()
                     .condition()
                     .has_num_pos_training_examples_without_weight()) {
              // Missing training set information, abort.
              has_training_information = false;
              return;
            }
            const double total_examples =
                node.node().condition().num_training_examples_without_weight();
            const double pos_examples =
                node.node()
                    .condition()
                    .num_pos_training_examples_without_weight();
            const double partition_ratio = pos_examples/total_examples;
            DCHECK_GT(total_examples, 0);
            const double partition_score =
                1. - 4. * (partition_ratio) * (1. - partition_ratio);
            auto& score = scores[node.node().condition().attribute()];
            score.total_score += partition_score;
            scores[node.node().condition().attribute()].num_usage++;
          }
        });
  }
  if (!has_training_information) {
    LOG(INFO)
        << "This model is missing some training information, cannot compute "
        << kVariableImportanceMeanPartitionScore;
    return std::vector<model::proto::VariableImportance>();
  }
  absl::flat_hash_map<int, double> importance;
  for (const auto x : scores) {
    STATUS_CHECK_GT(x.second.num_usage, 0);
    STATUS_CHECK_GE(x.second.total_score, 0);
    importance[x.first] = x.second.total_score / x.second.num_usage;
  }
  return decision_tree::VariableImportanceMapToSortedVector(importance);
}

struct DiffiInlierOutlier {
  UnsignedExampleIdx inliers = 0;
  UnsignedExampleIdx outliers = 0;

  // Add a constructor for C++17 compatibility.
  DiffiInlierOutlier(UnsignedExampleIdx inliers_val,
                     UnsignedExampleIdx outliers_val)
      : inliers(inliers_val), outliers(outliers_val) {}
};

struct DiffiIIC {
  double iic_in = 0.;
  double iic_out = 0.;
};

struct DiffiCFI {
  UnsignedExampleIdx usage_in = 0;
  UnsignedExampleIdx usage_out = 0;
  double cfi_in = 0;
  double cfi_out = 0;
};

// Score the training examples in `node` as inliers and outliers. The result is
// stored in pre-order in `inlier_outlier_counter`.
DiffiInlierOutlier PredictTrainingExamples(
    const decision_tree::NodeWithChildren& node,
    const UnsignedExampleIdx num_examples_per_tree,
    std::vector<DiffiInlierOutlier>& inlier_outlier_counter) {
  if (node.IsLeaf()) {
    const UnsignedExampleIdx num_examples =
        node.node().anomaly_detection().num_examples_without_weight();
    const int cur_depth = node.depth() + PreissAveragePathLength(num_examples);
    // Use 0.5 as the hard cutoff for inlier and outlier.
    if (IsolationForestPrediction(cur_depth, num_examples_per_tree) >= 0.5) {
      inlier_outlier_counter.emplace_back(0, num_examples);
    } else {
      inlier_outlier_counter.emplace_back(num_examples, 0);
    }
    return inlier_outlier_counter.back();
  } else {
    inlier_outlier_counter.emplace_back(-1, -1);  // Placeholder
    size_t current_index = inlier_outlier_counter.size() - 1;
    auto pos_counter = PredictTrainingExamples(
        *node.pos_child(), num_examples_per_tree, inlier_outlier_counter);
    auto neg_counter = PredictTrainingExamples(
        *node.neg_child(), num_examples_per_tree, inlier_outlier_counter);

    UnsignedExampleIdx total_inliers =
        neg_counter.inliers + pos_counter.inliers;
    UnsignedExampleIdx total_outliers =
        neg_counter.outliers + pos_counter.outliers;

    inlier_outlier_counter[current_index] = {total_inliers, total_outliers};
    return inlier_outlier_counter[current_index];
  }
}

// Compute the cumulative feature importances recursively by traversing the
// paths to the leaves. Corresponds to most of Algorithm 2 in the paper.
absl::Status ComputeCFIs(
    std::vector<std::pair<int, DiffiIIC>>& feature_and_iics_on_path,
    int& node_idx, const decision_tree::NodeWithChildren& node,
    const std::vector<DiffiIIC>& iics,
    const std::vector<DiffiInlierOutlier>& training_predictions,
    absl::flat_hash_map<int, DiffiCFI>& importances) {
  const auto current_pred = training_predictions[node_idx];
  STATUS_CHECK_GE(current_pred.inliers, 0);
  STATUS_CHECK_GE(current_pred.outliers, 0);
  const auto current_iics = iics[node_idx];
  STATUS_CHECK_GE(current_iics.iic_in, -1.1);   // Can be -1 for trivial split.
  STATUS_CHECK_GE(current_iics.iic_out, -1.1);  // Can be -1 for trivial split.
  node_idx++;
  if (node.IsLeaf()) {
    const auto depth = node.depth();
    STATUS_CHECK_GT(depth, 0);
    for (const auto [feature, iics] : feature_and_iics_on_path) {
      auto& importance = importances[feature];
      // The paper doesn't make it clear if the counter should increase for
      // nodes with zero IIC. In the official implementation, these node are not
      // counted.
      if (iics.iic_in >= 0) {
        importance.usage_in += current_pred.inliers;
        importance.cfi_in += current_pred.inliers * (iics.iic_in / depth);
      }
      if (iics.iic_out >= 0) {
        importance.usage_out += current_pred.outliers;
        importance.cfi_out += current_pred.outliers * (iics.iic_out / depth);
      }
    }
  } else {
    const auto feature = node.node().condition().attribute();
    feature_and_iics_on_path.push_back({feature, current_iics});
    RETURN_IF_ERROR(ComputeCFIs(feature_and_iics_on_path, node_idx,
                                *node.pos_child(), iics, training_predictions,
                                importances));
    RETURN_IF_ERROR(ComputeCFIs(feature_and_iics_on_path, node_idx,
                                *node.neg_child(), iics, training_predictions,
                                importances));
    feature_and_iics_on_path.pop_back();
  }
  return absl::OkStatus();
}

// DIFFI score. See https://arxiv.org/abs/2007.11117.
absl::StatusOr<std::vector<model::proto::VariableImportance>>
StructureDIFFIScore(
    const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
        decision_trees) {
  struct ImportancePerFeature {
    double total_score = 0;
    UnsignedExampleIdx num_usage = 0;
  };
  // Compute the Induced Imbalance Coefficient (IIC) of a node according to
  // equations (4) -- (6) in the paper.
  const auto compute_iic = [](UnsignedExampleIdx n_cur,
                              UnsignedExampleIdx n_neg,
                              UnsignedExampleIdx n_pos) -> double {
    DCHECK_GE(n_neg, 0);
    DCHECK_GE(n_pos, 0);
    DCHECK_EQ(n_cur, n_neg + n_pos);
    double iic = 0;
    if (n_cur == 0 || n_cur == 1) {
      // Do not assign an IIC for trivial splits. This condition is not explicit
      // in the paper, but it makes sense and exists in the reference
      // implementation.
      return -1.;
    }
    if (n_neg > 0 && n_pos > 0) {
      DCHECK_GT(n_cur, 0);
      const double lambda_min = static_cast<double>(n_cur / 2) / n_cur;
      const double lambda_max = static_cast<double>(n_cur - 1) / n_cur;
      iic = static_cast<double>(std::max(n_neg, n_pos)) / n_cur;
      if (lambda_min != lambda_max) {
        iic = (iic - lambda_min) / (2 * (lambda_max - lambda_min)) + 0.5;
      }
    }
    DCHECK(!std::isnan(iic));
    return iic;
  };
  bool has_training_information = true;

  // For each feature, stores its CFI.
  absl::flat_hash_map<int, DiffiCFI> cfis;
  for (auto& tree : decision_trees) {
    if (tree->root().IsLeaf()) {
      continue;
    }
    const auto& root = tree->root();
    if (!root.node().condition().has_num_training_examples_without_weight()) {
      has_training_information = false;
      break;
    }
    // For each tree node (in pre-order), record the number of
    // inliers and outliers and, in a second step, the corresponding IICs.
    std::vector<DiffiInlierOutlier> training_example_predictions;
    training_example_predictions.reserve(tree->NumNodes());
    std::vector<DiffiIIC> iics(tree->NumNodes());
    PredictTrainingExamples(
        root, root.node().condition().num_training_examples_without_weight(),
        training_example_predictions);
    int depth_first_index = 0;
    // For each internal node, compute the IICs (Algorithm 1 in the paper).
    // This code assumes that IterateOnNodes follows pre-order.
    tree->IterateOnNodes([&](const decision_tree::NodeWithChildren& node,
                             const int depth) {
      if (!node.IsLeaf()) {
        DCHECK_LT(depth_first_index + node.pos_child()->NumNodes() + 1,
                  iics.size());
        const auto& cur_pred = training_example_predictions[depth_first_index];
        const auto& pos_pred =
            training_example_predictions[depth_first_index + 1];
        const auto& neg_pred =
            training_example_predictions[depth_first_index +
                                         node.pos_child()->NumNodes() + 1];
        const double iic_in =
            compute_iic(cur_pred.inliers, pos_pred.inliers, neg_pred.inliers);
        const double iic_out = compute_iic(cur_pred.outliers, pos_pred.outliers,
                                           neg_pred.outliers);
        iics[depth_first_index] = {iic_in, iic_out};
      }
      depth_first_index++;
    });

    std::vector<std::pair<int, DiffiIIC>> feature_and_iics_on_path;
    int node_idx = 0;
    RETURN_IF_ERROR(ComputeCFIs(feature_and_iics_on_path, node_idx, root, iics,
                                training_example_predictions, cfis));
  }

  if (!has_training_information) {
    LOG(INFO)
        << "This model is missing some training information, cannot compute "
        << kVariableImportanceDIFFI;
    return std::vector<model::proto::VariableImportance>();
  }

  // Convert the CFIs into feature importances by normalizing between inliers
  // and outliers.
  absl::flat_hash_map<int, double> importance;
  for (const auto& [feature, cfi] : cfis) {
    STATUS_CHECK_GE(cfi.usage_in, 0);
    STATUS_CHECK_GE(cfi.usage_out, 0);
    STATUS_CHECK_GE(cfi.cfi_in, 0);
    STATUS_CHECK_GE(cfi.cfi_out, 0);
    if (cfi.cfi_in * cfi.usage_out == 0) {
      importance[feature] = 0.0;
    } else {
      importance[feature] =
          (cfi.cfi_out * cfi.usage_in) / (cfi.cfi_in * cfi.usage_out);
    }
  }
  return decision_tree::VariableImportanceMapToSortedVector(importance);
}

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
  OPTIONAL_ASSIGN_OR_RETURN(const auto abstract_size,
                            AbstractAttributesSizeInBytes());
  OPTIONAL_ASSIGN_OR_RETURN(const auto tree_size,
                            decision_tree::EstimateSizeInByte(decision_trees_));
  return abstract_size + tree_size;
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

// Add a new tree to the model.
void IsolationForestModel::AddTree(
    std::unique_ptr<decision_tree::DecisionTree> decision_tree) {
  decision_trees_.push_back(std::move(decision_tree));
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

std::vector<std::string> IsolationForestModel::AvailableVariableImportances()
    const {
  auto variable_importances = AbstractModel::AvailableVariableImportances();
  const auto structural = AvailableStructuralVariableImportances();
  variable_importances.insert(variable_importances.end(), structural.begin(),
                              structural.end());

  // Remove possible duplicates.
  std::sort(variable_importances.begin(), variable_importances.end());
  variable_importances.erase(
      std::unique(variable_importances.begin(), variable_importances.end()),
      variable_importances.end());

  return variable_importances;
}

std::vector<std::string>
IsolationForestModel::AvailableStructuralVariableImportances() const {
  std::vector<std::string> variable_importances;
  variable_importances.push_back(kVariableImportanceDIFFI);
  variable_importances.push_back(kVariableImportanceMeanPartitionScore);
  variable_importances.push_back(
      decision_tree::kVariableImportanceNumberOfNodes);
  return variable_importances;
}

absl::StatusOr<std::vector<model::proto::VariableImportance>>
IsolationForestModel::GetVariableImportance(absl::string_view key) const {
  const auto general_vi = AbstractModel::GetVariableImportance(key);
  if (general_vi.ok()) {
    return std::move(general_vi.value());
  } else if (general_vi.status().code() == absl::StatusCode::kNotFound) {
    if (key == decision_tree::kVariableImportanceNumberOfNodes) {
      return decision_tree::StructureNumberOfTimesInNode(decision_trees());
    }
    if (key == kVariableImportanceMeanPartitionScore) {
      if (is_pure_model_) {
        LOG(INFO) << "Variable importance " << key
                  << " may not be available for pure serving models.";
      }
      return StructureMeanPartitionScore(decision_trees());
    }
    if (key == kVariableImportanceDIFFI) {
      if (is_pure_model_) {
        LOG(INFO) << "Variable importance " << key
                  << " may not be available for pure serving models.";
      }
      return StructureDIFFIScore(decision_trees());
    }
  }
  return general_vi.status();
}

REGISTER_AbstractModel(IsolationForestModel,
                       IsolationForestModel::kRegisteredName);

}  // namespace yggdrasil_decision_forests::model::isolation_forest
