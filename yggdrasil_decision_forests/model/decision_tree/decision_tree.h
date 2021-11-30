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

// Generic decision tree model and learning.
// Support multi-class classification and regression.
//
// For classification, the information gain is used as the split score. For
// regression, the standard deviation reduction is used as the split score.
#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_DECISION_TREE_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_DECISION_TREE_H_

#include <stddef.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/sharded_io.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

using row_t = dataset::VerticalDataset::row_t;

// The total number of "conditions" allocated in PerThreadCache.condition_list
// is equal to the number of threads times this factor.
//
// A larger value might increase the memory usage while a lower value might slow
// down the training."
constexpr int32_t kConditionPoolGrowthFactor = 2;

// Variable importance names to be used for all decision tree based model.
static constexpr char kVariableImportanceNumberOfNodes[] = "NUM_NODES";
static constexpr char kVariableImportanceNumberOfTimesAsRoot[] = "NUM_AS_ROOT";
static constexpr char kVariableImportanceSumScore[] = "SUM_SCORE";
static constexpr char kVariableImportanceMeanMinDepth[] = "MEAN_MIN_DEPTH";

// Return an identifier of a type of condition.
std::string ConditionTypeToString(proto::Condition::TypeCase type);

// Evaluate a condition on an example contained in a vertical dataset.
bool EvalCondition(const proto::NodeCondition& condition,
                   const dataset::VerticalDataset& dataset,
                   dataset::VerticalDataset::row_t example_idx);

bool EvalConditionFromColumn(
    const proto::NodeCondition& condition,
    const dataset::VerticalDataset::AbstractColumn* column_data,
    const dataset::VerticalDataset& dataset, row_t example_idx);

bool EvalCondition(const proto::NodeCondition& condition,
                   const dataset::proto::Example& example);

// A node and its two children (if any).
class NodeWithChildren {
 public:
  // Approximate size in memory (expressed in byte) of the node and all its
  // children.
  size_t EstimateSizeInByte() const;

  // Exports the node (and its children) to a RecordIO writer. The nodes are
  // stored sequentially with a depth-first exploration.
  absl::Status WriteNodes(utils::ShardedWriter<proto::Node>* writer) const;

  // Imports the node (and its children) to a RecordIO reader.
  absl::Status ReadNodes(utils::ShardedReader<proto::Node>* reader);

  // Indicates the node is a leaf i.e. if the node DOES NOT have children.
  bool IsLeaf() const { return !children_[0]; }

  // Clear the detailed label distribution i.e. we only keep the top category
  // (in case of classification) or the mean (in case of regression).
  void ClearLabelDistributionDetails();

  // See definition of "CountFeatureUsage" in Random Forest.
  void CountFeatureUsage(
      std::unordered_map<int32_t, int64_t>* feature_usage) const;

  const proto::Node& node() const { return node_; }

  proto::Node* mutable_node() { return &node_; }

  // The "positive" child i.e. the child that is responsible for the prediction
  // when the condition evaluates to true.
  const NodeWithChildren* pos_child() const { return children_[1].get(); }
  NodeWithChildren* mutable_pos_child() { return children_[1].get(); }

  // The "negative" child.
  const NodeWithChildren* neg_child() const { return children_[0].get(); }
  NodeWithChildren* mutable_neg_child() { return children_[0].get(); }

  // Instantiate the children.
  void CreateChildren();

  // Number of nodes.
  int64_t NumNodes() const;

  // Check the validity of a node and its children.
  absl::Status Validate(
      const dataset::proto::DataSpecification& data_spec,
      std::function<absl::Status(const decision_tree::proto::Node& node)>
          check_leaf) const;

  // Call "call_back" on the node and all its children.
  void IterateOnNodes(const std::function<void(const NodeWithChildren& node,
                                               const int depth)>& call_back,
                      int depth = 0) const;

  void IterateOnMutableNodes(
      const std::function<void(NodeWithChildren* node, const int depth)>&
          call_back,
      bool neg_before_pos_child, int depth);

  // Append a human readable semi-graphical description of the model structure.
  void AppendModelStructure(const dataset::proto::DataSpecification& data_spec,
                            const int label_col_idx, int depth,
                            std::string* description) const;

  // Convert a node that was previously "FinalizeAsNonLeaf" into a leaf.
  void TurnIntoLeaf();

  // Finalize the node structure as a leaf. After this function is called, this
  // node is guaranteed to be a leaf.
  void FinalizeAsLeaf(bool store_detailed_label_distribution);

  // Finalize the node structure as a non-leaf. After this function is called,
  // this node is guaranteed not to be a leaf.
  void FinalizeAsNonLeaf(bool keep_non_leaf_label_distribution,
                         bool store_detailed_label_distribution);

  // If true, all the "na_value" values of the conditions (i.e. the value of the
  // condition when a value is missing) is equal to the condition applied with
  // global imputation (i.e. replacing the value by the global mean or median;
  // see GLOBAL_IMPUTATION strategy for more details).
  //
  // Such models are faster to serve as the inference engine does not need to
  // check for the absence of value.
  bool IsMissingValueConditionResultFollowGlobalImputation(
      const dataset::proto::DataSpecification& data_spec) const;

  int32_t leaf_idx() const { return leaf_idx_; }
  void set_leaf_idx(const int32_t v) { leaf_idx_ = v; }

 private:
  // Node content (i.e. value and condition).
  proto::Node node_;

  // Children (if any).
  std::unique_ptr<NodeWithChildren> children_[2];

  // Index of the leaf (if the node is a leaf) in the tree in a depth first
  // exploration. It is set by calling "SetLeafIndices()".
  int32_t leaf_idx_ = -1;
};

// A generic decision tree. This class is designed for cheap modification (by
// opposition to fast serving).
class DecisionTree {
 public:
  // Estimates the memory usage of the model in RAM. The serialized or the
  // compiled version of the model can be much smaller.
  size_t EstimateModelSizeInBytes() const;

  // Number of nodes in the tree.
  int64_t NumNodes() const;

  // Number of leafs in the tree.
  int64_t NumLeafs() const;

  // Exports the tree to a RecordIO writer. Cannot export a tree without a root
  // node.
  absl::Status WriteNodes(utils::ShardedWriter<proto::Node>* writer) const;

  // Imports the tree from a RecordIO reader.
  absl::Status ReadNodes(utils::ShardedReader<proto::Node>* reader);

  // Creates a root node. Fails if the tree is not empty (i.e. if there is
  // already a root node).
  void CreateRoot();

  const NodeWithChildren& root() const { return *root_; }
  NodeWithChildren* mutable_root() const { return root_.get(); }

  // Check the validity of a tree.
  absl::Status Validate(
      const dataset::proto::DataSpecification& data_spec,
      std::function<absl::Status(const decision_tree::proto::Node& node)>
          check_leaf) const;

  // See definition of "CountFeatureUsage" in Random Forest.
  void CountFeatureUsage(
      std::unordered_map<int32_t, int64_t>* feature_usage) const;

  // Apply the decision tree on an example and returns the leaf.
  const proto::Node& GetLeaf(const dataset::VerticalDataset& dataset,
                             dataset::VerticalDataset::row_t row_idx) const;

  const proto::Node& GetLeaf(const dataset::proto::Example& example) const;

  const NodeWithChildren& GetLeafAlt(
      const dataset::VerticalDataset& dataset,
      dataset::VerticalDataset::row_t row_idx) const;

  // Apply the decision tree on an example and returns the path.
  const void GetPath(const dataset::VerticalDataset& dataset,
                     dataset::VerticalDataset::row_t row_idx,
                     std::vector<const NodeWithChildren*>* path) const;

  // Apply the decision tree similarly to "GetLeaf". However, during inference,
  // simulate the replacement of the value of the attribute
  // "selected_attribute_idx" with the value of the example
  // "row_id_for_selected_attribute" (instead of "row_idx" for the other
  // attributes).
  const proto::Node& GetLeafWithSwappedAttribute(
      const dataset::VerticalDataset& dataset,
      dataset::VerticalDataset::row_t row_idx, int selected_attribute_idx,
      dataset::VerticalDataset::row_t row_id_for_selected_attribute) const;

  // Call "call_back" on all the nodes of the model.
  void IterateOnNodes(
      const std::function<void(const NodeWithChildren& node, const int depth)>&
          call_back) const;

  void IterateOnMutableNodes(
      const std::function<void(NodeWithChildren* node, const int depth)>&
          call_back,
      bool neg_before_pos_child = false);

  // Append a human readable semi-graphical description of the model structure.
  void AppendModelStructure(const dataset::proto::DataSpecification& data_spec,
                            const int label_col_idx,
                            std::string* description) const;

  // Maximum depth of the forest. A depth of "0" means a stump i.e. a tree with
  // a single node.  A depth of "-1" is an empty tree.
  int MaximumDepth() const;

  // Scales the output of a regressor tree. If the tree is not a regressor, an
  // error is raised.
  void ScaleRegressorOutput(float scale);

  // See the same method in "NodeWithChildren".
  bool IsMissingValueConditionResultFollowGlobalImputation(
      const dataset::proto::DataSpecification& data_spec) const;

  // Set the "leaf_idx" field for all the leaves. The index of a leaf is
  // assigned in the depth first iteration over all the nods (negative before
  // positive).
  void SetLeafIndices();

 private:
  // Root of the decision tree.
  std::unique_ptr<NodeWithChildren> root_;
};

// A list of trees without specific semantic.
typedef std::vector<std::unique_ptr<DecisionTree>> DecisionForest;

// Sets the leaf indices of all the trees.
void SetLeafIndices(DecisionForest* trees);

// Estimate the size (in bytes) of a list of decision trees.
size_t EstimateSizeInByte(const DecisionForest& trees);

// Number of nodes in a list of decision trees.
int64_t NumberOfNodes(const DecisionForest& trees);

bool IsMissingValueConditionResultFollowGlobalImputation(
    const dataset::proto::DataSpecification& data_spec,
    const DecisionForest& trees);

// Append a human readable semi-graphical description of the model structure.
void AppendModelStructure(const DecisionForest& trees,
                          const dataset::proto::DataSpecification& data_spec,
                          const int label_col_idx, std::string* description);

// Gets the number of time each feature is used as root in a set of trees.
std::vector<model::proto::VariableImportance> StructureNumberOfTimesAsRoot(
    const DecisionForest& decision_trees);

// Gets the number of time each feature is used in a set of trees.
std::vector<model::proto::VariableImportance> StructureNumberOfTimesInNode(
    const DecisionForest& decision_trees);

// Gets the average minimum depth of each feature.
std::vector<model::proto::VariableImportance> StructureMeanMinDepth(
    const DecisionForest& decision_trees, int num_features);

// Gets the weighted sum of the score (the semantic of the score depends on the
// loss function) of each feature.
std::vector<model::proto::VariableImportance> StructureSumScore(
    const DecisionForest& decision_trees);

// Append a human readable description of a node condition (i.e. split).
void AppendConditionDescription(
    const dataset::proto::DataSpecification& data_spec,
    const proto::NodeCondition& node, std::string* description);

// Checks if two sets represented by sorted containers intersect i.e. have at
// least one element in common.
template <typename Iter1, typename Iter2>
bool DoSortedRangesIntersect(Iter1 begin1, Iter1 end1, Iter2 begin2,
                             Iter2 end2) {
  DCHECK(std::is_sorted(begin1, end1));
  DCHECK(std::is_sorted(begin2, end2));
  while (begin1 != end1 && begin2 != end2) {
    if (*begin1 < *begin2) {
      ++begin1;
      continue;
    }
    if (*begin2 < *begin1) {
      ++begin2;
      continue;
    }
    return true;
  }
  return false;
}

// Extracts the list of positive elements from a "contains" type conditions.
std::vector<int32_t> ExactElementsFromContainsCondition(
    int vocab_size, const proto::Condition& condition);

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_DECISION_TREE_H_
