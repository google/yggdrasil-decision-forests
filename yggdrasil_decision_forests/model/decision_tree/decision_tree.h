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

// Generic decision tree model and learning.
// Support multi-class classification and regression.
//
// For classification, the information gain is used as the split score. For
// regression, the standard deviation reduction is used as the split score.
#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_DECISION_TREE_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_DECISION_TREE_H_

#include <stddef.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

using row_t = dataset::VerticalDataset::row_t;

// Variable importance names to be used for all decision tree based model.
static constexpr char kVariableImportanceNumberOfNodes[] = "NUM_NODES";
static constexpr char kVariableImportanceNumberOfTimesAsRoot[] = "NUM_AS_ROOT";
static constexpr char kVariableImportanceSumScore[] = "SUM_SCORE";
// "INV_MEAN_MIN_DEPTH" is 1/(1+d) where "d" is the average depth of the feature
// in the forest.
static constexpr char kVariableImportanceInvMeanMinDepth[] =
    "INV_MEAN_MIN_DEPTH";

// Return an identifier of a type of condition.
std::string ConditionTypeToString(proto::Condition::TypeCase type);

// Evaluate a condition on an example contained in a vertical dataset.
absl::StatusOr<bool> EvalCondition(const proto::NodeCondition& condition,
                                   const dataset::VerticalDataset& dataset,
                                   dataset::VerticalDataset::row_t example_idx);

absl::StatusOr<bool> EvalConditionFromColumn(
    const proto::NodeCondition& condition,
    const dataset::VerticalDataset::AbstractColumn* column_data,
    const dataset::VerticalDataset& dataset, row_t example_idx);

absl::StatusOr<bool> EvalCondition(const proto::NodeCondition& condition,
                                   const dataset::proto::Example& example);

// A list of selected examples, and a related buffer used to do some
// computation.
struct SelectedExamplesRollingBuffer {
  absl::Span<UnsignedExampleIdx> active;
  absl::Span<UnsignedExampleIdx> inactive;

  size_t size() const { return active.size(); }
  bool empty() const { return active.empty(); }

  static SelectedExamplesRollingBuffer Create(
      const absl::Span<UnsignedExampleIdx> active,
      std::vector<UnsignedExampleIdx>* buffer) {
    buffer->resize(active.size());
    return {.active = active, .inactive = absl::MakeSpan(*buffer)};
  }
};

struct ExampleSplitRollingBuffer {
  SelectedExamplesRollingBuffer positive_examples;
  SelectedExamplesRollingBuffer negative_examples;

  size_t num_positive() const { return positive_examples.size(); }
  size_t num_negative() const { return negative_examples.size(); }
};

absl::Status EvalConditionOnDataset(const dataset::VerticalDataset& dataset,
                                    SelectedExamplesRollingBuffer examples,
                                    const proto::NodeCondition& condition,
                                    bool dataset_is_dense,
                                    ExampleSplitRollingBuffer* example_split);

// Argument to the "CheckStructure" method that tests various aspects of the
// model structure. By default, "CheckStructureOptions" checks if the model
// was trained with global imputation.
struct CheckStructureOptions {
  // "global_imputation_*" tests if the model structure looks as if it was
  // trained with global imputation. That is, the "na_value" values of the
  // conditions (i.e. the value of the condition when a value is missing) are
  // equal to the condition applied with global imputation feature replacement
  // (i.e. replacing the value by the global mean or median; see
  // GLOBAL_IMPUTATION strategy for more details).

  // For the "is higher" conditions.
  bool global_imputation_is_higher = true;

  // For all the other conditions.
  bool global_imputation_others = true;

  // Check if the model does not contain any IsNA condition. Should not be
  // combined with other options.
  bool check_no_na_conditions = false;

  static CheckStructureOptions GlobalImputation() {
    return {
        /*.global_imputation_is_higher =*/true,
        /*.global_imputation_others =*/true,
        /*.check_no_na_conditions =*/false,
    };
  }

  static CheckStructureOptions NACondition() {
    return {
        /*.global_imputation_is_higher =*/false,
        /*.global_imputation_others =*/false,
        /*.check_no_na_conditions =*/true,
    };
  }
};

// A node and its two children (if any).
class NodeWithChildren {
 public:
  // Approximate size in memory (expressed in bytes) of the node and all its
  // children.
  std::optional<size_t> EstimateSizeInByte() const;

  // Exports the node (and its children) to a RecordIO writer. The nodes are
  // stored sequentially with a depth-first exploration.
  absl::Status WriteNodes(
      utils::ProtoWriterInterface<proto::Node>* writer) const;

  // Imports the node (and its children) to a RecordIO reader.
  absl::Status ReadNodes(utils::ProtoReaderInterface<proto::Node>* reader);

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

  // Removes the children.
  void ClearChildren();

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
                            int label_col_idx, int depth,
                            std::optional<bool> is_pos,
                            const std::string& prefix,
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

  // Tests if the model satisfy the condition defined in
  // "CheckStructureOptions".
  bool CheckStructure(const CheckStructureOptions& options,
                      const dataset::proto::DataSpecification& data_spec) const;

  int32_t leaf_idx() const { return leaf_idx_; }
  void set_leaf_idx(const int32_t v) { leaf_idx_ = v; }

  int32_t depth() const { return depth_; }
  void set_depth(const int32_t v) { depth_ = v; }

  // Compare a tree to another tree. If they are equal, return an empty string.
  // If they are different, returns an explanation of the differences.
  std::string DebugCompare(const dataset::proto::DataSpecification& dataspec,
                           const int label_idx,
                           const NodeWithChildren& other) const;

 private:
  // Node content (i.e. value and condition).
  proto::Node node_;

  // Children (if any).
  std::unique_ptr<NodeWithChildren> children_[2];

  // Index of the leaf (if the node is a leaf) in the tree in a depth first
  // exploration. It is set by calling "SetLeafIndices()".
  int32_t leaf_idx_ = -1;

  // Depth of the node. Assuming that the root node has depth 0. It is set by
  // calling "SetLeafIndices()".
  int32_t depth_ = -1;
};

// A generic decision tree. This class is designed for cheap modification (by
// opposition to fast serving).
class DecisionTree {
 public:
  // Estimates the memory usage of the model in RAM. The serialized or the
  // compiled version of the model can be much smaller.
  std::optional<size_t> EstimateModelSizeInBytes() const;

  // Number of nodes in the tree.
  int64_t NumNodes() const;

  // Number of leafs in the tree.
  int64_t NumLeafs() const;

  // Exports the tree to a RecordIO writer. Cannot export a tree without a root
  // node.
  absl::Status WriteNodes(
      utils::ProtoWriterInterface<proto::Node>* writer) const;

  // Imports the tree from a RecordIO reader.
  absl::Status ReadNodes(utils::ProtoReaderInterface<proto::Node>* reader);

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

  const NodeWithChildren& GetLeafAlt(
      const dataset::proto::Example& example) const;

  // Apply the decision tree on an example and returns the path.
  void GetPath(const dataset::VerticalDataset& dataset,
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

  // Tests if the model satisfy the condition defined in
  // "CheckStructureOptions".
  bool CheckStructure(const CheckStructureOptions& options,
                      const dataset::proto::DataSpecification& data_spec) const;

  // Set the "leaf_idx" field for all the leaves. The index of a leaf is
  // assigned in the depth first iteration over all the nods (negative before
  // positive).
  void SetLeafIndices();

  // Compare a tree to another tree. If they are equal, return an empty string.
  // If they are different, returns an explanation of the differences.
  std::string DebugCompare(const dataset::proto::DataSpecification& dataspec,
                           int label_idx, const DecisionTree& other) const;

 private:
  // Root of the decision tree.
  std::unique_ptr<NodeWithChildren> root_;
};

// A list of trees without specific semantic.
typedef std::vector<std::unique_ptr<DecisionTree>> DecisionForest;

// Sets the leaf indices of all the trees.
void SetLeafIndices(DecisionForest* trees);

// Estimate the size (in bytes) of a list of decision trees.
// Returns 0 if the size cannot be estimated.
std::optional<size_t> EstimateSizeInByte(const DecisionForest& trees);

// Number of nodes in a list of decision trees.
int64_t NumberOfNodes(const DecisionForest& trees);

// Tests if the model satisfy the condition defined in
// "CheckStructureOptions".
bool CheckStructure(const CheckStructureOptions& options,
                    const dataset::proto::DataSpecification& data_spec,
                    const DecisionForest& trees);

// Append a human readable semi-graphical description of the model structure.
void AppendModelStructure(const DecisionForest& trees,
                          const dataset::proto::DataSpecification& data_spec,
                          int label_col_idx, std::string* description);

// Append the header of AppendModelStructure.
void AppendModelStructureHeader(
    const DecisionForest& trees,
    const dataset::proto::DataSpecification& data_spec, int label_col_idx,
    std::string* description);

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

// Computes the pairwise distance between examples in "dataset1" and
// "dataset2".
//
// The distance is computed as one minus the ratio of common active leaves
// between two examples.
//
// "distances[i * dataset2.nrows() +j]" will be the distance between the i-th
// example of "dataset1" and the j-th example of "dataset2".
//
// "tree_weights" are optional tree weights. If specified, the size of
// "tree_weights" should be the same as "trees".
absl::Status Distance(
    absl::Span<const std::unique_ptr<decision_tree::DecisionTree>> trees,
    const dataset::VerticalDataset& dataset1,
    const dataset::VerticalDataset& dataset2, absl::Span<float> distances,
    const std::optional<std::reference_wrapper<std::vector<float>>>&
        tree_weights = {});

// Lists the input features used by the trees. The input features are given as
// sorted column indices.
std::vector<int> input_features(
    absl::Span<const std::unique_ptr<decision_tree::DecisionTree>> trees);

// Compare two forests. If they are equal, return an empty string. If they are
// different, returns an explanation of the differences.
std::string DebugCompare(
    const dataset::proto::DataSpecification& dataspec, int label_idx,
    absl::Span<const std::unique_ptr<decision_tree::DecisionTree>> a,
    absl::Span<const std::unique_ptr<decision_tree::DecisionTree>> b);

// Square of the euclidean distance between two vectors.
float SquaredDistance(absl::Span<const float> a, absl::Span<const float> b);

// A dot product between two vectors.
float DotProduct(absl::Span<const float> a, absl::Span<const float> b);

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_DECISION_TREE_DECISION_TREE_H_
