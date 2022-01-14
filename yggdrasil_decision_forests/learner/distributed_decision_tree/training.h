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

// Utility for the distributed training (compute and memory) of individual
// decision trees.
//
// Glossary:
//
// An "open node" is a leaf that is candidate to be splitted i.e. the node can
// be transformed in a non-leaf node. In layer-wise learning, the open nodes are
// all the nodes in the currently trained layer that satisfy the nodes splitting
// constraints (e.g. minimum number of examples).
//
// A "column index" is a dense index identifying a column. A column can be a
// label, input feature, weight, ranking group, ignored column, etc.
//
// A "feature index" is a "column index" where we know that the column is an
// input features.
//
// A "split evaluation" is an array of bits (one bit for each example in an open
// node) that describes the result of evaluating a split/condition on those
// examples.
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_TRAINING_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_TRAINING_H_

#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/label_accessor.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/splitter.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/training.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {

// How to re-map the node index after a split.
//
// NodeRemapping a;
// a[i].indices[0(neg)] => The index of negative node created after the split of
// the i-th open node.
struct SplitNodeIndex {
  // Negative (0) and positive (1) node indices.
  NodeIndex indices[2];
};
typedef std::vector<SplitNodeIndex> NodeRemapping;

// Bitmap of the evaluation of a split i.e. evaluation of the boolean condition
// defined by the split.
//
// SplitEvaluationPerOpenNode a;
// ReadBitmap(/*bitmap=*/a[i], /*bit_index=*/j) is the boolean evaluation of a
// split on the j-th example contained in the i-th open node.
typedef std::string SplitEvaluation;
typedef std::vector<SplitEvaluation> SplitEvaluationPerOpenNode;

// Selection of a label accessor.
enum class LabelAccessorType {
  kAutomatic,  // Select the label accessor matching the task.
  kNumericalWithHessian
};

// Default logic to set the value of a leaf.
absl::Status SetLeafValue(
    const decision_tree::proto::LabelStatistics& label_stats,
    decision_tree::proto::Node* leaf);

// Signature of a function that sets the value (i.e. the prediction) of a leaf
// from the gradient label statistics.
typedef std::function<absl::Status(
    const decision_tree::proto::LabelStatistics& label_stats,
    decision_tree::proto::Node* leaf)>
    SetLeafValueFromLabelStatsFunctor;

// A decision tree being build.
//
// Usage example:
//
//   // Create the builder.
//   auto builder = TreeBuilder::Create();
//
//   // Compute the statistics of the labels.
//   builder->AggregateLabelStatistics(&label_satistics);
//
//   // Create the root node.
//   builder->SetRootValue(label_satistics);
//
//   // Initialy, all the examples are in the root (which is the only node).
//   auto example_to_node = CreateExampleToNodeMap(num_examples);
//   LabelStatsPerNode label_stats_per_node({label_satistics});
//
//   for(int i=0; i<=5; i++) { // Train until depth 5
//
//     // Look for splits.
//     SplitPerOpenNode splits;
//     builder->FindBestSplits(example_to_node, label_stats_per_node, &splits);
//
//     // Update the tree structure with the splits.
//     auto node_remapping = tree_builder->ApplySplitToTree(splits);
//
//     // Evaluate the split on all the examples.
//     SplitEvaluationPerOpenNode split_evaluation;
//     EvaluateSplits(example_to_node, splits, &split_evaluation);
//
//     // Update the example->node mapping.
//     UpdateExampleNodeMap(splits, split_evaluation, node_remapping,
//     &example_to_node);
//
//     // Update the statistics of the label in the open nodes.
//     UpdateLabelStatistics(splits, node_remapping, &label_stats_per_node);
//   }
//
// // Do something with the tree.
// builder->mutable_tree()
//
class TreeBuilder {
 public:
  // Creates a new builder.
  static utils::StatusOr<std::unique_ptr<TreeBuilder>> Create(
      const model::proto::TrainingConfig& config,
      const model::proto::TrainingConfigLinking& config_link,
      const decision_tree::proto::DecisionTreeTrainingConfig& dt_config,
      LabelAccessorType label_accessor_type_ = LabelAccessorType::kAutomatic,
      SetLeafValueFromLabelStatsFunctor set_leaf_functor = SetLeafValue);

  // Computes the label statistics over all the examples.
  absl::Status AggregateLabelStatistics(
      const AbstractLabelAccessor& label_accessor,
      decision_tree::proto::LabelStatistics* label_stats,
      utils::concurrency::ThreadPool* thread_pool) const;

  // Finds the best splits for the open nodes.
  absl::Status FindBestSplits(const FindBestSplitsCommonArgs& common) const;

  // Finds the best splits for the open nodes. Unlike "FindBestSplits",
  // "FindBestSplitsWithThreadPool" schedules the split finding in the thread
  // pool and returns immediately. Multiple "FindBestSplitsWithThreadPool" can
  // run at the same time in the same thread pool (as long as the output
  // variables are different or protected by the same mutex).
  //
  // Usage example:
  //
  //  utils::concurrency::BlockingCounter counter(2);
  //  utils::concurrency::Mutex mutex_1, mutex_2;
  //   FindBestSplitsWithThreadPool({&splits_1}, ..., &mutex_1, &counter,...);
  //   FindBestSplitsWithThreadPool({&splits_2}, ..., &mutex_2, &counter,...);
  //   // The two "FindBestSplitsWithThreadPool" are running in parallel. In
  //   // addition, the individual features (in both
  //   // FindBestSplitsWithThreadPool calls are evaluated in parallel).
  //   counter.Wait();
  //   // The work of both FindBestSplitsWithThreadPool calls is available.
  //
  // Other example (using the same mutex and splits):
  //
  //  utils::concurrency::BlockingCounter counter(2);
  //  utils::concurrency::Mutex mutex;
  //   FindBestSplitsWithThreadPool({&splits}, ..., &mutex, &counter,...);
  //   FindBestSplitsWithThreadPool({&splits}, ..., &mutex, &counter,...);
  //   // The two "FindBestSplitsWithThreadPool" are running in parallel. In
  //   // addition, the individual features (in both
  //   // FindBestSplitsWithThreadPool calls are evaluated in parallel).
  //   counter.Wait();
  //   // The work of both FindBestSplitsWithThreadPool calls is available.
  //
  // Args:
  //   common: Input and outputs arguments. Same as "FindBestSplits".
  //   unique_active_features: Number of unique features to test among all the
  //     leaves i.e. the intersection of "common.features_per_open_node".
  //   thread_pool: Thread pool where to run the jobs. Will run one jobs per
  //     unique active feature.
  //   mutex: Protect "status" and "common.best_splits".
  //   counter: "counter.DecrementCount" is called each time a job is done i.e.
  //     once per unique active features.
  //   status: Aggregated status of the jobs.
  absl::Status FindBestSplitsWithThreadPool(
      const FindBestSplitsCommonArgs& common,
      const std::vector<int>& unique_active_features,
      utils::concurrency::ThreadPool* thread_pool,
      utils::concurrency::Mutex* mutex,
      utils::concurrency::BlockingCounter* counter, absl::Status* status) const;

  // Applies a list of splits (one for each open node) to the tree structure.
  utils::StatusOr<NodeRemapping> ApplySplitToTree(
      const SplitPerOpenNode& splits);

  // Creates a remapping of example->node that will close all the open nodes.
  //
  // Note that this function has not impact on the tree being build.
  NodeRemapping CreateClosingNodeRemapping() const;

  // Initializes the "value" of the tree root i.e. the predicted value is the
  // root was a leaf.
  absl::Status SetRootValue(
      const decision_tree::proto::LabelStatistics& label_stats);

  const decision_tree::DecisionTree& tree() const { return tree_; }

  decision_tree::DecisionTree* mutable_tree() { return &tree_; }

  size_t num_open_nodes() const { return open_nodes_.size(); }

  const SetLeafValueFromLabelStatsFunctor& set_leaf_functor() const {
    return set_leaf_functor_;
  }

 private:
  TreeBuilder(const model::proto::TrainingConfig& config,
              const model::proto::TrainingConfigLinking& config_link,
              const decision_tree::proto::DecisionTreeTrainingConfig& dt_config,
              LabelAccessorType label_accessor_type_,
              SetLeafValueFromLabelStatsFunctor set_leaf_functor);

  // Specialization of "FindBestSplits" for a single feature.
  absl::Status FindBestSplitsWithFeature(const FindBestSplitsCommonArgs& common,
                                         int feature, int num_theads) const;

  // Specialization of "FindBestSplitsWithFeatureNumerical" for a single
  // sorted numerical feature.
  absl::Status FindBestSplitsWithFeatureSortedNumerical(
      const FindBestSplitsCommonArgs& common, int feature,
      const std::vector<bool>& is_target_node) const;

  // Specialization of "FindBestSplitsWithFeatureNumerical" for a single
  // discretized numerical feature.
  absl::Status FindBestSplitsWithFeatureDiscretizedNumerical(
      const FindBestSplitsCommonArgs& common, int feature,
      const std::vector<bool>& is_target_node, int num_threads) const;

  // Specialization of "FindBestSplitsWithFeatureNumerical" for a single
  // categorical feature.
  absl::Status FindBestSplitsWithFeatureCategorical(
      const FindBestSplitsCommonArgs& common, int feature,
      const std::vector<bool>& is_target_node) const;

  // Specialization of "FindBestSplitsWithFeatureNumerical" for a single
  // boolean feature.
  absl::Status FindBestSplitsWithFeatureBoolean(
      const FindBestSplitsCommonArgs& common, int feature,
      const std::vector<bool>& is_target_node) const;

  // Training configuration of the tree.
  model::proto::TrainingConfig config_;
  model::proto::TrainingConfigLinking config_link_;
  decision_tree::proto::DecisionTreeTrainingConfig dt_config_;

  // Tree in construction.
  decision_tree::DecisionTree tree_;

  // List of open nodes i.e. leaf that can later be divided.
  std::vector<decision_tree::NodeWithChildren*> open_nodes_;

  // How to build the label accessor.
  LabelAccessorType label_accessor_type_ = LabelAccessorType::kAutomatic;

  SetLeafValueFromLabelStatsFunctor set_leaf_functor_;
};

// Computes the label statistics over all the examples.
absl::Status AggregateLabelStatistics(
    const AbstractLabelAccessor& label_accessor, model::proto::Task task,
    LabelAccessorType label_accessor_type_,
    decision_tree::proto::LabelStatistics* label_stats,
    utils::concurrency::ThreadPool* thread_pool);

// Initialize an example->node mapping where all the examples are in node 0.
ExampleToNodeMap CreateExampleToNodeMap(ExampleIndex num_examples);

// Merges two sets of splits element per element. "dst" will contain the split
// (from "src" or "dst") with the highest scores.
absl::Status MergeBestSplits(const SplitPerOpenNode& src,
                             SplitPerOpenNode* dst);

// Evaluates a collection of splits.
absl::Status EvaluateSplits(const ExampleToNodeMap& example_to_node,
                            const SplitPerOpenNode& splits,
                            SplitEvaluationPerOpenNode* split_evaluation,
                            dataset_cache::DatasetCacheReader* dataset,
                            utils::concurrency::ThreadPool* thread_pool);

// Evaluates a collection of splits on a specific numerical feature.
absl::Status EvaluateSplitsPerNumericalFeature(
    const ExampleToNodeMap& example_to_node, const SplitPerOpenNode& splits,
    FeatureIndex feature, const std::vector<int>& active_node_idxs,
    SplitEvaluationPerOpenNode* split_evaluation,
    dataset_cache::DatasetCacheReader* dataset);

// Evaluates a collection of splits on a specific categorical feature.
absl::Status EvaluateSplitsPerCategoricalFeature(
    const ExampleToNodeMap& example_to_node, const SplitPerOpenNode& splits,
    FeatureIndex feature, const std::vector<int>& active_node_idxs,
    SplitEvaluationPerOpenNode* split_evaluation,
    dataset_cache::DatasetCacheReader* dataset);

// Evaluates a collection of splits on a specific boolean feature.
absl::Status EvaluateSplitsPerBooleanFeature(
    const ExampleToNodeMap& example_to_node, const SplitPerOpenNode& splits,
    FeatureIndex feature, const std::vector<int>& active_node_idxs,
    SplitEvaluationPerOpenNode* split_evaluation,
    dataset_cache::DatasetCacheReader* dataset);

// Update the node index of each example according to the split.
absl::Status UpdateExampleNodeMap(
    const SplitPerOpenNode& splits,
    const SplitEvaluationPerOpenNode& split_evaluation,
    const NodeRemapping& node_remapping, ExampleToNodeMap* example_to_node,
    utils::concurrency::ThreadPool* thread_pool);

absl::Status UpdateLabelStatistics(const SplitPerOpenNode& splits,
                                   const NodeRemapping& node_remapping,
                                   LabelStatsPerNode* label_stats);

// Gets the number of valid splits.
int NumValidSplits(const SplitPerOpenNode& splits);

// Tests if a split is valid.
bool IsSplitValid(const Split& split);

// Converts between the proto and struct representation.
void ConvertFromProto(const proto::SplitPerOpenNode& src,
                      SplitPerOpenNode* dst);
void ConvertToProto(const SplitPerOpenNode& src, proto::SplitPerOpenNode* dst);

// Only converts the splits with index in "split_idxs". Other splits are set to
// be invalid.
void ConvertToProto(const SplitPerOpenNode& src,
                    const std::vector<int>& split_idxs,
                    proto::SplitPerOpenNode* dst);

}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_TRAINING_H_
