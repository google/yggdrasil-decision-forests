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

// Random Forest model.
//
#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_RANDOM_FOREST_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_RANDOM_FOREST_H_

#include <stddef.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/distribution.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace random_forest {

class RandomForestModel : public AbstractModel, public DecisionForestInterface {
 public:
  static constexpr char kRegisteredName[] = "RANDOM_FOREST";

  // Variable importance introduced by Breiman
  // (https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf).
  static constexpr char kVariableImportanceMeanDecreaseInAccuracy[] =
      "MEAN_DECREASE_IN_ACCURACY";

  static constexpr char kVariableImportanceMeanIncreaseInRmse[] =
      "MEAN_INCREASE_IN_RMSE";

  RandomForestModel() : AbstractModel(kRegisteredName) {}
  absl::Status Save(absl::string_view directory,
                    const ModelIOOptions& io_options) const override;
  absl::Status Load(absl::string_view directory,
                    const ModelIOOptions& io_options) override;

  absl::Status Validate() const override;

  // Compute a single prediction of a model on a VerticalDataset. See the
  // documentation of "AbstractModel" for mode details.
  void Predict(const dataset::VerticalDataset& dataset,
               dataset::VerticalDataset::row_t row_idx,
               model::proto::Prediction* prediction) const override;

  // Compute a single prediction for a classification random forest.
  void PredictClassification(const dataset::VerticalDataset& dataset,
                             dataset::VerticalDataset::row_t row_idx,
                             model::proto::Prediction* prediction) const;

  // Compute a single prediction for a regression random forest.
  void PredictRegression(const dataset::VerticalDataset& dataset,
                         dataset::VerticalDataset::row_t row_idx,
                         model::proto::Prediction* prediction) const;

  // Compute a single prediction for an uplift random forest.
  void PredictUplift(const dataset::VerticalDataset& dataset,
                     dataset::VerticalDataset::row_t row_idx,
                     model::proto::Prediction* prediction) const;

  void Predict(const dataset::proto::Example& example,
               model::proto::Prediction* prediction) const override;

  // Compute a single prediction for a classification random forest.
  void PredictClassification(const dataset::proto::Example& example,
                             model::proto::Prediction* prediction) const;

  // Compute a single prediction for a regression random forest.
  void PredictRegression(const dataset::proto::Example& example,
                         model::proto::Prediction* prediction) const;

  // Compute a single prediction for an uplift random forest.
  void PredictUplift(const dataset::proto::Example& example,
                     model::proto::Prediction* prediction) const;

  // Computes the indices of the active leaves.
  absl::Status PredictGetLeaves(const dataset::VerticalDataset& dataset,
                                dataset::VerticalDataset::row_t row_idx,
                                absl::Span<int32_t> leaves) const override;

  // Add a new tree to the model.
  void AddTree(std::unique_ptr<decision_tree::DecisionTree> decision_tree);

  // Estimates the memory usage of the model in RAM. The serialized or the
  // compiled version of the model can be much smaller.
  absl::optional<size_t> ModelSizeInBytes() const override;

  // Number of nodes in the model.
  int64_t NumNodes() const;

  // See "IsMissingValueConditionResultFollowGlobalImputation" in
  // "NodeWithChildren".
  bool IsMissingValueConditionResultFollowGlobalImputation() const;

  // Number of trees in the model.
  size_t NumTrees() const { return decision_trees_.size(); }

  int num_trees() const override { return NumTrees(); }

  // Number of times each feature is used in the model. Returns a map, indexed
  // by feature index, and counting the number of time a feature is used.
  void CountFeatureUsage(
      std::unordered_map<int32_t, int64_t>* feature_usage) const;

  const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
  decision_trees() const {
    return decision_trees_;
  }

  std::vector<std::unique_ptr<decision_tree::DecisionTree>>*
  mutable_decision_trees() {
    return &decision_trees_;
  }

  // Call the function "callback" on all the leafs in which the example (defined
  // by a dataset and a row index) is falling.
  void CallOnAllLeafs(
      const dataset::VerticalDataset& dataset,
      dataset::VerticalDataset::row_t row_idx,
      const std::function<void(const decision_tree::proto::Node& node)>&
          callback) const;

  void CallOnAllLeafs(
      const dataset::proto::Example& example,
      const std::function<void(const decision_tree::proto::Node& node)>&
          callback) const;

  void AppendDescriptionAndStatistics(bool full_definition,
                                      std::string* description) const override;

  // Append a human readable semi-graphical description of the model structure.
  void AppendModelStructure(std::string* description) const;

  // Call "call_back" on all the nodes of the model.
  void IterateOnNodes(
      const std::function<void(const decision_tree::NodeWithChildren& node,
                               const int depth)>& call_back) const;

  void IterateOnMutableNodes(
      const std::function<void(decision_tree::NodeWithChildren* node,
                               const int depth)>& call_back) const;

  void set_winner_take_all_inference(bool value) {
    winner_take_all_inference_ = value;
  }

  bool winner_take_all_inference() const { return winner_take_all_inference_; }

  std::vector<proto::OutOfBagTrainingEvaluations>*
  mutable_out_of_bag_evaluations() {
    return &out_of_bag_evaluations_;
  }

  std::vector<std::string> AvailableVariableImportances() const override;

  // List the variable importances that can be computed from the model
  // structure.
  std::vector<std::string> AvailableStructuralVariableImportances() const;

  utils::StatusOr<std::vector<model::proto::VariableImportance>>
  GetVariableImportance(absl::string_view key) const override;

  metric::proto::EvaluationResults ValidationEvaluation() const override;

  // Maximum depth of the model. A depth of "0" means a stump i.e. a tree with a
  // single node. A depth of -1 is only possible for an empty forest.
  int MaximumDepth() const;

  // Minimum number of training observations in a node.
  int MinNumberObs() const;

  // Updates the format used to save the model on disk. If not specified, the
  // recommended format `RecommendedSerializationFormat` is used.
  void set_node_format(const absl::optional<std::string>& format) {
    node_format_ = format;
  }

  // Number of nodes pruned during training.
  absl::optional<int64_t> num_pruned_nodes() const { return num_pruned_nodes_; }

  void set_num_pruned_nodes(int64_t value) { num_pruned_nodes_ = value; }

 private:
  // The decision trees.
  std::vector<std::unique_ptr<decision_tree::DecisionTree>> decision_trees_;

  // Whether the vote of individual trees are distributions or winner-take-all.
  bool winner_take_all_inference_ = true;

  // Evaluation of the model computed during training on the out of bag
  // examples.
  std::vector<proto::OutOfBagTrainingEvaluations> out_of_bag_evaluations_;

  // Variable importance.
  std::vector<model::proto::VariableImportance> mean_decrease_in_accuracy_;
  std::vector<model::proto::VariableImportance> mean_increase_in_rmse_;

  // Format used to stored the node on disk.
  // If not specified, the format `RecommendedSerializationFormat()` will be
  // used. When loading a model from disk, this field is populated with the
  // format.
  absl::optional<std::string> node_format_;

  // Number of nodes trained and then pruned during the training.
  // The classical random forest learning algorithm does not prune nodes.
  absl::optional<int64_t> num_pruned_nodes_;
};

namespace internal {

// Create a single line string containing the result of the evaluation computed
// by "EvaluateOOBPredictions".
std::string EvaluationSnippet(
    const metric::proto::EvaluationResults& evaluation);

void AddClassificationLeafToAccumulator(
    const bool winner_take_all_inference,
    const decision_tree::proto::Node& node,
    utils::IntegerDistribution<float>* accumulator);

void FinalizeClassificationLeafToAccumulator(
    const utils::IntegerDistribution<float>& accumulator,
    model::proto::Prediction* prediction);

// Add a node prediction to a prediction accumulator for regression.
void AddRegressionLeafToAccumulator(const decision_tree::proto::Node& node,
                                    double* accumulator);

// Add a node prediction to a prediction accumulator for uplift.
typedef absl::InlinedVector<float, 2> UplifLeafAccumulator;
void AddUpliftLeafToAccumulator(const decision_tree::proto::Node& node,
                                UplifLeafAccumulator* accumulator);

}  // namespace internal

}  // namespace random_forest
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_RANDOM_FOREST_H_
