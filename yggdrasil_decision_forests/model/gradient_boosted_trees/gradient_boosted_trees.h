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

// Implementation of Gradient Boosted Trees.

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_GRADIENT_BOOSTED_TREES_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_GRADIENT_BOOSTED_TREES_H_

#include <stddef.h>

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/utils/plot.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

class GradientBoostedTreesLearner;

// A GBT model i.e. takes as input an example and outputs a prediction.
// See the file header for a description of the GBT learning algorithm/model.
class GradientBoostedTreesModel : public AbstractModel,
                                  public DecisionForestInterface {
 public:
  static constexpr char kRegisteredName[] = "GRADIENT_BOOSTED_TREES";

  GradientBoostedTreesModel() : AbstractModel(kRegisteredName) {}
  absl::Status Save(absl::string_view directory,
                    const ModelIOOptions& io_options) const override;
  absl::Status Load(absl::string_view directory,
                    const ModelIOOptions& io_options) override;

  absl::Status Validate() const override;

  // Computes the indices of the active leaves.
  absl::Status PredictGetLeaves(const dataset::VerticalDataset& dataset,
                                dataset::VerticalDataset::row_t row_idx,
                                absl::Span<int32_t> leaves) const override;

  void Predict(const dataset::VerticalDataset& dataset,
               dataset::VerticalDataset::row_t row_idx,
               model::proto::Prediction* prediction) const override;

  void Predict(const dataset::proto::Example& example,
               model::proto::Prediction* prediction) const override;

  // Number of nodes in the model.
  int64_t NumNodes() const;

  bool CheckStructure(
      const decision_tree::CheckStructureOptions& options) const override;

  // Number of trees in the model. Note that this value can be different from
  // the "num_trees" parameter used during training for two reasons: (1) Early
  // stopping was triggered, (2) the model is trained with multiple trees per
  // iteration.
  size_t NumTrees() const { return decision_trees_.size(); }
  int num_trees() const override { return NumTrees(); }

  // Number of times each feature is used in the model. Returns a map, indexed
  // by feature index, and counting the number of time a feature is used.
  void CountFeatureUsage(
      std::unordered_map<int32_t, int64_t>* feature_usage) const;

  const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
  decision_trees() const override {
    return decision_trees_;
  }

  std::vector<std::unique_ptr<decision_tree::DecisionTree>>*
  mutable_decision_trees() override {
    return &decision_trees_;
  }

  void set_loss(proto::Loss loss, const proto::LossConfiguration& loss_config);
  proto::Loss loss() const { return loss_; }
  proto::LossConfiguration loss_config() const { return loss_config_; }

  int num_trees_per_iter() const { return num_trees_per_iter_; }
  void set_num_trees_per_iter(const int num_trees_per_iter) {
    num_trees_per_iter_ = num_trees_per_iter;
  }

  const proto::TrainingLogs& training_logs() const { return training_logs_; }
  proto::TrainingLogs* mutable_training_logs() { return &training_logs_; }

  // Sets and gets the initial predictions of the model.
  //
  // The initial prediction is the constant part of the model i.e. the
  // prediction of the model if all the trees are removed.
  //
  // For people familiar with Neural Nets, it can be seen as the "bias" weight
  // of a neuron.
  void set_initial_predictions(std::vector<float> initial_predictions) {
    initial_predictions_ = std::move(initial_predictions);
  }
  const std::vector<float>& initial_predictions() const {
    return initial_predictions_;
  }
  std::vector<float>* mutable_initial_predictions() {
    return &initial_predictions_;
  }

  metric::proto::EvaluationResults ValidationEvaluation() const override;

  std::vector<std::string> AvailableVariableImportances() const override;

  // List the variable importances that can be computed from the model
  // structure.
  std::vector<std::string> AvailableStructuralVariableImportances() const;

  absl::StatusOr<std::vector<model::proto::VariableImportance>>
  GetVariableImportance(absl::string_view key) const override;

  float validation_loss() const { return validation_loss_; }
  void set_validation_loss(const float loss) { validation_loss_ = loss; }

  bool output_logits() const { return output_logits_; }
  void set_output_logits(bool value) { output_logits_ = value; }

  // Updates the format used to save the model on disk. If not specified, the
  // recommended format `model::decision_tree::RecommendedSerializationFormat()`
  // is used.
  void set_node_format(const absl::optional<std::string>& format) override {
    node_format_ = format;
  }

  // Adds a new tree to the model.
  void AddTree(std::unique_ptr<decision_tree::DecisionTree> decision_tree);

  absl::Status MakePureServing() override;

  // Fields related to unit testing.
  struct Testing {
    // If true, the "CheckStructure" method will fail if
    // "global_imputation_is_higher"=True.
    bool force_fail_check_structure_global_imputation_is_higher = false;
  };
  Testing* Testing() { return &testing_; }

  absl::StatusOr<utils::plot::MultiPlot> PlotTrainingLogs() const override;

  // The distance between two examples is computed at the ratio of shared
  // leaves, weighted by the number of training examples in the leaf and the
  // average absolute value of the leaves in the tree.
  absl::Status Distance(const dataset::VerticalDataset& dataset1,
                        const dataset::VerticalDataset& dataset2,
                        absl::Span<float> distances) const override;

  std::string DebugCompare(const AbstractModel& other) const override;

 private:
  void PredictClassification(const dataset::VerticalDataset& dataset,
                             dataset::VerticalDataset::row_t row_idx,
                             model::proto::Prediction* prediction) const;

  void PredictRegression(const dataset::VerticalDataset& dataset,
                         dataset::VerticalDataset::row_t row_idx,
                         model::proto::Prediction* prediction) const;

  void PredictClassification(const dataset::proto::Example& example,
                             model::proto::Prediction* prediction) const;

  void PredictRegression(const dataset::proto::Example& example,
                         model::proto::Prediction* prediction) const;

  // Estimates the memory usage of the model in RAM. The serialized or the
  // compiled version of the model can be much smaller.
  absl::optional<size_t> ModelSizeInBytes() const override;

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

  // The decision trees.
  std::vector<std::unique_ptr<decision_tree::DecisionTree>> decision_trees_;
  // Prediction constant of the model.
  std::vector<float> initial_predictions_;

  // Loss evaluated on the validation dataset. Only available is a validation
  // dataset was provided during training.
  float validation_loss_ = std::numeric_limits<float>::quiet_NaN();

  // Number of trees extracted at each gradient boosting operation.
  int num_trees_per_iter_;

  // Evaluation metrics and other meta-data computed during training.
  proto::TrainingLogs training_logs_;
  // If true, call to predict methods return logits (e.g. instead of probability
  // in the case of classification).
  bool output_logits_ = false;

  // Format used to stored the node on disk.
  // If not specified, the format `RecommendedSerializationFormat()` will be
  // used. When loading a model from disk, this field is populated with the
  // format.
  absl::optional<std::string> node_format_;

  struct Testing testing_;

  absl::Status SerializeModelImpl(model::proto::SerializedModel* dst_proto,
                                  std::string* dst_raw) const override;

  absl::Status DeserializeModelImpl(
      const model::proto::SerializedModel& src_proto,
      absl::string_view src_raw) override;

  friend GradientBoostedTreesLearner;

 private:
  proto::Header BuildHeaderProto() const;
  void ApplyHeaderProto(const proto::Header& header);
  // Full name of the loss, including NDCG truncation where applicable.
  std::string GetLossName() const;
  // Loss used to train the model.
  proto::Loss loss_;
  // Options of the loss.
  proto::LossConfiguration loss_config_;
};

namespace internal {

// Average of the absolute value of the leafs weighted by the number of training
// examples.
float WeightedMeanAbsLeafValue(const decision_tree::DecisionTree& tree);

}  // namespace internal

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_GRADIENT_BOOSTED_TREES_H_
