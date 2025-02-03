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

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_ISOLATION_FOREST_ISOLATION_FOREST_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_ISOLATION_FOREST_ISOLATION_FOREST_H_

#include <stddef.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/isolation_forest/isolation_forest.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"

namespace yggdrasil_decision_forests::model::isolation_forest {

// Isolation-Forest specific variable importances
static constexpr char kVariableImportanceDIFFI[] = "DIFFI";
static constexpr char kVariableImportanceMeanPartitionScore[] =
    "MEAN_PARTITION_SCORE";

class IsolationForestModel : public AbstractModel,
                             public DecisionForestInterface {
 public:
  inline static constexpr char kRegisteredName[] = "ISOLATION_FOREST";

  IsolationForestModel() : AbstractModel(kRegisteredName) {}

  void Predict(const dataset::VerticalDataset& dataset,
               dataset::VerticalDataset::row_t row_idx,
               model::proto::Prediction* prediction) const override;

  void Predict(const dataset::proto::Example& example,
               model::proto::Prediction* prediction) const override;

  absl::Status PredictGetLeaves(const dataset::VerticalDataset& dataset,
                                dataset::VerticalDataset::row_t row_idx,
                                absl::Span<int32_t> leaves) const override;

  bool CheckStructure(
      const decision_tree::CheckStructureOptions& options) const override;

  void AddTree(std::unique_ptr<decision_tree::DecisionTree> decision_tree);

  std::optional<size_t> ModelSizeInBytes() const override;

  void AppendDescriptionAndStatistics(bool full_definition,
                                      std::string* description) const override;

  absl::Status MakePureServing() override;

  absl::Status Distance(const dataset::VerticalDataset& dataset1,
                        const dataset::VerticalDataset& dataset2,
                        absl::Span<float> distances) const override;

  int num_trees() const override { return decision_trees_.size(); }

  // For the serving engines.
  // TODO: Move in DecisionForestInterface.
  size_t NumTrees() const { return num_trees(); }
  int64_t NumNodes() const {
    return decision_tree::NumberOfNodes(decision_trees_);
  }
  void CountFeatureUsage(
      std::unordered_map<int32_t, int64_t>* feature_usage) const {
    for (const auto& tree : decision_trees_) {
      tree->CountFeatureUsage(feature_usage);
    }
  }

  const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
  decision_trees() const override {
    return decision_trees_;
  }

  std::vector<std::unique_ptr<decision_tree::DecisionTree>>*
  mutable_decision_trees() override {
    return &decision_trees_;
  }

  void set_node_format(const std::optional<std::string>& format) override {
    node_format_ = format;
  }

  void set_num_examples_per_trees(int64_t value) {
    num_examples_per_trees_ = value;
  }

  int64_t num_examples_per_trees() const { return num_examples_per_trees_; }

  std::string DebugCompare(const AbstractModel& other) const override;

  absl::Status Save(absl::string_view directory,
                    const ModelIOOptions& io_options) const override;

  absl::Status Load(absl::string_view directory,
                    const ModelIOOptions& io_options) override;

  absl::Status Validate() const override;

  // List the variable importances that can be computed from the model
  // structure.
  std::vector<std::string> AvailableStructuralVariableImportances() const;

 private:
  void PredictLambda(std::function<const decision_tree::NodeWithChildren&(
                         const decision_tree::DecisionTree&)>
                         get_leaf,
                     model::proto::Prediction* prediction) const;

  // The decision trees.
  std::vector<std::unique_ptr<decision_tree::DecisionTree>> decision_trees_;

  // Node storage format.
  std::optional<std::string> node_format_;

  absl::Status SerializeModelImpl(model::proto::SerializedModel* dst_proto,
                                  std::string* dst_raw) const override;

  absl::Status DeserializeModelImpl(
      const model::proto::SerializedModel& src_proto,
      absl::string_view src_raw) override;

  proto::Header BuildHeaderProto() const;
  void ApplyHeaderProto(const proto::Header& header);

  std::vector<std::string> AvailableVariableImportances() const override;

  absl::StatusOr<std::vector<model::proto::VariableImportance>>
  GetVariableImportance(absl::string_view key) const override;

  // Number of examples used to grow each tree.
  int64_t num_examples_per_trees_ = -1;
};

// Analytical expected number of examples in a binary tree trained with
// "num_examples" examples. Called "c" in "Isolation-Based Anomaly Detection" by
// Liu et al.
float PreissAveragePathLength(UnsignedExampleIdx num_examples);

// Isolation forest prediction.
float IsolationForestPrediction(float average_h,
                                UnsignedExampleIdx num_examples);

// Isolation forest prediction, from the pre-computed denominator.
float IsolationForestPredictionFromDenominator(float average_h,
                                               float denominator);

}  // namespace yggdrasil_decision_forests::model::isolation_forest

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_ISOLATION_FOREST_ISOLATION_FOREST_H_
