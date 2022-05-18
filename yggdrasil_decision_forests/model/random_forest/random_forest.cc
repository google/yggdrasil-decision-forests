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

#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"

#include <stddef.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree_io.h"
#include "yggdrasil_decision_forests/model/decision_tree/structure_analysis.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/usage.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace random_forest {

namespace {
// Basename for the shards containing the nodes.
constexpr char kNodeBaseFilename[] = "nodes";
// Filename containing the random forest header.
constexpr char kHeaderBaseFilename[] = "random_forest_header.pb";

}  // namespace

constexpr char RandomForestModel::kRegisteredName[];
constexpr char RandomForestModel::kVariableImportanceMeanDecreaseInAccuracy[];
constexpr char RandomForestModel::kVariableImportanceMeanIncreaseInRmse[];

absl::Status RandomForestModel::Save(absl::string_view directory,
                                     const ModelIOOptions& io_options) const {
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
  proto::Header header;
  header.set_node_format(format);
  header.set_num_node_shards(num_shards);
  header.set_num_trees(decision_trees_.size());
  header.set_winner_take_all_inference(winner_take_all_inference_);

  *header.mutable_out_of_bag_evaluations() = {out_of_bag_evaluations_.begin(),
                                              out_of_bag_evaluations_.end()};
  *header.mutable_mean_decrease_in_accuracy() = {
      mean_decrease_in_accuracy_.begin(), mean_decrease_in_accuracy_.end()};
  *header.mutable_mean_increase_in_rmse() = {mean_increase_in_rmse_.begin(),
                                             mean_increase_in_rmse_.end()};

  if (num_pruned_nodes_.has_value()) {
    header.set_num_pruned_nodes(num_pruned_nodes_.value());
  }

  const auto header_filename =
      absl::StrCat(io_options.file_prefix.value(), kHeaderBaseFilename);
  RETURN_IF_ERROR(file::SetBinaryProto(
      file::JoinPath(directory, header_filename), header, file::Defaults()));
  return absl::OkStatus();
}

absl::Status RandomForestModel::Load(absl::string_view directory,
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
  winner_take_all_inference_ = header.winner_take_all_inference();
  out_of_bag_evaluations_.assign(header.out_of_bag_evaluations().begin(),
                                 header.out_of_bag_evaluations().end());

  mean_decrease_in_accuracy_.assign(header.mean_decrease_in_accuracy().begin(),
                                    header.mean_decrease_in_accuracy().end());

  mean_increase_in_rmse_.assign(header.mean_increase_in_rmse().begin(),
                                header.mean_increase_in_rmse().end());

  if (header.has_num_pruned_nodes()) {
    num_pruned_nodes_ = header.num_pruned_nodes();
  }

  return absl::OkStatus();
}

absl::Status RandomForestModel::Validate() const {
  RETURN_IF_ERROR(AbstractModel::Validate());

  if (decision_trees_.empty()) {
    return absl::InvalidArgumentError("Empty random forest");
  }

  const auto validate_classification =
      [&](const decision_tree::proto::Node& node) -> absl::Status {
    const int num_classes =
        label_col_spec().categorical().number_of_unique_values();
    if (!node.has_classifier()) {
      return absl::InvalidArgumentError("Classifier missing in RF");
    }
    const auto& classifier = node.classifier();
    if (classifier.top_value() < 0 || classifier.top_value() >= num_classes ||
        classifier.distribution().counts_size() != num_classes) {
      return absl::InvalidArgumentError("Invalid classifier in RF");
    }
    return absl::OkStatus();
  };

  const auto validate_regression =
      [](const decision_tree::proto::Node& node) -> absl::Status {
    if (!node.has_regressor()) {
      return absl::InvalidArgumentError("Regressor missing in RF");
    }
    return absl::OkStatus();
  };

  const auto validate_generic_uplift =
      [&](const decision_tree::proto::Node& node) -> absl::Status {
    if (!node.has_uplift()) {
      return absl::InvalidArgumentError("Uplift missing in RF");
    }

    const auto& treatment_col_spec =
        data_spec().columns(uplift_treatment_col_idx());

    const int num_treatments =
        treatment_col_spec.categorical().number_of_unique_values() - 1;

    if (node.uplift().sum_weights_per_treatment_size() != num_treatments) {
      return absl::InvalidArgumentError("Invalid uplift in RF");
    }
    if (node.uplift().sum_weights_per_treatment_and_outcome_size() !=
        num_treatments) {
      return absl::InvalidArgumentError("Invalid uplift in RF");
    }
    if (node.uplift().treatment_effect_size() != num_treatments - 1) {
      return absl::InvalidArgumentError("Invalid uplift in RF");
    }
    return absl::OkStatus();
  };

  const auto validate_categorical_uplift =
      [&](const decision_tree::proto::Node& node) -> absl::Status {
    RETURN_IF_ERROR(validate_generic_uplift(node));
    const auto& outcome_col_spec = label_col_spec();
    if (outcome_col_spec.type() != dataset::proto::ColumnType::CATEGORICAL) {
      return absl::InvalidArgumentError("The outcome is not categorical.");
    }
    if (outcome_col_spec.categorical().number_of_unique_values() - 1 != 2) {
      return absl::InvalidArgumentError("Only binary outcome is supported.");
    }
    return absl::OkStatus();
  };

  const auto validate_numerical_uplift =
      [&](const decision_tree::proto::Node& node) -> absl::Status {
    RETURN_IF_ERROR(validate_generic_uplift(node));
    const auto& outcome_col_spec = label_col_spec();
    if (outcome_col_spec.type() != dataset::proto::ColumnType::NUMERICAL) {
      return absl::InvalidArgumentError("The outcome is not numerical.");
    }
    return absl::OkStatus();
  };

  switch (task_) {
    case model::proto::Task::CLASSIFICATION:
      for (const auto& tree : decision_trees_) {
        RETURN_IF_ERROR(tree->Validate(data_spec(), validate_classification));
      }
      break;
    case model::proto::Task::REGRESSION:
      for (const auto& tree : decision_trees_) {
        RETURN_IF_ERROR(tree->Validate(data_spec(), validate_regression));
      }
      break;
    case model::proto::Task::CATEGORICAL_UPLIFT:
      for (const auto& tree : decision_trees_) {
        RETURN_IF_ERROR(
            tree->Validate(data_spec(), validate_categorical_uplift));
      }
      break;
    case model::proto::Task::NUMERICAL_UPLIFT:
      for (const auto& tree : decision_trees_) {
        RETURN_IF_ERROR(tree->Validate(data_spec(), validate_numerical_uplift));
      }
      break;
    default:
      return absl::InvalidArgumentError("Non supported task in RF.");
      break;
  }

  return absl::OkStatus();
}

absl::optional<size_t> RandomForestModel::ModelSizeInBytes() const {
  return AbstractAttributesSizeInBytes() +
         decision_tree::EstimateSizeInByte(decision_trees_);
}

int64_t RandomForestModel::NumNodes() const {
  return decision_tree::NumberOfNodes(decision_trees_);
}

bool RandomForestModel::IsMissingValueConditionResultFollowGlobalImputation()
    const {
  return decision_tree::IsMissingValueConditionResultFollowGlobalImputation(
      data_spec(), decision_trees_);
}

// Add a new tree to the model.
void RandomForestModel::AddTree(
    std::unique_ptr<decision_tree::DecisionTree> decision_tree) {
  decision_trees_.push_back(std::move(decision_tree));
}

void RandomForestModel::CountFeatureUsage(
    std::unordered_map<int32_t, int64_t>* feature_usage) const {
  for (const auto& tree : decision_trees_) {
    tree->CountFeatureUsage(feature_usage);
  }
}

absl::Status RandomForestModel::PredictGetLeaves(
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

void RandomForestModel::Predict(const dataset::VerticalDataset& dataset,
                                dataset::VerticalDataset::row_t row_idx,
                                model::proto::Prediction* prediction) const {
  utils::usage::OnInference(1, metadata());
  switch (task_) {
    case model::proto::Task::CLASSIFICATION:
      PredictClassification(dataset, row_idx, prediction);
      break;
    case model::proto::Task::REGRESSION:
      PredictRegression(dataset, row_idx, prediction);
      break;
    case model::proto::Task::CATEGORICAL_UPLIFT:
    case model::proto::Task::NUMERICAL_UPLIFT:
      PredictUplift(dataset, row_idx, prediction);
      break;
    default:
      LOG(FATAL) << "Non supported task.";
      break;
  }
}

void RandomForestModel::Predict(const dataset::proto::Example& example,
                                model::proto::Prediction* prediction) const {
  utils::usage::OnInference(1, metadata());
  switch (task_) {
    case model::proto::Task::CLASSIFICATION:
      PredictClassification(example, prediction);
      break;
    case model::proto::Task::REGRESSION:
      PredictRegression(example, prediction);
      break;
    case model::proto::Task::CATEGORICAL_UPLIFT:
    case model::proto::Task::NUMERICAL_UPLIFT:
      PredictUplift(example, prediction);
      break;
    default:
      LOG(FATAL) << "Non supported task.";
      break;
  }
}

void RandomForestModel::PredictClassification(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t row_idx,
    model::proto::Prediction* prediction) const {
  utils::IntegerDistribution<float> accumulator(
      data_spec_.columns(label_col_idx_)
          .categorical()
          .number_of_unique_values());
  CallOnAllLeafs(dataset, row_idx,
                 [&accumulator, this](const decision_tree::proto::Node& node) {
                   internal::AddClassificationLeafToAccumulator(
                       winner_take_all_inference_, node, &accumulator);
                 });
  internal::FinalizeClassificationLeafToAccumulator(accumulator, prediction);
}

void RandomForestModel::PredictRegression(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t row_idx,
    model::proto::Prediction* prediction) const {
  double accumulator = 0;
  CallOnAllLeafs(dataset, row_idx,
                 [&accumulator](const decision_tree::proto::Node& node) {
                   internal::AddRegressionLeafToAccumulator(node, &accumulator);
                 });
  DCHECK_GT(NumTrees(), 0);
  accumulator /= NumTrees();
  prediction->mutable_regression()->set_value(accumulator);
}

void RandomForestModel::PredictUplift(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t row_idx,
    model::proto::Prediction* prediction) const {
  absl::InlinedVector<float, 2> accumulator(
      data_spec_.columns(uplift_treatment_col_idx_)
              .categorical()
              .number_of_unique_values() -
          2,
      0);
  CallOnAllLeafs(dataset, row_idx,
                 [&accumulator](const decision_tree::proto::Node& node) {
                   internal::AddUpliftLeafToAccumulator(node, &accumulator);
                 });
  const auto num_trees = NumTrees();
  DCHECK_GT(num_trees, 0);
  for (auto& value : accumulator) {
    value /= num_trees;
  }
  *prediction->mutable_uplift()->mutable_treatment_effect() = {
      accumulator.begin(), accumulator.end()};
}

void RandomForestModel::PredictClassification(
    const dataset::proto::Example& example,
    model::proto::Prediction* prediction) const {
  utils::IntegerDistribution<float> accumulator(
      data_spec_.columns(label_col_idx_)
          .categorical()
          .number_of_unique_values());

  CallOnAllLeafs(example,
                 [&accumulator, this](const decision_tree::proto::Node& node) {
                   internal::AddClassificationLeafToAccumulator(
                       winner_take_all_inference_, node, &accumulator);
                 });
  internal::FinalizeClassificationLeafToAccumulator(accumulator, prediction);
}

void RandomForestModel::PredictRegression(
    const dataset::proto::Example& example,
    model::proto::Prediction* prediction) const {
  double accumulator = 0;
  CallOnAllLeafs(example,
                 [&accumulator](const decision_tree::proto::Node& node) {
                   internal::AddRegressionLeafToAccumulator(node, &accumulator);
                 });
  DCHECK_GT(NumTrees(), 0);
  accumulator /= NumTrees();
  prediction->mutable_regression()->set_value(accumulator);
}

void RandomForestModel::PredictUplift(
    const dataset::proto::Example& example,
    model::proto::Prediction* prediction) const {
  internal::UplifLeafAccumulator accumulator(
      data_spec_.columns(uplift_treatment_col_idx_)
              .categorical()
              .number_of_unique_values() -
          2,
      0);
  CallOnAllLeafs(example,
                 [&accumulator](const decision_tree::proto::Node& node) {
                   internal::AddUpliftLeafToAccumulator(node, &accumulator);
                 });
  const auto num_trees = NumTrees();
  DCHECK_GT(num_trees, 0);
  for (auto& value : accumulator) {
    value /= num_trees;
  }
  *prediction->mutable_uplift()->mutable_treatment_effect() = {
      accumulator.begin(), accumulator.end()};
}

void RandomForestModel::CallOnAllLeafs(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t row_idx,
    const std::function<void(const decision_tree::proto::Node& node)>& callback)
    const {
  for (const auto& tree : decision_trees_) {
    callback(tree->GetLeaf(dataset, row_idx));
  }
}

void RandomForestModel::CallOnAllLeafs(
    const dataset::proto::Example& example,
    const std::function<void(const decision_tree::proto::Node& node)>& callback)
    const {
  for (const auto& tree : decision_trees_) {
    callback(tree->GetLeaf(example));
  }
}

void RandomForestModel::AppendDescriptionAndStatistics(
    bool full_definition, std::string* description) const {
  AbstractModel::AppendDescriptionAndStatistics(full_definition, description);
  absl::StrAppend(description, "\n");

  if (task() == model::proto::Task::CLASSIFICATION) {
    absl::SubstituteAndAppend(description, "Winner take all: $0\n",
                              winner_take_all_inference_);
  }

  if (!out_of_bag_evaluations_.empty()) {
    absl::SubstituteAndAppend(description, "Out-of-bag evaluation: $0\n",
                              internal::EvaluationSnippet(
                                  out_of_bag_evaluations_.back().evaluation()));
  } else {
    absl::StrAppend(description, "Out-of-bag evaluation disabled.\n");
  }

  StrAppendForestStructureStatistics(data_spec(), decision_trees(),
                                     description);

  absl::StrAppend(description,
                  "Node format: ", node_format_.value_or("NOT_SET"), "\n");

  if (num_pruned_nodes_.has_value()) {
    absl::StrAppend(description,
                    "Pruned nodes during training: ", num_pruned_nodes_.value(),
                    "\n");
  }

  if (!out_of_bag_evaluations_.empty()) {
    absl::StrAppend(description, "\nTraining OOB:\n");
    for (const auto& oob_eval : out_of_bag_evaluations_) {
      absl::SubstituteAndAppend(
          description, "\ttrees: $0, Out-of-bag evaluation: $1\n",
          oob_eval.number_of_trees(),
          internal::EvaluationSnippet(oob_eval.evaluation()));
    }
  }

  if (full_definition) {
    absl::StrAppend(description, "\nModel Structure:\n");
    AppendModelStructure(description);
  }
}

void RandomForestModel::IterateOnNodes(
    const std::function<void(const decision_tree::NodeWithChildren& node,
                             const int depth)>& call_back) const {
  for (auto& tree : decision_trees_) {
    tree->IterateOnNodes(call_back);
  }
}

void RandomForestModel::IterateOnMutableNodes(
    const std::function<void(decision_tree::NodeWithChildren* node,
                             const int depth)>& call_back) const {
  for (auto& tree : decision_trees_) {
    tree->IterateOnMutableNodes(call_back);
  }
}

void RandomForestModel::AppendModelStructure(std::string* description) const {
  decision_tree::AppendModelStructure(decision_trees_, data_spec(),
                                      label_col_idx_, description);
}

std::vector<std::string> RandomForestModel::AvailableVariableImportances()
    const {
  auto variable_importances = AbstractModel::AvailableVariableImportances();
  switch (task()) {
    case model::proto::Task::CLASSIFICATION:
      if (!mean_decrease_in_accuracy_.empty()) {
        variable_importances.push_back(
            kVariableImportanceMeanDecreaseInAccuracy);
      }
      break;
    case model::proto::Task::REGRESSION:
      if (!mean_increase_in_rmse_.empty()) {
        variable_importances.push_back(kVariableImportanceMeanIncreaseInRmse);
      }
      break;
    case model::proto::Task::CATEGORICAL_UPLIFT:
    case model::proto::Task::NUMERICAL_UPLIFT:
      // TODO(gbm): Add uplift variable importances.
      break;
    default:
      LOG(FATAL) << "RandomForest for task " << model::proto::Task_Name(task())
                 << " does not implement VariableImportances.";
  }
  const auto structual = AvailableStructuralVariableImportances();
  variable_importances.insert(variable_importances.end(), structual.begin(),
                              structual.end());

  // Remove possible duplicates.
  std::sort(variable_importances.begin(), variable_importances.end());
  variable_importances.erase(
      std::unique(variable_importances.begin(), variable_importances.end()),
      variable_importances.end());

  return variable_importances;
}

std::vector<std::string>
RandomForestModel::AvailableStructuralVariableImportances() const {
  std::vector<std::string> variable_importances;
  variable_importances.push_back(
      decision_tree::kVariableImportanceNumberOfNodes);
  variable_importances.push_back(
      decision_tree::kVariableImportanceNumberOfTimesAsRoot);
  variable_importances.push_back(decision_tree::kVariableImportanceSumScore);
  variable_importances.push_back(
      decision_tree::kVariableImportanceMeanMinDepth);
  return variable_importances;
}

utils::StatusOr<std::vector<model::proto::VariableImportance>>
RandomForestModel::GetVariableImportance(absl::string_view key) const {
  const auto general_vi = AbstractModel::GetVariableImportance(key);
  if (general_vi.ok()) {
    return std::move(general_vi.value());
  } else if (general_vi.status().code() == absl::StatusCode::kNotFound) {
    if (key == kVariableImportanceMeanDecreaseInAccuracy &&
        !mean_decrease_in_accuracy_.empty()) {
      return mean_decrease_in_accuracy_;
    } else if (key == kVariableImportanceMeanIncreaseInRmse &&
               !mean_increase_in_rmse_.empty()) {
      return mean_increase_in_rmse_;
    } else if (key == decision_tree::kVariableImportanceNumberOfNodes) {
      return decision_tree::StructureNumberOfTimesInNode(decision_trees());
    } else if (key == decision_tree::kVariableImportanceNumberOfTimesAsRoot) {
      return decision_tree::StructureNumberOfTimesAsRoot(decision_trees());
    } else if (key == decision_tree::kVariableImportanceSumScore) {
      return decision_tree::StructureSumScore(decision_trees());
    } else if (key == decision_tree::kVariableImportanceMeanMinDepth) {
      return decision_tree::StructureMeanMinDepth(decision_trees(),
                                                  data_spec().columns_size());
    }
  }
  return general_vi.status();
}

metric::proto::EvaluationResults RandomForestModel::ValidationEvaluation()
    const {
  if (out_of_bag_evaluations_.empty()) {
    LOG(FATAL) << "Cannot call ValidationEvaluation on a Random Forest model "
                  "without OOB evaluation. The model should be trained with "
                  "compute_oob_performances:true.";
  }
  return out_of_bag_evaluations_.back().evaluation();
}

int RandomForestModel::MaximumDepth() const {
  int max_depth = -1;
  for (const auto& tree : decision_trees_) {
    max_depth = std::max(max_depth, tree->MaximumDepth());
  }
  return max_depth;
}

int RandomForestModel::MinNumberObs() const {
  int min_num_obs = std::numeric_limits<int>::max();
  IterateOnNodes([&min_num_obs](const decision_tree::NodeWithChildren& node,
                                const int depth) {
    if (node.IsLeaf()) {
      const auto candidate =
          node.node().num_pos_training_examples_without_weight();
      if (candidate < min_num_obs) {
        min_num_obs = candidate;
      }
    }
  });
  return min_num_obs;
}

namespace internal {
std::string EvaluationSnippet(
    const metric::proto::EvaluationResults& evaluation) {
  switch (evaluation.task()) {
    case model::proto::Task::CLASSIFICATION:
      return absl::Substitute("accuracy:$0 logloss:$1",
                              metric::Accuracy(evaluation),
                              metric::LogLoss(evaluation));
    case model::proto::Task::REGRESSION:
      return absl::Substitute("rmse:$0", metric::RMSE(evaluation));
    case model::proto::Task::CATEGORICAL_UPLIFT:
      return absl::Substitute("qini:$0 auuc:$1", metric::Qini(evaluation),
                              metric::AUUC(evaluation));
    default:
      LOG(FATAL) << "Not implemented";
  }
}

void AddClassificationLeafToAccumulator(
    const bool winner_take_all_inference,
    const decision_tree::proto::Node& node,
    utils::IntegerDistribution<float>* accumulator) {
  if (winner_take_all_inference) {
    accumulator->Add(node.classifier().top_value());
  } else {
    DCHECK(node.classifier().has_distribution());
    accumulator->AddNormalizedProto(node.classifier().distribution());
  }
}

void FinalizeClassificationLeafToAccumulator(
    const utils::IntegerDistribution<float>& accumulator,
    model::proto::Prediction* prediction) {
  prediction->mutable_classification()->set_value(accumulator.TopClass());
  accumulator.Save(
      prediction->mutable_classification()->mutable_distribution());
}

void AddRegressionLeafToAccumulator(const decision_tree::proto::Node& node,
                                    double* accumulator) {
  *accumulator += node.regressor().top_value();
}

void AddUpliftLeafToAccumulator(const decision_tree::proto::Node& node,
                                UplifLeafAccumulator* accumulator) {
  DCHECK_EQ(accumulator->size(), node.uplift().treatment_effect_size());
  const int n = accumulator->size();
  for (int i = 0; i < n; i++) {
    (*accumulator)[i] += node.uplift().treatment_effect(i);
  }
}

}  // namespace internal

REGISTER_AbstractModel(RandomForestModel, RandomForestModel::kRegisteredName);

}  // namespace random_forest
}  // namespace model
}  // namespace yggdrasil_decision_forests
