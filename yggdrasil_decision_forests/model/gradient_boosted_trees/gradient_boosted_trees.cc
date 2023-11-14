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

#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree_io.h"
#include "yggdrasil_decision_forests/model/decision_tree/structure_analysis.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/plot.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/usage.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

constexpr char GradientBoostedTreesModel::kRegisteredName[];

namespace {
using ::yggdrasil_decision_forests::model::decision_tree::
    StrAppendForestStructureStatistics;
// Basename for the shards containing the nodes.
constexpr char kNodeBaseFilename[] = "nodes";
// Filename containing the gradient boosted trees header.
constexpr char kHeaderBaseFilename[] = "gradient_boosted_trees_header.pb";

}  // namespace

absl::Status GradientBoostedTreesModel::Save(
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
  std::string node_base_filename =
      absl::StrCat(io_options.file_prefix.value(), kNodeBaseFilename);
  RETURN_IF_ERROR(decision_tree::SaveTreesToDisk(
      directory, node_base_filename, decision_trees_, format, &num_shards));
  proto::Header header;
  header.set_node_format(format);
  header.set_num_node_shards(num_shards);
  header.set_num_trees(decision_trees_.size());
  header.set_loss(loss_);
  header.set_num_trees_per_iter(num_trees_per_iter_);
  header.set_validation_loss(validation_loss_);
  header.set_output_logits(output_logits_);
  *header.mutable_initial_predictions() = google::protobuf::RepeatedField<float>(
      initial_predictions_.begin(), initial_predictions_.end());
  *header.mutable_training_logs() = training_logs_;
  std::string header_filename =
      absl::StrCat(io_options.file_prefix.value(), kHeaderBaseFilename);
  RETURN_IF_ERROR(file::SetBinaryProto(
      file::JoinPath(directory, header_filename), header, file::Defaults()));
  return absl::OkStatus();
}

absl::Status GradientBoostedTreesModel::Load(absl::string_view directory,
                                             const ModelIOOptions& io_options) {
  RETURN_IF_ERROR(ValidateModelIOOptions(io_options));

  proto::Header header;
  decision_trees_.clear();
  std::string header_filename =
      absl::StrCat(io_options.file_prefix.value(), kHeaderBaseFilename);
  RETURN_IF_ERROR(file::GetBinaryProto(
      file::JoinPath(directory, header_filename), &header, file::Defaults()));
  std::string node_base_filename =
      absl::StrCat(io_options.file_prefix.value(), kNodeBaseFilename);
  RETURN_IF_ERROR(decision_tree::LoadTreesFromDisk(
      directory, node_base_filename, header.num_node_shards(),
      header.num_trees(), header.node_format(), &decision_trees_));
  node_format_ = header.node_format();
  loss_ = header.loss();
  initial_predictions_.assign(header.initial_predictions().begin(),
                              header.initial_predictions().end());
  num_trees_per_iter_ = header.num_trees_per_iter();
  validation_loss_ = header.validation_loss();
  training_logs_ = header.training_logs();
  output_logits_ = header.output_logits();
  return absl::OkStatus();
}

absl::Status GradientBoostedTreesModel::Validate() const {
  RETURN_IF_ERROR(AbstractModel::Validate());

  const auto validate_leaf =
      [](const decision_tree::proto::Node& node) -> absl::Status {
    if (!node.has_regressor()) {
      return absl::InvalidArgumentError("Regressor missing");
    }
    return absl::OkStatus();
  };

  for (const auto& tree : decision_trees_) {
    RETURN_IF_ERROR(tree->Validate(data_spec(), validate_leaf));
  }

  if ((decision_trees_.size() % num_trees_per_iter_) != 0) {
    return absl::InvalidArgumentError("Invalid number of trees in GBDT");
  }

  int expected_initial_predictions_size = -1;
  switch (task()) {
    case model::proto::Task::CLASSIFICATION:
      if (loss_ == proto::Loss::MULTINOMIAL_LOG_LIKELIHOOD) {
        expected_initial_predictions_size =
            label_col_spec().categorical().number_of_unique_values() - 1;
      } else if (loss_ == proto::Loss::BINOMIAL_LOG_LIKELIHOOD ||
                 loss_ == proto::Loss::BINARY_FOCAL_LOSS) {
        expected_initial_predictions_size = 1;
      } else {
        return absl::InvalidArgumentError("Invalid loss in GBDT");
      }
      break;
    case model::proto::Task::REGRESSION:
      expected_initial_predictions_size = 1;
      break;
    case model::proto::Task::RANKING:
      expected_initial_predictions_size = 1;
      if (ranking_group_col_idx() == -1) {
        return absl::InvalidArgumentError("Invalid ranking_group_col in GBDT");
      }
      break;
    default:
      return absl::InvalidArgumentError("Unknown task in GBDT");
  }
  if (initial_predictions_.size() != expected_initial_predictions_size) {
    return absl::InvalidArgumentError("Invalid initial_predictions in GBDT");
  }
  if (expected_initial_predictions_size != num_trees_per_iter_) {
    return absl::InvalidArgumentError("Invalid num_trees_per_iter_ in GBDT");
  }
  return absl::OkStatus();
}

absl::optional<size_t> GradientBoostedTreesModel::ModelSizeInBytes() const {
  return AbstractAttributesSizeInBytes() +
         decision_tree::EstimateSizeInByte(decision_trees_);
}

int64_t GradientBoostedTreesModel::NumNodes() const {
  return decision_tree::NumberOfNodes(decision_trees_);
}

bool GradientBoostedTreesModel::CheckStructure(
    const decision_tree::CheckStructureOptions& options) const {
  if (options.global_imputation_is_higher &&
      testing_.force_fail_check_structure_global_imputation_is_higher) {
    return false;
  }
  return decision_tree::CheckStructure(options, data_spec(), decision_trees_);
}

// Add a new tree to the model.
void GradientBoostedTreesModel::AddTree(
    std::unique_ptr<decision_tree::DecisionTree> decision_tree) {
  decision_trees_.push_back(std::move(decision_tree));
}

void GradientBoostedTreesModel::CountFeatureUsage(
    std::unordered_map<int32_t, int64_t>* feature_usage) const {
  for (const auto& tree : decision_trees_) {
    tree->CountFeatureUsage(feature_usage);
  }
}

absl::Status GradientBoostedTreesModel::PredictGetLeaves(
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

void GradientBoostedTreesModel::Predict(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t row_idx,
    model::proto::Prediction* prediction) const {
  utils::usage::OnInference(1, metadata());
  switch (loss_) {
    case proto::Loss::BINOMIAL_LOG_LIKELIHOOD:
    case proto::Loss::BINARY_FOCAL_LOSS: {
      double accumulator = initial_predictions_[0];
      CallOnAllLeafs(dataset, row_idx,
                     [&accumulator](const decision_tree::proto::Node& node) {
                       accumulator += node.regressor().top_value();
                     });

      prediction->mutable_classification()->set_value(accumulator > 0.f ? 2
                                                                        : 1);
      auto* dist = prediction->mutable_classification()->mutable_distribution();
      dist->mutable_counts()->Resize(3, 0.f);
      dist->set_sum(1.f);

      float proba_true;
      if (output_logits_) {
        proba_true = accumulator;
      } else {
        proba_true = 1.f / (1.f + std::exp(-accumulator));
      }
      dist->set_counts(1, 1.f - proba_true);
      dist->set_counts(2, proba_true);
    } break;
    case proto::Loss::MULTINOMIAL_LOG_LIKELIHOOD: {
      absl::FixedArray<float> accumulator(num_trees_per_iter_);
      // Zero initial prediction for the MULTINOMIAL_LOG_LIKELIHOOD.
      std::fill(accumulator.begin(), accumulator.end(), 0);

      {
        int accumulator_cell_idx = 0;
        CallOnAllLeafs(dataset, row_idx,
                       [&accumulator, &accumulator_cell_idx,
                        this](const decision_tree::proto::Node& node) {
                         accumulator[accumulator_cell_idx] +=
                             node.regressor().top_value();
                         accumulator_cell_idx++;
                         if (accumulator_cell_idx == num_trees_per_iter_) {
                           accumulator_cell_idx = 0;
                         }
                       });
      }

      auto* dist = prediction->mutable_classification()->mutable_distribution();
      dist->mutable_counts()->Resize(num_trees_per_iter_ + 1, 0.f);

      // Top class.
      if (output_logits_) {
        float sum_logit = 0;
        int highest_cell_idx = 0;
        float highest_cell_value = 0;
        for (int accumulator_idx = 0; accumulator_idx < num_trees_per_iter_;
             accumulator_idx++) {
          auto value = accumulator[accumulator_idx];
          sum_logit += value;
          dist->set_counts(accumulator_idx + 1, value);
          if (value > highest_cell_value) {
            highest_cell_value = value;
            highest_cell_idx = accumulator_idx;
          }
        }
        prediction->mutable_classification()->set_value(highest_cell_idx + 1);
        dist->set_sum(sum_logit);
      } else {
        // Sum logits.
        float sum_exp = 0;
        for (int accumulator_idx = 0; accumulator_idx < num_trees_per_iter_;
             accumulator_idx++) {
          const float exp_val = std::exp(accumulator[accumulator_idx]);
          sum_exp += exp_val;
          // The offset of 1 between the class idx and the accumulator_idx is to
          // skill the special OOD value with index 0.
          dist->set_counts(accumulator_idx + 1, exp_val);
        }
        // Softmax
        int highest_cell_idx = 0;
        float highest_cell_value = 0;
        const float normalization = (sum_exp > 0) ? (1.f / sum_exp) : 0.f;
        for (int accumulator_idx = 0; accumulator_idx < num_trees_per_iter_;
             accumulator_idx++) {
          const float value = dist->counts(accumulator_idx + 1);
          dist->set_counts(accumulator_idx + 1, value * normalization);
          if (value > highest_cell_value) {
            highest_cell_value = value;
            highest_cell_idx = accumulator_idx;
          }
        }
        prediction->mutable_classification()->set_value(highest_cell_idx + 1);
        dist->set_sum(1.f);
      }
    } break;
    case proto::Loss::MEAN_AVERAGE_ERROR:
    case proto::Loss::SQUARED_ERROR: {
      double accumulator = initial_predictions_[0];
      CallOnAllLeafs(dataset, row_idx,
                     [&accumulator](const decision_tree::proto::Node& node) {
                       accumulator += node.regressor().top_value();
                     });
      if (task() == model::proto::RANKING) {
        prediction->mutable_ranking()->set_relevance(accumulator);
      } else if (task() == model::proto::REGRESSION) {
        prediction->mutable_regression()->set_value(accumulator);
      } else {
        YDF_LOG(FATAL) << "Non supported task";
      }
    } break;
    case proto::Loss::POISSON: {
      double accumulator = initial_predictions_[0];
      CallOnAllLeafs(dataset, row_idx,
                     [&accumulator](const decision_tree::proto::Node& node) {
                       accumulator += node.regressor().top_value();
                     });
      if (task() == model::proto::REGRESSION) {
        double clamped_accumulator = utils::clamp(accumulator, -19., 19.);
        prediction->mutable_regression()->set_value(
            std::exp(clamped_accumulator));
      } else {
        YDF_LOG(FATAL) << "Non supported task";
      }
    } break;
    case proto::Loss::LAMBDA_MART_NDCG5:
    case proto::Loss::XE_NDCG_MART: {
      double accumulator = initial_predictions_[0];
      CallOnAllLeafs(dataset, row_idx,
                     [&accumulator](const decision_tree::proto::Node& node) {
                       accumulator += node.regressor().top_value();
                     });
      prediction->mutable_ranking()->set_relevance(accumulator);
    } break;
    default:
      YDF_LOG(FATAL) << "Not implemented";
  }
}

void GradientBoostedTreesModel::Predict(
    const dataset::proto::Example& example,
    model::proto::Prediction* prediction) const {
  utils::usage::OnInference(1, metadata());
  switch (loss_) {
    case proto::Loss::BINOMIAL_LOG_LIKELIHOOD: {
      double accumulator = initial_predictions_[0];
      CallOnAllLeafs(example,
                     [&accumulator](const decision_tree::proto::Node& node) {
                       accumulator += node.regressor().top_value();
                     });
      const float proba_true = 1.f / (1.f + std::exp(-accumulator));
      prediction->mutable_classification()->set_value(proba_true > 0.5f ? 2
                                                                        : 1);
      auto* dist = prediction->mutable_classification()->mutable_distribution();
      dist->mutable_counts()->Resize(3, 0.f);
      dist->set_sum(1.f);
      dist->set_counts(1, 1.f - proba_true);
      dist->set_counts(2, proba_true);
    } break;

    case proto::Loss::MULTINOMIAL_LOG_LIKELIHOOD: {
      absl::FixedArray<float> accumulator(num_trees_per_iter_);
      // Zero initial prediction for the MULTINOMIAL_LOG_LIKELIHOOD.
      std::fill(accumulator.begin(), accumulator.end(), 0);

      {
        int accumulator_cell_idx = 0;
        CallOnAllLeafs(example, [&accumulator, &accumulator_cell_idx,
                                 this](const decision_tree::proto::Node& node) {
          accumulator[accumulator_cell_idx] += node.regressor().top_value();
          accumulator_cell_idx++;
          if (accumulator_cell_idx == num_trees_per_iter_) {
            accumulator_cell_idx = 0;
          }
        });
        CHECK_EQ(accumulator_cell_idx, 0);
      }

      // Note: Why the "+1"? : "prediction" reserves the first value for the out
      // of vocabulary which is not taken into account in "accumulator'.

      auto* dist = prediction->mutable_classification()->mutable_distribution();
      dist->mutable_counts()->Resize(num_trees_per_iter_ + 1, 0.f);

      float sum_exp = 0;
      for (int accumulator_idx = 0; accumulator_idx < num_trees_per_iter_;
           accumulator_idx++) {
        const float exp_val = std::exp(accumulator[accumulator_idx]);
        sum_exp += exp_val;
        dist->set_counts(accumulator_idx + 1, exp_val);
      }

      const float normalization = 1.f / sum_exp;

      float highest_cell_value = 0;
      int highest_cell_idx = 0;

      for (int accumulator_idx = 0; accumulator_idx < num_trees_per_iter_;
           accumulator_idx++) {
        const float value = dist->counts(accumulator_idx + 1);
        if (value > highest_cell_value) {
          highest_cell_value = value;
          highest_cell_idx = accumulator_idx;
        }
        dist->set_counts(accumulator_idx + 1, value * normalization);
      }
      dist->set_sum(1.f);
      prediction->mutable_classification()->set_value(highest_cell_idx + 1);
    } break;

    case proto::Loss::SQUARED_ERROR: {
      double accumulator = initial_predictions_[0];
      CallOnAllLeafs(example,
                     [&accumulator](const decision_tree::proto::Node& node) {
                       accumulator += node.regressor().top_value();
                     });
      prediction->mutable_regression()->set_value(accumulator);
    } break;
    case proto::Loss::LAMBDA_MART_NDCG5:
    case proto::Loss::XE_NDCG_MART: {
      double accumulator = initial_predictions_[0];
      CallOnAllLeafs(example,
                     [&accumulator](const decision_tree::proto::Node& node) {
                       accumulator += node.regressor().top_value();
                     });
      prediction->mutable_ranking()->set_relevance(accumulator);
    } break;
    default:
      YDF_LOG(FATAL) << "Not implemented";
  }
}

void GradientBoostedTreesModel::CallOnAllLeafs(
    const dataset::VerticalDataset& dataset,
    dataset::VerticalDataset::row_t row_idx,
    const std::function<void(const decision_tree::proto::Node& node)>& callback)
    const {
  for (const auto& tree : decision_trees_) {
    callback(tree->GetLeaf(dataset, row_idx));
  }
}

void GradientBoostedTreesModel::CallOnAllLeafs(
    const dataset::proto::Example& example,
    const std::function<void(const decision_tree::proto::Node& node)>& callback)
    const {
  for (const auto& tree : decision_trees_) {
    callback(tree->GetLeaf(example));
  }
}

void GradientBoostedTreesModel::IterateOnNodes(
    const std::function<void(const decision_tree::NodeWithChildren& node,
                             const int depth)>& call_back) const {
  for (auto& tree : decision_trees_) {
    tree->IterateOnNodes(call_back);
  }
}

void GradientBoostedTreesModel::IterateOnMutableNodes(
    const std::function<void(decision_tree::NodeWithChildren* node,
                             const int depth)>& call_back) const {
  for (auto& tree : decision_trees_) {
    tree->IterateOnMutableNodes(call_back);
  }
}

void GradientBoostedTreesModel::AppendModelStructure(
    std::string* description) const {
  decision_tree::AppendModelStructure(decision_trees_, data_spec(),
                                      label_col_idx_, description);
}

metric::proto::EvaluationResults
GradientBoostedTreesModel::ValidationEvaluation() const {
  if (std::isnan(validation_loss_)) {
    YDF_LOG(WARNING)
        << "Validation evaluation not available for the Gradient "
           "Boosted Tree model as no validation dataset was provided "
           "for training (i.e. validation_set_ratio == 0).";
    return {};
  }
  metric::proto::EvaluationResults validation_evaluation;
  validation_evaluation.set_task(task_);
  validation_evaluation.set_loss_value(validation_loss_);
  validation_evaluation.set_loss_name(proto::Loss_Name(loss_));

  for (const auto& log : training_logs_.entries()) {
    // The log entry corresponding to the final model is identified with the
    // number of trees in the final model.
    if (log.number_of_trees() !=
        training_logs_.number_of_trees_in_final_model()) {
      continue;
    }
    // `log` is the training log that corresponds to the final model.
    for (int metrix_idx = 0;
         metrix_idx < training_logs_.secondary_metric_names_size();
         metrix_idx++) {
      const auto& metric_name =
          training_logs_.secondary_metric_names(metrix_idx);
      const auto metric_value = log.validation_secondary_metrics(metrix_idx);
      // Some classical metric names.
      if (metric_name == "accuracy") {
        validation_evaluation.mutable_classification()->set_accuracy(
            metric_value);
      } else if (metric_name == "rmse") {
        validation_evaluation.mutable_regression()->set_sum_square_error(
            metric_value);
        validation_evaluation.set_count_predictions(1.f);
      } else if (metric_name == "NDCG@5") {
        validation_evaluation.mutable_ranking()->mutable_ndcg()->set_value(
            metric_value);
        validation_evaluation.mutable_ranking()->set_ndcg_truncation(5);
      } else {
        YDF_LOG(WARNING) << "Unknown metric name:" << metric_name;
      }
    }
    if (task_ == model::proto::Task::CLASSIFICATION &&
        log.has_validation_confusion_matrix()) {
      *validation_evaluation.mutable_label_column() = label_col_spec();
      *validation_evaluation.mutable_classification()->mutable_confusion() =
          log.validation_confusion_matrix();
    }
  }

  return validation_evaluation;
}

void GradientBoostedTreesModel::AppendDescriptionAndStatistics(
    bool full_definition, std::string* description) const {
  AbstractModel::AppendDescriptionAndStatistics(full_definition, description);
  absl::StrAppend(description, "\n");

  absl::StrAppend(description, "Loss: ", proto::Loss_Name(loss_), "\n");
  if (!std::isnan(validation_loss_)) {
    absl::StrAppend(description, "Validation loss value: ", validation_loss_,
                    "\n");
  }
  absl::StrAppend(description,
                  "Number of trees per iteration: ", num_trees_per_iter_, "\n");

  absl::StrAppend(description,
                  "Node format: ", node_format_.value_or("NOT_SET"), "\n");

  StrAppendForestStructureStatistics(data_spec(), decision_trees(),
                                     description);

  // Training logs.
  if (!training_logs_.entries().empty()) {
    absl::StrAppend(description, "\nTraining logs:\n");
    absl::StrAppend(description, "Number of iteration to final model: ",
                    training_logs_.number_of_trees_in_final_model(), "\n");
    int entry_idx = 0;
    while (entry_idx < training_logs_.entries().size()) {
      const auto& entry = training_logs_.entries(entry_idx);
      absl::StrAppendFormat(description,
                            "\tIter:%d train-loss:%f valid-loss:%f ",
                            entry.number_of_trees(), entry.training_loss(),
                            entry.validation_loss());
      for (int metric_idx = 0;
           metric_idx < training_logs_.secondary_metric_names_size();
           metric_idx++) {
        // Metric name.
        const auto& metric_name =
            training_logs_.secondary_metric_names(metric_idx);

        // Metric values.
        float train_value = std::numeric_limits<float>::quiet_NaN();
        float valid_value = std::numeric_limits<float>::quiet_NaN();
        if (metric_idx < entry.training_secondary_metrics().size()) {
          train_value = entry.training_secondary_metrics(metric_idx);
        }
        if (metric_idx < entry.validation_secondary_metrics().size()) {
          valid_value = entry.validation_secondary_metrics(metric_idx);
        }

        absl::StrAppendFormat(description, " train-%s:%f valid-%s:%f",
                              metric_name, train_value, metric_name,
                              valid_value);
      }
      absl::StrAppend(description, "\n");

      // Print the first 5 entries, and then, print once every 10 entries.
      if (entry_idx < 5) {
        entry_idx++;
      } else {
        entry_idx += 10;
      }
    }
  }

  if (full_definition) {
    absl::StrAppend(description, "\nModel Structure:\n");
    absl::SubstituteAndAppend(description, "Initial predictions: $0\n",
                              absl::StrJoin(initial_predictions_, ","));
    absl::StrAppend(description, "\n");
    AppendModelStructure(description);
  }
}

std::vector<std::string>
GradientBoostedTreesModel::AvailableVariableImportances() const {
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
GradientBoostedTreesModel::AvailableStructuralVariableImportances() const {
  std::vector<std::string> variable_importances;
  variable_importances.push_back(
      decision_tree::kVariableImportanceNumberOfNodes);
  variable_importances.push_back(
      decision_tree::kVariableImportanceNumberOfTimesAsRoot);
  variable_importances.push_back(decision_tree::kVariableImportanceSumScore);
  variable_importances.push_back(
      decision_tree::kVariableImportanceInvMeanMinDepth);
  return variable_importances;
}

absl::StatusOr<std::vector<model::proto::VariableImportance>>
GradientBoostedTreesModel::GetVariableImportance(absl::string_view key) const {
  const auto general_vi = AbstractModel::GetVariableImportance(key);
  if (general_vi.ok()) {
    return std::move(general_vi.value());
  } else if (general_vi.status().code() == absl::StatusCode::kNotFound) {
    // Tree structure variable importances.
    if (key == decision_tree::kVariableImportanceNumberOfNodes) {
      return decision_tree::StructureNumberOfTimesInNode(decision_trees());
    } else if (key == decision_tree::kVariableImportanceNumberOfTimesAsRoot) {
      return decision_tree::StructureNumberOfTimesAsRoot(decision_trees());
    } else if (key == decision_tree::kVariableImportanceSumScore) {
      return decision_tree::StructureSumScore(decision_trees());
    } else if (key == decision_tree::kVariableImportanceInvMeanMinDepth) {
      return decision_tree::StructureMeanMinDepth(decision_trees(),
                                                  data_spec().columns_size());
    }
  }
  return general_vi.status();
}

absl::Status GradientBoostedTreesModel::MakePureServing() {
  training_logs_.Clear();
  for (auto& tree : decision_trees_) {
    tree->IterateOnMutableNodes(
        [](decision_tree::NodeWithChildren* node, const int depth) {
          if (node->IsLeaf()) {
            // Remove the unused information.
            auto* output = node->mutable_node()->mutable_regressor();
            output->clear_sum_gradients();
            output->clear_sum_hessians();
            output->clear_sum_weights();
            output->clear_distribution();
          } else {
            // Remove the label information from the non-leaf nodes.
            node->mutable_node()->clear_output();
          }
        });
  }
  return AbstractModel::MakePureServing();
}

absl::StatusOr<utils::plot::MultiPlot>
GradientBoostedTreesModel::PlotTrainingLogs() const {
  if (training_logs_.entries_size() == 0) {
    return absl::InvalidArgumentError(
        "The Gradient Boosted Trees model does not have training logs. Make "
        "sure the training logs have not been removed with "
        "pure_serving_model=True.");
  }

  utils::plot::MultiPlot multiplot;

  // One plot for the loss, and one plot for each metric.
  ASSIGN_OR_RETURN(
      auto placer,
      utils::plot::PlotPlacer::Create(
          1 + training_logs_.secondary_metric_names_size(), 1, &multiplot));

  // Setup loss plot
  ASSIGN_OR_RETURN(auto* loss_plot, placer.NewPlot());
  loss_plot->x_axis.label = "iteration";
  loss_plot->y_axis.label = "loss";
  auto* training_loss = loss_plot->AddCurve();
  auto* validation_loss = loss_plot->AddCurve();
  training_loss->label = "training";
  validation_loss->label = "validation";

  // Fill loss plot
  for (const auto& entry : training_logs_.entries()) {
    training_loss->xs.push_back(entry.number_of_trees());
    validation_loss->xs.push_back(entry.number_of_trees());
    training_loss->ys.push_back(entry.training_loss());
    validation_loss->ys.push_back(entry.validation_loss());
  }

  // Metric plots
  for (int metric_idx = 0;
       metric_idx < training_logs_.secondary_metric_names().size();
       metric_idx++) {
    ASSIGN_OR_RETURN(auto* metric_plot, placer.NewPlot());
    metric_plot->x_axis.label = "iteration";
    metric_plot->y_axis.label =
        training_logs_.secondary_metric_names(metric_idx);

    auto* training_metric = metric_plot->AddCurve();
    auto* validation_metric = metric_plot->AddCurve();
    training_metric->label = "training";
    validation_metric->label = "validation";

    for (const auto& entry : training_logs_.entries()) {
      // X axis
      training_metric->xs.push_back(entry.number_of_trees());
      validation_metric->xs.push_back(entry.number_of_trees());

      // Y axis
      training_metric->ys.push_back(
          entry.training_secondary_metrics(metric_idx));
      validation_metric->ys.push_back(
          entry.validation_secondary_metrics(metric_idx));
    }
  }

  RETURN_IF_ERROR(placer.Finalize());
  return multiplot;
}

absl::Status GradientBoostedTreesModel::Distance(
    const dataset::VerticalDataset& dataset1,
    const dataset::VerticalDataset& dataset2,
    absl::Span<float> distances) const {
  std::vector<float> tree_weights;
  tree_weights.reserve(decision_trees_.size());
  double sum_values = 0;
  for (const auto& tree : decision_trees_) {
    const double value = internal::WeightedMeanAbsLeafValue(*tree);
    sum_values += value;
    tree_weights.push_back(value);
  }
  // Normalize the tree weights.
  if (sum_values > 0) {
    for (auto& v : tree_weights) {
      v /= sum_values;
    }
  }
  return decision_tree::Distance(decision_trees(), dataset1, dataset2,
                                 distances, tree_weights);
}

namespace internal {

float WeightedMeanAbsLeafValue(const decision_tree::DecisionTree& tree) {
  double sum = 0;
  double total_weight = 0;
  tree.IterateOnNodes(
      [&](const decision_tree::NodeWithChildren& node, const int depth) {
        if (node.IsLeaf()) {
          const auto& regressor = node.node().regressor();
          double leaf_weight;
          if (regressor.has_sum_weights()) {
            leaf_weight = regressor.sum_weights();
          } else if (regressor.distribution().has_count()) {
            leaf_weight = regressor.distribution().count();
          } else {
            leaf_weight = 1;
          }
          sum += std::abs(regressor.top_value()) * leaf_weight;
          total_weight += leaf_weight;
        }
      });

  if (total_weight > 0) {
    return sum / total_weight;
  } else {
    return 0;
  }
}

}  // namespace internal

REGISTER_AbstractModel(GradientBoostedTreesModel,
                       GradientBoostedTreesModel::kRegisteredName);

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
