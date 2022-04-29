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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/formats.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/dataset/weight.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/generic_parameters.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_library.h"
#include "yggdrasil_decision_forests/learner/types.h"
#include "yggdrasil_decision_forests/metric/ranking_ndcg.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/utils/adaptive_work.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/csv.h"
#include "yggdrasil_decision_forests/utils/feature_importance.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/snapshot.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/usage.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

using row_t = dataset::VerticalDataset::row_t;

constexpr char GradientBoostedTreesLearner::kRegisteredName[];

// Generic hyper parameter names.
constexpr char GradientBoostedTreesLearner::kHParamNumTrees[];
constexpr char GradientBoostedTreesLearner::kHParamShrinkage[];
constexpr char GradientBoostedTreesLearner::kHParamL1Regularization[];
constexpr char GradientBoostedTreesLearner::kHParamL2Regularization[];
constexpr char
    GradientBoostedTreesLearner::kHParamL2CategoricalRegularization[];
constexpr char GradientBoostedTreesLearner::kHParamLambdaLoss[];
constexpr char GradientBoostedTreesLearner::kHParamDartDropOut[];
constexpr char GradientBoostedTreesLearner::
    kHParamAdaptSubsampleForMaximumTrainingDuration[];
constexpr char GradientBoostedTreesLearner::kHParamUseHessianGain[];
constexpr char GradientBoostedTreesLearner::kHParamSamplingMethod[];
constexpr char GradientBoostedTreesLearner::kSamplingMethodNone[];
constexpr char GradientBoostedTreesLearner::kSamplingMethodRandom[];
constexpr char GradientBoostedTreesLearner::kSamplingMethodGOSS[];
constexpr char GradientBoostedTreesLearner::kSamplingMethodSelGB[];
constexpr char GradientBoostedTreesLearner::kHParamGossAlpha[];
constexpr char GradientBoostedTreesLearner::kHParamGossBeta[];
constexpr char GradientBoostedTreesLearner::kHParamSelGBRatio[];
constexpr char GradientBoostedTreesLearner::kHParamSubsample[];

constexpr char GradientBoostedTreesLearner::kHParamForestExtraction[];
constexpr char GradientBoostedTreesLearner::kHParamForestExtractionMart[];
constexpr char GradientBoostedTreesLearner::kHParamForestExtractionDart[];

constexpr char GradientBoostedTreesLearner::kHParamValidationSetRatio[];
constexpr char GradientBoostedTreesLearner::kHParamEarlyStopping[];
constexpr char GradientBoostedTreesLearner::kHParamEarlyStoppingNone[];
constexpr char
    GradientBoostedTreesLearner::kHParamEarlyStoppingMinLossFullModel[];
constexpr char GradientBoostedTreesLearner::kHParamEarlyStoppingLossIncrease[];
constexpr char
    GradientBoostedTreesLearner::kHParamEarlyStoppingNumTreesLookAhead[];
constexpr char GradientBoostedTreesLearner::kHParamApplyLinkFunction[];
constexpr char
    GradientBoostedTreesLearner::kHParamComputePermutationVariableImportance[];
constexpr char GradientBoostedTreesLearner::kHParamValidationIntervalInTrees[];
constexpr char GradientBoostedTreesLearner::kHParamLoss[];
constexpr char GradientBoostedTreesLearner::kHParamFocalLossGamma[];
constexpr char GradientBoostedTreesLearner::kHParamFocalLossAlpha[];

using dataset::VerticalDataset;
using CategoricalColumn = VerticalDataset::CategoricalColumn;

constexpr double kAdaptativeWarmUpSeconds = 5.0;

namespace {

// During training, the training dataset is duplicated (shallow copy) and
// modified to train each individual tree. This modified dataset is called the
// "gradient dataset" (because the label is the gradient of the loss).
// Base name of the gradient and hessian column in the gradient dataset.
// Note: The hessian column is only created if necessary (e.g. non-constant
// hessian and use_hessian_gain=true).
constexpr char kBaseGradientColumnName[] = "__gradient__";
constexpr char kBaseHessianColumnName[] = "__hessian__";

// Name of the gradient column in the gradient dataset.
std::string GradientColumnName(const int grad_idx) {
  return absl::StrCat(kBaseGradientColumnName, grad_idx);
}

// Name of the hessian column in the gradient dataset.
std::string HessianColumnName(const int grad_idx) {
  return absl::StrCat(kBaseHessianColumnName, grad_idx);
}

// Creates the training configurations to "learn the gradients".
// Note: Gradients are the target of the learning.
void ConfigureTrainingConfigForGradients(
    const model::proto::TrainingConfig& base_config,
    const model::proto::TrainingConfigLinking& base_config_link,
    const model::proto::Task gradient_task,
    const dataset::VerticalDataset& dataset,
    std::vector<GradientData>* gradients) {
  for (auto& gradient : *gradients) {
    gradient.config = base_config;
    gradient.config.set_label(gradient.gradient_column_name);
    gradient.config.set_task(gradient_task);
    gradient.config_link = base_config_link;
    gradient.config_link.set_label(
        dataset.ColumnNameToColumnIdx(gradient.gradient_column_name));
  }
}

// Returns the task used to train the individual decision trees. This task might
// be different from the task that the GBT model is trained to solve.
//
// For example, in case of loss=BINOMIAL_LOG_LIKELIHOOD (which implies a binary
// classification), the trees are "regression trees".
model::proto::Task SubTask(const proto::Loss loss) {
  // GBT trees are always (so far) regression trees.
  return model::proto::REGRESSION;
}

// Set the default value of non-specified hyper-parameters.
absl::Status SetDefaultHyperParameters(model::proto::TrainingConfig* config) {
  auto* gbt_config = config->MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  return internal::SetDefaultHyperParameters(gbt_config);
}

// Splits the training shards between effective training and validation.
absl::Status SplitShards(std::vector<std::string> all,
                         const int num_validation_shards,
                         std::vector<std::string>* training,
                         std::vector<std::string>* validation,
                         utils::RandomEngine* rnd) {
  if (all.size() < num_validation_shards) {
    return absl::InternalError("Not enough shards");
  }

  training->clear();
  validation->clear();

  if (num_validation_shards == 0) {
    // No validation.
    *training = all;
    return absl::OkStatus();
  }

  std::shuffle(all.begin(), all.end(), *rnd);
  validation->insert(validation->end(), all.begin(),
                     all.begin() + num_validation_shards);
  training->insert(training->end(), all.begin() + num_validation_shards,
                   all.end());
  return absl::OkStatus();
}

// Sample a subset of the candidate shards without replacement.
std::vector<std::string> SampleTrainingShards(
    const std::vector<std::string>& candidates, const int num_selected,
    utils::RandomEngine* rnd) {
  // Note: Could use std::sample in C++17.
  std::vector<std::string> selected = candidates;
  std::shuffle(selected.begin(), selected.end(), *rnd);
  selected.resize(num_selected);
  return selected;
}

// Truncate the model (if early stopping is enabled), update the validation loss
// and display the final snippet.
absl::Status FinalizeModelWithValidationDataset(
    const internal::AllTrainingConfiguration& config,
    const internal::EarlyStopping& early_stopping,
    const dataset::VerticalDataset& validation_dataset,
    GradientBoostedTreesModel* mdl) {
  std::vector<float> final_secondary_metrics;
  if (config.gbt_config->early_stopping() ==
          proto::GradientBoostedTreesTrainingConfig::
              MIN_VALIDATION_LOSS_ON_FULL_MODEL ||
      config.gbt_config->early_stopping() ==
          proto::GradientBoostedTreesTrainingConfig::VALIDATION_LOSS_INCREASE) {
    LOG(INFO) << "Truncates the model to " << early_stopping.best_num_trees()
              << " tree(s) i.e. "
              << early_stopping.best_num_trees() / mdl->num_trees_per_iter()
              << "  iteration(s).";
    if (early_stopping.best_num_trees() < 0) {
      return absl::InvalidArgumentError(
          "The model should be evaluated once on the validation dataset.");
    }
    mdl->set_validation_loss(early_stopping.best_loss());
    final_secondary_metrics = early_stopping.best_metrics();
    mdl->mutable_decision_trees()->resize(early_stopping.best_num_trees());
  } else {
    mdl->set_validation_loss(early_stopping.last_loss());
    final_secondary_metrics = early_stopping.last_metrics();
  }

  // Final snippet
  std::string snippet;
  absl::StrAppendFormat(
      &snippet, "Final model num-trees:%d valid-loss:%f",
      early_stopping.best_num_trees() / mdl->num_trees_per_iter(),
      mdl->validation_loss());

  if (!final_secondary_metrics.empty()) {
    for (int secondary_metric_idx = 0;
         secondary_metric_idx <
         mdl->training_logs().secondary_metric_names().size();
         secondary_metric_idx++) {
      absl::StrAppendFormat(
          &snippet, " valid-%s:%f",
          mdl->training_logs().secondary_metric_names(secondary_metric_idx),
          final_secondary_metrics[secondary_metric_idx]);
    }
  }
  LOG(INFO) << snippet;

  if (config.gbt_config->compute_permutation_variable_importance()) {
    LOG(INFO) << "Compute permutation variable importances";
    RETURN_IF_ERROR(
        utils::ComputePermutationFeatureImportance(validation_dataset, mdl));
  }

  return absl::OkStatus();
}

absl::Status MaybeExportTrainingLogs(const absl::string_view log_directory,
                                     GradientBoostedTreesModel* mdl) {
  mdl->mutable_training_logs()->set_number_of_trees_in_final_model(
      mdl->NumTrees() / mdl->num_trees_per_iter());
  if (!log_directory.empty()) {
    RETURN_IF_ERROR(
        internal::ExportTrainingLogs(mdl->training_logs(), log_directory));
  }
  return absl::OkStatus();
}

absl::Status FinalizeModel(const absl::string_view log_directory,
                           GradientBoostedTreesModel* mdl) {
  // Cache the structural variable importance in the model data.
  RETURN_IF_ERROR(mdl->PrecomputeVariableImportances(
      mdl->AvailableStructuralVariableImportances()));

  return MaybeExportTrainingLogs(log_directory, mdl);
}

// Returns a non owning vector of tree pointers from a vector of tree
// unique_ptr.
std::vector<const decision_tree::DecisionTree*> RemoveUniquePtr(
    const std::vector<std::unique_ptr<decision_tree::DecisionTree>>& src) {
  std::vector<const decision_tree::DecisionTree*> dst;
  dst.reserve(src.size());
  for (const auto& tree : src) {
    dst.push_back(tree.get());
  }
  return dst;
}

// Builds the internal (i.e. generally not accessible to user) configuration for
// the weak learner.
decision_tree::InternalTrainConfig BuildWeakLearnerInternalConfig(
    const internal::AllTrainingConfiguration& config, const int num_threads,
    const int grad_idx, const std::vector<GradientData>& gradients,
    const std::vector<float>& predictions, const absl::Time& begin_training) {
  // Timeout in the tree training.
  absl::optional<absl::Time> timeout;
  if (config.train_config.has_maximum_training_duration_seconds()) {
    timeout =
        begin_training +
        absl::Seconds(config.train_config.maximum_training_duration_seconds());
  }

  decision_tree::InternalTrainConfig internal_config;
  internal_config.set_leaf_value_functor = config.loss->SetLeafFunctor(
      predictions, gradients, config.train_config_link.label());
  internal_config.use_hessian_gain = config.gbt_config->use_hessian_gain();
  internal_config.hessian_col_idx = gradients[grad_idx].hessian_col_idx;
  internal_config.hessian_l1 = config.gbt_config->l1_regularization();
  internal_config.hessian_l2_numerical = config.gbt_config->l2_regularization();
  internal_config.hessian_l2_categorical =
      config.gbt_config->l2_regularization_categorical();
  internal_config.num_threads = num_threads;
  internal_config.duplicated_selected_examples = false;
  internal_config.timeout = timeout;
  return internal_config;
}

}  // namespace

GradientBoostedTreesLearner::GradientBoostedTreesLearner(
    const model::proto::TrainingConfig& training_config)
    : AbstractLearner(training_config) {}

absl::Status GradientBoostedTreesLearner::CheckConfiguration(
    const dataset::proto::DataSpecification& data_spec,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::GradientBoostedTreesTrainingConfig& gbt_config,
    const model::proto::DeploymentConfig& deployment) {
  RETURN_IF_ERROR(AbstractLearner::CheckConfiguration(data_spec, config,
                                                      config_link, deployment));

  if ((gbt_config.has_subsample() && gbt_config.subsample() < 1) &&
      gbt_config.sampling_methods_case() !=
          gbt_config.SAMPLING_METHODS_NOT_SET) {
    LOG(WARNING) << "More than one sampling strategy is present.";
  }

  if (gbt_config.has_sample_with_shards()) {
    if (config.task() == model::proto::RANKING) {
      return absl::InvalidArgumentError(
          "Ranking is not supported for per-shard sampling. Unset "
          "sample_with_shards.");
    }
    if (gbt_config.has_dart()) {
      return absl::InvalidArgumentError(
          "Dart is not supported for per-shard sampling. Unset "
          "sample_with_shards.");
    }
    if (gbt_config.adapt_subsample_for_maximum_training_duration()) {
      return absl::InvalidArgumentError(
          "Adaptive sub-sampling is not supported for per-shard sampling. "
          "Unset sample_with_shards.");
    }
  }

  return absl::OkStatus();
}

absl::Status GradientBoostedTreesLearner::BuildAllTrainingConfiguration(
    const dataset::proto::DataSpecification& data_spec,
    internal::AllTrainingConfiguration* all_config) const {
  all_config->train_config = training_config();
  RETURN_IF_ERROR(SetDefaultHyperParameters(&all_config->train_config));

  all_config->gbt_config = &all_config->train_config.GetExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);

  RETURN_IF_ERROR(AbstractLearner::LinkTrainingConfig(
      all_config->train_config, data_spec, &all_config->train_config_link));

  RETURN_IF_ERROR(CheckConfiguration(data_spec, all_config->train_config,
                                     all_config->train_config_link,
                                     *all_config->gbt_config, deployment()));

  auto* mutable_gbt_config = all_config->train_config.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);

  // Select the loss function.
  if (mutable_gbt_config->loss() == proto::Loss::DEFAULT) {
    ASSIGN_OR_RETURN(
        const auto default_loss,
        internal::DefaultLoss(
            all_config->train_config.task(),
            data_spec.columns(all_config->train_config_link.label())));
    mutable_gbt_config->set_loss(default_loss);
    LOG(INFO) << "Default loss set to "
              << proto::Loss_Name(mutable_gbt_config->loss());
  }

  ASSIGN_OR_RETURN(
      all_config->loss,
      CreateLoss(all_config->gbt_config->loss(),
                 all_config->train_config.task(),
                 data_spec.columns(all_config->train_config_link.label()),
                 *all_config->gbt_config));

  if (all_config->loss->RequireGroupingAttribute()) {
    if (!all_config->gbt_config->validation_set_group_feature().empty()) {
      return absl::InvalidArgumentError(
          "\"validation_set_group_feature\" cannot be specified for "
          "a ranking task. Instead, use \"ranking_group\".");
    }
    all_config->effective_validation_set_group =
        all_config->train_config_link.ranking_group();
  } else {
    if (!all_config->gbt_config->validation_set_group_feature().empty()) {
      RETURN_IF_ERROR(dataset::GetSingleColumnIdxFromName(
          all_config->gbt_config->validation_set_group_feature(), data_spec,
          &all_config->effective_validation_set_group));
    }
  }

  return absl::OkStatus();
}

std::unique_ptr<GradientBoostedTreesModel>
GradientBoostedTreesLearner::InitializeModel(
    const internal::AllTrainingConfiguration& config,
    const dataset::proto::DataSpecification& data_spec) const {
  auto mdl = absl::make_unique<GradientBoostedTreesModel>();
  mdl->set_data_spec(data_spec);
  internal::InitializeModelWithTrainingConfig(
      config.train_config, config.train_config_link, mdl.get());
  mdl->set_loss(config.gbt_config->loss());
  const auto secondary_metric_names = config.loss->SecondaryMetricNames();
  *mdl->training_logs_.mutable_secondary_metric_names() = {
      secondary_metric_names.begin(), secondary_metric_names.end()};

  if (mdl->task() == model::proto::Task::CLASSIFICATION &&
      !config.gbt_config->apply_link_function()) {
    // The model output might not be a probability.
    mdl->set_classification_outputs_probabilities(false);
  }
  mdl->set_output_logits(!config.gbt_config->apply_link_function());
  return mdl;
}

utils::StatusOr<std::unique_ptr<AbstractModel>>
GradientBoostedTreesLearner::TrainWithStatus(
    const absl::string_view typed_path,
    const dataset::proto::DataSpecification& data_spec,
    const absl::optional<std::string>& typed_valid_path) const {
  const auto& gbt_config = training_config().GetExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  if (!gbt_config.has_sample_with_shards()) {
    // Regular training.
    return AbstractLearner::TrainWithStatus(typed_path, data_spec,
                                            typed_valid_path);
  }

  return ShardedSamplingTrain(typed_path, data_spec, typed_valid_path);
}

utils::StatusOr<std::unique_ptr<AbstractModel>>
GradientBoostedTreesLearner::ShardedSamplingTrain(
    const absl::string_view typed_path,
    const dataset::proto::DataSpecification& data_spec,
    const absl::optional<std::string>& typed_valid_path) const {
  // The logic of this method is similar to "TrainWithStatus", with the
  // exceptions:
  // - The loss on the training dataset is  computed.
  // - Instead of using a dataset loaded in memory, each tree is trained on a
  //   dataset sampled using shards.
  // - No support for the DART algorithm.
  // - No support for Ranking.

  // TODO(gbm): Splitting method.

  const auto begin_training = absl::Now();

  // Initialize the configuration.
  internal::AllTrainingConfiguration config;
  RETURN_IF_ERROR(BuildAllTrainingConfiguration(data_spec, &config));

  utils::usage::OnTrainingStart(data_spec, config.train_config,
                                config.train_config_link,
                                /*num_examples=*/-1);

  // Initialize the model.
  auto mdl = InitializeModel(config, data_spec);

  utils::RandomEngine random(config.train_config.random_seed());

  // Get the dataset shards.
  std::string dataset_path, dataset_prefix;
  ASSIGN_OR_RETURN(std::tie(dataset_prefix, dataset_path),
                   dataset::SplitTypeAndPath(typed_path));

  std::vector<std::string> all_shards;
  RETURN_IF_ERROR(utils::ExpandInputShards(dataset_path, &all_shards));

  LOG(INFO) << "Training gradient boosted tree on " << all_shards.size()
            << " shard(s) and " << config.train_config_link.features().size()
            << " feature(s).";
  if (all_shards.size() < 10) {
    return absl::InvalidArgumentError(absl::Substitute(
        "The number of shards in $0 is too small $1<10. For best "
        "performances, sampling in the shards should be approximately similar, "
        "for the model training, as sampling the examples e.g. >100.",
        typed_path, all_shards.size()));
  }

  // Split the shards between train and validation.
  std::vector<std::string> training_shards;
  std::vector<std::string> validation_shards;
  bool has_validation_dataset;

  if (typed_valid_path.has_value()) {
    // The user provided a validation dataset.
    training_shards = all_shards;
    std::string valid_dataset_path, valid_dataset_prefix;
    ASSIGN_OR_RETURN(std::tie(valid_dataset_prefix, valid_dataset_path),
                     dataset::SplitTypeAndPath(typed_valid_path.value()));
    RETURN_IF_ERROR(
        utils::ExpandInputShards(valid_dataset_path, &validation_shards));
    has_validation_dataset = !validation_shards.empty();
  } else {
    // Extract a validation dataset from training dataset.
    int num_validation_shards = std::lround(
        all_shards.size() * config.gbt_config->validation_set_ratio());
    has_validation_dataset = num_validation_shards > 0;
    if (config.gbt_config->validation_set_ratio() > 0.f &&
        num_validation_shards == 0) {
      num_validation_shards = 1;
    }
    RETURN_IF_ERROR(SplitShards(all_shards, num_validation_shards,
                                &training_shards, &validation_shards, &random));
  }

  // Load and prepare the validation dataset.
  std::unique_ptr<internal::CompleteTrainingDatasetForWeakLearner> validation;
  if (has_validation_dataset) {
    const auto begin_load_validation = absl::Now();
    LOG(INFO) << "Loading validation dataset from " << validation_shards.size()
              << " shards";
    ASSIGN_OR_RETURN(validation,
                     internal::LoadCompleteDatasetForWeakLearner(
                         validation_shards, dataset_prefix, data_spec, config,
                         /*allocate_gradient=*/false, mdl.get()));
    LOG(INFO) << validation->dataset.nrow()
              << " examples loaded in the validation dataset in "
              << (absl::Now() - begin_load_validation);
  }

  internal::EarlyStopping early_stopping(
      config.gbt_config->early_stopping_num_trees_look_ahead());

  // Load the first sample of training dataset.
  int num_sample_train_shards =
      std::lround(training_shards.size() *
                  config.gbt_config->stochastic_gradient_boosting().ratio());
  if (num_sample_train_shards == 0) {
    num_sample_train_shards = 1;
  }
  std::unique_ptr<internal::CompleteTrainingDatasetForWeakLearner>
      current_train_dataset, next_train_dataset;
  LOG(INFO) << "Loading first training sample dataset from "
            << num_sample_train_shards << " shards";
  const auto begin_load_first_sample = absl::Now();
  ASSIGN_OR_RETURN(current_train_dataset,
                   internal::LoadCompleteDatasetForWeakLearner(
                       SampleTrainingShards(training_shards,
                                            num_sample_train_shards, &random),
                       dataset_prefix, data_spec, config,
                       /*allocate_gradient=*/true, mdl.get()));
  RETURN_IF_ERROR(CheckNumExamples(current_train_dataset->dataset.nrow()));
  LOG(INFO) << current_train_dataset->dataset.nrow()
            << " examples loaded in the first training sample in "
            << (absl::Now() - begin_load_first_sample);

  // Timer accumulators.
  struct {
    // Amount of time spent waiting for the preparation thread (IO + parsing +
    // preprocess).
    absl::Duration sum_duration_wait_prepare;
    utils::concurrency::Mutex mutex_sum_duration;
    // Amount of time in the shard loading logic (IO + parsing).
    absl::Duration sum_duration_load GUARDED_BY(mutex_sum_duration);
    // Amount of time in the shard preprocess logic (running the previously
    // learned trees).
    absl::Duration sum_duration_preprocess GUARDED_BY(mutex_sum_duration);
  } time_accumulators;

  // Fast version of the model. The fast engine is cheaper to run but more
  // expensive to construct.
  std::unique_ptr<serving::FastEngine> last_engine;
  int num_trees_in_last_engine = 0;

  // Load a random sample of training data, prepare it for the weak learner
  // training, and compute the cached predictions with "trees".
  utils::RandomEngine shard_random(random());
  utils::concurrency::Mutex shard_random_mutex;
  auto load_and_prepare_next_sample =
      [&training_shards, num_sample_train_shards, &shard_random,
       &shard_random_mutex, &dataset_prefix, &data_spec, &config, &mdl,
       &time_accumulators, &last_engine, &num_trees_in_last_engine](
          const std::vector<decision_tree::DecisionTree*>& trees)
      -> utils::StatusOr<
          std::unique_ptr<internal::CompleteTrainingDatasetForWeakLearner>> {
    auto time_begin_load = absl::Now();
    std::vector<std::string> selected_shards;
    {
      utils::concurrency::MutexLock lock(&shard_random_mutex);
      selected_shards = SampleTrainingShards(
          training_shards, num_sample_train_shards, &shard_random);
    }

    ASSIGN_OR_RETURN(auto dataset,
                     internal::LoadCompleteDatasetForWeakLearner(
                         selected_shards, dataset_prefix, data_spec, config,
                         /*allocate_gradient=*/true, mdl.get()));

    auto time_begin_predict = absl::Now();
    RETURN_IF_ERROR(internal::ComputePredictions(
        mdl.get(), last_engine.get(), trees, config, dataset->gradient_dataset,
        &dataset->predictions));
    dataset->predictions_from_num_trees =
        num_trees_in_last_engine + trees.size();

    auto time_end_all = absl::Now();
    {
      utils::concurrency::MutexLock results_lock(
          &time_accumulators.mutex_sum_duration);
      time_accumulators.sum_duration_load +=
          time_begin_predict - time_begin_load;
      time_accumulators.sum_duration_preprocess +=
          time_end_all - time_begin_predict;
    }
    return dataset;
  };

  // List of selected examples. Always contains all the training examples.
  std::vector<row_t> selected_examples;

  // Thread loading the sample of shard for the next tree.
  // Note: The shard loaded in multi-threaded by the vertical dataset IO lib.
  std::unique_ptr<utils::concurrency::Thread> thread_load_next_shards;

  // Begin time of the training, excluding the model preparation. Used to
  // compute the IO bottle neck.
  const auto begin_training_loop = absl::Now();

  // Gets the fraction of time spend waiting for the loader thread (and not
  // training).
  const auto get_ratio_waiting_for_loader = [&]() {
    const auto denominator =
        absl::ToDoubleSeconds(absl::Now() - begin_training_loop);
    if (denominator == 0.0) {
      return 0.;
    }
    return absl::ToDoubleSeconds(time_accumulators.sum_duration_wait_prepare) /
           denominator;
  };

  // Amount of time spend in preprocessing in the preparation of the shards.
  const auto get_ratio_prepare_in_shard_preparation = [&]() {
    utils::concurrency::MutexLock results_lock(
        &time_accumulators.mutex_sum_duration);
    const auto denominator =
        absl::ToDoubleSeconds(time_accumulators.sum_duration_preprocess +
                              time_accumulators.sum_duration_load);
    if (denominator == 0.0) {
      return 0.;
    }
    return absl::ToDoubleSeconds(time_accumulators.sum_duration_preprocess) /
           denominator;
  };

  for (int iter_idx = 0; iter_idx < config.gbt_config->num_trees();
       iter_idx++) {
    // If true, the sample in "current_train_dataset" will be re-used (instead
    // of discarded and replaced by "next_train_dataset").
    const bool recycle_current =
        (iter_idx %
         (1 + config.gbt_config->sample_with_shards().num_recycling())) != 0;

    // Same as "recycle_current", but for the next iteration.
    const bool recycle_next =
        ((iter_idx + 1) %
         (1 + config.gbt_config->sample_with_shards().num_recycling())) != 0;

    if (!recycle_current) {
      // Retrieve the set of sharded being loaded.
      if (iter_idx > 0) {
        // Wait for the loading thread.
        const auto begin_wait_loader = absl::Now();
        thread_load_next_shards->Join();
        time_accumulators.sum_duration_wait_prepare +=
            absl::Now() - begin_wait_loader;
        thread_load_next_shards = {};
        if (!next_train_dataset) {
          return absl::InternalError("Missing next sample");
        }

        // Note: At this point, the pre-computed predictions do not take into
        // account the trees added in the last iteration.

        // Add the predictions of the trees learned in the last iteration(s).
        const int num_redo_iters =
            1 + config.gbt_config->sample_with_shards().num_recycling();
        DCHECK_EQ(mdl->NumTrees(),
                  next_train_dataset->predictions_from_num_trees +
                      num_redo_iters * mdl->num_trees_per_iter());
        for (int redo_iter_idx = 0; redo_iter_idx < num_redo_iters;
             redo_iter_idx++) {
          std::vector<const decision_tree::DecisionTree*> last_trees;
          last_trees.reserve(mdl->num_trees_per_iter());
          const auto begin_tree_idx =
              mdl->NumTrees() -
              (num_redo_iters - redo_iter_idx) * mdl->num_trees_per_iter();
          for (int tree_idx_in_iter = 0;
               tree_idx_in_iter < mdl->num_trees_per_iter();
               tree_idx_in_iter++) {
            last_trees.push_back(
                mdl->decision_trees()[begin_tree_idx + tree_idx_in_iter].get());
          }

          // Caches the predictions of the trees.
          RETURN_IF_ERROR(config.loss->UpdatePredictions(
              last_trees, next_train_dataset->gradient_dataset,
              &next_train_dataset->predictions,
              /*mean_abs_prediction=*/nullptr));
          next_train_dataset->predictions_from_num_trees +=
              mdl->num_trees_per_iter();
        }

        current_train_dataset = std::move(next_train_dataset);
      }

      // Start the loading of the next training sample.
      //
      // Note: We don't need to do it for the last tree.
      if (iter_idx < config.gbt_config->num_trees() - 1) {
        // Compile the trees into an engine.
        mdl->set_output_logits(true);
        auto engine_or = mdl->BuildFastEngine();
        mdl->set_output_logits(false);
        if (engine_or.ok()) {
          last_engine = std::move(engine_or.value());
          num_trees_in_last_engine = mdl->NumTrees();
        }

        // Extract the trees of the current model.
        std::vector<decision_tree::DecisionTree*> trees;
        for (int tree_idx = num_trees_in_last_engine;
             tree_idx < mdl->NumTrees(); tree_idx++) {
          trees.push_back(&*mdl->decision_trees()[tree_idx]);
        }

        thread_load_next_shards = absl::make_unique<utils::concurrency::Thread>(
            [&load_and_prepare_next_sample, &next_train_dataset, trees]() {
              next_train_dataset = load_and_prepare_next_sample(trees).value();
            });
      }
    }

    if (current_train_dataset->dataset.nrow() != selected_examples.size()) {
      // Select all the training examples in the sample.
      selected_examples.resize(current_train_dataset->dataset.nrow());
      std::iota(selected_examples.begin(), selected_examples.end(), 0);
    }

    if (iter_idx == 0) {
      // The first dataset is used to compute the initial predictions (a.k.a
      // bias).
      mdl->num_trees_per_iter_ = current_train_dataset->gradients.size();
      early_stopping.set_trees_per_iterations(mdl->num_trees_per_iter_);
      ASSIGN_OR_RETURN(
          const auto initial_predictions,
          config.loss->InitialPredictions(current_train_dataset->dataset,
                                          config.train_config_link.label(),
                                          current_train_dataset->weights));
      mdl->set_initial_predictions(initial_predictions);

      internal::SetInitialPredictions(mdl->initial_predictions(),
                                      current_train_dataset->dataset.nrow(),
                                      &current_train_dataset->predictions);

      if (has_validation_dataset) {
        internal::SetInitialPredictions(mdl->initial_predictions(),
                                        validation->dataset.nrow(),
                                        &validation->predictions);
      }
    }

    // Compute the gradient.
    // Compute the gradient of the residual relative to the examples.
    RETURN_IF_ERROR(config.loss->UpdateGradients(
        current_train_dataset->gradient_dataset,
        config.train_config_link.label(), current_train_dataset->predictions,
        nullptr, &current_train_dataset->gradients, &random));

    // Train a tree on the gradient.
    DCHECK_EQ(current_train_dataset->predictions_from_num_trees,
              mdl->NumTrees());
    std::vector<std::unique_ptr<decision_tree::DecisionTree>> new_trees;
    new_trees.reserve(mdl->num_trees_per_iter());
    for (int grad_idx = 0; grad_idx < mdl->num_trees_per_iter(); grad_idx++) {
      auto tree = absl::make_unique<decision_tree::DecisionTree>();

      const auto internal_config = BuildWeakLearnerInternalConfig(
          config, deployment().num_threads(), grad_idx,
          current_train_dataset->gradients, current_train_dataset->predictions,
          begin_training);

      RETURN_IF_ERROR(decision_tree::Train(
          current_train_dataset->gradient_dataset, selected_examples,
          current_train_dataset->gradients[grad_idx].config,
          current_train_dataset->gradients[grad_idx].config_link,
          config.gbt_config->decision_tree(), deployment(),
          current_train_dataset->weights, &random, tree.get(),
          internal_config));
      new_trees.push_back(std::move(tree));
    }

    if (has_validation_dataset) {
      // Update the predictions on the validation dataset.
      RETURN_IF_ERROR(config.loss->UpdatePredictions(
          RemoveUniquePtr(new_trees), validation->gradient_dataset,
          &validation->predictions,
          /*mean_abs_prediction=*/nullptr));
      validation->predictions_from_num_trees += new_trees.size();
    }

    if (recycle_next) {
      // Update the predictions on the sample because it will be recycled.
      RETURN_IF_ERROR(config.loss->UpdatePredictions(
          RemoveUniquePtr(new_trees), current_train_dataset->gradient_dataset,
          &current_train_dataset->predictions,
          /*mean_abs_prediction=*/nullptr));
      current_train_dataset->predictions_from_num_trees += new_trees.size();
    }

    // Add the tree to the model.
    for (auto& tree : new_trees) {
      mdl->AddTree(std::move(tree));
    }

    // Validation & training logs
    if (((iter_idx + 1) % config.gbt_config->validation_interval_in_trees()) ==
        0) {
      float training_loss;
      std::vector<float> train_secondary_metrics;
      DCHECK_EQ(validation->predictions_from_num_trees, mdl->NumTrees());
      RETURN_IF_ERROR(config.loss->Loss(
          current_train_dataset->gradient_dataset,
          config.train_config_link.label(), current_train_dataset->predictions,
          current_train_dataset->weights, nullptr, &training_loss,
          &train_secondary_metrics));

      auto* log_entry = mdl->training_logs_.mutable_entries()->Add();
      log_entry->set_number_of_trees(iter_idx + 1);
      log_entry->set_training_loss(training_loss);
      *log_entry->mutable_training_secondary_metrics() = {
          train_secondary_metrics.begin(), train_secondary_metrics.end()};

      std::string snippet = absl::StrFormat("\tnum-trees:%d train-loss:%f",
                                            iter_idx + 1, training_loss);

      for (int secondary_metric_idx = 0;
           secondary_metric_idx <
           mdl->training_logs_.secondary_metric_names().size();
           secondary_metric_idx++) {
        absl::StrAppendFormat(
            &snippet, " train-%s:%f",
            mdl->training_logs_.secondary_metric_names(secondary_metric_idx),
            train_secondary_metrics[secondary_metric_idx]);
      }

      if (has_validation_dataset) {
        float validation_loss;
        std::vector<float> validation_secondary_metrics;
        RETURN_IF_ERROR(config.loss->Loss(
            validation->gradient_dataset, config.train_config_link.label(),
            validation->predictions, validation->weights, nullptr,
            &validation_loss, &validation_secondary_metrics));
        log_entry->set_validation_loss(validation_loss);
        *log_entry->mutable_validation_secondary_metrics() = {
            validation_secondary_metrics.begin(),
            validation_secondary_metrics.end()};
        absl::StrAppendFormat(&snippet, " valid-loss:%f", validation_loss);

        for (int secondary_metric_idx = 0;
             secondary_metric_idx <
             mdl->training_logs_.secondary_metric_names().size();
             secondary_metric_idx++) {
          absl::StrAppendFormat(
              &snippet, " valid-%s:%f",
              mdl->training_logs_.secondary_metric_names(secondary_metric_idx),
              validation_secondary_metrics[secondary_metric_idx]);
        }

        // Early stopping.
        RETURN_IF_ERROR(early_stopping.Update(validation_loss,
                                              validation_secondary_metrics,
                                              mdl->decision_trees().size()));

        if (config.gbt_config->early_stopping() ==
                proto::GradientBoostedTreesTrainingConfig::
                    VALIDATION_LOSS_INCREASE &&
            early_stopping.ShouldStop()) {
          break;
        }
      }  // End of validation

      absl::StrAppendFormat(
          &snippet, " loader-blocking:%d%% preprocessing-load:%d%%",
          std::lround(100 * get_ratio_waiting_for_loader()),
          std::lround(100 * get_ratio_prepare_in_shard_preparation()));

      if (iter_idx == 0 || iter_idx == config.gbt_config->num_trees() - 1) {
        LOG(INFO) << snippet;
      } else {
        LOG_INFO_EVERY_N_SEC(30, _ << snippet);
      }

      if (training_config().has_maximum_training_duration_seconds() &&
          (absl::Now() - begin_training) >
              absl::Seconds(
                  training_config().maximum_training_duration_seconds())) {
        LOG(INFO) << "Stop training because of the maximum training duration.";
        break;
      }
    }  // End of training loss.

    // Export intermediate training logs.
    if (config.gbt_config->export_logs_during_training_in_trees() > 0 &&
        (((iter_idx + 1) %
          config.gbt_config->export_logs_during_training_in_trees()) == 0)) {
      RETURN_IF_ERROR(MaybeExportTrainingLogs(log_directory_, mdl.get()));
    }
  }

  // Wait for the loader to stop. This is possible if the training was stopping
  // by early stopping.
  if (thread_load_next_shards) {
    thread_load_next_shards->Join();
    thread_load_next_shards = {};
  }

  if (has_validation_dataset) {
    RETURN_IF_ERROR(FinalizeModelWithValidationDataset(
        config, early_stopping, validation->dataset, mdl.get()));
  }

  RETURN_IF_ERROR(FinalizeModel(log_directory_, mdl.get()));

  utils::usage::OnTrainingEnd(
      data_spec, config.train_config, config.train_config_link,
      /*num_examples=*/-1, *mdl, absl::Now() - begin_training);

  return mdl;
}

utils::StatusOr<std::unique_ptr<AbstractModel>>
GradientBoostedTreesLearner::TrainWithStatus(
    const dataset::VerticalDataset& train_dataset,
    absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
        valid_dataset) const {
  // The training of the model works as follows:
  //
  // 1. Determine the "constant prediction" part of the model
  // 2. Compute the predictions and the loss of the model.
  // 3. Determine the gradient of the loss according to each training example.
  // 4. Train a tree to predict this loss gradient.
  // 5. Add this tree to the model (with a small negative factor e.g. -0.1 i.e.
  // minus the shrinkage).
  // 6. Goto 2.
  //
  // Extra:
  //   - The algorithm extracts a validation dataset to monitor the
  //     performances of the model during training (i.e. at regular interval,
  //     the model is evaluated on the validation dataset).
  //   - Each tree is trained on a random sample of the gradients (instead of
  //     all the gradients).

  const auto begin_training = absl::Now();
  RETURN_IF_ERROR(CheckNumExamples(train_dataset.nrow()));

  // Initialize the configuration.
  internal::AllTrainingConfiguration config;
  RETURN_IF_ERROR(
      BuildAllTrainingConfiguration(train_dataset.data_spec(), &config));

  LOG(INFO) << "Training gradient boosted tree on " << train_dataset.nrow()
            << " example(s) and " << config.train_config_link.features().size()
            << " feature(s).";

  utils::usage::OnTrainingStart(train_dataset.data_spec(), config.train_config,
                                config.train_config_link, train_dataset.nrow());

  if (config.gbt_config->has_sample_with_shards()) {
    return absl::InvalidArgumentError(
        "\"sample_with_shards\" is not compatible with training "
        "dataset specified as VerticalDataset. Instead you should "
        "specify the training dataset as a typed path.");
  }

  // Initialize the model.
  auto mdl = InitializeModel(config, train_dataset.data_spec());

  utils::RandomEngine random(config.train_config.random_seed());

  dataset::VerticalDataset sub_train_dataset;
  dataset::VerticalDataset validation_dataset;
  bool has_validation_dataset;

  if (valid_dataset.has_value()) {
    has_validation_dataset = true;
    sub_train_dataset = train_dataset.ShallowNonOwningClone();
    validation_dataset = valid_dataset.value().get().ShallowNonOwningClone();
  } else {
    // Divide the original training dataset into a validation and a training
    // dataset.
    RETURN_IF_ERROR(internal::ExtractValidationDataset(
        train_dataset, config.gbt_config->validation_set_ratio(),
        config.effective_validation_set_group, &sub_train_dataset,
        &validation_dataset, &random));
    has_validation_dataset = validation_dataset.nrow() > 0;
    if (config.gbt_config->validation_set_ratio() > 0 &&
        !has_validation_dataset) {
      LOG(WARNING)
          << "The validation dataset is empty. Either increase the number "
             "of training examples, or increase the validation set ratio.";
    }
  }

  LOG(INFO) << sub_train_dataset.nrow() << " examples used for training and "
            << validation_dataset.nrow() << " examples used for validation";

  // Compute the example weights.
  std::vector<float> weights, validation_weights;
  RETURN_IF_ERROR(dataset::GetWeights(sub_train_dataset,
                                      config.train_config_link, &weights));
  if (has_validation_dataset) {
    RETURN_IF_ERROR(dataset::GetWeights(
        validation_dataset, config.train_config_link, &validation_weights));
  }

  // Initialize the training dataset for individual trees (called gradient
  // dataset). Note: Individual trees are trained to predict a gradient (instead
  // of the actual dataset labels).
  dataset::VerticalDataset gradient_sub_train_dataset;
  // Gradient specific information (values and training configuration).
  std::vector<GradientData> gradients;
  // Current predictions of the model on the "sub_train_dataset" dataset BEFORE
  // the activation function. Stored example wise i.e. sub_train_predictions[i +
  // j * col_size] is the i-th coordinate the prediction for the j-th example.
  std::vector<float> sub_train_predictions;
  // Initialize the gradient dataset.
  RETURN_IF_ERROR(internal::CreateGradientDataset(
      sub_train_dataset, config.train_config_link.label(),
      config.gbt_config->use_hessian_gain(), *config.loss,
      &gradient_sub_train_dataset, &gradients, &sub_train_predictions));
  // Note: At each iteration, one tree is created for each gradient dimensions.
  mdl->num_trees_per_iter_ = gradients.size();

  dataset::VerticalDataset gradient_validation_dataset;
  std::vector<float> validation_predictions;
  RETURN_IF_ERROR(internal::CreateGradientDataset(
      validation_dataset, config.train_config_link.label(),
      config.gbt_config->use_hessian_gain(), *config.loss,
      &gradient_validation_dataset,
      /*gradients=*/nullptr, &validation_predictions));

  // Training configuration for individual trees i.e. to predict the gradient.
  ConfigureTrainingConfigForGradients(
      config.train_config, config.train_config_link, SubTask(mdl->loss()),
      gradient_sub_train_dataset, &gradients);

  // Controller for the adaptive subsample parameter.
  std::unique_ptr<utils::AdaptativeWork> adaptive_work;
  if (config.gbt_config->adapt_subsample_for_maximum_training_duration()) {
    if (!training_config().has_maximum_training_duration_seconds()) {
      return absl::InvalidArgumentError(
          "\"maximum_training_duration_seconds\" required if "
          "\"subsample_for_maximum_training_duration\" is enabled.");
    }
    adaptive_work = absl::make_unique<utils::AdaptativeWork>(
        config.gbt_config->num_trees(),
        training_config().maximum_training_duration_seconds(),
        kAdaptativeWarmUpSeconds, config.gbt_config->min_adapted_subsample());
  }

  // Compute and set the initial prediction of the model i.e. the "constant
  // prediction" independent of the trees.
  ASSIGN_OR_RETURN(const auto initial_predictions,
                   config.loss->InitialPredictions(
                       gradient_sub_train_dataset,
                       config.train_config_link.label(), weights));
  mdl->set_initial_predictions(initial_predictions);
  internal::SetInitialPredictions(mdl->initial_predictions(),
                                  sub_train_dataset.nrow(),
                                  &sub_train_predictions);
  if (has_validation_dataset) {
    internal::SetInitialPredictions(mdl->initial_predictions(),
                                    validation_dataset.nrow(),
                                    &validation_predictions);
  }

  bool dart_extraction = config.gbt_config->forest_extraction_case() ==
                         proto::GradientBoostedTreesTrainingConfig::kDart;

  // Initialize the Dart accumulator (if Dart is used).
  internal::DartPredictionAccumulator dart_predictions_training;
  internal::DartPredictionAccumulator dart_predictions_validation;
  if (dart_extraction) {
    dart_predictions_training.Initialize(mdl->initial_predictions(),
                                         sub_train_dataset.nrow());
    if (has_validation_dataset) {
      dart_predictions_validation.Initialize(mdl->initial_predictions(),
                                             validation_dataset.nrow());
    }
  }

  std::unique_ptr<RankingGroupsIndices> train_ranking_index;
  std::unique_ptr<RankingGroupsIndices> valid_ranking_index;
  if (mdl->task() == model::proto::Task::RANKING) {
    train_ranking_index = absl::make_unique<RankingGroupsIndices>();
    train_ranking_index->Initialize(sub_train_dataset,
                                    config.train_config_link.label(),
                                    config.train_config_link.ranking_group());
    if (has_validation_dataset) {
      valid_ranking_index = absl::make_unique<RankingGroupsIndices>();
      valid_ranking_index->Initialize(validation_dataset,
                                      config.train_config_link.label(),
                                      config.train_config_link.ranking_group());
    }
  }

  proto::TrainingLogs& training_logs = mdl->training_logs_;

  internal::EarlyStopping early_stopping(
      config.gbt_config->early_stopping_num_trees_look_ahead());
  early_stopping.set_trees_per_iterations(mdl->num_trees_per_iter_);

  if (config.gbt_config->use_hessian_gain() &&
      gradients.front().hessian_col_idx == -1) {
    return absl::InvalidArgumentError(
        "Loss does not support hessian optimization");
  }

  ASSIGN_OR_RETURN(
      const auto preprocessing,
      decision_tree::PreprocessTrainingDataset(
          gradient_sub_train_dataset, config.train_config,
          config.train_config_link, config.gbt_config->decision_tree(),
          deployment_.num_threads()));

  // Time of the next snapshot if training resume is enabled.
  auto next_snapshot =
      absl::Now() +
      absl::Seconds(deployment_.resume_training_snapshot_interval_seconds());

  // Path to the root snapshot directory used to resume interrupted training.
  // Empty if resuming training is disabled.
  std::string snapshot_directory;

  // Try to resume training.
  int iter_idx = 0;
  if (deployment_.try_resume_training()) {
    if (deployment_.cache_path().empty()) {
      return absl::InvalidArgumentError(
          "\"try_resume_training=True\" requires a \"cache_path\" in the "
          "deployment configuration.");
    }
    snapshot_directory = file::JoinPath(deployment_.cache_path(), "snapshot");

    const auto snapshot_idx_or = utils::GetGreatestSnapshot(snapshot_directory);
    if (snapshot_idx_or.ok()) {
      // Load the snapshot.
      LOG(INFO) << "Resume the GBT training from tree #"
                << snapshot_idx_or.value();
      const auto model_path =
          file::JoinPath(deployment_.cache_path(),
                         absl::StrCat("model_", snapshot_idx_or.value()));
      // Load the model structure.
      RETURN_IF_ERROR(
          mdl->Load(model_path, /*io_options=*/{/*file_prefix=*/""}));
      iter_idx = mdl->NumTrees();

      // Recompute the prediction caches.
      auto time_begin_recompute_accumulators = absl::Now();
      mdl->set_output_logits(true);
      ASSIGN_OR_RETURN(auto engine, mdl->BuildFastEngine());
      mdl->set_output_logits(false);

      RETURN_IF_ERROR(internal::ComputePredictions(
          mdl.get(), engine.get(), {}, config, gradient_sub_train_dataset,
          &sub_train_predictions));

      if (has_validation_dataset) {
        RETURN_IF_ERROR(internal::ComputePredictions(
            mdl.get(), engine.get(), {}, config, gradient_validation_dataset,
            &validation_predictions));
      }
      LOG(INFO) << "Re-compute the prediction accumulators in "
                << absl::FormatDuration(absl::Now() -
                                        time_begin_recompute_accumulators);
    }
  }

  // Train the trees one by one.
  std::vector<row_t> selected_examples;

  // Switch between weights and GOSS-specific weights if necessary.
  std::vector<float>* tree_weights = &weights;
  std::vector<float> goss_weights;
  if (config.gbt_config->has_gradient_one_side_sampling()) {
    goss_weights = weights;
    tree_weights = &goss_weights;
  }
  for (; iter_idx < config.gbt_config->num_trees(); iter_idx++) {
    // The user interrupted the training.
    if (stop_training_trigger_ != nullptr && *stop_training_trigger_) {
      LOG(INFO) << "Training interrupted per request.";
      break;
    }

    const auto begin_iter_training = absl::Now();
    std::vector<int> dropout_trees_idxs;
    if (dart_extraction) {
      dropout_trees_idxs = dart_predictions_training.SampleIterIndices(
          config.gbt_config->dart().dropout_rate(), &random);
      RETURN_IF_ERROR(dart_predictions_training.GetSampledPredictions(
          dropout_trees_idxs, &sub_train_predictions));
    }

    // Compute the gradient of the residual relative to the examples.
    RETURN_IF_ERROR(config.loss->UpdateGradients(
        gradient_sub_train_dataset, config.train_config_link.label(),
        sub_train_predictions, train_ranking_index.get(), &gradients, &random));

    float subsample_factor = 1.f;
    // Select a random set of examples (without replacement).
    if (adaptive_work) {
      subsample_factor = adaptive_work->OptimalApproximationFactor();
    }

    switch (config.gbt_config->sampling_methods_case()) {
      case proto::GradientBoostedTreesTrainingConfig::kGradientOneSideSampling:
        // Reset train weights.
        std::copy(weights.begin(), weights.end(), goss_weights.begin());

        // Sample examples with GOSS and adjust train weights accordingly.
        internal::SampleTrainingExamplesWithGoss(
            gradients, gradient_sub_train_dataset.nrow(),
            subsample_factor *
                config.gbt_config->gradient_one_side_sampling().alpha(),
            subsample_factor *
                config.gbt_config->gradient_one_side_sampling().beta(),
            &random, &selected_examples, &goss_weights);
        break;
      case proto::GradientBoostedTreesTrainingConfig::
          kSelectiveGradientBoosting:
        RETURN_IF_ERROR(internal::SampleTrainingExamplesWithSelGB(
            mdl->task(), gradient_sub_train_dataset.nrow(),
            train_ranking_index.get(), sub_train_predictions,
            config.gbt_config->selective_gradient_boosting().ratio(),
            &selected_examples));
        break;
      case proto::GradientBoostedTreesTrainingConfig::
          kStochasticGradientBoosting:
      case proto::GradientBoostedTreesTrainingConfig::SAMPLING_METHODS_NOT_SET:
      default:
        internal::SampleTrainingExamples(
            gradient_sub_train_dataset.nrow(),
            config.gbt_config->stochastic_gradient_boosting().ratio() *
                subsample_factor,
            &random, &selected_examples);
        break;
    }

    // Train a tree on the gradient.
    std::vector<std::unique_ptr<decision_tree::DecisionTree>> new_trees;
    new_trees.reserve(gradients.size());
    for (int grad_idx = 0; grad_idx < gradients.size(); grad_idx++) {
      auto tree = absl::make_unique<decision_tree::DecisionTree>();

      auto internal_config = BuildWeakLearnerInternalConfig(
          config, deployment().num_threads(), grad_idx, gradients,
          sub_train_predictions, begin_training);
      internal_config.preprocessing = &preprocessing;

      RETURN_IF_ERROR(decision_tree::Train(
          gradient_sub_train_dataset, selected_examples,
          gradients[grad_idx].config, gradients[grad_idx].config_link,
          config.gbt_config->decision_tree(), deployment(), *tree_weights,
          &random, tree.get(), internal_config));
      new_trees.push_back(std::move(tree));
    }

    // Note: Since the batch size is only impacting the training time (i.e.
    // not the update prediction time), and since the adaptive work manager
    // assumes a linear relation between work and time, we only measure the
    // duration of the training step.
    if (adaptive_work) {
      adaptive_work->ReportTaskDone(
          subsample_factor,
          absl::ToDoubleSeconds(absl::Now() - begin_iter_training));
    }

    double mean_abs_prediction = 0;
    if (dart_extraction) {
      // Update the Dart cache and the predictions on the training dataset.
      RETURN_IF_ERROR(dart_predictions_training.UpdateWithNewIteration(
          dropout_trees_idxs, mdl->loss(), *config.loss, new_trees,
          gradient_sub_train_dataset, gradients.size(), &mean_abs_prediction));
      RETURN_IF_ERROR(
          dart_predictions_training.GetAllPredictions(&sub_train_predictions));

      if (has_validation_dataset) {
        // Update the dart cache and the predictions on the validation dataset.
        RETURN_IF_ERROR(dart_predictions_validation.UpdateWithNewIteration(
            dropout_trees_idxs, mdl->loss(), *config.loss, new_trees,
            gradient_validation_dataset, gradients.size()));
        RETURN_IF_ERROR(dart_predictions_validation.GetAllPredictions(
            &validation_predictions));
      }
    } else {
      // Update the predictions on the training dataset.
      RETURN_IF_ERROR(config.loss->UpdatePredictions(
          RemoveUniquePtr(new_trees), gradient_sub_train_dataset,
          &sub_train_predictions, &mean_abs_prediction));

      if (has_validation_dataset) {
        // Update the predictions on the validation dataset.
        RETURN_IF_ERROR(config.loss->UpdatePredictions(
            RemoveUniquePtr(new_trees), gradient_validation_dataset,
            &validation_predictions,
            /*mean_abs_prediction=*/nullptr));
      }
    }

    // Add the tree to the model.
    for (auto& tree : new_trees) {
      mdl->AddTree(std::move(tree));
    }

    if (((iter_idx + 1) % config.gbt_config->validation_interval_in_trees()) ==
        0) {
      float training_loss;
      std::vector<float> train_secondary_metrics;
      RETURN_IF_ERROR(config.loss->Loss(
          gradient_sub_train_dataset, config.train_config_link.label(),
          sub_train_predictions, weights, train_ranking_index.get(),
          &training_loss, &train_secondary_metrics));

      auto* log_entry = training_logs.mutable_entries()->Add();
      log_entry->set_number_of_trees(iter_idx + 1);
      log_entry->set_training_loss(training_loss);
      log_entry->set_subsample_factor(subsample_factor);
      *log_entry->mutable_training_secondary_metrics() = {
          train_secondary_metrics.begin(), train_secondary_metrics.end()};
      log_entry->set_mean_abs_prediction(mean_abs_prediction);

      std::string snippet = absl::StrFormat("\tnum-trees:%d train-loss:%f",
                                            iter_idx + 1, training_loss);
      if (subsample_factor < 1.f) {
        absl::StrAppendFormat(&snippet, " subsample_factor:%f",
                              subsample_factor);
      }

      for (int secondary_metric_idx = 0;
           secondary_metric_idx < training_logs.secondary_metric_names().size();
           secondary_metric_idx++) {
        absl::StrAppendFormat(
            &snippet, " train-%s:%f",
            training_logs.secondary_metric_names(secondary_metric_idx),
            train_secondary_metrics[secondary_metric_idx]);
      }

      if (has_validation_dataset) {
        float validation_loss;
        std::vector<float> validation_secondary_metrics;
        RETURN_IF_ERROR(config.loss->Loss(
            gradient_validation_dataset, config.train_config_link.label(),
            validation_predictions, validation_weights,
            valid_ranking_index.get(), &validation_loss,
            &validation_secondary_metrics));
        log_entry->set_validation_loss(validation_loss);
        *log_entry->mutable_validation_secondary_metrics() = {
            validation_secondary_metrics.begin(),
            validation_secondary_metrics.end()};
        absl::StrAppendFormat(&snippet, " valid-loss:%f", validation_loss);

        for (int secondary_metric_idx = 0;
             secondary_metric_idx <
             training_logs.secondary_metric_names().size();
             secondary_metric_idx++) {
          absl::StrAppendFormat(
              &snippet, " valid-%s:%f",
              training_logs.secondary_metric_names(secondary_metric_idx),
              validation_secondary_metrics[secondary_metric_idx]);
        }

        // Early stopping.
        RETURN_IF_ERROR(early_stopping.Update(validation_loss,
                                              validation_secondary_metrics,
                                              mdl->decision_trees().size()));

        if (config.gbt_config->early_stopping() ==
                proto::GradientBoostedTreesTrainingConfig::
                    VALIDATION_LOSS_INCREASE &&
            early_stopping.ShouldStop()) {
          break;
        }
      }  // End of validation loss.

      if (iter_idx == 0 || iter_idx == config.gbt_config->num_trees() - 1) {
        LOG(INFO) << snippet;
      } else {
        LOG_INFO_EVERY_N_SEC(30, _ << snippet);
      }

      if (training_config().has_maximum_training_duration_seconds() &&
          (absl::Now() - begin_training) >
              absl::Seconds(
                  training_config().maximum_training_duration_seconds())) {
        LOG(INFO) << "Stop training because of the maximum training duration.";
        break;
      }
    }  // End of training loss.

    // Export intermediate training logs.
    if (config.gbt_config->export_logs_during_training_in_trees() > 0 &&
        (((iter_idx + 1) %
          config.gbt_config->export_logs_during_training_in_trees()) == 0)) {
      RETURN_IF_ERROR(MaybeExportTrainingLogs(log_directory_, mdl.get()));
    }

    // Export a snapshot
    if (deployment_.try_resume_training() && next_snapshot < absl::Now()) {
      LOG(INFO) << "Create a snapshot of the model at iteration " << iter_idx;
      const auto model_path = file::JoinPath(deployment_.cache_path(),
                                             absl::StrCat("model_", iter_idx));

      // Save the model structure.
      RETURN_IF_ERROR(
          mdl->Save(model_path, /*io_options=*/{/*file_prefix=*/""}));

      // Record the snapshot.
      RETURN_IF_ERROR(utils::AddSnapshot(snapshot_directory, iter_idx));

      next_snapshot =
          absl::Now() +
          absl::Seconds(
              deployment_.resume_training_snapshot_interval_seconds());
    }
  }  // End of training iteration.

  // Create a final snapshot
  if (deployment_.try_resume_training()) {
    const auto last_snapshot = utils::GetGreatestSnapshot(snapshot_directory);
    if (!last_snapshot.ok() || last_snapshot.value() < iter_idx) {
      LOG(INFO) << "Create final snapshot of the model at iteration "
                << iter_idx;
      const auto model_path = file::JoinPath(deployment_.cache_path(),
                                             absl::StrCat("model_", iter_idx));

      // Save the model structure.
      RETURN_IF_ERROR(
          mdl->Save(model_path, /*io_options=*/{/*file_prefix=*/""}));

      // Record the snapshot.
      RETURN_IF_ERROR(utils::AddSnapshot(snapshot_directory, iter_idx));
    }
  }

  if (has_validation_dataset) {
    RETURN_IF_ERROR(FinalizeModelWithValidationDataset(
        config, early_stopping, validation_dataset, mdl.get()));
  }

  if (dart_extraction) {
    // Scale the trees output values.
    const auto per_tree_weights = dart_predictions_training.TreeOutputScaling();
    const int num_iters = mdl->NumTrees() / mdl->num_trees_per_iter();
    if (per_tree_weights.size() < num_iters) {
      return absl::InternalError("Wrong number of trees");
    }
    for (int sub_iter_idx = 0; sub_iter_idx < num_iters; sub_iter_idx++) {
      for (int sub_tree_idx = 0; sub_tree_idx < mdl->num_trees_per_iter();
           sub_tree_idx++) {
        (*mdl->mutable_decision_trees())[sub_iter_idx *
                                             mdl->num_trees_per_iter() +
                                         sub_tree_idx]
            ->ScaleRegressorOutput(per_tree_weights[sub_iter_idx]);
      }
    }
  }

  RETURN_IF_ERROR(FinalizeModel(log_directory_, mdl.get()));

  utils::usage::OnTrainingEnd(train_dataset.data_spec(), training_config(),
                              config.train_config_link, train_dataset.nrow(),
                              *mdl, absl::Now() - begin_training);

  decision_tree::SetLeafIndices(mdl->mutable_decision_trees());
  return std::move(mdl);
}

absl::Status GradientBoostedTreesLearner::SetHyperParametersImpl(
    utils::GenericHyperParameterConsumer* generic_hyper_params) {
  RETURN_IF_ERROR(
      AbstractLearner::SetHyperParametersImpl(generic_hyper_params));
  auto* gbt_config = training_config_.MutableExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);
  // Decision tree specific hyper-parameters.
  absl::flat_hash_set<std::string> consumed_hparams;
  RETURN_IF_ERROR(decision_tree::SetHyperParameters(
      &consumed_hparams, gbt_config->mutable_decision_tree(),
      generic_hyper_params));

  {
    const auto hparam = generic_hyper_params->Get(kHParamLoss);
    if (hparam.has_value()) {
      const auto& str_loss = hparam.value().value().categorical();
      model::gradient_boosted_trees::proto::Loss loss;
      if (!model::gradient_boosted_trees::proto::Loss_Parse(str_loss, &loss)) {
        return absl::InvalidArgumentError(
            absl::Substitute("The loss value \"$0\" is unknown.", str_loss));
      }
      gbt_config->set_loss(loss);
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamNumTrees);
    if (hparam.has_value()) {
      gbt_config->set_num_trees(hparam.value().value().integer());
    }
  }
  {
    const auto hparam = generic_hyper_params->Get(kHParamShrinkage);
    if (hparam.has_value()) {
      gbt_config->set_shrinkage(hparam.value().value().real());
    }
  }
  {
    const auto hparam = generic_hyper_params->Get(kHParamL1Regularization);
    if (hparam.has_value()) {
      gbt_config->set_l1_regularization(hparam.value().value().real());
    }
  }
  {
    const auto hparam = generic_hyper_params->Get(kHParamL2Regularization);
    if (hparam.has_value()) {
      gbt_config->set_l2_regularization(hparam.value().value().real());
    }
  }
  {
    const auto hparam =
        generic_hyper_params->Get(kHParamL2CategoricalRegularization);
    if (hparam.has_value()) {
      gbt_config->set_l2_regularization_categorical(
          hparam.value().value().real());
    }
  }
  {
    const auto hparam = generic_hyper_params->Get(kHParamLambdaLoss);
    if (hparam.has_value()) {
      gbt_config->set_lambda_loss(hparam.value().value().real());
    }
  }
  {
    const auto hparam = generic_hyper_params->Get(kHParamForestExtraction);
    if (hparam.has_value()) {
      if (hparam.value().value().categorical() == kHParamForestExtractionMart) {
        gbt_config->mutable_mart();
      } else if (hparam.value().value().categorical() ==
                 kHParamForestExtractionDart) {
        gbt_config->mutable_dart();
      } else {
        LOG(WARNING) << "Unknown value " << hparam.value().value().categorical()
                     << " for " << kHParamForestExtraction << ".";
      }
    }
  }
  {
    const auto hparam = generic_hyper_params->Get(kHParamDartDropOut);
    if (hparam.has_value() && gbt_config->has_dart()) {
      gbt_config->mutable_dart()->set_dropout_rate(
          hparam.value().value().real());
    }
  }
  {
    const auto hparam = generic_hyper_params->Get(
        kHParamAdaptSubsampleForMaximumTrainingDuration);
    if (hparam.has_value()) {
      gbt_config->set_adapt_subsample_for_maximum_training_duration(
          hparam.value().value().categorical() == "true");
    }
  }
  {
    const auto hparam = generic_hyper_params->Get(kHParamUseHessianGain);
    if (hparam.has_value()) {
      gbt_config->set_use_hessian_gain(hparam.value().value().categorical() ==
                                       "true");
    }
  }

  // Determine the sampling strategy.
  const auto sampling_method_hparam =
      generic_hyper_params->Get(kHParamSamplingMethod);
  if (sampling_method_hparam.has_value()) {
    // The preferred way of choosing a sampling method and setting its params.
    const auto sampling_method =
        sampling_method_hparam.value().value().categorical();

    if (sampling_method == kSamplingMethodRandom) {
      gbt_config->mutable_stochastic_gradient_boosting();
    } else if (sampling_method == kSamplingMethodGOSS) {
      gbt_config->mutable_gradient_one_side_sampling();
    } else if (sampling_method == kSamplingMethodSelGB) {
      gbt_config->mutable_selective_gradient_boosting();
    }
  } else {

    if (gbt_config->sampling_methods_case() ==
        proto::GradientBoostedTreesTrainingConfig::SAMPLING_METHODS_NOT_SET) {
      // If no other sampling method is selected, we activate the "subsample"
      // field for backwards compatibility.
      gbt_config->mutable_stochastic_gradient_boosting();
    } else {
      if (gbt_config->has_subsample()) {
        LOG(WARNING)
            << "Too many sampling methods are set in the config. "
               "Only one may be configured at any given time. Ignoring "
               "'subsample'.";
        gbt_config->clear_subsample();
      }
    }
  }

  // Set sampling strategy's hyperparameters.
  {
    const auto subsample = generic_hyper_params->Get(kHParamSubsample);
    if (subsample.has_value()) {
      if (gbt_config->has_stochastic_gradient_boosting()) {
        gbt_config->mutable_stochastic_gradient_boosting()->set_ratio(
            subsample.value().value().real());
      } else {
        LOG(WARNING) << "Subsample hyperparameter given but sampling method "
                        "does not match.";
      }
    }
  }

  {
    const auto alpha = generic_hyper_params->Get(kHParamGossAlpha);
    if (alpha.has_value()) {
      if (gbt_config->has_gradient_one_side_sampling()) {
        gbt_config->mutable_gradient_one_side_sampling()->set_alpha(
            alpha.value().value().real());
      } else {
        LOG(WARNING) << "GOSS alpha hyperparameter given but GOSS is disabled.";
      }
    }
    const auto beta = generic_hyper_params->Get(kHParamGossBeta);
    if (beta.has_value()) {
      if (gbt_config->has_gradient_one_side_sampling()) {
        gbt_config->mutable_gradient_one_side_sampling()->set_beta(
            beta.value().value().real());
      } else {
        LOG(WARNING) << "GOSS beta hyperparameter given but GOSS is disabled.";
      }
    }
  }

  {
    const auto selgb = generic_hyper_params->Get(kHParamSelGBRatio);
    if (selgb.has_value()) {
      if (gbt_config->has_selective_gradient_boosting()) {
        gbt_config->mutable_selective_gradient_boosting()->set_ratio(
            selgb.value().value().real());
      } else {
        LOG(WARNING)
            << "SelGB ratio hyperparameter given but SelGB is disabled.";
      }
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamValidationSetRatio);
    if (hparam.has_value()) {
      gbt_config->set_validation_set_ratio(hparam.value().value().real());
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamEarlyStoppingNumTreesLookAhead);
    if (hparam.has_value()) {
      gbt_config->set_early_stopping_num_trees_look_ahead(
          hparam.value().value().integer());
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamValidationIntervalInTrees);
    if (hparam.has_value()) {
      gbt_config->set_validation_interval_in_trees(
          hparam.value().value().integer());
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamEarlyStopping);
    if (hparam.has_value()) {
      const auto early_stopping = hparam.value().value().categorical();
      if (early_stopping == kHParamEarlyStoppingNone) {
        gbt_config->set_early_stopping(
            proto::GradientBoostedTreesTrainingConfig::NONE);
      } else if (early_stopping == kHParamEarlyStoppingMinLossFullModel) {
        gbt_config->set_early_stopping(
            proto::GradientBoostedTreesTrainingConfig::
                MIN_VALIDATION_LOSS_ON_FULL_MODEL);
      } else if (early_stopping == kHParamEarlyStoppingLossIncrease) {
        gbt_config->set_early_stopping(
            proto::GradientBoostedTreesTrainingConfig::
                VALIDATION_LOSS_INCREASE);
      }
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamApplyLinkFunction);
    if (hparam.has_value()) {
      gbt_config->set_apply_link_function(
          hparam.value().value().categorical() == "true");
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamComputePermutationVariableImportance);
    if (hparam.has_value()) {
      gbt_config->set_compute_permutation_variable_importance(
          hparam.value().value().categorical() == "true");
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamFocalLossGamma);
    if (hparam.has_value()) {
      gbt_config->mutable_binary_focal_loss_options()
          ->set_misprediction_exponent(hparam.value().value().real());
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamFocalLossAlpha);
    if (hparam.has_value()) {
      gbt_config->mutable_binary_focal_loss_options()
          ->set_positive_sample_coefficient(hparam.value().value().real());
    }
  }

  return absl::OkStatus();
}

utils::StatusOr<model::proto::HyperParameterSpace>
GradientBoostedTreesLearner::PredefinedHyperParameterSpace() const {
  model::proto::HyperParameterSpace space;

  decision_tree::PredefinedHyperParameterAxisSplitSpace(&space);

  // Note: We don't optimize the number of tree as it is always beneficial
  // metric wise, and we don't optimize the inference time or model size (yet).

  {
    auto* field = space.add_fields();
    field->set_name(decision_tree::kHParamMaxDepth);
    auto* cands = field->mutable_discrete_candidates();
    cands->add_possible_values()->set_integer(-1);
    cands->add_possible_values()->set_integer(3);
    cands->add_possible_values()->set_integer(4);
    cands->add_possible_values()->set_integer(6);
    cands->add_possible_values()->set_integer(8);
  }

  {
    auto* field = space.add_fields();
    field->set_name(decision_tree::kHParamGrowingStrategy);
    auto* cands = field->mutable_discrete_candidates();
    cands->add_possible_values()->set_categorical(
        decision_tree::kGrowingStrategyLocal);
    cands->add_possible_values()->set_categorical(
        decision_tree::kGrowingStrategyBestFirstGlobal);

    auto* child = field->add_children();
    child->set_name(decision_tree::kHParamMaxNumNodes);
    auto* parent_values = child->mutable_parent_discrete_values();
    parent_values->add_possible_values()->set_categorical(
        decision_tree::kGrowingStrategyBestFirstGlobal);
    auto* child_cands = child->mutable_discrete_candidates();
    child_cands->add_possible_values()->set_integer(8);
    child_cands->add_possible_values()->set_integer(16);
    child_cands->add_possible_values()->set_integer(31);
    child_cands->add_possible_values()->set_integer(63);
    child_cands->add_possible_values()->set_integer(127);
    child_cands->add_possible_values()->set_integer(255);
    child_cands->add_possible_values()->set_integer(512);
  }

  {
    auto* field = space.add_fields();
    field->set_name(kHParamSamplingMethod);
    auto* cands = field->mutable_discrete_candidates();
    cands->add_possible_values()->set_categorical(kSamplingMethodNone);
    cands->add_possible_values()->set_categorical(kSamplingMethodRandom);
    cands->add_possible_values()->set_categorical(kSamplingMethodGOSS);

    // Random sampling method (aka, subsample).
    auto* random = field->add_children();
    random->set_name(kHParamSubsample);
    auto* random_parent_values = random->mutable_parent_discrete_values();
    random_parent_values->add_possible_values()->set_categorical(
        kSamplingMethodRandom);
    auto* random_cands = random->mutable_discrete_candidates();
    random_cands->add_possible_values()->set_real(0.6);
    random_cands->add_possible_values()->set_real(0.8);
    random_cands->add_possible_values()->set_real(0.9);
    random_cands->add_possible_values()->set_real(1.0);

    // GOSS sampling method.
    auto* goss_alpha = field->add_children();
    goss_alpha->set_name(kHParamGossAlpha);
    auto* goss_alpha_parent_values =
        goss_alpha->mutable_parent_discrete_values();
    goss_alpha_parent_values->add_possible_values()->set_categorical(
        kSamplingMethodGOSS);
    auto* goss_alpha_cands = goss_alpha->mutable_discrete_candidates();
    goss_alpha_cands->add_possible_values()->set_real(0.05);
    goss_alpha_cands->add_possible_values()->set_real(0.1);
    goss_alpha_cands->add_possible_values()->set_real(0.15);
    goss_alpha_cands->add_possible_values()->set_real(0.2);

    auto* goss_beta = field->add_children();
    goss_beta->set_name(kHParamGossBeta);
    auto* goss_beta_parent_values = goss_beta->mutable_parent_discrete_values();
    goss_beta_parent_values->add_possible_values()->set_categorical(
        kSamplingMethodGOSS);
    auto* goss_beta_cands = goss_beta->mutable_discrete_candidates();
    goss_beta_cands->add_possible_values()->set_real(0.05);
    goss_beta_cands->add_possible_values()->set_real(0.1);
    goss_beta_cands->add_possible_values()->set_real(0.15);
    goss_beta_cands->add_possible_values()->set_real(0.2);

    // Selective Gradient Boosting sampling method.
    if (training_config_.task() == model::proto::Task::RANKING) {
      cands->add_possible_values()->set_categorical(kSamplingMethodSelGB);

      auto* child = field->add_children();
      child->set_name(kHParamSelGBRatio);
      auto* parent_values = child->mutable_parent_discrete_values();
      parent_values->add_possible_values()->set_categorical(
          kSamplingMethodSelGB);
      auto* child_cands = child->mutable_discrete_candidates();
      child_cands->add_possible_values()->set_real(0.01);
      child_cands->add_possible_values()->set_real(0.05);
      child_cands->add_possible_values()->set_real(0.1);
      child_cands->add_possible_values()->set_real(0.2);
      child_cands->add_possible_values()->set_real(1.0);
    }
  }

  {
    auto* field = space.add_fields();
    field->set_name(kHParamShrinkage);
    auto* cands = field->mutable_discrete_candidates();
    cands->add_possible_values()->set_real(0.02);
    cands->add_possible_values()->set_real(0.05);
    cands->add_possible_values()->set_real(0.10);
    cands->add_possible_values()->set_real(0.15);
  }

  {
    auto* field = space.add_fields();
    field->set_name(decision_tree::kHParamMinExamples);
    auto* cands = field->mutable_discrete_candidates();
    cands->add_possible_values()->set_integer(5);
    cands->add_possible_values()->set_integer(10);
    cands->add_possible_values()->set_integer(50);
  }

  if (training_config_.task() != model::proto::Task::REGRESSION) {
    auto* field = space.add_fields();
    field->set_name(kHParamUseHessianGain);
    auto* cands = field->mutable_discrete_candidates();
    cands->add_possible_values()->set_categorical("true");
    cands->add_possible_values()->set_categorical("false");
  }

  const std::vector<float> l2_loss_regularization = {0.f, 0.1f, 1.0f, 10.f,
                                                     100.f};
  const std::vector<float> l1_loss_regularization = {0.f, 0.1f, 1.0f, 10.f};
  {
    auto* field = space.add_fields();
    field->set_name(kHParamL2Regularization);
    auto* candidates = field->mutable_discrete_candidates();
    for (const auto& value : l2_loss_regularization) {
      candidates->add_possible_values()->set_real(value);
    }
  }

  {
    auto* field = space.add_fields();
    field->set_name(kHParamL2CategoricalRegularization);
    auto* candidates = field->mutable_discrete_candidates();
    for (const auto& value : l2_loss_regularization) {
      candidates->add_possible_values()->set_real(value);
    }
  }

  {
    auto* field = space.add_fields();
    field->set_name(kHParamL1Regularization);
    auto* candidates = field->mutable_discrete_candidates();
    for (const auto& value : l1_loss_regularization) {
      candidates->add_possible_values()->set_real(value);
    }
  }

  return space;
}

utils::StatusOr<model::proto::GenericHyperParameterSpecification>
GradientBoostedTreesLearner::GetGenericHyperParameterSpecification() const {
  ASSIGN_OR_RETURN(auto hparam_def,
                   AbstractLearner::GetGenericHyperParameterSpecification());

  hparam_def.mutable_documentation()->set_description(
      "A GBT (Gradient Boosted [Decision] Tree; "
      "https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) is a set of shallow "
      "decision trees trained sequentially. Each tree is trained to predict "
      "and then \"correct\" for the errors of the previously trained trees "
      "(more precisely each tree predict the gradient of the loss relative to "
      "the model output).");

  const auto proto_path =
      "learner/gradient_boosted_trees/gradient_boosted_trees.proto";

  model::proto::TrainingConfig config;
  RETURN_IF_ERROR(SetDefaultHyperParameters(&config));
  const auto& gbt_config = config.GetExtension(
      gradient_boosted_trees::proto::gradient_boosted_trees_config);

  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamLoss);
    param.mutable_categorical()->set_default_value(
        proto::Loss_Name(gbt_config.loss()));

    for (int loss_idx = 0; loss_idx < proto::Loss_ARRAYSIZE; loss_idx++) {
      if (proto::Loss_IsValid(loss_idx)) {
        param.mutable_categorical()->add_possible_values(
            proto::Loss_Name(loss_idx));
      }
    }

    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(The loss optimized by the model. If not specified (DEFAULT) the loss is selected automatically according to the \"task\" and label statistics. For example, if task=CLASSIFICATION and the label has two possible values, the loss will be set to BINOMIAL_LOG_LIKELIHOOD. Possible values are:
- `DEFAULT`: Select the loss automatically according to the task and label statistics.
- `BINOMIAL_LOG_LIKELIHOOD`: Binomial log likelihood. Only valid for binary classification.
- `SQUARED_ERROR`: Least square loss. Only valid for regression.
- `MULTINOMIAL_LOG_LIKELIHOOD`: Multinomial log likelihood i.e. cross-entropy. Only valid for binary or multi-class classification.
- `LAMBDA_MART_NDCG5`: LambdaMART with NDCG5.
- `XE_NDCG_MART`:  Cross Entropy Loss NDCG. See arxiv.org/abs/1911.09798.
)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamNumTrees);
    param.mutable_integer()->set_minimum(1);
    param.mutable_integer()->set_default_value(gbt_config.num_trees());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Maximum number of decision trees. The effective number of trained tree can be smaller if early stopping is enabled.)");
  }
  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamShrinkage);
    param.mutable_real()->set_minimum(0.f);
    param.mutable_real()->set_maximum(1.f);
    param.mutable_real()->set_default_value(gbt_config.shrinkage());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Coefficient applied to each tree prediction. A small value (0.02) tends to give more accurate results (assuming enough trees are trained), but results in larger models. Analogous to neural network learning rate.)");
  }
  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamL1Regularization);
    param.mutable_real()->set_minimum(0.f);
    param.mutable_real()->set_default_value(gbt_config.l1_regularization());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(L1 regularization applied to the training loss. Impact the tree structures and lead values.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamL2Regularization);
    param.mutable_real()->set_minimum(0.f);
    param.mutable_real()->set_default_value(gbt_config.l2_regularization());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(L2 regularization applied to the training loss for all features except the categorical ones.)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](
        kHParamL2CategoricalRegularization);
    param.mutable_real()->set_minimum(0.f);
    param.mutable_real()->set_default_value(
        gbt_config.l2_regularization_categorical());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_proto_field(
        "l2_regularization_categorical");
    param.mutable_documentation()->set_description(
        R"(L2 regularization applied to the training loss for categorical features. Impact the tree structures and lead values.)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamLambdaLoss);
    param.mutable_real()->set_minimum(0.f);
    param.mutable_real()->set_default_value(gbt_config.lambda_loss());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Lambda regularization applied to certain training loss functions. Only for NDCG loss.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamForestExtraction);
    param.mutable_categorical()->set_default_value(kHParamForestExtractionMart);
    param.mutable_categorical()->add_possible_values(
        kHParamForestExtractionMart);
    param.mutable_categorical()->add_possible_values(
        kHParamForestExtractionDart);
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_proto_field("forest_extraction");
    param.mutable_documentation()->set_description(
        R"(How to construct the forest:
- MART: For Multiple Additive Regression Trees. The "classical" way to build a GBDT i.e. each tree tries to "correct" the mistakes of the previous trees.
- DART: For Dropout Additive Regression Trees. A modification of MART proposed in http://proceedings.mlr.press/v38/korlakaivinayak15.pdf. Here, each tree tries to "correct" the mistakes of a random subset of the previous trees.)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamDartDropOut);
    param.mutable_real()->set_minimum(0.f);
    param.mutable_real()->set_maximum(1.f);
    param.mutable_real()->set_default_value(gbt_config.dart().dropout_rate());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_proto_field("dropout_rate");
    param.mutable_documentation()->set_description(
        R"(Dropout rate applied when using the DART i.e. when forest_extraction=DART.)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](
        kHParamAdaptSubsampleForMaximumTrainingDuration);
    param.mutable_categorical()->set_default_value(
        gbt_config.adapt_subsample_for_maximum_training_duration() ? "true"
                                                                   : "false");
    param.mutable_categorical()->add_possible_values("true");
    param.mutable_categorical()->add_possible_values("false");
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Control how the maximum training duration (if set) is applied. If false, the training stop when the time is used. If true, the size of the sampled datasets used train individual trees are adapted dynamically so that all the trees are trained in time.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamUseHessianGain);
    param.mutable_categorical()->set_default_value(
        gbt_config.use_hessian_gain() ? "true" : "false");
    param.mutable_categorical()->add_possible_values("true");
    param.mutable_categorical()->add_possible_values("false");
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Use true, uses a formulation of split gain with a hessian term i.e. optimizes the splits to minimize the variance of "gradient / hessian. Available for all losses except regression.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamSamplingMethod);
    param.mutable_categorical()->set_default_value(kSamplingMethodNone);
    param.mutable_categorical()->add_possible_values(kSamplingMethodNone);
    param.mutable_categorical()->add_possible_values(kSamplingMethodRandom);
    param.mutable_categorical()->add_possible_values(kSamplingMethodGOSS);
    if (config.task() == model::proto::Task::RANKING) {
      param.mutable_categorical()->add_possible_values(kSamplingMethodSelGB);
    }
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Control the sampling of the datasets used to train individual trees.
- NONE: No sampling is applied.
- RANDOM: Uniform random sampling. Automatically selected if "subsample" is set.
- GOSS: Gradient-based One-Side Sampling. Automatically selected if "goss_alpha" or "goss_beta" is set.
- SELGB: Selective Gradient Boosting. Automatically selected if "selective_gradient_boosting_ratio" is set.
)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamSubsample);
    param.mutable_real()->set_minimum(0.f);
    param.mutable_real()->set_maximum(1.f);
    param.mutable_real()->set_default_value(gbt_config.subsample());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Ratio of the dataset (sampling without replacement) used to train individual trees for the random sampling method.)");
  }
  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamGossAlpha);
    param.mutable_real()->set_minimum(0.f);
    param.mutable_real()->set_maximum(1.f);
    param.mutable_real()->set_default_value(
        gbt_config.gradient_one_side_sampling().alpha());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Alpha parameter for the GOSS (Gradient-based One-Side Sampling; "See LightGBM: A Highly Efficient Gradient Boosting Decision Tree") sampling method.)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamGossBeta);
    param.mutable_real()->set_minimum(0.f);
    param.mutable_real()->set_maximum(1.f);
    param.mutable_real()->set_default_value(
        gbt_config.gradient_one_side_sampling().beta());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Beta parameter for the GOSS (Gradient-based One-Side Sampling) sampling method.)");
  }
  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamSelGBRatio);
    param.mutable_real()->set_minimum(0.f);
    param.mutable_real()->set_maximum(1.f);
    param.mutable_real()->set_default_value(
        gbt_config.selective_gradient_boosting().ratio());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_proto_field("ratio");
    param.mutable_documentation()->set_description(
        R"(Ratio of the dataset used to train individual tree for the selective Gradient Boosting (Selective Gradient Boosting for Effective Learning to Rank; Lucchese et al; http://quickrank.isti.cnr.it/selective-data/selective-SIGIR2018.pdf) sampling method.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamValidationSetRatio);
    param.mutable_real()->set_minimum(0.f);
    param.mutable_real()->set_maximum(1.f);
    param.mutable_real()->set_default_value(gbt_config.validation_set_ratio());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Ratio of the training dataset used to monitor the training. Require to be >0 if early stopping is enabled.)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](
        kHParamEarlyStoppingNumTreesLookAhead);
    param.mutable_integer()->set_minimum(1);
    param.mutable_integer()->set_default_value(
        gbt_config.early_stopping_num_trees_look_ahead());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Rolling number of trees used to detect validation loss increase and trigger early stopping.)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](
        kHParamValidationIntervalInTrees);
    param.mutable_integer()->set_minimum(1);
    param.mutable_integer()->set_default_value(
        gbt_config.validation_interval_in_trees());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Evaluate the model on the validation set every "validation_interval_in_trees" trees. Increasing this value reduce the cost of validation and can impact the early stopping policy (as early stopping is only tested during the validation).)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamEarlyStopping);
    param.mutable_categorical()->set_default_value(
        kHParamEarlyStoppingLossIncrease);
    param.mutable_categorical()->add_possible_values(kHParamEarlyStoppingNone);
    param.mutable_categorical()->add_possible_values(
        kHParamEarlyStoppingMinLossFullModel);
    param.mutable_categorical()->add_possible_values(
        kHParamEarlyStoppingLossIncrease);
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Early stopping detects the overfitting of the model and halts it training using the validation dataset controlled by `validation_ratio`.
- `NONE`: No early stopping. The model is trained entirely.
- `MIN_LOSS_FINAL`: No early stopping. However, the model is then truncated to maximize the validation loss.
- `LOSS_INCREASE`: Stop the training when the validation does not decrease for `early_stopping_num_trees_look_ahead` trees.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamApplyLinkFunction);
    param.mutable_categorical()->set_default_value(
        gbt_config.apply_link_function() ? "true" : "false");
    param.mutable_categorical()->add_possible_values("true");
    param.mutable_categorical()->add_possible_values("false");
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(If true, applies the link function (a.k.a. activation function), if any, before returning the model prediction. If false, returns the pre-link function model output.
For example, in the case of binary classification, the pre-link function output is a logic while the post-link function is a probability.)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](
        kHParamComputePermutationVariableImportance);
    param.mutable_categorical()->set_default_value(
        gbt_config.compute_permutation_variable_importance() ? "true"
                                                             : "false");
    param.mutable_categorical()->add_possible_values("true");
    param.mutable_categorical()->add_possible_values("false");
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(If true, compute the permutation variable importance of the model at the end of the training using the validation dataset. Enabling this feature can increase the training time significantly.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamFocalLossGamma);
    param.mutable_real()->set_minimum(0.f);
    param.mutable_real()->set_default_value(
        gbt_config.binary_focal_loss_options().misprediction_exponent());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(EXPERIMENTAL. Exponent of the misprediction exponent term in focal loss, corresponds to gamma parameter in https://arxiv.org/pdf/1708.02002.pdf. Only used with focal loss i.e. `loss="BINARY_FOCAL_LOSS"`)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamFocalLossAlpha);
    param.mutable_real()->set_minimum(0.f);
    param.mutable_real()->set_maximum(1.f);
    param.mutable_real()->set_default_value(
        gbt_config.binary_focal_loss_options().positive_sample_coefficient());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(EXPERIMENTAL. Weighting parameter for focal loss, positive samples weighted by alpha, negative samples by (1-alpha). The default 0.5 value means no active class-level weighting. Only used with focal loss i.e. `loss="BINARY_FOCAL_LOSS"`)");
  }

  RETURN_IF_ERROR(decision_tree::GetGenericHyperParameterSpecification(
      gbt_config.decision_tree(), &hparam_def));
  return hparam_def;
}

namespace internal {

utils::StatusOr<proto::Loss> DefaultLoss(
    const model::proto::Task task, const dataset::proto::Column& label_spec) {
  if (task == model::proto::Task::CLASSIFICATION &&
      label_spec.type() == dataset::proto::ColumnType::CATEGORICAL) {
    if (label_spec.categorical().number_of_unique_values() == 3) {
      // Note: "number_of_unique_values() == 3" because of the reserved
      // "out-of-dictionary" item.
      return proto::Loss::BINOMIAL_LOG_LIKELIHOOD;
    } else if (label_spec.categorical().number_of_unique_values() > 3) {
      return proto::Loss::MULTINOMIAL_LOG_LIKELIHOOD;
    } else {
      return absl::InvalidArgumentError(
          "No default loss available for a categorical label with a single "
          "unique value. 1) Make sure you want classification (e.g. instead of "
          "regression), 2) Make sure your training dataset contains at least "
          "two different categorical label values. 3) Alternatively, specify "
          "manually the loss e.g. loss=BINOMIAL_LOG_LIKELIHOOD.");
    }
  }

  if (task == model::proto::Task::REGRESSION &&
      label_spec.type() == dataset::proto::ColumnType::NUMERICAL) {
    return proto::Loss::SQUARED_ERROR;
  }

  if (task == model::proto::Task::RANKING &&
      label_spec.type() == dataset::proto::ColumnType::NUMERICAL) {
    return proto::Loss::LAMBDA_MART_NDCG5;
  }

  return absl::InvalidArgumentError(
      "No defined default loss for this combination of label type and task");
}

utils::StatusOr<std::unique_ptr<CompleteTrainingDatasetForWeakLearner>>
LoadCompleteDatasetForWeakLearner(
    const std::vector<std::string>& shards,
    const absl::string_view format_prefix,
    const dataset::proto::DataSpecification& data_spec,
    const AllTrainingConfiguration& config, const bool allocate_gradient,
    const GradientBoostedTreesModel* mdl) {
  auto complete_dataset =
      absl::make_unique<CompleteTrainingDatasetForWeakLearner>();

  const auto dataset_loading_config =
      OptimalDatasetLoadingConfig(config.train_config_link);

  RETURN_IF_ERROR(dataset::LoadVerticalDataset(
      absl::StrCat(format_prefix, ":", absl::StrJoin(shards, ",")), data_spec,
      &complete_dataset->dataset, {}, dataset_loading_config));

  RETURN_IF_ERROR(dataset::GetWeights(complete_dataset->dataset,
                                      config.train_config_link,
                                      &complete_dataset->weights));

  RETURN_IF_ERROR(internal::CreateGradientDataset(
      complete_dataset->dataset, config.train_config_link.label(),
      config.gbt_config->use_hessian_gain(), *config.loss,
      &complete_dataset->gradient_dataset,
      allocate_gradient ? &complete_dataset->gradients : nullptr,
      &complete_dataset->predictions));

  // Training configuration for individual trees i.e. to predict the gradient.
  ConfigureTrainingConfigForGradients(
      config.train_config, config.train_config_link, SubTask(mdl->loss()),
      complete_dataset->gradient_dataset, &complete_dataset->gradients);

  return complete_dataset;
}

absl::Status ExtractValidationDataset(const VerticalDataset& dataset,
                                      const float validation_set_ratio,
                                      const int group_column_idx,
                                      VerticalDataset* train,
                                      VerticalDataset* validation,
                                      utils::RandomEngine* random) {
  if (validation_set_ratio < 0.f || validation_set_ratio > 1.f) {
    return absl::InvalidArgumentError(
        "The validation set ratio should be in [0,1].");
  }
  if (validation_set_ratio == 0) {
    *train = dataset.ShallowNonOwningClone();
  } else {
    std::uniform_real_distribution<float> unif_dist_01;
    std::vector<row_t> training_rows;
    std::vector<row_t> validation_rows;

    if (group_column_idx == -1) {
      // Sampling per example.
      for (row_t row = 0; row < dataset.nrow(); row++) {
        const bool in_training = unif_dist_01(*random) > validation_set_ratio;
        (in_training ? training_rows : validation_rows).push_back(row);
      }
      *train = dataset.Extract(training_rows).value();
      *validation = dataset.Extract(validation_rows).value();
    } else {
      // Extract groups.
      const auto* group_categorical_values = dataset.ColumnWithCastOrNull<
          dataset::VerticalDataset::CategoricalColumn>(group_column_idx);
      const auto* group_hash_values =
          dataset.ColumnWithCastOrNull<dataset::VerticalDataset::HashColumn>(
              group_column_idx);

      absl::flat_hash_map<uint64_t, std::vector<row_t>> rows_per_groups;
      for (row_t row_idx = 0; row_idx < dataset.nrow(); row_idx++) {
        // Get the value of the group.
        uint64_t group_value;
        if (group_categorical_values) {
          group_value = group_categorical_values->values()[row_idx];
        } else if (group_hash_values) {
          group_value = group_hash_values->values()[row_idx];
        } else {
          return absl::InvalidArgumentError("Invalid group type");
        }

        rows_per_groups[group_value].push_back(row_idx);
      }

      std::vector<std::vector<row_t>> rows_per_groups_decreasing_volume;
      rows_per_groups_decreasing_volume.reserve(rows_per_groups.size());
      for (auto& group : rows_per_groups) {
        rows_per_groups_decreasing_volume.push_back(std::move(group.second));
      }
      std::shuffle(rows_per_groups_decreasing_volume.begin(),
                   rows_per_groups_decreasing_volume.end(), *random);
      std::sort(rows_per_groups_decreasing_volume.begin(),
                rows_per_groups_decreasing_volume.end(),
                [](const std::vector<row_t>& a, const std::vector<row_t>& b) {
                  if (a.size() == b.size()) {
                    return std::lexicographical_compare(a.begin(), a.end(),
                                                        b.begin(), b.end());
                  }
                  return a.size() > b.size();
                });

      // Sampling per groups.
      for (const auto& group : rows_per_groups_decreasing_volume) {
        auto& dst = (training_rows.size() * validation_set_ratio <
                     validation_rows.size())
                        ? training_rows
                        : validation_rows;
        dst.insert(dst.end(), group.begin(), group.end());
      }
      LOG(INFO) << "Split training/validation dataset by \""
                << dataset.data_spec().columns(group_column_idx).name()
                << "\". " << rows_per_groups.size() << " groups found in "
                << dataset.nrow() << " examples i.e. "
                << static_cast<float>(dataset.nrow()) / rows_per_groups.size()
                << " examples/groups.";
    }

    *train = dataset.Extract(training_rows).value();
    *validation = dataset.Extract(validation_rows).value();
  }
  return absl::OkStatus();
}

absl::Status CreateGradientDataset(const dataset::VerticalDataset& dataset,
                                   const int label_col_idx,
                                   const bool hessian_splits,
                                   const AbstractLoss& loss_impl,
                                   dataset::VerticalDataset* gradient_dataset,
                                   std::vector<GradientData>* gradients,
                                   std::vector<float>* predictions) {
  const auto loss_shape = loss_impl.Shape();
  *gradient_dataset = dataset.ShallowNonOwningClone();

  if (gradients) {
    gradients->reserve(loss_shape.gradient_dim);
    // Allocate gradients.
    for (int gradient_idx = 0; gradient_idx < loss_shape.gradient_dim;
         gradient_idx++) {
      const auto grad_col_name = GradientColumnName(gradient_idx);
      dataset::proto::Column gradient_spec;
      gradient_spec.set_name(grad_col_name);
      gradient_spec.set_type(dataset::proto::ColumnType::NUMERICAL);

      GradientData gradient = {
          /*.gradient =*/
          *(dynamic_cast<dataset::VerticalDataset::NumericalColumn*>(
                gradient_dataset->AddColumn(gradient_spec).value())
                ->mutable_values()),
          /*.gradient_column_name =*/grad_col_name};

      if (loss_shape.has_hessian) {
        const auto hessian_col_name = HessianColumnName(gradient_idx);
        dataset::proto::Column hessian_col_spec;
        hessian_col_spec.set_name(hessian_col_name);
        hessian_col_spec.set_type(dataset::proto::ColumnType::NUMERICAL);

        // Note: These values will be set correctly before use.
        gradient.hessian =
            dynamic_cast<dataset::VerticalDataset::NumericalColumn*>(
                gradient_dataset->AddColumn(hessian_col_spec).value())
                ->mutable_values();
        gradient.hessian_col_idx =
            gradient_dataset->ColumnNameToColumnIdx(hessian_col_name);
        if (gradient.hessian_col_idx < 0) {
          return absl::InternalError("No allocated hessian column");
        }
      }
      gradients->push_back(std::move(gradient));
    }
  }

  // Allocate predictions.
  if (predictions) {
    predictions->resize(dataset.nrow() * loss_shape.prediction_dim);
  }
  return absl::OkStatus();
}

absl::Status ComputePredictions(
    const GradientBoostedTreesModel* mdl,
    const serving::FastEngine* optional_engine,
    const std::vector<decision_tree::DecisionTree*>& trees,
    const internal::AllTrainingConfiguration& config,
    const dataset::VerticalDataset& gradient_dataset,
    std::vector<float>* predictions) {
  if (optional_engine) {
    // Prediction using the engine (fast).

    if (optional_engine->NumPredictionDimension() !=
        mdl->initial_predictions().size()) {
      return absl::InternalError("Unexpected number of prediction dimensions");
    }

    const size_t batch_size = 1000;
    const size_t num_examples = gradient_dataset.nrow();
    auto examples =
        optional_engine->AllocateExamples(std::min(batch_size, num_examples));
    const size_t num_batches = (num_examples + batch_size - 1) / batch_size;
    std::vector<float> batch_predictions;

    predictions->resize(num_examples * mdl->initial_predictions().size());

    for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
      const size_t begin_example_idx = batch_idx * batch_size;
      const size_t end_example_idx =
          std::min(begin_example_idx + batch_size, num_examples);
      RETURN_IF_ERROR(serving::CopyVerticalDatasetToAbstractExampleSet(
          gradient_dataset,
          /*begin_example_idx=*/begin_example_idx,
          /*end_example_idx=*/end_example_idx, optional_engine->features(),
          examples.get()));
      optional_engine->Predict(*examples, end_example_idx - begin_example_idx,
                               &batch_predictions);
      std::copy(batch_predictions.begin(), batch_predictions.end(),
                predictions->begin() +
                    begin_example_idx * mdl->initial_predictions().size());
    }

  } else {
    SetInitialPredictions(mdl->initial_predictions(), gradient_dataset.nrow(),
                          predictions);
  }

  // Predictions using the trees (slow).
  const int num_iterations = trees.size() / mdl->num_trees_per_iter();
  std::vector<const decision_tree::DecisionTree*> selected_trees(
      mdl->num_trees_per_iter());
  for (int iter_idx = 0; iter_idx < num_iterations; iter_idx++) {
    for (int tree_idx = 0; tree_idx < mdl->num_trees_per_iter(); tree_idx++) {
      selected_trees[tree_idx] =
          trees[iter_idx * mdl->num_trees_per_iter() + tree_idx];
    }
    RETURN_IF_ERROR(config.loss->UpdatePredictions(
        selected_trees, gradient_dataset, predictions, nullptr));
  }
  return absl::OkStatus();
}

void SampleTrainingExamples(const dataset::VerticalDataset::row_t num_rows,
                            const float sample, utils::RandomEngine* random,
                            std::vector<row_t>* selected_examples) {
  if (sample >= 1.f - std::numeric_limits<float>::epsilon()) {
    selected_examples->resize(num_rows);
    std::iota(selected_examples->begin(), selected_examples->end(), 0);
    return;
  }

  selected_examples->clear();
  std::uniform_real_distribution<float> unif_dist_unit;
  for (row_t example_idx = 0; example_idx < num_rows; example_idx++) {
    if (unif_dist_unit(*random) < sample) {
      selected_examples->push_back(example_idx);
    }
  }
  if (selected_examples->empty()) {
    // Ensure at least one example is selected.
    selected_examples->push_back(
        std::uniform_int_distribution<row_t>(num_rows - 1)(*random));
  }
}

void SampleTrainingExamplesWithGoss(
    const std::vector<GradientData>& gradients,
    const dataset::VerticalDataset::row_t num_rows, const float alpha,
    const float beta, utils::RandomEngine* random,
    std::vector<row_t>* selected_examples, std::vector<float>* weights) {
  // Compute L1 norm of the gradient vector for every example.
  std::vector<std::pair<row_t, float>> l1_norm;
  l1_norm.reserve(num_rows);
  for (row_t example_idx = 0; example_idx < num_rows; example_idx++) {
    float example_l1_norm = 0.f;
    for (const auto& gradient_data : gradients) {
      example_l1_norm += std::fabs(gradient_data.gradient[example_idx]);
    }
    l1_norm.push_back(std::make_pair(example_idx, example_l1_norm));
  }

  // Sort examples by the L1 norm of gradients in decreasing order.
  std::sort(l1_norm.begin(), l1_norm.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  // Add examples with a large gradient to the set of selected examples.
  selected_examples->clear();

  int cutoff = std::ceil(alpha * num_rows);
  for (row_t idx = 0; idx < cutoff; idx++) {
    selected_examples->push_back(l1_norm[idx].first);
  }

  // From the remaining examples, randomly select a subset and adjust weights.
  if (beta > 0) {
    const float amplification_factor = (1.f - alpha) / beta;
    std::uniform_real_distribution<float> unif_dist_unit;
    for (row_t idx = cutoff; idx < num_rows; idx++) {
      if (unif_dist_unit(*random) < beta) {
        const row_t example_idx = l1_norm[idx].first;
        selected_examples->push_back(example_idx);
        (*weights)[example_idx] *= amplification_factor;
      }
    }
  }

  // Ensure at least one example is selected.
  if (selected_examples->empty()) {
    selected_examples->push_back(
        std::uniform_int_distribution<row_t>(num_rows - 1)(*random));
  }
}

absl::Status SampleTrainingExamplesWithSelGB(
    model::proto::Task task, const dataset::VerticalDataset::row_t num_rows,
    const RankingGroupsIndices* ranking_index,
    const std::vector<float>& predictions, const float ratio,
    std::vector<row_t>* selected_examples) {
  if (task != model::proto::Task::RANKING) {
    return absl::InvalidArgumentError(
        "Selective Gradient Boosting is only applicable to ranking");
  }

  if (ratio >= .999) {
    selected_examples->resize(num_rows);
    std::iota(selected_examples->begin(), selected_examples->end(), 0);
    return absl::OkStatus();
  }

  selected_examples->clear();
  std::vector<std::pair<row_t, float>> negative_predictions;

  for (const auto& group : ranking_index->groups()) {
    const auto group_size = group.items.size();
    negative_predictions.reserve(group_size);
    negative_predictions.clear();

    // Add positive examples to the set, and prepare negative examples for
    // down-sampling.
    for (int idx = 0; idx < group_size; idx++) {
      const auto example_idx = group.items[idx].example_idx;
      if (group.items[idx].relevance > 0) {
        selected_examples->push_back(example_idx);
      } else {
        negative_predictions.push_back(
            std::make_pair(example_idx, predictions[example_idx]));
      }
    }

    // Sort negative examples by prediction scores in decreasing order.
    std::sort(negative_predictions.begin(), negative_predictions.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    // Add the top examples to the set of selected examples.
    const int cutoff_idx = std::ceil(ratio * negative_predictions.size());
    for (int idx = 0; idx < cutoff_idx && idx < negative_predictions.size();
         idx++) {
      selected_examples->push_back(negative_predictions[idx].first);
    }
  }
  return absl::OkStatus();
}

absl::Status ExportTrainingLogs(const proto::TrainingLogs& training_logs,
                                absl::string_view directory) {
  // Add methods to plot training logs here.
  CHECK_OK(file::RecursivelyCreateDir(directory, file::Defaults()));

  // Export the logs in a .csv file.
  ASSIGN_OR_RETURN(auto file_handle, file::OpenOutputFile(file::JoinPath(
                                         directory, "training_logs.csv")));
  file::OutputFileCloser file(std::move(file_handle));

  utils::csv::Writer writer(file.stream());
  std::vector<std::string> fields = {"num_trees", "valid_loss", "train_loss"};
  for (const auto& metric : training_logs.secondary_metric_names()) {
    fields.push_back(absl::StrCat("valid_", metric));
  }
  for (const auto& metric : training_logs.secondary_metric_names()) {
    fields.push_back(absl::StrCat("train_", metric));
  }
  RETURN_IF_ERROR(writer.WriteRowStrings(fields));
  for (const auto& entry : training_logs.entries()) {
    std::vector<std::string> row;
    row.push_back(absl::StrCat(entry.number_of_trees()));
    row.push_back(absl::StrCat(entry.validation_loss()));
    row.push_back(absl::StrCat(entry.training_loss()));
    for (const auto metric : entry.validation_secondary_metrics()) {
      row.push_back(absl::StrCat(metric));
    }
    for (const auto metric : entry.training_secondary_metrics()) {
      row.push_back(absl::StrCat(metric));
    }
    RETURN_IF_ERROR(writer.WriteRowStrings(row));
  }

  return absl::OkStatus();
}

void InitializeModelWithTrainingConfig(
    const model::proto::TrainingConfig& training_config,
    const model::proto::TrainingConfigLinking& training_config_linking,
    GradientBoostedTreesModel* model) {
  InitializeModelWithAbstractTrainingConfig(training_config,
                                            training_config_linking, model);
}

void DartPredictionAccumulator::Initialize(
    const std::vector<float>& initial_predictions, const row_t num_rows) {
  SetInitialPredictions(initial_predictions, num_rows, &predictions_);
}

std::vector<int> DartPredictionAccumulator::SampleIterIndices(
    float dropout, utils::RandomEngine* random) const {
  if (prediction_per_tree_.empty()) {
    return {};
  }

  // Select each iter with the "dropout" probability.
  std::vector<int> selected_iter_idxs;
  std::uniform_real_distribution<float> unif_01;
  for (int iter_idx = 0; iter_idx < prediction_per_tree_.size(); iter_idx++) {
    if (unif_01(*random) < dropout) {
      selected_iter_idxs.push_back(iter_idx);
    }
  }

  // Ensure at least one iter is selected.
  if (selected_iter_idxs.empty()) {
    selected_iter_idxs.push_back(std::uniform_int_distribution<int>(
        0, prediction_per_tree_.size() - 1)(*random));
  }

  return selected_iter_idxs;
}

absl::Status DartPredictionAccumulator::GetAllPredictions(
    std::vector<float>* predictions) {
  if (predictions_.size() != predictions->size()) {
    return absl::InternalError("Wrong number of predictions");
  }
  std::copy(predictions_.begin(), predictions_.end(), predictions->begin());
  return absl::OkStatus();
}

absl::Status DartPredictionAccumulator::GetSampledPredictions(
    const std::vector<int>& dropout_iter_idxs,
    std::vector<float>* predictions) {
  if (dropout_iter_idxs.empty()) {
    return GetAllPredictions(predictions);
  }
  for (row_t example_idx = 0; example_idx < predictions_.size();
       example_idx++) {
    float acc = predictions_[example_idx];
    if (std::isnan(acc)) {
      return absl::InvalidArgumentError("Found NaN in predictions");
    }
    for (const auto iter_idx : dropout_iter_idxs) {
      acc -= prediction_per_tree_[iter_idx].predictions[example_idx] *
             prediction_per_tree_[iter_idx].weight;
    }
    if (std::isnan(acc)) {
      return absl::InvalidArgumentError("Found NaN in predictions");
    }
    (*predictions)[example_idx] = acc;
  }
  return absl::OkStatus();
}

absl::Status DartPredictionAccumulator::UpdateWithNewIteration(
    const std::vector<int>& selected_iter_idxs, proto::Loss loss,
    const AbstractLoss& loss_impl,
    const std::vector<std::unique_ptr<decision_tree::DecisionTree>>& new_trees,
    const dataset::VerticalDataset& gradient_dataset,
    int num_gradient_dimensions, double* mean_abs_prediction) {
  // Compute the predictions of the new trees.
  TreePredictions tree_prediction;
  tree_prediction.predictions.assign(predictions_.size(), 0.f);
  tree_prediction.weight = 1.0f / (selected_iter_idxs.size() + 1);
  RETURN_IF_ERROR(loss_impl.UpdatePredictions(
      RemoveUniquePtr(new_trees), gradient_dataset,
      &tree_prediction.predictions, mean_abs_prediction));

  const float sampled_factor = static_cast<float>(selected_iter_idxs.size()) /
                               (selected_iter_idxs.size() + 1);

  // Update the global predictions.
  for (row_t example_idx = 0; example_idx < predictions_.size();
       example_idx++) {
    if (std::isnan(predictions_[example_idx])) {
      return absl::InvalidArgumentError("Found NaN in predictions");
    }

    predictions_[example_idx] +=
        tree_prediction.predictions[example_idx] * tree_prediction.weight;

    if (std::isnan(predictions_[example_idx])) {
      return absl::InvalidArgumentError("Found NaN in predictions");
    }

    for (const auto iter_idx : selected_iter_idxs) {
      predictions_[example_idx] +=
          prediction_per_tree_[iter_idx].predictions[example_idx] *
          prediction_per_tree_[iter_idx].weight * (sampled_factor - 1.f);

      if (std::isnan(predictions_[example_idx])) {
        return absl::InvalidArgumentError("Found NaN in predictions");
      }
    }
  }

  // Update the weight of the selected iterations.
  for (const auto iter_idx : selected_iter_idxs) {
    prediction_per_tree_[iter_idx].weight *= sampled_factor;
  }

  prediction_per_tree_.push_back(std::move(tree_prediction));
  return absl::OkStatus();
}

std::vector<float> DartPredictionAccumulator::TreeOutputScaling() const {
  std::vector<float> scaling;
  scaling.reserve(prediction_per_tree_.size());
  for (const auto& per_tree : prediction_per_tree_) {
    scaling.push_back(per_tree.weight);
  }
  return scaling;
}

absl::Status EarlyStopping::Update(
    const float validation_loss,
    const std::vector<float>& validation_secondary_metrics,
    const int num_trees) {
  if (trees_per_iterations_ == -1) {
    return absl::InternalError(
        "The number of trees per iterations should be set before the update");
  }
  if (best_num_trees_ == -1 || validation_loss < best_loss_) {
    best_loss_ = validation_loss;
    best_metrics_ = validation_secondary_metrics;
    best_num_trees_ = num_trees;
  }
  last_loss_ = validation_loss;
  last_metrics_ = validation_secondary_metrics;
  last_num_trees_ = num_trees;
  return absl::OkStatus();
}

bool EarlyStopping::ShouldStop() {
  if (last_num_trees_ - best_num_trees_ >= num_trees_look_ahead_) {
    LOG(INFO) << "Early stop of the training because the validation "
                 "loss does not decrease anymore. Best valid-loss: "
              << best_loss_;
    return true;
  }
  return false;
}

void SetInitialPredictions(const std::vector<float>& initial_predictions,
                           const row_t num_rows,
                           std::vector<float>* predictions) {
  predictions->resize(num_rows * initial_predictions.size());
  size_t cur = 0;
  for (row_t example_idx = 0; example_idx < num_rows; example_idx++) {
    for (const auto initial_prediction : initial_predictions) {
      (*predictions)[cur++] = initial_prediction;
    }
  }
}

absl::Status SetDefaultHyperParameters(
    gradient_boosted_trees::proto::GradientBoostedTreesTrainingConfig*
        gbt_config) {
  decision_tree::SetDefaultHyperParameters(gbt_config->mutable_decision_tree());

  if (gbt_config->has_sample_with_shards()) {
    gbt_config->mutable_decision_tree()
        ->mutable_internal()
        ->set_sorting_strategy(
            decision_tree::proto::DecisionTreeTrainingConfig::Internal::
                IN_NODE);
  }

  if (!gbt_config->decision_tree().has_max_depth()) {
    if (gbt_config->decision_tree().has_growing_strategy_best_first_global()) {
      gbt_config->mutable_decision_tree()->set_max_depth(-1);
    } else {
      gbt_config->mutable_decision_tree()->set_max_depth(6);
    }
  }
  if (!gbt_config->decision_tree().has_num_candidate_attributes() &&
      !gbt_config->decision_tree().has_num_candidate_attributes_ratio()) {
    // The basic definition of GBT does not have any attribute sampling.
    gbt_config->mutable_decision_tree()->set_num_candidate_attributes(-1);
  }
  if (!gbt_config->has_shrinkage()) {
    if (gbt_config->forest_extraction_case() ==
        proto::GradientBoostedTreesTrainingConfig::kDart) {
      gbt_config->set_shrinkage(1.f);
    }
  }
  if (gbt_config->has_use_goss()) {
    if (gbt_config->has_gradient_one_side_sampling()) {
      LOG(WARNING) << "Ignoring deprecated use_goss, goss_alpha, and goss_beta "
                      "values because `gradient_one_side_sampling` is already "
                      "present in "
                      "the train config.";
    } else if ((gbt_config->has_subsample() && gbt_config->subsample() < 1) ||
               gbt_config->sampling_methods_case() !=
                   proto::GradientBoostedTreesTrainingConfig::
                       SAMPLING_METHODS_NOT_SET) {
      LOG(WARNING)
          << "Ignoring deprecated use_goss, goss_alpha, and goss_beta "
             "values because another sampling method is already present in the "
             "train config.";
    } else {
      gbt_config->mutable_gradient_one_side_sampling()->set_alpha(
          gbt_config->goss_alpha());
      gbt_config->mutable_gradient_one_side_sampling()->set_beta(
          gbt_config->goss_beta());
    }

    // Clear deprecated fields.
    gbt_config->clear_subsample();
    gbt_config->clear_use_goss();
    gbt_config->clear_goss_alpha();
    gbt_config->clear_goss_beta();
  }
  if (gbt_config->has_subsample()) {
    if (gbt_config->has_stochastic_gradient_boosting()) {
      LOG(WARNING)
          << "Ignoring deprecated subsample value because "
             "`stochastic_gradient_boosting` is already present in the config.";
    } else if (gbt_config->sampling_methods_case() !=
               proto::GradientBoostedTreesTrainingConfig::
                   SAMPLING_METHODS_NOT_SET) {
      LOG(WARNING) << "Ignoring deprecated subsample value because another "
                      "sampling method is already present in the train config.";
    } else {
      gbt_config->mutable_stochastic_gradient_boosting()->set_ratio(
          gbt_config->subsample());
    }

    // Clear deprecated fields.
    gbt_config->clear_subsample();
  }

  if (gbt_config->early_stopping() !=
          proto::GradientBoostedTreesTrainingConfig::NONE &&
      gbt_config->validation_set_ratio() == 0) {
    LOG(WARNING)
        << "early_stopping != \"NONE\" requires validation_set_ratio>0. "
           "Setting early_stopping=\"NONE\" (was \""
        << proto::GradientBoostedTreesTrainingConfig::EarlyStopping_Name(
               gbt_config->early_stopping())
        << "\") i.e. sabling early stopping.";
    gbt_config->set_early_stopping(
        proto::GradientBoostedTreesTrainingConfig::NONE);
  }
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
