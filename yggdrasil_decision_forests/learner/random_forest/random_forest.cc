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

#include "yggdrasil_decision_forests/learner/random_forest/random_forest.h"

#include <algorithm>
#include <atomic>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example_writer.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/weight.h"
#include "yggdrasil_decision_forests/dataset/weight.pb.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/generic_parameters.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/learner/types.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/utils/adaptive_work.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/feature_importance.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"
#include "yggdrasil_decision_forests/utils/usage.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace random_forest {

using row_t = dataset::VerticalDataset::row_t;

constexpr double kAdaptativeWarmUpSeconds = 5.0;
constexpr char RandomForestLearner::kRegisteredName[];
constexpr char RandomForestLearner::kHParamNumTrees[];
constexpr char RandomForestLearner::kHParamWinnerTakeAll[];
constexpr char RandomForestLearner::
    kHParamAdaptBootstrapSizeRatioForMaximumTrainingDuration[];
constexpr char RandomForestLearner::kHParamComputeOOBPerformances[];
constexpr char RandomForestLearner::kHParamComputeOOBVariableImportance[];
constexpr char RandomForestLearner::kHParamBootstrapTrainingDataset[];
constexpr char RandomForestLearner::kHParamBootstrapSizeRatio[];
constexpr char
    RandomForestLearner::kHParamNumOOBVariableImportancePermutations[];
constexpr char RandomForestLearner::kHParamSamplingWithReplacement[];

RandomForestLearner::RandomForestLearner(
    const model::proto::TrainingConfig& training_config)
    : AbstractLearner(training_config) {}

absl::Status RandomForestLearner::SetHyperParametersImpl(
    utils::GenericHyperParameterConsumer* generic_hyper_params) {
  RETURN_IF_ERROR(
      AbstractLearner::SetHyperParametersImpl(generic_hyper_params));
  const auto& rf_config = training_config_.MutableExtension(
      random_forest::proto::random_forest_config);

  // Decision tree specific hyper-parameters.
  absl::flat_hash_set<std::string> consumed_hparams;
  RETURN_IF_ERROR(decision_tree::SetHyperParameters(
      &consumed_hparams, rf_config->mutable_decision_tree(),
      generic_hyper_params));

  {
    const auto hparam = generic_hyper_params->Get(kHParamNumTrees);
    if (hparam.has_value()) {
      rf_config->set_num_trees(hparam.value().value().integer());
    }
  }
  {
    const auto hparam = generic_hyper_params->Get(kHParamWinnerTakeAll);
    if (hparam.has_value()) {
      rf_config->set_winner_take_all_inference(
          hparam.value().value().categorical() == "true");
    }
  }
  {
    const auto hparam = generic_hyper_params->Get(
        kHParamAdaptBootstrapSizeRatioForMaximumTrainingDuration);
    if (hparam.has_value()) {
      rf_config->set_adapt_bootstrap_size_ratio_for_maximum_training_duration(
          hparam.value().value().categorical() == "true");
    }
  }
  {
    const auto hparam =
        generic_hyper_params->Get(kHParamComputeOOBPerformances);
    if (hparam.has_value()) {
      rf_config->set_compute_oob_performances(
          hparam.value().value().categorical() == "true");
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamComputeOOBVariableImportance);
    if (hparam.has_value()) {
      rf_config->set_compute_oob_variable_importances(
          hparam.value().value().categorical() == "true");
      if (rf_config->compute_oob_variable_importances()) {
        rf_config->set_compute_oob_performances(true);
      }
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamBootstrapTrainingDataset);
    if (hparam.has_value()) {
      rf_config->set_bootstrap_training_dataset(
          hparam.value().value().categorical() == "true");
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamNumOOBVariableImportancePermutations);
    if (hparam.has_value()) {
      rf_config->set_num_oob_variable_importances_permutations(
          hparam.value().value().integer());
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamBootstrapSizeRatio);
    if (hparam.has_value()) {
      rf_config->set_bootstrap_size_ratio(hparam.value().value().real());
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamSamplingWithReplacement);
    if (hparam.has_value()) {
      rf_config->set_sampling_with_replacement(
          hparam.value().value().categorical() == "true");
    }
  }

  return absl::OkStatus();
}

utils::StatusOr<model::proto::HyperParameterSpace>
RandomForestLearner::PredefinedHyperParameterSpace() const {
  // Note: We don't optimize the number of tree as it is always beneficial
  // metric wise, and we don't optimize the inference time or model size (yet).
  model::proto::HyperParameterSpace space;

  decision_tree::PredefinedHyperParameterAxisSplitSpace(&space);

  {
    auto* field = space.add_fields();
    field->set_name(decision_tree::kHParamMaxDepth);
    auto* cands = field->mutable_discrete_candidates();
    cands->add_possible_values()->set_integer(12);
    cands->add_possible_values()->set_integer(16);
    cands->add_possible_values()->set_integer(20);
    cands->add_possible_values()->set_integer(25);
    cands->add_possible_values()->set_integer(30);
  }

  {
    auto* field = space.add_fields();
    field->set_name(decision_tree::kHParamMinExamples);
    auto* cands = field->mutable_discrete_candidates();
    cands->add_possible_values()->set_integer(1);
    cands->add_possible_values()->set_integer(2);
    cands->add_possible_values()->set_integer(5);
    cands->add_possible_values()->set_integer(10);
    cands->add_possible_values()->set_integer(40);
  }

  return space;
}

utils::StatusOr<model::proto::GenericHyperParameterSpecification>
RandomForestLearner::GetGenericHyperParameterSpecification() const {
  ASSIGN_OR_RETURN(auto hparam_def,
                   AbstractLearner::GetGenericHyperParameterSpecification());
  model::proto::TrainingConfig config;
  const auto proto_path = "learner/random_forest/random_forest.proto";

  hparam_def.mutable_documentation()->set_description(
      R"(A Random Forest (https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) is a collection of deep CART decision trees trained independently and without pruning. Each tree is trained on a random subset of the original training  dataset (sampled with replacement).

The algorithm is unique in that it is robust to overfitting, even in extreme cases e.g. when there is more features than training examples.

It is probably the most well-known of the Decision Forest training algorithms.)");

  const auto& rf_config =
      config.GetExtension(random_forest::proto::random_forest_config);

  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamNumTrees);
    param.mutable_integer()->set_minimum(1);
    param.mutable_integer()->set_default_value(rf_config.num_trees());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Number of individual decision trees. Increasing the number of trees can increase the quality of the model at the expense of size, training speed, and inference latency.)");
  }
  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamWinnerTakeAll);
    param.mutable_categorical()->set_default_value(
        rf_config.winner_take_all_inference() ? "true" : "false");
    param.mutable_categorical()->add_possible_values("true");
    param.mutable_categorical()->add_possible_values("false");
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_proto_field("winner_take_all_inference");

    param.mutable_documentation()->set_description(
        R"(Control how classification trees vote. If true, each tree votes for one class. If false, each tree vote for a distribution of classes. winner_take_all_inference=false is often preferable.)");
  }
  {
    auto& param = hparam_def.mutable_fields()->operator[](
        kHParamAdaptBootstrapSizeRatioForMaximumTrainingDuration);
    param.mutable_categorical()->set_default_value(
        rf_config.adapt_bootstrap_size_ratio_for_maximum_training_duration()
            ? "true"
            : "false");
    param.mutable_categorical()->add_possible_values("true");
    param.mutable_categorical()->add_possible_values("false");
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Control how the maximum training duration (if set) is applied. If false, the training stop when the time is used. If true, adapts the size of the sampled dataset used to train each tree such that `num_trees` will train within `maximum_training_duration`. Has no effect if there is no maximum training duration specified.)");
  }
  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamComputeOOBPerformances);
    param.mutable_categorical()->set_default_value(
        rf_config.compute_oob_performances() ? "true" : "false");
    param.mutable_categorical()->add_possible_values("true");
    param.mutable_categorical()->add_possible_values("false");
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(If true, compute the Out-of-bag evaluation (then available in the summary and model inspector). This evaluation is a cheap alternative to cross-validation evaluation.)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](
        kHParamComputeOOBVariableImportance);
    param.mutable_categorical()->set_default_value(
        rf_config.compute_oob_variable_importances() ? "true" : "false");
    param.mutable_categorical()->add_possible_values("true");
    param.mutable_categorical()->add_possible_values("false");
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(If true, compute the Out-of-bag feature importance (then available in the summary and model inspector). Note that the OOB feature importance can be expensive to compute.)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](
        kHParamBootstrapTrainingDataset);
    param.mutable_categorical()->set_default_value(
        rf_config.bootstrap_training_dataset() ? "true" : "false");
    param.mutable_categorical()->add_possible_values("true");
    param.mutable_categorical()->add_possible_values("false");
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(If true (default), each tree is trained on a separate dataset sampled with replacement from the original dataset. If false, all the trees are trained on the entire same dataset. If bootstrap_training_dataset:false, OOB metrics are not available. bootstrap_training_dataset=false is used in "Extremely randomized trees" (https://link.springer.com/content/pdf/10.1007%2Fs10994-006-6226-1.pdf).)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](
        kHParamNumOOBVariableImportancePermutations);
    param.mutable_integer()->set_minimum(1);
    param.mutable_integer()->set_default_value(
        rf_config.num_oob_variable_importances_permutations());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Number of time the dataset is re-shuffled to compute the permutation variable importances. Increasing this value increase the training time (if "compute_oob_variable_importances:true") as well as the stability of the oob variable importance metrics.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamBootstrapSizeRatio);
    param.mutable_real()->set_minimum(0);
    param.mutable_real()->set_default_value(rf_config.bootstrap_size_ratio());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Number of examples used to train each trees; expressed as a ratio of the training dataset size.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamSamplingWithReplacement);
    param.mutable_categorical()->set_default_value(
        rf_config.sampling_with_replacement() ? "true" : "false");
    param.mutable_categorical()->add_possible_values("true");
    param.mutable_categorical()->add_possible_values("false");
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(If true, the training examples are sampled with replacement. If false, the training samples are sampled without replacement. Only used when "bootstrap_training_dataset=true". If false (sampling without replacement) and if "bootstrap_size_ratio=1" (default), all the examples are used to train all the trees (you probably do not want that).)");
  }

  RETURN_IF_ERROR(decision_tree::GetGenericHyperParameterSpecification(
      rf_config.decision_tree(), &hparam_def));
  return hparam_def;
}

absl::Status RandomForestLearner::CheckConfiguration(
    const dataset::proto::DataSpecification& data_spec,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::RandomForestTrainingConfig& rf_config,
    const model::proto::DeploymentConfig& deployment) {
  RETURN_IF_ERROR(AbstractLearner::CheckConfiguration(data_spec, config,
                                                      config_link, deployment));
  // Check that the decision tree will contain prediction weighting is needed.
  if (!rf_config.winner_take_all_inference()) {
    if (!rf_config.decision_tree().store_detailed_label_distribution())
      return absl::InvalidArgumentError(
          "store_detailed_label_label_distribution should be true if "
          "winner_take_all is false. The decision trees need to contain the "
          "detailed label distributions.");
  }
  return absl::OkStatus();
}

utils::StatusOr<std::unique_ptr<AbstractModel>>
RandomForestLearner::TrainWithStatus(
    const dataset::VerticalDataset& train_dataset,
    absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
        valid_dataset) const {
  const auto begin_training = absl::Now();
  RETURN_IF_ERROR(CheckNumExamples(train_dataset.nrow()));

  if (training_config().task() != model::proto::Task::CLASSIFICATION &&
      training_config().task() != model::proto::Task::REGRESSION &&
      training_config().task() != model::proto::Task::CATEGORICAL_UPLIFT &&
      training_config().task() != model::proto::Task::NUMERICAL_UPLIFT) {
    std::string tip;
    if (training_config().task() == model::proto::Task::RANKING) {
      tip =
          " You probably want to try the GRADIENT_BOOSTED_TREES learner that "
          "support ranking.";
    }
    return absl::InvalidArgumentError(absl::StrCat(
        "The RANDOM_FOREST learner does not support the task ",
        model::proto::Task_Name(training_config().task()), ".", tip));
  }

  auto config_with_default = training_config();
  auto& rf_config = *config_with_default.MutableExtension(
      random_forest::proto::random_forest_config);
  decision_tree::SetDefaultHyperParameters(rf_config.mutable_decision_tree());

  // If the maximum model size is limited, "keep_non_leaf_label_distribution"
  // defaults to false.
  if (!rf_config.decision_tree().has_keep_non_leaf_label_distribution() &&
      config_with_default.has_maximum_model_size_in_memory_in_bytes()) {
    rf_config.mutable_decision_tree()->set_keep_non_leaf_label_distribution(
        false);
  }

  if (training_config().task() == model::proto::Task::NUMERICAL_UPLIFT &&
      rf_config.compute_oob_performances()) {
    LOG(WARNING) << "RF does not support OOB performances with the numerical "
                    "uplift task (yet).";
    rf_config.set_compute_oob_performances(false);
  }

  auto mdl = absl::make_unique<RandomForestModel>();
  mdl->set_data_spec(train_dataset.data_spec());
  model::proto::TrainingConfigLinking config_link;
  RETURN_IF_ERROR(AbstractLearner::LinkTrainingConfig(
      config_with_default, train_dataset.data_spec(), &config_link));
  internal::InitializeModelWithTrainingConfig(config_with_default, config_link,
                                              mdl.get());
  LOG(INFO) << "Training random forest on " << train_dataset.nrow()
            << " example(s) and " << config_link.features().size()
            << " feature(s).";
  RETURN_IF_ERROR(CheckConfiguration(train_dataset.data_spec(),
                                     config_with_default, config_link,
                                     rf_config, deployment()));
  utils::usage::OnTrainingStart(train_dataset.data_spec(), config_with_default,
                                config_link, train_dataset.nrow());

  std::vector<float> weights;

  // Determines if the training code supports `weights` to be empty if
  // all the examples have the same weight.
  //
  // Currently, this feature is supported for:
  // - Binary classification without oblique splits (default) and with local
  //   imputation policy (default) to handle missing values.
  bool use_optimized_unit_weights = false;
  if (training_config().task() == model::proto::Task::CLASSIFICATION &&
      rf_config.decision_tree().split_axis_case() !=
          decision_tree::proto::DecisionTreeTrainingConfig::
              kSparseObliqueSplit) {
    // Only use optimized unit weights for binary classification for now.
    if (config_link.num_label_classes() == 3) use_optimized_unit_weights = true;
  }

  RETURN_IF_ERROR(dataset::GetWeights(train_dataset, config_link, &weights,
                                      use_optimized_unit_weights));

  ASSIGN_OR_RETURN(const auto preprocessing,
                   decision_tree::PreprocessTrainingDataset(
                       train_dataset, config_with_default, config_link,
                       rf_config.decision_tree(), deployment_.num_threads()));

  utils::RandomEngine global_random(config_with_default.random_seed());
  // Individual seeds for each tree.
  std::vector<int64_t> tree_seeds;
  tree_seeds.reserve(rf_config.num_trees());

  if (!rf_config.internal().individual_tree_seeds().empty()) {
    if (rf_config.internal().individual_tree_seeds().size() !=
        rf_config.num_trees()) {
      return absl::InternalError("Wrong number of trees");
    }
    tree_seeds = {rf_config.internal().individual_tree_seeds().begin(),
                  rf_config.internal().individual_tree_seeds().end()};
  } else {
    for (int tree_idx = 0; tree_idx < rf_config.num_trees(); tree_idx++) {
      tree_seeds.push_back(global_random());
    }
  }
  for (int tree_idx = 0; tree_idx < rf_config.num_trees(); tree_idx++) {
    mdl->AddTree(absl::make_unique<decision_tree::DecisionTree>());
  }

  // OOB (out-of-bag) predictions.
  utils::concurrency::Mutex
      oob_metrics_mutex;  // Protects all the "oob_*" fields.

  // Prediction accumulator for each example in the training dataset
  // (oob_predictions.size()==training_dataset.nrow()).
  std::vector<internal::PredictionAccumulator> oob_predictions;

  // Time of the last display of OOB metrics in the console. Expressed in
  // seconds from an arbitrary referential. Protected by "oob_metrics_mutex".
  absl::Time last_oob_computation_time = absl::InfinitePast();
  // Number of trees the last time the OOB metrics was computed and displayed in
  // the console.
  int last_oob_computation_num_trees = 0;

  // Prediction accumulator for each example in the training dataset and
  // shuffled according to each input feature:
  // "oob_predictions_per_input_features[i][j]" is the prediction accumulator,
  // for the example "j" (i.e. row "j" in training_dataset), where the value of
  // the input feature "i" has been shuffled. "shuffled" means that, during
  // inference, the value of feature "i" for the example "j" is replaced by the
  // value of the example "k" (of the same feature), where "k" is uniformly
  // sampled in [0, dataset.nrow()[.
  std::vector<std::vector<internal::PredictionAccumulator>>
      oob_predictions_per_input_features;

  // OOB Performance and variable importance are only computed when training is
  // bootstrapped.
  const bool compute_oob_performances = rf_config.compute_oob_performances() &&
                                        rf_config.bootstrap_training_dataset();
  const bool compute_oob_variable_importances =
      rf_config.compute_oob_variable_importances() &&
      rf_config.bootstrap_training_dataset();

  if (compute_oob_performances) {
    internal::InitializeOOBPredictionAccumulators(
        train_dataset.nrow(), config_with_default, config_link,
        train_dataset.data_spec(), &oob_predictions);
  }
  if (compute_oob_variable_importances) {
    if (!rf_config.compute_oob_performances())
      return absl::InvalidArgumentError(
          "The OOB metric computation should be enabled to compute the "
          "Variable Importance i.e. \"compute_oob_variable_importances=true\" "
          "requires \"compute_oob_performances=true\".");
    oob_predictions_per_input_features.resize(
        train_dataset.data_spec().columns_size());
    for (const int feature_idx : config_link.features()) {
      internal::InitializeOOBPredictionAccumulators(
          train_dataset.nrow(), config_with_default, config_link,
          train_dataset.data_spec(),
          &oob_predictions_per_input_features[feature_idx]);
    }
  }

  // If true, only a subset of trees will have been trained.
  bool training_stopped_early = false;

  std::unique_ptr<utils::AdaptativeWork> adaptative_work;
  if (rf_config.adapt_bootstrap_size_ratio_for_maximum_training_duration()) {
    if (!rf_config.bootstrap_training_dataset()) {
      return absl::InvalidArgumentError(
          "\"bootstrap_training_dataset\" required for adaptive "
          "bootstrap_size_ratio");
    }
    adaptative_work = absl::make_unique<utils::AdaptativeWork>(
        rf_config.num_trees(),
        training_config().maximum_training_duration_seconds() *
            deployment().num_threads(),
        kAdaptativeWarmUpSeconds, rf_config.min_adapted_subsample());
  }

  // Count the number of nodes in the trees so far. Used to enforce the
  // "total_max_num_nodes" parameter.
  //
  // Protects all the following variables.
  utils::concurrency::Mutex mutex_max_total_num_nodes;
  // Number of nodes in the "completed" trees.
  std::vector<int64_t> num_nodes_completed_trees(rf_config.num_trees(), -1);
  // Number of nodes in the accounted trees i.e. the first
  // "next_tree_idx_to_account" trees.
  int64_t total_num_nodes_accounted = 0;
  // Index of the next tree to account.
  int next_tree_idx_to_account = 0;
  int64_t model_size_in_bytes = mdl->AbstractAttributesSizeInBytes();

  // Note: "num_trained_trees" is defined outside of the following brackets so
  // to make use it is not released before "pool".
  std::atomic<int> num_trained_trees{0};
  {
    yggdrasil_decision_forests::utils::concurrency::ThreadPool pool(
        "TrainRF", deployment().num_threads());

    pool.StartWorkers();
    for (int tree_idx = 0; tree_idx < rf_config.num_trees(); tree_idx++) {
      pool.Schedule([&, tree_idx]() {
        // The user interrupted the training.
        if (stop_training_trigger_ != nullptr && *stop_training_trigger_) {
          if (!training_stopped_early) {
            training_stopped_early = true;
            LOG(INFO) << "Training interrupted per request";
          }
          return;
        }

        float bootstrap_size_ratio_factor = 1.f;
        if (adaptative_work) {
          bootstrap_size_ratio_factor =
              adaptative_work->OptimalApproximationFactor();
        }

        // Maximum training time.
        if (training_config().has_maximum_training_duration_seconds()) {
          // Stop the training if it lasted too long.
          if ((absl::Now() - begin_training) >
              absl::Seconds(
                  training_config().maximum_training_duration_seconds())) {
            if (!training_stopped_early) {
              training_stopped_early = true;
              LOG(INFO) << "Stop training because of the maximum training "
                           "duration.";
            }
            return;
          }
        }

        // Maximum model size.
        if (training_config().has_maximum_model_size_in_memory_in_bytes()) {
          utils::concurrency::MutexLock lock(&mutex_max_total_num_nodes);
          if (model_size_in_bytes >
              training_config().maximum_model_size_in_memory_in_bytes()) {
            return;
          }
        }

        if (rf_config.total_max_num_nodes() > 0) {
          utils::concurrency::MutexLock lock(&mutex_max_total_num_nodes);
          if (total_num_nodes_accounted > rf_config.total_max_num_nodes()) {
            // The num node limits is already exceeded.
            return;
          }
        }

        const auto begin_tree_training = absl::Now();

        utils::RandomEngine random(tree_seeds[tree_idx]);
        // Examples selected for the training.
        // Note: This in the inverse of the Out-of-bag (OOB) set.
        std::vector<row_t> selected_examples;
        auto& decision_tree = (*mdl->mutable_decision_trees())[tree_idx];
        if (rf_config.bootstrap_training_dataset()) {
          if (!rf_config.sampling_with_replacement() &&
              rf_config.bootstrap_size_ratio() == 1.f) {
            static bool already_shown = false;
            if (!already_shown) {
              already_shown = true;
              LOG(WARNING)
                  << "Example sampling without replacement "
                     "(sampling_with_replacement=false) with a sampling ratio "
                     "of 1 (bootstrap_size_ratio=1). All the examples "
                     "will be used for all the trees. You likely want to "
                     "reduce the sampling ratio e.g. bootstrap_size_ratio=0.5.";
            }
          }

          const auto num_samples = std::max(
              int64_t{1},
              static_cast<int64_t>(static_cast<double>(train_dataset.nrow()) *
                                   rf_config.bootstrap_size_ratio() *
                                   bootstrap_size_ratio_factor));
          internal::SampleTrainingExamples(
              train_dataset.nrow(), num_samples,
              rf_config.sampling_with_replacement(), &random,
              &selected_examples);
        } else {
          selected_examples.resize(train_dataset.nrow());
          std::iota(selected_examples.begin(), selected_examples.end(), 0);
        }

        // Timeout in the tree training.
        absl::optional<absl::Time> timeout;
        if (training_config().has_maximum_training_duration_seconds()) {
          timeout = begin_training +
                    absl::Seconds(
                        training_config().maximum_training_duration_seconds());
        }

        decision_tree::InternalTrainConfig internal_config;
        internal_config.preprocessing = &preprocessing;
        internal_config.timeout = timeout;
        CHECK_OK(decision_tree::Train(
            train_dataset, selected_examples, config_with_default, config_link,
            rf_config.decision_tree(), deployment(), weights, &random,
            decision_tree.get(), internal_config));

        if (training_config().has_maximum_model_size_in_memory_in_bytes()) {
          const auto tree_size_in_bytes =
              decision_tree->EstimateModelSizeInBytes();
          utils::concurrency::MutexLock lock(&mutex_max_total_num_nodes);
          model_size_in_bytes += tree_size_in_bytes;
          // Note: A model should contain at least one tree.
          if (num_trained_trees > 0 &&
              model_size_in_bytes >
                  training_config().maximum_model_size_in_memory_in_bytes()) {
            if (!training_stopped_early) {
              training_stopped_early = true;
              LOG(INFO)
                  << "Stop training after " << num_trained_trees
                  << " trees because the model size exceeded "
                     "maximum_model_size_in_memory_in_bytes="
                  << training_config().maximum_model_size_in_memory_in_bytes();
            }
            // Remove the tree that was just trained.
            decision_tree.reset();
            return;
          }
        }

        const auto current_num_trained_trees = ++num_trained_trees;

        if (rf_config.total_max_num_nodes() > 0) {
          utils::concurrency::MutexLock lock(&mutex_max_total_num_nodes);
          num_nodes_completed_trees[tree_idx] = decision_tree->NumNodes();
          while (next_tree_idx_to_account < num_nodes_completed_trees.size() &&
                 num_nodes_completed_trees[next_tree_idx_to_account] >= 0) {
            total_num_nodes_accounted +=
                num_nodes_completed_trees[next_tree_idx_to_account];
            next_tree_idx_to_account++;
          }
        }

        // Note: Since the batch size is only impacting the training time (i.e.
        // the oob computation), and since the adaptive work manager assumes a
        // linear relation between work and time, we only measure the duration
        // of the training step.
        //
        // Note: The OOB computation does not impact the quality of the model
        // (only the computation of model metrics). Disabling OOB computation
        // will make the work manager inference more accurate.
        if (adaptative_work) {
          adaptative_work->ReportTaskDone(
              bootstrap_size_ratio_factor,
              absl::ToDoubleSeconds(absl::Now() - begin_tree_training));
        }

        // OOB Metrics.
        if (compute_oob_performances) {
          utils::concurrency::MutexLock lock(&oob_metrics_mutex);
          // Update the prediction accumulator.
          internal::UpdateOOBPredictionsWithNewTree(
              train_dataset, config_with_default, selected_examples,
              rf_config.winner_take_all_inference(), *decision_tree, {},
              &random, &oob_predictions);

          // Evaluate the accumulated predictions.
          // Compute OOB if one of the condition is true:
          //   - This is the last tree of the model.
          //   - The last OOB was computed more than
          //     "oob_evaluation_interval_in_seconds" ago.
          //   - This last OOB was computed more than
          //     "oob_evaluation_interval_in_trees" trees ago.
          const bool compute_oob =
              ((absl::Now() - last_oob_computation_time) >=
               absl::Seconds(rf_config.oob_evaluation_interval_in_seconds())) ||
              (current_num_trained_trees == rf_config.num_trees()) ||
              ((current_num_trained_trees - last_oob_computation_num_trees) >=
               rf_config.oob_evaluation_interval_in_trees());

          if (compute_oob) {
            last_oob_computation_time = absl::Now();
            last_oob_computation_num_trees = current_num_trained_trees;
            proto::OutOfBagTrainingEvaluations evaluation;
            evaluation.set_number_of_trees(current_num_trained_trees);
            *evaluation.mutable_evaluation() = internal::EvaluateOOBPredictions(
                train_dataset, mdl->task(), mdl->label_col_idx(),
                mdl->uplift_treatment_col_idx(), mdl->weights(),
                oob_predictions,
                /*for_permutation_importance=*/false);
            mdl->mutable_out_of_bag_evaluations()->push_back(evaluation);

            // Print progress in the console.
            std::string snippet = absl::StrFormat("Training of tree  %d/%d",
                                                  current_num_trained_trees,
                                                  rf_config.num_trees());
            if (bootstrap_size_ratio_factor < 1.f) {
              absl::StrAppendFormat(&snippet, " work-factor:%f",
                                    bootstrap_size_ratio_factor);
            }
            if (training_config().has_maximum_model_size_in_memory_in_bytes()) {
              utils::concurrency::MutexLock lock2(&mutex_max_total_num_nodes);
              absl::StrAppendFormat(&snippet, " model-size:%d bytes",
                                    model_size_in_bytes);
            }
            absl::StrAppendFormat(
                &snippet, " (tree index:%d) done %s", tree_idx,
                internal::EvaluationSnippet(evaluation.evaluation()));
            LOG(INFO) << snippet;
          }

          // Variable importance.
          if (compute_oob_variable_importances) {
            for (const int feature_idx : config_link.features()) {
              for (int permutation_idx = 0;
                   permutation_idx <
                   rf_config.num_oob_variable_importances_permutations();
                   permutation_idx++) {
                internal::UpdateOOBPredictionsWithNewTree(
                    train_dataset, config_with_default, selected_examples,
                    rf_config.winner_take_all_inference(), *decision_tree,
                    feature_idx, &random,
                    &oob_predictions_per_input_features[feature_idx]);
              }
            }
          }
        } else {
          LOG_INFO_EVERY_N_SEC(
              20, _ << "Training of tree " << current_num_trained_trees << "/"
                    << rf_config.num_trees() << " (tree index:" << tree_idx
                    << ") done");
        }
      });
    }
  }

  if (training_stopped_early) {
    // Remove the non-trained trees.
    auto& trees = *mdl->mutable_decision_trees();
    trees.erase(
        std::remove_if(
            trees.begin(), trees.end(),
            [](const std::unique_ptr<decision_tree::DecisionTree>& tree)
                -> bool { return !tree || tree->mutable_root() == nullptr; }),
        trees.end());
  }

  if (rf_config.total_max_num_nodes() > 0 &&
      total_num_nodes_accounted > rf_config.total_max_num_nodes()) {
    // Keep the first trees such that the maximum number of nodes constraint is
    // satisfied.
    auto& trees = *mdl->mutable_decision_trees();
    int num_trees_to_keep = 0;
    int64_t num_nodes = 0;
    while (num_trees_to_keep < trees.size() &&
           num_nodes + num_nodes_completed_trees[num_trees_to_keep] <
               rf_config.total_max_num_nodes()) {
      num_nodes += num_nodes_completed_trees[num_trees_to_keep];
      num_trees_to_keep++;
    }
    if (num_trees_to_keep == 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The first tree alone exceeds the \"total_max_num_nodes\" "
          "parameter with ",
          num_nodes_completed_trees[0], ">", rf_config.total_max_num_nodes(),
          " nodes. Relax \"total_max_num_nodes\" or limit the "
          "growth of the tree (e.g. maximum depth)"));
    }
    trees.erase(trees.begin() + num_trees_to_keep, trees.end());
    LOG(INFO) << "Retaining the first " << num_trees_to_keep
              << " trees to satisfy the "
                 "\"total_max_num_nodes\" constraint.";
  }

  if (compute_oob_performances &&
      !mdl->mutable_out_of_bag_evaluations()->empty()) {
    LOG(INFO)
        << "Final OOB metrics: "
        << internal::EvaluationSnippet(
               mdl->mutable_out_of_bag_evaluations()->back().evaluation());
  }

  if (compute_oob_variable_importances) {
    ComputeVariableImportancesFromAccumulatedPredictions(
        oob_predictions, oob_predictions_per_input_features, train_dataset,
        mdl.get());
  }

  utils::usage::OnTrainingEnd(train_dataset.data_spec(), config_with_default,
                              config_link, train_dataset.nrow(), *mdl,
                              absl::Now() - begin_training);

  if (!rf_config.export_oob_prediction_path().empty()) {
    RETURN_IF_ERROR(ExportOOBPredictions(
        config_with_default, config_link, train_dataset.data_spec(),
        oob_predictions, rf_config.export_oob_prediction_path()));
  }

  // Cache the structural variable importance in the model data.
  RETURN_IF_ERROR(mdl->PrecomputeVariableImportances(
      mdl->AvailableStructuralVariableImportances()));

  decision_tree::SetLeafIndices(mdl->mutable_decision_trees());
  return std::move(mdl);
}

namespace internal {

void InitializeOOBPredictionAccumulators(
    const dataset::VerticalDataset::row_t num_predictions,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const dataset::proto::DataSpecification& data_spec,
    std::vector<PredictionAccumulator>* predictions) {
  predictions->resize(num_predictions);

  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION:
      for (auto& prediction : *predictions) {
        prediction.classification.SetNumClasses(
            config_link.num_label_classes());
      }
      break;

    case model::proto::Task::CATEGORICAL_UPLIFT:
      // Note: -2 because: The value 0 is reserved for the out-of-vocab item,
      // and there is one less predictions than treatments.
      for (auto& prediction : *predictions) {
        prediction.uplift.assign(
            data_spec.columns(config_link.uplift_treatment())
                    .categorical()
                    .number_of_unique_values() -
                2,
            0);
      }
      break;

    default:
      // Nothing to do.
      break;
  }
}

void UpdateOOBPredictionsWithNewTree(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    std::vector<row_t> sorted_non_oob_example_indices,
    const bool winner_take_all_inference,
    const decision_tree::DecisionTree& new_decision_tree,
    const absl::optional<int> shuffled_attribute_idx, utils::RandomEngine* rnd,
    std::vector<PredictionAccumulator>* oob_predictions) {
  // "next_non_oob_example_idx" is the index in "sorted_non_oob_example_indices"
  // of the example, with the smallest index which is greater or equal to the
  // index of the example being iterator on in the following "for loop".
  dataset::VerticalDataset::row_t next_non_oob_example_idx = 0;

  std::uniform_int_distribution<row_t> row_distribution(
      0, train_dataset.nrow() - 1);

  for (dataset::VerticalDataset::row_t example_idx = 0;
       example_idx < train_dataset.nrow(); example_idx++) {
    // Skip the example_idx in "sorted_non_oob_example_indices".
    while (next_non_oob_example_idx < sorted_non_oob_example_indices.size() &&
           sorted_non_oob_example_indices[next_non_oob_example_idx] <
               example_idx) {
      next_non_oob_example_idx++;
    }
    if (next_non_oob_example_idx < sorted_non_oob_example_indices.size() &&
        sorted_non_oob_example_indices[next_non_oob_example_idx] ==
            example_idx) {
      continue;
    }

    // Apply the decision tree.
    const decision_tree::proto::Node* leaf;
    if (shuffled_attribute_idx.has_value()) {
      const auto random_example_idx = row_distribution(*rnd);
      leaf = &new_decision_tree.GetLeafWithSwappedAttribute(
          train_dataset, example_idx, shuffled_attribute_idx.value(),
          random_example_idx);
    } else {
      leaf = &new_decision_tree.GetLeaf(train_dataset, example_idx);
    }

    // Accumulate the decision prediction to the oob accumulator.
    auto& accumulator = (*oob_predictions)[example_idx];
    accumulator.num_trees++;
    switch (config.task()) {
      case model::proto::Task::CLASSIFICATION:
        AddClassificationLeafToAccumulator(winner_take_all_inference, *leaf,
                                           &accumulator.classification);
        break;
      case model::proto::Task::REGRESSION:
        AddRegressionLeafToAccumulator(*leaf, &accumulator.regression);
        break;
      case model::proto::Task::RANKING:
        LOG(FATAL) << "OOB not implemented for Uplift.";
        break;
      case model::proto::Task::CATEGORICAL_UPLIFT:
        AddUpliftLeafToAccumulator(*leaf, &accumulator.uplift);
        break;
      default:
        LOG(WARNING) << "Not implemented";
    }
  }
}

metric::proto::EvaluationResults EvaluateOOBPredictions(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::Task task, const int label_col_idx,
    const int uplift_treatment_col_idx,
    const absl::optional<dataset::proto::LinkedWeightDefinition>& weight_links,
    const std::vector<PredictionAccumulator>& oob_predictions,
    const bool for_permutation_importance) {
  // Configure the evaluation options.
  metric::proto::EvaluationOptions eval_options;
  eval_options.set_task(task);
  // Disable the computation of expensive metrics that are not needed for the
  // monitoring of training.
  eval_options.set_bootstrapping_samples(-1);
  switch (task) {
    case model::proto::Task::CLASSIFICATION:
      eval_options.mutable_classification()->set_roc_enable(
          for_permutation_importance);
      break;
    case model::proto::Task::REGRESSION:
      eval_options.mutable_regression()->set_enable_regression_plots(false);
      break;
    case model::proto::Task::CATEGORICAL_UPLIFT:
    case model::proto::Task::NUMERICAL_UPLIFT:
      break;
    default:
      LOG(WARNING) << "Not implemented";
  }
  if (weight_links.has_value()) {
    // Note: The "weights" of "eval_options" won't be used, but "has_weights()"
    // needs to be true.
    eval_options.mutable_weights();
  }

  const auto& label_colum_spec =
      train_dataset.data_spec().columns(label_col_idx);
  utils::RandomEngine rnd;
  metric::proto::EvaluationResults evaluation;
  metric::InitializeEvaluation(eval_options, label_colum_spec, &evaluation);
  model::proto::Prediction prediction;

  for (dataset::VerticalDataset::row_t example_idx = 0;
       example_idx < train_dataset.nrow(); example_idx++) {
    auto& prediction_accumulator = oob_predictions[example_idx];
    if (prediction_accumulator.num_trees == 0) {
      // Not decision tree has been trained (yet) with this example in the oob
      // set i.e. all the trees have been trained using this example for
      // training.
      continue;
    }

    switch (task) {
      case model::proto::Task::CLASSIFICATION:
        FinalizeClassificationLeafToAccumulator(
            prediction_accumulator.classification, &prediction);
        break;
      case model::proto::Task::REGRESSION:
        prediction.mutable_regression()->set_value(
            prediction_accumulator.regression /
            prediction_accumulator.num_trees);
        break;
      case model::proto::Task::CATEGORICAL_UPLIFT:
        *prediction.mutable_uplift()->mutable_treatment_effect() = {
            prediction_accumulator.uplift.begin(),
            prediction_accumulator.uplift.end()};
        break;
      default:
        LOG(WARNING) << "Not implemented";
    }
    model::SetGroundTruth(
        train_dataset, example_idx,
        model::GroundTruthColumnIndices(label_col_idx, model::kNoRankingGroup,
                                        uplift_treatment_col_idx),
        eval_options.task(), &prediction);
    if (weight_links.has_value()) {
      prediction.set_weight(
          dataset::GetWeight(train_dataset, example_idx, weight_links.value()));
    }
    metric::AddPrediction(eval_options, prediction, &rnd, &evaluation);
  }
  metric::FinalizeEvaluation(eval_options, label_colum_spec, &evaluation);
  if (!for_permutation_importance &&
      evaluation.sampled_predictions_size() != 0) {
    LOG(WARNING) << "Internal error: Non empty oob evaluation";
    evaluation.clear_sampled_predictions();
  }
  return evaluation;
}

void ComputeVariableImportancesFromAccumulatedPredictions(
    const std::vector<internal::PredictionAccumulator>& oob_predictions,
    const std::vector<std::vector<internal::PredictionAccumulator>>&
        oob_predictions_per_input_features,
    const dataset::VerticalDataset& dataset, RandomForestModel* model) {
  // Note: "for_permutation_importance=true" allows to compute AUC, PR-AUC and
  // other expensive evaluation metrics.
  const auto base_evaluation = EvaluateOOBPredictions(
      dataset, model->task(), model->label_col_idx(),
      model->uplift_treatment_col_idx(), model->weights(), oob_predictions,
      /*for_permutation_importance=*/true);
  const auto permutation_evaluation = [&](const int feature_idx)
      -> absl::optional<metric::proto::EvaluationResults> {
    if (oob_predictions_per_input_features[feature_idx].empty()) {
      return {};
    }
    return EvaluateOOBPredictions(
        dataset, model->task(), model->label_col_idx(),
        model->uplift_treatment_col_idx(), model->weights(),
        oob_predictions_per_input_features[feature_idx],
        /*for_permutation_importance=*/true);
  };
  utils::ComputePermutationFeatureImportance(base_evaluation,
                                             permutation_evaluation, model);
}

void InitializeModelWithTrainingConfig(
    const model::proto::TrainingConfig& training_config,
    const model::proto::TrainingConfigLinking& training_config_linking,
    RandomForestModel* model) {
  InitializeModelWithAbstractTrainingConfig(training_config,
                                            training_config_linking, model);
  const auto& rf_config =
      training_config.GetExtension(random_forest::proto::random_forest_config);
  model->set_winner_take_all_inference(rf_config.winner_take_all_inference());
}

void SampleTrainingExamples(const row_t num_examples, const row_t num_samples,
                            const bool with_replacement,
                            utils::RandomEngine* random,
                            std::vector<row_t>* selected) {
  selected->resize(num_samples);

  if (with_replacement) {
    selected->resize(num_samples);
    // Sampling with replacement.
    std::uniform_int_distribution<row_t> example_idx_distrib(0,
                                                             num_examples - 1);
    for (row_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      (*selected)[sample_idx] = example_idx_distrib(*random);
    }
    std::sort(selected->begin(), selected->end());
  } else {
    selected->clear();
    selected->reserve(num_samples);
    // Sampling without replacement.
    std::uniform_real_distribution<float> dist_01;
    for (row_t example_idx = 0; example_idx < num_examples; example_idx++) {
      // The probability of selection is p/n where p is the remaining number of
      // items to sample, and n the remaining number of items to test.
      const float proba_select =
          static_cast<float>(num_samples - selected->size()) /
          (num_examples - example_idx);
      if (dist_01(*random) < proba_select) {
        selected->push_back(example_idx);
      }
    }
  }
}

absl::Status ExportOOBPredictions(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const dataset::proto::DataSpecification& dataspec,
    const std::vector<PredictionAccumulator>& oob_predictions,
    absl::string_view typed_path) {
  // Create the dataspec that describes the exported prediction dataset.
  dataset::proto::DataSpecification pred_dataspec;

  // Buffer example.
  dataset::proto::Example example;

  // Number of classification classes. Unused if the label is not categorical.
  int num_label_classes = -1;

  const auto& label_spec = dataspec.columns(config_link.label());
  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION: {
      num_label_classes = label_spec.categorical().number_of_unique_values();
      for (int i = 1 /*skip the OOV*/; i < num_label_classes; i++) {
        auto* col = pred_dataspec.add_columns();
        col->set_name(dataset::CategoricalIdxToRepresentation(label_spec, i));
        col->set_type(dataset::proto::ColumnType::NUMERICAL);
        example.add_attributes()->set_numerical(0);
      }
    } break;

    case model::proto::Task::REGRESSION: {
      auto* col = pred_dataspec.add_columns();
      col->set_name(label_spec.name());
      col->set_type(dataset::proto::ColumnType::NUMERICAL);
      example.add_attributes()->set_numerical(0);
    } break;

    case model::proto::Task::CATEGORICAL_UPLIFT: {
      num_label_classes = label_spec.categorical().number_of_unique_values();
      for (int i = 2 /*skip the OOV and treatement*/; i < num_label_classes;
           i++) {
        auto* col = pred_dataspec.add_columns();
        col->set_name(dataset::CategoricalIdxToRepresentation(label_spec, i));
        col->set_type(dataset::proto::ColumnType::NUMERICAL);
        example.add_attributes()->set_numerical(0);
      }
    } break;

    case model::proto::Task::NUMERICAL_UPLIFT: {
      auto* col = pred_dataspec.add_columns();
      col->set_name(label_spec.name());
      col->set_type(dataset::proto::ColumnType::NUMERICAL);
      example.add_attributes()->set_numerical(0);
    } break;

    default:
      return absl::InvalidArgumentError(
          "Exporting oob-predictions not supported for this task");
  }

  ASSIGN_OR_RETURN(auto writer,
                   dataset::CreateExampleWriter(typed_path, pred_dataspec));

  // Write the predictions one by one.
  for (const auto& pred : oob_predictions) {
    switch (config.task()) {
      case model::proto::Task::CLASSIFICATION:
        DCHECK_EQ(pred.classification.NumClasses(), num_label_classes);
        for (int i = 1 /*skip the OOV*/; i < num_label_classes; i++) {
          example.mutable_attributes(i - 1)->set_numerical(
              pred.classification.NumObservations() > 0
                  ? pred.classification.SafeProportionOrMinusInfinity(i)
                  : 0);
        }
        break;

      case model::proto::Task::REGRESSION:
        example.mutable_attributes(0)->set_numerical(pred.regression);
        break;

      case model::proto::Task::CATEGORICAL_UPLIFT:
        DCHECK_EQ(pred.uplift.size(), num_label_classes - 2);
        for (int i = 2; i < num_label_classes; i++) {
          example.mutable_attributes(i - 2)->set_numerical(pred.uplift[i - 2]);
        }
        break;

      case model::proto::Task::NUMERICAL_UPLIFT:
        DCHECK_EQ(pred.uplift.size(), 1);
        example.mutable_attributes(0)->set_numerical(pred.uplift[0]);
        break;

      default:
        return absl::InvalidArgumentError("Unsupported task");
    }
    RETURN_IF_ERROR(writer->Write(example));
  }

  return absl::OkStatus();
}

}  // namespace internal

}  // namespace random_forest
}  // namespace model
}  // namespace yggdrasil_decision_forests
