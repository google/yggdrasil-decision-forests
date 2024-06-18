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

#include "yggdrasil_decision_forests/learner/isolation_forest/isolation_forest.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/generic_parameters.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/isolation_forest/isolation_forest.pb.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/isolation_forest/isolation_forest.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests::model::isolation_forest {

namespace {

using ::yggdrasil_decision_forests::model::isolation_forest::internal::
    Configuration;
using ::yggdrasil_decision_forests::model::isolation_forest::internal::
    DefaultMaximumDepth;
using ::yggdrasil_decision_forests::model::isolation_forest::internal::
    GetNumExamplesPerTrees;

// Assembles and checks the configuration.
absl::StatusOr<internal::Configuration> BuildConfig(
    const IsolationForestLearner& learner,
    const dataset::proto::DataSpecification& data_spec,
    const UnsignedExampleIdx num_training_examples) {
  internal::Configuration config;

  config.training_config = learner.training_config();
  config.if_config = config.training_config.MutableExtension(
      isolation_forest::proto::isolation_forest_config);

  RETURN_IF_ERROR(AbstractLearner::LinkTrainingConfig(
      config.training_config, data_spec, &config.config_link));

  if (config.training_config.task() != model::proto::Task::ANOMALY_DETECTION) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The ISOLATION_FOREST learner does not support the task ",
        model::proto::Task_Name(config.training_config.task()), "."));
  }

  decision_tree::SetDefaultHyperParameters(
      config.if_config->mutable_decision_tree());

  if (!config.if_config->decision_tree().has_max_depth()) {
    const auto num_examples_per_trees =
        GetNumExamplesPerTrees(*config.if_config, num_training_examples);
    config.if_config->mutable_decision_tree()->set_max_depth(
        DefaultMaximumDepth(num_examples_per_trees));
  }

  if (!config.if_config->decision_tree().has_min_examples()) {
    config.if_config->mutable_decision_tree()->set_min_examples(1);
  }

  RETURN_IF_ERROR(learner.CheckConfiguration(data_spec, config.training_config,
                                             config.config_link,
                                             learner.deployment()));

  if (config.config_link.has_weight_definition()) {
    return absl::InvalidArgumentError(
        "Isolation forest does not support weights");
  }
  return config;
}

}  // namespace

namespace internal {

absl::StatusOr<bool> FindSplit(
    const Configuration& config, const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    decision_tree::NodeWithChildren* node, utils::RandomEngine* rnd) {
  DCHECK_GT(selected_examples.size(), 0);

  // Sample the order in which features are tested.
  // TODO: Use cache.
  std::vector<int> feature_order = {config.config_link.features().begin(),
                                    config.config_link.features().end()};
  std::shuffle(feature_order.begin(), feature_order.end(), *rnd);

  // Test features one after another.
  for (const auto& attribute_idx : feature_order) {
    const auto& col_spec = train_dataset.data_spec().columns(attribute_idx);
    if (col_spec.type() != dataset::proto::ColumnType::NUMERICAL) {
      // TODO: Add support for other types of features.
      continue;
    }

    const auto na_replacement = col_spec.numerical().mean();
    ASSIGN_OR_RETURN(
        const dataset::VerticalDataset::NumericalColumn* value_container,
        train_dataset.ColumnWithCastWithStatus<
            dataset::VerticalDataset::NumericalColumn>(attribute_idx));
    const auto& values = value_container->values();

    // Find minimum and maximum value.
    float min_value;
    float max_value;
    UnsignedExampleIdx num_valid_examples = 0;
    for (const auto example_idx : selected_examples) {
      const auto value = values[example_idx];
      if (std::isnan(value)) {
        continue;
      }
      if (num_valid_examples == 0 || value < min_value) {
        min_value = value;
      }
      if (num_valid_examples == 0 || value > max_value) {
        max_value = value;
      }
      num_valid_examples++;
    }

    if (num_valid_examples == 0 || max_value == min_value) {
      // Cannot split.
      continue;
    }

    // Randomly select a threshold in (min_value, max_value).
    const float threshold = std::uniform_real_distribution<float>(
        std::nextafter(min_value, std::numeric_limits<float>::max()),
        max_value)(*rnd);
    DCHECK_GT(threshold, min_value);
    DCHECK_LE(threshold, max_value);

    // Count the number of positive examples.
    UnsignedExampleIdx num_pos_examples = 0;
    for (const auto example_idx : selected_examples) {
      auto value = values[example_idx];
      if (std::isnan(value)) {
        value = na_replacement;
      }
      if (value >= threshold) {
        num_pos_examples++;
      }
    }

    DCHECK_GT(num_pos_examples, 0);
    DCHECK_LT(num_pos_examples, selected_examples.size());

    // Set split.
    auto* condition = node->mutable_node()->mutable_condition();
    condition->set_attribute(attribute_idx);
    condition->mutable_condition()->mutable_higher_condition()->set_threshold(
        threshold);
    condition->set_na_value(na_replacement >= threshold);
    condition->set_num_training_examples_without_weight(
        selected_examples.size());
    condition->set_num_pos_training_examples_without_weight(num_pos_examples);

    return true;
  }

  return false;  // No split found
}

// Grows recursively a node.
absl::Status GrowNode(const Configuration& config,
                      const dataset::VerticalDataset& train_dataset,
                      const std::vector<UnsignedExampleIdx>& selected_examples,
                      const int depth, decision_tree::NodeWithChildren* node,
                      utils::RandomEngine* rnd) {
  if (selected_examples.empty()) {
    return absl::InternalError("No examples fed to the node trainer");
  }

  const auto& dt_config = config.if_config->decision_tree();

  // Set node value
  node->mutable_node()->set_num_pos_training_examples_without_weight(
      selected_examples.size());
  node->mutable_node()
      ->mutable_anomaly_detection()
      ->set_num_examples_without_weight(selected_examples.size());

  // Stop growth
  if (selected_examples.size() < dt_config.min_examples() ||
      (dt_config.max_depth() >= 0 && depth >= dt_config.max_depth())) {
    node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
    return absl::OkStatus();
  }

  // Look for a split
  ASSIGN_OR_RETURN(
      const bool found_condition,
      FindSplit(config, train_dataset, selected_examples, node, rnd));

  if (!found_condition) {
    // No split found
    node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
    return absl::OkStatus();
  }

  // Turn the node into a non-leaf node
  STATUS_CHECK_EQ(
      selected_examples.size(),
      node->node().condition().num_training_examples_without_weight());
  node->CreateChildren();
  node->FinalizeAsNonLeaf(dt_config.keep_non_leaf_label_distribution(),
                          dt_config.store_detailed_label_distribution());

  // Branch examples to children
  // TODO: Use cache to avoid re-allocating selected example
  // buffers.
  std::vector<UnsignedExampleIdx> positive_examples;
  std::vector<UnsignedExampleIdx> negative_examples;
  RETURN_IF_ERROR(decision_tree::internal::SplitExamples(
      train_dataset, selected_examples, node->node().condition(), false,
      dt_config.internal_error_on_wrong_splitter_statistics(),
      &positive_examples, &negative_examples));

  // Split children
  RETURN_IF_ERROR(GrowNode(config, train_dataset, positive_examples, depth + 1,
                           node->mutable_pos_child(), rnd));
  positive_examples = {};  // Release memory of "positive_examples".
  RETURN_IF_ERROR(GrowNode(config, train_dataset, negative_examples, depth + 1,
                           node->mutable_neg_child(), rnd));
  return absl::OkStatus();
}

// Grows and return a tree.
absl::StatusOr<std::unique_ptr<decision_tree::DecisionTree>> GrowTree(
    const Configuration& config, const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    utils::RandomEngine* rnd) {
  auto tree = std::make_unique<decision_tree::DecisionTree>();
  tree->CreateRoot();
  RETURN_IF_ERROR(GrowNode(config, train_dataset, selected_examples,
                           /*depth=*/0, tree->mutable_root(), rnd));
  return std::move(tree);
}

int DefaultMaximumDepth(UnsignedExampleIdx num_examples_per_trees) {
  return std::ceil(std::log2(num_examples_per_trees));
}

std::vector<UnsignedExampleIdx> SampleExamples(
    const UnsignedExampleIdx num_examples,
    const UnsignedExampleIdx num_examples_to_sample, utils::RandomEngine* rnd) {
  std::vector<UnsignedExampleIdx> examples(num_examples);
  std::iota(examples.begin(), examples.end(), 0);
  std::shuffle(examples.begin(), examples.end(), *rnd);
  examples.resize(num_examples_to_sample);
  examples.shrink_to_fit();
  std::sort(examples.begin(), examples.end());
  return examples;
}

SignedExampleIdx GetNumExamplesPerTrees(
    const proto::IsolationForestTrainingConfig& if_config,
    const SignedExampleIdx num_training_examples) {
  switch (if_config.sampling_method_case()) {
    case proto::IsolationForestTrainingConfig::kSubsampleRatio:
      return static_cast<SignedExampleIdx>(
          std::ceil(static_cast<double>(if_config.subsample_ratio()) *
                    num_training_examples));
    default:
      return if_config.subsample_count();
  }
}

}  // namespace internal

IsolationForestLearner::IsolationForestLearner(
    const model::proto::TrainingConfig& training_config)
    : AbstractLearner(training_config) {}

absl::Status IsolationForestLearner::SetHyperParametersImpl(
    utils::GenericHyperParameterConsumer* generic_hyper_params) {
  RETURN_IF_ERROR(
      AbstractLearner::SetHyperParametersImpl(generic_hyper_params));
  const auto& if_config = training_config_.MutableExtension(
      isolation_forest::proto::isolation_forest_config);

  // Decision tree specific hyper-parameters.
  absl::flat_hash_set<std::string> consumed_hparams;
  RETURN_IF_ERROR(decision_tree::SetHyperParameters(
      &consumed_hparams, if_config->mutable_decision_tree(),
      generic_hyper_params));

  {
    const auto hparam = generic_hyper_params->Get(kHParamNumTrees);
    if (hparam.has_value()) {
      if_config->set_num_trees(hparam.value().value().integer());
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamSubsampleRatio);
    if (hparam.has_value()) {
      if_config->set_subsample_ratio(hparam.value().value().real());
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamSubsampleCount);
    if (hparam.has_value()) {
      if_config->set_subsample_count(hparam.value().value().integer());
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<model::proto::GenericHyperParameterSpecification>
IsolationForestLearner::GetGenericHyperParameterSpecification() const {
  ASSIGN_OR_RETURN(auto hparam_def,
                   AbstractLearner::GetGenericHyperParameterSpecification());
  model::proto::TrainingConfig config;
  const auto proto_path = "learner/isolation_forest/isolation_forest.proto";

  hparam_def.mutable_documentation()->set_description(
      R"(An Isolation Forest (https://ieeexplore.ieee.org/abstract/document/4781136) is a collection of decision trees trained without labels and independently to partition the feature space. The Isolation Forest prediction is an anomaly score that indicates whether an example originates from a same distribution to the training examples. We refer to Isolation Forest as both the original algorithm by Liu et al. and its extensions.)");

  const auto& if_config =
      config.GetExtension(isolation_forest::proto::isolation_forest_config);

  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamNumTrees);
    param.mutable_integer()->set_minimum(0);
    param.mutable_integer()->set_default_value(if_config.num_trees());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Number of individual decision trees. Increasing the number of trees can increase the quality of the model at the expense of size, training speed, and inference latency.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamSubsampleCount);
    param.mutable_integer()->set_minimum(0);
    param.mutable_integer()->set_default_value(if_config.num_trees());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Number of examples used to grow each tree. Only one of "subsample_ratio" and "subsample_count" can be set. If neither is set, "subsample_count" is assumed to be equal to 256. This is the default value recommended in the isolation forest paper.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamSubsampleRatio);
    param.mutable_integer()->set_minimum(0);
    param.mutable_integer()->set_default_value(if_config.num_trees());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Ratio of number of training examples used to grow each tree. Only one of "subsample_ratio" and "subsample_count" can be set. If neither is set, "subsample_count" is assumed to be equal to 256. This is the default value recommended in the isolation forest paper.)");
  }

  RETURN_IF_ERROR(decision_tree::GetGenericHyperParameterSpecification(
      if_config.decision_tree(), &hparam_def));
  return hparam_def;
}

absl::StatusOr<std::unique_ptr<AbstractModel>>
IsolationForestLearner::TrainWithStatusImpl(
    const dataset::VerticalDataset& train_dataset,
    absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
        valid_dataset) const {
  RETURN_IF_ERROR(dataset::CheckNumExamples(train_dataset.nrow()));

  ASSIGN_OR_RETURN(
      const internal::Configuration config,
      BuildConfig(*this, train_dataset.data_spec(), train_dataset.nrow()));

  auto model = absl::make_unique<IsolationForestModel>();
  InitializeModelWithAbstractTrainingConfig(config.training_config,
                                            config.config_link, model.get());
  model->set_data_spec(train_dataset.data_spec());
  model->set_num_examples_per_trees(
      GetNumExamplesPerTrees(*config.if_config, train_dataset.nrow()));

  YDF_LOG(INFO) << "Training isolation forest on " << train_dataset.nrow()
                << " example(s) and " << config.config_link.features_size()
                << " feature(s).";

  utils::RandomEngine global_random(config.training_config.random_seed());

  absl::Status global_status;
  utils::concurrency::Mutex global_mutex;
  {
    yggdrasil_decision_forests::utils::concurrency::ThreadPool pool(
        "TrainIF", deployment().num_threads());
    pool.StartWorkers();
    const auto num_trees = config.if_config->num_trees();
    model->mutable_decision_trees()->resize(num_trees);
    for (int tree_idx = 0; tree_idx < num_trees; tree_idx++) {
      pool.Schedule([&train_dataset, &model, &config, tree_idx, &global_status,
                     &global_mutex, seed = global_random()]() {
        {
          utils::concurrency::MutexLock lock(&global_mutex);
          if (!global_status.ok()) {
            return;
          }
        }
        utils::RandomEngine local_random(seed);
        const auto selected_examples = internal::SampleExamples(
            train_dataset.nrow(), model->num_examples_per_trees(),
            &local_random);
        auto tree_or =
            GrowTree(config, train_dataset, selected_examples, &local_random);
        if (!tree_or.ok()) {
          utils::concurrency::MutexLock lock(&global_mutex);
          global_status.Update(tree_or.status());
          return;
        }
        (*model->mutable_decision_trees())[tree_idx] = std::move(*tree_or);
      });
    }
  }
  decision_tree::SetLeafIndices(model->mutable_decision_trees());
  return std::move(model);
}

}  // namespace yggdrasil_decision_forests::model::isolation_forest
