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

#include "yggdrasil_decision_forests/learner/cart/cart.h"

#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/weight.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/cart/cart.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/generic_parameters.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/utils/adaptive_work.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/usage.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace cart {

constexpr char CartLearner::kRegisteredName[];
constexpr char CartLearner::kHParamValidationRatio[];

namespace {
using row_t = dataset::VerticalDataset::row_t;

// Generates the indices for the training and validation datasets.
void GenTrainAndValidIndices(const float validation_ratio,
                             const row_t num_examples,
                             std::vector<row_t>* train,
                             std::vector<row_t>* valid,
                             utils::RandomEngine* rnd) {
  std::uniform_real_distribution<float> unif_dist_01;
  for (row_t example_idx = 0; example_idx < num_examples; example_idx++) {
    const bool in_training = unif_dist_01(*rnd) > validation_ratio;
    (in_training ? train : valid)->push_back(example_idx);
  }
}

}  // namespace

CartLearner::CartLearner(const model::proto::TrainingConfig& training_config)
    : AbstractLearner(training_config) {}

absl::Status CartLearner::SetHyperParametersImpl(
    utils::GenericHyperParameterConsumer* generic_hyper_params) {
  RETURN_IF_ERROR(
      AbstractLearner::SetHyperParametersImpl(generic_hyper_params));
  const auto& cart_config =
      training_config_.MutableExtension(cart::proto::cart_config);

  // Decision tree specific hyper-parameters.
  absl::flat_hash_set<std::string> consumed_hparams;
  RETURN_IF_ERROR(decision_tree::SetHyperParameters(
      &consumed_hparams, cart_config->mutable_decision_tree(),
      generic_hyper_params));

  {
    const auto hparam = generic_hyper_params->Get(kHParamValidationRatio);
    if (hparam.has_value()) {
      cart_config->set_validation_ratio(hparam.value().value().real());
    }
  }

  return absl::OkStatus();
}

utils::StatusOr<model::proto::HyperParameterSpace>
CartLearner::PredefinedHyperParameterSpace() const {
  model::proto::HyperParameterSpace space;
  decision_tree::PredefinedHyperParameterAxisSplitSpace(&space);
  return space;
}

utils::StatusOr<model::proto::GenericHyperParameterSpecification>
CartLearner::GetGenericHyperParameterSpecification() const {
  ASSIGN_OR_RETURN(auto hparam_def,
                   AbstractLearner::GetGenericHyperParameterSpecification());

  hparam_def.mutable_documentation()->set_description(
      R"(A CART (Classification and Regression Trees) a decision tree. The non-leaf nodes contains conditions (also known as splits) while the leaf nodes contains prediction values. The training dataset is divided in two parts. The first is used to grow the tree while the second is used to prune the tree.)");

  model::proto::TrainingConfig config;
  const auto proto_path = "learner/cart/cart.proto";
  const auto& cart_config = config.GetExtension(cart::proto::cart_config);

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamValidationRatio);
    param.mutable_real()->set_minimum(0);
    param.mutable_real()->set_maximum(1);
    param.mutable_real()->set_default_value(cart_config.validation_ratio());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Ratio of the training dataset used to create the validation dataset used to prune the tree.)");
  }

  RETURN_IF_ERROR(decision_tree::GetGenericHyperParameterSpecification(
      cart_config.decision_tree(), &hparam_def));

  return hparam_def;
}

utils::StatusOr<std::unique_ptr<AbstractModel>> CartLearner::TrainWithStatus(
    const dataset::VerticalDataset& train_dataset) const {
  const auto begin_training = absl::Now();

  if (training_config().task() != model::proto::Task::CLASSIFICATION &&
      training_config().task() != model::proto::Task::REGRESSION) {
    return absl::InvalidArgumentError(
        absl::StrCat("The CART learner does not support the task ",
                     model::proto::Task_Name(training_config().task()), "."));
  }

  // Assemble and check the training configuration.
  auto config = training_config();
  auto& cart_config = *config.MutableExtension(cart::proto::cart_config);
  decision_tree::SetDefaultHyperParameters(cart_config.mutable_decision_tree());
  // There is no need for pre-sorting.
  cart_config.mutable_decision_tree()->mutable_internal()->set_sorting_strategy(
      decision_tree::proto::DecisionTreeTrainingConfig::Internal::IN_NODE);
  model::proto::TrainingConfigLinking config_link;
  RETURN_IF_ERROR(AbstractLearner::LinkTrainingConfig(
      config, train_dataset.data_spec(), &config_link));
  RETURN_IF_ERROR(CheckConfiguration(train_dataset.data_spec(), config,
                                     config_link, deployment()));

  // Initialize the model.
  auto mdl = absl::make_unique<model::random_forest::RandomForestModel>();
  mdl->set_data_spec(train_dataset.data_spec());
  InitializeModelWithAbstractTrainingConfig(config, config_link, mdl.get());

  // Outputs probabilities.
  mdl->set_winner_take_all_inference(false);

  mdl->AddTree(absl::make_unique<decision_tree::DecisionTree>());
  auto* decision_tree = mdl->mutable_decision_trees()->front().get();

  LOG(INFO) << "Training CART on " << train_dataset.nrow() << " example(s) and "
            << config_link.features().size() << " feature(s).";
  utils::usage::OnTrainingStart(train_dataset.data_spec(), config, config_link,
                                train_dataset.nrow());

  std::vector<float> weights;
  RETURN_IF_ERROR(dataset::GetWeights(train_dataset, config_link, &weights));
  utils::RandomEngine random(config.random_seed());

  // Select the example for training and for pruning.
  std::vector<row_t> train_examples, valid_examples;
  GenTrainAndValidIndices(cart_config.validation_ratio(), train_dataset.nrow(),
                          &train_examples, &valid_examples, &random);

  // Timeout in the tree training.
  absl::optional<absl::Time> timeout;
  if (training_config().has_maximum_training_duration_seconds()) {
    timeout =
        begin_training +
        absl::Seconds(training_config().maximum_training_duration_seconds());
  }

  // Trains the tree.
  decision_tree::InternalTrainConfig internal_config;
  internal_config.timeout = timeout;
  RETURN_IF_ERROR(decision_tree::Train(train_dataset, train_examples, config,
                                       config_link, cart_config.decision_tree(),
                                       deployment(), weights, &random,
                                       decision_tree, internal_config));

  // Prune the tree.
  RETURN_IF_ERROR(internal::PruneTree(train_dataset, weights, valid_examples,
                                      config, config_link, decision_tree));

  utils::usage::OnTrainingEnd(train_dataset.data_spec(), config, config_link,
                              train_dataset.nrow(), *mdl,
                              absl::Now() - begin_training);

  // Cache the structural variable importance in the model data.
  RETURN_IF_ERROR(mdl->PrecomputeVariableImportances(
      mdl->AvailableStructuralVariableImportances()));

  return std::move(mdl);
}

namespace internal {

template <typename ScoreAccumulator, typename Label>
absl::Status PruneNode(const dataset::VerticalDataset& dataset,
                       const std::vector<float> weights,
                       const std::vector<Label>& labels,
                       const std::vector<row_t>& example_idxs,
                       std::vector<Label>* predictions,
                       model::decision_tree::NodeWithChildren* node) {
  if (node->IsLeaf()) {
    // Compute the predictions and return.
    // Leaf cannot be pruned "more".
    for (const auto& example_idx : example_idxs) {
      (*predictions)[example_idx] = node->node().classifier().top_value();
    }
    return absl::OkStatus();
  }

  // Maybe prune the children.
  std::vector<row_t> positive_examples, negative_examples;
  RETURN_IF_ERROR(decision_tree::internal::SplitExamples(
      dataset, example_idxs, node->node().condition(),
      /*dataset_is_dense=*/false, /*error_on_wrong_splitter_statistics=*/false,
      &positive_examples, &negative_examples,
      /*examples_are_training_examples=*/false));

  RETURN_IF_ERROR((PruneNode<ScoreAccumulator, Label>(
      dataset, weights, labels, positive_examples, predictions,
      node->mutable_pos_child())));
  positive_examples.clear();
  positive_examples.shrink_to_fit();

  RETURN_IF_ERROR((PruneNode<ScoreAccumulator, Label>(
      dataset, weights, labels, negative_examples, predictions,
      node->mutable_neg_child())));
  negative_examples.clear();
  negative_examples.shrink_to_fit();

  // Compare the quality of the current node as a leaf or as a non-leaf.
  ScoreAccumulator score_as_leaf;
  ScoreAccumulator score_as_non_leaf;
  for (const auto& example_idx : example_idxs) {
    score_as_non_leaf.Add(labels[example_idx], (*predictions)[example_idx],
                          weights[example_idx]);
    score_as_leaf.Add(labels[example_idx], score_as_leaf.LeafToPrediction(node),
                      weights[example_idx]);
  }

  if (score_as_leaf.Score() < score_as_non_leaf.Score()) {
    // The node is better as a non-leaf than as a leaf: Don't prune the node.
    return absl::OkStatus();
  }

  // Turn the node into a leaf and prune the children.
  node->TurnIntoLeaf();

  // Update the predictions with this node as a leaf.
  for (const auto& example_idx : example_idxs) {
    (*predictions)[example_idx] = node->node().classifier().top_value();
  }
  return absl::OkStatus();
}

absl::Status PruneTreeClassification(
    const dataset::VerticalDataset& dataset, const std::vector<float> weights,
    const std::vector<row_t>& example_idxs,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    model::decision_tree::DecisionTree* tree) {
  const auto& labels =
      dataset
          .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
              config_link.label())
          ->values();

  class AccuracyAccumulator {
   public:
    int32_t LeafToPrediction(model::decision_tree::NodeWithChildren* leaf) {
      return leaf->node().classifier().top_value();
    }

    void Add(const int32_t label, const int32_t prediction,
             const float weight) {
      good_predictions_ += weight * (label == prediction);
      predictions_ += weight;
    }

    // Accuracy.
    float Score() { return good_predictions_ / predictions_; }

   private:
    double good_predictions_ = 0;
    double predictions_ = 0;
  };

  std::vector<int32_t> predictions(dataset.nrow());
  return PruneNode<AccuracyAccumulator>(dataset, weights, labels, example_idxs,
                                        &predictions, tree->mutable_root());
}

absl::Status PruneTreeRegression(
    const dataset::VerticalDataset& dataset, const std::vector<float> weights,
    const std::vector<row_t>& example_idxs,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    model::decision_tree::DecisionTree* tree) {
  const auto& labels =
      dataset
          .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
              config_link.label())
          ->values();

  class NegMSEAccumulator {
   public:
    int32_t LeafToPrediction(model::decision_tree::NodeWithChildren* leaf) {
      return leaf->node().regressor().top_value();
    }

    void Add(const float label, const float prediction, const float weight) {
      sum_squared_error_ +=
          weight * (label - prediction) * (label - prediction);
      sum_weights_ += weight;
    }

    // -MSE.
    float Score() { return -sum_squared_error_ / sum_weights_; }

   private:
    double sum_squared_error_ = 0;
    double sum_weights_;
  };

  std::vector<float> predictions(dataset.nrow());
  return PruneNode<NegMSEAccumulator>(dataset, weights, labels, example_idxs,
                                      &predictions, tree->mutable_root());
}

absl::Status PruneTree(const dataset::VerticalDataset& dataset,
                       const std::vector<float> weights,
                       const std::vector<row_t>& example_idxs,
                       const model::proto::TrainingConfig& config,
                       const model::proto::TrainingConfigLinking& config_link,
                       model::decision_tree::DecisionTree* tree) {
  if (example_idxs.empty()) {
    LOG(WARNING) << "Validation set is empty, not pruning decision tree. This "
                    "will likely result in lower quality (and larger) trees "
                    "than usual, consider setting validation_set_ratio > 0.0.";
    return absl::OkStatus();
  }
  const auto num_nodes_pre_pruning = tree->NumNodes();
  if (config.task() == model::proto::Task::CLASSIFICATION) {
    RETURN_IF_ERROR(PruneTreeClassification(dataset, weights, example_idxs,
                                            config, config_link, tree));
  } else if (config.task() == model::proto::Task::REGRESSION) {
    RETURN_IF_ERROR(PruneTreeRegression(dataset, weights, example_idxs, config,
                                        config_link, tree));
  } else {
    return absl::UnimplementedError("Non supported task");
  }
  const auto num_nodes_post_pruning = tree->NumNodes();
  LOG(INFO) << num_nodes_pre_pruning << " nodes before pruning. "
            << num_nodes_post_pruning << " nodes after pruning.";
  return absl::OkStatus();
}

}  // namespace internal

}  // namespace cart
}  // namespace model
}  // namespace yggdrasil_decision_forests
