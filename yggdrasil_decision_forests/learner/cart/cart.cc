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

#include "yggdrasil_decision_forests/learner/cart/cart.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/weight.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/cart/cart.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/generic_parameters.h"
#include "yggdrasil_decision_forests/learner/decision_tree/label.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace cart {

constexpr char CartLearner::kRegisteredName[];
constexpr char CartLearner::kHParamValidationRatio[];

namespace {

// Generates the indices for the training and validation datasets.
void GenTrainAndValidIndices(const float validation_ratio,
                             const UnsignedExampleIdx num_examples,
                             std::vector<UnsignedExampleIdx>* train,
                             std::vector<UnsignedExampleIdx>* valid,
                             utils::RandomEngine* rnd) {
  std::uniform_real_distribution<float> unif_dist_01;
  for (UnsignedExampleIdx example_idx = 0; example_idx < num_examples;
       example_idx++) {
    const bool in_training = unif_dist_01(*rnd) > validation_ratio;
    (in_training ? train : valid)->push_back(example_idx);
  }
}

absl::Status SetDefaultHyperParameters(
    cart::proto::CartTrainingConfig* cart_config) {
  // The basic definition of CART does not have any attribute sampling.
  if (!cart_config->decision_tree().has_num_candidate_attributes() &&
      !cart_config->decision_tree().has_num_candidate_attributes_ratio()) {
    cart_config->mutable_decision_tree()->set_num_candidate_attributes(-1);
  }

  // There is no need for pre-sorting.
  cart_config->mutable_decision_tree()
      ->mutable_internal()
      ->set_sorting_strategy(
          decision_tree::proto::DecisionTreeTrainingConfig::Internal::IN_NODE);

  // Set the default generic decision tree hyper-parameter which might have not
  // be set by this CART specific function.
  decision_tree::SetDefaultHyperParameters(
      cart_config->mutable_decision_tree());

  return absl::OkStatus();
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

absl::StatusOr<model::proto::HyperParameterSpace>
CartLearner::PredefinedHyperParameterSpace() const {
  model::proto::HyperParameterSpace space;
  decision_tree::PredefinedHyperParameterAxisSplitSpace(&space);
  return space;
}

absl::StatusOr<model::proto::GenericHyperParameterSpecification>
CartLearner::GetGenericHyperParameterSpecification() const {
  ASSIGN_OR_RETURN(auto hparam_def,
                   AbstractLearner::GetGenericHyperParameterSpecification());

  hparam_def.mutable_documentation()->set_description(
      R"(A CART (Classification and Regression Trees) a decision tree. The non-leaf nodes contains conditions (also known as splits) while the leaf nodes contain prediction values. The training dataset is divided in two parts. The first is used to grow the tree while the second is used to prune the tree.)");

  model::proto::TrainingConfig config;
  const auto proto_path = "learner/cart/cart.proto";
  auto& cart_config = *config.MutableExtension(cart::proto::cart_config);
  RETURN_IF_ERROR(SetDefaultHyperParameters(&cart_config));

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamValidationRatio);
    param.mutable_real()->set_minimum(0);
    param.mutable_real()->set_maximum(1);
    param.mutable_real()->set_default_value(cart_config.validation_ratio());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Ratio of the training dataset used to create the validation dataset for pruning the tree. If set to 0, the entire dataset is used for training, and the tree is not pruned.)");
  }

  RETURN_IF_ERROR(decision_tree::GetGenericHyperParameterSpecification(
      cart_config.decision_tree(), &hparam_def));

  return hparam_def;
}

absl::StatusOr<std::unique_ptr<AbstractModel>> CartLearner::TrainWithStatusImpl(
    const dataset::VerticalDataset& train_dataset,
    std::optional<std::reference_wrapper<const dataset::VerticalDataset>>
        valid_dataset) const {
  const auto begin_training = absl::Now();

  if (training_config().task() != model::proto::Task::CLASSIFICATION &&
      training_config().task() != model::proto::Task::REGRESSION &&
      training_config().task() != model::proto::Task::CATEGORICAL_UPLIFT) {
    return absl::InvalidArgumentError(
        absl::StrCat("The CART learner does not support the task ",
                     model::proto::Task_Name(training_config().task()), "."));
  }

  // Assemble and check the training configuration.
  auto config = training_config();
  auto& cart_config = *config.MutableExtension(cart::proto::cart_config);

  RETURN_IF_ERROR(SetDefaultHyperParameters(&cart_config));

  model::proto::TrainingConfigLinking config_link;
  RETURN_IF_ERROR(AbstractLearner::LinkTrainingConfig(
      config, train_dataset.data_spec(), &config_link));
  decision_tree::SetInternalDefaultHyperParameters(
      config, config_link, train_dataset.data_spec(),
      cart_config.mutable_decision_tree());
  RETURN_IF_ERROR(CheckConfiguration(train_dataset.data_spec(), config,
                                     config_link, deployment()));

  // Initialize the model.
  auto mdl = std::make_unique<model::random_forest::RandomForestModel>();
  mdl->set_data_spec(train_dataset.data_spec());
  InitializeModelWithAbstractTrainingConfig(config, config_link, mdl.get());

  // Outputs probabilities.
  mdl->set_winner_take_all_inference(false);

  mdl->AddTree(std::make_unique<decision_tree::DecisionTree>());
  auto* decision_tree = mdl->mutable_decision_trees()->front().get();

  LOG(INFO) << "Training CART on " << train_dataset.nrow() << " example(s) and "
            << config_link.features().size() << " feature(s).";

  std::vector<float> weights;
  RETURN_IF_ERROR(dataset::GetWeights(train_dataset, config_link, &weights));
  utils::RandomEngine random(config.random_seed());

  // Select the example for training and for validation.
  // The validation dataset is used for tree pruning and for self reported
  // validation.

  // Note: "effective_valid_dataset" will be either null, point to
  // "valid_dataset" or point to "buffer_valid_dataset".
  dataset::VerticalDataset const* effective_valid_dataset = nullptr;
  dataset::VerticalDataset buffer_valid_dataset;
  std::vector<UnsignedExampleIdx> train_examples_in_train_ds;
  if (valid_dataset.has_value()) {
    // The user provided a validation dataset.
    effective_valid_dataset = &valid_dataset.value().get();
    // Uses all the training examples for growing the tree.
    train_examples_in_train_ds.resize(train_dataset.nrow());
    std::iota(train_examples_in_train_ds.begin(),
              train_examples_in_train_ds.end(), 0);
  } else {
    // Extract a validation dataset from the training dataset.
    std::vector<UnsignedExampleIdx> valid_examples_in_train_ds;
    GenTrainAndValidIndices(cart_config.validation_ratio(),
                            train_dataset.nrow(), &train_examples_in_train_ds,
                            &valid_examples_in_train_ds, &random);
    if (!valid_examples_in_train_ds.empty()) {
      ASSIGN_OR_RETURN(buffer_valid_dataset,
                       train_dataset.Extract(valid_examples_in_train_ds));
      effective_valid_dataset = &buffer_valid_dataset;
    } else {
      // No validation examples.
      effective_valid_dataset = nullptr;
    }
  }

  // Timeout in the tree training.
  std::optional<absl::Time> timeout;
  if (training_config().has_maximum_training_duration_seconds()) {
    timeout =
        begin_training +
        absl::Seconds(training_config().maximum_training_duration_seconds());
  }

  // Trains the tree.
  decision_tree::InternalTrainConfig internal_config;
  internal_config.timeout = timeout;
  RETURN_IF_ERROR(decision_tree::Train(
      train_dataset, train_examples_in_train_ds, config, config_link,
      cart_config.decision_tree(), deployment(), weights, &random,
      decision_tree, internal_config));

  if (effective_valid_dataset) {
    std::vector<UnsignedExampleIdx> valid_examples_in_valid_ds;
    valid_examples_in_valid_ds.resize(effective_valid_dataset->nrow());
    std::iota(valid_examples_in_valid_ds.begin(),
              valid_examples_in_valid_ds.end(), 0);

    std::vector<UnsignedExampleIdx> valid_examples_in_valid_ds_buffer;
    auto valid_examples_in_valid_ds_rb =
        decision_tree::SelectedExamplesRollingBuffer::Create(
            absl::MakeSpan(valid_examples_in_valid_ds),
            &valid_examples_in_valid_ds_buffer);

    // "valid_examples_in_valid_ds" always contains all the examples in
    // "effective_valid_dataset".
    DCHECK_EQ(effective_valid_dataset->nrow(),
              valid_examples_in_valid_ds.size());
    // Prune the tree.
    const auto num_nodes_pre_pruning = decision_tree->NumNodes();
    RETURN_IF_ERROR(internal::PruneTree(*effective_valid_dataset, weights,
                                        valid_examples_in_valid_ds_rb, config,
                                        config_link, decision_tree));
    mdl->set_num_pruned_nodes(num_nodes_pre_pruning -
                              decision_tree->NumNodes());

    // Evaluate model on the validation dataset.
    metric::proto::EvaluationOptions valid_evaluation_options;
    valid_evaluation_options.set_task(training_config().task());
    valid_evaluation_options.set_bootstrapping_samples(0);
    if (training_config().has_weight_definition()) {
      *valid_evaluation_options.mutable_weights() =
          training_config().weight_definition();
    }
    utils::RandomEngine rnd;
    ASSIGN_OR_RETURN(const auto valid_evaluation,
                     mdl->EvaluateWithStatus(*effective_valid_dataset,
                                             valid_evaluation_options, &rnd));

    // Store the validation evaluation in the RF OOB evaluation field.
    random_forest::proto::OutOfBagTrainingEvaluations oob_evaluation;
    oob_evaluation.set_number_of_trees(1);
    *oob_evaluation.mutable_evaluation() = valid_evaluation;
    mdl->mutable_out_of_bag_evaluations()->push_back(oob_evaluation);
  }

  // Cache the structural variable importance in the model data.
  RETURN_IF_ERROR(mdl->PrecomputeVariableImportances(
      mdl->AvailableStructuralVariableImportances()));

  decision_tree::SetLeafIndices(mdl->mutable_decision_trees());
  return std::move(mdl);
}

namespace internal {

// Prunes the node of a tree that would lead to an improvement of the "Score"
// computed by the "ScoreAccumulator".
//
// Template:
//   ScoreAccumulator: Class accumulating label+predictions and outputting a
//     score. Must implement a `.Add()` method to accumulate individual
//     examples, and a `.Score()` that returns the aggregated score.
//   Label: Representation of the label.
//   Prediction: Representation of the predictions.
//   Secondary: Representation of an optional secondary label. Must have
//     a default constructor, and is only passed to the ScoreAccumulator.Add
//     method.
//
// Args:
//   dataset: Validation dataset.
//   weights: Example weights.
//   labels: Example labels.
//   secondary_labels: Examples secondary labels. Empty if secondary labels are
//     not used. Secondary labels are an optional second column that store label
//     values. It is used in task where the label is stored on two columns (e.g.
//     ranking, uplifting). It is only up to "ScoreAccumulator" to use (or not)
//     secondary labels.
//   example_idxs: Indices of the examples to evaluate.
//   predictions: Example predictions.
//   node: Current node.
//
template <typename ScoreAccumulator, typename Label, typename Prediction,
          typename Secondary>
absl::Status PruneNode(
    const dataset::VerticalDataset& dataset, const std::vector<float>& weights,
    const std::vector<Label>& labels,
    const std::vector<Secondary>& secondary_labels,
    const decision_tree::SelectedExamplesRollingBuffer example_idxs,
    std::vector<Prediction>* predictions,
    model::decision_tree::NodeWithChildren* node) {
  if (node->IsLeaf()) {
    // Compute the predictions and return.
    // Leaf cannot be pruned "more".
    for (const auto example_idx : example_idxs.active) {
      (*predictions)[example_idx] = ScoreAccumulator::LeafToPrediction(node);
    }
    return absl::OkStatus();
  }

  // Note: This copy ensures the values are tested in the same order.
  // This is a bit less efficiently memory-wise, but make the computation of the
  // evaluation more local (i.e., faster).
  const auto save_example_idxs_order = example_idxs.active;

  // Maybe prune the children.
  ASSIGN_OR_RETURN(auto example_split,
                   decision_tree::internal::SplitExamplesInPlace(
                       dataset, example_idxs, node->node().condition(),
                       /*dataset_is_dense=*/false,
                       /*error_on_wrong_splitter_statistics=*/false,
                       /*examples_are_training_examples=*/false));

  RETURN_IF_ERROR((PruneNode<ScoreAccumulator, Label, Prediction, Secondary>(
      dataset, weights, labels, secondary_labels,
      example_split.positive_examples, predictions,
      node->mutable_pos_child())));

  RETURN_IF_ERROR((PruneNode<ScoreAccumulator, Label, Prediction, Secondary>(
      dataset, weights, labels, secondary_labels,
      example_split.negative_examples, predictions,
      node->mutable_neg_child())));

  // Compare the quality of the current node as a leaf or as a non-leaf.
  ScoreAccumulator score_as_leaf;
  ScoreAccumulator score_as_non_leaf;
  for (const auto example_idx : save_example_idxs_order) {
    Secondary secondary_label{};
    if (!secondary_labels.empty()) {
      secondary_label = secondary_labels[example_idx];
    }
    score_as_non_leaf.Add(labels[example_idx], secondary_label,
                          (*predictions)[example_idx], weights[example_idx]);
    score_as_leaf.Add(labels[example_idx], secondary_label,
                      ScoreAccumulator::LeafToPrediction(node),
                      weights[example_idx]);
  }

  if (score_as_leaf.Score() < score_as_non_leaf.Score()) {
    // The node is better as a non-leaf than as a leaf: Don't prune the node.
    return absl::OkStatus();
  }

  // Turn the node into a leaf and prune the children.
  node->TurnIntoLeaf();

  // Update the predictions with this node as a leaf.
  for (const auto example_idx : save_example_idxs_order) {
    (*predictions)[example_idx] = ScoreAccumulator::LeafToPrediction(node);
  }
  return absl::OkStatus();
}

absl::Status PruneTreeClassification(
    const dataset::VerticalDataset& dataset, const std::vector<float> weights,
    const decision_tree::SelectedExamplesRollingBuffer example_idxs,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    model::decision_tree::DecisionTree* tree) {
  const auto& labels =
      dataset
          .ColumnWithCastWithStatus<
              dataset::VerticalDataset::CategoricalColumn>(config_link.label())
          .value()
          ->values();

  class AccuracyAccumulator {
   public:
    static int32_t LeafToPrediction(
        model::decision_tree::NodeWithChildren* leaf) {
      return leaf->node().classifier().top_value();
    }

    void Add(const int32_t label, const bool ignored, const int32_t prediction,
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
  return PruneNode<AccuracyAccumulator, int32_t, int32_t, bool>(
      dataset, weights, labels, {}, example_idxs, &predictions,
      tree->mutable_root());
}

absl::Status PruneTreeRegression(
    const dataset::VerticalDataset& dataset, const std::vector<float> weights,
    const decision_tree::SelectedExamplesRollingBuffer example_idxs,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    model::decision_tree::DecisionTree* tree) {
  const auto& labels =
      dataset
          .ColumnWithCastWithStatus<dataset::VerticalDataset::NumericalColumn>(
              config_link.label())
          .value()
          ->values();

  class NegMSEAccumulator {
   public:
    static float LeafToPrediction(
        model::decision_tree::NodeWithChildren* leaf) {
      return leaf->node().regressor().top_value();
    }

    void Add(const float label, const bool ignored, const float prediction,
             const float weight) {
      sum_squared_error_ +=
          weight * (label - prediction) * (label - prediction);
      sum_weights_ += weight;
    }

    // -MSE.
    float Score() { return -sum_squared_error_ / sum_weights_; }

   private:
    double sum_squared_error_ = 0;
    double sum_weights_ = 0;
  };

  std::vector<float> predictions(dataset.nrow());
  return PruneNode<NegMSEAccumulator, float, float, bool>(
      dataset, weights, labels, {}, example_idxs, &predictions,
      tree->mutable_root());
}

absl::Status PruneTreeUpliftCategorical(
    const dataset::VerticalDataset& dataset, const std::vector<float> weights,
    const decision_tree::SelectedExamplesRollingBuffer example_idxs,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    model::decision_tree::DecisionTree* tree) {
  const auto& outcomes =
      dataset
          .ColumnWithCastWithStatus<
              dataset::VerticalDataset::CategoricalColumn>(config_link.label())
          .value()
          ->values();

  const auto& treatments = dataset
                               .ColumnWithCastWithStatus<
                                   dataset::VerticalDataset::CategoricalColumn>(
                                   config_link.uplift_treatment())
                               .value()
                               ->values();

  class UpliftAccumulator {
   public:
    typedef absl::InlinedVector<float, 2> Prediction;

    static Prediction LeafToPrediction(
        model::decision_tree::NodeWithChildren* leaf) {
      const auto& uplift = leaf->node().uplift();
      return {uplift.treatment_effect().begin(),
              uplift.treatment_effect().end()};
    }

    UpliftAccumulator() {
      options_.set_task(model::proto::Task::CATEGORICAL_UPLIFT);
      options_.mutable_weights();
      outcome_column_.set_type(dataset::proto::ColumnType::CATEGORICAL);
      outcome_column_.mutable_categorical()->set_number_of_unique_values(3);
      CHECK_OK(metric::InitializeEvaluation(options_, outcome_column_,
                                            &evaluation_));
    }

    void Add(const int32_t outcome, const int32_t treatment,
             const Prediction& prediction, const float weight) {
      model::proto::Prediction proto_pred;
      proto_pred.set_weight(weight);
      model::proto::Prediction::Uplift& uplift_pred =
          *proto_pred.mutable_uplift();
      uplift_pred.set_outcome_categorical(outcome);
      uplift_pred.set_treatment(treatment);
      *uplift_pred.mutable_treatment_effect() = {prediction.begin(),
                                                 prediction.end()};
      CHECK_OK(
          metric::AddPrediction(options_, proto_pred, &rnd_, &evaluation_));
    }

    float Score() {
      if (evaluation_.uplift().num_treatments() < 2) {
        // The leaf does not contain at least two treatments.
        return 0;
      }

      CHECK_OK(
          metric::FinalizeEvaluation(options_, outcome_column_, &evaluation_));
      return metric::AUUC(evaluation_);
    }

   private:
    metric::proto::EvaluationOptions options_;
    utils::RandomEngine rnd_;
    metric::proto::EvaluationResults evaluation_;
    dataset::proto::Column outcome_column_;
  };

  std::vector<typename UpliftAccumulator::Prediction> predictions(
      dataset.nrow());

  return PruneNode<UpliftAccumulator, int32_t,
                   typename UpliftAccumulator::Prediction, int32_t>(
      dataset, weights, outcomes, treatments, example_idxs, &predictions,
      tree->mutable_root());
}

absl::Status PruneTree(
    const dataset::VerticalDataset& dataset, const std::vector<float>& weights,
    const decision_tree::SelectedExamplesRollingBuffer example_idxs,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    model::decision_tree::DecisionTree* tree) {
  const auto num_nodes_pre_pruning = tree->NumNodes();

  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION:
      RETURN_IF_ERROR(PruneTreeClassification(dataset, weights, example_idxs,
                                              config, config_link, tree));
      break;
    case model::proto::Task::REGRESSION:
      RETURN_IF_ERROR(PruneTreeRegression(dataset, weights, example_idxs,
                                          config, config_link, tree));
      break;

    case model::proto::Task::CATEGORICAL_UPLIFT:
      RETURN_IF_ERROR(PruneTreeUpliftCategorical(dataset, weights, example_idxs,
                                                 config, config_link, tree));
      break;

    default:
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
