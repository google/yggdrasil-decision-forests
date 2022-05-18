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

// Random Forest learner.
//
// Note: "OOB" stands for "out of bag". The OOB examples of a tree (within a
// random forest) are the examples that are NOT used to trained this tree.
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_RANDOM_FOREST_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_RANDOM_FOREST_H_

#include <memory>
#include <random>
#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/weight.pb.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace random_forest {

class RandomForestLearner : public AbstractLearner {
 public:
  explicit RandomForestLearner(
      const model::proto::TrainingConfig& training_config);

  static constexpr char kRegisteredName[] = "RANDOM_FOREST";

  // Generic hyper parameter names.
  static constexpr char kHParamNumTrees[] = "num_trees";
  static constexpr char kHParamWinnerTakeAll[] = "winner_take_all";
  static constexpr char
      kHParamAdaptBootstrapSizeRatioForMaximumTrainingDuration[] =
          "adapt_bootstrap_size_ratio_for_maximum_training_duration";
  static constexpr char kHParamComputeOOBPerformances[] =
      "compute_oob_performances";
  static constexpr char kHParamComputeOOBVariableImportance[] =
      "compute_oob_variable_importances";

  static constexpr char kHParamBootstrapTrainingDataset[] =
      "bootstrap_training_dataset";
  static constexpr char kHParamBootstrapSizeRatio[] = "bootstrap_size_ratio";
  static constexpr char kHParamNumOOBVariableImportancePermutations[] =
      "num_oob_variable_importances_permutations";

  static constexpr char kHParamSamplingWithReplacement[] =
      "sampling_with_replacement";

  utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset = {}) const override;

  // Detects configuration errors and warnings.
  static absl::Status CheckConfiguration(
      const dataset::proto::DataSpecification& data_spec,
      const model::proto::TrainingConfig& config,
      const model::proto::TrainingConfigLinking& config_link,
      const proto::RandomForestTrainingConfig& rf_config,
      const model::proto::DeploymentConfig& deployment);

  absl::Status SetHyperParametersImpl(
      utils::GenericHyperParameterConsumer* generic_hyper_params) override;

  utils::StatusOr<model::proto::GenericHyperParameterSpecification>
  GetGenericHyperParameterSpecification() const override;

  utils::StatusOr<model::proto::HyperParameterSpace>
  PredefinedHyperParameterSpace() const override;

  std::vector<model::proto::PredefinedHyperParameterTemplate>
  PredefinedHyperParameters() const override;

  model::proto::LearnerCapabilities Capabilities() const override {
    model::proto::LearnerCapabilities capabilities;
    capabilities.set_support_max_training_duration(true);
    capabilities.set_support_max_model_size_in_memory(true);
    return capabilities;
  }
};

REGISTER_AbstractLearner(RandomForestLearner,
                         RandomForestLearner::kRegisteredName);

namespace internal {

void InitializeModelWithTrainingConfig(
    const model::proto::TrainingConfig& training_config,
    const model::proto::TrainingConfigLinking& training_config_linking,
    RandomForestModel* model);

// Accumulator of individual tree predictions. Can then be combined to compute
// the random forest predictions.
struct PredictionAccumulator {
  utils::IntegerDistribution<float> classification;
  double regression = 0;
  internal::UplifLeafAccumulator uplift;
  // Number of tree predictions being accumulated.
  int num_trees = 0;
};

// Initialize a vector of accumulators to support the task specified in
// "config*".
void InitializeOOBPredictionAccumulators(
    const dataset::VerticalDataset::row_t num_predictions,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const dataset::proto::DataSpecification& data_spec,
    std::vector<PredictionAccumulator>* predictions);

// Add the predictions of a decision tree to a set of predictor accumulators.
// The tree is applied only on the example indices NOT contained in
// "sorted_non_oob_example_indices".
//
// If "shuffled_attribute_idx" is set, the decision tree will be applied while
// simulating the random shuffling of the value of the attribute
// "shuffled_attribute_idx.value()" using "rnd" as source of randomness.
void UpdateOOBPredictionsWithNewTree(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    std::vector<dataset::VerticalDataset::row_t> sorted_non_oob_example_indices,
    const bool winner_take_all_inference,
    const decision_tree::DecisionTree& new_decision_tree,
    const absl::optional<int> shuffled_attribute_idx, utils::RandomEngine* rnd,
    std::vector<PredictionAccumulator>* oob_predictions);

// Evaluates the OOB predictions. Examples without any tree predictions are
// skipped.
metric::proto::EvaluationResults EvaluateOOBPredictions(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::Task task, const int label_col_idx,
    int uplift_treatment_col_idx,
    const absl::optional<dataset::proto::LinkedWeightDefinition>& weight_links,
    const std::vector<PredictionAccumulator>& oob_predictions,
    bool for_permutation_importance = false);

// Update the variable importance of the model with set of oob predictions.
void ComputeVariableImportancesFromAccumulatedPredictions(
    const std::vector<internal::PredictionAccumulator>& oob_predictions,
    const std::vector<std::vector<internal::PredictionAccumulator>>&
        oob_predictions_per_input_features,
    const dataset::VerticalDataset& dataset, RandomForestModel* model);

// Selects the examples to train one tree. Selects "num_samples" integers in [0,
// num_examples[ with replacement.
void SampleTrainingExamples(
    const dataset::VerticalDataset::row_t num_examples,
    const dataset::VerticalDataset::row_t num_samples,
    const bool with_replacement, utils::RandomEngine* random,
    std::vector<dataset::VerticalDataset::row_t>* selected);

// Exports the Out-of-bag predictions of a model to disk.
absl::Status ExportOOBPredictions(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const dataset::proto::DataSpecification& dataspec,
    const std::vector<PredictionAccumulator>& oob_predictions,
    absl::string_view typed_path);

}  // namespace internal

}  // namespace random_forest
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_RANDOM_FOREST_H_
