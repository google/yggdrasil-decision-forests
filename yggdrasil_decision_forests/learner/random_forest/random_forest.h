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

// Random Forest learner.
//
// Note: "OOB" stands for "out of bag". The OOB examples of a tree (within a
// random forest) are the examples that are NOT used to trained this tree.
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_RANDOM_FOREST_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_RANDOM_FOREST_H_

#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
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
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

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

  absl::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatusImpl(
      const dataset::VerticalDataset& train_dataset,
      std::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset) const override;

  // Detects configuration errors and warnings.
  static absl::Status CheckConfiguration(
      const dataset::proto::DataSpecification& data_spec,
      const model::proto::TrainingConfig& config,
      const model::proto::TrainingConfigLinking& config_link,
      const proto::RandomForestTrainingConfig& rf_config,
      const model::proto::DeploymentConfig& deployment);

  absl::Status SetHyperParametersImpl(
      utils::GenericHyperParameterConsumer* generic_hyper_params) override;

  absl::StatusOr<model::proto::GenericHyperParameterSpecification>
  GetGenericHyperParameterSpecification() const override;

  absl::StatusOr<model::proto::HyperParameterSpace>
  PredefinedHyperParameterSpace() const override;

  std::vector<model::proto::PredefinedHyperParameterTemplate>
  PredefinedHyperParameters() const override;

  model::proto::LearnerCapabilities Capabilities() const override {
    model::proto::LearnerCapabilities capabilities;
    capabilities.set_support_max_training_duration(true);
    capabilities.set_support_max_model_size_in_memory(true);
    capabilities.set_support_return_in_bag_example_indices(true);
    return capabilities;
  }

  // Returns the indices of the in-bag examples for the ith tree without
  // performing the training.
  //
  // If the random seeds for the individual trees are not stored in the training
  // configuration, this function re-generates a random seed for each tree every
  // time it is called.
  absl::StatusOr<std::vector<UnsignedExampleIdx>> GetTrainingExampleIndices(
      UnsignedExampleIdx dataset_size, int tree_idx) const;
};

REGISTER_AbstractLearner(RandomForestLearner,
                         RandomForestLearner::kRegisteredName);

namespace internal {

void InitializeModelWithTrainingConfig(
    const model::proto::TrainingConfig& training_config,
    const model::proto::TrainingConfigLinking& training_config_linking,
    RandomForestModel* model);

// Coordinates worker thread execution and periodic evaluation during Random
// Forest training. Allows worker threads to accumulate tree predictions in
// parallel across stripes, while periodically admitting batches and draining
// in-flight workers so that a single elected worker can perform OOB metric
// evaluation exclusively without data races.
class EvaluationGate {
 public:
  struct Config {
    int num_trees = 0;
    int num_stripes = 1;
    int evaluation_interval_in_trees = 1;
    double evaluation_interval_in_seconds = 0.0;
  };

  // RAII Token representing an admitted worker inside the gate.
  // If destroyed before being passed to LeaveAndMaybeElectEvaluator (e.g. on
  // early return or error), automatically notifies the gate to decrement the
  // in-flight worker count and reopen the gate if needed.
  class [[nodiscard]] Token {
   public:
    Token(const Token&) = delete;
    Token& operator=(const Token&) = delete;

    Token(Token&& other) noexcept
        : gate_(other.gate_),
          start_stripe_(other.start_stripe_),
          active_(other.active_) {
      other.active_ = false;
    }

    Token& operator=(Token&& other) noexcept {
      if (this != &other) {
        Release();
        gate_ = other.gate_;
        start_stripe_ = other.start_stripe_;
        active_ = other.active_;
        other.active_ = false;
      }
      return *this;
    }

    ~Token() { Release(); }

    int start_stripe() const { return start_stripe_; }

   private:
    friend class EvaluationGate;

    Token(EvaluationGate* gate, int start_stripe)
        : gate_(gate), start_stripe_(start_stripe), active_(true) {}

    void Release() {
      if (active_ && gate_ != nullptr) {
        gate_->OnWorkerAbort();
        active_ = false;
      }
    }

    void Dismiss() { active_ = false; }

    EvaluationGate* gate_ = nullptr;
    int start_stripe_ = 0;
    bool active_ = false;
  };

  // RAII Ticket given to the elected evaluator worker.
  // If destroyed without calling Complete() (e.g. if evaluation fails),
  // automatically cleans up and reopens the gate.
  class [[nodiscard]] EvaluationTicket {
   public:
    EvaluationTicket(const EvaluationTicket&) = delete;
    EvaluationTicket& operator=(const EvaluationTicket&) = delete;

    EvaluationTicket(EvaluationTicket&& other) noexcept
        : gate_(other.gate_),
          eval_tree_count_(other.eval_tree_count_),
          completed_(other.completed_) {
      other.completed_ = true;
    }

    EvaluationTicket& operator=(EvaluationTicket&& other) noexcept {
      if (this != &other) {
        if (!completed_ && gate_ != nullptr) {
          gate_->OnWorkerAbort();
        }
        gate_ = other.gate_;
        eval_tree_count_ = other.eval_tree_count_;
        completed_ = other.completed_;
        other.completed_ = true;
      }
      return *this;
    }

    ~EvaluationTicket() {
      if (!completed_ && gate_ != nullptr) {
        gate_->OnWorkerAbort();
      }
    }

    void Complete() {
      if (!completed_ && gate_ != nullptr) {
        gate_->CompleteEvaluation();
        completed_ = true;
      }
    }

    int eval_tree_count() const { return eval_tree_count_; }

   private:
    friend class EvaluationGate;

    EvaluationTicket(EvaluationGate* gate, int eval_tree_count)
        : gate_(gate), eval_tree_count_(eval_tree_count), completed_(false) {}

    EvaluationGate* gate_ = nullptr;
    int eval_tree_count_ = 0;
    bool completed_ = true;
  };

  explicit EvaluationGate(Config config);

  // Waits until the gate is open, admits the calling worker, and returns an
  // RAII Token with the tree's starting stripe index.
  Token Enter();

  // Called when a worker finishes processing its tree.
  // Returns an EvaluationTicket if this worker is elected to run evaluation
  // (i.e. all in-flight workers in this batch have finished and evaluation
  // criteria are met). Otherwise returns std::nullopt.
  std::optional<EvaluationTicket> LeaveAndMaybeElectEvaluator(Token token);

 private:
  void OnWorkerAbort();
  void CompleteEvaluation();

  const Config config_;

  utils::concurrency::Mutex mutex_;
  utils::concurrency::CondVar cv_;

  bool gate_closed_ GUARDED_BY(mutex_) = false;
  int in_flight_workers_ GUARDED_BY(mutex_) = 0;
  int trees_since_last_eval_ GUARDED_BY(mutex_) = 0;
  int trees_admitted_ GUARDED_BY(mutex_) = 0;
  int trees_completed_ GUARDED_BY(mutex_) = 0;

  absl::Time last_evaluation_time_ GUARDED_BY(mutex_);
  int last_eval_tree_count_ GUARDED_BY(mutex_) = 0;
};

// Encapsulates Out-Of-Bag (OOB) prediction accumulation, variable importance
// calculation, and periodic metric evaluation during Random Forest training.
class OOBEvaluator {
 public:
  // Accumulator of individual tree predictions. Can then be combined to compute
  // the random forest predictions.
  struct PredictionAccumulator {
    utils::IntegerDistribution<float> classification;
    double regression = 0;
    internal::UplifLeafAccumulator uplift;
    // Number of tree predictions being accumulated.
    int num_trees = 0;
  };

  static void InitializeAccumulators(
      UnsignedExampleIdx num_predictions,
      const model::proto::TrainingConfig& config,
      const model::proto::TrainingConfigLinking& config_link,
      const dataset::proto::DataSpecification& data_spec,
      std::vector<PredictionAccumulator>* predictions);

  static absl::StatusOr<std::unique_ptr<OOBEvaluator>> Create(
      bool compute_oob_performances, bool compute_oob_variable_importances,
      const dataset::VerticalDataset& train_dataset,
      const model::proto::TrainingConfig& config,
      const model::proto::TrainingConfigLinking& config_link, int num_threads,
      RandomForestModel* model);

  // Determines the optimal number of stripes for concurrent OOB prediction
  // accumulation.
  //
  // Balances two considerations:
  // 1. Concurrency: Matches the maximum number of worker threads that can
  //    concurrently execute inside the evaluation gate (bounded by thread
  //    count, total trees, and the tree evaluation interval).
  // 2. Granularity: Ensures each stripe contains at least
  // `kMinExamplesPerStripe`
  //    (1024) examples to amortize mutex acquisition overhead.
  static int DetermineNumStripes(
      UnsignedExampleIdx num_examples,
      const random_forest::proto::RandomForestTrainingConfig& rf_config,
      int num_threads);

  // Called by a worker thread immediately after training a decision tree.
  // Updates OOB accumulators and checks if a periodic OOB evaluation should
  // be computed and recorded.
  absl::Status UpdateAndMaybeEvaluate(
      const dataset::VerticalDataset& train_dataset,
      const std::vector<UnsignedExampleIdx>& selected_examples,
      const decision_tree::DecisionTree& new_tree, utils::RandomEngine* random,
      absl::string_view extra_log_info = "");

  // Called after all trees have been trained to compute variable importances,
  // export predictions if requested, and log final OOB metrics.
  absl::Status FinalizeTraining(const dataset::VerticalDataset& train_dataset,
                                int num_threads);

 private:
  OOBEvaluator(
      bool compute_oob_performances, bool compute_oob_variable_importances,
      const model::proto::TrainingConfig& config,
      const model::proto::TrainingConfigLinking& config_link,
      const random_forest::proto::RandomForestTrainingConfig& rf_config,
      int num_stripes, UnsignedExampleIdx num_examples,
      EvaluationGate::Config gate_config, RandomForestModel* model);

  absl::Status ExportPredictions(
      const dataset::proto::DataSpecification& dataspec,
      absl::string_view typed_path) const;

  absl::Status UpdateAccumulators(
      const dataset::VerticalDataset& train_dataset,
      const std::vector<UnsignedExampleIdx>& selected_examples,
      const decision_tree::DecisionTree& new_tree, utils::RandomEngine* random,
      int local_start_stripe);

  absl::Status RunEvaluation(const dataset::VerticalDataset& train_dataset,
                             int eval_tree_count,
                             absl::string_view extra_log_info);

  const bool compute_oob_performances_;
  const bool compute_oob_variable_importances_;
  const model::proto::TrainingConfig& config_;
  const model::proto::TrainingConfigLinking& config_link_;
  const random_forest::proto::RandomForestTrainingConfig& rf_config_;
  RandomForestModel* model_;

  // Stripes configuration
  int num_stripes_;
  UnsignedExampleIdx stripe_size_;
  absl::FixedArray<utils::concurrency::Mutex> stripe_mutexes_;

  std::vector<PredictionAccumulator> oob_predictions_;
  std::vector<std::vector<PredictionAccumulator>>
      oob_predictions_per_input_features_;

  EvaluationGate evaluation_gate_;
};

// Add the predictions of a decision tree to a set of predictor accumulators.
// The tree is applied only on the example indices NOT contained in
// "sorted_non_oob_example_indices".
//
// If "shuffled_attribute_idx" is set, the decision tree will be applied while
// simulating the random shuffling of the value of the attribute
// "shuffled_attribute_idx.value()" using "rnd" as source of randomness.
absl::Status UpdateOOBPredictionsWithNewTree(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const std::vector<UnsignedExampleIdx>& sorted_non_oob_example_indices,
    const bool winner_take_all_inference,
    const decision_tree::DecisionTree& new_decision_tree,
    const std::optional<int> shuffled_attribute_idx,
    UnsignedExampleIdx begin_example_idx, UnsignedExampleIdx end_example_idx,
    utils::RandomEngine* rnd,
    std::vector<OOBEvaluator::PredictionAccumulator>* oob_predictions);

// Evaluates the OOB predictions. Examples without any tree predictions are
// skipped.
absl::StatusOr<metric::proto::EvaluationResults> EvaluateOOBPredictions(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::Task task, const int label_col_idx,
    int uplift_treatment_col_idx,
    const std::optional<dataset::proto::LinkedWeightDefinition>& weight_links,
    const std::vector<OOBEvaluator::PredictionAccumulator>& oob_predictions,
    bool for_permutation_importance = false);

// Update the variable importance of the model with set of oob predictions.
absl::Status ComputeVariableImportancesFromAccumulatedPredictions(
    const std::vector<OOBEvaluator::PredictionAccumulator>& oob_predictions,
    const std::vector<std::vector<OOBEvaluator::PredictionAccumulator>>&
        oob_predictions_per_input_features,
    const dataset::VerticalDataset& dataset, const int num_threads,
    RandomForestModel* model);

// Randomly samples a list of training examples to use for training a single
// decision tree. The number of sampled examples is determined from the Random
// Forest Training Config. If `bootstrap_size_ratio_factor` is provided, the
// number of sampled examples is further scaled by this factor, provided that
// this is enabled in the training configuration.
absl::Status SampleTrainingExamples(
    UnsignedExampleIdx num_examples,
    const proto::RandomForestTrainingConfig& rf_config,
    std::optional<double> bootstrap_size_ratio_factor,
    utils::RandomEngine* random, std::vector<UnsignedExampleIdx>* selected);

absl::Status SetDefaultHyperParameters(
    random_forest::proto::RandomForestTrainingConfig* rf_config);

}  // namespace internal

}  // namespace random_forest
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_RANDOM_FOREST_H_
