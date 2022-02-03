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

// Implementation of Gradient Boosted Trees (one of the popular versions of
// Gradient Boosted Model).
//
// A GBT model is a set of shallow decision trees whose predictions are computed
// by summing the output of each tree. The space of the summation depends on the
// model e.g. logit space is common for binary classification.
//
// The trees are trained sequentially. Each tree is trained to predict and then
// "correct" for the errors of the previously trained trees. More precisely,
// each tree is trained to predict the gradient of a loss function relative to
// each training example. The prediction of the tree is then set to a small
// negative value (called the shrinking) times the predicted gradient.
//
// For more details:
// https://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting
//
// Uses the CART implementation of SimpleML i.e. this implementations supports
// numerical, categorical and categorical set input features.
//
// This implementation supports Classification (binary and multi-classes),
// regression and ranking.
//
// Use the loss functions and implementation details detailed in:
//   https://statweb.stanford.edu/~jhf/ftp/trebst.pdf
//   http://www.saedsayad.com/docs/gbm2.pdf
//   https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_H_

#include <memory>
#include <random>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/loss/loss_library.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

namespace internal {
class AllTrainingConfiguration;
};

// A GBT learner i.e. takes as input a dataset and outputs a GBT model.
// See the file header for a description of the GBT learning algorithm/model.
class GradientBoostedTreesLearner : public AbstractLearner {
 public:
  explicit GradientBoostedTreesLearner(
      const model::proto::TrainingConfig& training_config);

  static constexpr char kRegisteredName[] = "GRADIENT_BOOSTED_TREES";

  // Generic hyper parameter names.
  static constexpr char kHParamNumTrees[] = "num_trees";
  static constexpr char kHParamShrinkage[] = "shrinkage";
  static constexpr char kHParamL1Regularization[] = "l1_regularization";
  static constexpr char kHParamL2Regularization[] = "l2_regularization";
  static constexpr char kHParamL2CategoricalRegularization[] =
      "l2_categorical_regularization";
  static constexpr char kHParamLambdaLoss[] = "lambda_loss";
  static constexpr char kHParamDartDropOut[] = "dart_dropout";
  static constexpr char kHParamAdaptSubsampleForMaximumTrainingDuration[] =
      "adapt_subsample_for_maximum_training_duration";
  static constexpr char kHParamUseHessianGain[] = "use_hessian_gain";
  static constexpr char kHParamSamplingMethod[] = "sampling_method";
  static constexpr char kSamplingMethodNone[] = "NONE";
  static constexpr char kSamplingMethodRandom[] = "RANDOM";
  static constexpr char kSamplingMethodGOSS[] = "GOSS";
  static constexpr char kSamplingMethodSelGB[] = "SELGB";
  static constexpr char kHParamGossAlpha[] = "goss_alpha";
  static constexpr char kHParamGossBeta[] = "goss_beta";
  static constexpr char kHParamSelGBRatio[] =
      "selective_gradient_boosting_ratio";
  static constexpr char kHParamSubsample[] = "subsample";

  static constexpr char kHParamForestExtraction[] = "forest_extraction";
  static constexpr char kHParamForestExtractionMart[] = "MART";
  static constexpr char kHParamForestExtractionDart[] = "DART";

  static constexpr char kHParamValidationSetRatio[] = "validation_ratio";
  static constexpr char kHParamEarlyStopping[] = "early_stopping";
  static constexpr char kHParamEarlyStoppingNone[] = "NONE";
  static constexpr char kHParamEarlyStoppingMinLossFullModel[] =
      "MIN_LOSS_FINAL";
  static constexpr char kHParamEarlyStoppingLossIncrease[] = "LOSS_INCREASE";
  static constexpr char kHParamEarlyStoppingNumTreesLookAhead[] =
      "early_stopping_num_trees_look_ahead";
  static constexpr char kHParamApplyLinkFunction[] = "apply_link_function";
  static constexpr char kHParamComputePermutationVariableImportance[] =
      "compute_permutation_variable_importance";
  static constexpr char kHParamValidationIntervalInTrees[] =
      "validation_interval_in_trees";
  static constexpr char kHParamLoss[] = "loss";
  static constexpr char kHParamFocalLossGamma[] =
      "focal_loss_gamma";
  static constexpr char kHParamFocalLossAlpha[] =
      "focal_loss_alpha";

  utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset = {}) const override;

  utils::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatus(
      const absl::string_view typed_path,
      const dataset::proto::DataSpecification& data_spec,
      const absl::optional<std::string>& typed_valid_path = {}) const override;

  // Detects configuration errors and warnings.
  static absl::Status CheckConfiguration(
      const dataset::proto::DataSpecification& data_spec,
      const model::proto::TrainingConfig& config,
      const model::proto::TrainingConfigLinking& config_link,
      const proto::GradientBoostedTreesTrainingConfig& gbt_config,
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
    capabilities.set_resume_training(true);
    capabilities.set_support_validation_dataset(true);
    return capabilities;
  }

 private:
  // Generates, checks and groups all the configuration objects.
  absl::Status BuildAllTrainingConfiguration(
      const dataset::proto::DataSpecification& data_spec,
      internal::AllTrainingConfiguration* all_config) const;

  // Initializes and returns a model.
  std::unique_ptr<GradientBoostedTreesModel> InitializeModel(
      const internal::AllTrainingConfiguration& config,
      const dataset::proto::DataSpecification& data_spec) const;

  // Training with dataset sampling using shards.
  utils::StatusOr<std::unique_ptr<AbstractModel>> ShardedSamplingTrain(
      const absl::string_view typed_path,
      const dataset::proto::DataSpecification& data_spec,
      const absl::optional<std::string>& typed_valid_path) const;
};

REGISTER_AbstractLearner(GradientBoostedTreesLearner,
                         GradientBoostedTreesLearner::kRegisteredName);

namespace internal {

// Divide "dataset" into a training and a validation dataset.
//
// If "validation_set_ratio==0", "train" becomes a shallow copy of "dataset" and
// no data is copied.
//
// If "group_column_idx" is -1, the sampling is uniform. Otherwise, the split
// between train and validation ensures that all given "group_column_idx" values
// are either in the train or in the validation dataset. In this case, the
// sampling is a greedy heuristic that tries to optimize the balance.
absl::Status ExtractValidationDataset(const dataset::VerticalDataset& dataset,
                                      float validation_set_ratio,
                                      int group_column_idx,
                                      dataset::VerticalDataset* train,
                                      dataset::VerticalDataset* validation,
                                      utils::RandomEngine* random);

// Initialize the gradient dataset "gradient_dataset" used to train individual
// trees. This dataset contains a shallow copy of "dataset" and a new column
// for each gradient dimension (named with "GradientColumnName").
//
// If "hessian_splits=true", a column containing the hessian is also created
// with a name generated by "HessianColumnName".
//
// "gradients" is initialized with the name of the gradient columns as well as a
// non-owning pointer to the gradient data in "gradient_dataset".
//
// "predictions" is initialized to contain the predictions.
absl::Status CreateGradientDataset(const dataset::VerticalDataset& dataset,
                                   int label_col_idx, bool hessian_splits,
                                   const AbstractLoss& loss_impl,
                                   dataset::VerticalDataset* gradient_dataset,
                                   std::vector<GradientData>* gradients,
                                   std::vector<float>* predictions);

// Copy the initial model predictions to the accumulator of predictions.
template <typename V>
void SetInitialPredictions(const std::vector<float>& initial_predictions,
                           dataset::VerticalDataset::row_t num_rows,
                           std::vector<V>* predictions);

// Computes the predictions and gradient of the model without relying on
// existing predictions or gradient buffers.
//
// Only the meta-data are used from "mdl". If "optional_engine" is non-null, it
// will be used in conjunction with "trees".
absl::Status ComputePredictions(
    const GradientBoostedTreesModel* mdl,
    const serving::FastEngine* optional_engine,
    const std::vector<decision_tree::DecisionTree*>& trees,
    const internal::AllTrainingConfiguration& config,
    const dataset::VerticalDataset& gradient_dataset,
    std::vector<float>* predictions);

// Sample (without replacement) a set of example indices.
void SampleTrainingExamples(
    dataset::VerticalDataset::row_t num_rows, float sample,
    utils::RandomEngine* random,
    std::vector<dataset::VerticalDataset::row_t>* selected_examples);

// Sample a set of example indices using the GOSS algorithm.
void SampleTrainingExamplesWithGoss(
    const std::vector<GradientData>& gradients,
    dataset::VerticalDataset::row_t num_rows, float alpha, float beta,
    utils::RandomEngine* random,
    std::vector<dataset::VerticalDataset::row_t>* selected_examples,
    std::vector<float>* weights);

// Sample a set of example indices using the Selective Gradient Boosting
// algorithm. The algorithm always selects all positive examples, but selects
// only those negative training examples that are more difficult (i.e., those
// with larger scores).
absl::Status SampleTrainingExamplesWithSelGB(
    model::proto::Task task, dataset::VerticalDataset::row_t num_rows,
    const RankingGroupsIndices* ranking_index,
    const std::vector<float>& predictions, float ratio,
    std::vector<dataset::VerticalDataset::row_t>* selected_examples);

// Export the training logs. Creates:
// - A static plot (.svg) of the training/validation loss/secondary metric
//   according to the number of trees.
// - An interactive plot of the same type.
absl::Status ExportTrainingLogs(const proto::TrainingLogs& training_logs,
                                absl::string_view directory);

void InitializeModelWithTrainingConfig(
    const model::proto::TrainingConfig& training_config,
    const model::proto::TrainingConfigLinking& training_config_linking,
    GradientBoostedTreesModel* model);

// Accumulator of predictions for individual trees in the Dart algorithm.
// Supports the operations required by the Dart training (e.g. extracting
// predictions from a large number of randomly selected trees).
class DartPredictionAccumulator {
 public:
  // Initialize the prediction accumulator. This function must be called before
  // any other function. After this call, the accumulator represents a GBDT
  // model without any tree.
  void Initialize(const std::vector<float>& initial_predictions,
                  dataset::VerticalDataset::row_t num_rows);

  // Sample a set of iteration indices (Note: one iteration = one tree for
  // regression or binary classification) to be EXCLUDED (i.e. dropped) for the
  // computation of the gradient of the next iteration. "dropout=0" means only
  // one iteration will be excluded (i.e. the algorithm become almost equivalent
  // to a GBDT; modulo that at least one tree should be sampled), "dropout=1"
  // means maximum dropout i.e. all the iterations will be excluded i.e. the
  // algorithm is similar to a Random Forest (modulo the loss).
  std::vector<int> SampleIterIndices(float dropout,
                                     utils::RandomEngine* random) const;

  // Gets the predictions from the current model but without the iterations
  // specified in "dropout_iter_idxs". The predictions vector should already be
  // initialized.
  absl::Status GetSampledPredictions(const std::vector<int>& dropout_iter_idxs,
                                     std::vector<float>* predictions);

  // Gets the predictions from the current model i.e. from all the iterations.
  // The predictions vector should already be initialized.
  //
  // Equivalent to "GetSampledPredictions" with "dropout_iter_idxs" empty.
  absl::Status GetAllPredictions(std::vector<float>* predictions);

  // Updates the accumulator with a set of trees obtained thought a single
  // iteration.
  absl::Status UpdateWithNewIteration(
      const std::vector<int>& selected_iter_idxs, proto::Loss loss,
      const AbstractLoss& loss_impl,
      const std::vector<std::unique_ptr<decision_tree::DecisionTree>>&
          new_trees,
      const dataset::VerticalDataset& gradient_dataset,
      int num_gradient_dimensions, double* mean_abs_prediction = nullptr);

  // Returns the optimal multiplicative factor for each tree in the order
  // provided to "UpdateWithNewIteration". For example, if the model contains
  // two tree t_1 and t_2, and the multiplicative factors are f_1 and f_2, the
  // final predictions of the model is:  Activation(f_1 * t_1 + f_2 * t_2).
  std::vector<float> TreeOutputScaling() const;

 private:
  struct TreePredictions {
    // Weights over all the predictions.
    float weight;

    // Predictions of the tree before weighing.
    std::vector<float> predictions;
  };

  // Predictions of all the trees summed and weighed i.e. current predictions of
  // the model.
  //
  // Note:
  //   predictions_[i] = \sum_j prediction_per_tree_[j].predictions[i] *
  //   prediction_per_tree_[j].weights + initial_prediction.
  std::vector<float> predictions_;

  // Predictions of individual trees.
  std::vector<TreePredictions> prediction_per_tree_;
};

// All the configuration object derived from the user training configuration.
struct AllTrainingConfiguration {
  AllTrainingConfiguration() {}
  AllTrainingConfiguration(const AllTrainingConfiguration&) = delete;
  AllTrainingConfiguration& operator=(const AllTrainingConfiguration&) = delete;

  // The effective training configuration i.e. the user training configuration
  // with set default values and generic hyper-parameters.
  model::proto::TrainingConfig train_config;

  // Similar to "train_config_", but with feature name replaced by feature
  // indices.
  model::proto::TrainingConfigLinking train_config_link;

  // Configuration specific to the gbt in "train_config_".
  const proto::GradientBoostedTreesTrainingConfig* gbt_config;

  // Implementation of the loss. Have non-owning dependencies to
  // "train_config_"'s content.
  std::unique_ptr<AbstractLoss> loss;

  // Grouping column for the train/validation dataset split. A value of "-1"
  // indicates that the split is unconstrained.
  //
  // Note: Ranking problem with RMSE loss is not grouped.
  int effective_validation_set_group = -1;
};

// All the training dataset information from the point of view of the weak
// learner i.e. "classical" training dataset + gradient + weights.
struct CompleteTrainingDatasetForWeakLearner {
  dataset::VerticalDataset dataset;

  // Shallow copy of the training dataset with the gradient as label.
  dataset::VerticalDataset gradient_dataset;

  // Gradient data in "dataset".
  std::vector<GradientData> gradients;

  // Training weights.
  std::vector<float> weights;

  // Predictions of the model.
  std::vector<float> predictions;

  // Number of trees used to compute "predictions".
  int predictions_from_num_trees = 0;
};

// Loads a dataset for a weak learner.
utils::StatusOr<std::unique_ptr<CompleteTrainingDatasetForWeakLearner>>
LoadCompleteDatasetForWeakLearner(
    const std::vector<std::string>& shards,
    const absl::string_view format_prefix,
    const dataset::proto::DataSpecification& data_spec,
    const AllTrainingConfiguration& config, const bool allocate_gradient,
    const GradientBoostedTreesModel* mdl);

// Early stopping controller.
class EarlyStopping {
 public:
  EarlyStopping(const int early_stopping_num_trees_look_ahead)
      : num_trees_look_ahead_(early_stopping_num_trees_look_ahead) {}

  // Updates the internal state of the early stopping controller.
  //
  // "set_trees_per_iterations" should be called before the first update.
  absl::Status Update(const float validation_loss,
                      const std::vector<float>& validation_secondary_metrics,
                      const int num_trees);

  // Should the training stop?
  bool ShouldStop();

  // Best model.
  int best_num_trees() const { return best_num_trees_; }
  float best_loss() const { return best_loss_; }
  const std::vector<float>& best_metrics() const { return best_metrics_; }

  // Last model.
  float last_loss() const { return last_loss_; }
  const std::vector<float>& last_metrics() const { return last_metrics_; }

  // Number of trees trained at each iteration. "set_trees_per_iterations"
  // should be called before the first update.
  int trees_per_iterations() const { return trees_per_iterations_; }
  void set_trees_per_iterations(const int trees_per_iterations) {
    trees_per_iterations_ = trees_per_iterations;
  }

 private:
  // Minimum validation loss over all the step of the model. Only valid if
  // "min_validation_loss_num_trees>=0".
  float best_loss_ = 0.f;
  float last_loss_ = 0.f;

  std::vector<float> last_metrics_;
  std::vector<float> best_metrics_;

  // Number of trees in the model with the validation loss
  // "minimum_validation_loss_value".
  int best_num_trees_ = -1;
  int last_num_trees_ = 0;

  int num_trees_look_ahead_;

  int trees_per_iterations_ = -1;
};

// Computes the loss best adapted to the problem.
utils::StatusOr<proto::Loss> DefaultLoss(
    model::proto::Task task, const dataset::proto::Column& label_spec);

void SetInitialPredictions(const std::vector<float>& initial_predictions,
                           const dataset::VerticalDataset::row_t num_rows,
                           std::vector<float>* predictions);

// Sets the default hyper-parameters of the learner.
absl::Status SetDefaultHyperParameters(
    gradient_boosted_trees::proto::GradientBoostedTreesTrainingConfig*
        gbt_config);

}  // namespace internal

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_H_
