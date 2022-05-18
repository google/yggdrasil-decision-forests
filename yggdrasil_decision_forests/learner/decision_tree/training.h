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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_TRAINING_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_TRAINING_H_

#include <functional>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_accumulator.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_scanner.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/types.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/circular_buffer.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/concurrency_streamprocessor.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

// Defines a generic collection of label statistics.
struct LabelStats {
  virtual ~LabelStats() = default;
};

// Structure that encapsulates label statistics for Classification.
struct ClassificationLabelStats : LabelStats {
  explicit ClassificationLabelStats(const std::vector<int32_t>& label_data)
      : label_data(label_data) {}

  const std::vector<int32_t>& label_data;
  int32_t num_label_classes;
  utils::IntegerDistributionDouble label_distribution;
};

// Structure that encapsulates label statistics for Regression.
struct RegressionLabelStats : LabelStats {
  explicit RegressionLabelStats(const std::vector<float>& label_data)
      : label_data(label_data) {}

  const std::vector<float>& label_data;
  utils::NormalDistributionDouble label_distribution;
};

// Structure that encapsulates label statistics for Regression.
struct RegressionHessianLabelStats : LabelStats {
  RegressionHessianLabelStats(const std::vector<float>& gradient_data,
                              const std::vector<float>& hessian_data)
      : gradient_data(gradient_data), hessian_data(hessian_data) {}

  const std::vector<float>& gradient_data;
  const std::vector<float>& hessian_data;
  double sum_gradient;
  double sum_hessian;
  double sum_weights;
};

// Structure that encapsulates label statistics for uplift with categorical
// treatment and categorical outcome.
struct CategoricalUpliftLabelStats : LabelStats {
  explicit CategoricalUpliftLabelStats(
      const std::vector<int32_t>& outcome_values,
      const int num_unique_in_outcomes_column,
      const std::vector<int32_t>& treatment_values,
      const int num_unique_values_in_treatments_column)
      : outcome_values(outcome_values),
        treatment_values(treatment_values),
        num_unique_values_in_treatments_column(
            num_unique_values_in_treatments_column),
        num_unique_in_outcomes_column(num_unique_in_outcomes_column) {}

  const std::vector<int32_t>& outcome_values;
  const std::vector<int32_t>& treatment_values;
  int32_t num_unique_values_in_treatments_column;
  int32_t num_unique_in_outcomes_column;

  UpliftLabelDistribution label_distribution;
};

// Structure that encapsulates label statistics for uplift with categorical
// treatment and numerical outcome..
struct NumericalUpliftLabelStats : LabelStats {
  explicit NumericalUpliftLabelStats(
      const std::vector<float>& outcome_values,
      const std::vector<int32_t>& treatment_values,
      const int num_unique_values_in_treatments_column)
      : outcome_values(outcome_values),
        treatment_values(treatment_values),
        num_unique_values_in_treatments_column(
            num_unique_values_in_treatments_column) {}

  const std::vector<float>& outcome_values;
  const std::vector<int32_t>& treatment_values;
  int32_t num_unique_values_in_treatments_column;

  UpliftLabelDistribution label_distribution;
};

// A collection of objects used by split-finding methods.
//
// The purpose of this cache structure is to avoid repeated allocation of the
// contained objects over the course of training.
//
// A splitter (e.g. "FindBest...") is guarantied exclusive access at any time
// (thread safe). This structure is passed as place holder. Assume it is filled
// with garbage, it's not cleaned. The only thing preserved between calls to
// splitters is the size of the vectors, and the fact that they are
// pre-allocated from previous caller to specific purpose (e.g.
// "numerical_features" is allocated to contains one item for each numerical
// features).
struct SplitterPerThreadCache {
  // Objects used by splitters.
  utils::IntegersConfusionMatrixDouble confusion;
  utils::IntegersConfusionMatrixInt64 confusion_without_weights;
  utils::IntegerDistributionInt64 label_distribution_without_weights;
  std::vector<std::pair<float, int32_t>> positive_label_ratio_by_attr_value;
  utils::BinaryToIntegerConfusionMatrixDouble running_confusion;
  utils::BinaryToIntegerConfusionMatrixInt64 running_confusion_no_weights;
  std::vector<utils::NormalDistributionDouble> label_dist_by_attr_value;
  std::vector<int64_t> count_by_attr_value_without_weights;
  std::vector<std::pair<float, int32_t>> mean_label_by_attr_value;
  std::vector<utils::IntegerDistributionDouble> attr_label_distribution;
  std::vector<int64_t> attr_to_num_examples;

  std::vector<int> numerical_features;
  std::vector<float> projection_values;

  PerThreadCacheV2 cache_v2;

  utils::RandomEngine random;
};

// Set of immutable arguments in a splitter work request.
struct SplitterWorkRequestCommon {
  const dataset::VerticalDataset& train_dataset;
  const std::vector<dataset::VerticalDataset::row_t>& selected_examples;
  const proto::Node& parent;
  const LabelStats& label_stats;
};

// Work request for a splitter i.e. finding the best possible condition for a
// given attribute on a given dataset.
struct SplitterWorkRequest {
  // Index into PerThreadCache.work_status_list.
  int status_idx;
  // The attribute index to pass onto splitters.
  int attribute_idx;
  // Non-owning pointer to a "condition" in PerThreadCache.condition_list.
  proto::NodeCondition* dst_condition;
  // Non-owning pointer to an entry in PerThreadCache.splitter_cache_list.
  SplitterPerThreadCache* splitter_cache;
  // Best score found by splitters in previous calls.
  float best_score;
  // Set of immutable arguments passed onto splitters.
  SplitterWorkRequestCommon* common;
  // Seed used to initialize the random generator.
  utils::RandomEngine::result_type seed;
};

// Contains the result of a splitter.
struct SplitterWorkResponse {
  // Index into PerThreadCache.work_status_list.
  int status_idx;
  // Non-owning pointer to a "condition" in PerThreadCache.condition_list.
  proto::NodeCondition* condition;
  // The status returned by a splitter.
  SplitSearchResult status;
};

using SplitterFinderStreamProcessor =
    yggdrasil_decision_forests::utils::concurrency::StreamProcessor<
        SplitterWorkRequest, SplitterWorkResponse>;

// Records the status of workers in a concurrent setup.
struct SplitterWorkStatus {
  // If .condition is not null, then "condition =
  // PerThreadCache::condition_list[condition_idx].
  int condition_idx;
  // Index into PerThreadCache.splitter_cache_list.
  int cache_idx;
  // Non-owning pointer to a "condition" in PerThreadCache.condition_list.
  proto::NodeCondition* condition;
  // The status returned by a splitter.
  SplitSearchResult status;
};

// Memory cache used to reduce the number of allocation / de-allocation of
// memory during training. One mutable "PerThreadCache" object is required by
// the "train" method.
//
// A "PerThreadCache" object can be reused in successive calls to "train", but
// NOT for concurrent calls.
//
// Details: This cache allows to reduce the multithread locking of tcmalloc.
// Without the cache, most of the training time is spent in memory management.
struct PerThreadCache {
  // Object used by the splitter manager.
  std::vector<int32_t> candidate_attributes;

  struct PerDepth {
    // Indices of the positive and negative examples a split.
    std::vector<dataset::VerticalDataset::row_t> positive_examples;
    std::vector<dataset::VerticalDataset::row_t> negative_examples;

    // Indices of the positive and negative examples used only for the leaf
    // values in a split.
    std::vector<dataset::VerticalDataset::row_t> positive_node_only_examples;
    std::vector<dataset::VerticalDataset::row_t> negative_node_only_examples;
  };
  // Cache per depth.
  // Note: We use a unique pointer to guaranty stability of content.
  std::vector<std::unique_ptr<PerDepth>> per_depth;

  // A set of objects that are used by FindBestCondition.
  std::vector<SplitterPerThreadCache> splitter_cache_list;
  std::vector<SplitterWorkStatus> work_status_list;
  std::vector<proto::NodeCondition> condition_list;

  // List of available indices into splitter_cache_list.
  utils::CircularBuffer<int32_t> available_cache_idxs;
  // List of available indices into condition_list.
  utils::CircularBuffer<int32_t> available_condition_idxs;
};

// In a concurrent setup, this structure encapsulates all the objects that are
// needed to communicate with splitter workers.
struct SplitterConcurrencySetup {
  // Whether concurrent execution has been enabled.
  bool concurrent_execution;
  // The number of threads available in the worker pool.
  int num_threads;

  // Distributed split finder.
  std::unique_ptr<SplitterFinderStreamProcessor> split_finder_processor;
};

// Signature of a function that sets the value (i.e. the prediction) of a leaf
// from the gradient data.
typedef std::function<void(
    const dataset::VerticalDataset&,
    const std::vector<dataset::VerticalDataset::row_t>&,
    const std::vector<float>&, const model::proto::TrainingConfig&,
    const model::proto::TrainingConfigLinking&, NodeWithChildren* node)>
    CreateSetLeafValueFunctor;

// Signature of a function that sets the value (i.e. the prediction) of a leaf
// from the gradient label statistics.
typedef std::function<absl::Status(const decision_tree::proto::LabelStatistics&,
                                   decision_tree::proto::Node*)>
    SetLeafValueFromLabelStatsFunctor;

// Pre-computation on the training dataset used for the training of individual
// trees. The pre-processing is computed before any tree is trained.
class Preprocessing {
 public:
  struct PresortedNumericalFeature {
    // Feature value and example index sorted by feature values.
    // Missing values are replaced using the GLOBAL_IMPUTATION strategy.
    std::vector<SparseItem> items;
  };

  std::vector<PresortedNumericalFeature>*
  mutable_presorted_numerical_features() {
    return &presorted_numerical_features_;
  }

  const std::vector<PresortedNumericalFeature>& presorted_numerical_features()
      const {
    return presorted_numerical_features_;
  }

  uint64_t num_examples() const { return num_examples_; }

  void set_num_examples(const uint64_t value) { num_examples_ = value; }

 private:
  // List of presorted numerical features, indexed by feature index.
  // If feature "i" is not numerical or not presorted,
  // "presorted_numerical_features_[i]" will be an empty index.
  std::vector<PresortedNumericalFeature> presorted_numerical_features_;

  // Total number of examples.
  uint64_t num_examples_ = -1;
};

// The default policy to set the value of a leaf.
// - Distribution of the labels for classification.
// - Mean of the labels for regression.
void SetLabelDistribution(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    NodeWithChildren* node);

// Default policy to set the label value of a leaf in a regression tree i.e. set
// the value to the mean of the labels.
void SetRegressionLabelDistribution(
    const dataset::VerticalDataset& dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfigLinking& config_link, proto::Node* node);

// Training configuration for internal parameters not available to the user
// directly.
struct InternalTrainConfig {
  CreateSetLeafValueFunctor set_leaf_value_functor = SetLabelDistribution;

  // If true, evaluate split gain using the formulation relying on the hessian.
  // In this case, "hessian_col_idx" should be a numerical column containing the
  // hessian. The label column is expected to contain the gradient.
  bool use_hessian_gain = false;

  // Index of the hessian column in the dataset.
  int hessian_col_idx = -1;

  // Regularization terms.
  float hessian_l1 = 0.f;
  float hessian_l2_numerical = 0.f;
  float hessian_l2_categorical = 0.f;

  // Number of attributes tested in parallel (using fiber threads).
  int num_threads = 1;

  // Non owning, pointer to pre-processing information.
  // Depending on the decision tree configuration this field might be required.
  const Preprocessing* preprocessing = nullptr;

  // If true, the list of selected example index ("selected_examples") can
  // contain duplicated values. If false, all selected examples are expected to
  // be unique.
  bool duplicated_selected_examples = true;

  // If set, the training of the tree will stop after this time, leading to an
  // under-grow but valid decision tree. The growing strategy defines how the
  // tree is "under-grown".
  absl::optional<absl::Time> timeout;
};

// Find the best condition for this node. Return true iff a good condition has
// been found.
utils::StatusOr<bool> FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    PerThreadCache* cache);

// Contains logic to switch between a single-threaded splitter and a concurrent
// implementation.
utils::StatusOr<bool> FindBestConditionManager(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, PerThreadCache* cache);

// This is an implementation of FindBestConditionManager that is optimized for
// execution in a single thread.
utils::StatusOr<bool> FindBestConditionSingleThreadManager(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, PerThreadCache* cache);

// This is a concurrent implementation of FindBestConditionManager.
utils::StatusOr<bool> FindBestConditionConcurrentManager(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, PerThreadCache* cache);

// A worker that receives splitter work requests and dispatches those to the
// right specialized splitter function.
//
// Important: This function closes the "out" channel upon termination.
SplitterWorkResponse FindBestConditionFromSplitterWorkRequest(
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const InternalTrainConfig& internal_config,
    const SplitterWorkRequest& request);

// Specialization in the case of classification.
SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const ClassificationLabelStats& label_stats, const int32_t attribute_idx,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache);

// Specialization in the case of regression.
SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionLabelStats& label_stats, const int32_t attribute_idx,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache);

// Specialization in the case of regression with hessian gain.
SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionHessianLabelStats& label_stats, const int32_t attribute_idx,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache);

// Specialization in the case of uplift with categorical outcome.
SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const CategoricalUpliftLabelStats& label_stats, int32_t attribute_idx,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache);

// Specialization in the case of uplift with numerical outcome.
SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const NumericalUpliftLabelStats& label_stats, int32_t attribute_idx,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache);

// Following are the split finder function. Their name follow the patter:
// FindSplitLabel{label_type}Feature{feature_type}{algorithm_name}.

// Search for the best split of the type "Attribute is NA" (i.e. "Attribute is
// missing") for classification.
SplitSearchResult FindSplitLabelClassificationFeatureNA(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<int32_t>& labels, const int32_t num_label_classes,
    const dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Search for the best split of the type "Attribute is NA" (i.e. "Attribute is
// missing") for regression.
SplitSearchResult FindSplitLabelRegressionFeatureNA(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<float>& labels,
    const dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Search for the best split of the type "Attribute is NA" (i.e. "Attribute is
// missing") for hessian regression.
SplitSearchResult FindSplitLabelHessianRegressionFeatureNA(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    const dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Search for the best split of the type Boolean for classification.
SplitSearchResult FindSplitLabelClassificationFeatureBoolean(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<char>& attributes,
    const std::vector<int32_t>& labels, int32_t num_label_classes,
    bool na_replacement, dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Search for the best split of the type Boolean for regression.
SplitSearchResult FindSplitLabelRegressionFeatureBoolean(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<char>& attributes,
    const std::vector<float>& labels, bool na_replacement,
    dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

SplitSearchResult FindSplitLabelHessianRegressionFeatureBoolean(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<char>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    bool na_replacement, dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Search for the best split for a numerical attribute and a categorical label
// using the CART algorithm for a dataset loaded in memory. All the threshold
// are evaluated. All Na values are replaced by "na_replacement". Does not
// support weights.
//
// Currently the code only support conditions of the form: attribute>=threshold.
//
// "labels" and "attributes" contain respectively the label and attribute value
// of all the example in the dataset. Both vectors have the same size. From this
// dataset, only the example indexed in "selected_examples" should be considered
// to evaluation the quality of a split i.e. only consider the examples defined
// by attributes[selected_examples[i]] and labels[selected_examples[i]]. Note:
// As an example, non selected examples could be in other part of the tree, OOB
// or excluded for cross-validation purpose.
//
// "num_label_classes" represents the number of label classes i.e. all the
// number in "label" should be in [0, num_label_classes[.
//
// "na_replacement" replaces the NA (non-available) values in "attributes[i]"
// when evaluating the splits i.e. given a split threshold "t", an example "i"
// with attributes[i]==NA will be considered positive iif  "na_replacement" >=t.
//
// "min_num_obs" is the minimum number of (training) observations in either side
// of the split. Splits that invalidate this constraint are ignored.
//
// "label_distribution" is the label distribution of the selected examples i.e.
// the distribution of "labels" for all elements in "selected_examples".
//
// "attribute_idx" is the column index of the attribute. This argument is not
// used for computation. Instead, it is used to update the "condition" is a good
// split is found.
//
// "condition" will contain the best valid found split (if any). If "condition"
// already contains a valid condition (i.e. "condition.split_score()" is set),
// "condition" will be updated iff a new found split has a better score than the
// split initially contained in "condition".
//
// The function returns kBetterSplitFound is a better split was found,
// kNoBetterSplitFound if there are some valid splits but none of them were
// better than the split initially in "condition", and kInvalidAttribute is not
// valid split was found.
//
SplitSearchResult FindSplitLabelClassificationFeatureNumericalCart(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const std::vector<int32_t>& labels, int32_t num_label_classes,
    float na_replacement, dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

// Similarly to "FindSplitLabelClassificationFeatureNumericalCart", but uses an
// histogram approach to find the best split.
SplitSearchResult FindSplitLabelClassificationFeatureNumericalHistogram(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const std::vector<int32_t>& labels, int32_t num_label_classes,
    float na_replacement, dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition);

// Similar to "FindSplitLabelClassificationFeatureNumericalCart", but work on
// pre-discretized numerical values.
SplitSearchResult FindSplitLabelClassificationFeatureDiscretizedNumericalCart(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    int num_bins, const std::vector<int32_t>& labels, int32_t num_label_classes,
    dataset::DiscretizedNumericalIndex na_replacement,
    dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Search for the best split for a numerical attribute and a numerical label
// using the CART algorithm for a dataset loaded in memory.
//
// This functions works similarly as
// "FindSplitLabelClassificationFeatureNumericalCart" for categorical labels.
SplitSearchResult FindSplitLabelRegressionFeatureNumericalCart(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const std::vector<float>& labels, float na_replacement,
    dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

SplitSearchResult FindSplitLabelHessianRegressionFeatureNumericalCart(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    float na_replacement, dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

SplitSearchResult
FindSplitLabelHessianRegressionFeatureDiscretizedNumericalCart(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    int num_bins, const std::vector<float>& gradients,
    const std::vector<float>& hessians, float na_replacement,
    dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Similarly to "FindSplitLabelClassificationFeatureNumericalCart", but uses an
// histogram approach to find the best split.
SplitSearchResult FindSplitLabelRegressionFeatureNumericalHistogram(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const std::vector<float>& labels, float na_replacement,
    dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition);

// Similar to "FindSplitLabelClassificationFeatureNumericalCart", but work on
// pre-discretized numerical values.
SplitSearchResult FindSplitLabelRegressionFeatureDiscretizedNumericalCart(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    int num_bins, const std::vector<float>& labels,
    dataset::DiscretizedNumericalIndex na_replacement,
    dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Looks for the best split for a categorical attribute and a categorical label
// using the algorithm configured in "dt_config" for a dataset loaded in memory.
// Such split is defined as a subset of the possible values of the attribute.
// During inference, the node test will succeed iif the attribute value is
// contained in this subset.
//
// Arguments are similar to "FindSplitLabelClassificationFeatureNumericalCart".
// "num_attribute_classes" specifies the number of classes of the attribute
// (i.e. the maximum value for the elements in "attributes").
SplitSearchResult FindSplitLabelClassificationFeatureCategorical(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<int32_t>& labels, int32_t num_attribute_classes,
    int32_t num_label_classes, int32_t na_replacement,
    dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

// Looks for the best split for a categorical attribute and a numerical
// label using the algorithm set in "dt_config" for a dataset loaded in memory.
//
// This function works similarly as
// "FindSplitLabelClassificationFeatureCategorical" for categorical labels.
SplitSearchResult FindSplitLabelRegressionFeatureCategorical(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<float>& labels, const int32_t num_attribute_classes,
    int32_t na_replacement, const row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random);

SplitSearchResult FindSplitLabelHessianRegressionFeatureCategorical(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    const int32_t num_attribute_classes, int32_t na_replacement,
    dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random);

// Looks for the best split for a categorical set attribute and a categorical
// label for a dataset loaded in memory.
//
// The categorical set split is defined by examples where the intersection of
// "positive_set" and the examples attribute (a categorical set) is empty /
// non-empty.
//
// The best split algorithm works by greedily selecting categorical values into
// the "positive_set". It can be described as follow:
//
//    positive_set = {}
//    Sample uniformly a subset of candidate attribute item "candidate_items".
//    score = 0
//    while true
//      Find "a" in "candidate_items" such that the score of
//      "positive_set + a" (intersection) is maximized. This score is the
//      "new_score".
//      if new_score < score
//        break
//      add "s" to "positive_set".
//      score = new_score.
//    return positive_set
//
SplitSearchResult
FindSplitLabelClassificationFeatureCategoricalSetGreedyForward(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<int32_t>& labels, const int32_t num_attribute_classes,
    const int32_t num_label_classes,
    const dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random);

// Similar as the previous
// "FindSplitLabelClassificationFeatureCategoricalSetGreedyForward", but for
// regression.
// The "information gain" is replaced by the "variance reduction".
SplitSearchResult FindSplitLabelRegressionFeatureCategoricalSetGreedyForward(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<float>& labels, int32_t num_attribute_classes,
    dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random);

// Find the best possible condition for a uplift with categorical treatment,
// a numerical feature and categorical outcome.
SplitSearchResult FindSplitLabelUpliftCategoricalFeatureNumericalCart(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const CategoricalUpliftLabelStats& label_stats, float na_replacement,
    dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Find the best possible condition for a uplift with categorical treatment,
// a numerical feature and numerical outcome.
SplitSearchResult FindSplitLabelUpliftNumericalFeatureNumericalCart(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const NumericalUpliftLabelStats& label_stats, float na_replacement,
    dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Find the best possible condition for a uplift with categorical treatment,
// a categorical feature, and categorical outcome.
SplitSearchResult FindSplitLabelUpliftCategoricalFeatureCategorical(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const CategoricalUpliftLabelStats& label_stats, int num_attribute_classes,
    int32_t na_replacement, dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random);

// Find the best possible condition for a uplift with categorical treatment,
// a categorical feature and numerical outcome.
SplitSearchResult FindSplitLabelUpliftNumericalFeatureCategorical(
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const NumericalUpliftLabelStats& label_stats, int num_attribute_classes,
    int32_t na_replacement, dataset::VerticalDataset::row_t min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random);

// Returns the number of attributes to test ("num_attributes_to_test") and a
// list of candidate attributes to test in order ("candidate_attributes").
// "candidate_attributes" is guaranteed to have at least
// "num_attributes_to_test" elements.
void GetCandidateAttributes(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    int* num_attributes_to_test, std::vector<int32_t>* candidate_attributes,
    utils::RandomEngine* random);

// Generate a random imputation of NA (i.e. missing) values i.e.
// Copy the attributes in "attributes" from the examples indexed by "examples"
// from the source dataset "src" into the "dst" dataset while replacing the
// missing values from samples from the "examples" in "src". Columns not
// specified by "attributes" are not copied. If, for a given attribute, all the
// values are NA, the data is simply copied i.e. "dst" will contain na values.
void GenerateRandomImputation(
    const dataset::VerticalDataset& src, const std::vector<int>& attributes,
    const std::vector<dataset::VerticalDataset::row_t>& examples,
    dataset::VerticalDataset* dst, utils::RandomEngine* random);

// Random imputation on a given column. See documentation of
// "GenerateRandomImputation".
void GenerateRandomImputationOnColumn(
    const dataset::VerticalDataset::AbstractColumn* src,
    const std::vector<dataset::VerticalDataset::row_t>& examples,
    dataset::VerticalDataset::AbstractColumn* dst, utils::RandomEngine* random);

// Grows a decision tree with a "best-first" (or "leaf-wise") grow i.e. the leaf
// that best improve the overall tree is split.
absl::Status GrowTreeBestFirstGlobal(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& train_example_idxs,
    const std::vector<dataset::VerticalDataset::row_t>* optional_leaf_examples,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const std::vector<float>& weights,
    const InternalTrainConfig& internal_config, NodeWithChildren* root,
    utils::RandomEngine* random);

// The core training logic that is the same between single-threaded execution
// and concurrent execution.
//
// If "leaf_examples" is non null, it contains the examples to use to determine
// the value of the leaves while "selected_examples" contains the examples to
// use to determine the structure of the tree. If "leaf_examples" is null, the
// examples "selected_examples" are used for both.
absl::Status DecisionTreeCoreTrain(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<dataset::VerticalDataset::row_t>* optional_leaf_examples,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const std::vector<float>& weights, utils::RandomEngine* random,
    const InternalTrainConfig& internal_config,

    DecisionTree* dt);

// Train the tree. Fails if the tree is not empty.
absl::Status DecisionTreeTrain(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const std::vector<float>& weights, utils::RandomEngine* random,
    DecisionTree* dt,
    const InternalTrainConfig& internal_config = InternalTrainConfig());
constexpr auto Train = DecisionTreeTrain;

// Train a node and its children.
absl::Status NodeTrain(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<dataset::VerticalDataset::row_t>* optional_leaf_examples,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const std::vector<float>& weights, const int32_t depth,
    const InternalTrainConfig& internal_config, NodeWithChildren* node,
    utils::RandomEngine* random, PerThreadCache* cache);

// Preprocess the dataset before any tree training.
utils::StatusOr<Preprocessing> PreprocessTrainingDataset(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config, int num_threads);

// Component of "PreprocessTrainingDataset". Computes pre-sorted numerical
// features.
absl::Status PresortNumericalFeatures(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfigLinking& config_link, int num_threads,
    Preprocessing* preprocessing);

// Set the default values of the hyper-parameters.
void SetDefaultHyperParameters(proto::DecisionTreeTrainingConfig* config);

namespace internal {

// Initializes the item mask i.e. the bitmap of the items to consider or to
// ignore in the greedy selection for categorical-set attributes. An item is
// masked if:
//   1. It is "pure" i.e. the item is present in all or in none of the examples.
//   2. The item has been sampled-out (see
//   "categorical_set_split_greedy_sampling" in "dt_config").
//   3. The item is pruned by the maximum number of items (see
//   "categorical_set_split_max_num_items" in "dt_config").
// Return true iif at least one item is non masked.
bool MaskPureSampledOrPrunedItemsForCategoricalSetGreedySelection(
    const proto::DecisionTreeTrainingConfig& dt_config,
    int32_t num_attribute_classes,
    const std::vector<dataset::VerticalDataset::row_t>& selected_examples,
    const std::vector<int64_t>&
        count_examples_without_weights_by_attribute_class,
    std::vector<bool>* candidate_attributes_bitmap,
    utils::RandomEngine* random);

// Create the histogram bins (i.e. candidate threshold values) for an histogram
// based split finding on a numerical attribute.
std::vector<float> GenHistogramBins(proto::NumericalSplit::Type type,
                                    int num_splits,
                                    const std::vector<float>& attributes,
                                    float min_value, float max_value,
                                    utils::RandomEngine* random);

// Computes the indices of the subset of examples in "examples" that evaluates
// positively and negatively to the condition.
//
// If "examples_are_training_examples=true", optimizes the allocation by
// assuming "examples" are the examples used to train the tree.
absl::Status SplitExamples(
    const dataset::VerticalDataset& dataset,
    const std::vector<dataset::VerticalDataset::row_t>& examples,
    const proto::NodeCondition& condition, bool dataset_is_dense,
    bool error_on_wrong_splitter_statistics,
    std::vector<dataset::VerticalDataset::row_t>* positive_examples,
    std::vector<dataset::VerticalDataset::row_t>* negative_examples,
    const bool examples_are_training_examples = true);

// Copies the content on uplift categorical leaf output to a label distribution.
void UpliftLeafToLabelDist(const decision_tree::proto::NodeUpliftOutput& leaf,
                           UpliftLabelDistribution* dist);

// Copies the content on uplift categorical label distribution to the leafs.
void UpliftLabelDistToLeaf(const UpliftLabelDistribution& dist,
                           decision_tree::proto::NodeUpliftOutput* leaf);

}  // namespace internal

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_TRAINING_H_
