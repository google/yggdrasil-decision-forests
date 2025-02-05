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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_TRAINING_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_TRAINING_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/label.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_accumulator.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_scanner.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency_streamprocessor.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests::model::decision_tree {

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

// Applies a constraint over a leaf.
absl::Status ApplyConstraintOnNode(const NodeConstraints& constraint,
                                   NodeWithChildren* node);

// Divides a monotonic constraint over node's children.
absl::Status DivideMonotonicConstraintToChildren(
    const NodeConstraints& constraint, bool direction_increasing,
    bool check_monotonic, NodeWithChildren* parent_node,
    NodeWithChildren* pos_node, NodeWithChildren* neg_node,
    NodeConstraints* pos_constraint, NodeConstraints* neg_constraint);

// Set of immutable arguments in a splitter work request.
struct SplitterWorkRequestCommon {
  const dataset::VerticalDataset& train_dataset;
  absl::Span<const UnsignedExampleIdx> selected_examples;
  const proto::Node& parent;
  const LabelStats& label_stats;
  const NodeConstraints& constraints;
};

// Data packed with the work request that can be used by the manager to pass
// information to itself.
struct SplitterWorkManagerData {
  //  Index of the condition in the cache pool.
  int cache_idx;
  // Index of the job.
  int job_idx;
};

// Work request for a splitter i.e. finding the best possible condition for a
// given attribute on a given dataset.
struct SplitterWorkRequest {
  SplitterWorkManagerData manager_data;

  std::atomic<float>& best_score;

  // The attribute index to pass onto splitters.
  int attribute_idx;

  // Non-owning pointer to an entry in PerThreadCache.splitter_cache_list.
  SplitterPerThreadCache* splitter_cache;

  // Set of immutable arguments passed onto splitters.
  SplitterWorkRequestCommon* common;
  // Seed used to initialize the random generator.
  utils::RandomEngine::result_type seed;
  // If not -1, search for oblique split. In this case "attribute_idx" should be
  // -1.
  int num_oblique_projections_to_run;

  // Copy is not allowed.
  SplitterWorkRequest(SplitterWorkManagerData manager_data,
                      std::atomic<float>& best_score, int attribute_idx,
                      SplitterPerThreadCache* splitter_cache,
                      SplitterWorkRequestCommon* common,
                      utils::RandomEngine::result_type seed,
                      int num_oblique_projections_to_run)
      : manager_data(manager_data),
        best_score(best_score),
        attribute_idx(attribute_idx),
        splitter_cache(splitter_cache),
        common(common),
        seed(seed),
        num_oblique_projections_to_run(num_oblique_projections_to_run) {}
  SplitterWorkRequest(const SplitterWorkRequest&) = delete;
  SplitterWorkRequest& operator=(const SplitterWorkRequest&) = delete;
  SplitterWorkRequest(SplitterWorkRequest&&) = default;
  SplitterWorkRequest& operator=(SplitterWorkRequest&&) = default;
};

// Contains the result of a splitter.
struct SplitterWorkResponse {
  SplitterWorkManagerData manager_data;

  // The status returned by a splitter.
  SplitSearchResult status;

  std::unique_ptr<proto::NodeCondition> condition;

  // Copy is not allowed.
  SplitterWorkResponse() = default;
  SplitterWorkResponse(SplitterWorkManagerData manager_data,
                       SplitSearchResult status,
                       std::unique_ptr<proto::NodeCondition> condition)
      : manager_data(manager_data),
        status(status),
        condition(std::move(condition)) {}
  SplitterWorkResponse(const SplitterWorkResponse&) = delete;
  SplitterWorkResponse& operator=(const SplitterWorkResponse&) = delete;
  SplitterWorkResponse(SplitterWorkResponse&&) = default;
  SplitterWorkResponse& operator=(SplitterWorkResponse&&) = default;
};

using SplitterFinderStreamProcessor =
    yggdrasil_decision_forests::utils::concurrency::StreamProcessor<
        SplitterWorkRequest, absl::StatusOr<SplitterWorkResponse>>;

// Records the status of workers in a concurrent setup.
// Part of the worker response (SplitterWorkResponse) that need to be kept in
// order to simulate sequential feature splitting.
struct SplitterWorkDurableResponse {
  std::unique_ptr<proto::NodeCondition> condition;

  // The status returned by a splitter.
  SplitSearchResult status;

  // If not set, the other fields are not meaningful.
  bool set;
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

  // A set of objects that are used by FindBestCondition.
  std::vector<SplitterPerThreadCache> splitter_cache_list;
  std::vector<SplitterWorkDurableResponse> durable_response_list;

  // List of available indices into splitter_cache_list.
  std::vector<int32_t> available_cache_idxs;

  // Used to handle selected example indices.
  std::vector<UnsignedExampleIdx> selected_example_buffer;

  // Used to handle selected leaf example indices.
  std::vector<UnsignedExampleIdx> leaf_example_buffer;
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
// from the gradient label statistics.
typedef std::function<absl::Status(const decision_tree::proto::LabelStatistics&,
                                   decision_tree::proto::Node*)>
    SetLeafValueFromLabelStatsFunctor;

// Find the best condition for a leaf node. Return true if a condition better
// than the one initially in `best_condition` was found. If `best_condition` is
// a newly created object, return true if a condition was found (since
// `best_condition` does not yet define a condition).
//
// This is the entry point / main function to call to find a condition.
absl::StatusOr<bool> FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, PerThreadCache* cache);

// Following are the method to handle multithreading in FindBestCondition.
// =============================================================================

// Dispatches the condition search to either single thread or multithread
// computation.
absl::StatusOr<bool> FindBestConditionManager(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, const NodeConstraints& constraints,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    PerThreadCache* cache);

// Single thread search for conditions.
absl::StatusOr<bool> FindBestConditionSingleThreadManager(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, const NodeConstraints& constraints,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    PerThreadCache* cache);

// Multi-thread search for conditions.
absl::StatusOr<bool> FindBestConditionConcurrentManager(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, const NodeConstraints& constraints,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    PerThreadCache* cache);

// Starts the worker threads needed for "FindBestConditionConcurrentManager".
absl::Status FindBestConditionStartWorkers(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const InternalTrainConfig& internal_config,
    const std::vector<float>& weights,
    SplitterConcurrencySetup* splitter_concurrency_setup);

// A worker that receives splitter work requests and dispatches those to the
// right specialized splitter function.
//
// Important: This function closes the "out" channel upon termination.
absl::StatusOr<SplitterWorkResponse> FindBestConditionFromSplitterWorkRequest(
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const InternalTrainConfig& internal_config,
    const SplitterWorkRequest& request);

// Following are the "FindBestCondition" specialized for specific tasks.
// =============================================================================

absl::StatusOr<SplitSearchResult> FindBestConditionClassification(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const ClassificationLabelStats& label_stats, int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache);

absl::StatusOr<SplitSearchResult> FindBestConditionRegression(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionLabelStats& label_stats, int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache);

absl::StatusOr<SplitSearchResult> FindBestConditionRegressionHessianGain(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionHessianLabelStats& label_stats, int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache);

absl::StatusOr<SplitSearchResult> FindBestConditionUpliftCategorical(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const CategoricalUpliftLabelStats& label_stats, int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache);

absl::StatusOr<SplitSearchResult> FindBestConditionUpliftNumerical(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const NumericalUpliftLabelStats& label_stats, int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache);

// Following are the "FindBestCondition" specialized for both a task (i.e. label
// semantic) and feature semantic. The function names follow the pattern:
// FindSplitLabel{label_type}Feature{feature_type}{algorithm_name}.
//
// Some splitters are only specialized on the feature, but not one the label
// typee (e.g. "FindBestConditionOblique";
// =============================================================================

// Search for the best split of the type "Attribute is NA" (i.e. "Attribute is
// missing") for classification.
absl::StatusOr<SplitSearchResult> FindSplitLabelClassificationFeatureNA(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<int32_t>& labels, int32_t num_label_classes,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Search for the best split of the type "Attribute is NA" (i.e. "Attribute is
// missing") for regression.
template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelRegressionFeatureNA(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<float>& labels, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  FeatureIsMissingBucket::Filler feature_filler(attributes);

  typename LabelNumericalBucket<weighted>::Filler label_filler(labels, weights);

  typename LabelNumericalBucket<weighted>::Initializer initializer(
      label_distribution);

  return FindBestSplit_LabelRegressionFeatureNACart<weighted>(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

// Search for the best split of the type "Attribute is NA" (i.e. "Attribute is
// missing") for hessian regression.
template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelHessianRegressionFeatureNA(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Search for the best split of the type Boolean for classification.
absl::StatusOr<SplitSearchResult> FindSplitLabelClassificationFeatureBoolean(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<int32_t>& labels, int32_t num_label_classes,
    bool na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Search for the best split of the type Boolean for regression.
template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelRegressionFeatureBoolean(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<float>& labels, bool na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelHessianRegressionFeatureBoolean(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    bool na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
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
// `weights` may be empty and this is equivalent to unit weights.
absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureNumericalCart(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, absl::Span<const float> attributes,
    const std::vector<int32_t>& labels, int32_t num_label_classes,
    float na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

// Similarly to "FindSplitLabelClassificationFeatureNumericalCart", but uses an
// histogram approach to find the best split.
//
// `weights` may be empty and this is equivalent to unit weights.
absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureNumericalHistogram(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, absl::Span<const float> attributes,
    const std::vector<int32_t>& labels, int32_t num_label_classes,
    float na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition);

// Similar to "FindSplitLabelClassificationFeatureNumericalCart", but work on
// pre-discretized numerical values.
//
// `weights` may be empty and this is equivalent to unit weights.
absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureDiscretizedNumericalCart(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    int num_bins, const std::vector<int32_t>& labels, int32_t num_label_classes,
    dataset::DiscretizedNumericalIndex na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Search for the best split for a numerical attribute and a numerical label
// using the CART algorithm for a dataset loaded in memory.
//
// This functions works similarly as
// "FindSplitLabelClassificationFeatureNumericalCart" for categorical labels.
template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelRegressionFeatureNumericalCart(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, absl::Span<const float> attributes,
    const std::vector<float>& labels, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureNumericalCart(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, absl::Span<const float> attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    float na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureDiscretizedNumericalCart(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    int num_bins, const std::vector<float>& gradients,
    const std::vector<float>& hessians, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

// Similarly to "FindSplitLabelClassificationFeatureNumericalCart", but uses an
// histogram approach to find the best split.
template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureNumericalHistogram(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, absl::Span<const float> attributes,
    const std::vector<float>& labels, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition);

// Similar to "FindSplitLabelClassificationFeatureNumericalCart", but work on
// pre-discretized numerical values.
template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureDiscretizedNumericalCart(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    int num_bins, const std::vector<float>& labels,
    dataset::DiscretizedNumericalIndex na_replacement,
    UnsignedExampleIdx min_num_obs,
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
//
// `weights` may be empty and this is equivalent to unit weights.
absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureCategorical(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<int32_t>& labels, int32_t num_attribute_classes,
    int32_t num_label_classes, int32_t na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

// Looks for the best split for a categorical attribute and a numerical
// label using the algorithm set in "dt_config" for a dataset loaded in memory.
//
// This function works similarly as
// "FindSplitLabelClassificationFeatureCategorical" for categorical labels.
template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelRegressionFeatureCategorical(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<float>& labels, int32_t num_attribute_classes,
    int32_t na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random);

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureCategorical(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    int32_t num_attribute_classes, int32_t na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
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
// `weights` may be empty and this is equivalent to unit weights.
absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureCategoricalSetGreedyForward(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<int32_t>& labels, int32_t num_attribute_classes,
    int32_t num_label_classes, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random);

// Similar as the previous
// "FindSplitLabelClassificationFeatureCategoricalSetGreedyForward", but for
// regression.
// The "information gain" is replaced by the "variance reduction".
template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureCategoricalSetGreedyForward(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<float>& labels, int32_t num_attribute_classes,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random);

// Find the best possible condition for a uplift with categorical treatment,
// a numerical feature and categorical outcome.
absl::StatusOr<SplitSearchResult>
FindSplitLabelUpliftCategoricalFeatureNumericalCart(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, absl::Span<const float> attributes,
    const CategoricalUpliftLabelStats& label_stats, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Find the best possible condition for a uplift with categorical treatment,
// a numerical feature and numerical outcome.
absl::StatusOr<SplitSearchResult>
FindSplitLabelUpliftNumericalFeatureNumericalCart(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, absl::Span<const float> attributes,
    const NumericalUpliftLabelStats& label_stats, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

// Find the best possible condition for a uplift with categorical treatment,
// a categorical feature, and categorical outcome.
absl::StatusOr<SplitSearchResult>
FindSplitLabelUpliftCategoricalFeatureCategorical(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const CategoricalUpliftLabelStats& label_stats, int num_attribute_classes,
    int32_t na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random);

// Find the best possible condition for a uplift with categorical treatment,
// a categorical feature and numerical outcome.
absl::StatusOr<SplitSearchResult>
FindSplitLabelUpliftNumericalFeatureCategorical(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const NumericalUpliftLabelStats& label_stats, int num_attribute_classes,
    int32_t na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random);

// Find the best possible oblique condition.
absl::StatusOr<bool> FindBestConditionOblique(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache);

// End of the FindBestCondition specialization.
// =============================================================================

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
absl::Status GenerateRandomImputation(
    const dataset::VerticalDataset& src, const std::vector<int>& attributes,
    absl::Span<const UnsignedExampleIdx> examples,
    dataset::VerticalDataset* dst, utils::RandomEngine* random);

// Random imputation on a given column. See documentation of
// "GenerateRandomImputation".
absl::Status GenerateRandomImputationOnColumn(
    const dataset::VerticalDataset::AbstractColumn* src,
    absl::Span<const UnsignedExampleIdx> examples,
    dataset::VerticalDataset::AbstractColumn* dst, utils::RandomEngine* random);

// Grows a decision tree with a "best-first" (or "leaf-wise") grow i.e. the leaf
// that best improve the overall tree is split.
absl::Status GrowTreeBestFirstGlobal(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const std::vector<float>& weights,
    const InternalTrainConfig& internal_config, NodeWithChildren* root,
    utils::RandomEngine* random,
    SelectedExamplesRollingBuffer selected_examples,
    std::optional<SelectedExamplesRollingBuffer> leaf_examples);

// The core training logic that is the same between single-threaded execution
// and concurrent execution.
//
// If "leaf_examples" is non null, it contains the examples to use to determine
// the value of the leaves while "selected_examples" contains the examples to
// use to determine the structure of the tree. If "leaf_examples" is null, the
// examples "selected_examples" are used for both.
//
// The "selected_examples" buffer will be modified during training.
absl::Status DecisionTreeCoreTrain(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const std::vector<float>& weights, utils::RandomEngine* random,
    const InternalTrainConfig& internal_config, DecisionTree* dt,
    absl::Span<UnsignedExampleIdx> selected_examples,
    std::optional<absl::Span<UnsignedExampleIdx>> leaf_examples);

// Train the tree. Fails if the tree is not empty.
absl::Status DecisionTreeTrain(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
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
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const std::vector<float>& weights, int32_t depth,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, bool set_leaf_already_set,
    NodeWithChildren* node, utils::RandomEngine* random, PerThreadCache* cache,
    SelectedExamplesRollingBuffer selected_examples,
    std::optional<SelectedExamplesRollingBuffer> leaf_examples);

// Set the default values of the hyper-parameters.
void SetDefaultHyperParameters(proto::DecisionTreeTrainingConfig* config);

// Set the default values of the internal hyper-parameters. Should be called
// after "SetDefaultHyperParameters". Does not change user visible
// hyper-parameters.
void SetInternalDefaultHyperParameters(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& link_config,
    const dataset::proto::DataSpecification& data_spec,
    proto::DecisionTreeTrainingConfig* dt_config);

// Number of attributes to test when looking for an optimal split.
int NumAttributesToTest(const proto::DecisionTreeTrainingConfig& dt_config,
                        int num_attributes, model::proto::Task task);

// Returns -1 if a feature is decreasing monotonic, +1 if a feature is a
// increasing monotonic, and 0 if a feature is not constrained.
int8_t MonotonicConstraintSign(
    const model::proto::TrainingConfigLinking& config_link, int attribute_idx);

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
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<int64_t>&
        count_examples_without_weights_by_attribute_class,
    std::vector<bool>* candidate_attributes_bitmap,
    utils::RandomEngine* random);

// Create the histogram bins (i.e. candidate threshold values) for an histogram
// based split finding on a numerical attribute.
absl::StatusOr<std::vector<float>> GenHistogramBins(
    proto::NumericalSplit::Type type, int num_splits,
    absl::Span<const float> attributes, float min_value, float max_value,
    utils::RandomEngine* random);

// Sets in "positive_examples" and "negative_examples" the examples from
// "examples" that evaluate respectively positively and negatively to the
// condition "condition". The items in "examples" are expected to be sorted.
// When the function returns, "examples" might not be sorted anymore.
// "positive_examples" and "negative_examples" will be pointing to subsets of
// "examples".
//
// If "examples_are_training_examples=true", optimizes the allocation by
// assuming "examples" are the examples used to train the tree.
absl::StatusOr<ExampleSplitRollingBuffer> SplitExamplesInPlace(
    const dataset::VerticalDataset& dataset,
    SelectedExamplesRollingBuffer examples,
    const proto::NodeCondition& condition, bool dataset_is_dense,
    bool error_on_wrong_splitter_statistics,
    bool examples_are_training_examples = true);

}  // namespace internal

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_TRAINING_H_
