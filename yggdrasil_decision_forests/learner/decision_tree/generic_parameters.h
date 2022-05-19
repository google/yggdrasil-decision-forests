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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_GENERIC_PARAMETERS_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_GENERIC_PARAMETERS_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

// Generic hyper parameter names.
constexpr char kHParamMaxDepth[] = "max_depth";
constexpr char kHParamMinExamples[] = "min_examples";
constexpr char kHParamNumCandidateAttributes[] = "num_candidate_attributes";
constexpr char kHParamNumCandidateAttributesRatio[] =
    "num_candidate_attributes_ratio";
constexpr char kHParamInSplitMinExampleCheck[] = "in_split_min_examples_check";
constexpr char kHParamAllowNaConditions[] = "allow_na_conditions";
constexpr char kHParamMissingValuePolicy[] = "missing_value_policy";
constexpr char kHParamGrowingStrategy[] = "growing_strategy";
constexpr char kHParamMaxNumNodes[] = "max_num_nodes";

constexpr char kGrowingStrategyLocal[] = "LOCAL";
constexpr char kGrowingStrategyBestFirstGlobal[] = "BEST_FIRST_GLOBAL";

// Categorical-set splitter
constexpr char kHParamCategoricalSetSplitGreedySampling[] =
    "categorical_set_split_greedy_sampling";
constexpr char kHParamCategoricalSetSplitMaxNumItems[] =
    "categorical_set_split_max_num_items";
constexpr char kHParamCategoricalSetSplitMinItemFrequency[] =
    "categorical_set_split_min_item_frequency";

// Categorical splitter
constexpr char kHParamCategoricalAlgorithm[] = "categorical_algorithm";
constexpr char kCategoricalAlgorithmCART[] = "CART";
constexpr char kCategoricalAlgorithmOneHot[] = "ONE_HOT";
constexpr char kCategoricalAlgorithmRandom[] = "RANDOM";

// Axis splits.
constexpr char kHParamSplitAxis[] = "split_axis";
constexpr char kHParamSplitAxisAxisAligned[] = "AXIS_ALIGNED";
constexpr char kHParamSplitAxisSparseOblique[] = "SPARSE_OBLIQUE";

constexpr char kHParamSplitAxisSparseObliqueNumProjectionsExponent[] =
    "sparse_oblique_num_projections_exponent";
constexpr char kHParamSplitAxisSparseObliqueProjectionDensityFactor[] =
    "sparse_oblique_projection_density_factor";

constexpr char kHParamSplitAxisSparseObliqueWeights[] =
    "sparse_oblique_weights";
constexpr char kHParamSplitAxisSparseObliqueWeightsBinary[] = "BINARY";
constexpr char kHParamSplitAxisSparseObliqueWeightsContinuous[] = "CONTINUOUS";

constexpr char kHParamSplitAxisSparseObliqueNormalization[] =
    "sparse_oblique_normalization";
constexpr char kHParamSplitAxisSparseObliqueNormalizationNone[] = "NONE";
constexpr char kHParamSplitAxisSparseObliqueNormalizationStandardDeviation[] =
    "STANDARD_DEVIATION";
constexpr char kHParamSplitAxisSparseObliqueNormalizationMinMax[] = "MIN_MAX";

constexpr char kHParamSortingStrategy[] = "sorting_strategy";
constexpr char kHParamSortingStrategyInNode[] = "IN_NODE";
constexpr char kHParamSortingStrategyPresort[] = "PRESORT";

constexpr char kHParamKeepNonLeafLabelDistribution[] =
    "keep_non_leaf_label_distribution";

constexpr char kHParamUpliftSplitScore[] = "uplift_split_score";
constexpr char kHParamUpliftMinExamplesInTreatment[] =
    "uplift_min_examples_in_treatment";

constexpr char kHParamUpliftSplitScoreKL[] = "KULLBACK_LEIBLER";
constexpr char kHParamUpliftSplitScoreED[] = "EUCLIDEAN_DISTANCE";
constexpr char kHParamUpliftSplitScoreCS[] = "CHI_SQUARED";
constexpr char kHParamUpliftSplitScoreCED[] = "CONSERVATIVE_EUCLIDEAN_DISTANCE";
constexpr char kHParamUpliftSplitScoreKLAlt[] = "KL";
constexpr char kHParamUpliftSplitScoreEDAlt[] = "ED";
constexpr char kHParamUpliftSplitScoreCSAlt[] = "CS";
constexpr char kHParamUpliftSplitScoreCEDAlt[] = "CED";

constexpr char kHParamHonest[] = "honest";
constexpr char kHParamHonestRatioLeafExamples[] = "honest_ratio_leaf_examples";
constexpr char kHParamHonestFixedSeparation[] = "honest_fixed_separation";

// The values kTrue and kFalse represent boolean values of a categorical
// hyper-parameter.
// TODO(gbm): Add direct support for boolean hyper-parameter. Note: users are
// generally not using those hyper-parameter directly. Instead generic
// hyper-parameter are used in several sub-modules (automatic generation of
// documentation, generation of python code, hyper-parameter tuning, etc.).
constexpr char kTrue[] = "true";
constexpr char kFalse[] = "false";

// Fill decision tree specific generic hyper parameter specifications.
// This function is designed to be called by the learners using decision trees
// learning.
absl::Status GetGenericHyperParameterSpecification(
    const proto::DecisionTreeTrainingConfig& config,
    model::proto::GenericHyperParameterSpecification* hparam_def);

// Set the fields of a decision tree train proto from set of generic hyper
// parameters.
// "consumed_hparams" contains the list of already "consumed" hyper-parameters.
// An error is raised if "SetHyperParameters" tries to consume an
// hyper-parameter initially "consumed_hparams". All the consumed
// hyper-parameter keys are added to "consumed_hparams".
absl::Status SetHyperParameters(
    absl::flat_hash_set<std::string>* consumed_hparams,
    proto::DecisionTreeTrainingConfig* dt_config,
    utils::GenericHyperParameterConsumer* generic_hyper_params);

// Default predefined hyper-parameter space for axis splits.
void PredefinedHyperParameterAxisSplitSpace(
    model::proto::HyperParameterSpace* space);

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_GENERIC_PARAMETERS_H_
