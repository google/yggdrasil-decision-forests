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

#include "yggdrasil_decision_forests/learner/decision_tree/generic_parameters.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

namespace {

TEST(GenericParameters, AllHyperparameters) {
  proto::DecisionTreeTrainingConfig config;
  model::proto::GenericHyperParameterSpecification hparam_def;
  EXPECT_OK(GetGenericHyperParameterSpecification(config, &hparam_def));
}

TEST(GenericParameters, GiveValidAndInvalidHyperparameters) {
  proto::DecisionTreeTrainingConfig config;
  model::proto::GenericHyperParameterSpecification hparam_def;
  absl::flat_hash_set<std::string> valid_hyperparameters = {
      kHParamMaxDepth,
      kHParamMinExamples,
  };
  absl::flat_hash_set<std::string> invalid_hyperparameters = {
      kHParamNumCandidateAttributes,
      kHParamNumCandidateAttributesRatio,
      kHParamInSplitMinExampleCheck,
      kHParamAllowNaConditions,
      kHParamMissingValuePolicy,
      kHParamCategoricalSetSplitGreedySampling,
      kHParamCategoricalSetSplitMaxNumItems,
      kHParamCategoricalSetSplitMinItemFrequency,
      kHParamGrowingStrategy,
      kHParamMaxNumNodes,
      kHParamSplitAxis,
      kHParamSplitAxisSparseObliqueNumProjectionsExponent,
      kHParamSplitAxisSparseObliqueProjectionDensityFactor,
      kHParamSplitAxisSparseObliqueNormalization,
      kHParamSplitAxisSparseObliqueWeights,
      kHParamSplitAxisSparseObliqueMaxNumProjections,
      kHParamSplitAxisMhldObliqueMaxNumAttributes,
      kHParamSplitAxisMhldObliqueSampleAttributes,
      kHParamCategoricalAlgorithm,
      kHParamSortingStrategy,
      kHParamKeepNonLeafLabelDistribution,
      kHParamUpliftSplitScore,
      kHParamUpliftMinExamplesInTreatment,
      kHParamHonest,
      kHParamHonestRatioLeafExamples,
      kHParamHonestFixedSeparation};
  EXPECT_OK(GetGenericHyperParameterSpecification(
      config, &hparam_def, valid_hyperparameters, invalid_hyperparameters));
  EXPECT_EQ(hparam_def.fields().size(), 2);
  EXPECT_TRUE(hparam_def.fields().contains(kHParamMaxDepth));
  EXPECT_TRUE(hparam_def.fields().contains(kHParamMinExamples));
}

TEST(GenericParameters, MissingValidHyperparameters) {
  proto::DecisionTreeTrainingConfig config;
  model::proto::GenericHyperParameterSpecification hparam_def;
  absl::flat_hash_set<std::string> valid_hyperparameters = {
      kHParamMaxDepth,
      // kHParamMinExamples,
  };
  absl::flat_hash_set<std::string> invalid_hyperparameters = {
      kHParamNumCandidateAttributes,
      kHParamNumCandidateAttributesRatio,
      kHParamInSplitMinExampleCheck,
      kHParamAllowNaConditions,
      kHParamMissingValuePolicy,
      kHParamCategoricalSetSplitGreedySampling,
      kHParamCategoricalSetSplitMaxNumItems,
      kHParamCategoricalSetSplitMinItemFrequency,
      kHParamGrowingStrategy,
      kHParamMaxNumNodes,
      kHParamSplitAxis,
      kHParamSplitAxisSparseObliqueNumProjectionsExponent,
      kHParamSplitAxisSparseObliqueProjectionDensityFactor,
      kHParamSplitAxisSparseObliqueNormalization,
      kHParamSplitAxisSparseObliqueWeights,
      kHParamSplitAxisSparseObliqueMaxNumProjections,
      kHParamSplitAxisMhldObliqueMaxNumAttributes,
      kHParamSplitAxisMhldObliqueSampleAttributes,
      kHParamCategoricalAlgorithm,
      kHParamSortingStrategy,
      kHParamKeepNonLeafLabelDistribution,
      kHParamUpliftSplitScore,
      kHParamUpliftMinExamplesInTreatment,
      kHParamHonest,
      kHParamHonestRatioLeafExamples,
      kHParamHonestFixedSeparation};
  absl::Status status = GetGenericHyperParameterSpecification(
      config, &hparam_def, valid_hyperparameters, invalid_hyperparameters);
  EXPECT_THAT(status, test::StatusIs(absl::StatusCode::kInternal,
                                     "Hyperparameter min_examples is neither "
                                     "listed as valid nor invalid."));
}

TEST(GenericParameters, MissingInvalidHyperparameters) {
  proto::DecisionTreeTrainingConfig config;
  model::proto::GenericHyperParameterSpecification hparam_def;
  absl::flat_hash_set<std::string> valid_hyperparameters = {kHParamMaxDepth,
                                                            kHParamMinExamples};
  absl::flat_hash_set<std::string> invalid_hyperparameters = {
      // kHParamNumCandidateAttributes,
      kHParamNumCandidateAttributesRatio,
      kHParamInSplitMinExampleCheck,
      kHParamAllowNaConditions,
      kHParamMissingValuePolicy,
      kHParamCategoricalSetSplitGreedySampling,
      kHParamCategoricalSetSplitMaxNumItems,
      kHParamCategoricalSetSplitMinItemFrequency,
      kHParamGrowingStrategy,
      kHParamMaxNumNodes,
      kHParamSplitAxis,
      kHParamSplitAxisSparseObliqueNumProjectionsExponent,
      kHParamSplitAxisSparseObliqueProjectionDensityFactor,
      kHParamSplitAxisSparseObliqueNormalization,
      kHParamSplitAxisSparseObliqueWeights,
      kHParamSplitAxisSparseObliqueMaxNumProjections,
      kHParamSplitAxisMhldObliqueMaxNumAttributes,
      kHParamSplitAxisMhldObliqueSampleAttributes,
      kHParamCategoricalAlgorithm,
      kHParamSortingStrategy,
      kHParamKeepNonLeafLabelDistribution,
      kHParamUpliftSplitScore,
      kHParamUpliftMinExamplesInTreatment,
      kHParamHonest,
      kHParamHonestRatioLeafExamples,
      kHParamHonestFixedSeparation};
  absl::Status status = GetGenericHyperParameterSpecification(
      config, &hparam_def, valid_hyperparameters, invalid_hyperparameters);
  EXPECT_THAT(status, test::StatusIs(
                          absl::StatusCode::kInternal,
                          "Hyperparameter num_candidate_attributes is neither "
                          "listed as valid nor invalid."));
}

TEST(GenericParameters, UnknownValidHyperparameter) {
  proto::DecisionTreeTrainingConfig config;
  model::proto::GenericHyperParameterSpecification hparam_def;
  absl::flat_hash_set<std::string> valid_hyperparameters = {
      kHParamMaxDepth, kHParamMinExamples, "does_not_exist_valid"};
  absl::flat_hash_set<std::string> invalid_hyperparameters = {
      // kHParamNumCandidateAttributes,
      kHParamNumCandidateAttributesRatio,
      kHParamInSplitMinExampleCheck,
      kHParamAllowNaConditions,
      kHParamMissingValuePolicy,
      kHParamCategoricalSetSplitGreedySampling,
      kHParamCategoricalSetSplitMaxNumItems,
      kHParamCategoricalSetSplitMinItemFrequency,
      kHParamGrowingStrategy,
      kHParamMaxNumNodes,
      kHParamSplitAxis,
      kHParamSplitAxisSparseObliqueNumProjectionsExponent,
      kHParamSplitAxisSparseObliqueProjectionDensityFactor,
      kHParamSplitAxisSparseObliqueNormalization,
      kHParamSplitAxisSparseObliqueWeights,
      kHParamSplitAxisSparseObliqueMaxNumProjections,
      kHParamSplitAxisMhldObliqueMaxNumAttributes,
      kHParamSplitAxisMhldObliqueSampleAttributes,
      kHParamCategoricalAlgorithm,
      kHParamSortingStrategy,
      kHParamKeepNonLeafLabelDistribution,
      kHParamUpliftSplitScore,
      kHParamUpliftMinExamplesInTreatment,
      kHParamHonest,
      kHParamHonestRatioLeafExamples,
      kHParamHonestFixedSeparation};
  absl::Status status = GetGenericHyperParameterSpecification(
      config, &hparam_def, valid_hyperparameters, invalid_hyperparameters);
  EXPECT_THAT(
      status,
      test::StatusIs(absl::StatusCode::kInternal,
                     "Unknown valid hyperparameter: does_not_exist_valid"));
}

TEST(GenericParameters, UnknownInvalidHyperparameter) {
  proto::DecisionTreeTrainingConfig config;
  model::proto::GenericHyperParameterSpecification hparam_def;
  absl::flat_hash_set<std::string> valid_hyperparameters = {kHParamMaxDepth,
                                                            kHParamMinExamples};
  absl::flat_hash_set<std::string> invalid_hyperparameters = {
      // kHParamNumCandidateAttributes,
      kHParamNumCandidateAttributesRatio,
      kHParamInSplitMinExampleCheck,
      kHParamAllowNaConditions,
      kHParamMissingValuePolicy,
      kHParamCategoricalSetSplitGreedySampling,
      kHParamCategoricalSetSplitMaxNumItems,
      kHParamCategoricalSetSplitMinItemFrequency,
      kHParamGrowingStrategy,
      kHParamMaxNumNodes,
      kHParamSplitAxis,
      kHParamSplitAxisSparseObliqueNumProjectionsExponent,
      kHParamSplitAxisSparseObliqueProjectionDensityFactor,
      kHParamSplitAxisSparseObliqueNormalization,
      kHParamSplitAxisSparseObliqueWeights,
      kHParamSplitAxisSparseObliqueMaxNumProjections,
      kHParamSplitAxisMhldObliqueMaxNumAttributes,
      kHParamSplitAxisMhldObliqueSampleAttributes,
      kHParamCategoricalAlgorithm,
      kHParamSortingStrategy,
      kHParamKeepNonLeafLabelDistribution,
      kHParamUpliftSplitScore,
      kHParamUpliftMinExamplesInTreatment,
      kHParamHonest,
      kHParamHonestRatioLeafExamples,
      kHParamHonestFixedSeparation,
      "does_not_exist_invalid"};
  absl::Status status = GetGenericHyperParameterSpecification(
      config, &hparam_def, valid_hyperparameters, invalid_hyperparameters);
  EXPECT_THAT(
      status,
      test::StatusIs(absl::StatusCode::kInternal,
                     "Unknown invalid hyperparameter: does_not_exist_invalid"));
}

TEST(GenericParameters, ExistingHyperparameter) {
  proto::DecisionTreeTrainingConfig config;
  model::proto::GenericHyperParameterSpecification hparam_def;
  (*hparam_def.mutable_fields())["new_hp"];
  absl::flat_hash_set<std::string> valid_hyperparameters = {kHParamMaxDepth,
                                                            kHParamMinExamples};
  absl::flat_hash_set<std::string> invalid_hyperparameters = {
      kHParamNumCandidateAttributes,
      kHParamNumCandidateAttributesRatio,
      kHParamInSplitMinExampleCheck,
      kHParamAllowNaConditions,
      kHParamMissingValuePolicy,
      kHParamCategoricalSetSplitGreedySampling,
      kHParamCategoricalSetSplitMaxNumItems,
      kHParamCategoricalSetSplitMinItemFrequency,
      kHParamGrowingStrategy,
      kHParamMaxNumNodes,
      kHParamSplitAxis,
      kHParamSplitAxisSparseObliqueNumProjectionsExponent,
      kHParamSplitAxisSparseObliqueProjectionDensityFactor,
      kHParamSplitAxisSparseObliqueNormalization,
      kHParamSplitAxisSparseObliqueWeights,
      kHParamSplitAxisSparseObliqueMaxNumProjections,
      kHParamSplitAxisMhldObliqueMaxNumAttributes,
      kHParamSplitAxisMhldObliqueSampleAttributes,
      kHParamCategoricalAlgorithm,
      kHParamSortingStrategy,
      kHParamKeepNonLeafLabelDistribution,
      kHParamUpliftSplitScore,
      kHParamUpliftMinExamplesInTreatment,
      kHParamHonest,
      kHParamHonestRatioLeafExamples,
      kHParamHonestFixedSeparation};
  absl::Status status = GetGenericHyperParameterSpecification(
      config, &hparam_def, valid_hyperparameters, invalid_hyperparameters);
  EXPECT_THAT(
      status,
      test::StatusIs(absl::StatusCode::kInternal,
                     "Hyperparameter new_hp is neither listed as valid nor "
                     "invalid."));
}

TEST(GenericParameters, OnlyValidFails) {
  proto::DecisionTreeTrainingConfig config;
  model::proto::GenericHyperParameterSpecification hparam_def;
  (*hparam_def.mutable_fields())["new_hp"];
  absl::flat_hash_set<std::string> valid_hyperparameters = {};
  absl::Status status = GetGenericHyperParameterSpecification(
      config, &hparam_def, valid_hyperparameters, absl::nullopt);
  EXPECT_THAT(
      status,
      test::StatusIs(
          absl::StatusCode::kInternal,
          "A caller must either supply both the valid hyperparameter and the "
          "invalid hyperparameters or none of them"));
}

TEST(GenericParameters, OnlyInvalidFails) {
  proto::DecisionTreeTrainingConfig config;
  model::proto::GenericHyperParameterSpecification hparam_def;
  (*hparam_def.mutable_fields())["new_hp"];
  absl::flat_hash_set<std::string> invalid_hyperparameters = {};
  absl::Status status = GetGenericHyperParameterSpecification(
      config, &hparam_def, absl::nullopt, invalid_hyperparameters);
  EXPECT_THAT(
      status,
      test::StatusIs(
          absl::StatusCode::kInternal,
          "A caller must either supply both the valid hyperparameter and the "
          "invalid hyperparameters or none of them"));
}

}  // namespace

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
