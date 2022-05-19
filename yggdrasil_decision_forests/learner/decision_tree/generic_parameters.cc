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

#include "yggdrasil_decision_forests/learner/decision_tree/generic_parameters.h"

#include <string>
#include <type_traits>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

absl::Status GetGenericHyperParameterSpecification(
    const proto::DecisionTreeTrainingConfig& config,
    model::proto::GenericHyperParameterSpecification* hparam_def) {
  auto& fields = *hparam_def->mutable_fields();

  const auto get_params = [&fields](const absl::string_view key)
      -> utils::StatusOr<
          model::proto::GenericHyperParameterSpecification::Value*> {
    if (fields.find(std::string(key)) != fields.end()) {
      return absl::InternalError(
          absl::StrCat("Duplicated hyper parameter: ", key));
    }
    auto* field = &fields[std::string(key)];
    field->mutable_documentation()->set_proto_path(
        "learner/decision_tree/decision_tree.proto");
    return field;
  };

  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamMaxDepth));
    param->mutable_integer()->set_minimum(-1);
    param->mutable_integer()->set_default_value(config.max_depth());
    param->mutable_documentation()->set_description(
        R"(Maximum depth of the tree. `max_depth=1` means that all trees will be roots. Negative values are ignored.)");
  }
  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamMinExamples));
    param->mutable_integer()->set_minimum(1);
    param->mutable_integer()->set_default_value(config.min_examples());
    param->mutable_documentation()->set_description(
        R"(Minimum number of examples in a node.)");
  }
  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamNumCandidateAttributes));
    param->mutable_integer()->set_minimum(-1);
    param->mutable_integer()->set_default_value(
        config.num_candidate_attributes());
    param->mutable_documentation()->set_description(
        R"(Number of unique valid attributes tested for each node. An attribute is valid if it has at least a valid split. If `num_candidate_attributes=0`, the value is set to the classical default value for Random Forest: `sqrt(number of input attributes)` in case of classification and `number_of_input_attributes / 3` in case of regression. If `num_candidate_attributes=-1`, all the attributes are tested.)");
  }
  {
    ASSIGN_OR_RETURN(auto param,
                     get_params(kHParamNumCandidateAttributesRatio));
    param->mutable_real()->set_minimum(-1.);
    param->mutable_real()->set_maximum(1.);
    param->mutable_real()->set_default_value(
        config.num_candidate_attributes_ratio());
    param->mutable_documentation()->set_description(
        R"(Ratio of attributes tested at each node. If set, it is equivalent to `num_candidate_attributes = number_of_input_features x num_candidate_attributes_ratio`. The possible values are between ]0, and 1] as well as -1. If not set or equal to -1, the `num_candidate_attributes` is used.)");
  }
  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamInSplitMinExampleCheck));
    param->mutable_categorical()->set_default_value(
        config.in_split_min_examples_check() ? kTrue : kFalse);
    param->mutable_categorical()->add_possible_values(kTrue);
    param->mutable_categorical()->add_possible_values(kFalse);
    param->mutable_documentation()->set_description(
        R"(Whether to check the `min_examples` constraint in the split search (i.e. splits leading to one child having less than `min_examples` examples are considered invalid) or before the split search (i.e. a node can be derived only if it contains more than `min_examples` examples). If false, there can be nodes with less than `min_examples` training examples.)");
  }
  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamAllowNaConditions));
    param->mutable_categorical()->set_default_value(
        config.allow_na_conditions() ? kTrue : kFalse);
    param->mutable_categorical()->add_possible_values(kTrue);
    param->mutable_categorical()->add_possible_values(kFalse);
    param->mutable_documentation()->set_description(
        R"(If true, the tree training evaluates conditions of the type `X is NA` i.e. `X is missing`.)");
  }
  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamMissingValuePolicy));
    param->mutable_categorical()->set_default_value(
        proto::DecisionTreeTrainingConfig::MissingValuePolicy_Name(
            config.missing_value_policy()));
    param->mutable_categorical()->add_possible_values("GLOBAL_IMPUTATION");
    param->mutable_categorical()->add_possible_values("LOCAL_IMPUTATION");
    param->mutable_categorical()->add_possible_values(
        "RANDOM_LOCAL_IMPUTATION");
    param->mutable_documentation()->set_description(
        R"(Method used to handle missing attribute values.
- `GLOBAL_IMPUTATION`: Missing attribute values are imputed, with the mean (in case of numerical attribute) or the most-frequent-item (in case of categorical attribute) computed on the entire dataset (i.e. the information contained in the data spec).
- `LOCAL_IMPUTATION`: Missing attribute values are imputed with the mean (numerical attribute) or most-frequent-item (in the case of categorical attribute) evaluated on the training examples in the current node.
- `RANDOM_LOCAL_IMPUTATION`: Missing attribute values are imputed from randomly sampled values from the training examples in the current node. This method was proposed by Clinic et al. in "Random Survival Forests" (https://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908043).)");
  }

  {
    ASSIGN_OR_RETURN(auto param,
                     get_params(kHParamCategoricalSetSplitGreedySampling));
    param->mutable_real()->set_default_value(
        config.categorical_set_greedy_forward().sampling());
    param->mutable_real()->set_minimum(0);
    param->mutable_real()->set_maximum(1);
    param->mutable_documentation()->set_proto_field(
        "categorical_set_greedy_forward");
    param->mutable_documentation()->set_description(
        R"(For categorical set splits e.g. texts. Probability for a categorical value to be a candidate for the positive set. The sampling is applied once per node (i.e. not at every step of the greedy optimization).)");
  }

  {
    ASSIGN_OR_RETURN(auto param,
                     get_params(kHParamCategoricalSetSplitMaxNumItems));
    param->mutable_integer()->set_default_value(
        config.categorical_set_greedy_forward().max_num_items());
    param->mutable_integer()->set_minimum(-1);
    param->mutable_documentation()->set_proto_field("max_num_items");
    param->mutable_documentation()->set_description(
        R"(For categorical set splits e.g. texts. Maximum number of items (prior to the sampling). If more items are available, the least frequent items are ignored. Changing this value is similar to change the "max_vocab_count" before loading the dataset, with the following exception: With `max_vocab_count`, all the remaining items are grouped in a special Out-of-vocabulary item. With `max_num_items`, this is not the case.)");
  }

  {
    ASSIGN_OR_RETURN(auto param,
                     get_params(kHParamCategoricalSetSplitMinItemFrequency));
    param->mutable_integer()->set_default_value(
        config.categorical_set_greedy_forward().min_item_frequency());
    param->mutable_integer()->set_minimum(1);
    param->mutable_documentation()->set_proto_field("min_item_frequency");
    param->mutable_documentation()->set_description(
        R"(For categorical set splits e.g. texts. Minimum number of occurrences of an item to be considered.)");
  }

  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamGrowingStrategy));
    param->mutable_categorical()->set_default_value(kGrowingStrategyLocal);
    param->mutable_categorical()->add_possible_values(kGrowingStrategyLocal);
    param->mutable_categorical()->add_possible_values(
        kGrowingStrategyBestFirstGlobal);
    param->mutable_documentation()->set_description(
        R"(How to grow the tree.
- `LOCAL`: Each node is split independently of the other nodes. In other words, as long as a node satisfy the splits "constraints (e.g. maximum depth, minimum number of observations), the node will be split. This is the "classical" way to grow decision trees.
- `BEST_FIRST_GLOBAL`: The node with the best loss reduction among all the nodes of the tree is selected for splitting. This method is also called "best first" or "leaf-wise growth". See "Best-first decision tree learning", Shi and "Additive logistic regression : A statistical view of boosting", Friedman for more details.)");
  }

  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamMaxNumNodes));
    param->mutable_integer()->set_default_value(31);
    param->mutable_integer()->set_minimum(-1);
    param->mutable_conditional()->set_control_field(kHParamGrowingStrategy);
    param->mutable_conditional()->mutable_categorical()->add_values(
        kGrowingStrategyBestFirstGlobal);
    param->mutable_documentation()->set_description(
        R"(Maximum number of nodes in the tree. Set to -1 to disable this limit. Only available for `growing_strategy=BEST_FIRST_GLOBAL`.)");
  }

  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamSplitAxis));
    param->mutable_categorical()->set_default_value(
        kHParamSplitAxisAxisAligned);
    param->mutable_categorical()->add_possible_values(
        kHParamSplitAxisAxisAligned);
    param->mutable_categorical()->add_possible_values(
        kHParamSplitAxisSparseOblique);
    param->mutable_documentation()->set_description(
        R"(What structure of split to consider for numerical features.
- `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This is the "classical" way to train a tree. Default value.
- `SPARSE_OBLIQUE`: Sparse oblique splits (i.e. splits one a small number of features) from "Sparse Projection Oblique Random Forests", Tomita et al., 2020.)");
  }
  {
    ASSIGN_OR_RETURN(
        auto param,
        get_params(kHParamSplitAxisSparseObliqueNumProjectionsExponent));
    param->mutable_real()->set_default_value(
        config.sparse_oblique_split().num_projections_exponent());
    param->mutable_real()->set_minimum(0);
    param->mutable_documentation()->set_proto_field("num_projections_exponent");
    param->mutable_conditional()->set_control_field(kHParamSplitAxis);
    param->mutable_conditional()->mutable_categorical()->add_values(
        kHParamSplitAxisSparseOblique);
    param->mutable_documentation()->set_description(
        R"(For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections to test at each node as `num_features^num_projections_exponent`.)");
  }
  {
    ASSIGN_OR_RETURN(
        auto param,
        get_params(kHParamSplitAxisSparseObliqueProjectionDensityFactor));
    param->mutable_real()->set_default_value(
        config.sparse_oblique_split().projection_density_factor());
    param->mutable_real()->set_minimum(0);
    param->mutable_documentation()->set_proto_field(
        "projection_density_factor");
    param->mutable_conditional()->set_control_field(kHParamSplitAxis);
    param->mutable_conditional()->mutable_categorical()->add_values(
        kHParamSplitAxisSparseOblique);
    param->mutable_documentation()->set_description(
        R"(For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections to test at each node as `num_features^num_projections_exponent`.)");
  }

  {
    ASSIGN_OR_RETURN(auto param,
                     get_params(kHParamSplitAxisSparseObliqueNormalization));
    param->mutable_categorical()->set_default_value(
        kHParamSplitAxisSparseObliqueNormalizationNone);
    param->mutable_categorical()->add_possible_values(
        kHParamSplitAxisSparseObliqueNormalizationNone);
    param->mutable_categorical()->add_possible_values(
        kHParamSplitAxisSparseObliqueNormalizationStandardDeviation);
    param->mutable_categorical()->add_possible_values(
        kHParamSplitAxisSparseObliqueNormalizationMinMax);
    param->mutable_documentation()->set_proto_field("sparse_oblique_split");
    param->mutable_conditional()->set_control_field(kHParamSplitAxis);
    param->mutable_conditional()->mutable_categorical()->add_values(
        kHParamSplitAxisSparseOblique);
    param->mutable_documentation()->set_description(
        R"(For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Normalization applied on the features, before applying the sparse oblique projections.
- `NONE`: No normalization.
- `STANDARD_DEVIATION`: Normalize the feature by the estimated standard deviation on the entire train dataset. Also known as Z-Score normalization.
- `MIN_MAX`: Normalize the feature by the range (i.e. max-min) estimated on the entire train dataset.)");
  }

  {
    ASSIGN_OR_RETURN(auto param,
                     get_params(kHParamSplitAxisSparseObliqueWeights));
    param->mutable_categorical()->set_default_value(
        config.sparse_oblique_split().binary_weight()
            ? kHParamSplitAxisSparseObliqueWeightsBinary
            : kHParamSplitAxisSparseObliqueWeightsContinuous);
    param->mutable_categorical()->add_possible_values(
        kHParamSplitAxisSparseObliqueWeightsBinary);
    param->mutable_categorical()->add_possible_values(
        kHParamSplitAxisSparseObliqueWeightsContinuous);
    param->mutable_conditional()->set_control_field(kHParamSplitAxis);
    param->mutable_conditional()->mutable_categorical()->add_values(
        kHParamSplitAxisSparseOblique);
    param->mutable_documentation()->set_description(
        R"(For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Possible values:
- `BINARY`: The oblique weights are sampled in {-1,1} (default).
- `CONTINUOUS`: The oblique weights are be sampled in [-1,1].)");
  }

  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamCategoricalAlgorithm));
    param->mutable_categorical()->set_default_value(kCategoricalAlgorithmCART);
    param->mutable_categorical()->add_possible_values(
        kCategoricalAlgorithmCART);
    param->mutable_categorical()->add_possible_values(
        kCategoricalAlgorithmOneHot);
    param->mutable_categorical()->add_possible_values(
        kCategoricalAlgorithmRandom);
    param->mutable_documentation()->set_description(
        R"(How to learn splits on categorical attributes.
- `CART`: CART algorithm. Find categorical splits of the form "value \in mask". The solution is exact for binary classification, regression and ranking. It is approximated for multi-class classification. This is a good first algorithm to use. In case of overfitting (very small dataset, large dictionary), the "random" algorithm is a good alternative.
- `ONE_HOT`: One-hot encoding. Find the optimal categorical split of the form "attribute == param". This method is similar (but more efficient) than converting converting each possible categorical value into a boolean feature. This method is available for comparison purpose and generally performs worse than other alternatives.
- `RANDOM`: Best splits among a set of random candidate. Find the a categorical split of the form "value \in mask" using a random search. This solution can be seen as an approximation of the CART algorithm. This method is a strong alternative to CART. This algorithm is inspired from section "5.1 Categorical Variables" of "Random Forest", 2001.)");
  }

  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamSortingStrategy));
    switch (config.internal().sorting_strategy()) {
      case proto::DecisionTreeTrainingConfig::Internal::IN_NODE:
        param->mutable_categorical()->set_default_value(
            kHParamSortingStrategyInNode);
        break;
      case proto::DecisionTreeTrainingConfig::Internal::PRESORTED:
        param->mutable_categorical()->set_default_value(
            kHParamSortingStrategyPresort);
        break;
      default:
        return absl::InvalidArgumentError("Non implemented sorting strategy");
    }
    param->mutable_categorical()->add_possible_values(
        kHParamSortingStrategyInNode);
    param->mutable_categorical()->add_possible_values(
        kHParamSortingStrategyPresort);
    param->mutable_documentation()->set_description(
        R"(How are sorted the numerical features in order to find the splits
- PRESORT: The features are pre-sorted at the start of the training. This solution is faster but consumes much more memory than IN_NODE.
- IN_NODE: The features are sorted just before being used in the node. This solution is slow but consumes little amount of memory.
.)");
  }

  {
    ASSIGN_OR_RETURN(auto param,
                     get_params(kHParamKeepNonLeafLabelDistribution));
    param->mutable_categorical()->set_default_value(
        config.keep_non_leaf_label_distribution() ? kTrue : kFalse);
    param->mutable_categorical()->add_possible_values(kTrue);
    param->mutable_categorical()->add_possible_values(kFalse);
    param->mutable_documentation()->set_description(
        R"(Whether to keep the node value (i.e. the distribution of the labels of the training examples) of non-leaf nodes. This information is not used during serving, however it can be used for model interpretation as well as hyper parameter tuning. This can take lots of space, sometimes accounting for half of the model size.)");
  }

  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamUpliftSplitScore));
    param->mutable_categorical()->set_default_value(kHParamUpliftSplitScoreKL);

    for (const auto& value :
         {kHParamUpliftSplitScoreKL, kHParamUpliftSplitScoreKLAlt,
          kHParamUpliftSplitScoreED, kHParamUpliftSplitScoreEDAlt,
          kHParamUpliftSplitScoreCS, kHParamUpliftSplitScoreCSAlt,
          kHParamUpliftSplitScoreCED, kHParamUpliftSplitScoreCEDAlt}) {
      param->mutable_categorical()->add_possible_values(value);
    }

    param->mutable_documentation()->set_description(
        R"(For uplift models only. Splitter score i.e. score optimized by the splitters. The scores are introduced in "Decision trees for uplift modeling with single and multiple treatments", Rzepakowski et al. Notation: `p` probability / average value of the positive outcome, `q` probability / average value in the control group.
- `KULLBACK_LEIBLER` or `KL`: - p log (p/q)
- `EUCLIDEAN_DISTANCE` or `ED`: (p-q)^2
- `CHI_SQUARED` or `CS`: (p-q)^2/q
)");
  }

  {
    ASSIGN_OR_RETURN(auto param,
                     get_params(kHParamUpliftMinExamplesInTreatment));
    param->mutable_integer()->set_default_value(
        config.uplift().min_examples_in_treatment());
    param->mutable_integer()->set_minimum(0);
    param->mutable_documentation()->set_description(
        R"(For uplift models only. Minimum number of examples per treatment in a node.)");
  }

  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamHonest));
    param->mutable_categorical()->set_default_value(
        config.has_honest() ? kTrue : kFalse);
    param->mutable_categorical()->add_possible_values(kTrue);
    param->mutable_categorical()->add_possible_values(kFalse);
    param->mutable_documentation()->set_description(
        R"(In honest trees, different training examples are used to infer the structure and the leaf values. This regularization technique trades examples for bias estimates. It might increase or reduce the quality of the model. See "Generalized Random Forests", Athey et al. In this paper, Honest trees are trained with the Random Forest algorithm with a sampling without replacement.)");
  }

  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamHonestRatioLeafExamples));
    param->mutable_real()->set_minimum(0.);
    param->mutable_real()->set_maximum(1.);
    param->mutable_real()->set_default_value(
        config.honest().ratio_leaf_examples());
    param->mutable_documentation()->set_description(
        R"(For honest trees only i.e. honest=true. Ratio of examples used to set the leaf values.)");
  }

  {
    ASSIGN_OR_RETURN(auto param, get_params(kHParamHonestFixedSeparation));
    param->mutable_categorical()->set_default_value(
        config.honest().fixed_separation() ? kTrue : kFalse);
    param->mutable_categorical()->add_possible_values(kTrue);
    param->mutable_categorical()->add_possible_values(kFalse);
    param->mutable_documentation()->set_description(
        R"(For honest trees only i.e. honest=true. If true, a new random separation is generated for each tree. If false, the same separation is used for all the trees (e.g., in Gradient Boosted Trees containing multiple trees).)");
  }

  return absl::OkStatus();
}

absl::Status SetHyperParameters(
    absl::flat_hash_set<std::string>* consumed_hparams,
    proto::DecisionTreeTrainingConfig* dt_config,
    utils::GenericHyperParameterConsumer* generic_hyper_params) {
  int max_nodes = -1;
  bool max_nodes_is_set = false;

  {
    const auto hparam = generic_hyper_params->Get(kHParamMaxDepth);
    if (hparam.has_value()) {
      dt_config->set_max_depth(hparam.value().value().integer());
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamMinExamples);
    if (hparam.has_value()) {
      dt_config->set_min_examples(hparam.value().value().integer());
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamNumCandidateAttributes);
    if (hparam.has_value()) {
      dt_config->set_num_candidate_attributes(hparam.value().value().integer());
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamNumCandidateAttributesRatio);
    if (hparam.has_value()) {
      const auto value = hparam.value().value().real();
      if (value >= 0) {
        dt_config->set_num_candidate_attributes_ratio(value);
      }
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamInSplitMinExampleCheck);
    if (hparam.has_value()) {
      dt_config->set_in_split_min_examples_check(
          hparam.value().value().categorical() == kTrue);
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamAllowNaConditions);
    if (hparam.has_value()) {
      dt_config->set_allow_na_conditions(hparam.value().value().categorical() ==
                                         kTrue);
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamMissingValuePolicy);
    if (hparam.has_value()) {
      decision_tree::proto::DecisionTreeTrainingConfig::MissingValuePolicy
          value;
      if (!decision_tree::proto::DecisionTreeTrainingConfig::
              MissingValuePolicy_Parse(hparam.value().value().categorical(),
                                       &value)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Cannot parse MissingValuePolicy value \"",
                         hparam.value().value().categorical(), "\"."));
      }
      dt_config->set_missing_value_policy(value);
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamCategoricalSetSplitGreedySampling);
    if (hparam.has_value()) {
      dt_config->mutable_categorical_set_greedy_forward()->set_sampling(
          hparam.value().value().real());
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamCategoricalSetSplitMaxNumItems);
    if (hparam.has_value()) {
      dt_config->mutable_categorical_set_greedy_forward()->set_max_num_items(
          hparam.value().value().integer());
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamCategoricalSetSplitMinItemFrequency);
    if (hparam.has_value()) {
      dt_config->mutable_categorical_set_greedy_forward()
          ->set_min_item_frequency(hparam.value().value().integer());
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamGrowingStrategy);
    if (hparam.has_value()) {
      if (hparam.value().value().categorical() == kGrowingStrategyLocal) {
        dt_config->mutable_growing_strategy_local();
      } else if (hparam.value().value().categorical() ==
                 kGrowingStrategyBestFirstGlobal) {
        dt_config->mutable_growing_strategy_best_first_global();
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Unknown growing strategy: ",
                         hparam.value().value().categorical()));
      }
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamMaxNumNodes);
    if (hparam.has_value()) {
      max_nodes = hparam.value().value().integer();
      max_nodes_is_set = true;
    }
  }

  if (max_nodes_is_set) {
    if (!dt_config->has_growing_strategy_best_first_global()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "\"", kHParamMaxNumNodes, "\" is only available with \"",
          kHParamGrowingStrategy, "=", kGrowingStrategyBestFirstGlobal, "\"."));
    }
    dt_config->mutable_growing_strategy_best_first_global()->set_max_num_nodes(
        max_nodes);
  }

  // Oblique trees
  {
    const auto hparam = generic_hyper_params->Get(kHParamSplitAxis);
    if (hparam.has_value()) {
      const auto hparam_value = hparam.value().value().categorical();
      if (hparam_value == kHParamSplitAxisAxisAligned) {
        dt_config->mutable_axis_aligned_split();
      } else if (hparam_value == kHParamSplitAxisSparseOblique) {
        dt_config->mutable_sparse_oblique_split();
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Unknown axis split strategy: ", hparam_value));
      }
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(
        kHParamSplitAxisSparseObliqueNumProjectionsExponent);
    if (hparam.has_value()) {
      const auto hparam_value = hparam.value().value().real();
      if (dt_config->has_sparse_oblique_split()) {
        dt_config->mutable_sparse_oblique_split()->set_num_projections_exponent(
            hparam_value);
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat(kHParamSplitAxisSparseObliqueNumProjectionsExponent,
                         " only work with oblique trees"));
      }
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(
        kHParamSplitAxisSparseObliqueProjectionDensityFactor);
    if (hparam.has_value()) {
      const auto hparam_value = hparam.value().value().real();
      if (dt_config->has_sparse_oblique_split()) {
        dt_config->mutable_sparse_oblique_split()
            ->set_projection_density_factor(hparam_value);
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat(kHParamSplitAxisSparseObliqueProjectionDensityFactor,
                         " only work with oblique trees"));
      }
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamSplitAxisSparseObliqueNormalization);
    if (hparam.has_value()) {
      decision_tree::proto::DecisionTreeTrainingConfig::SparseObliqueSplit::
          Normalization value;
      if (!decision_tree::proto::DecisionTreeTrainingConfig::
              SparseObliqueSplit::Normalization_Parse(
                  hparam.value().value().categorical(), &value)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Cannot parse SparseObliqueSplit::Normalization value \"",
            hparam.value().value().categorical(), "\"."));
      }
      if (dt_config->has_sparse_oblique_split()) {
        dt_config->mutable_sparse_oblique_split()->set_normalization(value);
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat(kHParamSplitAxisSparseObliqueNormalization,
                         " only work with oblique trees"));
      }
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamSplitAxisSparseObliqueWeights);
    if (hparam.has_value()) {
      if (dt_config->has_sparse_oblique_split()) {
        const auto& value = hparam.value().value().categorical();
        if (value == kHParamSplitAxisSparseObliqueWeightsBinary) {
          dt_config->mutable_sparse_oblique_split()->set_binary_weight(true);
        } else if (value == kHParamSplitAxisSparseObliqueWeightsContinuous) {
          dt_config->mutable_sparse_oblique_split()->set_binary_weight(false);
        } else {
          return absl::InvalidArgumentError(absl::StrCat(
              "Unknown value for parameter ",
              kHParamSplitAxisSparseObliqueWeights, ". Possible values are: ",
              kHParamSplitAxisSparseObliqueWeightsBinary, " and ",
              kHParamSplitAxisSparseObliqueWeightsContinuous, "."));
        }
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat(kHParamSplitAxisSparseObliqueWeights,
                         " only work with oblique trees"));
      }
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamCategoricalAlgorithm);
    if (hparam.has_value()) {
      const auto value = hparam.value().value().categorical();
      if (value == kCategoricalAlgorithmCART) {
        dt_config->mutable_categorical()->mutable_cart();
      } else if (value == kCategoricalAlgorithmOneHot) {
        dt_config->mutable_categorical()->mutable_one_hot();
      } else if (value == kCategoricalAlgorithmRandom) {
        dt_config->mutable_categorical()->mutable_random();
      } else {
        return absl::InvalidArgumentError(
            absl::StrFormat(R"(Unknown value "%s" for parameter "%s")", value,
                            kHParamCategoricalAlgorithm));
      }
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamSortingStrategy);
    if (hparam.has_value()) {
      const auto value = hparam.value().value().categorical();
      if (value == kHParamSortingStrategyInNode) {
        dt_config->mutable_internal()->set_sorting_strategy(
            proto::DecisionTreeTrainingConfig::Internal::IN_NODE);
      } else if (value == kHParamSortingStrategyPresort) {
        dt_config->mutable_internal()->set_sorting_strategy(
            proto::DecisionTreeTrainingConfig::Internal::PRESORTED);
      } else {
        return absl::InvalidArgumentError(
            absl::StrFormat(R"(Unknown value "%s" for parameter "%s")", value,
                            kHParamSortingStrategy));
      }
    }
  }

  if (max_nodes_is_set) {
    if (!dt_config->has_growing_strategy_best_first_global()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "\"", kHParamMaxNumNodes, "\" is only available with \"",
          kHParamGrowingStrategy, "=", kGrowingStrategyBestFirstGlobal, "\"."));
    }
    dt_config->mutable_growing_strategy_best_first_global()->set_max_num_nodes(
        max_nodes);
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamKeepNonLeafLabelDistribution);
    if (hparam.has_value()) {
      dt_config->set_keep_non_leaf_label_distribution(
          hparam.value().value().categorical() == kTrue);
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamUpliftMinExamplesInTreatment);
    if (hparam.has_value()) {
      dt_config->mutable_uplift()->set_min_examples_in_treatment(
          hparam.value().value().integer());
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamUpliftSplitScore);
    if (hparam.has_value()) {
      const auto value = hparam.value().value().categorical();
      if (value == kHParamUpliftSplitScoreKL ||
          value == kHParamUpliftSplitScoreKLAlt) {
        dt_config->mutable_uplift()->set_split_score(
            proto::DecisionTreeTrainingConfig::Uplift::KULLBACK_LEIBLER);
      } else if (value == kHParamUpliftSplitScoreCS ||
                 value == kHParamUpliftSplitScoreCSAlt) {
        dt_config->mutable_uplift()->set_split_score(
            proto::DecisionTreeTrainingConfig::Uplift::CHI_SQUARED);
      } else if (value == kHParamUpliftSplitScoreED ||
                 value == kHParamUpliftSplitScoreEDAlt) {
        dt_config->mutable_uplift()->set_split_score(
            proto::DecisionTreeTrainingConfig::Uplift::EUCLIDEAN_DISTANCE);
      } else if (value == kHParamUpliftSplitScoreCED ||
                 value == kHParamUpliftSplitScoreCEDAlt) {
        dt_config->mutable_uplift()->set_split_score(
            proto::DecisionTreeTrainingConfig::Uplift::
                CONSERVATIVE_EUCLIDEAN_DISTANCE);
      } else {
        return absl::InvalidArgumentError(
            absl::StrFormat(R"(Unknown value "%s" for parameter "%s")", value,
                            kHParamUpliftSplitScore));
      }
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamHonest);
    if (hparam.has_value()) {
      if (hparam.value().value().categorical() == kTrue) {
        dt_config->mutable_honest();
      } else {
        dt_config->clear_honest();
      }
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamHonestRatioLeafExamples);
    if (hparam.has_value()) {
      if (dt_config->has_honest()) {
        dt_config->mutable_honest()->set_ratio_leaf_examples(
            hparam.value().value().real());
      }
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamHonestFixedSeparation);
    if (hparam.has_value()) {
      if (dt_config->has_honest()) {
        dt_config->mutable_honest()->set_fixed_separation(
            hparam.value().value().categorical() == kTrue);
      }
    }
  }

  return absl::OkStatus();
}

void PredefinedHyperParameterAxisSplitSpace(
    model::proto::HyperParameterSpace* space) {
  auto* field = space->add_fields();
  field->set_name(decision_tree::kHParamSplitAxis);
  auto* cands = field->mutable_discrete_candidates();
  cands->add_possible_values()->set_categorical(
      decision_tree::kHParamSplitAxisAxisAligned);
  cands->add_possible_values()->set_categorical(
      decision_tree::kHParamSplitAxisSparseOblique);

  auto* child = field->add_children();
  child->set_name(
      decision_tree::kHParamSplitAxisSparseObliqueProjectionDensityFactor);
  auto* parent_values = child->mutable_parent_discrete_values();
  parent_values->add_possible_values()->set_categorical(
      decision_tree::kHParamSplitAxisSparseOblique);
  auto* child_cands = child->mutable_discrete_candidates();
  child_cands->add_possible_values()->set_real(1);
  child_cands->add_possible_values()->set_real(2);
  child_cands->add_possible_values()->set_real(3);
  child_cands->add_possible_values()->set_real(4);
  child_cands->add_possible_values()->set_real(5);
}

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
