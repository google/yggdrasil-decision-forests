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

#include "yggdrasil_decision_forests/learner/isolation_forest/isolation_forest.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/generic_parameters.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/isolation_forest/isolation_forest.pb.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/model/isolation_forest/isolation_forest.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests::model::isolation_forest {

namespace {

using ::yggdrasil_decision_forests::model::isolation_forest::internal::
    Configuration;
using ::yggdrasil_decision_forests::model::isolation_forest::internal::
    DefaultMaximumDepth;
using ::yggdrasil_decision_forests::model::isolation_forest::internal::
    GetNumExamplesPerTrees;

// Assembles and checks the configuration.
absl::StatusOr<internal::Configuration> BuildConfig(
    const IsolationForestLearner& learner,
    const dataset::proto::DataSpecification& data_spec,
    const UnsignedExampleIdx num_training_examples) {
  internal::Configuration config;

  config.training_config = learner.training_config();
  config.if_config = config.training_config.MutableExtension(
      isolation_forest::proto::isolation_forest_config);

  RETURN_IF_ERROR(AbstractLearner::LinkTrainingConfig(
      config.training_config, data_spec, &config.config_link));

  if (config.training_config.task() != model::proto::Task::ANOMALY_DETECTION) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The ISOLATION_FOREST learner does not support the task ",
        model::proto::Task_Name(config.training_config.task()), "."));
  }

  decision_tree::SetDefaultHyperParameters(
      config.if_config->mutable_decision_tree());

  if (!config.if_config->decision_tree().has_max_depth() ||
      config.if_config->decision_tree().max_depth() == -2) {
    const auto num_examples_per_trees =
        GetNumExamplesPerTrees(*config.if_config, num_training_examples);
    config.if_config->mutable_decision_tree()->set_max_depth(
        DefaultMaximumDepth(num_examples_per_trees));
  }

  if (!config.if_config->decision_tree().has_min_examples()) {
    config.if_config->mutable_decision_tree()->set_min_examples(1);
  }

  RETURN_IF_ERROR(learner.CheckConfiguration(data_spec, config.training_config,
                                             config.config_link,
                                             learner.deployment()));

  if (config.config_link.has_weight_definition()) {
    return absl::InvalidArgumentError(
        "Isolation forest does not support weights");
  }
  return config;
}

// Check if this feature can be split.
template <typename T>
absl::StatusOr<bool> CanSplit(
    const dataset::VerticalDataset& train_dataset, int feature_idx,
    typename T::Format na_replacement,
    const std::vector<UnsignedExampleIdx>& selected_examples) {
  ASSIGN_OR_RETURN(const T* value_container,
                   train_dataset.ColumnWithCastWithStatus<T>(feature_idx));
  DCHECK_GT(selected_examples.size(), 1);
  const auto& values = value_container->values();
  float first_example = value_container->IsNa(selected_examples[0])
                            ? na_replacement
                            : values[selected_examples[0]];
  for (const auto example_idx : selected_examples) {
    float current_example = value_container->IsNa(example_idx)
                                ? na_replacement
                                : values[example_idx];
    if (first_example != current_example) {
      return true;
    }
  }
  return false;
}

// Compute the indices of the features that can be split, i.e. where not all
// values are equal. The indices of the nontrivial features are stored, by
// column type, in `nontrivial_features`. The total number of nontrivial
// features is returned.
absl::StatusOr<int> FindNontrivialFeatures(
    const Configuration& config, const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    absl::flat_hash_map<dataset::proto::ColumnType, std::vector<int>>*
        nontrivial_features) {
  DCHECK_GT(selected_examples.size(), 1);
  nontrivial_features->clear();
  size_t num_nontrivial_features = 0;
  const auto& data_spec = train_dataset.data_spec();
  for (const auto& feature_idx : config.config_link.features()) {
    const auto& feature = data_spec.columns(feature_idx);
    const auto feature_type = feature.type();
    bool can_split;
    switch (feature_type) {
      case dataset::proto::NUMERICAL: {
        ASSIGN_OR_RETURN(can_split,
                         CanSplit<dataset::VerticalDataset::NumericalColumn>(
                             train_dataset, feature_idx,
                             feature.numerical().mean(), selected_examples));
        break;
      }
      case dataset::proto::BOOLEAN: {
        const bool na_replacement =
            feature.boolean().count_true() >= feature.boolean().count_false();
        ASSIGN_OR_RETURN(
            can_split,
            CanSplit<dataset::VerticalDataset::BooleanColumn>(
                train_dataset, feature_idx, na_replacement, selected_examples));
        break;
      }
      case dataset::proto::CATEGORICAL: {
        const int64_t na_replacement =
            feature.categorical().most_frequent_value();
        ASSIGN_OR_RETURN(
            can_split,
            CanSplit<dataset::VerticalDataset::CategoricalColumn>(
                train_dataset, feature_idx, na_replacement, selected_examples));
        break;
      }
      case dataset::proto::CATEGORICAL_SET:
        LOG_FIRST_N(INFO, 1) << "Ignoring columns of unsupported type "
                             << dataset::proto::ColumnType_Name(feature.type())
                             << " e.g. " << feature.name();
        continue;
      case dataset::proto::DISCRETIZED_NUMERICAL:
        LOG_FIRST_N(INFO, 1) << "Ignoring columns of unsupported type "
                             << dataset::proto::ColumnType_Name(feature.type())
                             << " e.g. " << feature.name();
        continue;
      case dataset::proto::HASH:
        LOG_FIRST_N(INFO, 1) << "Ignoring columns of unsupported type "
                             << dataset::proto::ColumnType_Name(feature.type())
                             << " e.g. " << feature.name();
        continue;
      default:
        return absl::InvalidArgumentError(absl::Substitute(
            "Unsupported type $0 for feature $1",
            dataset::proto::ColumnType_Name(feature.type()), feature.name()));
    }
    if (can_split) {
      (*nontrivial_features)[feature_type].push_back(feature_idx);
      num_nontrivial_features++;
    }
  }
  return num_nontrivial_features;
}
}  // namespace

namespace internal {

absl::StatusOr<bool> FindSplit(
    const Configuration& config, const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    decision_tree::NodeWithChildren* node, utils::RandomEngine* rnd) {
  if (selected_examples.size() <= 1) {
    return false;
  }
  // TODO: Consider getting rid of this heap allocation
  absl::flat_hash_map<dataset::proto::ColumnType, std::vector<int>>
      nontrivial_features;

  ASSIGN_OR_RETURN(
      const auto num_nontrivial_features,
      FindNontrivialFeatures(config, train_dataset, selected_examples,
                             &nontrivial_features));
  if (num_nontrivial_features == 0) {
    return false;
  }
  // Sample a non-trivial feature type uniformly among the features. Note that
  // the number of column types is a small constant, so this algorithm is
  // reasonable.
  dataset::proto::ColumnType selected_feature_type;
  // For oblique numerical splits, this will not be used.
  int selected_feature_idx;
  const size_t random_idx =
      absl::Uniform<size_t>(*rnd, 0, num_nontrivial_features);
  size_t current_idx = 0;
  for (const auto& features_of_type : nontrivial_features) {
    if (random_idx < current_idx + features_of_type.second.size()) {
      selected_feature_type = features_of_type.first;
      selected_feature_idx = features_of_type.second[random_idx - current_idx];
      break;
    }
    current_idx += features_of_type.second.size();
  }
  DCHECK_NE(selected_feature_type, dataset::proto::ColumnType::UNKNOWN);
  switch (selected_feature_type) {
    case dataset::proto::NUMERICAL: {
      const auto& nontrivial_numerical_features =
          nontrivial_features.find(selected_feature_type);
      DCHECK(nontrivial_numerical_features != nontrivial_features.end());
      switch (config.if_config->decision_tree().split_axis_case()) {
        case decision_tree::proto::DecisionTreeTrainingConfig::SplitAxisCase::
            SPLIT_AXIS_NOT_SET:
        case decision_tree::proto::DecisionTreeTrainingConfig::SplitAxisCase::
            kAxisAlignedSplit:
          RETURN_IF_ERROR(SetRandomSplitNumericalAxisAligned(
              selected_feature_idx, config, train_dataset, selected_examples,
              node, rnd));
          break;
        case decision_tree::proto::DecisionTreeTrainingConfig::SplitAxisCase::
            kSparseObliqueSplit: {
          RETURN_IF_ERROR(SetRandomSplitNumericalSparseOblique(
              nontrivial_numerical_features->second, config, train_dataset,
              selected_examples, node, rnd));
          break;
        }
        default:
          return absl::InvalidArgumentError(
              "Only axis-aligned and sparse oblique splits are supported for "
              "isolation forests.");
      }
      break;
    }
    case dataset::proto::BOOLEAN: {
      const auto& nontrivial_boolean_features =
          nontrivial_features.find(selected_feature_type);
      DCHECK(nontrivial_boolean_features != nontrivial_features.end());
      RETURN_IF_ERROR(FindSplitBoolean(selected_feature_idx, config,
                                       train_dataset, selected_examples, node,
                                       rnd));
      break;
    }
    case dataset::proto::CATEGORICAL: {
      const auto& nontrivial_categorical_features =
          nontrivial_features.find(selected_feature_type);
      DCHECK(nontrivial_categorical_features != nontrivial_features.end());
      RETURN_IF_ERROR(FindSplitCategorical(selected_feature_idx, config,
                                           train_dataset, selected_examples,
                                           node, rnd));
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::Substitute(
          "Unsupported type $0",
          dataset::proto::ColumnType_Name(selected_feature_type)));
  }
  return true;
}

absl::Status SetRandomSplitNumericalSparseOblique(
    const std::vector<int>& nontrivial_features, const Configuration& config,
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    decision_tree::NodeWithChildren* node, utils::RandomEngine* rnd) {
  decision_tree::internal::Projection current_projection;
  const float projection_density = config.if_config->decision_tree()
                                       .sparse_oblique_split()
                                       .projection_density_factor() /
                                   config.config_link.numerical_features_size();
  decision_tree::internal::ProjectionEvaluator projection_evaluator(
      train_dataset, config.config_link.numerical_features());
  std::vector<float> projection_values;
  // An oblique split can be invalid even if all the features involved in the
  // split are nontrivial, if the weights are chosen so that features cancel
  // each other out. This is unlikely to happen. This function sets a
  // high number of trials and fails if it consistently fails to find a split
  // (which is likely indicative of an issue with the splitter).
  const int maximum__num_trials = 100 * nontrivial_features.size();
  int8_t unused_monotonic_direction;
  
  for (int i = 0; i < maximum__num_trials; i++) {
    decision_tree::internal::SampleProjection(
        nontrivial_features, config.if_config->decision_tree(),
        train_dataset.data_spec(), config.config_link, projection_density,
        &current_projection, &unused_monotonic_direction, rnd);

    // Pre-compute the result of the current_projection.
    RETURN_IF_ERROR(projection_evaluator.Evaluate(
        current_projection, selected_examples, &projection_values));

    // Find minimum and maximum value.
    int num_valid_examples = 0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    for (const auto value : projection_values) {
      if (!std::isnan(value)) {
        num_valid_examples++;
        if (value < min_value) {
          min_value = value;
        }
        if (value > max_value) {
          max_value = value;
        }
      }
    }
    if (num_valid_examples == 0 || min_value == max_value) {
      // Invalid split, try again.
      continue;
    }

    // Randomly select a threshold in (min_value, max_value).
    const float threshold = std::uniform_real_distribution<float>(
        std::nextafter(min_value, std::numeric_limits<float>::max()),
        max_value)(*rnd);
    DCHECK_GT(threshold, min_value);
    DCHECK_LE(threshold, max_value);

    // Count the number of positive examples.
    UnsignedExampleIdx num_pos_examples = 0;
    for (const auto value : projection_values) {
      if (value >= threshold) {
        num_pos_examples++;
      }
    }

    DCHECK_GT(num_pos_examples, 0);
    DCHECK_LT(num_pos_examples, selected_examples.size());

    // Set split.
    auto* condition = node->mutable_node()->mutable_condition();
    RETURN_IF_ERROR(decision_tree::internal::SetCondition(
        current_projection, threshold, train_dataset.data_spec(), condition));
    condition->set_num_training_examples_without_weight(
        selected_examples.size());
    condition->set_num_pos_training_examples_without_weight(num_pos_examples);
    return absl::OkStatus();
  }
  return absl::InternalError(absl::Substitute(
      "No valid oblique split found after $0 tries. This indicates an issue "
      "with the oblique Isolation Forest splitter.",
      maximum__num_trials));
}

absl::Status SetRandomSplitNumericalAxisAligned(
    const int feature_idx, const Configuration& config,
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    decision_tree::NodeWithChildren* node, utils::RandomEngine* rnd) {
  auto& col_spec = train_dataset.data_spec().columns(feature_idx);
  DCHECK_EQ(col_spec.type(), dataset::proto::NUMERICAL);
  DCHECK_GT(selected_examples.size(), 1);

  ASSIGN_OR_RETURN(
      const dataset::VerticalDataset::NumericalColumn* value_container,
      train_dataset
          .ColumnWithCastWithStatus<dataset::VerticalDataset::NumericalColumn>(
              feature_idx));
  const auto& values = value_container->values();
  const float na_replacement = col_spec.numerical().mean();

  // Check if this feature can be split.
  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();
  for (const auto example_idx : selected_examples) {
    auto value = values[example_idx];
    if (value_container->IsNa(example_idx)) {
      value = na_replacement;
    }
    if (value < min_value) {
      min_value = value;
    }
    if (value > max_value) {
      max_value = value;
    }
  }
  // `feature_idx` must be a nontrivial feature.
  DCHECK_LT(min_value, max_value);
  // Randomly select a threshold in (min_value, max_value).
  const float threshold = std::uniform_real_distribution<float>(
      std::nextafter(min_value, std::numeric_limits<float>::max()),
      max_value)(*rnd);
  DCHECK_GT(threshold, min_value);
  DCHECK_LE(threshold, max_value);

  // Count the number of positive examples.
  UnsignedExampleIdx num_pos_examples = 0;
  for (const auto example_idx : selected_examples) {
    auto value = values[example_idx];
    if (value_container->IsNa(example_idx)) {
      value = na_replacement;
    }
    if (value >= threshold) {
      num_pos_examples++;
    }
  }

  DCHECK_GT(num_pos_examples, 0);
  DCHECK_LT(num_pos_examples, selected_examples.size());

  // Set split.
  auto* condition = node->mutable_node()->mutable_condition();
  condition->set_attribute(feature_idx);
  condition->mutable_condition()->mutable_higher_condition()->set_threshold(
      threshold);
  condition->set_na_value(na_replacement >= threshold);
  condition->set_num_training_examples_without_weight(selected_examples.size());
  condition->set_num_pos_training_examples_without_weight(num_pos_examples);
  return absl::OkStatus();
}

absl::Status FindSplitBoolean(
    const int feature_idx, const Configuration& config,
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    decision_tree::NodeWithChildren* node, utils::RandomEngine* rnd) {
  auto& col_spec = train_dataset.data_spec().columns(feature_idx);
  DCHECK_EQ(col_spec.type(), dataset::proto::BOOLEAN);
  DCHECK_GT(selected_examples.size(), 0);

  // Positive values go to the positive branch, negative go to the negative
  // branch, NA goes to the majority side.
  ASSIGN_OR_RETURN(
      const auto* value_container,
      train_dataset
          .ColumnWithCastWithStatus<dataset::VerticalDataset::BooleanColumn>(
              feature_idx));
  const bool na_replacement =
      col_spec.boolean().count_true() >= col_spec.boolean().count_false();
  UnsignedExampleIdx num_pos_examples = 0;
  for (const auto example_idx : selected_examples) {
    if (value_container->IsTrue(example_idx) ||
        (na_replacement && value_container->IsNa(example_idx))) {
      num_pos_examples++;
    }
  }

  DCHECK_GT(num_pos_examples, 0);
  DCHECK_LT(num_pos_examples, selected_examples.size());

  // Set split.
  auto* condition = node->mutable_node()->mutable_condition();
  condition->set_attribute(feature_idx);
  condition->mutable_condition()->mutable_true_value_condition();
  condition->set_na_value(na_replacement);
  condition->set_num_training_examples_without_weight(selected_examples.size());
  condition->set_num_pos_training_examples_without_weight(num_pos_examples);
  return absl::OkStatus();
}

absl::Status FindSplitCategorical(
    const int feature_idx, const Configuration& config,
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    decision_tree::NodeWithChildren* node, utils::RandomEngine* rnd) {
  const auto& col_spec = train_dataset.data_spec().columns(feature_idx);
  const auto na_replacement = col_spec.categorical().most_frequent_value();
  DCHECK_EQ(col_spec.type(), dataset::proto::CATEGORICAL);
  DCHECK_GT(selected_examples.size(), 0);

  ASSIGN_OR_RETURN(
      const auto* value_container,
      train_dataset.ColumnWithCastWithStatus<
          dataset::VerticalDataset::CategoricalColumn>(feature_idx));
  const auto& values = value_container->values();
  const int num_unique_feature_values =
      col_spec.categorical().number_of_unique_values();

  // if num_unique_feature_values is very large (likely because of a user
  // feeding pre-integerized values), the following might be memory-hungry.
  std::vector<UnsignedExampleIdx> active_feature_values_count(
      num_unique_feature_values, 0);
  int num_active_feature_values = 0;
  for (const auto example_idx : selected_examples) {
    auto value = values[example_idx];
    if (value == dataset::VerticalDataset::CategoricalColumn::kNaValue) {
      value = na_replacement;
    }
    if (active_feature_values_count[value] == 0) {
      num_active_feature_values++;
    }
    active_feature_values_count[value]++;
  }
  DCHECK_GT(num_active_feature_values, 0);
  int ensure_chosen_idx = absl::Uniform(*rnd, 0, num_active_feature_values);
  int ensure_not_chosen_idx =
      (ensure_chosen_idx +
       absl::Uniform(*rnd, 1, num_active_feature_values - 1)) %
      num_active_feature_values;
  std::vector<int> chosen_values = {};
  chosen_values.reserve(num_unique_feature_values);

  int selected_values_idx = 0;
  int num_pos_examples = 0;
  bool na_is_chosen = false;
  // Flip a fair coin for every value in the selected examples.
  for (int item = 0; item < num_unique_feature_values; item++) {
    if (active_feature_values_count[item] > 0) {
      bool choose;
      if (selected_values_idx == ensure_not_chosen_idx) {
        choose = false;
      } else if (selected_values_idx == ensure_chosen_idx) {
        choose = true;
      } else {
        choose = absl::Bernoulli(*rnd, 0.5);
      }
      if (choose) {
        chosen_values.push_back(item);
        num_pos_examples += active_feature_values_count[item];
        if (item == na_replacement) {
          na_is_chosen = true;
        }
      }
      selected_values_idx++;
    } else {
      // Randomly pick from the non-observed values for a more balanced split.
      if (absl::Bernoulli(*rnd, 0.5)) {
        chosen_values.push_back(item);
        if (item == na_replacement) {
          na_is_chosen = true;
        }
      }
    }
  }
  DCHECK(!chosen_values.empty());
  DCHECK_GT(num_pos_examples, 0);
  DCHECK_LT(num_pos_examples, selected_examples.size());

  // Set split.
  auto* condition = node->mutable_node()->mutable_condition();
  condition->set_attribute(feature_idx);
  decision_tree::SetPositiveAttributeSetOfCategoricalContainsCondition(
      chosen_values, num_unique_feature_values, condition);
  condition->set_na_value(na_is_chosen);
  condition->set_num_training_examples_without_weight(selected_examples.size());
  condition->set_num_pos_training_examples_without_weight(num_pos_examples);
  return absl::OkStatus();
}

// Grows recursively a node.
absl::Status GrowNode(const Configuration& config,
                      const dataset::VerticalDataset& train_dataset,
                      const std::vector<UnsignedExampleIdx>& selected_examples,
                      const int depth, decision_tree::NodeWithChildren* node,
                      utils::RandomEngine* rnd) {
  if (selected_examples.empty()) {
    return absl::InternalError("No examples fed to the node trainer");
  }

  const auto& dt_config = config.if_config->decision_tree();

  // Set node value
  node->mutable_node()->set_num_pos_training_examples_without_weight(
      selected_examples.size());
  node->mutable_node()
      ->mutable_anomaly_detection()
      ->set_num_examples_without_weight(selected_examples.size());

  // Stop growth
  if (selected_examples.size() < dt_config.min_examples() ||
      (dt_config.max_depth() >= 0 && depth >= dt_config.max_depth())) {
    node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
    return absl::OkStatus();
  }

  // Look for a split
  ASSIGN_OR_RETURN(
      const bool found_condition,
      FindSplit(config, train_dataset, selected_examples, node, rnd));

  if (!found_condition) {
    // No split found
    node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
    return absl::OkStatus();
  }

  // Turn the node into a non-leaf node
  STATUS_CHECK_EQ(
      selected_examples.size(),
      node->node().condition().num_training_examples_without_weight());
  node->CreateChildren();
  node->FinalizeAsNonLeaf(dt_config.keep_non_leaf_label_distribution(),
                          dt_config.store_detailed_label_distribution());

  // Branch examples to children
  // TODO: Use cache to avoid re-allocating selected example
  // buffers.
  std::vector<UnsignedExampleIdx> positive_examples;
  std::vector<UnsignedExampleIdx> negative_examples;
  RETURN_IF_ERROR(decision_tree::internal::SplitExamples(
      train_dataset, selected_examples, node->node().condition(), false,
      dt_config.internal_error_on_wrong_splitter_statistics(),
      &positive_examples, &negative_examples));

  // Split children
  RETURN_IF_ERROR(GrowNode(config, train_dataset, positive_examples, depth + 1,
                           node->mutable_pos_child(), rnd));
  positive_examples = {};  // Release memory of "positive_examples".
  RETURN_IF_ERROR(GrowNode(config, train_dataset, negative_examples, depth + 1,
                           node->mutable_neg_child(), rnd));
  return absl::OkStatus();
}

// Grows and return a tree.
absl::StatusOr<std::unique_ptr<decision_tree::DecisionTree>> GrowTree(
    const Configuration& config, const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    utils::RandomEngine* rnd) {
  auto tree = std::make_unique<decision_tree::DecisionTree>();
  tree->CreateRoot();
  RETURN_IF_ERROR(GrowNode(config, train_dataset, selected_examples,
                           /*depth=*/0, tree->mutable_root(), rnd));
  return std::move(tree);
}

int DefaultMaximumDepth(UnsignedExampleIdx num_examples_per_trees) {
  return std::ceil(std::log2(num_examples_per_trees));
}

std::vector<UnsignedExampleIdx> SampleExamples(
    const UnsignedExampleIdx num_examples,
    const UnsignedExampleIdx num_examples_to_sample, utils::RandomEngine* rnd) {
  if (num_examples_to_sample < num_examples / 2) {
    // If the number of examples is not too large, use Floyd's algorithm for
    // sampling `num_examples_to_sample` examples from range [0, num_examples).
    // https://doi.org/10.1145/30401.315746

    absl::btree_set<UnsignedExampleIdx> sampled_examples;
    for (UnsignedExampleIdx j = num_examples - num_examples_to_sample;
         j < num_examples; j++) {
      UnsignedExampleIdx t = absl::Uniform<UnsignedExampleIdx>(*rnd, 0, j + 1);
      if (!sampled_examples.insert(t).second) {
        sampled_examples.insert(j);
      }
    }
    std::vector<UnsignedExampleIdx> examples(sampled_examples.begin(),
                                             sampled_examples.end());
    return {sampled_examples.begin(), sampled_examples.end()};
  }
  std::vector<UnsignedExampleIdx> examples(num_examples);
  std::iota(examples.begin(), examples.end(), 0);
  std::shuffle(examples.begin(), examples.end(), *rnd);
  examples.resize(num_examples_to_sample);
  examples.shrink_to_fit();
  std::sort(examples.begin(), examples.end());
  return examples;
}

SignedExampleIdx GetNumExamplesPerTrees(
    const proto::IsolationForestTrainingConfig& if_config,
    const SignedExampleIdx num_training_examples) {
  switch (if_config.sampling_method_case()) {
    case proto::IsolationForestTrainingConfig::kSubsampleRatio:
      return static_cast<SignedExampleIdx>(
          std::ceil(static_cast<double>(if_config.subsample_ratio()) *
                    num_training_examples));
    default:
      return if_config.subsample_count();
  }
}

}  // namespace internal

IsolationForestLearner::IsolationForestLearner(
    const model::proto::TrainingConfig& training_config)
    : AbstractLearner(training_config) {}

absl::Status IsolationForestLearner::SetHyperParametersImpl(
    utils::GenericHyperParameterConsumer* generic_hyper_params) {
  RETURN_IF_ERROR(
      AbstractLearner::SetHyperParametersImpl(generic_hyper_params));
  const auto& if_config = training_config_.MutableExtension(
      isolation_forest::proto::isolation_forest_config);

  // Decision tree specific hyper-parameters.
  absl::flat_hash_set<std::string> consumed_hparams;
  RETURN_IF_ERROR(decision_tree::SetHyperParameters(
      &consumed_hparams, if_config->mutable_decision_tree(),
      generic_hyper_params));

  {
    const auto hparam = generic_hyper_params->Get(kHParamNumTrees);
    if (hparam.has_value()) {
      if_config->set_num_trees(hparam.value().value().integer());
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamSubsampleRatio);
    if (hparam.has_value()) {
      if_config->set_subsample_ratio(hparam.value().value().real());
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamSubsampleCount);
    if (hparam.has_value()) {
      if_config->set_subsample_count(hparam.value().value().integer());
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<model::proto::GenericHyperParameterSpecification>
IsolationForestLearner::GetGenericHyperParameterSpecification() const {
  absl::flat_hash_set<std::string> valid_decision_tree_hyperparameters = {
      kHParamRandomSeed,
      kHParamPureServingModel,
      decision_tree::kHParamMaxDepth,
      decision_tree::kHParamMinExamples,
      decision_tree::kHParamSplitAxis,
      decision_tree::kHParamSplitAxisSparseObliqueProjectionDensityFactor,
      decision_tree::kHParamSplitAxisSparseObliqueNormalization,
      decision_tree::kHParamSplitAxisSparseObliqueWeights,
      decision_tree::kHParamSplitAxisSparseObliqueMaxNumFeatures,
      decision_tree::kHParamSplitAxisSparseObliqueWeightsPowerOfTwoMinExponent,
      decision_tree::kHParamSplitAxisSparseObliqueWeightsPowerOfTwoMaxExponent,
      decision_tree::kHParamSplitAxisSparseObliqueWeightsIntegerMinimum,
      decision_tree::kHParamSplitAxisSparseObliqueWeightsIntegerMaximum,
  };
  // Remove not yet implemented hyperparameters
  // TODO: b/345425508 - Implement more hyperparameters for isolation forests.
  absl::flat_hash_set<std::string> invalid_decision_tree_hyperparameters = {
      kHParamMaximumModelSizeInMemoryInBytes,
      kHParamMaximumTrainingDurationSeconds,
      decision_tree::kHParamGrowingStrategy,
      decision_tree::kHParamMaxNumNodes,
      decision_tree::kHParamNumCandidateAttributes,
      decision_tree::kHParamNumCandidateAttributesRatio,
      decision_tree::kHParamInSplitMinExampleCheck,
      decision_tree::kHParamAllowNaConditions,
      decision_tree::kHParamMissingValuePolicy,
      decision_tree::kHParamCategoricalSetSplitGreedySampling,
      decision_tree::kHParamCategoricalSetSplitMaxNumItems,
      decision_tree::kHParamCategoricalSetSplitMinItemFrequency,
      decision_tree::kHParamSplitAxisSparseObliqueNumProjectionsExponent,
      decision_tree::kHParamSplitAxisSparseObliqueMaxNumProjections,
      decision_tree::kHParamSplitAxisMhldObliqueMaxNumAttributes,
      decision_tree::kHParamSplitAxisMhldObliqueSampleAttributes,
      decision_tree::kHParamCategoricalAlgorithm,
      decision_tree::kHParamSortingStrategy,
      decision_tree::kHParamKeepNonLeafLabelDistribution,
      decision_tree::kHParamUpliftSplitScore,
      decision_tree::kHParamUpliftMinExamplesInTreatment,
      decision_tree::kHParamHonest,
      decision_tree::kHParamHonestRatioLeafExamples,
      decision_tree::kHParamHonestFixedSeparation};

  ASSIGN_OR_RETURN(auto hparam_def,
                   AbstractLearner::GetGenericHyperParameterSpecification());
  model::proto::TrainingConfig config;
  const auto proto_path = "learner/isolation_forest/isolation_forest.proto";

  hparam_def.mutable_documentation()->set_description(
      R"(An [Isolation Forest](https://ieeexplore.ieee.org/abstract/document/4781136) is a collection of decision trees trained without labels and independently to partition the feature space. The Isolation Forest prediction is an anomaly score that indicates whether an example originates from a same distribution to the training examples. We refer to Isolation Forest as both the original algorithm by Liu et al. and its extensions.)");

  const auto& if_config =
      config.GetExtension(isolation_forest::proto::isolation_forest_config);

  RETURN_IF_ERROR(decision_tree::GetGenericHyperParameterSpecification(
      if_config.decision_tree(), &hparam_def,
      valid_decision_tree_hyperparameters,
      invalid_decision_tree_hyperparameters));

  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamNumTrees);
    param.mutable_integer()->set_minimum(0);
    param.mutable_integer()->set_default_value(if_config.num_trees());
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Number of individual decision trees. Increasing the number of trees can increase the quality of the model at the expense of size, training speed, and inference latency.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamSubsampleCount);
    param.mutable_integer()->set_minimum(0);
    param.mutable_integer()->set_default_value(if_config.subsample_count());
    param.mutable_mutual_exclusive()->set_is_default(true);
    param.mutable_mutual_exclusive()->add_other_parameters(
        kHParamSubsampleRatio);
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Number of examples used to grow each tree. Only one of "subsample_ratio" and "subsample_count" can be set. By default, sample 256 examples per tree. Note that this parameter also restricts the tree's maximum depth to log2(examples used per tree) unless max_depth is set explicitly.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamSubsampleRatio);
    param.mutable_real()->set_minimum(0);
    param.mutable_real()->set_default_value(1.0);
    param.mutable_mutual_exclusive()->set_is_default(false);
    param.mutable_mutual_exclusive()->add_other_parameters(
        kHParamSubsampleCount);
    param.mutable_documentation()->set_proto_path(proto_path);
    param.mutable_documentation()->set_description(
        R"(Ratio of number of training examples used to grow each tree. Only one of "subsample_ratio" and "subsample_count" can be set. By default, sample 256 examples per tree. Note that this parameter also restricts the tree's maximum depth to log2(examples used per tree) unless max_depth is set explicitly.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](decision_tree::kHParamMaxDepth);
    param.mutable_integer()->set_minimum(-2);
    param.mutable_integer()->set_default_value(-2);
    param.mutable_documentation()->set_description(
        R"(Maximum depth of the tree. `max_depth=1` means that all trees will be roots. `max_depth=-1` means that tree depth unconstrained by this parameter. `max_depth=-2` means that the maximum depth is log2(number of sampled examples per tree) (default).)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](
        decision_tree::kHParamSplitAxis);
    param.mutable_categorical()->set_default_value(
        decision_tree::kHParamSplitAxisAxisAligned);
    param.mutable_categorical()->clear_possible_values();
    param.mutable_categorical()->add_possible_values(
        decision_tree::kHParamSplitAxisAxisAligned);
    param.mutable_categorical()->add_possible_values(
        decision_tree::kHParamSplitAxisSparseOblique);
    param.mutable_documentation()->set_description(
        R"(What structure of split to consider for numerical features.
- `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This is the "classical" way to train a tree. Default value.
- `SPARSE_OBLIQUE`: Sparse oblique splits (i.e. random splits on a small number of features) from "Sparse Projection Oblique Random Forests", Tomita et al., 2020. This includes the splits described in "Extended Isolation Forests" (Sahand Hariri et al., 2018).)");
  }

  return hparam_def;
}

absl::StatusOr<std::unique_ptr<AbstractModel>>
IsolationForestLearner::TrainWithStatusImpl(
    const dataset::VerticalDataset& train_dataset,
    std::optional<std::reference_wrapper<const dataset::VerticalDataset>>
        valid_dataset) const {
  RETURN_IF_ERROR(dataset::CheckNumExamples(train_dataset.nrow()));

  ASSIGN_OR_RETURN(
      const internal::Configuration config,
      BuildConfig(*this, train_dataset.data_spec(), train_dataset.nrow()));

  auto model = std::make_unique<IsolationForestModel>();
  InitializeModelWithAbstractTrainingConfig(config.training_config,
                                            config.config_link, model.get());
  model->set_data_spec(train_dataset.data_spec());
  model->set_num_examples_per_trees(
      GetNumExamplesPerTrees(*config.if_config, train_dataset.nrow()));

  LOG(INFO) << "Training isolation forest on " << train_dataset.nrow()
            << " example(s) and " << config.config_link.features_size()
            << " feature(s).";

  utils::RandomEngine global_random(config.training_config.random_seed());

  absl::Status global_status;
  utils::concurrency::Mutex global_mutex;
  {
    yggdrasil_decision_forests::utils::concurrency::ThreadPool pool(
        deployment().num_threads(), {.name_prefix = std::string("TrainIF")});
    pool.StartWorkers();
    const auto num_trees = config.if_config->num_trees();
    model->mutable_decision_trees()->resize(num_trees);
    for (int tree_idx = 0; tree_idx < num_trees; tree_idx++) {
      pool.Schedule([&train_dataset, &model, &config, tree_idx, &global_status,
                     &global_mutex, seed = global_random()]() {
        {
          utils::concurrency::MutexLock lock(&global_mutex);
          if (!global_status.ok()) {
            return;
          }
        }
        utils::RandomEngine local_random(seed);
        const auto selected_examples = internal::SampleExamples(
            train_dataset.nrow(), model->num_examples_per_trees(),
            &local_random);
        DCHECK(
            std::is_sorted(selected_examples.begin(), selected_examples.end()));
        DCHECK_EQ(selected_examples.size(), model->num_examples_per_trees());
        auto tree_or =
            GrowTree(config, train_dataset, selected_examples, &local_random);
        if (!tree_or.ok()) {
          utils::concurrency::MutexLock lock(&global_mutex);
          global_status.Update(tree_or.status());
          return;
        }
        (*model->mutable_decision_trees())[tree_idx] = std::move(*tree_or);
      });
    }
  }
  RETURN_IF_ERROR(global_status);
  decision_tree::SetLeafIndices(model->mutable_decision_trees());
  return std::move(model);
}

}  // namespace yggdrasil_decision_forests::model::isolation_forest
