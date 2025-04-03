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

#include "yggdrasil_decision_forests/learner/decision_tree/training.h"

#include <stddef.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/label.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_accumulator.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_scanner.h"
#include "yggdrasil_decision_forests/learner/decision_tree/uplift.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/decision_tree/vector_sequence.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/cast.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::model::decision_tree {

namespace {

// Generates a failure absl status if the configuration contains monotonic
// constraints.
absl::Status FailIfMonotonic(
    const model::proto::TrainingConfigLinking& config_link,
    const int attribute_idx, const NodeConstraints& constraints,
    const absl::string_view why) {
  if (config_link.per_columns_size() > 0 &&
      (config_link.per_columns(attribute_idx).has_monotonic_constraint() ||
       constraints.min_max_output.has_value())) {
    return absl::InternalError(
        absl::StrCat("Monotonic constraints not supported for ", why));
  }
  return absl::OkStatus();
}

// Number of trials to run when learning a categorical split with randomly
// generated masks.
//
// Args:
//   config: A random categorical split learning configuration.
//
// Returns:
//   A function active_dictionary_size -> number_of_trials.
//
// The "active_dictionary_size" is the number of unique categorical values that
// are present at least once in the training examples of the node.
std::function<int(const int active_dictionary_size)>
NumTrialsForRandomCategoricalSplit(const proto::Categorical::Random& config) {
  const auto num_trial_exponent = config.num_trial_exponent();
  const auto max_num_trials = config.max_num_trials();
  return
      [num_trial_exponent, max_num_trials](const int active_dictionary_size) {
        const int num_trials =
            32 + std::pow(active_dictionary_size, num_trial_exponent);
        return std::min(num_trials, max_num_trials);
      };
}

// Helper function to set a condition statistics. Do not set the following
// fields: "mutable_condition" and "na_value".
void SetConditionHelper(
    const double information_gain, const int32_t attribute_idx,
    const utils::BinaryToIntegerConfusionMatrixDouble& running_confusion,
    const utils::BinaryToIntegerConfusionMatrixInt64&
        running_confusion_no_weights,
    proto::NodeCondition* condition) {
  condition->set_split_score(information_gain);
  condition->set_attribute(attribute_idx);
  condition->set_num_pos_training_examples_without_weight(
      running_confusion_no_weights.pos().NumObservations());
  condition->set_num_pos_training_examples_with_weight(
      running_confusion.pos().NumObservations());
  condition->set_num_training_examples_without_weight(
      running_confusion_no_weights.NumObservations());
  condition->set_num_training_examples_with_weight(
      running_confusion.NumObservations());
}

// Helper function to set a condition statistics.
// Similar as "SetConditionHelper" above, but for a regression problem.
void SetConditionHelper(
    const double variance_reduction, const int32_t attribute_idx,
    const utils::BinaryToNormalDistributionDouble& running_confusion,
    const utils::BinaryToNormalDistributionDouble& running_confusion_no_weights,
    proto::NodeCondition* condition) {
  condition->set_split_score(variance_reduction);
  condition->set_attribute(attribute_idx);
  condition->set_num_pos_training_examples_without_weight(
      running_confusion_no_weights.pos().NumObservations());
  condition->set_num_pos_training_examples_with_weight(
      running_confusion.pos().NumObservations());
  condition->set_num_training_examples_without_weight(
      running_confusion_no_weights.NumObservations());
  condition->set_num_training_examples_with_weight(
      running_confusion.NumObservations());
}

// Computes and set in "na_replacement" the value to use as a replacement of
// missing values when the "local imputation" strategy is used.
//
// Explanation: The "local imputation" strategy to handle missing values
// consists in replacing these missing values by the mean of the feature in the
// training dataset.
//
// If the feature only contains missing values, the "na_replacement" argument is
// left unchanged.
//
// `weights` may be empty which is equivalent to unit weights.
void LocalImputationForNumericalAttribute(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    float* na_replacement) {
  double na_replacement_value_accumulator = 0;
  double na_replacement_weight_accumulator = 0;
  for (const auto example_idx : selected_examples) {
    const float attribute = attributes[example_idx];
    const float weight = weights.empty() ? 1.f : weights[example_idx];
    if (!std::isnan(attribute)) {
      na_replacement_value_accumulator += attribute * weight;
      na_replacement_weight_accumulator += weight;
    }
  }
  if (na_replacement_weight_accumulator > 0) {
    *na_replacement = static_cast<float>(na_replacement_value_accumulator /
                                         na_replacement_weight_accumulator);
  }
}

// Similar as "LocalImputationForNumericalAttribute", but for a categorical
// attribute. Return the most frequent attribute value.
void LocalImputationForCategoricalAttribute(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const int32_t num_attribute_classes, int32_t* na_replacement) {
  utils::IntegerDistributionDouble attribute_distribution;
  attribute_distribution.SetNumClasses(num_attribute_classes);
  for (const auto example_idx : selected_examples) {
    const auto attribute_value = attributes[example_idx];
    if (attribute_value !=
        dataset::VerticalDataset::CategoricalColumn::kNaValue) {
      const float weight = weights.empty() ? 1.f : weights[example_idx];
      attribute_distribution.Add(attribute_value, weight);
    }
  }
  if (attribute_distribution.NumObservations() > 0) {
    *na_replacement = attribute_distribution.TopClass();
  }
}

// Similar to "LocalImputationForCategoricalAttribute", but for a boolean
// attribute. Returns the most frequent attribute value.
void LocalImputationForBooleanAttribute(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    bool* na_replacement) {
  DCHECK(!weights.empty());
  utils::IntegerDistributionDouble attribute_distribution;
  attribute_distribution.SetNumClasses(2);
  for (const auto example_idx : selected_examples) {
    const auto attribute_value = attributes[example_idx];
    if (attribute_value != dataset::VerticalDataset::BooleanColumn::kNaValue) {
      const float weight = weights.empty() ? 1.f : weights[example_idx];
      attribute_distribution.Add(attribute_value, weight);
    }
  }
  if (attribute_distribution.NumObservations() > 0) {
    *na_replacement = attribute_distribution.TopClass();
  }
}

// Return the minimum and maximum values of a numerical attribute.
// Return false if there is no min-max e.g. selected_examples is empty or all
// the values are NAs.
bool MinMaxNumericalAttribute(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const absl::Span<const float> attributes, float* min_value,
    float* max_value) {
  float local_min_value = 0;
  float local_max_value = 0;
  bool first = true;
  for (const auto example_idx : selected_examples) {
    const float attribute = attributes[example_idx];
    if (first && !std::isnan(attribute)) {
      local_max_value = local_min_value = attribute;
      first = false;
    } else if (attribute > local_max_value) {
      local_max_value = attribute;
    } else if (attribute < local_min_value) {
      local_min_value = attribute;
    }
  }
  *min_value = local_min_value;
  *max_value = local_max_value;
  return !first;
}

// Search for the attribute item that maximize the immediate variance reduction.
template <bool weighted>
std::pair<int, double> GetAttributeValueWithMaximumVarianceReduction(
    const double variance_reduction, const int32_t num_attribute_classes,
    const utils::BinaryToNormalDistributionDouble& split_label_distribution,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<bool>& positive_selected_example_bitmap,
    const std::vector<std::pair<size_t, size_t>>& attribute_values,
    const std::vector<int>& attribute_bank, const std::vector<float>& weights,
    const std::vector<float>& labels, const double initial_variance,
    std::vector<UnsignedExampleIdx>* running_attr_bank_idx,
    std::vector<bool>* candidate_attributes_bitmap) {
  if (weighted) {
    DCHECK_EQ(weights.size(), labels.size());
  } else {
    DCHECK(weights.empty());
  }
  // Search for the attribute item that maximize the variance reduction. Note:
  // We ignore attribute that reduce the current variance reduction.
  double best_candidate_variance_reduction = variance_reduction;
  int best_attr_value = -1;
  for (int candidate_attr_value = 0;
       candidate_attr_value < num_attribute_classes; candidate_attr_value++) {
    if (!(*candidate_attributes_bitmap)[candidate_attr_value]) {
      // The attribute value was already selected or was excluded from the
      // search space.
      continue;
    }
    // Compute the variance reduction of the already selected values
    // "positive_attributes_vector" and the candidate value
    // "candidate_attr_value".
    utils::BinaryToNormalDistributionDouble candidate_split_label_distribution =
        split_label_distribution;
    int64_t num_absent_in_negative_set = 0;
    for (size_t select_idx = 0; select_idx < selected_examples.size();
         select_idx++) {
      const auto example_idx = selected_examples[select_idx];
      if (positive_selected_example_bitmap[select_idx]) {
        // The example is already in the positive set.
        continue;
      }

      // Search if X = attribute_bank[ attribute_values[example_idx].first,
      // attribute_values[example_idx].second ] contains the current candidate
      // attribute value "candidate_attr_value".
      //
      // We use "running_attr_bank_idx[select_idx]" that contains the index in
      // "X" of the last tested candidate attribute.
      //
      // Note: "X" does not contains duplicates and is sorted in increasing
      // order. The candidate attributes "candidate_attr_value" are testing in
      // increasing order.
      bool match = false;
      UnsignedExampleIdx last_attr;
      while (
          (*running_attr_bank_idx)[select_idx] <
              attribute_values[example_idx].second &&
          (last_attr = attribute_bank[(*running_attr_bank_idx)[select_idx]]) <=
              candidate_attr_value) {
        (*running_attr_bank_idx)[select_idx]++;
        if (last_attr == candidate_attr_value) {
          match = true;
          break;
        }
      }

      if (match) {
        // Add the example to the positive set and remove it from the
        // negative.
        if constexpr (weighted) {
          candidate_split_label_distribution.mutable_pos()->Add(
              labels[example_idx], weights[example_idx]);
          candidate_split_label_distribution.mutable_neg()->Add(
              labels[example_idx], -weights[example_idx]);
        } else {
          candidate_split_label_distribution.mutable_pos()->Add(
              labels[example_idx]);
          candidate_split_label_distribution.mutable_neg()->Sub(
              labels[example_idx]);
        }
      } else {
        num_absent_in_negative_set++;
      }
    }
    // Remove the attribute from the candidate set if the attribute is pure
    // for the current negative set.
    if (num_absent_in_negative_set == 0 ||
        num_absent_in_negative_set == selected_examples.size()) {
      (*candidate_attributes_bitmap)[candidate_attr_value] = false;
      continue;
    }

    const double candidate_variance_reduction =
        initial_variance - candidate_split_label_distribution.FinalVariance();
    if (candidate_variance_reduction > best_candidate_variance_reduction) {
      // Best score so far.
      best_candidate_variance_reduction = candidate_variance_reduction;
      best_attr_value = candidate_attr_value;
    }
  }
  return std::make_pair(best_attr_value, best_candidate_variance_reduction);
}

// Select which sorting strategy to use effectively.
//
// If the strategy is AUTO or PRESORTED, the fastest / selected strategy depends
// on the data.
proto::DecisionTreeTrainingConfig::Internal::SortingStrategy EffectiveStrategy(
    const proto::DecisionTreeTrainingConfig& dt_config,
    const int64_t num_selected_examples,
    const InternalTrainConfig& internal_config) {
  proto::DecisionTreeTrainingConfig::Internal::SortingStrategy strategy;

  if (internal_config.override_sorting_strategy.has_value()) {
    // The internal configuration configured by the learning algorithm takes
    // precedence on the sorting strategy.
    strategy = internal_config.override_sorting_strategy.value();
  } else {
    // Otherwise, the training configuration (controlled by the user or
    // unit-test controller) selects the strategy.
    strategy = dt_config.internal().sorting_strategy();
  }
  switch (strategy) {
    // User specified strategy.
    case proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED:
    case proto::DecisionTreeTrainingConfig::Internal::IN_NODE:
      return strategy;

    case proto::DecisionTreeTrainingConfig::Internal::AUTO:
      CHECK(false);  // The AUTO strategy should have been resolved before.
      break;
    case proto::DecisionTreeTrainingConfig::Internal::PRESORTED: {
      DCHECK(internal_config.preprocessing);
      const auto num_total_examples =
          internal_config.preprocessing->num_examples();
      const float ratio =
          static_cast<float>(num_selected_examples) / num_total_examples;
      return (num_selected_examples >= 25 && ratio >= 0.125)
                 ? proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED
                 : proto::DecisionTreeTrainingConfig::Internal::IN_NODE;
    }
  };
}

}  // namespace

// Specialization in the case of classification.
absl::StatusOr<SplitSearchResult> FindBestConditionClassification(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const ClassificationLabelStats& label_stats, const int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  if (dt_config.internal().generate_fake_error_in_splitter()) {
    return absl::InternalError("Fake error");
  }

  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  RETURN_IF_ERROR(FailIfMonotonic(config_link, attribute_idx, constraints,
                                  "classification"));

  SplitSearchResult result;

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::NumericalColumn>(attribute_idx));

      const auto na_replacement = attribute_column_spec.numerical().mean();
      if (dt_config.numerical_split().type() == proto::NumericalSplit::EXACT) {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelClassificationFeatureNumericalCart(
                        selected_examples, weights, attribute_data->values(),
                        label_stats.label_data, label_stats.num_label_classes,
                        na_replacement, min_num_obs, dt_config,
                        label_stats.label_distribution, attribute_idx,
                        internal_config, best_condition, cache));
      } else {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelClassificationFeatureNumericalHistogram(
                        selected_examples, weights, attribute_data->values(),
                        label_stats.label_data, label_stats.num_label_classes,
                        na_replacement, min_num_obs, dt_config,
                        label_stats.label_distribution, attribute_idx, random,
                        best_condition));
      }
    } break;

    case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::DiscretizedNumericalColumn>(
              attribute_idx));

      const auto na_replacement = attribute_column_spec.numerical().mean();
      const auto num_bins =
          attribute_column_spec.discretized_numerical().boundaries_size() + 1;
      const auto na_replacement_index =
          dataset::NumericalToDiscretizedNumerical(attribute_column_spec,
                                                   na_replacement);
      ASSIGN_OR_RETURN(
          result, FindSplitLabelClassificationFeatureDiscretizedNumericalCart(
                      selected_examples, weights, attribute_data->values(),
                      num_bins, label_stats.label_data,
                      label_stats.num_label_classes, na_replacement_index,
                      min_num_obs, dt_config, label_stats.label_distribution,
                      attribute_idx, best_condition, cache));
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::CategoricalColumn>(attribute_idx));

      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      ASSIGN_OR_RETURN(
          result, FindSplitLabelClassificationFeatureCategorical(
                      selected_examples, weights, attribute_data->values(),
                      label_stats.label_data, num_attribute_classes,
                      label_stats.num_label_classes, na_replacement,
                      min_num_obs, dt_config, label_stats.label_distribution,
                      attribute_idx, random, best_condition, cache));
    } break;

    case dataset::proto::ColumnType::CATEGORICAL_SET: {
      ASSIGN_OR_RETURN(
          const auto* attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::CategoricalSetColumn>(attribute_idx));
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelClassificationFeatureCategoricalSetGreedyForward(
              selected_examples, weights, *attribute_data,
              label_stats.label_data, num_attribute_classes,
              label_stats.num_label_classes, min_num_obs, dt_config,
              label_stats.label_distribution, attribute_idx, best_condition,
              random));
    } break;

    case dataset::proto::ColumnType::BOOLEAN: {
      // Condition of the type "Attr is True".
      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::BooleanColumn>(attribute_idx));

      const auto na_replacement =
          attribute_column_spec.boolean().count_true() >=
          attribute_column_spec.boolean().count_false();
      ASSIGN_OR_RETURN(
          result, FindSplitLabelClassificationFeatureBoolean(
                      selected_examples, weights, attribute_data->values(),
                      label_stats.label_data, label_stats.num_label_classes,
                      na_replacement, min_num_obs, dt_config,
                      label_stats.label_distribution, attribute_idx,
                      best_condition, cache));
    } break;

    case dataset::proto::ColumnType::NUMERICAL_VECTOR_SEQUENCE: {
      ASSIGN_OR_RETURN(
          const auto* attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::NumericalVectorSequenceColumn>(
              attribute_idx));
      ASSIGN_OR_RETURN(
          result, FindSplitAnyLabelFeatureNumericalVectorSequence(
                      model::proto::Task::CLASSIFICATION, selected_examples,
                      weights, *attribute_data, attribute_column_spec,
                      label_stats, min_num_obs, dt_config, attribute_idx,
                      internal_config, best_condition, random, cache));
    } break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          dataset::proto::ColumnType_Name(
              train_dataset.column(attribute_idx)->type()),
          " attribute ", train_dataset.column(attribute_idx)->name(),
          " is not supported."));
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    ASSIGN_OR_RETURN(
        const auto na_result,
        FindSplitLabelClassificationFeatureNA(
            selected_examples, weights, train_dataset.column(attribute_idx),
            label_stats.label_data, label_stats.num_label_classes, min_num_obs,
            dt_config, label_stats.label_distribution, attribute_idx,
            best_condition, cache));
    result = std::min(result, na_result);
  }

  return result;
}

absl::StatusOr<SplitSearchResult> FindBestConditionRegressionHessianGain(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionHessianLabelStats& label_stats, const int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  if (dt_config.internal().generate_fake_error_in_splitter()) {
    return absl::InternalError("Fake error");
  }

  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  SplitSearchResult result;

  const int8_t monotonic_direction =
      MonotonicConstraintSign(config_link, attribute_idx);

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      // Condition of the type "Attr >= threshold".
      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::NumericalColumn>(attribute_idx));

      const auto na_replacement = attribute_column_spec.numerical().mean();
      if (dt_config.numerical_split().type() == proto::NumericalSplit::EXACT) {
        if (weights.empty()) {
          ASSIGN_OR_RETURN(
              result,
              FindSplitLabelHessianRegressionFeatureNumericalCart<
                  /*weighted=*/false>(
                  selected_examples, weights, attribute_data->values(),
                  label_stats.gradient_data, label_stats.hessian_data,
                  na_replacement, min_num_obs, dt_config,
                  label_stats.sum_gradient, label_stats.sum_hessian,
                  label_stats.sum_weights, attribute_idx, internal_config,
                  constraints, monotonic_direction, best_condition, cache));
        } else {
          ASSIGN_OR_RETURN(
              result,
              FindSplitLabelHessianRegressionFeatureNumericalCart<
                  /*weighted=*/true>(
                  selected_examples, weights, attribute_data->values(),
                  label_stats.gradient_data, label_stats.hessian_data,
                  na_replacement, min_num_obs, dt_config,
                  label_stats.sum_gradient, label_stats.sum_hessian,
                  label_stats.sum_weights, attribute_idx, internal_config,
                  constraints, monotonic_direction, best_condition, cache));
        }
      } else {
        return absl::InvalidArgumentError(
            "Only split exact implemented for hessian gains.");
      }
    } break;

    case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      // Condition of the type "Attr >= threshold".
      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::DiscretizedNumericalColumn>(
              attribute_idx));

      const auto na_replacement = attribute_column_spec.numerical().mean();
      const auto num_bins =
          attribute_column_spec.discretized_numerical().boundaries_size() + 1;
      const auto na_replacement_index =
          dataset::NumericalToDiscretizedNumerical(attribute_column_spec,
                                                   na_replacement);
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelHessianRegressionFeatureDiscretizedNumericalCart<
                /*weighted=*/false>(
                selected_examples, weights, attribute_data->values(), num_bins,
                label_stats.gradient_data, label_stats.hessian_data,
                na_replacement_index, min_num_obs, dt_config,
                label_stats.sum_gradient, label_stats.sum_hessian,
                label_stats.sum_weights, attribute_idx, internal_config,
                constraints, monotonic_direction, best_condition, cache));
      } else {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelHessianRegressionFeatureDiscretizedNumericalCart<
                /*weighted=*/true>(
                selected_examples, weights, attribute_data->values(), num_bins,
                label_stats.gradient_data, label_stats.hessian_data,
                na_replacement_index, min_num_obs, dt_config,
                label_stats.sum_gradient, label_stats.sum_hessian,
                label_stats.sum_weights, attribute_idx, internal_config,
                constraints, monotonic_direction, best_condition, cache));
      }
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      // Condition of the type "Attr \in X".
      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::CategoricalColumn>(attribute_idx));

      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelHessianRegressionFeatureCategorical<
                /*weighted=*/false>(
                selected_examples, weights, attribute_data->values(),
                label_stats.gradient_data, label_stats.hessian_data,
                num_attribute_classes, na_replacement, min_num_obs, dt_config,
                label_stats.sum_gradient, label_stats.sum_hessian,
                label_stats.sum_weights, attribute_idx, internal_config,
                constraints, best_condition, cache, random));
      } else {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelHessianRegressionFeatureCategorical<
                /*weighted=*/true>(
                selected_examples, weights, attribute_data->values(),
                label_stats.gradient_data, label_stats.hessian_data,
                num_attribute_classes, na_replacement, min_num_obs, dt_config,
                label_stats.sum_gradient, label_stats.sum_hessian,
                label_stats.sum_weights, attribute_idx, internal_config,
                constraints, best_condition, cache, random));
      }
    } break;

    case dataset::proto::ColumnType::BOOLEAN: {
      // Condition of the type "Attr is True".
      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::BooleanColumn>(attribute_idx));

      const auto na_replacement =
          attribute_column_spec.boolean().count_true() >=
          attribute_column_spec.boolean().count_false();
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelHessianRegressionFeatureBoolean</*weighted=*/false>(
                selected_examples, weights, attribute_data->values(),
                label_stats.gradient_data, label_stats.hessian_data,
                na_replacement, min_num_obs, dt_config,
                label_stats.sum_gradient, label_stats.sum_hessian,
                label_stats.sum_weights, attribute_idx, internal_config,
                constraints, best_condition, cache));
      } else {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelHessianRegressionFeatureBoolean</*weighted=*/true>(
                selected_examples, weights, attribute_data->values(),
                label_stats.gradient_data, label_stats.hessian_data,
                na_replacement, min_num_obs, dt_config,
                label_stats.sum_gradient, label_stats.sum_hessian,
                label_stats.sum_weights, attribute_idx, internal_config,
                constraints, best_condition, cache));
      }
    } break;

    case dataset::proto::ColumnType::NUMERICAL_VECTOR_SEQUENCE: {
      ASSIGN_OR_RETURN(
          const auto* attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::NumericalVectorSequenceColumn>(
              attribute_idx));
      ASSIGN_OR_RETURN(result,
                       FindSplitAnyLabelFeatureNumericalVectorSequence(
                           model::proto::Task::REGRESSION, selected_examples,
                           weights, *attribute_data, attribute_column_spec,
                           label_stats, min_num_obs, dt_config, attribute_idx,
                           internal_config, best_condition, random, cache));
    } break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          dataset::proto::ColumnType_Name(
              train_dataset.column(attribute_idx)->type()),
          " attribute ", train_dataset.column(attribute_idx)->name(),
          " is not supported."));
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    if (weights.empty()) {
      ASSIGN_OR_RETURN(
          const auto na_result,
          FindSplitLabelHessianRegressionFeatureNA</*weighted=*/false>(
              selected_examples, weights, train_dataset.column(attribute_idx),
              label_stats.gradient_data, label_stats.hessian_data, min_num_obs,
              dt_config, label_stats.sum_gradient, label_stats.sum_hessian,
              label_stats.sum_weights, attribute_idx, internal_config,
              constraints, best_condition, cache));
      result = std::min(result, na_result);
    } else {
      ASSIGN_OR_RETURN(
          const auto na_result,
          FindSplitLabelHessianRegressionFeatureNA</*weighted=*/true>(
              selected_examples, weights, train_dataset.column(attribute_idx),
              label_stats.gradient_data, label_stats.hessian_data, min_num_obs,
              dt_config, label_stats.sum_gradient, label_stats.sum_hessian,
              label_stats.sum_weights, attribute_idx, internal_config,
              constraints, best_condition, cache));
      result = std::min(result, na_result);
    }
  }

  return result;
}

// Specialization in the case of regression.
absl::StatusOr<SplitSearchResult> FindBestConditionRegression(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionLabelStats& label_stats, const int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  if (dt_config.internal().generate_fake_error_in_splitter()) {
    return absl::InternalError("Fake error");
  }

  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  SplitSearchResult result;

  RETURN_IF_ERROR(
      FailIfMonotonic(config_link, attribute_idx, constraints, "regression"));

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      // Condition of the type "Attr >= threshold".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::NumericalColumn>(attribute_idx)
              .value()
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();
      if (dt_config.numerical_split().type() == proto::NumericalSplit::EXACT) {
        if (weights.empty()) {
          ASSIGN_OR_RETURN(
              result,
              FindSplitLabelRegressionFeatureNumericalCart</*weighted=*/false>(
                  selected_examples, weights, attribute_data,
                  label_stats.label_data, na_replacement, min_num_obs,
                  dt_config, label_stats.label_distribution, attribute_idx,
                  internal_config, best_condition, cache));
        } else {
          ASSIGN_OR_RETURN(
              result,
              FindSplitLabelRegressionFeatureNumericalCart</*weighted=*/true>(
                  selected_examples, weights, attribute_data,
                  label_stats.label_data, na_replacement, min_num_obs,
                  dt_config, label_stats.label_distribution, attribute_idx,
                  internal_config, best_condition, cache));
        }
      } else {
        if (weights.empty()) {
          ASSIGN_OR_RETURN(
              result, FindSplitLabelRegressionFeatureNumericalHistogram<
                          /*weighted=*/false>(
                          selected_examples, weights, attribute_data,
                          label_stats.label_data, na_replacement, min_num_obs,
                          dt_config, label_stats.label_distribution,
                          attribute_idx, random, best_condition));
        } else {
          ASSIGN_OR_RETURN(
              result, FindSplitLabelRegressionFeatureNumericalHistogram<
                          /*weighted=*/true>(
                          selected_examples, weights, attribute_data,
                          label_stats.label_data, na_replacement, min_num_obs,
                          dt_config, label_stats.label_distribution,
                          attribute_idx, random, best_condition));
        }
      }
    } break;

    case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      // Condition of the type "Attr >= threshold".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::DiscretizedNumericalColumn>(
                  attribute_idx)
              .value()
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();
      const auto num_bins =
          attribute_column_spec.discretized_numerical().boundaries_size() + 1;
      const auto na_replacement_index =
          dataset::NumericalToDiscretizedNumerical(attribute_column_spec,
                                                   na_replacement);
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelRegressionFeatureDiscretizedNumericalCart<
                        /*weighted=*/false>(
                        selected_examples, weights, attribute_data, num_bins,
                        label_stats.label_data, na_replacement_index,
                        min_num_obs, dt_config, label_stats.label_distribution,
                        attribute_idx, best_condition, cache));
      } else {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelRegressionFeatureDiscretizedNumericalCart<
                        /*weighted=*/true>(
                        selected_examples, weights, attribute_data, num_bins,
                        label_stats.label_data, na_replacement_index,
                        min_num_obs, dt_config, label_stats.label_distribution,
                        attribute_idx, best_condition, cache));
      }
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      // Condition of the type "Attr \in X".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::CategoricalColumn>(attribute_idx)
              .value()
              ->values();
      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelRegressionFeatureCategorical</*weighted=*/false>(
                selected_examples, weights, attribute_data,
                label_stats.label_data, num_attribute_classes, na_replacement,
                min_num_obs, dt_config, label_stats.label_distribution,
                attribute_idx, best_condition, cache, random));
      } else {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelRegressionFeatureCategorical</*weighted=*/true>(
                selected_examples, weights, attribute_data,
                label_stats.label_data, num_attribute_classes, na_replacement,
                min_num_obs, dt_config, label_stats.label_distribution,
                attribute_idx, best_condition, cache, random));
      }
    } break;

    case dataset::proto::ColumnType::CATEGORICAL_SET: {
      const auto* attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::CategoricalSetColumn>(attribute_idx)
              .value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelRegressionFeatureCategoricalSetGreedyForward<
                        /*weighted=*/false>(
                        selected_examples, weights, *attribute_data,
                        label_stats.label_data, num_attribute_classes,
                        min_num_obs, dt_config, label_stats.label_distribution,
                        attribute_idx, best_condition, random));
      } else {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelRegressionFeatureCategoricalSetGreedyForward<
                        /*weighted=*/true>(
                        selected_examples, weights, *attribute_data,
                        label_stats.label_data, num_attribute_classes,
                        min_num_obs, dt_config, label_stats.label_distribution,
                        attribute_idx, best_condition, random));
      }
    } break;

    case dataset::proto::ColumnType::BOOLEAN: {
      // Condition of the type "Attr is True".
      ASSIGN_OR_RETURN(
          const auto* attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::BooleanColumn>(attribute_idx));
      const auto na_replacement =
          attribute_column_spec.boolean().count_true() >=
          attribute_column_spec.boolean().count_false();
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelRegressionFeatureBoolean</*weighted=*/false>(
                        selected_examples, weights, attribute_data->values(),
                        label_stats.label_data, na_replacement, min_num_obs,
                        dt_config, label_stats.label_distribution,
                        attribute_idx, best_condition, cache));
      } else {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelRegressionFeatureBoolean</*weighted=*/true>(
                        selected_examples, weights, attribute_data->values(),
                        label_stats.label_data, na_replacement, min_num_obs,
                        dt_config, label_stats.label_distribution,
                        attribute_idx, best_condition, cache));
      }
    } break;

    case dataset::proto::ColumnType::NUMERICAL_VECTOR_SEQUENCE: {
      ASSIGN_OR_RETURN(
          const auto* attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::NumericalVectorSequenceColumn>(
              attribute_idx));
      ASSIGN_OR_RETURN(result,
                       FindSplitAnyLabelFeatureNumericalVectorSequence(
                           model::proto::Task::REGRESSION, selected_examples,
                           weights, *attribute_data, attribute_column_spec,
                           label_stats, min_num_obs, dt_config, attribute_idx,
                           internal_config, best_condition, random, cache));
    } break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          dataset::proto::ColumnType_Name(
              train_dataset.column(attribute_idx)->type()),
          " attribute ", train_dataset.column(attribute_idx)->name(),
          " is not supported."));
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    if (weights.empty()) {
      ASSIGN_OR_RETURN(
          const auto na_result,
          FindSplitLabelRegressionFeatureNA</*weighted=*/false>(
              selected_examples, weights, train_dataset.column(attribute_idx),
              label_stats.label_data, min_num_obs, dt_config,
              label_stats.label_distribution, attribute_idx, best_condition,
              cache));
      result = std::min(result, na_result);
    } else {
      ASSIGN_OR_RETURN(
          const auto na_result,
          FindSplitLabelRegressionFeatureNA</*weighted=*/true>(
              selected_examples, weights, train_dataset.column(attribute_idx),
              label_stats.label_data, min_num_obs, dt_config,
              label_stats.label_distribution, attribute_idx, best_condition,
              cache));
      result = std::min(result, na_result);
    }
  }

  return result;
}

// Specialization in the case of uplift with categorical outcome.
absl::StatusOr<SplitSearchResult> FindBestConditionUpliftCategorical(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const CategoricalUpliftLabelStats& label_stats, const int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;
  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  RETURN_IF_ERROR(FailIfMonotonic(config_link, attribute_idx, constraints,
                                  "categorical uplift"));

  SplitSearchResult result;

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::NumericalColumn>(attribute_idx)
              .value()
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();

      ASSIGN_OR_RETURN(
          result, FindSplitLabelUpliftCategoricalFeatureNumericalCart(
                      selected_examples, weights, attribute_data, label_stats,
                      na_replacement, min_num_obs, dt_config, attribute_idx,
                      internal_config, best_condition, cache));
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::CategoricalColumn>(attribute_idx)
              .value()
              ->values();
      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();

      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelUpliftCategoricalFeatureCategorical(
              selected_examples, weights, attribute_data, label_stats,
              num_attribute_classes, na_replacement, min_num_obs, dt_config,
              attribute_idx, internal_config, best_condition, cache, random));
    } break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          dataset::proto::ColumnType_Name(
              train_dataset.column(attribute_idx)->type()),
          " attribute ", train_dataset.column(attribute_idx)->name(),
          " is not supported."));
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    return absl::InvalidArgumentError("allow_na_conditions not supported");
  }

  return result;
}

// Specialization in the case of uplift with numerical outcome.
absl::StatusOr<SplitSearchResult> FindBestConditionUpliftNumerical(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const NumericalUpliftLabelStats& label_stats, const int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;
  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  RETURN_IF_ERROR(FailIfMonotonic(config_link, attribute_idx, constraints,
                                  "numerical uplift"));

  SplitSearchResult result;

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();

      ASSIGN_OR_RETURN(
          result, FindSplitLabelUpliftNumericalFeatureNumericalCart(
                      selected_examples, weights, attribute_data, label_stats,
                      na_replacement, min_num_obs, dt_config, attribute_idx,
                      internal_config, best_condition, cache));
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();

      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelUpliftNumericalFeatureCategorical(
              selected_examples, weights, attribute_data, label_stats,
              num_attribute_classes, na_replacement, min_num_obs, dt_config,
              attribute_idx, internal_config, best_condition, cache, random));
    } break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          dataset::proto::ColumnType_Name(
              train_dataset.column(attribute_idx)->type()),
          " attribute ", train_dataset.column(attribute_idx)->name(),
          " is not supported."));
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    return absl::InvalidArgumentError("allow_na_conditions not supported");
  }

  return result;
}

absl::StatusOr<SplitterWorkResponse> FindBestConditionFromSplitterWorkRequest(
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const InternalTrainConfig& internal_config,
    const SplitterWorkRequest& request) {
  SplitterWorkResponse response;
  response.manager_data = request.manager_data;
  request.splitter_cache->random.seed(request.seed);

  response.condition = absl::make_unique<proto::NodeCondition>();
  response.condition->set_split_score(request.best_score);

  if (request.num_oblique_projections_to_run != -1) {
    DCHECK_EQ(request.attribute_idx, -1);
    ASSIGN_OR_RETURN(
        const auto found_oblique_condition,
        FindBestConditionOblique(
            request.common->train_dataset, request.common->selected_examples,
            weights, config, config_link, dt_config, request.common->parent,
            internal_config, request.common->label_stats,
            request.num_oblique_projections_to_run, request.common->constraints,
            response.condition.get(), &request.splitter_cache->random,
            request.splitter_cache));

    // An oblique split cannot be invalid.
    response.status = found_oblique_condition
                          ? SplitSearchResult::kBetterSplitFound
                          : SplitSearchResult::kNoBetterSplitFound;
    return response;
  }

  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION: {
      const auto& label_stats =
          utils::down_cast<const ClassificationLabelStats&>(
              request.common->label_stats);

      ASSIGN_OR_RETURN(
          response.status,
          FindBestConditionClassification(
              request.common->train_dataset, request.common->selected_examples,
              weights, config, config_link, dt_config, request.common->parent,
              internal_config, label_stats, request.attribute_idx,
              request.common->constraints, response.condition.get(),
              &request.splitter_cache->random, request.splitter_cache));
    } break;
    case model::proto::Task::REGRESSION:
      if (internal_config.hessian_score) {
        const auto& label_stats =
            utils::down_cast<const RegressionHessianLabelStats&>(
                request.common->label_stats);

        ASSIGN_OR_RETURN(
            response.status,
            FindBestConditionRegressionHessianGain(
                request.common->train_dataset,
                request.common->selected_examples, weights, config, config_link,
                dt_config, request.common->parent, internal_config, label_stats,
                request.attribute_idx, request.common->constraints,
                response.condition.get(), &request.splitter_cache->random,
                request.splitter_cache));

      } else {
        const auto& label_stats = utils::down_cast<const RegressionLabelStats&>(
            request.common->label_stats);

        ASSIGN_OR_RETURN(
            response.status,
            FindBestConditionRegression(
                request.common->train_dataset,
                request.common->selected_examples, weights, config, config_link,
                dt_config, request.common->parent, internal_config, label_stats,
                request.attribute_idx, request.common->constraints,
                response.condition.get(), &request.splitter_cache->random,
                request.splitter_cache));
      }
      break;
    default:
      NOTREACHED();
  }

  return response;
}

absl::StatusOr<bool> FindBestConditionOblique(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION: {
      const auto& class_label_stats =
          utils::down_cast<const ClassificationLabelStats&>(label_stats);
      return FindBestConditionOblique(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, parent, internal_config, class_label_stats,
          override_num_projections, best_condition, random, cache);
    } break;
    case model::proto::Task::REGRESSION:
      if (internal_config.hessian_score) {
        const auto& reg_label_stats =
            utils::down_cast<const RegressionHessianLabelStats&>(label_stats);
        return FindBestConditionOblique(
            train_dataset, selected_examples, weights, config, config_link,
            dt_config, parent, internal_config, reg_label_stats,
            override_num_projections, constraints, best_condition, random,
            cache);
      } else {
        const auto& reg_label_stats =
            utils::down_cast<const RegressionLabelStats&>(label_stats);
        return FindBestConditionOblique(
            train_dataset, selected_examples, weights, config, config_link,
            dt_config, parent, internal_config, reg_label_stats,
            override_num_projections, best_condition, random, cache);
      }
      break;
    default:
      return absl::UnimplementedError(
          "Oblique splits not implemented for this task");
  }

  return false;
}

absl::StatusOr<bool> FindBestConditionSingleThreadManager(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, const NodeConstraints& constraints,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    PerThreadCache* cache) {
  // Single Thread Setup.
  cache->splitter_cache_list.resize(1);

  // Was a least one good split found?
  bool found_good_condition = false;

  switch (dt_config.split_axis_case()) {
    case proto::DecisionTreeTrainingConfig::SPLIT_AXIS_NOT_SET:
    case proto::DecisionTreeTrainingConfig::kAxisAlignedSplit:
      // Nothing to do.
      break;
    case proto::DecisionTreeTrainingConfig::kSparseObliqueSplit:
    case proto::DecisionTreeTrainingConfig::kMhldObliqueSplit:
      ASSIGN_OR_RETURN(
          found_good_condition,
          FindBestConditionOblique(
              train_dataset, selected_examples, weights, config, config_link,
              dt_config, parent, internal_config, label_stats, {}, constraints,
              best_condition, random, &cache->splitter_cache_list[0]));
      break;
  }

  // Get the indices of the attributes to test.
  int remaining_attributes_to_test;
  std::vector<int32_t>& candidate_attributes = cache->candidate_attributes;
  GetCandidateAttributes(config, config_link, dt_config,
                         &remaining_attributes_to_test, &candidate_attributes,
                         random);

  // Index of the next attribute to be tested in "candidate_attributes".
  int candidate_attribute_idx_in_candidate_list = 0;

  while (remaining_attributes_to_test >= 0 &&
         candidate_attribute_idx_in_candidate_list <
             candidate_attributes.size()) {
    // Get the attribute data.
    const int32_t attribute_idx =
        candidate_attributes[candidate_attribute_idx_in_candidate_list++];
    SplitSearchResult result;

    switch (config.task()) {
      case model::proto::Task::CLASSIFICATION: {
        const auto& class_label_stats =
            utils::down_cast<const ClassificationLabelStats&>(label_stats);

        ASSIGN_OR_RETURN(result, FindBestConditionClassification(
                                     train_dataset, selected_examples, weights,
                                     config, config_link, dt_config, parent,
                                     internal_config, class_label_stats,
                                     attribute_idx, constraints, best_condition,
                                     random, &cache->splitter_cache_list[0]));
      } break;
      case model::proto::Task::REGRESSION:
        if (internal_config.hessian_score) {
          const auto& reg_label_stats =
              utils::down_cast<const RegressionHessianLabelStats&>(label_stats);

          ASSIGN_OR_RETURN(
              result,
              FindBestConditionRegressionHessianGain(
                  train_dataset, selected_examples, weights, config,
                  config_link, dt_config, parent, internal_config,
                  reg_label_stats, attribute_idx, constraints, best_condition,
                  random, &cache->splitter_cache_list[0]));

        } else {
          const auto& reg_label_stats =
              utils::down_cast<const RegressionLabelStats&>(label_stats);

          ASSIGN_OR_RETURN(
              result,
              FindBestConditionRegression(
                  train_dataset, selected_examples, weights, config,
                  config_link, dt_config, parent, internal_config,
                  reg_label_stats, attribute_idx, constraints, best_condition,
                  random, &cache->splitter_cache_list[0]));
        }
        break;

      case model::proto::Task::CATEGORICAL_UPLIFT: {
        const auto& uplift_label_stats =
            utils::down_cast<const CategoricalUpliftLabelStats&>(label_stats);
        ASSIGN_OR_RETURN(result, FindBestConditionUpliftCategorical(
                                     train_dataset, selected_examples, weights,
                                     config, config_link, dt_config, parent,
                                     internal_config, uplift_label_stats,
                                     attribute_idx, constraints, best_condition,
                                     random, &cache->splitter_cache_list[0]));
      } break;

      case model::proto::Task::NUMERICAL_UPLIFT: {
        const auto& uplift_label_stats =
            utils::down_cast<const NumericalUpliftLabelStats&>(label_stats);
        ASSIGN_OR_RETURN(result, FindBestConditionUpliftNumerical(
                                     train_dataset, selected_examples, weights,
                                     config, config_link, dt_config, parent,
                                     internal_config, uplift_label_stats,
                                     attribute_idx, constraints, best_condition,
                                     random, &cache->splitter_cache_list[0]));
      } break;

      default:
        return absl::UnimplementedError("Non implemented");
    }
    if (result != SplitSearchResult::kInvalidAttribute) {
      remaining_attributes_to_test--;
      if (result == SplitSearchResult::kBetterSplitFound) {
        found_good_condition = true;
      }
    }
  }

  return found_good_condition;
}

absl::StatusOr<bool> FindBestConditionConcurrentManager(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, const NodeConstraints& constraints,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    PerThreadCache* cache) {
  // This method looks for the best split using worker threads.
  //
  // Background:
  // The best split is the split with the best score among the first
  // "min_num_jobs_to_test" valid split searches. The order of the split is
  // defined by the list of candidate attributes generated by
  // GetCandidateAttributes. A split search is valid if it tested at least one
  // split.
  //
  // Since the execution is multi-threaded, splits are computed/evaluated in
  // unpredictable order. However, this method's result is as if the splits were
  // evaluated sequentially according to the order defined by
  // GetCandidateAttributes. Oblique splits are always evaluated (if requested
  // by the user).
  //
  // A work unit (called a "job") is the evaluation of a single attribute or, in
  // the case of oblique splits, the evaluation of a given number of random
  // projections. A job with a given idx can be in one of multiple states (in
  // chronological order):
  //
  // 1. Before being scheduled (idx >= next_job_to_schedule).
  // 2. Scheduled and being computed by a worker.
  // 3. The worker is done with the computation, and the result was recorded by
  // the manager (cache->durable_response_list[idx].set).
  // 4. The result was processed by the manager (idx < next_job_to_process).
  //
  // This method guarantees that the jobs/splits are processed in order.
  //
  // Note that next_job_to_process < next_job_to_schedule always holds.
  if (internal_config.split_finder_processor == nullptr) {
    return absl::InternalError(
        absl::Substitute("Multi-threaded execution requested but no worker "
                         "threads created. Expected $0 threads",
                         internal_config.num_threads));
  }
  const int num_threads = internal_config.num_threads;

  if (config_link.features().empty()) {
    return false;
  }

  // Constant and static part of the requests.
  SplitterWorkRequestCommon common{
      .train_dataset = train_dataset,
      .selected_examples = selected_examples,
      .parent = parent,
      .label_stats = label_stats,
      .constraints = constraints,
      .weights = weights,
      .config = config,
      .config_link = config_link,
      .dt_config = dt_config,
      .internal_config = internal_config,
  };

  // Computes the number of oblique projections to evaluate and how to group
  // them into requests.
  int num_oblique_jobs = 0;
  int num_oblique_projections;
  int num_oblique_projections_per_oblique_job;

  if (config_link.numerical_features_size() > 0) {
    if (dt_config.split_axis_case() ==
        proto::DecisionTreeTrainingConfig::kSparseObliqueSplit) {
      num_oblique_projections =
          GetNumProjections(dt_config, config_link.numerical_features_size());

      if (num_oblique_projections > 0) {
        // Arbitrary minimum number of oblique projections to test in each job.
        // Because oblique jobs are expensive (more than non oblique jobs), it
        // is not efficient to create a request with too little work to do.
        //
        // In most real cases, this parameter does not matter as the limit is
        // effectively constraint by the number of threads.
        const int min_projections_per_request = 10;

        DCHECK_GE(num_threads, 1);
        num_oblique_jobs = std::min(
            num_threads,
            (num_oblique_projections + min_projections_per_request - 1) /
                min_projections_per_request);
        num_oblique_projections_per_oblique_job =
            (num_oblique_projections + num_oblique_jobs - 1) / num_oblique_jobs;
      }
    } else if (dt_config.split_axis_case() ==
               proto::DecisionTreeTrainingConfig::kMhldObliqueSplit) {
      num_oblique_projections = 1;
      num_oblique_projections_per_oblique_job = 1;
      num_oblique_jobs = 1;
    }
  }

  // Prepare caches.
  cache->splitter_cache_list.resize(num_threads);

  // Get the ordered indices of the attributes to test.
  int min_num_jobs_to_test;
  std::vector<int32_t>& candidate_attributes = cache->candidate_attributes;
  GetCandidateAttributes(config, config_link, dt_config, &min_num_jobs_to_test,
                         &candidate_attributes, random);

  const int num_jobs = candidate_attributes.size() + num_oblique_jobs;
  // All the oblique jobs need to be done.
  // Note: When do look for oblique splits, we also run the classical numerical
  // splitter.
  min_num_jobs_to_test += num_oblique_jobs;

  cache->durable_response_list.resize(num_jobs);

  // Marks all the caches "available".
  cache->available_cache_idxs.resize(cache->splitter_cache_list.size());
  std::iota(cache->available_cache_idxs.begin(),
            cache->available_cache_idxs.end(), 0);

  // Marks all the duration responses as "non set".
  for (auto& s : cache->durable_response_list) {
    s.set = false;
  }

  // Score and value of the best found condition.
  std::atomic<float> best_split_score = best_condition->split_score();
  std::unique_ptr<proto::NodeCondition> best_condition_ptr;

  // Get Channel readers and writers.
  auto& processor = *(internal_config.split_finder_processor);

  // Number of jobs currently scheduled.
  int num_in_flight = 0;

  // Helper function to create a WorkRequest.
  //
  // If attribute_idx is != -1 create a request for an axis-aligned split.
  //
  // If attribute_idx is == -1 and num_oblique_projections_to_run != -1, create
  // a request for an oblique split.
  //
  auto build_request =
      [&](const int job_idx, const int attribute_idx,
          const int num_oblique_projections_to_run) -> SplitterWorkRequest {
    DCHECK_NE(attribute_idx != -1, num_oblique_projections_to_run != -1);
    DCHECK(!cache->available_cache_idxs.empty());
    const int32_t cache_idx = cache->available_cache_idxs.back();
    cache->available_cache_idxs.pop_back();
    num_in_flight++;
    return SplitterWorkRequest(
        /*manager_data=*/
        {
            .cache_idx = cache_idx,
            .job_idx = job_idx,
        },
        /*best_score=*/best_split_score,
        /*attribute_idx=*/attribute_idx,
        /*splitter_cache=*/&cache->splitter_cache_list[cache_idx],
        /*common=*/&common,
        /*seed=*/(*random)(),
        /*num_oblique_projections_to_run=*/num_oblique_projections_to_run);
  };

  // Schedule all the oblique jobs.
  int next_job_to_schedule = 0;
  for (int oblique_job_idx = 0; oblique_job_idx < num_oblique_jobs;
       oblique_job_idx++) {
    int num_projections_in_request;
    if (oblique_job_idx == num_oblique_jobs - 1) {
      num_projections_in_request =
          num_oblique_projections -
          oblique_job_idx * num_oblique_projections_per_oblique_job;
    } else {
      num_projections_in_request = num_oblique_projections_per_oblique_job;
    }

    processor.Submit(build_request(
        next_job_to_schedule++,
        /*attribute_idx=*/-1,
        /*num_oblique_projections_to_run=*/num_projections_in_request));
  }

  // Schedule some non-oblique jobs if threads are still available.
  while (next_job_to_schedule < std::min(num_threads, num_jobs) &&
         !cache->available_cache_idxs.empty()) {
    DCHECK_GE(next_job_to_schedule, num_oblique_jobs);
    const int attribute_idx =
        candidate_attributes[next_job_to_schedule - num_oblique_jobs];

    processor.Submit(build_request(next_job_to_schedule,
                                   /*attribute_idx=*/attribute_idx,
                                   /*num_oblique_projections_to_run=*/-1));
    next_job_to_schedule++;
  }

  int num_valid_job_tested = 0;
  int next_job_to_process = 0;

  absl::Status status;

  while (true) {
    // Get a new result from a worker splitter.
    auto maybe_response = processor.GetResult();
    if (!maybe_response.has_value()) {
      break;
    }

    num_in_flight--;
    DCHECK_GE(num_in_flight, 0);

    {
      // Record, but do not process, the worker response.
      auto response_or = std::move(maybe_response).value();
      if (!response_or.ok()) {
        status.Update(response_or.status());
        break;
      }
      auto response = std::move(response_or).value();
      // Release the cache immediately to be reused by other workers.
      cache->available_cache_idxs.push_back(response.manager_data.cache_idx);

      // Record response for further processing.
      auto& durable_response =
          cache->durable_response_list[response.manager_data.job_idx];
      durable_response.status = response.status;
      durable_response.set = true;
      if (response.status == SplitSearchResult::kBetterSplitFound) {
        // The worker found a potentially better solution.
        durable_response.condition = std::move(response.condition);
      }
    }

    // Process new responses that can be processed.
    while (next_job_to_process < next_job_to_schedule &&
           num_valid_job_tested < min_num_jobs_to_test &&
           cache->durable_response_list[next_job_to_process].set) {
      // Something to process.
      auto durable_response =
          &cache->durable_response_list[next_job_to_process];
      next_job_to_process++;

      if (durable_response->status != SplitSearchResult::kInvalidAttribute) {
        num_valid_job_tested++;
      }
      if (durable_response->status == SplitSearchResult::kBetterSplitFound) {
        const float split_score = durable_response->condition->split_score();
        if (split_score > best_split_score) {
          best_condition_ptr = std::move(durable_response->condition);
          best_split_score = split_score;
        }
      }
    }

    if (num_valid_job_tested >= min_num_jobs_to_test) {
      // Enough jobs have been tested to take a decision.
      break;
    }

    if (next_job_to_process >= num_jobs) {
      // We have processed all the jobs.
      break;
    }

    // Schedule the testing of more conditions.
    while (!cache->available_cache_idxs.empty() &&
           next_job_to_schedule < num_jobs) {
      processor.Submit(build_request(
          next_job_to_schedule,
          /*attribute_idx=*/
          candidate_attributes[next_job_to_schedule - num_oblique_jobs],
          /*num_oblique_projections_to_run=*/-1));
      next_job_to_schedule++;
    }
  }

  // Drain the response channel.
  for (int i = 0; i < num_in_flight; i++) {
    auto maybe_response = processor.GetResult();
    if (!maybe_response.has_value()) {
      // The channel was closed.
      break;
    }
    auto response_or = std::move(maybe_response).value();
    status.Update(response_or.status());
  }

  // Move the random generator state to make the behavior deterministic.
  random->discard(num_jobs - next_job_to_schedule);

  if (!status.ok()) {
    return status;
  }

  if (best_condition_ptr) {
    *best_condition = std::move(*best_condition_ptr);
    return true;
  }
  return false;
}

absl::StatusOr<bool> FindBestConditionManager(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, const NodeConstraints& constraints,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    PerThreadCache* cache) {
  if (internal_config.split_finder_processor != nullptr) {
    return FindBestConditionConcurrentManager(
        train_dataset, selected_examples, weights, config, config_link,
        dt_config, parent, internal_config, label_stats, constraints,
        best_condition, random, cache);
  }

  // Single thread.
  return FindBestConditionSingleThreadManager(
      train_dataset, selected_examples, weights, config, config_link, dt_config,
      parent, internal_config, label_stats, constraints, best_condition, random,
      cache);
}

absl::StatusOr<bool> FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, PerThreadCache* cache) {
  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION: {
      STATUS_CHECK(!internal_config.hessian_score);
      ASSIGN_OR_RETURN(const auto labels,
                       train_dataset.ColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(
                           config_link.label()));
      ClassificationLabelStats label_stat(labels->values());

      const auto& label_column_spec =
          train_dataset.data_spec().columns(config_link.label());
      label_stat.num_label_classes =
          label_column_spec.categorical().number_of_unique_values();

      label_stat.label_distribution.Load(parent.classifier().distribution());

      if (label_stat.label_distribution.NumClasses() >= 1 &&
          label_stat.label_distribution.count(
              dataset::kOutOfDictionaryItemIndex) > 0) {
        return absl::InternalError(
            absl::StrCat("The training label column \"", config.label(),
                         "\" contain out-of-dictionary (=0) values."));
      }

      return FindBestConditionManager(train_dataset, selected_examples, weights,
                                      config, config_link, dt_config, parent,
                                      internal_config, label_stat, constraints,
                                      best_condition, random, cache);
    } break;

    case model::proto::Task::REGRESSION: {
      if (internal_config.hessian_score) {
        STATUS_CHECK_NE(internal_config.gradient_col_idx, -1);
        STATUS_CHECK_NE(internal_config.hessian_col_idx, -1);
        STATUS_CHECK_EQ(internal_config.gradient_col_idx, config_link.label());

        ASSIGN_OR_RETURN(const auto gradients,
                         train_dataset.ColumnWithCastWithStatus<
                             dataset::VerticalDataset::NumericalColumn>(
                             internal_config.gradient_col_idx));
        ASSIGN_OR_RETURN(const auto hessians,
                         train_dataset.ColumnWithCastWithStatus<
                             dataset::VerticalDataset::NumericalColumn>(
                             internal_config.hessian_col_idx));

        RegressionHessianLabelStats label_stat(gradients->values(),
                                               hessians->values());

        STATUS_CHECK(parent.regressor().has_sum_gradients());
        label_stat.sum_gradient = parent.regressor().sum_gradients();
        label_stat.sum_hessian = parent.regressor().sum_hessians();
        label_stat.sum_weights = parent.regressor().sum_weights();

        return FindBestConditionManager(
            train_dataset, selected_examples, weights, config, config_link,
            dt_config, parent, internal_config, label_stat, constraints,
            best_condition, random, cache);
      } else {
        ASSIGN_OR_RETURN(const auto labels,
                         train_dataset.ColumnWithCastWithStatus<
                             dataset::VerticalDataset::NumericalColumn>(
                             config_link.label()));
        RegressionLabelStats label_stat(labels->values());

        STATUS_CHECK(parent.regressor().has_distribution());
        label_stat.label_distribution.Load(parent.regressor().distribution());

        return FindBestConditionManager(
            train_dataset, selected_examples, weights, config, config_link,
            dt_config, parent, internal_config, label_stat, constraints,
            best_condition, random, cache);
      }
    } break;

    case model::proto::Task::CATEGORICAL_UPLIFT: {
      STATUS_CHECK(!internal_config.hessian_score);
      const auto& outcome_spec =
          train_dataset.data_spec().columns(config_link.label());
      const auto& treatment_spec =
          train_dataset.data_spec().columns(config_link.uplift_treatment());

      ASSIGN_OR_RETURN(const auto labels,
                       train_dataset.ColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(
                           config_link.label()));

      ASSIGN_OR_RETURN(const auto treatments,
                       train_dataset.ColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(
                           config_link.uplift_treatment()));

      CategoricalUpliftLabelStats label_stat(
          labels->values(),
          outcome_spec.categorical().number_of_unique_values(),
          treatments->values(),
          treatment_spec.categorical().number_of_unique_values());

      UpliftLeafToLabelDist(parent.uplift(), &label_stat.label_distribution);

      return FindBestConditionManager(train_dataset, selected_examples, weights,
                                      config, config_link, dt_config, parent,
                                      internal_config, label_stat, constraints,
                                      best_condition, random, cache);
    } break;

    case model::proto::Task::NUMERICAL_UPLIFT: {
      STATUS_CHECK(!internal_config.hessian_score);
      const auto& treatment_spec =
          train_dataset.data_spec().columns(config_link.uplift_treatment());

      ASSIGN_OR_RETURN(
          const auto labels,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::NumericalColumn>(config_link.label()));

      ASSIGN_OR_RETURN(const auto treatments,
                       train_dataset.ColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(
                           config_link.uplift_treatment()));

      NumericalUpliftLabelStats label_stat(
          labels->values(), treatments->values(),
          treatment_spec.categorical().number_of_unique_values());

      UpliftLeafToLabelDist(parent.uplift(), &label_stat.label_distribution);

      return FindBestConditionManager(train_dataset, selected_examples, weights,
                                      config, config_link, dt_config, parent,
                                      internal_config, label_stat, constraints,
                                      best_condition, random, cache);
    } break;

    default:
      return absl::UnimplementedError("Non implemented");
  }
  return false;
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureNumericalHistogram(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<int32_t>& labels, const int32_t num_label_classes,
    float na_replacement, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition) {
  DCHECK(condition != nullptr);
  if (!weights.empty()) {
    DCHECK_EQ(weights.size(), labels.size());
  }

  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }
  // Determine the minimum and maximum values of the attribute.
  float min_value, max_value;
  if (!MinMaxNumericalAttribute(selected_examples, attributes, &min_value,
                                &max_value)) {
    return SplitSearchResult::kInvalidAttribute;
  }
  // There should be at least two different unique values.
  if (min_value == max_value) {
    return SplitSearchResult::kInvalidAttribute;
  }
  // Randomly select some threshold values.
  struct CandidateSplit {
    float threshold;
    utils::IntegerDistributionDouble pos_label_distribution;
    int64_t num_positive_examples_without_weights = 0;
    bool operator<(const CandidateSplit& other) const {
      return threshold < other.threshold;
    }
  };

  ASSIGN_OR_RETURN(
      const auto bins,
      internal::GenHistogramBins(dt_config.numerical_split().type(),
                                 dt_config.numerical_split().num_candidates(),
                                 attributes, min_value, max_value, random));

  std::vector<CandidateSplit> candidate_splits(bins.size());
  for (int split_idx = 0; split_idx < candidate_splits.size(); split_idx++) {
    auto& candidate_split = candidate_splits[split_idx];
    candidate_split.pos_label_distribution.SetNumClasses(num_label_classes);
    candidate_split.threshold = bins[split_idx];
  }

  // Compute the split score of each threshold.
  for (const auto example_idx : selected_examples) {
    const int32_t label = labels[example_idx];
    const float weight = weights.empty() ? 1.f : weights[example_idx];
    float attribute = attributes[example_idx];
    if (std::isnan(attribute)) {
      attribute = na_replacement;
    }
    auto it_split = std::upper_bound(
        candidate_splits.begin(), candidate_splits.end(), attribute,
        [](const float a, const CandidateSplit& b) { return a < b.threshold; });
    if (it_split == candidate_splits.begin()) {
      continue;
    }
    --it_split;
    it_split->num_positive_examples_without_weights++;
    it_split->pos_label_distribution.Add(label, weight);
  }

  for (int split_idx = candidate_splits.size() - 2; split_idx >= 0;
       split_idx--) {
    const auto& src = candidate_splits[split_idx + 1];
    auto& dst = candidate_splits[split_idx];
    dst.num_positive_examples_without_weights +=
        src.num_positive_examples_without_weights;
    dst.pos_label_distribution.Add(src.pos_label_distribution);
  }

  const double initial_entropy = label_distribution.Entropy();
  utils::BinaryToIntegerConfusionMatrixDouble confusion;
  confusion.SetNumClassesIntDim(num_label_classes);

  // Select the best threshold.
  bool found_split = false;
  for (auto& candidate_split : candidate_splits) {
    if (selected_examples.size() -
                candidate_split.num_positive_examples_without_weights <
            min_num_obs ||
        candidate_split.num_positive_examples_without_weights < min_num_obs) {
      continue;
    }

    confusion.mutable_neg()->Set(label_distribution);
    confusion.mutable_neg()->Sub(candidate_split.pos_label_distribution);
    confusion.mutable_pos()->Set(candidate_split.pos_label_distribution);

    const double final_entropy = confusion.FinalEntropy();
    const double information_gain = initial_entropy - final_entropy;
    if (information_gain > condition->split_score()) {
      condition->set_split_score(information_gain);
      condition->mutable_condition()->mutable_higher_condition()->set_threshold(
          candidate_split.threshold);
      condition->set_attribute(attribute_idx);
      condition->set_num_training_examples_without_weight(
          selected_examples.size());
      condition->set_num_training_examples_with_weight(
          confusion.NumObservations());
      condition->set_num_pos_training_examples_without_weight(
          candidate_split.num_positive_examples_without_weights);
      condition->set_num_pos_training_examples_with_weight(
          confusion.pos().NumObservations());
      condition->set_na_value(na_replacement >= candidate_split.threshold);
      found_split = true;
    }
  }
  return found_split ? SplitSearchResult::kBetterSplitFound
                     : SplitSearchResult::kNoBetterSplitFound;
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<int32_t>& labels, const int32_t num_label_classes,
    float na_replacement, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  if (!weights.empty()) {
    DCHECK_EQ(weights.size(), labels.size());
  }
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  const auto sorting_strategy =
      EffectiveStrategy(dt_config, selected_examples.size(), internal_config);

  // "Why ==3" ?
  // Categorical attributes always have one class reserved for
  // "out-of-vocabulary" items. The "num_label_classes" takes into account this
  // class. In case of binary classification, "num_label_classes" is 3 (OOB,
  // False, True).
  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      LabelBinaryCategoricalOneValueBucket</*weighted=*/false>::Filler
          label_filler(labels, weights);
      LabelBinaryCategoricalOneValueBucket</*weighted=*/false>::Initializer
          initializer(label_distribution);

      if (sorting_strategy ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
        const auto& sorted_attributes =
            internal_config.preprocessing
                ->presorted_numerical_features()[attribute_idx];
        return ScanSplitsPresortedSparse<
            FeatureNumericalLabelUnweightedBinaryCategoricalOneValue,
            LabelBinaryCategoricalScoreAccumulator>(
            internal_config.preprocessing->num_examples(), selected_examples,
            sorted_attributes.items, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx,
            internal_config.duplicated_selected_examples, condition,
            &cache->cache_v2);
      } else if (sorting_strategy ==
                 proto::DecisionTreeTrainingConfig::Internal::IN_NODE) {
        return FindBestSplit_LabelUnweightedBinaryClassificationFeatureNumerical(
            selected_examples, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx, condition, &cache->cache_v2);
      } else {
        return absl::InvalidArgumentError("Non supported strategy.");
      }
    } else {
      LabelBinaryCategoricalOneValueBucket</*weighted=*/true>::Filler
          label_filler(labels, weights);
      LabelBinaryCategoricalOneValueBucket</*weighted=*/true>::Initializer
          initializer(label_distribution);
      if (sorting_strategy ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
        const auto& sorted_attributes =
            internal_config.preprocessing
                ->presorted_numerical_features()[attribute_idx];
        return ScanSplitsPresortedSparse<
            FeatureNumericalLabelBinaryCategoricalOneValue,
            LabelBinaryCategoricalScoreAccumulator>(
            internal_config.preprocessing->num_examples(), selected_examples,
            sorted_attributes.items, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx,
            internal_config.duplicated_selected_examples, condition,
            &cache->cache_v2);
      } else if (sorting_strategy ==
                 proto::DecisionTreeTrainingConfig::Internal::IN_NODE) {
        return FindBestSplit_LabelBinaryClassificationFeatureNumerical(
            selected_examples, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx, condition, &cache->cache_v2);
      } else {
        return absl::InvalidArgumentError("Non supported strategy");
      }
    }
  } else {
    // Multi-class classification.
    if (weights.empty()) {
      LabelCategoricalOneValueBucket</*weighted=*/false>::Filler label_filler(
          labels, weights);
      LabelCategoricalOneValueBucket</*weighted=*/false>::Initializer
          initializer(label_distribution);

      if (sorting_strategy ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
        const auto& sorted_attributes =
            internal_config.preprocessing
                ->presorted_numerical_features()[attribute_idx];
        return ScanSplitsPresortedSparse<
            FeatureNumericalLabelUnweightedCategoricalOneValue,
            LabelCategoricalScoreAccumulator>(
            internal_config.preprocessing->num_examples(), selected_examples,
            sorted_attributes.items, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx,
            internal_config.duplicated_selected_examples, condition,
            &cache->cache_v2);
      } else if (sorting_strategy ==
                 proto::DecisionTreeTrainingConfig::Internal::IN_NODE) {
        return FindBestSplit_LabelUnweightedClassificationFeatureNumerical(
            selected_examples, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx, condition, &cache->cache_v2);
      } else {
        return absl::InvalidArgumentError("Non supported strategy");
      }
    } else {
      LabelCategoricalOneValueBucket</*weighted=*/true>::Filler label_filler(
          labels, weights);
      LabelCategoricalOneValueBucket</*weighted=*/true>::Initializer
          initializer(label_distribution);

      if (sorting_strategy ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
        const auto& sorted_attributes =
            internal_config.preprocessing
                ->presorted_numerical_features()[attribute_idx];
        return ScanSplitsPresortedSparse<
            FeatureNumericalLabelCategoricalOneValue,
            LabelCategoricalScoreAccumulator>(
            internal_config.preprocessing->num_examples(), selected_examples,
            sorted_attributes.items, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx,
            internal_config.duplicated_selected_examples, condition,
            &cache->cache_v2);
      } else if (sorting_strategy ==
                 proto::DecisionTreeTrainingConfig::Internal::IN_NODE) {
        return FindBestSplit_LabelClassificationFeatureNumerical(
            selected_examples, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx, condition, &cache->cache_v2);
      } else {
        return absl::InvalidArgumentError("Non supported strategy");
      }
    }
  }
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureDiscretizedNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    const int num_bins, const std::vector<int32_t>& labels,
    const int32_t num_label_classes,
    const dataset::DiscretizedNumericalIndex na_replacement,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if (!weights.empty()) {
    DCHECK_EQ(weights.size(), labels.size());
  }
  FeatureDiscretizedNumericalBucket::Filler feature_filler(
      num_bins, na_replacement, attributes);
  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      LabelBinaryCategoricalBucket</*weighted=*/false>::Filler label_filler(
          labels, weights, label_distribution);
      LabelBinaryCategoricalBucket</*weighted=*/false>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedBinaryClassificationFeatureDiscretizedNumerical(  // NOLINT(whitespace/line_length)
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelBinaryCategoricalBucket</*weighted=*/true>::Filler label_filler(
          labels, weights, label_distribution);
      LabelBinaryCategoricalBucket</*weighted=*/true>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelBinaryClassificationFeatureDiscretizedNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  } else {
    // Multi-class classification.
    if (weights.empty()) {
      LabelCategoricalBucket</*weighted=*/false>::Filler label_filler(
          labels, weights, label_distribution);
      LabelCategoricalBucket</*weighted=*/false>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedClassificationFeatureDiscretizedNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelCategoricalBucket</*weighted=*/true>::Filler label_filler(
          labels, weights, label_distribution);
      LabelCategoricalBucket</*weighted=*/true>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelClassificationFeatureDiscretizedNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  }
}

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureNumericalHistogram(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<float>& labels, float na_replacement,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition) {
  DCHECK(condition != nullptr);
  if constexpr (weighted) {
    DCHECK_EQ(weights.size(), labels.size());
  } else {
    DCHECK(weights.empty());
  }

  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }
  // Determine the minimum and maximum values of the attribute.
  float min_value, max_value;
  if (!MinMaxNumericalAttribute(selected_examples, attributes, &min_value,
                                &max_value)) {
    return SplitSearchResult::kInvalidAttribute;
  }

  // There should be at least two different unique values.
  if (min_value == max_value) {
    return SplitSearchResult::kInvalidAttribute;
  }
  // Randomly select some threshold values.
  struct CandidateSplit {
    float threshold;
    utils::NormalDistributionDouble pos_label_dist;
    int64_t num_positive_examples_without_weights = 0;
    bool operator<(const CandidateSplit& other) const {
      return threshold < other.threshold;
    }
  };
  ASSIGN_OR_RETURN(
      const auto bins,
      internal::GenHistogramBins(dt_config.numerical_split().type(),
                                 dt_config.numerical_split().num_candidates(),
                                 attributes, min_value, max_value, random));

  std::vector<CandidateSplit> candidate_splits(bins.size());
  for (int split_idx = 0; split_idx < candidate_splits.size(); split_idx++) {
    auto& candidate_split = candidate_splits[split_idx];
    candidate_split.threshold = bins[split_idx];
  }

  // Compute the split score of each threshold.
  for (const auto example_idx : selected_examples) {
    const float label = labels[example_idx];
    float attribute = attributes[example_idx];
    if (std::isnan(attribute)) {
      attribute = na_replacement;
    }

    auto it_split = std::upper_bound(
        candidate_splits.begin(), candidate_splits.end(), attribute,
        [](const float a, const CandidateSplit& b) { return a < b.threshold; });
    if (it_split == candidate_splits.begin()) {
      continue;
    }
    --it_split;
    it_split->num_positive_examples_without_weights++;
    if constexpr (weighted) {
      it_split->pos_label_dist.Add(label, weights[example_idx]);
    } else {
      it_split->pos_label_dist.Add(label);
    }
  }

  for (int split_idx = candidate_splits.size() - 2; split_idx >= 0;
       split_idx--) {
    const auto& src = candidate_splits[split_idx + 1];
    auto& dst = candidate_splits[split_idx];
    dst.num_positive_examples_without_weights +=
        src.num_positive_examples_without_weights;
    dst.pos_label_dist.Add(src.pos_label_dist);
  }

  // Select the best threshold.
  const double initial_variance = label_distribution.Var();
  int best_candidate_split_idx = -1;
  double best_variance_reduction = condition->split_score();
  utils::NormalDistributionDouble neg_label_dist;
  for (int candidate_split_idx = 0;
       candidate_split_idx < candidate_splits.size(); candidate_split_idx++) {
    const auto& candidate_split = candidate_splits[candidate_split_idx];
    if (selected_examples.size() -
                candidate_split.num_positive_examples_without_weights <
            min_num_obs ||
        candidate_split.num_positive_examples_without_weights < min_num_obs) {
      continue;
    }
    neg_label_dist = label_distribution;
    neg_label_dist.Sub(candidate_split.pos_label_dist);
    const double frac_pos = candidate_split.pos_label_dist.NumObservations() /
                            (candidate_split.pos_label_dist.NumObservations() +
                             neg_label_dist.NumObservations());
    const double final_variance =
        frac_pos * candidate_split.pos_label_dist.Var() +
        (1 - frac_pos) * neg_label_dist.Var();
    const double variance_reduction = initial_variance - final_variance;
    if (variance_reduction > best_variance_reduction) {
      best_variance_reduction = variance_reduction;
      best_candidate_split_idx = candidate_split_idx;
    }
  }

  if (best_candidate_split_idx == -1) {
    return SplitSearchResult::kNoBetterSplitFound;
  } else {
    const auto& candidate_split = candidate_splits[best_candidate_split_idx];
    condition->set_split_score(best_variance_reduction);
    condition->mutable_condition()->mutable_higher_condition()->set_threshold(
        candidate_split.threshold);
    condition->set_attribute(attribute_idx);
    condition->set_num_training_examples_without_weight(
        selected_examples.size());
    condition->set_num_training_examples_with_weight(
        candidate_split.pos_label_dist.NumObservations() +
        neg_label_dist.NumObservations());
    condition->set_num_pos_training_examples_without_weight(
        candidate_split.num_positive_examples_without_weights);
    condition->set_num_pos_training_examples_with_weight(
        candidate_split.pos_label_dist.NumObservations());
    condition->set_na_value(na_replacement >= candidate_split.threshold);
    return SplitSearchResult::kBetterSplitFound;
  }
}

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    float na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, const int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }

  const auto sorting_strategy =
      EffectiveStrategy(dt_config, selected_examples.size(), internal_config);

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  typename LabelHessianNumericalOneValueBucket<weighted>::Filler label_filler(
      gradients, hessians, weights);

  typename LabelHessianNumericalOneValueBucket<weighted>::Initializer
      initializer(sum_gradient, sum_hessian, sum_weights,
                  internal_config.hessian_l1,
                  internal_config.hessian_l2_numerical,
                  dt_config.internal().hessian_split_score_subtract_parent(),
                  monotonic_direction, constraints);

  if (sorting_strategy ==
      proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
    const auto& sorted_attributes =
        internal_config.preprocessing
            ->presorted_numerical_features()[attribute_idx];
    return ScanSplitsPresortedSparse<
        FeatureNumericalLabelHessianNumericalOneValue<weighted>,
        LabelHessianNumericalScoreAccumulator>(
        internal_config.preprocessing->num_examples(), selected_examples,
        sorted_attributes.items, feature_filler, label_filler, initializer,
        min_num_obs, attribute_idx,
        internal_config.duplicated_selected_examples, condition,
        &cache->cache_v2);
  } else if (sorting_strategy ==
             proto::DecisionTreeTrainingConfig::Internal::IN_NODE) {
    return FindBestSplit_LabelHessianRegressionFeatureNumerical<weighted>(
        selected_examples, feature_filler, label_filler, initializer,
        min_num_obs, attribute_idx, condition, &cache->cache_v2);
  } else {
    return absl::InvalidArgumentError("Non supported strategy");
  }
}

template absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureNumericalCart<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    float na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureNumericalCart<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
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
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    int num_bins, const std::vector<float>& gradients,
    const std::vector<float>& hessians, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  FeatureDiscretizedNumericalBucket::Filler feature_filler(
      num_bins, na_replacement, attributes);

  typename LabelHessianNumericalBucket<weighted>::Filler label_filler(
      gradients, hessians, weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical);

  typename LabelHessianNumericalBucket<weighted>::Initializer initializer(
      sum_gradient, sum_hessian, sum_weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical,
      dt_config.internal().hessian_split_score_subtract_parent(),
      monotonic_direction, constraints);

  return FindBestSplit_LabelHessianRegressionFeatureDiscretizedNumerical<
      weighted>(selected_examples, feature_filler, label_filler, initializer,
                min_num_obs, attribute_idx, condition, &cache->cache_v2);
}

template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelRegressionFeatureNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<float>& labels, float na_replacement,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }

  const auto sorting_strategy =
      EffectiveStrategy(dt_config, selected_examples.size(), internal_config);

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  typename LabelNumericalOneValueBucket<weighted>::Filler label_filler(labels,
                                                                       weights);

  typename LabelNumericalOneValueBucket<weighted>::Initializer initializer(
      label_distribution);

  if (sorting_strategy ==
      proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
    const auto& sorted_attributes =
        internal_config.preprocessing
            ->presorted_numerical_features()[attribute_idx];
    return ScanSplitsPresortedSparse<
        FeatureNumericalLabelNumericalOneValue<weighted>,
        LabelNumericalScoreAccumulator>(
        internal_config.preprocessing->num_examples(), selected_examples,
        sorted_attributes.items, feature_filler, label_filler, initializer,
        min_num_obs, attribute_idx,
        internal_config.duplicated_selected_examples, condition,
        &cache->cache_v2);
  } else if (sorting_strategy ==
             proto::DecisionTreeTrainingConfig::Internal::IN_NODE) {
    return FindBestSplit_LabelRegressionFeatureNumerical<weighted>(
        selected_examples, feature_filler, label_filler, initializer,
        min_num_obs, attribute_idx, condition, &cache->cache_v2);
  } else {
    return absl::InvalidArgumentError("Non supported strategy");
  }
}

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureNumericalCart<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<float>& labels, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureNumericalCart<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<float>& labels, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureDiscretizedNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    const int num_bins, const std::vector<float>& labels,
    const dataset::DiscretizedNumericalIndex na_replacement,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  FeatureDiscretizedNumericalBucket::Filler feature_filler(
      num_bins, na_replacement, attributes);

  typename LabelNumericalBucket<weighted>::Filler label_filler(labels, weights);

  typename LabelNumericalBucket<weighted>::Initializer initializer(
      label_distribution);

  return FindBestSplit_LabelRegressionFeatureDiscretizedNumerical<weighted>(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

absl::StatusOr<SplitSearchResult> FindSplitLabelClassificationFeatureNA(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<int32_t>& labels, const int32_t num_label_classes,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if (!weights.empty()) {
    DCHECK_EQ(weights.size(), labels.size());
  }
  FeatureIsMissingBucket::Filler feature_filler(attributes);
  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      LabelBinaryCategoricalBucket</*weighted=*/false>::Filler label_filler(
          labels, {}, label_distribution);

      LabelBinaryCategoricalBucket</*weighted=*/false>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedBinaryClassificationFeatureNACart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelBinaryCategoricalBucket</*weighted=*/true>::Filler label_filler(
          labels, weights, label_distribution);

      LabelBinaryCategoricalBucket</*weighted=*/true>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelBinaryClassificationFeatureNACart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  } else {
    // Multi-class classification.
    if (weights.empty()) {
      LabelCategoricalBucket</*weighted=*/false>::Filler label_filler(
          labels, weights, label_distribution);
      LabelCategoricalBucket</*weighted=*/false>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedClassificationFeatureNACart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelCategoricalBucket</*weighted=*/true>::Filler label_filler(
          labels, weights, label_distribution);
      LabelCategoricalBucket</*weighted=*/true>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelClassificationFeatureNACart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  }
}

template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelHessianRegressionFeatureNA(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  FeatureIsMissingBucket::Filler feature_filler(attributes);

  typename LabelHessianNumericalBucket<weighted>::Filler label_filler(
      gradients, hessians, weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical);

  typename LabelHessianNumericalBucket<weighted>::Initializer initializer(
      sum_gradient, sum_hessian, sum_weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical,
      dt_config.internal().hessian_split_score_subtract_parent(),
      /*monotonic_direction=*/0, constraints);

  return FindBestSplit_LabelHessianRegressionFeatureNACart<weighted>(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

absl::StatusOr<SplitSearchResult> FindSplitLabelClassificationFeatureBoolean(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<int32_t>& labels, const int32_t num_label_classes,
    bool na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if (!weights.empty()) {
    DCHECK_EQ(weights.size(), labels.size());
  }
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForBooleanAttribute(selected_examples, weights, attributes,
                                       &na_replacement);
  }

  FeatureBooleanBucket::Filler feature_filler(na_replacement, attributes);

  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      LabelBinaryCategoricalBucket</*weighted=*/false>::Filler label_filler(
          labels, {}, label_distribution);

      LabelBinaryCategoricalBucket</*weighted=*/false>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedBinaryClassificationFeatureBooleanCart(  // NOLINT(whitespace/line_length)
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelBinaryCategoricalBucket</*weighted=*/true>::Filler label_filler(
          labels, weights, label_distribution);

      LabelBinaryCategoricalBucket</*weighted=*/true>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelBinaryClassificationFeatureBooleanCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  } else {
    // Multi-class classification.
    if (weights.empty()) {
      LabelCategoricalBucket</*weighted=*/false>::Filler label_filler(
          labels, weights, label_distribution);

      LabelCategoricalBucket</*weighted=*/false>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedClassificationFeatureBooleanCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelCategoricalBucket</*weighted=*/true>::Filler label_filler(
          labels, weights, label_distribution);

      LabelCategoricalBucket</*weighted=*/true>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelClassificationFeatureBooleanCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  }
}

template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelRegressionFeatureBoolean(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<float>& labels, bool na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }

  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForBooleanAttribute(selected_examples, weights, attributes,
                                       &na_replacement);
  }

  FeatureBooleanBucket::Filler feature_filler(na_replacement, attributes);
  typename LabelNumericalBucket<weighted>::Filler label_filler(labels, weights);
  typename LabelNumericalBucket<weighted>::Initializer initializer(
      label_distribution);

  return FindBestSplit_LabelRegressionFeatureBooleanCart<weighted>(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureBoolean<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<float>& labels, bool na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);
template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureBoolean<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<float>& labels, bool na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelHessianRegressionFeatureBoolean(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    bool na_replacement, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForBooleanAttribute(selected_examples, weights, attributes,
                                       &na_replacement);
  }

  FeatureBooleanBucket::Filler feature_filler(na_replacement, attributes);
  typename LabelHessianNumericalBucket<weighted>::Filler label_filler(
      gradients, hessians, weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical);

  typename LabelHessianNumericalBucket<weighted>::Initializer initializer(
      sum_gradient, sum_hessian, sum_weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical,
      dt_config.internal().hessian_split_score_subtract_parent(),
      /*monotonic_direction=*/0, constraints);

  return FindBestSplit_LabelHessianRegressionFeatureBooleanCart<weighted>(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureCategorical(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    const int32_t num_attribute_classes, int32_t na_replacement,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);
  typename LabelHessianNumericalBucket<weighted>::Filler label_filler(
      gradients, hessians, weights, internal_config.hessian_l1,
      internal_config.hessian_l2_categorical);

  typename LabelHessianNumericalBucket<weighted>::Initializer initializer(
      sum_gradient, sum_hessian, sum_weights, internal_config.hessian_l1,
      internal_config.hessian_l2_categorical,
      dt_config.internal().hessian_split_score_subtract_parent(),
      /*monotonic_direction=*/0, constraints);

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return FindBestSplit_LabelHessianRegressionFeatureCategoricalCart<
          weighted>(selected_examples, feature_filler, label_filler,
                    initializer, min_num_obs, attribute_idx, condition,
                    &cache->cache_v2);

    case proto::Categorical::kRandom:
      return FindBestSplit_LabelHessianRegressionFeatureCategoricalRandom<
          weighted>(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);

    default:
      return absl::InvalidArgumentError("Non supported");
  }
}

template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelRegressionFeatureCategorical(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<float>& labels, const int32_t num_attribute_classes,
    int32_t na_replacement, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }

  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);
  typename LabelNumericalBucket<weighted>::Filler label_filler(labels, weights);

  typename LabelNumericalBucket<weighted>::Initializer initializer(
      label_distribution);

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return FindBestSplit_LabelRegressionFeatureCategoricalCart<weighted>(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);

    case proto::Categorical::kRandom:
      return FindBestSplit_LabelRegressionFeatureCategoricalRandom<weighted>(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);

    default:
      return absl::InvalidArgumentError("Non supported");
  }
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureCategoricalSetGreedyForward(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<int32_t>& labels, const int32_t num_attribute_classes,
    const int32_t num_label_classes, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random) {
  if (!weights.empty()) {
    DCHECK_EQ(weights.size(), labels.size());
  }
  // Bitmap of available attribute values. During the course of the algorithm,
  // an attribute value is available if:
  //  - It is selected by the initial random sampling of candidate attribute
  //  values.
  //  - It is not (yet) selected in the positive set.
  //  - It is not pure in the negative examples i.e. it is not present in all
  //  or in none of the non-selected examples (ps: Initially, all the examples
  //  are non-selected).
  std::vector<bool> candidate_attributes_bitmap(num_attribute_classes, true);
  // The "positive attribute set" are the attribute values that, if present
  // in the example, evaluates the node condition as true.
  std::vector<int> positive_attributes_vector;
  // Bitmap of the example that are already in the positive set i.e. for which
  // the condition defined by "positive_attributes_vector" is positive.
  // Instead of being indexed by the example_idx, this bitmap is indexed by
  // "selected_examples" i.e. "positive_selected_example_bitmap[i]==true"
  // means that "selected_examples[i]" is selected.
  std::vector<bool> positive_selected_example_bitmap(selected_examples.size(),
                                                     false);
  // Weighted and non weighted distribution of the labels in the positive and
  // negative sets.
  utils::BinaryToIntegerConfusionMatrixDouble split_label_distribution;
  utils::BinaryToIntegerConfusionMatrixInt64
      split_label_distribution_no_weights;
  split_label_distribution.SetNumClassesIntDim(num_label_classes);
  split_label_distribution_no_weights.SetNumClassesIntDim(num_label_classes);
  // All the examples are initially in the negative set.
  *split_label_distribution.mutable_neg() = label_distribution;
  // Number of example (with weights) where the attribute value (an attribute
  // value is a set of categorical items) that contains the i-th  categorical
  // items.
  std::vector<int64_t> count_examples_without_weights_by_attribute_class(
      num_attribute_classes);
  // Count per categorical item value.
  const auto& attribute_values = attributes.values();
  const auto& attribute_bank = attributes.bank();
  for (const auto example_idx : selected_examples) {
    for (auto bank_idx = attribute_values[example_idx].first;
         bank_idx < attribute_values[example_idx].second; bank_idx++) {
      const auto value = attribute_bank[bank_idx];
      count_examples_without_weights_by_attribute_class[value]++;
    }
    split_label_distribution_no_weights.Add(false, labels[example_idx]);
  }
  // Sample-out items.
  if (!internal::MaskPureSampledOrPrunedItemsForCategoricalSetGreedySelection(
          dt_config, num_attribute_classes, selected_examples,
          count_examples_without_weights_by_attribute_class,
          &candidate_attributes_bitmap, random)) {
    return SplitSearchResult::kInvalidAttribute;
  }

  // Contains, for each example, the index in the "attribute_bank"
  // corresponding to the next candidate attribute value
  // "candidate_attr_value".
  //
  // When initialized, this corresponds to the index of the first values for
  // this particular example: i.e. running_attr_bank_idx[select_idx] ==
  // attribute_values[selected_examples[select_idx]].first.
  std::vector<UnsignedExampleIdx> running_attr_bank_idx(
      selected_examples.size());

  const int max_iterations =
      dt_config.categorical_set_greedy_forward().max_selected_items();
  int num_iterations = 0;

  const double initial_entropy = split_label_distribution.InitEntropy();
  // Information gain of the current condition i.e.
  // "positive_attributes_vector".
  double information_gain = 0;
  while (true) {
    if (max_iterations > 0 && num_iterations >= max_iterations) {
      break;
    }

    // Initialize the running attribute bank index.
    for (size_t select_idx = 0; select_idx < selected_examples.size();
         select_idx++) {
      const auto example_idx = selected_examples[select_idx];
      running_attr_bank_idx[select_idx] = attribute_values[example_idx].first;
    }
    // Search for the attribute item that maximize the information gain. Note:
    // We ignore attribute that reduce the current information gain.
    double best_candidate_information_gain = information_gain;
    int best_attr_value = -1;
    for (int candidate_attr_value = 0;
         candidate_attr_value < num_attribute_classes; candidate_attr_value++) {
      if (!candidate_attributes_bitmap[candidate_attr_value]) {
        // The attribute value was already selected or was excluded from the
        // search space.
        continue;
      }
      // Compute the information gains of the already selected values
      // "positive_attributes_vector" and the candidate value
      // "candidate_attr_value".
      utils::BinaryToIntegerConfusionMatrixDouble
          candidate_split_label_distribution = split_label_distribution;
      int64_t num_absent_in_negative_set = 0;
      for (size_t select_idx = 0; select_idx < selected_examples.size();
           select_idx++) {
        const auto example_idx = selected_examples[select_idx];
        if (positive_selected_example_bitmap[select_idx]) {
          // The example is already in the positive set.
          continue;
        }

        // Search if X = attribute_bank[ attribute_values[example_idx].first,
        // attribute_values[example_idx].second ] contains the current
        // candidate attribute value "candidate_attr_value".
        //
        // We use "running_attr_bank_idx[select_idx]" that contains the index
        // in "X" of the last tested candidate attribute.
        //
        // Note: "X" does not contains duplicates and is sorted in increasing
        // order. The candidate attributes "candidate_attr_value" are testing
        // in increasing order.
        bool match = false;
        UnsignedExampleIdx last_attr;
        while (
            running_attr_bank_idx[select_idx] <
                attribute_values[example_idx].second &&
            (last_attr = attribute_bank[running_attr_bank_idx[select_idx]]) <=
                candidate_attr_value) {
          running_attr_bank_idx[select_idx]++;
          if (last_attr == candidate_attr_value) {
            match = true;
            break;
          }
        }

        if (match) {
          // Add the example to the positive set and remove it from the
          // negative.
          float weight = weights.empty() ? 1.f : weights[example_idx];
          candidate_split_label_distribution.mutable_pos()->Add(
              labels[example_idx], weight);
          candidate_split_label_distribution.mutable_neg()->Add(
              labels[example_idx], -weight);
        } else {
          num_absent_in_negative_set++;
        }
      }
      // Remove the attribute from the candidate set if the attribute is pure
      // for the current negative set.
      if (num_absent_in_negative_set == 0 ||
          num_absent_in_negative_set == selected_examples.size()) {
        candidate_attributes_bitmap[candidate_attr_value] = false;
        continue;
      }

      const double candidate_information_gain =
          initial_entropy - candidate_split_label_distribution.FinalEntropy();
      if (candidate_information_gain > best_candidate_information_gain) {
        // Best score so far.
        best_candidate_information_gain = candidate_information_gain;
        best_attr_value = candidate_attr_value;
      }
    }
    // Check if a satisfying attribute item was found.
    if (best_attr_value == -1) {
      break;
    }
    // Add the attribute item to the positive set.
    candidate_attributes_bitmap[best_attr_value] = false;
    positive_attributes_vector.push_back(best_attr_value);
    information_gain = best_candidate_information_gain;
    // Update the label distributions in the positive and negative sets.
    for (size_t select_idx = 0; select_idx < selected_examples.size();
         select_idx++) {
      const auto example_idx = selected_examples[select_idx];
      if (positive_selected_example_bitmap[select_idx]) {
        // The example is already in the positive set.
        continue;
      }
      const bool match =
          (attribute_values[example_idx].first <
           attribute_values[example_idx].second) &&
          std::binary_search(
              attribute_bank.begin() + attribute_values[example_idx].first,
              attribute_bank.begin() + attribute_values[example_idx].second,
              best_attr_value);
      if (match) {
        positive_selected_example_bitmap[select_idx] = true;
        float weight = weights.empty() ? 1.f : weights[example_idx];
        split_label_distribution.mutable_pos()->Add(labels[example_idx],
                                                    weight);
        split_label_distribution.mutable_neg()->Add(labels[example_idx],
                                                    -weight);
        split_label_distribution_no_weights.mutable_pos()->Add(
            labels[example_idx], 1);
        split_label_distribution_no_weights.mutable_neg()->Add(
            labels[example_idx], -1);
      }
    }

    num_iterations++;
  }

  if (information_gain > condition->split_score()) {
    condition->set_na_value(false);
    SetConditionHelper(information_gain, attribute_idx,
                       split_label_distribution,
                       split_label_distribution_no_weights, condition);
    // Assign the positive set to the condition.
    std::sort(positive_attributes_vector.begin(),
              positive_attributes_vector.end());
    SetPositiveAttributeSetOfCategoricalContainsCondition(
        positive_attributes_vector, num_attribute_classes, condition);
    return SplitSearchResult::kBetterSplitFound;
  } else {
    return SplitSearchResult::kNoBetterSplitFound;
  }
}

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureCategoricalSetGreedyForward(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<float>& labels, int32_t num_attribute_classes,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random) {
  if constexpr (weighted) {
    DCHECK_EQ(weights.size(), labels.size());
  } else {
    DCHECK(weights.empty());
  }
  // Bitmap of available attribute values. During the course of the algorithm,
  // an attribute value is available if:
  //  - It is selected by the initial random sampling of candidate attribute
  //  values.
  //  - It is not (yet) selected in the positive set.
  //  - It is not pure in the negative examples i.e. it is not present in all
  //  or in none of the non-selected examples (ps: Initially, all the examples
  //  are non-selected).
  std::vector<bool> candidate_attributes_bitmap(num_attribute_classes, true);
  // The "positive attribute set" are the attribute values that, if present
  // in the example, evaluates the node condition as true.
  std::vector<int> positive_attributes_vector;
  // Bitmap of the example that are already in the positive set i.e. for which
  // the condition defined by "positive_attributes_vector" is positive.
  // Instead of being indexed by the example_idx, this bitmap is indexed by
  // "selected_examples" i.e. "positive_selected_example_bitmap[i]==true"
  // means that "selected_examples[i]" is selected.
  std::vector<bool> positive_selected_example_bitmap(selected_examples.size(),
                                                     false);
  // Weighted and non weighted distribution of the labels in the positive and
  // negative sets.
  utils::BinaryToNormalDistributionDouble split_label_distribution;
  utils::BinaryToNormalDistributionDouble split_label_distribution_no_weights;
  // All the examples are initially in the negative set.
  *split_label_distribution.mutable_neg() = label_distribution;
  // Number of example (with weights) where the attribute value (an attribute
  // value is a set of categorical items) that contains the i-th  categorical
  // items.
  std::vector<int64_t> count_examples_without_weights_by_attribute_class(
      num_attribute_classes);
  // Count per categorical item value.
  const auto& attribute_values = attributes.values();
  const auto& attribute_bank = attributes.bank();
  for (const auto example_idx : selected_examples) {
    for (auto bank_idx = attribute_values[example_idx].first;
         bank_idx < attribute_values[example_idx].second; bank_idx++) {
      const auto value = attribute_bank[bank_idx];
      count_examples_without_weights_by_attribute_class[value]++;
    }
    split_label_distribution_no_weights.Add(false, labels[example_idx]);
  }

  // Sample-out items.
  if (!internal::MaskPureSampledOrPrunedItemsForCategoricalSetGreedySelection(
          dt_config, num_attribute_classes, selected_examples,
          count_examples_without_weights_by_attribute_class,
          &candidate_attributes_bitmap, random)) {
    return SplitSearchResult::kInvalidAttribute;
  }

  // Contains, for each example, the index in the "attribute_bank"
  // corresponding to the next candidate attribute value
  // "candidate_attr_value".
  //
  // When initialized, this corresponds to the index of the first values for
  // this particular example: i.e. running_attr_bank_idx[select_idx] ==
  // attribute_values[selected_examples[select_idx]].first.
  std::vector<UnsignedExampleIdx> running_attr_bank_idx(
      selected_examples.size());

  const double initial_variance = label_distribution.Var();
  // Variance reduction of the current condition i.e.
  // "positive_attributes_vector".
  double variance_reduction = 0;
  while (true) {
    // Initialize the running attribute bank index.
    for (size_t select_idx = 0; select_idx < selected_examples.size();
         select_idx++) {
      const auto example_idx = selected_examples[select_idx];
      running_attr_bank_idx[select_idx] = attribute_values[example_idx].first;
    }
    // Find the attribute with best immediate variance reduction.
    int best_attr_value;
    double best_candidate_variance_reduction;
    std::tie(best_attr_value, best_candidate_variance_reduction) =
        GetAttributeValueWithMaximumVarianceReduction<weighted>(
            variance_reduction, num_attribute_classes, split_label_distribution,
            selected_examples, positive_selected_example_bitmap,
            attribute_values, attribute_bank, weights, labels, initial_variance,
            &running_attr_bank_idx, &candidate_attributes_bitmap);
    // Check if a satisfying attribute item was found.
    if (best_attr_value == -1) {
      break;
    }
    // Add the attribute item to the positive set.
    candidate_attributes_bitmap[best_attr_value] = false;
    positive_attributes_vector.push_back(best_attr_value);
    variance_reduction = best_candidate_variance_reduction;
    // Update the label distributions in the positive and negative sets.
    for (size_t select_idx = 0; select_idx < selected_examples.size();
         select_idx++) {
      const auto example_idx = selected_examples[select_idx];
      if (positive_selected_example_bitmap[select_idx]) {
        // The example is already in the positive set.
        continue;
      }
      const bool match =
          (attribute_values[example_idx].first <
           attribute_values[example_idx].second) &&
          std::binary_search(
              attribute_bank.begin() + attribute_values[example_idx].first,
              attribute_bank.begin() + attribute_values[example_idx].second,
              best_attr_value);
      if (match) {
        positive_selected_example_bitmap[select_idx] = true;
        if constexpr (weighted) {
          split_label_distribution.mutable_pos()->Add(labels[example_idx],
                                                      weights[example_idx]);
          split_label_distribution.mutable_neg()->Sub(labels[example_idx],
                                                      weights[example_idx]);
        } else {
          split_label_distribution.mutable_pos()->Add(labels[example_idx]);
          split_label_distribution.mutable_neg()->Sub(labels[example_idx]);
        }
        split_label_distribution_no_weights.mutable_pos()->Add(
            labels[example_idx]);
        split_label_distribution_no_weights.mutable_neg()->Sub(
            labels[example_idx]);
      }
    }
  }

  if (variance_reduction > condition->split_score()) {
    condition->set_na_value(false);
    SetConditionHelper(variance_reduction, attribute_idx,
                       split_label_distribution,
                       split_label_distribution_no_weights, condition);
    // Assign the positive set to the condition.
    std::sort(positive_attributes_vector.begin(),
              positive_attributes_vector.end());
    SetPositiveAttributeSetOfCategoricalContainsCondition(
        positive_attributes_vector, num_attribute_classes, condition);
    return SplitSearchResult::kBetterSplitFound;
  } else {
    return SplitSearchResult::kNoBetterSplitFound;
  }
}

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureCategoricalSetGreedyForward<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<float>& labels, int32_t num_attribute_classes,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureCategoricalSetGreedyForward<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<float>& labels, int32_t num_attribute_classes,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random);

template <typename LabelBucket, typename ExampleBucketSet,
          typename LabelScoreAccumulator>
absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureCategorical(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<int32_t>& labels, int32_t num_attribute_classes,
    int32_t num_label_classes, int32_t na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);
  typename LabelBucket::Filler label_filler(labels, weights,
                                            label_distribution);

  typename LabelBucket::Initializer initializer(label_distribution);

  // Create buckets.
  ExampleBucketSet& example_set_accumulator =
      *GetCachedExampleBucketSet<ExampleBucketSet>(&cache->cache_v2);
  FillExampleBucketSet<ExampleBucketSet, /*require_label_sorting=*/false>(
      selected_examples, feature_filler, label_filler, &example_set_accumulator,
      &cache->cache_v2);

  // Scanner for the "one label value vs others".
  const auto one_vs_other_scan = [&]() -> SplitSearchResult {
    // Value and index of the buckets.
    auto& bucket_order = cache->cache_v2.bucket_order;
    bucket_order.resize(example_set_accumulator.items.size());

    SplitSearchResult split_status = SplitSearchResult::kInvalidAttribute;
    for (int32_t positive_label_value = 0;
         positive_label_value < num_label_classes; positive_label_value++) {
      if (label_distribution.count(positive_label_value) == 0) {
        // Never observed label value.
        continue;
      }
      if (num_label_classes == 3 && positive_label_value == 1) {
        // "True vs others" or "False vs others" are equivalent for binary
        // classification.
        continue;
      }

      // Order value of the buckets.
      for (int bucket_idx = 0; bucket_idx < bucket_order.size(); bucket_idx++) {
        const auto& bucket = example_set_accumulator.items[bucket_idx];
        const float ratio_positive_label =
            bucket.label.SafeProportionOrMinusInfinity(positive_label_value);
        DCHECK(!std::isnan(ratio_positive_label));
        bucket_order[bucket_idx] = {ratio_positive_label, bucket_idx};
      }

      // Sort the bucket indices.
      std::sort(bucket_order.begin(), bucket_order.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

      // Scan the buckets in order.
      const auto scan_result =
          ScanSplitsCustomOrder<ExampleBucketSet, LabelScoreAccumulator>(
              bucket_order, feature_filler, initializer,
              example_set_accumulator, selected_examples.size(), min_num_obs,
              attribute_idx, condition, &cache->cache_v2);
      if (scan_result < split_status) {
        split_status = scan_result;
      }
    }
    return split_status;
  };

  // Scanner for the "one hot" type condition i.e. conditions of the type:
  // "attribute == value".
  //
  // Note: In the majority of cases, one-hot (on attribute value) is worst that
  // "one class vs others". This is however a common solution, and this code is
  // present for comparison purpose.
  const auto one_hot_scan = [&]() -> absl::StatusOr<SplitSearchResult> {
    STATUS_CHECK_EQ(example_set_accumulator.items.size(),
                    num_attribute_classes);

    std::uniform_real_distribution<float> sampling_dist;

    auto& neg = *GetCachedLabelScoreAccumulator<LabelScoreAccumulator>(
        false, &cache->cache_v2);
    auto& pos = *GetCachedLabelScoreAccumulator<LabelScoreAccumulator>(
        true, &cache->cache_v2);

    initializer.InitFull(&pos);
    const double weighted_num_examples = pos.WeightedNumExamples();

    double best_score = condition->split_score();
    bool tried_one_split = false;
    int best_bucket_idx = -1;

    for (int attribute_value = 0; attribute_value < num_attribute_classes;
         attribute_value++) {
      if (dt_config.categorical().one_hot().sampling() < 1.f &&
          sampling_dist(*random) >
              dt_config.categorical().one_hot().sampling()) {
        continue;
      }
      const auto bucket_idx = attribute_value;
      const auto& item = example_set_accumulator.items[bucket_idx];

      const int64_t num_pos_examples = item.label.count;
      const int64_t num_neg_examples =
          selected_examples.size() - item.label.count;

      // Enough examples?
      if (num_pos_examples < min_num_obs || num_neg_examples < min_num_obs) {
        continue;
      }

      initializer.InitFull(&neg);
      initializer.InitEmpty(&pos);

      item.label.SubToScoreAcc(&neg);
      item.label.AddToScoreAcc(&pos);

      const auto score = Score<>(initializer, weighted_num_examples, pos, neg);
      tried_one_split = true;

      if (score > best_score) {
        // Memorize the split.
        best_bucket_idx = bucket_idx;
        best_score = score;
        condition->set_num_pos_training_examples_without_weight(
            num_pos_examples);
        condition->set_num_pos_training_examples_with_weight(
            pos.WeightedNumExamples());
      }
    }

    if (best_bucket_idx != -1) {
      // Finalize the best found split.
      condition->set_na_value(na_replacement == best_bucket_idx);
      SetPositiveAttributeSetOfCategoricalContainsCondition(
          {best_bucket_idx}, num_attribute_classes, condition);

      condition->set_attribute(attribute_idx);
      condition->set_num_training_examples_without_weight(
          selected_examples.size());
      condition->set_num_training_examples_with_weight(weighted_num_examples);
      condition->set_split_score(best_score);
      return SplitSearchResult::kBetterSplitFound;
    } else {
      return tried_one_split ? SplitSearchResult::kNoBetterSplitFound
                             : SplitSearchResult::kInvalidAttribute;
    }
    return absl::OkStatus();
  };

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return one_vs_other_scan();

    case proto::Categorical::kOneHot:
      return one_hot_scan();
      break;

    case proto::Categorical::kRandom:
      return ScanSplitsRandomBuckets<ExampleBucketSet, LabelScoreAccumulator>(
          feature_filler, label_filler, initializer, example_set_accumulator,
          selected_examples.size(), min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);
  }
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureCategorical(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<int32_t>& labels, int32_t num_attribute_classes,
    int32_t num_label_classes, int32_t na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  if (!weights.empty()) {
    DCHECK_EQ(weights.size(), labels.size());
  }
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      return FindSplitLabelClassificationFeatureCategorical<
          LabelBinaryCategoricalBucket</*weighted=*/false>,
          FeatureCategoricalLabelUnweightedBinaryCategorical,
          LabelBinaryCategoricalScoreAccumulator>(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, attribute_idx, random, condition, cache);
    } else {
      return FindSplitLabelClassificationFeatureCategorical<
          LabelBinaryCategoricalBucket</*weighted=*/true>,
          FeatureCategoricalLabelBinaryCategorical,
          LabelBinaryCategoricalScoreAccumulator>(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, attribute_idx, random, condition, cache);
    }
  } else {
    // Multi-class classification.
    if (weights.empty()) {
      return FindSplitLabelClassificationFeatureCategorical<
          LabelCategoricalBucket</*weighted=*/false>,
          FeatureCategoricalLabelUnweightedCategorical,
          LabelCategoricalScoreAccumulator>(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, attribute_idx, random, condition, cache);
    } else {
      return FindSplitLabelClassificationFeatureCategorical<
          LabelCategoricalBucket</*weighted=*/true>,
          FeatureCategoricalLabelCategorical, LabelCategoricalScoreAccumulator>(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, attribute_idx, random, condition, cache);
    }
  }
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelUpliftCategoricalFeatureNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const CategoricalUpliftLabelStats& label_stats, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  DCHECK(!weights.empty());
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  LabelUpliftCategoricalOneValueBucket::Initializer initializer(
      label_stats.label_distribution,
      dt_config.uplift().min_examples_in_treatment(),
      dt_config.uplift().split_score());
  LabelUpliftCategoricalOneValueBucket::Filler label_filler(
      label_stats.outcome_values, label_stats.treatment_values, weights);

  // TODO: Add support for-presorted splitting.

  return FindBestSplit_LabelUpliftClassificationFeatureNumerical(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelUpliftNumericalFeatureNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const NumericalUpliftLabelStats& label_stats, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  DCHECK(!weights.empty());
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  LabelUpliftNumericalOneValueBucket::Initializer initializer(
      label_stats.label_distribution,
      dt_config.uplift().min_examples_in_treatment(),
      dt_config.uplift().split_score());
  LabelUpliftNumericalOneValueBucket::Filler label_filler(
      label_stats.outcome_values, label_stats.treatment_values, weights);

  // TODO: Add support for pre-sorted splitting.

  return FindBestSplit_LabelUpliftNumericalFeatureNumerical(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelUpliftCategoricalFeatureCategorical(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const CategoricalUpliftLabelStats& label_stats, int num_attribute_classes,
    int32_t na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random) {
  DCHECK(!weights.empty());
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);

  LabelUpliftCategoricalBucket::Initializer initializer(
      label_stats.label_distribution,
      dt_config.uplift().min_examples_in_treatment(),
      dt_config.uplift().split_score());
  LabelUpliftCategoricalBucket::Filler label_filler(
      label_stats.label_distribution, label_stats.outcome_values,
      label_stats.treatment_values, weights,
      dt_config.uplift().empty_bucket__ordering());

  // TODO: Add support for pre-sorted splitting.

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return FindBestSplit_LabelUpliftClassificationFeatureCategoricalCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);

    case proto::Categorical::kRandom:
      return FindBestSplit_LabelUpliftClassificationFeatureCategoricalRandom(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);

    default:
      return absl::InvalidArgumentError("Non supported");
  }
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelUpliftNumericalFeatureCategorical(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const NumericalUpliftLabelStats& label_stats, int num_attribute_classes,
    int32_t na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random) {
  DCHECK(!weights.empty());
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);

  LabelUpliftNumericalBucket::Initializer initializer(
      label_stats.label_distribution,
      dt_config.uplift().min_examples_in_treatment(),
      dt_config.uplift().split_score());
  LabelUpliftNumericalBucket::Filler label_filler(
      label_stats.label_distribution, label_stats.outcome_values,
      label_stats.treatment_values, weights,
      dt_config.uplift().empty_bucket__ordering());

  // TODO: Add support for pre-sorted splitting.

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return FindBestSplit_LabelUpliftNumericalFeatureCategoricalCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);

    case proto::Categorical::kRandom:
      return FindBestSplit_LabelUpliftNumericalFeatureCategoricalRandom(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);

    default:
      return absl::InvalidArgumentError("Non supported");
  }
}

int NumAttributesToTest(const proto::DecisionTreeTrainingConfig& dt_config,
                        const int num_attributes,
                        const model::proto::Task task) {
  int num_attributes_to_test;
  // User specified number of candidate attributes.
  if (dt_config.has_num_candidate_attributes_ratio() &&
      dt_config.num_candidate_attributes_ratio() >= 0) {
    if (dt_config.has_num_candidate_attributes() &&
        dt_config.num_candidate_attributes() > 0) {
      LOG(WARNING) << "Both \"num_candidate_attributes\" and "
                      "\"num_candidate_attributes_ratio\" are specified. "
                      "Ignoring \"num_candidate_attributes\".";
    }
    num_attributes_to_test = static_cast<int>(
        std::ceil(dt_config.num_candidate_attributes_ratio() * num_attributes));
  } else {
    num_attributes_to_test = dt_config.num_candidate_attributes();
  }

  // Automatic number of attribute selection logic.
  if (num_attributes_to_test == 0) {
    switch (task) {
      default:
      case model::proto::Task::CATEGORICAL_UPLIFT:
      case model::proto::Task::CLASSIFICATION:
        num_attributes_to_test = static_cast<int>(
            ceil(std::sqrt(static_cast<double>(num_attributes))));
        break;
      case model::proto::Task::REGRESSION:
        num_attributes_to_test =
            static_cast<int>(ceil(static_cast<double>(num_attributes) / 3));
        break;
    }
  }

  // Special value to use all the available attributes.
  if (num_attributes_to_test == -1) {
    num_attributes_to_test = static_cast<int>(num_attributes);
  }

  // Make sure we don't select more than the available attributes.
  num_attributes_to_test =
      std::min(num_attributes_to_test, static_cast<int>(num_attributes));

  return num_attributes_to_test;
}

void GetCandidateAttributes(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    int* num_attributes_to_test, std::vector<int32_t>* candidate_attributes,
    utils::RandomEngine* random) {
  candidate_attributes->assign(config_link.features().begin(),
                               config_link.features().end());
  std::shuffle(candidate_attributes->begin(), candidate_attributes->end(),
               *random);

  *num_attributes_to_test = NumAttributesToTest(
      dt_config, candidate_attributes->size(), config.task());
}

absl::Status GenerateRandomImputation(
    const dataset::VerticalDataset& src, const std::vector<int>& attributes,
    const absl::Span<const UnsignedExampleIdx> examples,
    dataset::VerticalDataset* dst, utils::RandomEngine* random) {
  STATUS_CHECK_EQ(dst->ncol(), 0);
  dst->set_data_spec(src.data_spec());
  RETURN_IF_ERROR(dst->CreateColumnsFromDataspec());
  dst->set_nrow(examples.size());
  for (const auto col_idx : attributes) {
    RETURN_IF_ERROR(GenerateRandomImputationOnColumn(
        src.column(col_idx), examples, dst->mutable_column(col_idx), random));
  }
  return absl::OkStatus();
}

absl::Status GenerateRandomImputationOnColumn(
    const dataset::VerticalDataset::AbstractColumn* src,
    const absl::Span<const UnsignedExampleIdx> examples,
    dataset::VerticalDataset::AbstractColumn* dst,
    utils::RandomEngine* random) {
  STATUS_CHECK_EQ(src->type(), dst->type());
  // Extract the indices of the example with non-na values i.e. the candidate
  // for sampling.
  std::vector<UnsignedExampleIdx> non_na_examples;
  for (const auto example_idx : examples) {
    if (!src->IsNa(example_idx)) {
      non_na_examples.push_back(example_idx);
    }
  }

  if (non_na_examples.empty()) {
    return src->ExtractAndAppend(examples, dst);
  }

  std::uniform_int_distribution<SignedExampleIdx> non_na_example_dist(
      0, std::max(static_cast<SignedExampleIdx>(0),
                  static_cast<SignedExampleIdx>(non_na_examples.size()) - 1));

  std::vector<SignedExampleIdx> source_indices;
  source_indices.resize(examples.size());

  UnsignedExampleIdx local_example_idx = 0;
  for (const auto example_idx : examples) {
    if (src->IsNa(example_idx)) {
      // Sample a random non-na value.
      source_indices[local_example_idx] =
          non_na_examples[non_na_example_dist(*random)];
    } else {
      source_indices[local_example_idx] = example_idx;
    }
    local_example_idx++;
  }
  return src->ExtractAndAppend(source_indices, dst);
}

void SetInternalDefaultHyperParameters(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& link_config,
    const dataset::proto::DataSpecification& data_spec,
    proto::DecisionTreeTrainingConfig* dt_config) {}

void SetDefaultHyperParameters(proto::DecisionTreeTrainingConfig* config) {
  // Emulation of histogram splits.
  if (!config->numerical_split().has_num_candidates()) {
    switch (config->numerical_split().type()) {
      case proto::NumericalSplit::HISTOGRAM_RANDOM:
        config->mutable_numerical_split()->set_num_candidates(1);
        break;
      case proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH:
        config->mutable_numerical_split()->set_num_candidates(255);
        break;
      default:
        break;
    }
  }

  // By default, use axis aligned splits.
  if (config->split_axis_case() ==
      proto::DecisionTreeTrainingConfig::SPLIT_AXIS_NOT_SET) {
    config->mutable_axis_aligned_split();
  }

  // By default, use the cart categorical algorithm for categorical features.
  if (config->categorical().algorithm_case() ==
      proto::Categorical::ALGORITHM_NOT_SET) {
    config->mutable_categorical()->mutable_cart();
  }

  // By default, use the local growing strategy i.e. divide and conquer.
  if (config->growing_strategy_case() ==
      proto::DecisionTreeTrainingConfig::GROWING_STRATEGY_NOT_SET) {
    config->mutable_growing_strategy_local();
  }

  // Change the pre-sorting strategy if not supported by the splitter.
  using Internal = proto::DecisionTreeTrainingConfig::Internal;
  auto sorting_strategy = config->internal().sorting_strategy();

  // If possible, use presorting by default.
  if (sorting_strategy == Internal::AUTO) {
    sorting_strategy = Internal::PRESORTED;
  }

  if (sorting_strategy == Internal::PRESORTED ||
      sorting_strategy == Internal::FORCE_PRESORTED) {
    if (config->has_sparse_oblique_split() ||
        config->has_mhld_oblique_split() ||
        config->missing_value_policy() !=
            proto::DecisionTreeTrainingConfig::GLOBAL_IMPUTATION) {
      sorting_strategy = Internal::IN_NODE;
    }
  }

  config->mutable_internal()->set_sorting_strategy(sorting_strategy);

  // The binary weight hyperparameter is deprecated for the more general weights
  // hyperparameter.
  if (config->sparse_oblique_split().has_binary_weight()) {
    if (config->sparse_oblique_split().binary_weight()) {
      config->mutable_sparse_oblique_split()->mutable_binary();
    } else {
      config->mutable_sparse_oblique_split()->mutable_continuous();
    }
    config->mutable_sparse_oblique_split()->clear_binary_weight();
  }

  // By default, we use binary weights.
  if (config->has_sparse_oblique_split() &&
      config->sparse_oblique_split().weights_case() ==
          proto::DecisionTreeTrainingConfig::SparseObliqueSplit::
              WEIGHTS_NOT_SET) {
    config->mutable_sparse_oblique_split()->mutable_binary();
  }
}

absl::Status GrowTreeBestFirstGlobal(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const std::vector<float>& weights,
    const InternalTrainConfig& internal_config, NodeWithChildren* root,
    utils::RandomEngine* random,
    SelectedExamplesRollingBuffer selected_examples,
    std::optional<SelectedExamplesRollingBuffer> leaf_examples) {
  if (config.monotonic_constraints_size() > 0) {
    return absl::InvalidArgumentError(
        "Global growth of decision trees (i.e. "
        "growing_strategy=kGrowingStrategyBestFirstGlobal) does not support "
        "monotonic constraints.");
  }

  if (leaf_examples.has_value()) {
    return absl::InvalidArgumentError(
        "honest trees are not (yet) supported with "
        "growing_strategy_best_first_global strategy.");
  }

  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::RANDOM_LOCAL_IMPUTATION) {
    return absl::InvalidArgumentError(
        "Random local imputation not supported in best first global "
        "tree growth.");
  }
  if (config_link.per_columns_size() > 0) {
    for (const auto feature : config_link.features()) {
      if (config_link.per_columns(feature).has_monotonic_constraint()) {
        return absl::InvalidArgumentError(
            "GBT with growing_strategy_best_first_global does not support "
            "monotonic constraints.");
      }
    }
  }

  PerThreadCache cache;

  struct CandidateSplit {
    // Split.
    proto::NodeCondition condition;
    // Indices of examples in the node.
    SelectedExamplesRollingBuffer example_idxs;
    // Global score of the split.
    float score;
    // The currently leaf node.
    NodeWithChildren* node;
    // Depth of the node.
    int depth;

    bool operator<(const CandidateSplit& other) const {
      return score < other.score;
    }
  };

  // List of candidate splits.
  std::priority_queue<CandidateSplit> candidate_splits;

  // Initialize a node and update the list of candidate splits with a given
  // node.
  const auto ingest_node = [&](const SelectedExamplesRollingBuffer example_idxs,
                               NodeWithChildren* node,
                               const int depth) -> absl::Status {
    RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
        train_dataset, example_idxs.active, weights, config, config_link,
        node));

    if (example_idxs.size() < dt_config.min_examples() ||
        (dt_config.max_depth() >= 0 && depth >= dt_config.max_depth())) {
      // Stop the grow of the branch.
      node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
      return absl::OkStatus();
    }
    proto::NodeCondition condition;
    ASSIGN_OR_RETURN(
        const auto has_better_condition,
        FindBestCondition(train_dataset, example_idxs.active, weights, config,
                          config_link, dt_config, node->node(), internal_config,
                          {}, &condition, random, &cache));
    if (!has_better_condition) {
      // No good condition found. Close the branch.
      node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
      return absl::OkStatus();
    }

    const float score = condition.split_score() * example_idxs.size();
    candidate_splits.push({/*.condition =*/std::move(condition),
                           /*.example_idxs =*/example_idxs,
                           /*.score =*/score,
                           /*.node =*/node,
                           /*.depth =*/depth});
    return absl::OkStatus();
  };

  RETURN_IF_ERROR(ingest_node(selected_examples, root, /*depth=*/0));

  // Total number of nodes in the tree.
  int num_nodes = 1;

  const int max_num_nodes =
      dt_config.growing_strategy_best_first_global().max_num_nodes();

  while (!candidate_splits.empty() &&
         (max_num_nodes < 0 || num_nodes < max_num_nodes) &&
         (!internal_config.timeout.has_value() ||
          internal_config.timeout >= absl::Now())) {
    // Ensure the candidate set is not larger than  "max_num_nodes". Note:
    // There is not need for mode than "max_num_nodes" candidate splits.
    while (max_num_nodes >= 0 && candidate_splits.size() > max_num_nodes) {
      candidate_splits.top().node->FinalizeAsLeaf(
          dt_config.store_detailed_label_distribution());
      candidate_splits.pop();
    }

    // Split the node.
    auto split = candidate_splits.top();
    candidate_splits.pop();

    *split.node->mutable_node()->mutable_condition() = split.condition;
    split.node->CreateChildren();
    split.node->FinalizeAsNonLeaf(
        dt_config.keep_non_leaf_label_distribution(),
        dt_config.store_detailed_label_distribution());

    const auto& condition = split.node->node().condition();

    // Add new candidate splits for children.

    ASSIGN_OR_RETURN(
        auto exemple_split,
        internal::SplitExamplesInPlace(
            train_dataset, split.example_idxs, condition,
            /*dataset_is_dense=*/false,
            dt_config.internal_error_on_wrong_splitter_statistics()));

    RETURN_IF_ERROR(ingest_node(exemple_split.positive_examples,
                                split.node->mutable_pos_child(),
                                split.depth + 1));
    RETURN_IF_ERROR(ingest_node(exemple_split.negative_examples,
                                split.node->mutable_neg_child(),
                                split.depth + 1));
    num_nodes++;
  }

  // Finalize the remaining candidates.
  while (!candidate_splits.empty()) {
    candidate_splits.top().node->FinalizeAsLeaf(
        dt_config.store_detailed_label_distribution());
    candidate_splits.pop();
  }
  return absl::OkStatus();
}

absl::Status DecisionTreeTrain(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const std::vector<float>& weights, utils::RandomEngine* random,
    DecisionTree* dt, const InternalTrainConfig& internal_config) {
  // Note: This function is the entry point of all decision tree learning.

  // Check the sorting strategy.
  if (dt_config.internal().has_ensure_effective_sorting_strategy() &&
      (dt_config.internal().ensure_effective_sorting_strategy() !=
       dt_config.internal().sorting_strategy())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Non expected effective sorting strategy:",
        proto::DecisionTreeTrainingConfig::Internal::SortingStrategy_Name(
            dt_config.internal().ensure_effective_sorting_strategy()),
        "(expected) != ",
        proto::DecisionTreeTrainingConfig::Internal::SortingStrategy_Name(
            dt_config.internal().sorting_strategy()),
        "(actual)"));
  }

  // Decide if execution should happen in single-thread or concurrent mode.

  std::optional<std::vector<UnsignedExampleIdx>> leaf_examples;
  std::vector<UnsignedExampleIdx> working_selected_examples;

  // Fail if the data spec has invalid columns.
  for (const auto feature_idx : config_link.features()) {
    const auto& data_spec_columns = train_dataset.data_spec().columns();
    const auto column_type = data_spec_columns[feature_idx].type();
    if (column_type != dataset::proto::NUMERICAL &&
        column_type != dataset::proto::CATEGORICAL &&
        column_type != dataset::proto::CATEGORICAL_SET &&
        column_type != dataset::proto::BOOLEAN &&
        column_type != dataset::proto::DISCRETIZED_NUMERICAL &&
        column_type != dataset::proto::NUMERICAL_VECTOR_SEQUENCE) {
      return absl::InvalidArgumentError(
          absl::Substitute("Column $0 has type $1, which is not supported "
                           "for decision tree training.",
                           data_spec_columns[feature_idx].name(),
                           dataset::proto::ColumnType_Name(column_type)));
    }
  }

  // Check monotonic constraints
  if (config.monotonic_constraints_size() > 0 &&
      !dt_config.keep_non_leaf_label_distribution()) {
    return absl::InvalidArgumentError(
        "keep_non_leaf_label_distribution=false is not compatible with "
        "monotonic constraints. To minimize the size of your serving model "
        "(with or without monotonic constraints), use "
        "pure_serving_model=true.");
  }

  // Check if oblique splits are correctly specified
  if (dt_config.has_sparse_oblique_split()) {
    if (dt_config.sparse_oblique_split().has_binary_weight() &&
        dt_config.sparse_oblique_split().weights_case() !=
            dt_config.sparse_oblique_split().WEIGHTS_NOT_SET) {
      return absl::InvalidArgumentError(
          "Both sparse_oblique_split.binary_weights and "
          "sparse_oblique_split.weights are set. Setting "
          "sparse_oblique_split.binary_weights is deprecated and replaced by "
          "just setting sparse_oblique_split.weights.");
    }
    if (dt_config.sparse_oblique_split().power_of_two().max_exponent() > 31) {
      return absl::InvalidArgumentError(
          "The maximum exponent for sparse oblique power-of-two weights cannot "
          "be larger than 31.");
    }
    if (dt_config.sparse_oblique_split().power_of_two().min_exponent() < -31) {
      return absl::InvalidArgumentError(
          "The minimum exponent for sparse oblique power-of-two weights cannot "
          "be smaller than -31.");
    }
    if (dt_config.sparse_oblique_split().power_of_two().min_exponent() >
        dt_config.sparse_oblique_split().power_of_two().max_exponent()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "The minimum exponent for sparse oblique power-of-two weights cannot "
          "be larger than the maximum exponent. Got minimum: $0, maximum: $1",
          dt_config.sparse_oblique_split().power_of_two().min_exponent(),
          dt_config.sparse_oblique_split().power_of_two().max_exponent()));
    }
    if (dt_config.sparse_oblique_split().integer().minimum() >
        dt_config.sparse_oblique_split().integer().maximum()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "The minimum value for sparse oblique integer weights cannot "
          "be larger than the maximum value. Got minimum: $0, maximum: $1",
          dt_config.sparse_oblique_split().integer().minimum(),
          dt_config.sparse_oblique_split().integer().maximum()));
    }
  }

  if (dt_config.has_honest()) {
    // Split the examples in two parts. One ("selected_examples_buffer") will be
    // used to infer the structure of the trees while the second
    // ("leaf_examples_buffer") will be used to determine the leaf values (i.e.
    // the predictions).

    const float leaf_rate = dt_config.honest().ratio_leaf_examples();
    std::uniform_real_distribution<float> dist_01;

    // Reduce the risk of std::vector re-allocations.
    const float error_margin = 1.1f;
    leaf_examples = std::vector<UnsignedExampleIdx>();
    auto& leaf_examples_value = leaf_examples.value();
    leaf_examples_value.reserve(selected_examples.size() * leaf_rate *
                                error_margin);
    working_selected_examples.reserve(selected_examples.size() *
                                      (1.f - leaf_rate) * error_margin);

    auto* effective_random = random;
    utils::RandomEngine fixed_random(12345678);
    if (dt_config.honest().fixed_separation()) {
      effective_random = &fixed_random;
    }

    for (const auto& example : selected_examples) {
      if (dist_01(*effective_random) < leaf_rate) {
        leaf_examples_value.push_back(example);
      } else {
        working_selected_examples.push_back(example);
      }
    }
  } else {
    working_selected_examples.assign(selected_examples.begin(),
                                     selected_examples.end());
  }

  auto leaf_example_span = leaf_examples.has_value()
                               ? std::optional<absl::Span<UnsignedExampleIdx>>(
                                     absl::MakeSpan(leaf_examples.value()))
                               : std::nullopt;

  return DecisionTreeCoreTrain(train_dataset, config, config_link, dt_config,
                               deployment, weights, random, internal_config, dt,
                               absl::MakeSpan(working_selected_examples),
                               leaf_example_span);
}

std::unique_ptr<SplitterFinderStreamProcessor>
CreateSplitterFinderStreamProcessor(int num_threads) {
  if (num_threads <= 1) {
    return nullptr;
  }
  LOG(INFO) << "Create processor with " << num_threads
            << " threads for split computation";
  auto find_condition =
      [](SplitterWorkRequest request) -> absl::StatusOr<SplitterWorkResponse> {
    const auto& common = *(request.common);
    if (common.dt_config.internal().generate_fake_error_in_splitter()) {
      return absl::InternalError("Fake error");
    }
    return FindBestConditionFromSplitterWorkRequest(
        common.weights, common.config, common.config_link, common.dt_config,
        common.internal_config, request);
  };
  auto split_finder_processor = std::make_unique<SplitterFinderStreamProcessor>(
      "SplitFinder", num_threads, find_condition);
  split_finder_processor->StartWorkers();

  return split_finder_processor;
}

absl::Status DecisionTreeCoreTrain(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const std::vector<float>& weights, utils::RandomEngine* random,
    const InternalTrainConfig& internal_config, DecisionTree* dt,
    absl::Span<UnsignedExampleIdx> selected_examples,
    std::optional<absl::Span<UnsignedExampleIdx>> leaf_examples) {
  dt->CreateRoot();
  PerThreadCache cache;

  auto selected_examples_rb = SelectedExamplesRollingBuffer::Create(
      selected_examples, &cache.selected_example_buffer);
  std::optional<SelectedExamplesRollingBuffer> leaf_examples_rb;
  if (leaf_examples.has_value()) {
    leaf_examples_rb = SelectedExamplesRollingBuffer::Create(
        leaf_examples.value(), &cache.leaf_example_buffer);
  }

  switch (dt_config.growing_strategy_case()) {
    case proto::DecisionTreeTrainingConfig::kGrowingStrategyLocal: {
      const auto constraints = NodeConstraints::CreateNodeConstraints();
      return NodeTrain(train_dataset, config, config_link, dt_config,
                       deployment, weights, 1, internal_config, constraints,
                       false, dt->mutable_root(), random, &cache,
                       selected_examples_rb, leaf_examples_rb);
    } break;
    case proto::DecisionTreeTrainingConfig::kGrowingStrategyBestFirstGlobal:
      return GrowTreeBestFirstGlobal(
          train_dataset, config, config_link, dt_config, deployment, weights,
          internal_config, dt->mutable_root(), random, selected_examples_rb,
          leaf_examples_rb);
      break;
    default:
      return absl::InvalidArgumentError("Grow strategy not set");
  }
}

absl::Status NodeTrain(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const std::vector<float>& weights, const int32_t depth,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, bool set_leaf_already_set,
    NodeWithChildren* node, utils::RandomEngine* random, PerThreadCache* cache,
    SelectedExamplesRollingBuffer selected_examples,
    std::optional<SelectedExamplesRollingBuffer> leaf_examples) {
  if (selected_examples.empty()) {
    return absl::InternalError("No examples fed to the node trainer");
  }
  node->mutable_node()->set_num_pos_training_examples_without_weight(
      selected_examples.size());

  if (!set_leaf_already_set) {
    // Set the node value (i.e. the label distribution).
    RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
        train_dataset, selected_examples.active, weights, config, config_link,
        node));
    RETURN_IF_ERROR(ApplyConstraintOnNode(constraints, node));
  }

  if (selected_examples.size() < dt_config.min_examples() ||
      (dt_config.max_depth() >= 0 && depth >= dt_config.max_depth()) ||
      (internal_config.timeout.has_value() &&
       internal_config.timeout < absl::Now())) {
    if (leaf_examples.has_value()) {
      // Override the leaf values.
      RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
          train_dataset, leaf_examples->active, weights, config, config_link,
          node));
      RETURN_IF_ERROR(ApplyConstraintOnNode(constraints, node));
    }

    // Stop the growth of the branch.
    node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
    return absl::OkStatus();
  }

  // Dataset used to train this node.
  const dataset::VerticalDataset* train_dataset_for_splitter;
  absl::Span<const UnsignedExampleIdx> selected_examples_for_splitter;
  // If true, the entire dataset "local_train_dataset" is composed of training
  // examples for this node. If false, only the subset of
  // "local_train_dataset" indexed by "selected_examples" are to be considered
  // for this node i.e. local_train_dataset[selected_examples[i]].
  bool splitter_dataset_is_compact;

  // Extract the random local imputation.
  dataset::VerticalDataset random_local_imputation_train_dataset;
  std::vector<UnsignedExampleIdx> random_local_imputation_selected_examples;
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::RANDOM_LOCAL_IMPUTATION) {
    std::vector<int> label_and_input_features(config_link.features().begin(),
                                              config_link.features().end());
    label_and_input_features.push_back(config_link.label());
    RETURN_IF_ERROR(GenerateRandomImputation(
        train_dataset, label_and_input_features, selected_examples.active,
        &random_local_imputation_train_dataset, random));
    random_local_imputation_selected_examples.resize(selected_examples.size());
    std::iota(random_local_imputation_selected_examples.begin(),
              random_local_imputation_selected_examples.end(), 0);

    train_dataset_for_splitter = &random_local_imputation_train_dataset;
    selected_examples_for_splitter =
        absl::MakeConstSpan(random_local_imputation_selected_examples);
    splitter_dataset_is_compact = true;
  } else {
    selected_examples_for_splitter =
        absl::MakeConstSpan(selected_examples.active);
    splitter_dataset_is_compact = false;
    train_dataset_for_splitter = &train_dataset;
  }

  // Determine the best split.
  if (selected_examples_for_splitter.empty()) {
    return absl::InternalError("No examples fed to the splitter");
  }

  ASSIGN_OR_RETURN(
      const auto has_better_condition,
      FindBestCondition(*train_dataset_for_splitter,
                        selected_examples_for_splitter, weights, config,
                        config_link, dt_config, node->node(), internal_config,
                        constraints, node->mutable_node()->mutable_condition(),
                        random, cache));
  if (!has_better_condition) {
    // No good condition found. Close the branch.
    node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
    return absl::OkStatus();
  }
  STATUS_CHECK_EQ(
      selected_examples.size(),
      node->node().condition().num_training_examples_without_weight());
  node->CreateChildren();
  node->FinalizeAsNonLeaf(dt_config.keep_non_leaf_label_distribution(),
                          dt_config.store_detailed_label_distribution());

  // Separate the positive and negative examples.
  ASSIGN_OR_RETURN(
      auto example_split,
      internal::SplitExamplesInPlace(
          *train_dataset_for_splitter, selected_examples,
          node->node().condition(), splitter_dataset_is_compact,
          dt_config.internal_error_on_wrong_splitter_statistics()));

  if (example_split.positive_examples.empty() ||
      example_split.negative_examples.empty()) {
    // The splitter statistics don't match exactly the condition evaluation and
    // one of the children is pure.
    node->ClearChildren();
    node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
    return absl::OkStatus();
  }

  // Separate the positive and negative examples used only to determine the node
  // value.
  std::optional<ExampleSplitRollingBuffer> node_only_example_split;
  if (leaf_examples.has_value()) {
    ASSIGN_OR_RETURN(
        node_only_example_split,
        internal::SplitExamplesInPlace(
            train_dataset, *leaf_examples, node->node().condition(), false,
            dt_config.internal_error_on_wrong_splitter_statistics(),
            /*examples_are_training_examples=*/false));
  }

  // Set leaf outputs
  RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
      train_dataset, example_split.positive_examples.active, weights, config,
      config_link, node->mutable_pos_child()));
  RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
      train_dataset, example_split.negative_examples.active, weights, config,
      config_link, node->mutable_neg_child()));
  RETURN_IF_ERROR(
      ApplyConstraintOnNode(constraints, node->mutable_pos_child()));
  RETURN_IF_ERROR(
      ApplyConstraintOnNode(constraints, node->mutable_neg_child()));

  // Children constraints
  auto pos_constraints = constraints;
  auto neg_constraints = constraints;
  const int monotonic_constraint_sign = MonotonicConstraintSign(
      config_link, node->node().condition().attribute());
  if (monotonic_constraint_sign != 0) {
    RETURN_IF_ERROR(DivideMonotonicConstraintToChildren(
        constraints, monotonic_constraint_sign == 1,
        dt_config.internal().check_monotonic_constraints(), node,
        node->mutable_pos_child(), node->mutable_neg_child(), &pos_constraints,
        &neg_constraints));
  }

  // Positive child.
  RETURN_IF_ERROR(NodeTrain(
      train_dataset, config, config_link, dt_config, deployment, weights,
      depth + 1, internal_config, pos_constraints, true,
      node->mutable_pos_child(), random, cache, example_split.positive_examples,
      node_only_example_split.has_value()
          ? std::optional<SelectedExamplesRollingBuffer>(
                node_only_example_split->positive_examples)
          : std::nullopt));

  // Negative child.
  RETURN_IF_ERROR(NodeTrain(
      train_dataset, config, config_link, dt_config, deployment, weights,
      depth + 1, internal_config, neg_constraints, true,
      node->mutable_neg_child(), random, cache, example_split.negative_examples,
      node_only_example_split.has_value()
          ? std::optional<SelectedExamplesRollingBuffer>(
                node_only_example_split->negative_examples)
          : std::nullopt));
  return absl::OkStatus();
}

absl::Status ApplyConstraintOnNode(const NodeConstraints& constraint,
                                   NodeWithChildren* node) {
  if (!constraint.min_max_output.has_value()) {
    return absl::OkStatus();
  }
  auto* reg = node->mutable_node()->mutable_regressor();
  STATUS_CHECK(reg->has_top_value());
  reg->set_top_value(std::clamp(reg->top_value(),
                                constraint.min_max_output.value().min,
                                constraint.min_max_output.value().max));
  return absl::OkStatus();
}

absl::Status DivideMonotonicConstraintToChildren(
    const NodeConstraints& constraint, bool direction_increasing,
    const bool check_monotonic, NodeWithChildren* parent_node,
    NodeWithChildren* pos_node, NodeWithChildren* neg_node,
    NodeConstraints* pos_constraint, NodeConstraints* neg_constraint) {
  STATUS_CHECK(parent_node->node().regressor().has_top_value());
  STATUS_CHECK(pos_node->node().regressor().has_top_value());
  STATUS_CHECK(neg_node->node().regressor().has_top_value());

  // TODO: Experiment with other ways to select limit.
  float limit = parent_node->node().regressor().top_value();

  if (check_monotonic) {
    // A failure is indicative of an issue with the splitter i.e. the
    // "FindCondition" function call just before.

    const auto check = [direction_increasing](auto a, auto b) {
      if (direction_increasing) {
        STATUS_CHECK_GE(a, b);
      } else {
        STATUS_CHECK_LE(a, b);
      }
      return absl::OkStatus();
    };

    const float pos_value = pos_node->node().regressor().top_value();
    const float neg_value = neg_node->node().regressor().top_value();
    const float parent_value = parent_node->node().regressor().top_value();
    RETURN_IF_ERROR(check(pos_value, neg_value));
    RETURN_IF_ERROR(check(pos_value, parent_value));
    RETURN_IF_ERROR(check(parent_value, neg_value));
  }
  if ((pos_node->node().regressor().top_value() <
       neg_node->node().regressor().top_value()) == direction_increasing) {
    const float center = (pos_node->node().regressor().top_value() +
                          neg_node->node().regressor().top_value()) /
                         2;
    pos_node->mutable_node()->mutable_regressor()->set_top_value(center);
    neg_node->mutable_node()->mutable_regressor()->set_top_value(center);
    limit = center;
  }

  if (!pos_constraint->min_max_output.has_value()) {
    pos_constraint->min_max_output = NodeConstraints::MinMax();
  }
  if (!neg_constraint->min_max_output.has_value()) {
    neg_constraint->min_max_output = NodeConstraints::MinMax();
  }

  if (direction_increasing) {
    pos_constraint->min_max_output.value().min = limit;
    neg_constraint->min_max_output.value().max = limit;
  } else {
    pos_constraint->min_max_output.value().max = limit;
    neg_constraint->min_max_output.value().min = limit;
  }

  return absl::OkStatus();
}

int8_t MonotonicConstraintSign(
    const model::proto::TrainingConfigLinking& config_link,
    const int attribute_idx) {
  if (config_link.per_columns_size() == 0) {
    return 0;
  }
  const auto& link_condition_attribute = config_link.per_columns(attribute_idx);
  if (link_condition_attribute.has_monotonic_constraint()) {
    const bool direction_increasing =
        link_condition_attribute.monotonic_constraint().direction() ==
        model::proto::MonotonicConstraint::INCREASING;
    return direction_increasing ? +1 : -1;
  }
  return 0;
}

namespace internal {

bool MaskPureSampledOrPrunedItemsForCategoricalSetGreedySelection(
    const proto::DecisionTreeTrainingConfig& dt_config,
    int32_t num_attribute_classes,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<int64_t>&
        count_examples_without_weights_by_attribute_class,
    std::vector<bool>* candidate_attributes_bitmap,
    utils::RandomEngine* random) {
  std::uniform_real_distribution<float> sampling_dist;
  int64_t valid_items = 0;
  for (int attr_value = 0; attr_value < num_attribute_classes; attr_value++) {
    if (dt_config.categorical_set_greedy_forward().max_num_items() >= 0 &&
        attr_value >=
            dt_config.categorical_set_greedy_forward().max_num_items()) {
      // Too much candidate items.
      (*candidate_attributes_bitmap)[attr_value] = false;
    } else if (dt_config.categorical_set_greedy_forward().sampling() < 1.f &&
               sampling_dist(*random) >
                   dt_config.categorical_set_greedy_forward().sampling()) {
      // Randomly masked item.
      (*candidate_attributes_bitmap)[attr_value] = false;
    } else if (count_examples_without_weights_by_attribute_class[attr_value] <
                   dt_config.categorical_set_greedy_forward()
                       .min_item_frequency() ||
               count_examples_without_weights_by_attribute_class[attr_value] >
                   selected_examples.size() -
                       dt_config.categorical_set_greedy_forward()
                           .min_item_frequency()) {
      // Pure item.
      (*candidate_attributes_bitmap)[attr_value] = false;
    } else {
      valid_items++;
    }
  }
  return valid_items > 0;
}

absl::StatusOr<std::vector<float>> GenHistogramBins(
    const proto::NumericalSplit::Type type, const int num_splits,
    const absl::Span<const float> attributes, const float min_value,
    const float max_value, utils::RandomEngine* random) {
  STATUS_CHECK_GE(num_splits, 0);
  std::vector<float> candidate_splits(num_splits);
  switch (type) {
    case proto::NumericalSplit::HISTOGRAM_RANDOM: {
      std::uniform_real_distribution<float> threshold_distribution(min_value,
                                                                   max_value);
      for (auto& candidate_split : candidate_splits) {
        candidate_split = threshold_distribution(*random);
      }
    } break;
    case proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH: {
      for (int split_idx = 0; split_idx < candidate_splits.size();
           split_idx++) {
        candidate_splits[split_idx] = min_value + (max_value - min_value) *
                                                      (split_idx + 0.5f) /
                                                      candidate_splits.size();
      }
    } break;
    default:
      return absl::InvalidArgumentError("Numerical histogram not implemented");
  }
  std::sort(candidate_splits.begin(), candidate_splits.end());
  return candidate_splits;
}

absl::StatusOr<ExampleSplitRollingBuffer> SplitExamplesInPlace(
    const dataset::VerticalDataset& dataset,
    const SelectedExamplesRollingBuffer examples,
    const proto::NodeCondition& condition, const bool dataset_is_dense,
    const bool error_on_wrong_splitter_statistics,
    const bool examples_are_training_examples) {
  DCHECK(std::is_sorted(examples.active.begin(), examples.active.end()));

  ExampleSplitRollingBuffer example_split;
  RETURN_IF_ERROR(EvalConditionOnDataset(dataset, examples, condition,
                                         dataset_is_dense, &example_split));

  DCHECK(std::is_sorted(example_split.positive_examples.active.begin(),
                        example_split.positive_examples.active.end()));
  DCHECK(std::is_sorted(example_split.negative_examples.active.begin(),
                        example_split.negative_examples.active.end()));

  // The following test ensure that the effective number of positive examples is
  // equal to the expected number of positive examples. A miss alignment
  // generally indicates that the splitter does not work correctly.
  //
  // Incorrectly working splitters can make the model worst than expected if
  // the error happens often. If such error happen rarely, the impact is likely
  // insignificant.
  //
  // This generates an error in unit testing and a warning otherwise.
  if (examples_are_training_examples &&
      ABSL_PREDICT_FALSE(
          (example_split.num_positive() !=
           condition.num_pos_training_examples_without_weight()) ||
          (example_split.num_negative() !=
           examples.size() -
               condition.num_pos_training_examples_without_weight()))) {
    const std::string message = absl::Substitute(
        "[This message is only shown once] The effective split of examples "
        "does not match the expected split "
        "returned by the splitter algorithm. This problem can be caused by (1) "
        "large floating point values (e.g. value>=10e30) or (2) a bug in the "
        "software. You can turn this error in a warning with "
        "internal_error_on_wrong_splitter_statistics=false.\n\nDetails:\n"
        "Num examples: $0\n"
        "Num positive examples (from split evaluation): $1\n"
        "Num positive examples (from split learning): $4\n"
        "Num negative examples (from split evaluation): $2\n"
        "\n"
        "Condition: $3\n"
        "Attribute spec: $5",
        /*$0*/ examples.size(),
        /*$1*/ example_split.num_positive(),
        /*$2*/ example_split.num_negative(),
        /*$3*/ condition.DebugString(),
        /*$4*/ condition.num_pos_training_examples_without_weight(),
        /*$5*/
        dataset.data_spec().columns(condition.attribute()).DebugString());
    if (error_on_wrong_splitter_statistics) {
      return absl::InternalError(message);
    } else {
      // Logging this message too often will crash.
      LOG_FIRST_N(WARNING, 1) << message;
    }
  }
  return example_split;
}

}  // namespace internal

}  // namespace yggdrasil_decision_forests::model::decision_tree
