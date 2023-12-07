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
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_accumulator.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_accumulator_uplift.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_scanner.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/cast.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

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

// Set the label value for a classification label on a vertical dataset.
template <bool weighted>
absl::Status SetClassificationLabelDistribution(
    const dataset::VerticalDataset& dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfigLinking& config_link, proto::Node* node) {
  if constexpr (weighted) {
    DCHECK_LE(selected_examples.size(), weights.size());
  } else {
    DCHECK(weights.empty());
  }
  ASSIGN_OR_RETURN(
      const auto* const labels,
      dataset.ColumnWithCastWithStatus<
          dataset::VerticalDataset::CategoricalColumn>(config_link.label()));
  utils::IntegerDistributionDouble label_distribution;
  const int32_t num_classes = dataset.data_spec()
                                  .columns(config_link.label())
                                  .categorical()
                                  .number_of_unique_values();
  label_distribution.SetNumClasses(num_classes);
  for (const UnsignedExampleIdx example_idx : selected_examples) {
    if constexpr (weighted) {
      label_distribution.Add(labels->values()[example_idx],
                             weights[example_idx]);
    } else {
      label_distribution.Add(labels->values()[example_idx]);
    }
  }
  label_distribution.Save(node->mutable_classifier()->mutable_distribution());
  node->mutable_classifier()->set_top_value(label_distribution.TopClass());
  return absl::OkStatus();
}

absl::Status SetCategoricalUpliftLabelDistribution(
    const dataset::VerticalDataset& dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfigLinking& config_link, proto::Node* node) {
  DCHECK(!weights.empty());
  // TODO: Update.
  ASSIGN_OR_RETURN(
      const auto* const outcomes,
      dataset.ColumnWithCastWithStatus<
          dataset::VerticalDataset::CategoricalColumn>(config_link.label()));

  ASSIGN_OR_RETURN(const auto* const treatments,
                   dataset.ColumnWithCastWithStatus<
                       dataset::VerticalDataset::CategoricalColumn>(
                       config_link.uplift_treatment()));

  const auto& outcome_spec = dataset.data_spec().columns(config_link.label());
  const auto& treatment_spec =
      dataset.data_spec().columns(config_link.uplift_treatment());

  UpliftLabelDistribution label_dist;
  label_dist.InitializeAndClearCategoricalOutcome(
      outcome_spec.categorical().number_of_unique_values(),
      treatment_spec.categorical().number_of_unique_values());

  for (const UnsignedExampleIdx example_idx : selected_examples) {
    label_dist.AddCategoricalOutcome(outcomes->values()[example_idx],
                                     treatments->values()[example_idx],
                                     weights[example_idx]);
  }
  internal::UpliftLabelDistToLeaf(label_dist, node->mutable_uplift());
  return absl::OkStatus();
}

absl::Status SetRegressiveUpliftLabelDistribution(
    const dataset::VerticalDataset& dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfigLinking& config_link, proto::Node* node) {
  const auto* const outcomes =
      dataset.ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
          config_link.label());

  const auto* const treatments =
      dataset.ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
          config_link.uplift_treatment());

  const auto& treatment_spec =
      dataset.data_spec().columns(config_link.uplift_treatment());

  UpliftLabelDistribution label_dist;
  label_dist.InitializeAndClearNumericalOutcome(
      treatment_spec.categorical().number_of_unique_values());

  for (const UnsignedExampleIdx example_idx : selected_examples) {
    label_dist.AddNumericalOutcome(outcomes->values()[example_idx],
                                   treatments->values()[example_idx],
                                   weights[example_idx]);
  }
  internal::UpliftLabelDistToLeaf(label_dist, node->mutable_uplift());
  return absl::OkStatus();
}

// Compute the ratio of true label for all attribute values.
void ComputeTrueLabelValuePerAttributeValue(
    const utils::IntegersConfusionMatrixDouble& confusion,
    const int32_t true_label_value,
    std::vector<std::pair<float, int32_t>>* ratio_true_label_by_attr_value) {
  ratio_true_label_by_attr_value->resize(confusion.ncol());
  for (int32_t attribute_value = 0; attribute_value < confusion.ncol();
       attribute_value++) {
    const float count_true_label =
        confusion.at(true_label_value, attribute_value);
    float count_all_label = 0;
    for (int32_t label_value = 0; label_value < confusion.nrow();
         label_value++) {
      count_all_label += confusion.at(label_value, attribute_value);
    }
    const float ratio_true_label =
        (count_all_label > 0) ? (count_true_label / count_all_label) : 0;
    (*ratio_true_label_by_attr_value)[attribute_value].first = ratio_true_label;
    (*ratio_true_label_by_attr_value)[attribute_value].second = attribute_value;
  }
  // Order the attribute values in increasing order of true label ratio.
  std::sort(ratio_true_label_by_attr_value->begin(),
            ratio_true_label_by_attr_value->end());
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
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
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
    const std::vector<UnsignedExampleIdx>& selected_examples,
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
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<char>& attributes,
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
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& attributes, float* min_value, float* max_value) {
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
    const std::vector<UnsignedExampleIdx>& selected_examples,
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

// Predicts if the sorting-in-node numerical splitter is slower or faster than
// the numerical splitter using a pre-sorted index. Returns true if learning a
// split on "num_selected_examples" while the entire training dataset contains
// "num_total_examples" is more efficient with a pre-sorted splitter. If false,
// a splitter with in-node sorting is more efficient.
//
// This function is an heuristic that impact the training speed but not the
// model.
//
// This function is used when PRESORTING is enabled (see
// "SetDefaultHyperParameters").
bool IsPresortingOnNumericalSplitMoreEfficient(
    const int64_t num_selected_examples, const int64_t num_total_examples) {
  const float ratio =
      static_cast<float>(num_selected_examples) / num_total_examples;
  return num_selected_examples >= 25 && ratio >= 0.125;
}

}  // namespace

absl::Status SetLabelDistribution(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    NodeWithChildren* node) {
  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION:
      if (weights.empty()) {
        RETURN_IF_ERROR(SetClassificationLabelDistribution</*weighted=*/false>(
            train_dataset, selected_examples, weights, config_link,
            node->mutable_node()));
      } else {
        RETURN_IF_ERROR(SetClassificationLabelDistribution</*weighted=*/true>(
            train_dataset, selected_examples, weights, config_link,
            node->mutable_node()));
      }
      break;
    case model::proto::Task::REGRESSION:
      if (weights.empty()) {
        RETURN_IF_ERROR(SetRegressionLabelDistribution</*weighted=*/false>(
            train_dataset, selected_examples, weights, config_link,
            node->mutable_node()));
        break;
      } else {
        RETURN_IF_ERROR(SetRegressionLabelDistribution</*weighted=*/true>(
            train_dataset, selected_examples, weights, config_link,
            node->mutable_node()));
        break;
      }

    case model::proto::Task::CATEGORICAL_UPLIFT:
      RETURN_IF_ERROR(SetCategoricalUpliftLabelDistribution(
          train_dataset, selected_examples, weights, config_link,
          node->mutable_node()));
      break;

    case model::proto::Task::NUMERICAL_UPLIFT:
      RETURN_IF_ERROR(SetRegressiveUpliftLabelDistribution(
          train_dataset, selected_examples, weights, config_link,
          node->mutable_node()));
      break;

    default:
      NOTREACHED();
  }
  return absl::OkStatus();
}

// Specialization in the case of classification.
SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const ClassificationLabelStats& label_stats, const int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  CHECK_OK(FailIfMonotonic(config_link, attribute_idx, constraints,
                           "classification"));

  SplitSearchResult result;

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::NumericalColumn>(attribute_idx)
              .value()
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();
      if (dt_config.numerical_split().type() == proto::NumericalSplit::EXACT) {
        result = FindSplitLabelClassificationFeatureNumericalCart(
            selected_examples, weights, attribute_data, label_stats.label_data,
            label_stats.num_label_classes, na_replacement, min_num_obs,
            dt_config, label_stats.label_distribution, attribute_idx,
            internal_config, best_condition, cache);
      } else {
        result = FindSplitLabelClassificationFeatureNumericalHistogram(
            selected_examples, weights, attribute_data, label_stats.label_data,
            label_stats.num_label_classes, na_replacement, min_num_obs,
            dt_config, label_stats.label_distribution, attribute_idx, random,
            best_condition);
      }
    } break;

    case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

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
      result = FindSplitLabelClassificationFeatureDiscretizedNumericalCart(
          selected_examples, weights, attribute_data, num_bins,
          label_stats.label_data, label_stats.num_label_classes,
          na_replacement_index, min_num_obs, dt_config,
          label_stats.label_distribution, attribute_idx, best_condition, cache);
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
      result = FindSplitLabelClassificationFeatureCategorical(
          selected_examples, weights, attribute_data, label_stats.label_data,
          num_attribute_classes, label_stats.num_label_classes, na_replacement,
          min_num_obs, dt_config, label_stats.label_distribution, attribute_idx,
          random, best_condition, cache);
    } break;

    case dataset::proto::ColumnType::CATEGORICAL_SET: {
      const auto* attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::CategoricalSetColumn>(attribute_idx)
              .value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      result = FindSplitLabelClassificationFeatureCategoricalSetGreedyForward(
          selected_examples, weights, *attribute_data, label_stats.label_data,
          num_attribute_classes, label_stats.num_label_classes, min_num_obs,
          dt_config, label_stats.label_distribution, attribute_idx,
          best_condition, random);
    } break;

    case dataset::proto::ColumnType::BOOLEAN: {
      // Condition of the type "Attr is True".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::BooleanColumn>(attribute_idx)
              .value()
              ->values();
      const auto na_replacement =
          attribute_column_spec.boolean().count_true() >=
          attribute_column_spec.boolean().count_false();
      result = FindSplitLabelClassificationFeatureBoolean(
          selected_examples, weights, attribute_data, label_stats.label_data,
          label_stats.num_label_classes, na_replacement, min_num_obs, dt_config,
          label_stats.label_distribution, attribute_idx, best_condition, cache);
    } break;

    default:
      YDF_LOG(FATAL) << dataset::proto::ColumnType_Name(
                            train_dataset.column(attribute_idx)->type())
                     << " attribute "
                     << train_dataset.column(attribute_idx)->name()
                     << " is not supported.";
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    const auto na_result = FindSplitLabelClassificationFeatureNA(
        selected_examples, weights, train_dataset.column(attribute_idx),
        label_stats.label_data, label_stats.num_label_classes, min_num_obs,
        dt_config, label_stats.label_distribution, attribute_idx,
        best_condition, cache);
    result = std::min(result, na_result);
  }

  return result;
}

SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionHessianLabelStats& label_stats, const int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
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
      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::NumericalColumn>(attribute_idx)
              .value()
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();
      if (dt_config.numerical_split().type() == proto::NumericalSplit::EXACT) {
        if (weights.empty()) {
          result = FindSplitLabelHessianRegressionFeatureNumericalCart<
              /*weighted=*/false>(
              selected_examples, weights, attribute_data,
              label_stats.gradient_data, label_stats.hessian_data,
              na_replacement, min_num_obs, dt_config, label_stats.sum_gradient,
              label_stats.sum_hessian, label_stats.sum_weights, attribute_idx,
              internal_config, constraints, monotonic_direction, best_condition,
              cache);
        } else {
          result = FindSplitLabelHessianRegressionFeatureNumericalCart<
              /*weighted=*/true>(
              selected_examples, weights, attribute_data,
              label_stats.gradient_data, label_stats.hessian_data,
              na_replacement, min_num_obs, dt_config, label_stats.sum_gradient,
              label_stats.sum_hessian, label_stats.sum_weights, attribute_idx,
              internal_config, constraints, monotonic_direction, best_condition,
              cache);
        }
      } else {
        YDF_LOG(FATAL) << "Only split exact implemented for hessian gains.";
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
        result = FindSplitLabelHessianRegressionFeatureDiscretizedNumericalCart<
            /*weighted=*/false>(
            selected_examples, weights, attribute_data, num_bins,
            label_stats.gradient_data, label_stats.hessian_data,
            na_replacement_index, min_num_obs, dt_config,
            label_stats.sum_gradient, label_stats.sum_hessian,
            label_stats.sum_weights, attribute_idx, internal_config,
            constraints, monotonic_direction, best_condition, cache);
      } else {
        result = FindSplitLabelHessianRegressionFeatureDiscretizedNumericalCart<
            /*weighted=*/true>(selected_examples, weights, attribute_data,
                               num_bins, label_stats.gradient_data,
                               label_stats.hessian_data, na_replacement_index,
                               min_num_obs, dt_config, label_stats.sum_gradient,
                               label_stats.sum_hessian, label_stats.sum_weights,
                               attribute_idx, internal_config, constraints,
                               monotonic_direction, best_condition, cache);
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
        result = FindSplitLabelHessianRegressionFeatureCategorical<
            /*weighted=*/false>(
            selected_examples, weights, attribute_data,
            label_stats.gradient_data, label_stats.hessian_data,
            num_attribute_classes, na_replacement, min_num_obs, dt_config,
            label_stats.sum_gradient, label_stats.sum_hessian,
            label_stats.sum_weights, attribute_idx, internal_config,
            constraints, best_condition, cache, random);
      } else {
        result = FindSplitLabelHessianRegressionFeatureCategorical<
            /*weighted=*/true>(
            selected_examples, weights, attribute_data,
            label_stats.gradient_data, label_stats.hessian_data,
            num_attribute_classes, na_replacement, min_num_obs, dt_config,
            label_stats.sum_gradient, label_stats.sum_hessian,
            label_stats.sum_weights, attribute_idx, internal_config,
            constraints, best_condition, cache, random);
      }
    } break;

    case dataset::proto::ColumnType::BOOLEAN: {
      // Condition of the type "Attr is True".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::BooleanColumn>(attribute_idx)
              .value()
              ->values();
      const auto na_replacement =
          attribute_column_spec.boolean().count_true() >=
          attribute_column_spec.boolean().count_false();
      if (weights.empty()) {
        result =
            FindSplitLabelHessianRegressionFeatureBoolean</*weighted=*/false>(
                selected_examples, weights, attribute_data,
                label_stats.gradient_data, label_stats.hessian_data,
                na_replacement, min_num_obs, dt_config,
                label_stats.sum_gradient, label_stats.sum_hessian,
                label_stats.sum_weights, attribute_idx, internal_config,
                constraints, best_condition, cache);
      } else {
        result =
            FindSplitLabelHessianRegressionFeatureBoolean</*weighted=*/true>(
                selected_examples, weights, attribute_data,
                label_stats.gradient_data, label_stats.hessian_data,
                na_replacement, min_num_obs, dt_config,
                label_stats.sum_gradient, label_stats.sum_hessian,
                label_stats.sum_weights, attribute_idx, internal_config,
                constraints, best_condition, cache);
      }
    } break;

    default:
      YDF_LOG(FATAL) << dataset::proto::ColumnType_Name(
                            train_dataset.column(attribute_idx)->type())
                     << " attribute "
                     << train_dataset.column(attribute_idx)->name()
                     << " is not supported.";
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    if (weights.empty()) {
      const auto na_result =
          FindSplitLabelHessianRegressionFeatureNA</*weighted=*/false>(
              selected_examples, weights, train_dataset.column(attribute_idx),
              label_stats.gradient_data, label_stats.hessian_data, min_num_obs,
              dt_config, label_stats.sum_gradient, label_stats.sum_hessian,
              label_stats.sum_weights, attribute_idx, internal_config,
              constraints, best_condition, cache);
      result = std::min(result, na_result);
    } else {
      const auto na_result =
          FindSplitLabelHessianRegressionFeatureNA</*weighted=*/true>(
              selected_examples, weights, train_dataset.column(attribute_idx),
              label_stats.gradient_data, label_stats.hessian_data, min_num_obs,
              dt_config, label_stats.sum_gradient, label_stats.sum_hessian,
              label_stats.sum_weights, attribute_idx, internal_config,
              constraints, best_condition, cache);
      result = std::min(result, na_result);
    }
  }

  return result;
}

// Specialization in the case of regression.
SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionLabelStats& label_stats, const int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  SplitSearchResult result;

  CHECK_OK(
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
          result =
              FindSplitLabelRegressionFeatureNumericalCart</*weighted=*/false>(
                  selected_examples, weights, attribute_data,
                  label_stats.label_data, na_replacement, min_num_obs,
                  dt_config, label_stats.label_distribution, attribute_idx,
                  internal_config, best_condition, cache);
        } else {
          result =
              FindSplitLabelRegressionFeatureNumericalCart</*weighted=*/true>(
                  selected_examples, weights, attribute_data,
                  label_stats.label_data, na_replacement, min_num_obs,
                  dt_config, label_stats.label_distribution, attribute_idx,
                  internal_config, best_condition, cache);
        }
      } else {
        if (weights.empty()) {
          result = FindSplitLabelRegressionFeatureNumericalHistogram<
              /*weighted=*/false>(selected_examples, weights, attribute_data,
                                  label_stats.label_data, na_replacement,
                                  min_num_obs, dt_config,
                                  label_stats.label_distribution, attribute_idx,
                                  random, best_condition);
        } else {
          result = FindSplitLabelRegressionFeatureNumericalHistogram<
              /*weighted=*/true>(selected_examples, weights, attribute_data,
                                 label_stats.label_data, na_replacement,
                                 min_num_obs, dt_config,
                                 label_stats.label_distribution, attribute_idx,
                                 random, best_condition);
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
        result = FindSplitLabelRegressionFeatureDiscretizedNumericalCart<
            /*weighted=*/false>(selected_examples, weights, attribute_data,
                                num_bins, label_stats.label_data,
                                na_replacement_index, min_num_obs, dt_config,
                                label_stats.label_distribution, attribute_idx,
                                best_condition, cache);
      } else {
        result = FindSplitLabelRegressionFeatureDiscretizedNumericalCart<
            /*weighted=*/true>(selected_examples, weights, attribute_data,
                               num_bins, label_stats.label_data,
                               na_replacement_index, min_num_obs, dt_config,
                               label_stats.label_distribution, attribute_idx,
                               best_condition, cache);
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
        result = FindSplitLabelRegressionFeatureCategorical</*weighted=*/false>(
            selected_examples, weights, attribute_data, label_stats.label_data,
            num_attribute_classes, na_replacement, min_num_obs, dt_config,
            label_stats.label_distribution, attribute_idx, best_condition,
            cache, random);
      } else {
        result = FindSplitLabelRegressionFeatureCategorical</*weighted=*/true>(
            selected_examples, weights, attribute_data, label_stats.label_data,
            num_attribute_classes, na_replacement, min_num_obs, dt_config,
            label_stats.label_distribution, attribute_idx, best_condition,
            cache, random);
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
        result = FindSplitLabelRegressionFeatureCategoricalSetGreedyForward<
            /*weighted=*/false>(selected_examples, weights, *attribute_data,
                                label_stats.label_data, num_attribute_classes,
                                min_num_obs, dt_config,
                                label_stats.label_distribution, attribute_idx,
                                best_condition, random);
      } else {
        result = FindSplitLabelRegressionFeatureCategoricalSetGreedyForward<
            /*weighted=*/true>(selected_examples, weights, *attribute_data,
                               label_stats.label_data, num_attribute_classes,
                               min_num_obs, dt_config,
                               label_stats.label_distribution, attribute_idx,
                               best_condition, random);
      }
    } break;

    case dataset::proto::ColumnType::BOOLEAN: {
      // Condition of the type "Attr is True".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::BooleanColumn>(attribute_idx)
              .value()
              ->values();
      const auto na_replacement =
          attribute_column_spec.boolean().count_true() >=
          attribute_column_spec.boolean().count_false();
      if (weights.empty()) {
        result = FindSplitLabelRegressionFeatureBoolean</*weighted=*/false>(
            selected_examples, weights, attribute_data, label_stats.label_data,
            na_replacement, min_num_obs, dt_config,
            label_stats.label_distribution, attribute_idx, best_condition,
            cache);
      } else {
        result = FindSplitLabelRegressionFeatureBoolean</*weighted=*/true>(
            selected_examples, weights, attribute_data, label_stats.label_data,
            na_replacement, min_num_obs, dt_config,
            label_stats.label_distribution, attribute_idx, best_condition,
            cache);
      }
    } break;

    default:
      YDF_LOG(FATAL) << dataset::proto::ColumnType_Name(
                            train_dataset.column(attribute_idx)->type())
                     << " attribute "
                     << train_dataset.column(attribute_idx)->name()
                     << " is not supported.";
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    if (weights.empty()) {
      const auto na_result =
          FindSplitLabelRegressionFeatureNA</*weighted=*/false>(
              selected_examples, weights, train_dataset.column(attribute_idx),
              label_stats.label_data, min_num_obs, dt_config,
              label_stats.label_distribution, attribute_idx, best_condition,
              cache);
      result = std::min(result, na_result);
    } else {
      const auto na_result =
          FindSplitLabelRegressionFeatureNA</*weighted=*/true>(
              selected_examples, weights, train_dataset.column(attribute_idx),
              label_stats.label_data, min_num_obs, dt_config,
              label_stats.label_distribution, attribute_idx, best_condition,
              cache);
      result = std::min(result, na_result);
    }
  }

  return result;
}

// Specialization in the case of uplift with categorical outcome.
SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
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

  CHECK_OK(FailIfMonotonic(config_link, attribute_idx, constraints,
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

      result = FindSplitLabelUpliftCategoricalFeatureNumericalCart(
          selected_examples, weights, attribute_data, label_stats,
          na_replacement, min_num_obs, dt_config, attribute_idx,
          internal_config, best_condition, cache);
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

      result = FindSplitLabelUpliftCategoricalFeatureCategorical(
          selected_examples, weights, attribute_data, label_stats,
          num_attribute_classes, na_replacement, min_num_obs, dt_config,
          attribute_idx, internal_config, best_condition, cache, random);
    } break;

    default:
      YDF_LOG(FATAL) << dataset::proto::ColumnType_Name(
                            train_dataset.column(attribute_idx)->type())
                     << " attribute "
                     << train_dataset.column(attribute_idx)->name()
                     << " is not supported.";
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    YDF_LOG(FATAL) << "allow_na_conditions not supported";
  }

  return result;
}

// Specialization in the case of uplift with numerical outcome.
SplitSearchResult FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
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

  CHECK_OK(FailIfMonotonic(config_link, attribute_idx, constraints,
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

      result = FindSplitLabelUpliftNumericalFeatureNumericalCart(
          selected_examples, weights, attribute_data, label_stats,
          na_replacement, min_num_obs, dt_config, attribute_idx,
          internal_config, best_condition, cache);
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

      result = FindSplitLabelUpliftNumericalFeatureCategorical(
          selected_examples, weights, attribute_data, label_stats,
          num_attribute_classes, na_replacement, min_num_obs, dt_config,
          attribute_idx, internal_config, best_condition, cache, random);
    } break;

    default:
      YDF_LOG(FATAL) << dataset::proto::ColumnType_Name(
                            train_dataset.column(attribute_idx)->type())
                     << " attribute "
                     << train_dataset.column(attribute_idx)->name()
                     << " is not supported.";
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    YDF_LOG(FATAL) << "allow_na_conditions not supported";
  }

  return result;
}

SplitterWorkResponse FindBestConditionFromSplitterWorkRequest(
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const InternalTrainConfig& internal_config,
    const SplitterWorkRequest& request) {
  SplitterWorkResponse response;
  response.manager_data = request.manager_data;
  request.splitter_cache->random.seed(request.seed);

  if (request.num_oblique_projections_to_run.has_value()) {
    DCHECK_EQ(request.attribute_idx, -1);
    const auto found_oblique_condition =
        FindBestConditionOblique(
            request.common->train_dataset, request.common->selected_examples,
            weights, config, config_link, dt_config, request.common->parent,
            internal_config, request.common->label_stats,
            request.num_oblique_projections_to_run.value(),
            request.common->constraints, request.condition,
            &request.splitter_cache->random, request.splitter_cache)
            .value();

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

      response.status = FindBestCondition(
          request.common->train_dataset, request.common->selected_examples,
          weights, config, config_link, dt_config, request.common->parent,
          internal_config, label_stats, request.attribute_idx,
          request.common->constraints, request.condition,
          &request.splitter_cache->random, request.splitter_cache);
    } break;
    case model::proto::Task::REGRESSION:
      if (internal_config.hessian_score) {
        const auto& label_stats =
            utils::down_cast<const RegressionHessianLabelStats&>(
                request.common->label_stats);

        response.status = FindBestCondition(
            request.common->train_dataset, request.common->selected_examples,
            weights, config, config_link, dt_config, request.common->parent,
            internal_config, label_stats, request.attribute_idx,
            request.common->constraints, request.condition,
            &request.splitter_cache->random, request.splitter_cache);

      } else {
        const auto& label_stats = utils::down_cast<const RegressionLabelStats&>(
            request.common->label_stats);

        response.status = FindBestCondition(
            request.common->train_dataset, request.common->selected_examples,
            weights, config, config_link, dt_config, request.common->parent,
            internal_config, label_stats, request.attribute_idx,
            request.common->constraints, request.condition,
            &request.splitter_cache->random, request.splitter_cache);
      }
      break;
    default:
      NOTREACHED();
  }

  return response;
}

absl::StatusOr<bool> FindBestConditionOblique(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats,
    const absl::optional<int>& override_num_projections,
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
    const std::vector<UnsignedExampleIdx>& selected_examples,
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

        result = FindBestCondition(train_dataset, selected_examples, weights,
                                   config, config_link, dt_config, parent,
                                   internal_config, class_label_stats,
                                   attribute_idx, constraints, best_condition,
                                   random, &cache->splitter_cache_list[0]);
      } break;
      case model::proto::Task::REGRESSION:
        if (internal_config.hessian_score) {
          const auto& reg_label_stats =
              utils::down_cast<const RegressionHessianLabelStats&>(label_stats);

          result = FindBestCondition(train_dataset, selected_examples, weights,
                                     config, config_link, dt_config, parent,
                                     internal_config, reg_label_stats,
                                     attribute_idx, constraints, best_condition,
                                     random, &cache->splitter_cache_list[0]);

        } else {
          const auto& reg_label_stats =
              utils::down_cast<const RegressionLabelStats&>(label_stats);

          result = FindBestCondition(train_dataset, selected_examples, weights,
                                     config, config_link, dt_config, parent,
                                     internal_config, reg_label_stats,
                                     attribute_idx, constraints, best_condition,
                                     random, &cache->splitter_cache_list[0]);
        }
        break;

      case model::proto::Task::CATEGORICAL_UPLIFT: {
        const auto& uplift_label_stats =
            utils::down_cast<const CategoricalUpliftLabelStats&>(label_stats);
        result = FindBestCondition(train_dataset, selected_examples, weights,
                                   config, config_link, dt_config, parent,
                                   internal_config, uplift_label_stats,
                                   attribute_idx, constraints, best_condition,
                                   random, &cache->splitter_cache_list[0]);
      } break;

      case model::proto::Task::NUMERICAL_UPLIFT: {
        const auto& uplift_label_stats =
            utils::down_cast<const NumericalUpliftLabelStats&>(label_stats);
        result = FindBestCondition(train_dataset, selected_examples, weights,
                                   config, config_link, dt_config, parent,
                                   internal_config, uplift_label_stats,
                                   attribute_idx, constraints, best_condition,
                                   random, &cache->splitter_cache_list[0]);
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
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
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

  int num_threads = splitter_concurrency_setup.num_threads;
  int num_features = config_link.features().size();

  if (num_features == 0) {
    return false;
  }

  SplitterWorkRequestCommon common{
      .train_dataset = train_dataset,
      .selected_examples = selected_examples,
      .parent = parent,
      .label_stats = label_stats,
      .constraints = constraints,
  };

  // Computes the number of oblique projections to evaluate and how to group
  // them into requests.
  bool oblique = false;
  int num_oblique_jobs = 0;
  int num_oblique_projections = 0;
  if (dt_config.split_axis_case() ==
      proto::DecisionTreeTrainingConfig::kSparseObliqueSplit) {
    num_oblique_projections =
        GetNumProjections(dt_config, config_link.numerical_features_size());

    // Arbitrary minimum number of oblique projections to test in each job.
    // Because oblique jobs are expensive (more than non oblique jobs), it is
    // not efficient to create a request with too little work to do.
    //
    // In most real cases, this parameter does not matter as the limit is
    // effectively constraint by the number of threads.
    const int min_projections_per_request = 10;

    DCHECK_GE(num_threads, 1);
    num_oblique_jobs = std::min(num_threads, (num_oblique_projections +
                                              min_projections_per_request - 1) /
                                                 min_projections_per_request);
    oblique = num_oblique_projections > 0;
  } else if (config_link.numerical_features_size() > 0 &&
             dt_config.split_axis_case() ==
                 proto::DecisionTreeTrainingConfig::kMhldObliqueSplit) {
    num_oblique_projections = 1;
    num_oblique_jobs = 1;
    oblique = true;
  }

  // Prepare caches.
  cache->splitter_cache_list.resize(num_threads);
  cache->condition_list.resize(num_threads * kConditionPoolGrowthFactor);
  const int num_jobs = num_features + num_oblique_jobs;
  cache->durable_response_list.resize(num_jobs);

  // Get the ordered indices of the attributes to test.
  int min_num_jobs_to_test;
  std::vector<int32_t>& candidate_attributes = cache->candidate_attributes;
  GetCandidateAttributes(config, config_link, dt_config, &min_num_jobs_to_test,
                         &candidate_attributes, random);
  // All the oblique requests need to be tested.
  min_num_jobs_to_test += num_oblique_jobs;

  // Marks all the caches and conditions as "available".
  cache->available_cache_idxs.fill_iota(cache->splitter_cache_list.size(), 0);
  cache->available_condition_idxs.fill_iota(cache->condition_list.size(), 0);

  // Marks all the duration responses as "non set".
  for (auto& s : cache->durable_response_list) {
    s.set = false;
  }

  // Get Channel readers and writers.
  auto& processor = *splitter_concurrency_setup.split_finder_processor;

  // Helper function to create a WorkRequest.
  //
  // If attribute_idx is != -1 create a request for an axis-aligned split.
  //
  // If attribute_idx is == -1 and num_oblique_projections_to_run != -1, create
  // a request for an oblique split.
  //
  auto produce =
      [&](const int job_idx, const float best_score, const int attribute_idx,
          const int num_oblique_projections_to_run) -> SplitterWorkRequest {
    // Get a cache and a condition.
    DCHECK(!cache->available_cache_idxs.empty());
    DCHECK(!cache->available_condition_idxs.empty());
    int32_t cache_idx = cache->available_cache_idxs.back();
    cache->available_cache_idxs.pop_back();
    int32_t condition_idx = cache->available_condition_idxs.back();
    DCHECK_GE(condition_idx, -1);
    cache->available_condition_idxs.pop_back();

    SplitterWorkRequest request;
    request.manager_data.condition_idx = condition_idx;
    request.manager_data.cache_idx = cache_idx;
    request.manager_data.job_idx = job_idx;
    DCHECK((attribute_idx == -1) != (num_oblique_projections_to_run == -1));
    if (attribute_idx != -1) {
      request.attribute_idx = attribute_idx;
    } else {
      request.num_oblique_projections_to_run = num_oblique_projections_to_run;
      request.attribute_idx = -1;
    }
    request.condition = &cache->condition_list[condition_idx];
    request.splitter_cache = &cache->splitter_cache_list[cache_idx];
    request.condition->set_split_score(best_score);  // Best score so far.
    request.common = &common;
    request.seed = (*random)();  // Create a new seed.

    return request;
  };

  // Schedule all the oblique jobs.
  int next_job_to_schedule = 0;
  if (oblique) {
    const int num_oblique_projections_per_job =
        (num_oblique_projections + num_oblique_jobs - 1) / num_oblique_jobs;

    for (int oblique_job_idx = 0; oblique_job_idx < num_oblique_jobs;
         oblique_job_idx++) {
      const int num_projections_in_request =
          std::min((oblique_job_idx + 1) * num_oblique_projections_per_job,
                   num_oblique_projections) -
          oblique_job_idx * num_oblique_projections_per_job;
      processor.Submit(produce(
          next_job_to_schedule++, best_condition->split_score(),
          /*attribute_idx=*/-1,
          /*num_oblique_projections_to_run=*/num_projections_in_request));
    }
  }

  // Schedule some non-oblique jobs.
  while (next_job_to_schedule < std::min(num_threads, num_jobs) &&
         !cache->available_condition_idxs.empty() &&
         !cache->available_cache_idxs.empty()) {
    const int attribute_idx =
        candidate_attributes[next_job_to_schedule - num_oblique_jobs];
    processor.Submit(produce(next_job_to_schedule++,
                             best_condition->split_score(),
                             /*attribute_idx=*/attribute_idx,
                             /*num_oblique_projections_to_run=*/-1));
  }

  int num_valid_job_tested = 0;
  int next_job_to_process = 0;

  // Index of the best condition. -1 if not better condition was found.
  int best_condition_idx = -1;
  // Score of the best found condition, or minimum condition score to look for.
  float best_split_score = best_condition->split_score();

  while (true) {
    auto maybe_response = processor.GetResult();
    if (!maybe_response.has_value()) {
      break;
    }

    {
      // Record, but do not process, the worker response.
      SplitterWorkResponse& response = maybe_response.value();

      // Release the cache immediately to be reused by other workers.
      cache->available_cache_idxs.push_front(response.manager_data.cache_idx);

      auto& durable_response =
          cache->durable_response_list[response.manager_data.job_idx];
      durable_response.status = response.status;
      durable_response.set = true;
      if (response.status == SplitSearchResult::kBetterSplitFound) {
        // The worker found a better solution compared from when the worker
        // started working.

        const float new_split_score =
            cache->condition_list[response.manager_data.condition_idx]
                .split_score();

        if ((new_split_score > best_split_score) ||
            (new_split_score == best_split_score &&
             response.manager_data.condition_idx <
                 durable_response.condition_idx)) {
          // This is the best condition so far. Keep it for processing.
          durable_response.condition_idx = response.manager_data.condition_idx;
        } else {
          // Acctually, a better condition was found by another worker and
          // processed in the mean time. No need to keep the condition.
          cache->available_condition_idxs.push_front(
              response.manager_data.condition_idx);
          durable_response.condition_idx = -1;
          durable_response.status = SplitSearchResult::kNoBetterSplitFound;
        }
      } else {
        // Return the condition to the condition pool.
        cache->available_condition_idxs.push_front(
            response.manager_data.condition_idx);
        durable_response.condition_idx = -1;
      }
    }

    // Process all the responses that can be processed.
    // Simulate a deterministic sequential processing of the responses.
    while (next_job_to_process < next_job_to_schedule &&
           num_valid_job_tested < min_num_jobs_to_test &&
           cache->durable_response_list[next_job_to_process].set) {
      auto durable_response =
          &cache->durable_response_list[next_job_to_process];
      next_job_to_process++;

      if (durable_response->status == SplitSearchResult::kNoBetterSplitFound) {
        // Even if no better split was found, this is still a valid job.
        num_valid_job_tested++;
      } else if (durable_response->status ==
                 SplitSearchResult::kBetterSplitFound) {
        num_valid_job_tested++;
        DCHECK_NE(durable_response->condition_idx, -1);

        const float process_split_score =
            cache->condition_list[durable_response->condition_idx]
                .split_score();

        if (process_split_score > best_split_score) {
          if (best_condition_idx != -1) {
            cache->available_condition_idxs.push_front(best_condition_idx);
          }
          best_condition_idx = durable_response->condition_idx;
          best_split_score = process_split_score;
        } else {
          // Return the condition to the condition pool.
          cache->available_condition_idxs.push_front(
              durable_response->condition_idx);
        }
      }
    }

    if (num_valid_job_tested >= min_num_jobs_to_test) {
      // Enough jobs have been tested to take a decision.
      break;
    }

    // Schedule the testing of more conditions.

    while (!cache->available_condition_idxs.empty() &&
           !cache->available_cache_idxs.empty() &&
           next_job_to_schedule < num_jobs) {
      const int attribute_idx =
          candidate_attributes[next_job_to_schedule - num_oblique_jobs];
      processor.Submit(produce(next_job_to_schedule++, best_split_score,
                               /*attribute_idx=*/attribute_idx,
                               /*num_oblique_projections_to_run=*/-1));
    }

    // The following condition means that no work is in the pipeline and no more
    // work will be generated.
    if (cache->available_cache_idxs.full()) {
      break;
    }
  }

  // Drain the response channel.
  while (!cache->available_cache_idxs.full()) {
    auto maybe_response = processor.GetResult();
    if (!maybe_response.has_value()) {
      break;
    }
    SplitterWorkResponse& response = maybe_response.value();
    cache->available_cache_idxs.push_front(response.manager_data.cache_idx);
  }

  // Move the random generator state to facilitate deterministic behavior.
  random->discard(num_jobs - next_job_to_schedule);

  if (best_condition_idx != -1) {
    *best_condition = cache->condition_list[best_condition_idx];
    return true;
  }
  return false;
}

absl::StatusOr<bool> FindBestConditionManager(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, const NodeConstraints& constraints,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    PerThreadCache* cache) {
  if (splitter_concurrency_setup.concurrent_execution) {
    return FindBestConditionConcurrentManager(
        train_dataset, selected_examples, weights, config, config_link,
        dt_config, splitter_concurrency_setup, parent, internal_config,
        label_stats, constraints, best_condition, random, cache);
  }
  return FindBestConditionSingleThreadManager(
      train_dataset, selected_examples, weights, config, config_link, dt_config,
      parent, internal_config, label_stats, constraints, best_condition, random,
      cache);
}

// This is the entry point when searching for a condition.
// All other "FindBestCondition*" functions are called by this one.
absl::StatusOr<bool> FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, PerThreadCache* cache) {
  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION: {
      STATUS_CHECK(!internal_config.hessian_score);
      ClassificationLabelStats label_stat(
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::CategoricalColumn>(
                  config_link.label())
              .value()
              ->values());

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

      return FindBestConditionManager(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, splitter_concurrency_setup, parent, internal_config,
          label_stat, constraints, best_condition, random, cache);
    } break;

    case model::proto::Task::REGRESSION: {
      if (internal_config.hessian_score) {
        DCHECK_NE(internal_config.gradient_col_idx, -1);
        DCHECK_NE(internal_config.hessian_col_idx, -1);

        DCHECK_EQ(internal_config.gradient_col_idx, config_link.label());
        RegressionHessianLabelStats label_stat(
            train_dataset
                .ColumnWithCastWithStatus<
                    dataset::VerticalDataset::NumericalColumn>(
                    internal_config.gradient_col_idx)
                .value()
                ->values(),
            train_dataset
                .ColumnWithCastWithStatus<
                    dataset::VerticalDataset::NumericalColumn>(
                    internal_config.hessian_col_idx)
                .value()
                ->values());

        DCHECK(parent.regressor().has_sum_gradients());
        label_stat.sum_gradient = parent.regressor().sum_gradients();
        label_stat.sum_hessian = parent.regressor().sum_hessians();
        label_stat.sum_weights = parent.regressor().sum_weights();

        return FindBestConditionManager(
            train_dataset, selected_examples, weights, config, config_link,
            dt_config, splitter_concurrency_setup, parent, internal_config,
            label_stat, constraints, best_condition, random, cache);
      } else {
        RegressionLabelStats label_stat(
            train_dataset
                .ColumnWithCastWithStatus<
                    dataset::VerticalDataset::NumericalColumn>(
                    config_link.label())
                .value()
                ->values());

        DCHECK(parent.regressor().has_distribution());
        label_stat.label_distribution.Load(parent.regressor().distribution());

        return FindBestConditionManager(
            train_dataset, selected_examples, weights, config, config_link,
            dt_config, splitter_concurrency_setup, parent, internal_config,
            label_stat, constraints, best_condition, random, cache);
      }
    } break;

    case model::proto::Task::CATEGORICAL_UPLIFT: {
      STATUS_CHECK(!internal_config.hessian_score);
      const auto& outcome_spec =
          train_dataset.data_spec().columns(config_link.label());
      const auto& treatment_spec =
          train_dataset.data_spec().columns(config_link.uplift_treatment());

      CategoricalUpliftLabelStats label_stat(
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::CategoricalColumn>(
                  config_link.label())
              .value()
              ->values(),
          outcome_spec.categorical().number_of_unique_values(),
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::CategoricalColumn>(
                  config_link.uplift_treatment())
              .value()
              ->values(),
          treatment_spec.categorical().number_of_unique_values());

      internal::UpliftLeafToLabelDist(parent.uplift(),
                                      &label_stat.label_distribution);

      return FindBestConditionManager(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, splitter_concurrency_setup, parent, internal_config,
          label_stat, constraints, best_condition, random, cache);
    } break;

    case model::proto::Task::NUMERICAL_UPLIFT: {
      STATUS_CHECK(!internal_config.hessian_score);
      const auto& treatment_spec =
          train_dataset.data_spec().columns(config_link.uplift_treatment());

      NumericalUpliftLabelStats label_stat(
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                  config_link.label())
              ->values(),
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
                  config_link.uplift_treatment())
              ->values(),
          treatment_spec.categorical().number_of_unique_values());

      internal::UpliftLeafToLabelDist(parent.uplift(),
                                      &label_stat.label_distribution);

      return FindBestConditionManager(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, splitter_concurrency_setup, parent, internal_config,
          label_stat, constraints, best_condition, random, cache);
    } break;

    default:
      return absl::UnimplementedError("Non implemented");
  }
  return false;
}

SplitSearchResult FindSplitLabelClassificationFeatureNumericalHistogram(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
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

  const auto bins =
      internal::GenHistogramBins(dt_config.numerical_split().type(),
                                 dt_config.numerical_split().num_candidates(),
                                 attributes, min_value, max_value, random);

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

SplitSearchResult FindSplitLabelClassificationFeatureNumericalCart(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
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

  const auto sorting_strategy = dt_config.internal().sorting_strategy();

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
              proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
          sorting_strategy ==
              proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
        if (!internal_config.preprocessing) {
          YDF_LOG(FATAL) << "Preprocessing missing for PRESORTED sorting "
                            "strategy";
        }
        if (sorting_strategy ==
                proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED ||
            IsPresortingOnNumericalSplitMoreEfficient(
                selected_examples.size(),
                internal_config.preprocessing->num_examples())) {
          const auto& sorted_attributes =
              internal_config.preprocessing
                  ->presorted_numerical_features()[attribute_idx];
          return ScanSplitsPresortedSparse<
              FeatureNumericalLabelUnweightedBinaryCategoricalOneValue,
              LabelBinaryCategoricalScoreAccumulator>(
              internal_config.preprocessing->num_examples(), selected_examples,
              sorted_attributes.items, feature_filler, label_filler,
              initializer, min_num_obs, attribute_idx,
              internal_config.duplicated_selected_examples, condition,
              &cache->cache_v2);
        }
      }

      return FindBestSplit_LabelUnweightedBinaryClassificationFeatureNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelBinaryCategoricalOneValueBucket</*weighted=*/true>::Filler
          label_filler(labels, weights);
      LabelBinaryCategoricalOneValueBucket</*weighted=*/true>::Initializer
          initializer(label_distribution);

      if (sorting_strategy ==
              proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
          sorting_strategy ==
              proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
        if (!internal_config.preprocessing) {
          YDF_LOG(FATAL) << "Preprocessing missing for PRESORTED sorting "
                            "strategy";
        }
        if (sorting_strategy ==
                proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED ||
            IsPresortingOnNumericalSplitMoreEfficient(
                selected_examples.size(),
                internal_config.preprocessing->num_examples())) {
          const auto& sorted_attributes =
              internal_config.preprocessing
                  ->presorted_numerical_features()[attribute_idx];
          return ScanSplitsPresortedSparse<
              FeatureNumericalLabelBinaryCategoricalOneValue,
              LabelBinaryCategoricalScoreAccumulator>(
              internal_config.preprocessing->num_examples(), selected_examples,
              sorted_attributes.items, feature_filler, label_filler,
              initializer, min_num_obs, attribute_idx,
              internal_config.duplicated_selected_examples, condition,
              &cache->cache_v2);
        }
      }

      return FindBestSplit_LabelBinaryClassificationFeatureNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  } else {
    // Multi-class classification.
    if (weights.empty()) {
      LabelCategoricalOneValueBucket</*weighted=*/false>::Filler label_filler(
          labels, weights);
      LabelCategoricalOneValueBucket</*weighted=*/false>::Initializer
          initializer(label_distribution);

      if (sorting_strategy ==
              proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
          sorting_strategy ==
              proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
        if (!internal_config.preprocessing) {
          YDF_LOG(FATAL) << "Preprocessing missing for PRESORTED sorting "
                            "strategy";
        }
        if (sorting_strategy ==
                proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED ||
            IsPresortingOnNumericalSplitMoreEfficient(
                selected_examples.size(),
                internal_config.preprocessing->num_examples())) {
          const auto& sorted_attributes =
              internal_config.preprocessing
                  ->presorted_numerical_features()[attribute_idx];
          return ScanSplitsPresortedSparse<
              FeatureNumericalLabelUnweightedCategoricalOneValue,
              LabelCategoricalScoreAccumulator>(
              internal_config.preprocessing->num_examples(), selected_examples,
              sorted_attributes.items, feature_filler, label_filler,
              initializer, min_num_obs, attribute_idx,
              internal_config.duplicated_selected_examples, condition,
              &cache->cache_v2);
        }
      }

      return FindBestSplit_LabelUnweightedClassificationFeatureNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelCategoricalOneValueBucket</*weighted=*/true>::Filler label_filler(
          labels, weights);
      LabelCategoricalOneValueBucket</*weighted=*/true>::Initializer
          initializer(label_distribution);

      if (sorting_strategy ==
              proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
          sorting_strategy ==
              proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
        if (!internal_config.preprocessing) {
          YDF_LOG(FATAL) << "Preprocessing missing for PRESORTED sorting "
                            "strategy";
        }
        if (sorting_strategy ==
                proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED ||
            IsPresortingOnNumericalSplitMoreEfficient(
                selected_examples.size(),
                internal_config.preprocessing->num_examples())) {
          const auto& sorted_attributes =
              internal_config.preprocessing
                  ->presorted_numerical_features()[attribute_idx];
          return ScanSplitsPresortedSparse<
              FeatureNumericalLabelCategoricalOneValue,
              LabelCategoricalScoreAccumulator>(
              internal_config.preprocessing->num_examples(), selected_examples,
              sorted_attributes.items, feature_filler, label_filler,
              initializer, min_num_obs, attribute_idx,
              internal_config.duplicated_selected_examples, condition,
              &cache->cache_v2);
        }
      }

      return FindBestSplit_LabelClassificationFeatureNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  }
}

SplitSearchResult FindSplitLabelClassificationFeatureDiscretizedNumericalCart(
    const std::vector<UnsignedExampleIdx>& selected_examples,
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
SplitSearchResult FindSplitLabelRegressionFeatureNumericalHistogram(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
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
  const auto bins =
      internal::GenHistogramBins(dt_config.numerical_split().type(),
                                 dt_config.numerical_split().num_candidates(),
                                 attributes, min_value, max_value, random);

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
SplitSearchResult FindSplitLabelHessianRegressionFeatureNumericalCart(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
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

  if (dt_config.internal().sorting_strategy() ==
          proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
      dt_config.internal().sorting_strategy() ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
    if (!internal_config.preprocessing) {
      YDF_LOG(FATAL) << "Preprocessing missing for PRESORTED sorting "
                        "strategy";
    }
    if (dt_config.internal().sorting_strategy() ==
            proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED ||
        IsPresortingOnNumericalSplitMoreEfficient(
            selected_examples.size(),
            internal_config.preprocessing->num_examples())) {
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
    }
  }

  return FindBestSplit_LabelHessianRegressionFeatureNumerical<weighted>(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

template SplitSearchResult
FindSplitLabelHessianRegressionFeatureNumericalCart<true>(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    float na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template SplitSearchResult
FindSplitLabelHessianRegressionFeatureNumericalCart<false>(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    float na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template <bool weighted>
SplitSearchResult
FindSplitLabelHessianRegressionFeatureDiscretizedNumericalCart(
    const std::vector<UnsignedExampleIdx>& selected_examples,
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
SplitSearchResult FindSplitLabelRegressionFeatureNumericalCart(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
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

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  typename LabelNumericalOneValueBucket<weighted>::Filler label_filler(labels,
                                                                       weights);

  typename LabelNumericalOneValueBucket<weighted>::Initializer initializer(
      label_distribution);
  const auto sorting_strategy = dt_config.internal().sorting_strategy();
  if (sorting_strategy ==
          proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
      sorting_strategy ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
    if (!internal_config.preprocessing) {
      YDF_LOG(FATAL) << "Preprocessing missing for PRESORTED sorting "
                        "strategy";
    }

    if (sorting_strategy ==
            proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED ||
        IsPresortingOnNumericalSplitMoreEfficient(
            -selected_examples.size(),
            internal_config.preprocessing->num_examples())) {
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
    }
  }

  return FindBestSplit_LabelRegressionFeatureNumerical<weighted>(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

template SplitSearchResult FindSplitLabelRegressionFeatureNumericalCart<true>(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const std::vector<float>& labels, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template SplitSearchResult FindSplitLabelRegressionFeatureNumericalCart<false>(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
    const std::vector<float>& labels, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template <bool weighted>
SplitSearchResult FindSplitLabelRegressionFeatureDiscretizedNumericalCart(
    const std::vector<UnsignedExampleIdx>& selected_examples,
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

SplitSearchResult FindSplitLabelClassificationFeatureNA(
    const std::vector<UnsignedExampleIdx>& selected_examples,
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
SplitSearchResult FindSplitLabelHessianRegressionFeatureNA(
    const std::vector<UnsignedExampleIdx>& selected_examples,
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

SplitSearchResult FindSplitLabelClassificationFeatureBoolean(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<char>& attributes,
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
SplitSearchResult FindSplitLabelRegressionFeatureBoolean(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<char>& attributes,
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

template SplitSearchResult FindSplitLabelRegressionFeatureBoolean<true>(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<char>& attributes,
    const std::vector<float>& labels, bool na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);
template SplitSearchResult FindSplitLabelRegressionFeatureBoolean<false>(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<char>& attributes,
    const std::vector<float>& labels, bool na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

template <bool weighted>
SplitSearchResult FindSplitLabelHessianRegressionFeatureBoolean(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<char>& attributes,
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
SplitSearchResult FindSplitLabelHessianRegressionFeatureCategorical(
    const std::vector<UnsignedExampleIdx>& selected_examples,
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
      YDF_LOG(FATAL) << "Non supported";
  }
}

template <bool weighted>
SplitSearchResult FindSplitLabelRegressionFeatureCategorical(
    const std::vector<UnsignedExampleIdx>& selected_examples,
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
      YDF_LOG(FATAL) << "Non supported";
  }
}

SplitSearchResult
FindSplitLabelClassificationFeatureCategoricalSetGreedyForward(
    const std::vector<UnsignedExampleIdx>& selected_examples,
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
SplitSearchResult FindSplitLabelRegressionFeatureCategoricalSetGreedyForward(
    const std::vector<UnsignedExampleIdx>& selected_examples,
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

template <typename LabelBucket, typename ExampleBucketSet,
          typename LabelScoreAccumulator>
SplitSearchResult FindSplitLabelClassificationFeatureCategorical(
    const std::vector<UnsignedExampleIdx>& selected_examples,
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
  const auto one_hot_scan = [&]() -> SplitSearchResult {
    CHECK_EQ(example_set_accumulator.items.size(), num_attribute_classes);

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

SplitSearchResult FindSplitLabelClassificationFeatureCategorical(
    const std::vector<UnsignedExampleIdx>& selected_examples,
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

SplitSearchResult FindSplitLabelUpliftCategoricalFeatureNumericalCart(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
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

SplitSearchResult FindSplitLabelUpliftNumericalFeatureNumericalCart(
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<float>& weights, const std::vector<float>& attributes,
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

SplitSearchResult FindSplitLabelUpliftCategoricalFeatureCategorical(
    const std::vector<UnsignedExampleIdx>& selected_examples,
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
      YDF_LOG(FATAL) << "Non supported";
  }
}

SplitSearchResult FindSplitLabelUpliftNumericalFeatureCategorical(
    const std::vector<UnsignedExampleIdx>& selected_examples,
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
      YDF_LOG(FATAL) << "Non supported";
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
      YDF_LOG(WARNING) << "Both \"num_candidate_attributes\" and "
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

void GenerateRandomImputation(const dataset::VerticalDataset& src,
                              const std::vector<int>& attributes,
                              const std::vector<UnsignedExampleIdx>& examples,
                              dataset::VerticalDataset* dst,
                              utils::RandomEngine* random) {
  CHECK_EQ(dst->ncol(), 0) << "The destination dataset should be empty.";
  dst->set_data_spec(src.data_spec());
  CHECK_OK(dst->CreateColumnsFromDataspec());
  dst->set_nrow(examples.size());
  for (const auto col_idx : attributes) {
    GenerateRandomImputationOnColumn(src.column(col_idx), examples,
                                     dst->mutable_column(col_idx), random);
  }
}

void GenerateRandomImputationOnColumn(
    const dataset::VerticalDataset::AbstractColumn* src,
    const std::vector<UnsignedExampleIdx>& examples,
    dataset::VerticalDataset::AbstractColumn* dst,
    utils::RandomEngine* random) {
  CHECK_EQ(src->type(), dst->type());
  // Extract the indices of the example with non-na values i.e. the candidate
  // for sampling.
  std::vector<UnsignedExampleIdx> non_na_examples;
  for (const auto example_idx : examples) {
    if (!src->IsNa(example_idx)) {
      non_na_examples.push_back(example_idx);
    }
  }

  if (non_na_examples.empty()) {
    CHECK_OK(src->ExtractAndAppend(examples, dst));
    return;
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
  CHECK_OK(src->ExtractAndAppend(source_indices, dst));
}

void SetDefaultHyperParameters(proto::DecisionTreeTrainingConfig* config) {
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

  if (config->split_axis_case() ==
      proto::DecisionTreeTrainingConfig::SPLIT_AXIS_NOT_SET) {
    config->mutable_axis_aligned_split();
  }

  // Disable pre-sorting if not supported by the splitters.
  if (config->internal().sorting_strategy() ==
          proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
      config->internal().sorting_strategy() ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
    if (config->has_sparse_oblique_split() ||
        config->has_mhld_oblique_split() ||
        config->missing_value_policy() !=
            proto::DecisionTreeTrainingConfig::GLOBAL_IMPUTATION) {
      config->mutable_internal()->set_sorting_strategy(
          proto::DecisionTreeTrainingConfig::Internal::IN_NODE);
    }
  }
}

template <class T, class S, class C>
S& Container(std::priority_queue<T, S, C>& q) {
  struct HackedQueue : private std::priority_queue<T, S, C> {
    static S& Container(std::priority_queue<T, S, C>& q) {
      return q.*&HackedQueue::c;
    }
  };
  return HackedQueue::Container(q);
}

absl::Status GrowTreeBestFirstGlobal(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& train_example_idxs,
    const std::vector<UnsignedExampleIdx>* optional_leaf_examples,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const std::vector<float>& weights,
    const InternalTrainConfig& internal_config, NodeWithChildren* root,
    utils::RandomEngine* random) {
  if (config.monotonic_constraints_size() > 0) {
    return absl::InvalidArgumentError(
        "Global growth of decision trees (i.e. "
        "growing_strategy=kGrowingStrategyBestFirstGlobal) does not support "
        "monotonic constraints.");
  }

  if (optional_leaf_examples) {
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
    std::vector<UnsignedExampleIdx> example_idxs;
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
  const auto ingest_node =
      [&](const std::vector<UnsignedExampleIdx>& example_idxs,
          NodeWithChildren* node, const int depth) -> absl::Status {
    RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
        train_dataset, example_idxs, weights, config, config_link, node));

    if (example_idxs.size() < dt_config.min_examples() ||
        (dt_config.max_depth() >= 0 && depth >= dt_config.max_depth())) {
      // Stop the grow of the branch.
      node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
      return absl::OkStatus();
    }
    proto::NodeCondition condition;
    ASSIGN_OR_RETURN(
        const auto has_better_condition,
        FindBestCondition(train_dataset, example_idxs, weights, config,
                          config_link, dt_config, splitter_concurrency_setup,
                          node->node(), internal_config, {}, &condition, random,
                          &cache));
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

  RETURN_IF_ERROR(ingest_node(train_example_idxs, root, /*depth=*/0));

  // Total number of nodes in the tree.
  int num_nodes = 1;

  const int max_num_nodes =
      dt_config.growing_strategy_best_first_global().max_num_nodes();

  std::vector<UnsignedExampleIdx> positive_examples;
  std::vector<UnsignedExampleIdx> negative_examples;

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
    RETURN_IF_ERROR(internal::SplitExamples(
        train_dataset, split.example_idxs, condition,
        /*dataset_is_dense=*/false,
        dt_config.internal_error_on_wrong_splitter_statistics(),
        &positive_examples, &negative_examples));

    RETURN_IF_ERROR(ingest_node(
        positive_examples, split.node->mutable_pos_child(), split.depth + 1));
    RETURN_IF_ERROR(ingest_node(
        negative_examples, split.node->mutable_neg_child(), split.depth + 1));
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
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const std::vector<float>& weights, utils::RandomEngine* random,
    DecisionTree* dt, const InternalTrainConfig& internal_config) {
  // Decide if execution should happen in single-thread or concurrent mode.

  const std::vector<UnsignedExampleIdx>* effective_selected_examples;
  const std::vector<UnsignedExampleIdx>* leaf_examples;
  std::vector<UnsignedExampleIdx> selected_examples_buffer;
  std::vector<UnsignedExampleIdx> leaf_examples_buffer;

  // Check monotonic constraints
  if (config.monotonic_constraints_size() > 0 &&
      !dt_config.keep_non_leaf_label_distribution()) {
    return absl::InvalidArgumentError(
        "keep_non_leaf_label_distribution=false is not compatible with "
        "monotonic constraints. To minimize the size of your serving model "
        "(with or without monotonic constraints), use "
        "pure_serving_model=true.");
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
    leaf_examples_buffer.reserve(selected_examples.size() * leaf_rate *
                                 error_margin);
    selected_examples_buffer.reserve(selected_examples.size() *
                                     (1.f - leaf_rate) * error_margin);

    auto* effective_random = random;
    utils::RandomEngine fixed_random(12345678);
    if (dt_config.honest().fixed_separation()) {
      effective_random = &fixed_random;
    }

    for (const auto& example : selected_examples) {
      if (dist_01(*effective_random) < leaf_rate) {
        leaf_examples_buffer.push_back(example);
      } else {
        selected_examples_buffer.push_back(example);
      }
    }
    effective_selected_examples = &selected_examples_buffer;
    leaf_examples = &leaf_examples_buffer;
  } else {
    effective_selected_examples = &selected_examples;
    leaf_examples = nullptr;
  }

  SplitterConcurrencySetup splitter_concurrency_setup;
  if (internal_config.num_threads <= 1) {
    splitter_concurrency_setup.concurrent_execution = false;
    return DecisionTreeCoreTrain(train_dataset, *effective_selected_examples,
                                 leaf_examples, config, config_link, dt_config,
                                 deployment, splitter_concurrency_setup,
                                 weights, random, internal_config, dt);
  } else {
    splitter_concurrency_setup.concurrent_execution = true;
    splitter_concurrency_setup.num_threads = internal_config.num_threads;
  }

  splitter_concurrency_setup.split_finder_processor =
      absl::make_unique<SplitterFinderStreamProcessor>(
          "SplitFinder", internal_config.num_threads,
          [&](SplitterWorkRequest request) -> SplitterWorkResponse {
            return FindBestConditionFromSplitterWorkRequest(
                weights, config, config_link, dt_config,
                splitter_concurrency_setup, internal_config, request);
          });
  splitter_concurrency_setup.split_finder_processor->StartWorkers();

  return DecisionTreeCoreTrain(train_dataset, *effective_selected_examples,
                               leaf_examples, config, config_link, dt_config,
                               deployment, splitter_concurrency_setup, weights,
                               random, internal_config, dt);
}

absl::Status DecisionTreeCoreTrain(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<UnsignedExampleIdx>* optional_leaf_examples,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const std::vector<float>& weights, utils::RandomEngine* random,
    const InternalTrainConfig& internal_config, DecisionTree* dt) {
  dt->CreateRoot();
  PerThreadCache cache;
  switch (dt_config.growing_strategy_case()) {
    case proto::DecisionTreeTrainingConfig::GROWING_STRATEGY_NOT_SET:
    case proto::DecisionTreeTrainingConfig::kGrowingStrategyLocal: {
      const auto constraints = NodeConstraints::CreateNodeConstraints();
      return NodeTrain(train_dataset, selected_examples, optional_leaf_examples,
                       config, config_link, dt_config, deployment,
                       splitter_concurrency_setup, weights, 1, internal_config,
                       constraints, false, dt->mutable_root(), random, &cache);
    } break;
    case proto::DecisionTreeTrainingConfig::kGrowingStrategyBestFirstGlobal:
      return GrowTreeBestFirstGlobal(
          train_dataset, selected_examples, optional_leaf_examples, config,
          config_link, dt_config, deployment, splitter_concurrency_setup,
          weights, internal_config, dt->mutable_root(), random);
      break;
  }
}

absl::Status NodeTrain(
    const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    const std::vector<UnsignedExampleIdx>* optional_leaf_examples,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const SplitterConcurrencySetup& splitter_concurrency_setup,
    const std::vector<float>& weights, const int32_t depth,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, bool set_leaf_already_set,
    NodeWithChildren* node, utils::RandomEngine* random,
    PerThreadCache* cache) {
  if (selected_examples.empty()) {
    return absl::InternalError("No examples fed to the node trainer");
  }
  node->mutable_node()->set_num_pos_training_examples_without_weight(
      selected_examples.size());

  if (!set_leaf_already_set) {
    // Set the node value (i.e. the label distribution).
    RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
        train_dataset, selected_examples, weights, config, config_link, node));
    RETURN_IF_ERROR(ApplyConstraintOnNode(constraints, node));
  }

  if (selected_examples.size() < dt_config.min_examples() ||
      (dt_config.max_depth() >= 0 && depth >= dt_config.max_depth()) ||
      (internal_config.timeout.has_value() &&
       internal_config.timeout < absl::Now())) {
    if (optional_leaf_examples) {
      // Override the leaf values.
      RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
          train_dataset, *optional_leaf_examples, weights, config, config_link,
          node));
      RETURN_IF_ERROR(ApplyConstraintOnNode(constraints, node));
    }

    // Stop the growth of the branch.
    node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
    return absl::OkStatus();
  }

  // Dataset used to train this node.
  const dataset::VerticalDataset* local_train_dataset = &train_dataset;
  const std::vector<UnsignedExampleIdx>* local_selected_examples =
      &selected_examples;
  // If true, the entire dataset "local_train_dataset" is composed of training
  // examples for this node. If false, only the subset of
  // "local_train_dataset" indexed by "selected_examples" are to be considered
  // for this node i.e. local_train_dataset[selected_examples[i]].
  bool local_train_dataset_is_compact = false;

  // Extract the random local imputation.
  dataset::VerticalDataset random_local_imputation_train_dataset;
  std::vector<UnsignedExampleIdx> random_local_imputation_selected_examples;
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::RANDOM_LOCAL_IMPUTATION) {
    std::vector<int> label_and_input_features(config_link.features().begin(),
                                              config_link.features().end());
    label_and_input_features.push_back(config_link.label());
    GenerateRandomImputation(train_dataset, label_and_input_features,
                             selected_examples,
                             &random_local_imputation_train_dataset, random);
    random_local_imputation_selected_examples.resize(selected_examples.size());
    std::iota(random_local_imputation_selected_examples.begin(),
              random_local_imputation_selected_examples.end(), 0);

    local_train_dataset = &random_local_imputation_train_dataset;
    local_selected_examples = &random_local_imputation_selected_examples;
    local_train_dataset_is_compact = true;
  }

  // Determine the best split.
  ASSIGN_OR_RETURN(
      const auto has_better_condition,
      FindBestCondition(
          *local_train_dataset, *local_selected_examples, weights, config,
          config_link, dt_config, splitter_concurrency_setup, node->node(),
          internal_config, constraints,
          node->mutable_node()->mutable_condition(), random, cache));
  if (!has_better_condition) {
    // No good condition found. Close the branch.
    node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
    return absl::OkStatus();
  }
  CHECK_EQ(selected_examples.size(),
           node->node().condition().num_training_examples_without_weight());
  node->CreateChildren();
  node->FinalizeAsNonLeaf(dt_config.keep_non_leaf_label_distribution(),
                          dt_config.store_detailed_label_distribution());

  // Ensure the per-depth cache is allocated.
  while (cache->per_depth.size() < depth) {
    cache->per_depth.push_back(absl::make_unique<PerThreadCache::PerDepth>());
  }

  // Separate the positive and negative examples.
  auto& per_depth_cache = *cache->per_depth[depth - 1];
  std::vector<UnsignedExampleIdx>& positive_examples =
      per_depth_cache.positive_examples;
  std::vector<UnsignedExampleIdx>& negative_examples =
      per_depth_cache.negative_examples;
  RETURN_IF_ERROR(internal::SplitExamples(
      *local_train_dataset, selected_examples, node->node().condition(),
      local_train_dataset_is_compact,
      dt_config.internal_error_on_wrong_splitter_statistics(),
      &positive_examples, &negative_examples));

  // Separate the positive and negative examples used only to determine the node
  // value.
  std::vector<UnsignedExampleIdx>* positive_node_only_examples = nullptr;
  std::vector<UnsignedExampleIdx>* negative_node_only_examples = nullptr;
  if (optional_leaf_examples) {
    positive_node_only_examples = &per_depth_cache.positive_node_only_examples;
    negative_node_only_examples = &per_depth_cache.negative_node_only_examples;
    RETURN_IF_ERROR(internal::SplitExamples(
        train_dataset, *optional_leaf_examples, node->node().condition(), false,
        dt_config.internal_error_on_wrong_splitter_statistics(),
        positive_node_only_examples, negative_node_only_examples,
        /*examples_are_training_examples=*/false));
  }

  // Set leaf outputs
  RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
      train_dataset, positive_examples, weights, config, config_link,
      node->mutable_pos_child()));
  RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
      train_dataset, negative_examples, weights, config, config_link,
      node->mutable_neg_child()));
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
      train_dataset, positive_examples, positive_node_only_examples, config,
      config_link, dt_config, deployment, splitter_concurrency_setup, weights,
      depth + 1, internal_config, pos_constraints, true,
      node->mutable_pos_child(), random, cache));

  // Negative child.
  RETURN_IF_ERROR(NodeTrain(
      train_dataset, negative_examples, negative_node_only_examples, config,
      config_link, dt_config, deployment, splitter_concurrency_setup, weights,
      depth + 1, internal_config, neg_constraints, true,
      node->mutable_neg_child(), random, cache));
  return absl::OkStatus();
}

absl::StatusOr<Preprocessing> PreprocessTrainingDataset(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config, const int num_threads) {
  const auto time_begin = absl::Now();
  Preprocessing preprocessing;
  preprocessing.set_num_examples(train_dataset.nrow());

  if (dt_config.internal().sorting_strategy() ==
          proto::DecisionTreeTrainingConfig::Internal::PRESORTED ||
      dt_config.internal().sorting_strategy() ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
    RETURN_IF_ERROR(PresortNumericalFeatures(train_dataset, config_link,
                                             num_threads, &preprocessing));
  }

  const auto duration = absl::Now() - time_begin;
  if (duration > absl::Seconds(10)) {
    YDF_LOG(INFO) << "Feature index computed in "
                  << absl::FormatDuration(duration);
  }
  return preprocessing;
}

absl::Status PresortNumericalFeatures(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfigLinking& config_link,
    const int num_threads, Preprocessing* preprocessing) {
  // Check number of examples.
  RETURN_IF_ERROR(dataset::CheckNumExamples(train_dataset.nrow()));

  preprocessing->mutable_presorted_numerical_features()->resize(
      train_dataset.data_spec().columns_size());

  utils::concurrency::ThreadPool pool(
      "presort_numerical_features",
      std::min(num_threads, config_link.features().size()));
  pool.StartWorkers();

  // For all the input features in the model.
  for (const auto feature_idx : config_link.features()) {
    // Skip non numerical features.
    if (train_dataset.data_spec().columns(feature_idx).type() !=
        dataset::proto::NUMERICAL) {
      continue;
    }

    pool.Schedule([feature_idx, &train_dataset, preprocessing]() {
      const UnsignedExampleIdx num_examples = train_dataset.nrow();
      const auto& values =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::NumericalColumn>(feature_idx)
              .value()
              ->values();
      CHECK_EQ(num_examples, values.size());

      // Global imputation replacement.
      const float na_replacement_value =
          train_dataset.data_spec().columns(feature_idx).numerical().mean();

      std::vector<std::pair<float, SparseItem::ExampleIdx>> items(
          values.size());
      for (UnsignedExampleIdx example_idx = 0; example_idx < num_examples;
           example_idx++) {
        auto value = values[example_idx];
        if (std::isnan(value)) {
          value = na_replacement_value;
        }
        items[example_idx] = {value, example_idx};
      }

      // Sort by feature value and example index.
      std::sort(items.begin(), items.end());

      auto& sorted_values =
          (*preprocessing->mutable_presorted_numerical_features())[feature_idx];
      sorted_values.items.resize(values.size());

      for (UnsignedExampleIdx sorted_example_idx = 0;
           sorted_example_idx < num_examples; sorted_example_idx++) {
        SparseItem::ExampleIdx example_idx = items[sorted_example_idx].second;
        const bool change_value =
            sorted_example_idx > 0 && (items[sorted_example_idx].first !=
                                       items[sorted_example_idx - 1].first);
        if (change_value) {
          example_idx |= ((SparseItem::ExampleIdx)1)
                         << (sizeof(SparseItem::ExampleIdx) * 8 - 1);
        }
        sorted_values.items[sorted_example_idx].example_idx_and_extra =
            example_idx;
      }
    });
  }
  return absl::OkStatus();
}

absl::Status ApplyConstraintOnNode(const NodeConstraints& constraint,
                                   NodeWithChildren* node) {
  if (!constraint.min_max_output.has_value()) {
    return absl::OkStatus();
  }
  auto* reg = node->mutable_node()->mutable_regressor();
  STATUS_CHECK(reg->has_top_value());
  reg->set_top_value(utils::clamp(reg->top_value(),
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
    const std::vector<UnsignedExampleIdx>& selected_examples,
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

std::vector<float> GenHistogramBins(const proto::NumericalSplit::Type type,
                                    const int num_splits,
                                    const std::vector<float>& attributes,
                                    const float min_value,
                                    const float max_value,
                                    utils::RandomEngine* random) {
  CHECK_GE(num_splits, 0);
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
      YDF_LOG(FATAL) << "Numerical histogram not implemented";
  }
  std::sort(candidate_splits.begin(), candidate_splits.end());
  return candidate_splits;
}

absl::Status SplitExamples(const dataset::VerticalDataset& dataset,
                           const std::vector<UnsignedExampleIdx>& examples,
                           const proto::NodeCondition& condition,
                           const bool dataset_is_dense,
                           const bool error_on_wrong_splitter_statistics,
                           std::vector<UnsignedExampleIdx>* positive_examples,
                           std::vector<UnsignedExampleIdx>* negative_examples,
                           const bool examples_are_training_examples) {
  if (examples_are_training_examples) {
    positive_examples->reserve(
        condition.num_pos_training_examples_without_weight());
    negative_examples->reserve(
        examples.size() - condition.num_pos_training_examples_without_weight());
  }

  positive_examples->clear();
  negative_examples->clear();

  std::vector<UnsignedExampleIdx>* example_sets[] = {negative_examples,
                                                     positive_examples};

  // Index of the example selected for this node.
  const auto column_data = dataset.column(condition.attribute());

  if (!dataset_is_dense) {
    for (const UnsignedExampleIdx example_idx : examples) {
      const auto dst = example_sets[EvalConditionFromColumn(
          condition, column_data, dataset, example_idx)];
      dst->push_back(example_idx);
    }
  } else {
    UnsignedExampleIdx dense_example_idx = 0;
    for (const UnsignedExampleIdx example_idx : examples) {
      const auto dst = example_sets[EvalConditionFromColumn(
          condition, column_data, dataset, dense_example_idx)];
      dense_example_idx++;
      dst->push_back(example_idx);
    }
  }

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
          (positive_examples->size() !=
           condition.num_pos_training_examples_without_weight()) ||
          (negative_examples->size() !=
           examples.size() -
               condition.num_pos_training_examples_without_weight()))) {
    const std::string message = absl::Substitute(
        "The effective split of examples does not match the expected split "
        "returned by the splitter algorithm. This problem can be caused by (1) "
        "large floating point values (e.g. value>=10e30) or (2) a bug in the "
        "software. You can turn this error in a warning with "
        "internal_error_on_wrong_splitter_statistics=false.\n\nDetails:\n"
        "Num examples: $0\n"
        "Effective num positive examples: $1\n"
        "Expected num positive example: $4\n"
        "Effective num negative examples: $2\n"
        "Condition: $3\n"
        "Attribute spec: $5",
        /*$0*/ examples.size(),
        /*$1*/ positive_examples->size(),
        /*$2*/ negative_examples->size(),
        /*$3*/ condition.DebugString(),
        /*$4*/ condition.num_pos_training_examples_without_weight(),
        /*$5*/
        dataset.data_spec().columns(condition.attribute()).DebugString());
    if (error_on_wrong_splitter_statistics) {
      return absl::InternalError(message);
    } else {
      YDF_LOG(WARNING) << message;
    }
  }
  return absl::OkStatus();
}

void UpliftLeafToLabelDist(const decision_tree::proto::NodeUpliftOutput& leaf,
                           UpliftLabelDistribution* dist) {
  dist->ImportSetFromLeafProto(leaf);
}

void UpliftLabelDistToLeaf(const UpliftLabelDistribution& dist,
                           decision_tree::proto::NodeUpliftOutput* leaf) {
  dist.ExportToLeafProto(leaf);
}

}  // namespace internal

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
